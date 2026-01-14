from pathlib import Path
import h5py
import os
import threadpoolctl

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from joblib import Parallel, delayed

from deepx_dock.misc import load_json_file, load_poscar_file
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.compute.eigen.matrix import AOMatrixR

class HamiltonianObj:
    """
    Tight-binding Hamiltonian in the matrix form.
    
    This class constructs the Hamiltonian operator from the standard DeepH 
    format data. The Hamiltonian and overlap matrix in real space (H(R) and S(R))
    are constructed and can be Fourier transformed to the reciprocal space 
    (H(k) and S(k)). The diagonalization of the Hamiltonian is also supported.
    
    Parameters
    ----------
    info_dir_path : str 
        Path to the directory containing the POSCAR, info.json and overlap.h5.
    
    H_file_path : str (optional)
        Path to the Hamiltonian file. Default: hamiltonian.h5 under `info_dir_path`.
    
    Properties:
    ----------
    lattice : np.array((3, 3), dtype=float)
        Lattice vectors. Each row is a lattice vector.

    reciprocal_lattice : np.array((3, 3), dtype=float)
        Reciprocal lattice vectors. Each row is a reciprocal lattice vector.

    Rijk_list : np.array((N_R, 3), dtype=int)
        Lattice displacements for inter-cell hoppings.
        The displacements are expressed in terms of the lattice vectors.
        N_R is the number of displacements.

    SR : np.array((N_R, N_b, N_b), dtype=float)
        Overlap matrix in real space. SR[i, :, :] = S(Rijk_list[i, :]).
        N_b is the number of basis functions in the unit cell (including the spin DOF if spinful is true).
    
    HR : np.array((N_R, N_b, N_b), dtype=float/complex)
        Hamiltonian matrix in real space. HR[i, :, :] = H(Rijk_list[i, :]).
        The dtype is float if spinful is false, otherwise the dtype is complex.
    """
    def __init__(self, info_dir_path, H_file_path=None):
        self._get_necessary_data_path(info_dir_path, H_file_path)
        #
        self.mat_S = None
        self.mat_H = None
        #
        self.parse_data()

    @property
    def Rijk_list(self):
        return self.mat_S.Rs if self.mat_S is not None else None

    @property
    def SR(self):
        return self.mat_S.MRs if self.mat_S is not None else None

    @property
    def HR(self):
        return self.mat_H.MRs if self.mat_H is not None else None
    
    @property
    def R_quantity(self):
        return len(self.mat_S.Rs) if self.mat_S is not None else 0

    def _get_necessary_data_path(self,
        info_dir_path: str | Path, H_file_path: str | Path | None = None
    ):
        info_dir_path = Path(info_dir_path)
        self.poscar_path = info_dir_path / DEEPX_POSCAR_FILENAME
        self.info_json_path = info_dir_path / DEEPX_INFO_FILENAME
        self.SR_path = info_dir_path / DEEPX_OVERLAP_FILENAME
        self.HR_path = (info_dir_path / DEEPX_HAMILTONIAN_FILENAME) if H_file_path is None else Path(H_file_path)

    def parse_data(self):
        self._parse_info()
        self._parse_poscar()
        self._parse_orbit_types()
        self._parse_overlap()
        self._parse_hamiltonian()

    def _parse_info(self):
        raw_info = self._read_info_json(self.info_json_path)
        #
        self.atoms_quantity = raw_info["atoms_quantity"]
        self.orbits_quantity = raw_info["orbits_quantity"]
        self.is_orthogonal_basis = raw_info["orthogonal_basis"]
        self.spinful = raw_info["spinful"]
        self.fermi_energy = raw_info["fermi_energy_eV"]
        self.elements_orbital_map = raw_info["elements_orbital_map"]
        self.occupation = raw_info.get("occupation", None)
    
    def _parse_poscar(self):
        raw_poscar = self._read_poscar(self.poscar_path)
        #
        self.lattice = raw_poscar["lattice"]
        self.elements = raw_poscar["elements"]
        self.frac_coords = raw_poscar["frac_coords"]
        self.reciprocal_lattice = self.get_reciprocal_lattice(self.lattice)
    
    def _parse_orbit_types(self):
        self.atom_num_orbits = [
            np.sum(2 * np.array(self.elements_orbital_map[el]) + 1)
            for el in self.elements
        ]
        self.atom_num_orbits_cumsum = np.insert(
            np.cumsum(self.atom_num_orbits), 0, 0
        )
        assert self.orbits_quantity == self.atom_num_orbits_cumsum[-1], f"Number of orbitals {self.orbits_quantity}(info.json) and {self.atom_num_orbits_cumsum[-1]}(POSCAR) do not match"

    def _parse_overlap(self):
        S_R = {}
        atom_pairs, bounds, shapes, entries = self._read_h5(self.SR_path)
        self.atom_pairs = atom_pairs
        for i_ap, ap in enumerate(atom_pairs):
            # Gen Data
            Rijk = (ap[0], ap[1], ap[2])
            i_atom, j_atom  = ap[3], ap[4]
            if Rijk not in S_R:
                S_R[Rijk] = np.zeros(
                    (self.orbits_quantity, self.orbits_quantity),
                    dtype=np.float64
                )
            # Get Chunk
            _bound_slice = slice(bounds[i_ap], bounds[i_ap+1])
            _shape = shapes[i_ap]
            _S_chunk = entries[_bound_slice].reshape(_shape)
            # Fill Values
            _i_slice = slice(
                self.atom_num_orbits_cumsum[i_atom],
                self.atom_num_orbits_cumsum[i_atom+1]
            )
            _j_slice = slice(
                self.atom_num_orbits_cumsum[j_atom],
                self.atom_num_orbits_cumsum[j_atom+1]
            )
            S_R[Rijk][_i_slice, _j_slice] = _S_chunk
        #
        R_quantity = len(S_R)
        Rijk_list = np.zeros((R_quantity, 3), dtype=int)
        SR = np.zeros(
            (R_quantity, self.orbits_quantity, self.orbits_quantity),
            dtype=np.float64
        )
        for i_R, (Rijk, S_val) in enumerate(S_R.items()):
            Rijk_list[i_R] = Rijk
            SR[i_R] = S_val
        #
        if self.spinful:
            _zeros_S = np.zeros_like(SR)
            SR = np.block(
                [[SR, _zeros_S], [_zeros_S, SR]]
            )
        self.mat_S = AOMatrixR(Rijk_list, SR)
    
    def _parse_hamiltonian(self):
        H_R = {}
        dtype = np.complex128 if self.spinful else np.float64
        atom_pairs, bounds, shapes, entries = \
            self._read_h5(self.HR_path, dtype=dtype)
        assert np.array_equal(self.atom_pairs, atom_pairs), "The atom pairs is not the same."
        bands_quantity = self.orbits_quantity * (1 + self.spinful)
        R_quantity = self.R_quantity
        _matrix_shape = (R_quantity, bands_quantity, bands_quantity)
        for i_ap, ap in enumerate(atom_pairs):
            # Gen Data
            R_ijk = (ap[0], ap[1], ap[2])
            i_atom, j_atom  = ap[3], ap[4]
            if R_ijk not in H_R:
                H_R[R_ijk] = np.zeros(
                    (bands_quantity, bands_quantity), dtype=dtype
                )
            # Get Chunk
            _bound_slice = slice(bounds[i_ap], bounds[i_ap+1])
            _shape = shapes[i_ap]
            _H_chunk = entries[_bound_slice].reshape(_shape)
            # Fill Values
            if self.spinful:
                _i_slice_up = slice(
                    self.atom_num_orbits_cumsum[i_atom],
                    self.atom_num_orbits_cumsum[i_atom+1]
                )
                _i_slice_dn = slice(
                    self.atom_num_orbits_cumsum[i_atom] + self.orbits_quantity,
                    self.atom_num_orbits_cumsum[i_atom+1] + self.orbits_quantity
                )
                _j_slice_up = slice(
                    self.atom_num_orbits_cumsum[j_atom],
                    self.atom_num_orbits_cumsum[j_atom+1]
                )
                _j_slice_dn = slice(
                    self.atom_num_orbits_cumsum[j_atom] + self.orbits_quantity,
                    self.atom_num_orbits_cumsum[j_atom+1] + self.orbits_quantity
                )
                _i_orb_num = self.atom_num_orbits[i_atom]
                _j_orb_num = self.atom_num_orbits[j_atom]
                H_R[R_ijk][_i_slice_up, _j_slice_up] = \
                    _H_chunk[:_i_orb_num, :_j_orb_num]
                H_R[R_ijk][_i_slice_up, _j_slice_dn] = \
                    _H_chunk[:_i_orb_num, _j_orb_num:]
                H_R[R_ijk][_i_slice_dn, _j_slice_up] = \
                    _H_chunk[_i_orb_num:, :_j_orb_num]
                H_R[R_ijk][_i_slice_dn, _j_slice_dn] = \
                    _H_chunk[_i_orb_num:, _j_orb_num:]
            else:
                _i_slice = slice(
                    self.atom_num_orbits_cumsum[i_atom],
                    self.atom_num_orbits_cumsum[i_atom+1]
                )
                _j_slice = slice(
                    self.atom_num_orbits_cumsum[j_atom],
                    self.atom_num_orbits_cumsum[j_atom+1]
                )
                H_R[R_ijk][_i_slice, _j_slice] = _H_chunk
        #
        assert self.Rijk_list is not None, "You must read in the overlaps first!"
        assert R_quantity == len(H_R), f"The overlap R quantity `{R_quantity}` is not agree with the hamiltonian `{len(H_R)}`."
        HR = np.zeros(_matrix_shape, dtype=dtype)
        for i_R in range(R_quantity):
            R_ijk = self.Rijk_list[i_R]
            HR[i_R] = H_R[tuple(R_ijk)]
        self.mat_H = AOMatrixR(self.Rijk_list, HR)

    @staticmethod
    def get_reciprocal_lattice(lattice):
        a = np.array(lattice)
        #
        volume = abs(np.dot(a[0], np.cross(a[1], a[2])))
        if np.isclose(volume, 0):
            raise ValueError("Invalid lattice: Volume is zero")
        #
        b1 = 2 * np.pi * np.cross(a[1], a[2]) / volume
        b2 = 2 * np.pi * np.cross(a[2], a[0]) / volume
        b3 = 2 * np.pi * np.cross(a[0], a[1]) / volume
        #
        return np.vstack([b1, b2, b3])

    @staticmethod
    def _read_h5(h5_path, dtype=np.float64):
        with h5py.File(h5_path, 'r') as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            entries = np.array(f['entries'][:], dtype=dtype)
        return atom_pairs, boundaries, shapes, entries

    @staticmethod
    def _read_info_json(json_path):
        return load_json_file(json_path)

    @staticmethod
    def _read_poscar(filename):
        result = load_poscar_file(filename)
        elements = [
            elem for elem, n in zip(
                result["elements_unique"], result["elements_counts"]
            ) for _ in range(n)
        ]
        return {
            "lattice": result["lattice"],
            "elements": elements,
            "cart_coords": result["cart_coords"],
            "frac_coords": result["frac_coords"],
        }
    
    def Sk_and_Hk(self, k):
        # Support batch k or single k.
        # k: (3,) or (Nk, 3)
        if k.ndim == 1:
            ks = k[None, :]
            squeeze = True
        else:
            ks = k
            squeeze = False
            
        Sk = self.mat_S.r2k(ks)
        Hk = self.mat_H.r2k(ks)
        
        if squeeze:
            return Sk[0], Hk[0]
        return Sk, Hk

    def diag(self, ks, k_process_num=1, thread_num=None, sparse_calc=False, bands_only=False, **kwargs):
        """
        Diagonalize the Hamiltonian at specified k-points to obtain eigenvalues (bands) 
        and optionally eigenvectors (wave functions).

        This function supports both dense (scipy.linalg.eigh) and sparse (scipy.sparse.linalg.eigsh) 
        solvers and utilizes parallel computing via joblib.

        Parameters
        ----------
        ks : array_like, shape (Nk, 3)
            List of k-points in reduced coordinates (fractional).
        k_process_num : int, optional
            Number of parallel processes to use (default is 1).
            If > 1, BLAS threads per process are restricted to 1 to avoid oversubscription.
        sparse_calc : bool, optional
            If True, use sparse solver (eigsh). If False, use dense solver (eigh).
            Default is False.
        bands_only : bool, optional
            If True, only compute and return eigenvalues. Faster and uses less memory.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the solver.
            - For sparse_calc=True (eigsh): 'k' (num eigenvalues), 'which' (e.g., 'SA'), 'sigma', etc.
            - For sparse_calc=False (eigh): 'driver', 'type', etc.

        Returns
        -------
        eigvals : np.ndarray
            The eigenvalues (band energies).
            Shape: (Nband, Nk)
        eigvecs : np.ndarray, optional
            The eigenvectors (coefficients). Returned only if bands_only is False.
            Shape: (Norb, Nband, Nk)
        """

        mat_S = self.mat_S
        mat_H = self.mat_H

        def process_k(k):
            # Hk, Sk: (Norb, Norb)
            # Use vectorized r2k for single k (1, 3) -> (1, Norb, Norb) -> (Norb, Norb)
            Sk = mat_S.r2k(k[None, :])[0]
            Hk = mat_H.r2k(k[None, :])[0]
            
            if sparse_calc:
                if bands_only:
                    # vals: (k,)
                    vals = eigsh(Hk, M=Sk, return_eigenvectors=False, **kwargs)
                    return np.sort(vals)
                else:
                    # vals: (k,), vecs: (Norb, k)
                    vals, vecs = eigsh(Hk, M=Sk, **kwargs)
                    idx = np.argsort(vals)
                    return vals[idx], vecs[:, idx]
            else:
                if bands_only:
                    # vals: (Norb,)
                    vals = eigh(Hk, Sk, eigvals_only=True)
                    return vals 
                else:
                    # vals: (Norb,), vecs: (Norb, Norb)
                    vals, vecs = eigh(Hk, Sk)
                    return vals, vecs

        # Limit BLAS threads per process to prevent CPU contention during parallel execution
        if thread_num is None:
            thread_num = int(os.environ.get('OPENBLAS_NUM_THREADS', "1"))
        with threadpoolctl.threadpool_limits(limits=thread_num, user_api='blas'):
            if k_process_num == 1:
                results = [process_k(k) for k in tqdm(ks, leave=False)]
            else:
                results = Parallel(n_jobs=k_process_num)(
                    delayed(process_k)(k) for k in tqdm(ks, leave=False)
                )

        # Reorganize results into arrays
        if bands_only:
            # results: List of (Nband,) -> Stack -> (Nband, Nk)
            return np.stack(results, axis=1)
        else:
            # results: List of ((Nband,), (Norb, Nband))
            
            # vals: List of (Nband,) -> Stack -> (Nband, Nk)
            eigvals = np.stack([res[0] for res in results], axis=1)
            
            # vecs: List of (Norb, Nband) -> Stack -> (Norb, Nband, Nk)
            eigvecs = np.stack([res[1] for res in results], axis=2)
            
            return eigvals, eigvecs
