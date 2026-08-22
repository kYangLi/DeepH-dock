"""
Atomic Orbital Matrix Container Classes

This module provides container classes for atomic orbital matrices in both
real space and reciprocal space, as well as a base class for loading matrices
from DeepH format files.
"""

from pathlib import Path
import h5py
import threadpoolctl
import numpy as np

from deepx_dock.parallel import parallel_map
from deepx_dock.misc import load_json_file, load_poscar_file
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL


class AOMatrixR:
    """
    Atomic orbital matrix in real space.

    Properties
    ----------
    Rs : np.ndarray, shape (N_R, 3), dtype=int
        Lattice displacements for inter-cell hoppings, in fractional
        coordinates (integers). The displacements are expressed in terms
        of the lattice vectors. N_R is the number of displacements.

    MRs : np.ndarray, shape (N_R, N_b, N_b), dtype=float or complex
        Matrix in real space. MRs[i, :, :] = M(R_list[i, :]).
        The dtype is float if spinful is false, otherwise complex.
    """

    def __init__(self, Rs, MRs):
        self.Rs = Rs
        self.MRs = MRs

    def r2k(self, ks):
        """
        Fourier transform from real space to reciprocal space.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3)
            k-points in fractional coordinates.

        Returns
        -------
        MKs : np.ndarray, shape (Nk, N_b, N_b)
            Matrices in reciprocal space.
        """
        phase = np.exp(2j * np.pi * np.matmul(ks, self.Rs.T))
        MRs_flat = self.MRs.reshape(len(self.Rs), -1)
        Mks_flat = np.matmul(phase, MRs_flat)
        return Mks_flat.reshape(len(ks), *self.MRs.shape[1:])


class AOMatrixK:
    """
    Atomic orbital matrix in reciprocal space.

    Properties
    ----------
    ks : np.ndarray, shape (N_k, 3), dtype=float
        Reciprocal lattice points for the Fourier transform, in fractional
        coordinates. N_k is the number of points.

    MKs : np.ndarray, shape (N_k, N_b, N_b), dtype=float or complex
        Matrix in reciprocal space. MKs[i, :, :] = M(ks[i, :]).
        The dtype is float if spinful is false, otherwise complex.
    """

    def __init__(self, ks, MKs):
        self.ks = ks
        self.MKs = MKs

    def k2r(self, Rs, weights=None):
        """
        Inverse Fourier transform from reciprocal space to real space.

        Parameters
        ----------
        Rs : np.ndarray, shape (NR, 3)
            R-vectors in fractional coordinates.
        weights : np.ndarray, shape (Nk,), optional
            Weights for k-points. Default is uniform weights.

        Returns
        -------
        MRs : np.ndarray, shape (NR, N_b, N_b)
            Matrices in real space.
        """
        if weights is None:
            weights = np.ones(len(self.ks)) / len(self.ks)
        else:
            weights = np.array(weights)

        phase = np.exp(-2j * np.pi * np.matmul(Rs, self.ks.T))
        MKs_flat = self.MKs.reshape(len(self.ks), -1)
        MRs_flat = np.matmul(phase, MKs_flat * weights[:, None])
        return MRs_flat.reshape(len(Rs), *self.MKs.shape[1:])


class AOMatrixObj:
    """
    Tight-binding operator (matrix) loaded from DeepH format files.

    This class constructs the one-body operator (matrix) from the standard
    DeepH format data. The operator in real space (e.g. H(R) or S(R)) is
    constructed and can be Fourier transformed to the reciprocal space.

    Parameters
    ----------
    info_dir_path : str or Path
        Path to the directory containing the POSCAR, info.json and overlap.h5.

    matrix_file_path : str or Path, optional
        Path to the matrix file. Default: hamiltonian.h5 under `info_dir_path`.

    matrix_type : str, optional
        Type of the matrix. Options: "hamiltonian", "overlap", "density_matrix".
        Default: "hamiltonian".

    mats : np.ndarray, shape (N_R, N_b, N_b), optional
        Matrix in real space. If provided, the object will not load the matrix
        from the file. The R-vectors MUST be sorted to avoid bugs.

    Properties
    ----------
    lattice : np.ndarray, shape (3, 3), dtype=float
        Lattice vectors. Each row is a lattice vector.

    reciprocal_lattice : np.ndarray, shape (3, 3), dtype=float
        Reciprocal lattice vectors. Each row is a reciprocal lattice vector.

    Rijk_list : np.ndarray, shape (N_R, 3), dtype=int
        Lattice displacements for inter-cell hoppings.
        Sorted with z-index varying most slowly and x-index most rapidly
        (C-style row-major order).

    mats : np.ndarray, shape (N_R, N_b, N_b), dtype=float or complex
        Matrix in real space. mats[i, :, :] = matrix(Rijk_list[i, :]).
        N_b is the number of basis functions in the unit cell.

    atoms_quantity : int
        Number of atoms in the unit cell.

    orbits_quantity : int
        Number of orbitals in the unit cell.

    spinful : bool
        Whether the system is spinful.

    fermi_energy : float
        Fermi energy in eV.
    """

    def __init__(self, info_dir_path, matrix_file_path=None, matrix_type="hamiltonian", mats=None):
        self._get_necessary_data_path(info_dir_path, matrix_file_path, matrix_type)

        self.mats = None
        self.Rijk_list = None

        Rijk_only = mats is not None

        self.parse_data(matrix_type, Rijk_only)
        self._sort_Rijk()

        if mats is not None:
            self.mats = mats
            assert self.R_quantity == len(mats), f"Mismatch: R_quantity={self.R_quantity}, mats_len={len(mats)}"

    @classmethod
    def from_kspace(cls, info_dir_path, AOMatrixK_obj, matrix_type="hamiltonian", n_jobs=-1, parallel_k=True):
        """
        Construct a real-space AOMatrixObj from a k-space AOMatrixK object.

        This factory method performs an Inverse Fourier Transform on the
        provided `AOMatrixK_obj` to reconstruct the real-space matrices H(R)
        or S(R) projected onto the R-vectors from the info directory.

        Parameters
        ----------
        info_dir_path : str or Path
            Path to the directory containing DeepH input files.

        AOMatrixK_obj : AOMatrixK
            The source k-space matrix object.

        matrix_type : str, optional
            Type of the matrix. Default is "hamiltonian".

        n_jobs : int, optional
            Number of parallel workers. Default is -1 (auto-detect).
            - If parallel_k=True: number of R-chunks to process in parallel.
            - If parallel_k=False: number of BLAS threads for k2r.

        parallel_k : bool, optional
            Parallelization strategy. Default is True.
            - True: Multiple R-chunks in parallel, each with 1 BLAS thread.
            - False: R-chunks processed sequentially with n_jobs BLAS threads.

        Returns
        -------
        AOMatrixObj
            A new instance with real-space matrices.
        """
        import os

        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        obj = cls(info_dir_path, matrix_type=matrix_type)

        Rs = obj.Rijk_list
        if Rs is None:
            raise ValueError("Failed to initialize Rijk_list from info directory.")

        def process_r_chunk(rs_chunk):
            return AOMatrixK_obj.k2r(rs_chunk)

        if parallel_k:
            n_blas_threads = 1
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                if n_jobs == 1:
                    mats = AOMatrixK_obj.k2r(Rs)
                else:
                    if len(Rs) > 0:
                        n_chunks = n_jobs * 4
                        rs_chunks = np.array_split(Rs, n_chunks)
                        results = parallel_map(process_r_chunk, rs_chunks, n_jobs=n_jobs, desc="K to R")
                        mats = np.concatenate(results, axis=0)
                    else:
                        mats = np.zeros((0, obj.orbits_quantity, obj.orbits_quantity))
        else:
            n_blas_threads = n_jobs
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                mats = AOMatrixK_obj.k2r(Rs)

        obj.mats = mats

        if matrix_type == "overlap" and np.iscomplexobj(obj.mats):
            obj.mats = np.real(obj.mats)

        return obj

    @classmethod
    def from_data(cls, structure_dict, info_dict, matrix_data, matrix_type="hamiltonian"):
        """
        Construct an AOMatrixObj from in-memory DeepH-format data.

        This factory bypasses file I/O entirely; all path attributes of the
        returned object are ``None``.

        For ``matrix_type="overlap"`` the input must always be **spinless**,
        and it will be expanded to ``[[S,0],[0,S]]`` when
        ``info_dict["spinful"]=True``.
        When generating the info and overlap with
        :func:`deepx_dock.compute.overlap.calc_overlap_in_memory`, call it
        with ``spinful=False`` and modify ``info_dict["spinful"]`` manually.

        Parameters
        ----------
        structure_dict : dict
            Structure data with keys:
            ``lattice`` (3, 3) [required],
            ``atomic_symbols`` (list[str], per-atom) or ``atomic_numbers`` (Natom,),
            ``frac_coords`` (Natom, 3) or ``cart_coords`` (Natom, 3).
            The missing member of each pair is derived from the one provided.
            Atoms of the same element need not be contiguous.
        info_dict : dict
            System metadata with keys:
            Required: ``spinful``, ``elements_orbital_map``.
            Cross-checked: ``atoms_quantity``, ``orbits_quantity``.
            Optional: ``orthogonal_basis`` (False), ``fermi_energy_eV`` (None),
            ``occupation`` (None).
        matrix_data : dict
            DeepH-format matrix arrays with keys:
            ``atom_pairs`` (N, 5),
            ``chunk_shapes`` (N, 2),
            ``chunk_boundaries`` (N+1,),
            ``entries`` (flattened).
        matrix_type : str, optional
            "hamiltonian", "overlap", or "density_matrix". Default: "hamiltonian".

        Returns
        -------
        AOMatrixObj
            A new instance with no file backing.
        """
        obj = cls.__new__(cls)
        obj.info_dir_path = None
        obj.poscar_path = None
        obj.info_json_path = None
        obj.matrix_path = None

        obj._apply_structure_data(structure_dict)
        obj._apply_info_data(info_dict)
        obj._parse_orbit_types()

        atom_pairs = np.asarray(matrix_data["atom_pairs"])
        chunk_boundaries = np.asarray(matrix_data["chunk_boundaries"])
        chunk_shapes = np.asarray(matrix_data["chunk_shapes"])
        entries = matrix_data["entries"]

        obj.atom_pairs = atom_pairs
        obj.bounds = chunk_boundaries
        obj.shapes = chunk_shapes

        obj.Rijk_list, obj.mats = cls._assemble_matrix_from_deeph_data(
            atom_pairs,
            chunk_boundaries,
            chunk_shapes,
            entries,
            obj.atom_num_orbits,
            obj.spinful,
            matrix_type=matrix_type,
        )
        obj._sort_Rijk()
        return obj

    @property
    def R_quantity(self):
        """Number of R-vectors."""
        return len(self.Rijk_list)

    def _get_necessary_data_path(self, info_dir_path, matrix_file_path=None, matrix_type="hamiltonian"):
        info_dir_path = Path(info_dir_path)
        self.info_dir_path = info_dir_path
        self.poscar_path = info_dir_path / DEEPX_POSCAR_FILENAME
        self.info_json_path = info_dir_path / DEEPX_INFO_FILENAME

        if matrix_file_path is not None:
            self.matrix_path = Path(matrix_file_path)
        else:
            if matrix_type == "hamiltonian":
                self.matrix_path = info_dir_path / DEEPX_HAMILTONIAN_FILENAME
            elif matrix_type == "overlap":
                self.matrix_path = info_dir_path / DEEPX_OVERLAP_FILENAME
            elif matrix_type == "density_matrix":
                self.matrix_path = info_dir_path / DEEPX_DENSITY_MATRIX_FILENAME
            else:
                raise ValueError(f"Invalid matrix_type: {matrix_type}")

    def parse_data(self, matrix_type="hamiltonian", Rijk_only=False):
        """Parse all necessary data from files."""
        self._parse_info()
        self._parse_poscar()
        self._parse_orbit_types()

        if Rijk_only:
            self._parse_matrix_S_like(Rijk_only=True)
        else:
            if matrix_type in ("hamiltonian", "density_matrix"):
                self._parse_matrix_H_like()
            elif matrix_type == "overlap":
                self._parse_matrix_S_like()
            else:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

    def _parse_info(self):
        """Parse info.json file."""
        raw_info = self._read_info_json(self.info_json_path)

        self.atoms_quantity = raw_info["atoms_quantity"]
        self.orbits_quantity = raw_info["orbits_quantity"]
        self.is_orthogonal_basis = raw_info["orthogonal_basis"]
        self.spinful = raw_info["spinful"]
        self.fermi_energy = raw_info["fermi_energy_eV"]
        self.elements_orbital_map = raw_info["elements_orbital_map"]
        self.occupation = raw_info.get("occupation", None)

    def _parse_poscar(self):
        """Parse POSCAR file."""
        raw_poscar = self._read_poscar(self.poscar_path)

        self.lattice = raw_poscar["lattice"]
        self.elements = raw_poscar["elements"]
        self.frac_coords = raw_poscar["frac_coords"]
        self.reciprocal_lattice = self.get_reciprocal_lattice(self.lattice)

    def _parse_orbit_types(self):
        """Parse orbital type information."""
        self.atom_num_orbits = [np.sum(2 * np.array(self.elements_orbital_map[el]) + 1) for el in self.elements]
        self.atom_num_orbits_cumsum = np.insert(np.cumsum(self.atom_num_orbits), 0, 0)

        expected_orbits = self.atom_num_orbits_cumsum[-1]
        assert (
            self.orbits_quantity == expected_orbits
        ), f"Orbital count mismatch: {self.orbits_quantity} (info.json) vs {expected_orbits} (POSCAR)"

    def _apply_structure_data(self, structure_dict):
        """
        Populate structure attributes from an in-memory structure dict.
        No assumption that atoms of the same element are contiguous.
        """
        lattice = np.array(structure_dict["lattice"], dtype=float)
        self.lattice = lattice
        self.reciprocal_lattice = self.get_reciprocal_lattice(lattice)

        if "atomic_symbols" in structure_dict:
            atomic_symbols = list(structure_dict["atomic_symbols"])
        elif "atomic_numbers" in structure_dict:
            atomic_symbols = [PERIODIC_TABLE_INDEX_TO_SYMBOL[int(z)] for z in structure_dict["atomic_numbers"]]
        else:
            raise KeyError("structure_dict must contain 'atomic_symbols' or 'atomic_numbers'")

        if "frac_coords" in structure_dict:
            frac_coords = np.array(structure_dict["frac_coords"], dtype=float)
        elif "cart_coords" in structure_dict:
            cart_coords = np.array(structure_dict["cart_coords"], dtype=float)
            frac_coords = cart_coords @ np.linalg.inv(lattice)
        else:
            raise KeyError("structure_dict must contain 'frac_coords' or 'cart_coords'")

        self.elements = atomic_symbols
        self.frac_coords = frac_coords
        self.atoms_quantity = len(atomic_symbols)

    def _apply_info_data(self, info_dict):
        """
        Populate info attributes from an in-memory info dict.

        Required keys: ``spinful``, ``elements_orbital_map``.
        Cross-checked: ``atoms_quantity``, ``orbits_quantity``.
        Defaulted: ``orthogonal_basis`` (False), ``fermi_energy_eV`` (None),
        ``occupation`` (None).

        Must be called after ``_apply_structure_data`` (uses ``self.elements``).
        """

        assert "spinful" in info_dict, "info_dict must contain 'spinful'"
        assert "elements_orbital_map" in info_dict, "info_dict must contain 'elements_orbital_map'"
        self.spinful = bool(info_dict["spinful"])
        self.elements_orbital_map = info_dict["elements_orbital_map"]

        self.is_orthogonal_basis = bool(info_dict.get("orthogonal_basis", False))
        _fe = info_dict.get("fermi_energy_eV", None)
        self.fermi_energy = None if _fe is None else float(_fe)
        self.occupation = info_dict.get("occupation", None)

        derived_atoms = len(self.elements)
        if "atoms_quantity" in info_dict:
            provided = int(info_dict["atoms_quantity"])
            assert (
                provided == derived_atoms
            ), f"atoms_quantity mismatch: info_dict={provided} vs structure-derived={derived_atoms}"
        self.atoms_quantity = derived_atoms

        derived_orbits = sum(int(np.sum(2 * np.array(self.elements_orbital_map[sym]) + 1)) for sym in self.elements)
        if "orbits_quantity" in info_dict:
            provided = int(info_dict["orbits_quantity"])
            assert (
                provided == derived_orbits
            ), f"orbits_quantity mismatch: info_dict={provided} vs derived={derived_orbits}"
        self.orbits_quantity = derived_orbits

    @staticmethod
    def _assemble_matrix_from_deeph_data(
        atom_pairs,
        chunk_boundaries,
        chunk_shapes,
        entries,
        atom_num_orbits,
        spinful,
        matrix_type="hamiltonian",
        Rijk_only=False,
    ):
        """
        Assemble (Rijk_list, mats) from DeepH-format HDF5 arrays.

        Note that for ``matrix_type="overlap"``, input must always be
        **spinless**, and it will be expanded to ``[[S, 0], [0, S]]`` in this
        function when ``spinful=True``.
        For other ``matrix_type``, input should match the ``spinful`` flag.

        Parameters
        ----------
        atom_pairs : np.ndarray, shape (N_edges, 5)
            Each row: [Rx, Ry, Rz, i_atom, j_atom].
        chunk_boundaries : np.ndarray, shape (N_edges + 1,)
            Chunk boundaries in ``entries``.
        chunk_shapes : np.ndarray, shape (N_edges, 2)
            Shape of each chunk.
        entries : np.ndarray
            Flattened matrix entries.
        atom_num_orbits : list[int] or np.ndarray
            Orbital count of each atom (spinless).
        spinful : bool
            Whether the output includes spin DOF.
        matrix_type : str
            "hamiltonian"/"density_matrix" (H-like) or "overlap" (S-like).
        Rijk_only : bool
            If True, only compute the unique R-vectors and return
            ``(Rijk_list, None)`` without assembling matrices.

        Returns
        -------
        Rijk_list : np.ndarray, shape (N_R, 3), dtype=int
        mats : np.ndarray, shape (N_R, N_b, N_b) or None (if Rijk_only)
        """
        if matrix_type in ("hamiltonian", "density_matrix"):
            is_overlap = False
        elif matrix_type == "overlap":
            is_overlap = True
        else:
            raise ValueError(f"Invalid matrix_type: {matrix_type}")

        R_to_idx = {}
        for ap in atom_pairs:
            R = (int(ap[0]), int(ap[1]), int(ap[2]))
            if R not in R_to_idx:
                R_to_idx[R] = len(R_to_idx)
        Rijk_list = np.array(list(R_to_idx.keys()), dtype=int)

        if Rijk_only:
            return Rijk_list, None

        if is_overlap:
            dtype = np.float64
            spin_factor = 1
        else:
            dtype = np.complex128 if spinful else np.float64
            spin_factor = 1 + spinful

        entries = np.asarray(entries, dtype=dtype)
        atom_num_orbits = np.asarray(atom_num_orbits, dtype=int)
        cumsum = np.zeros(len(atom_num_orbits) + 1, dtype=int)
        np.cumsum(atom_num_orbits, out=cumsum[1:])
        orbits_quantity = int(cumsum[-1])
        mat_dim = orbits_quantity * spin_factor

        mats = np.zeros((len(R_to_idx), mat_dim, mat_dim), dtype=dtype)

        for i_ap, ap in enumerate(atom_pairs):
            i_R = R_to_idx[(int(ap[0]), int(ap[1]), int(ap[2]))]
            i_atom, j_atom = int(ap[3]), int(ap[4])
            chunk = entries[chunk_boundaries[i_ap] : chunk_boundaries[i_ap + 1]].reshape(chunk_shapes[i_ap])

            i_sl_u = slice(cumsum[i_atom], cumsum[i_atom + 1])
            j_sl_u = slice(cumsum[j_atom], cumsum[j_atom + 1])

            if is_overlap or not spinful:
                mats[i_R][i_sl_u, j_sl_u] = chunk
            else:
                n_i = int(atom_num_orbits[i_atom])
                n_j = int(atom_num_orbits[j_atom])
                i_sl_d = slice(cumsum[i_atom] + orbits_quantity, cumsum[i_atom + 1] + orbits_quantity)
                j_sl_d = slice(cumsum[j_atom] + orbits_quantity, cumsum[j_atom + 1] + orbits_quantity)
                mats[i_R][i_sl_u, j_sl_u] = chunk[:n_i, :n_j]
                mats[i_R][i_sl_u, j_sl_d] = chunk[:n_i, n_j:]
                mats[i_R][i_sl_d, j_sl_u] = chunk[n_i:, :n_j]
                mats[i_R][i_sl_d, j_sl_d] = chunk[n_i:, n_j:]

        if is_overlap and spinful:
            _zeros = np.zeros_like(mats)
            mats = np.block([[mats, _zeros], [_zeros, mats]])

        return Rijk_list, mats

    def _parse_matrix_S_like(self, Rijk_only=False):
        """Parse overlap-like matrix (real values, no spin structure)."""
        if not Rijk_only:
            matrix_path = self.matrix_path
        else:
            matrix_path = self.info_dir_path / DEEPX_OVERLAP_FILENAME

        atom_pairs, bounds, shapes, entries = self._read_h5(matrix_path)
        self.atom_pairs = atom_pairs
        self.bounds = bounds
        self.shapes = shapes

        self.Rijk_list, mats = self._assemble_matrix_from_deeph_data(
            atom_pairs,
            bounds,
            shapes,
            entries,
            self.atom_num_orbits,
            self.spinful,
            matrix_type="overlap",
            Rijk_only=Rijk_only,
        )
        if not Rijk_only:
            self.mats = mats

    def _parse_matrix_H_like(self):
        """Parse Hamiltonian-like matrix (complex for spinful)."""
        dtype = np.complex128 if self.spinful else np.float64

        atom_pairs, bounds, shapes, entries = self._read_h5(self.matrix_path, dtype=dtype)
        self.atom_pairs = atom_pairs
        self.bounds = bounds
        self.shapes = shapes

        self.Rijk_list, self.mats = self._assemble_matrix_from_deeph_data(
            atom_pairs,
            bounds,
            shapes,
            entries,
            self.atom_num_orbits,
            self.spinful,
            matrix_type="hamiltonian",
        )

    def _build_entries_H_like(self, R_to_index=None, dtype=None):
        """
        Reverse of _parse_matrix_H_like: construct entries from atom_pairs, bounds, shapes, mats.

        Returns
        -------
        entries : np.ndarray
            Flattened matrix entries, ready to be written to HDF5.
        """
        bounds = self.bounds
        shapes = self.shapes

        if R_to_index is None:
            R_to_index = {tuple(R): i_R for i_R, R in enumerate(self.Rijk_list)}
        if dtype is None:
            dtype = np.complex128 if self.spinful else np.float64

        entries = np.empty(bounds[-1], dtype=dtype)

        for i_ap, ap in enumerate(self.atom_pairs):
            R_ijk = (ap[0], ap[1], ap[2])
            i_atom, j_atom = ap[3], ap[4]
            i_R = R_to_index[R_ijk]
            mat = self.mats[i_R]

            if self.spinful:
                _i_slice_up = slice(self.atom_num_orbits_cumsum[i_atom], self.atom_num_orbits_cumsum[i_atom + 1])
                _i_slice_dn = slice(
                    self.atom_num_orbits_cumsum[i_atom] + self.orbits_quantity,
                    self.atom_num_orbits_cumsum[i_atom + 1] + self.orbits_quantity,
                )
                _j_slice_up = slice(self.atom_num_orbits_cumsum[j_atom], self.atom_num_orbits_cumsum[j_atom + 1])
                _j_slice_dn = slice(
                    self.atom_num_orbits_cumsum[j_atom] + self.orbits_quantity,
                    self.atom_num_orbits_cumsum[j_atom + 1] + self.orbits_quantity,
                )
                _i_orb_num = self.atom_num_orbits[i_atom]
                _j_orb_num = self.atom_num_orbits[j_atom]

                chunk = np.empty((2 * _i_orb_num, 2 * _j_orb_num), dtype=dtype)
                chunk[:_i_orb_num, :_j_orb_num] = mat[_i_slice_up, _j_slice_up]
                chunk[:_i_orb_num, _j_orb_num:] = mat[_i_slice_up, _j_slice_dn]
                chunk[_i_orb_num:, :_j_orb_num] = mat[_i_slice_dn, _j_slice_up]
                chunk[_i_orb_num:, _j_orb_num:] = mat[_i_slice_dn, _j_slice_dn]
            else:
                _i_slice = slice(self.atom_num_orbits_cumsum[i_atom], self.atom_num_orbits_cumsum[i_atom + 1])
                _j_slice = slice(self.atom_num_orbits_cumsum[j_atom], self.atom_num_orbits_cumsum[j_atom + 1])
                chunk = mat[_i_slice, _j_slice]

            expected_size = bounds[i_ap + 1] - bounds[i_ap]
            flat_chunk = chunk.reshape(-1)
            assert (
                flat_chunk.shape[0] == expected_size
            ), f"Chunk size mismatch at atom_pair {i_ap}: got {flat_chunk.shape[0]}, expected {expected_size}"
            entries[bounds[i_ap] : bounds[i_ap + 1]] = flat_chunk

        return entries

    def _sort_Rijk(self):
        """Sort Rijk_list in C-style row-major order (x fastest, z slowest)."""
        tx = self.Rijk_list[:, 0]
        ty = self.Rijk_list[:, 1]
        tz = self.Rijk_list[:, 2]

        sort_indices = np.lexsort((tx, ty, tz))
        self.Rijk_list = self.Rijk_list[sort_indices]

        if self.mats is not None:
            self.mats = self.mats[sort_indices]

    @staticmethod
    def get_reciprocal_lattice(lattice):
        """
        Calculate reciprocal lattice vectors.

        Parameters
        ----------
        lattice : np.ndarray, shape (3, 3)
            Direct lattice vectors (each row is a vector).

        Returns
        -------
        reciprocal_lattice : np.ndarray, shape (3, 3)
            Reciprocal lattice vectors (each row is a vector).
        """
        a = np.array(lattice)

        volume = abs(np.dot(a[0], np.cross(a[1], a[2])))
        if np.isclose(volume, 0):
            raise ValueError("Invalid lattice: Volume is zero")

        b1 = 2 * np.pi * np.cross(a[1], a[2]) / volume
        b2 = 2 * np.pi * np.cross(a[2], a[0]) / volume
        b3 = 2 * np.pi * np.cross(a[0], a[1]) / volume

        return np.vstack([b1, b2, b3])

    @staticmethod
    def _read_h5(h5_path, dtype=np.float64):
        """
        Read matrix data from HDF5 file.

        Returns
        -------
        atom_pairs : np.ndarray, shape (N_pairs, 5)
            Each row: [R1, R2, R3, i_atom, j_atom]
        boundaries : np.ndarray, shape (N_pairs + 1,)
            Chunk boundaries in entries array.
        shapes : np.ndarray, shape (N_pairs, 2)
            Shape of each chunk.
        entries : np.ndarray
            Flattened matrix entries.
        """
        h5_path_obj = Path(h5_path)
        if not h5_path_obj.exists():
            raise FileNotFoundError(f"File not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            entries = np.array(f["entries"][:], dtype=dtype)

        return atom_pairs, boundaries, shapes, entries

    @staticmethod
    def _read_info_json(json_path):
        """Read info.json file."""
        return load_json_file(json_path)

    @staticmethod
    def _read_poscar(filename):
        """Read POSCAR file."""
        result = load_poscar_file(filename)
        elements = [elem for elem, n in zip(result["elements_unique"], result["elements_counts"]) for _ in range(n)]
        return {
            "lattice": result["lattice"],
            "elements": elements,
            "cart_coords": result["cart_coords"],
            "frac_coords": result["frac_coords"],
        }

    def r2k(self, ks):
        """
        Fourier transform from real space to reciprocal space.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3)
            k-points in fractional coordinates.

        Returns
        -------
        MKs : np.ndarray, shape (Nk, N_b, N_b)
            Matrices in reciprocal space.
        """
        phase = np.exp(2j * np.pi * np.matmul(ks, self.Rijk_list.T))
        MRs_flat = self.mats.reshape(len(self.Rijk_list), -1)
        Mks_flat = np.matmul(phase, MRs_flat)
        return Mks_flat.reshape(len(ks), *self.mats.shape[1:])

    def assert_compatible(self, other):
        """
        Assert that another AOMatrixObj is structurally compatible.

        Raises
        ------
        AssertionError
            If mismatch found.
        """
        assert self.spinful == other.spinful, "Spin mismatch"
        assert self.orbits_quantity == other.orbits_quantity, "Orbital number mismatch"
        assert self.is_orthogonal_basis == other.is_orthogonal_basis, "Basis orthogonality mismatch"

        assert np.allclose(self.lattice, other.lattice), "Lattice vector mismatch"
        assert self.elements == other.elements, "Element mismatch"
        assert np.allclose(self.frac_coords, other.frac_coords), "Fractional coordinates mismatch"
        assert np.array_equal(self.Rijk_list, other.Rijk_list), "Rijk_list mismatch"

        if self.atom_pairs is not None and other.atom_pairs is not None:
            assert np.array_equal(self.atom_pairs, other.atom_pairs), "Atom pairs storage mismatch"

        assert np.array_equal(self.atom_num_orbits_cumsum, other.atom_num_orbits_cumsum), "Orbital indexing mismatch"

        return True
