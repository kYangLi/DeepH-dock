from dataclasses import dataclass
import numpy as np

from deepx_dock.misc import load_json_file, load_poscar_file
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME

class AOMatrixR: 
    """
    Properties:
    ----------
    Rs : np.array((N_R, 3), dtype=int)
        Lattice displacements for inter-cell hoppings.
        The displacements are expressed in terms of the lattice vectors.
        N_R is the number of displacements.
    
    MRs : np.array((N_R, N_b, N_b), dtype=float/complex)
        Overlap matrix in real space. MRs[i, :, :] = S(Rijk_list[i, :]).
        The dtype is float if spinful is false, otherwise the dtype is complex.
    """
    def __init__(self, Rs, MRs):
        self.Rs = Rs
        self.MRs = MRs

    def r2k(self, ks):
        # ks: (Nk, 3), Rs: (NR, 3) -> phase: (Nk, NR)
        phase = np.exp(2j * np.pi * np.matmul(ks, self.Rs.T))
        # MRs: (NR, Nb, Nb) -> flat: (NR, Nb*Nb)
        MRs_flat = self.MRs.reshape(len(self.Rs), -1)
        # (Nk, NR) @ (NR, Nb*Nb) -> (Nk, Nb*Nb)
        Mks_flat = np.matmul(phase, MRs_flat)
        return Mks_flat.reshape(len(ks), *self.MRs.shape[1:])

class AOMatrixK:
    """
    Properties:
    ----------
    ks : np.array((N_k, 3), dtype=float)
        Reciprocal lattice points for the Fourier transform.
        N_k is the number of points.
    
    MKs : np.array((N_k, N_b, N_b), dtype=float/complex)
        Overlap matrix in reciprocal space. MKs[i, :, :] = S(ks[i, :]).
        The dtype is float if spinful is false, otherwise the dtype is complex.
    """
    def __init__(self, ks, MKs):
        self.ks = ks
        self.MKs = MKs  
    
    def k2r(self, Rs, weights=None):
        # weights: (Nk,)
        if weights is None:
            weights = np.ones(len(self.ks)) / len(self.ks)
        else:
            weights = np.array(weights)
        # Rs: (NR, 3), ks: (Nk, 3) -> phase: (NR, Nk)
        phase = np.exp(-2j * np.pi * np.matmul(Rs, self.ks.T))
        # MKs: (Nk, Nb, Nb) -> flat: (Nk, Nb*Nb)
        MKs_flat = self.MKs.reshape(len(self.ks), -1)
        # (NR, Nk) @ (Nk, Nb*Nb) -> (NR, Nb*Nb)
        MRs_flat = np.matmul(phase, MKs_flat * weights[:, None])
        return MRs_flat.reshape(len(Rs), *self.MKs.shape[1:])

@dataclass
class AOMatrixObj:
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
    Rijk_list: np.array
    mats: np.array
    type: str
    spinful: bool

    @classmethod
    def from_file(cls, info_dir_path, matrix_file_path, type="hamiltonian"):
        poscar_path, info_json_path, matrix_path = cls._get_necessary_data_path(info_dir_path, matrix_file_path, type)
        #
        atoms_quantity, orbits_quantity, is_orthogonal_basis, \
        spinful, fermi_energy, elements_orbital_map, occupation = \
        cls._parse_info(info_json_path)
        #
        lattice, elements, frac_coords, reciprocal_lattice = \
        cls._parse_poscar(poscar_path)
        #
        atom_num_orbits, atom_num_orbits_cumsum = \
        cls._parse_orbital_types(elements, elements_orbital_map, orbits_quantity)
        #
        Rijk_list, mats = cls._parse_matrix(type, matrix_path, spinful, atom_num_orbits, atom_num_orbits_cumsum)
        
        return cls(Rijk_list, mats, type, spinful)

    @staticmethod
    def _get_necessary_data_path(
        info_dir_path: str | Path, file_path: str | Path | None = None, type: str = "hamiltonian"
    ):
        info_dir_path = Path(info_dir_path)
        poscar_path = info_dir_path / DEEPX_POSCAR_FILENAME
        info_json_path = info_dir_path / DEEPX_INFO_FILENAME

        matrix_path = None
        if type == "hamiltonian":
            matrix_path = info_dir_path / DEEPX_HAMILTONIAN_FILENAME
        elif type == "overlap":
            matrix_path = info_dir_path / DEEPX_OVERLAP_FILENAME
        elif type == "density_matrix":
            matrix_path = info_dir_path / DEEPX_DENSITY_MATRIX_FILENAME
        else:
            raise ValueError(f"Invalid type: {type}")

        return poscar_path, info_json_path, matrix_path

    @staticmethod
    def _parse_info(info_json_path):
        raw_info = AOMatrixObj._read_info_json(info_json_path)
        #
        atoms_quantity = raw_info["atoms_quantity"]
        orbits_quantity = raw_info["orbits_quantity"]
        is_orthogonal_basis = raw_info["orthogonal_basis"]
        spinful = raw_info["spinful"]
        fermi_energy = raw_info["fermi_energy_eV"]
        elements_orbital_map = raw_info["elements_orbital_map"]
        occupation = raw_info.get("occupation", None)
        return atoms_quantity, orbits_quantity, is_orthogonal_basis, spinful, fermi_energy, elements_orbital_map, occupation
    
    @staticmethod
    def _parse_poscar(poscar_path):
        raw_poscar = AOMatrixObj._read_poscar(poscar_path)
        #
        lattice = raw_poscar["lattice"]
        elements = raw_poscar["elements"]
        frac_coords = raw_poscar["frac_coords"]
        reciprocal_lattice = AOMatrixObj._get_reciprocal_lattice(lattice)
        return lattice, elements, frac_coords, reciprocal_lattice

    @staticmethod
    def _parse_orbital_types(elements, elements_orbital_map, orbits_quantity):
        atom_num_orbits = [
            np.sum(2 * np.array(elements_orbital_map[el]) + 1)
            for el in elements
        ]
        atom_num_orbits_cumsum = np.insert(
            np.cumsum(atom_num_orbits), 0, 0
        )
        assert orbits_quantity == atom_num_orbits_cumsum[-1], f"Number of orbitals {orbits_quantity}(info.json) and {atom_num_orbits_cumsum[-1]}(POSCAR) do not match"

        return atom_num_orbits, atom_num_orbits_cumsum

    @staticmethod
    def _parse_matrix(type, matrix_path, orbits_quantity, atom_num_orbits_cumsum, spinful):
        if type == "overlap":
            return AOMatrixObj._parse_matrix_S_like(matrix_path, orbits_quantity, atom_num_orbits_cumsum, spinful)
        elif type == "hamiltonian" or type == "density_matrix":
            return AOMatrixObj._parse_matrix_H_like(matrix_path, orbits_quantity, atom_num_orbits_cumsum, spinful)
        else:
            raise ValueError(f"Unknown matrix type: {type}")

    @staticmethod
    def _parse_matrix_S_like(matrix_path, orbits_quantity, atom_num_orbits_cumsum, spinful):
        mats_R = {}
        atom_pairs, bounds, shapes, entries = AOMatrixObj._read_h5(matrix_path)
        for i_ap, ap in enumerate(atom_pairs):
            # Gen Data
            Rijk = (ap[0], ap[1], ap[2])
            i_atom, j_atom  = ap[3], ap[4]
            if Rijk not in mats_R:
                mats_R[Rijk] = np.zeros(
                    (orbits_quantity, orbits_quantity),
                    dtype=np.float64
                )
            # Get Chunk
            _bound_slice = slice(bounds[i_ap], bounds[i_ap+1])
            _shape = shapes[i_ap]
            _S_chunk = entries[_bound_slice].reshape(_shape)
            # Fill Values
            _i_slice = slice(
                atom_num_orbits_cumsum[i_atom],
                atom_num_orbits_cumsum[i_atom+1]
            )
            _j_slice = slice(
                atom_num_orbits_cumsum[j_atom],
                atom_num_orbits_cumsum[j_atom+1]
            )
            mats_R[Rijk][_i_slice, _j_slice] = _S_chunk
        #
        R_quantity = len(mats_R)
        Rijk_list = np.zeros((R_quantity, 3), dtype=int)
        mats = np.zeros(
            (R_quantity, orbits_quantity, orbits_quantity),
            dtype=np.float64
        )
        for i_R, (Rijk, mat_val) in enumerate(mats_R.items()):
            Rijk_list[i_R] = Rijk
            mats[i_R] = mat_val
        #
        if spinful:
            mats = AOMatrixObj._spinless_to_spinful(mats)
        return Rijk_list, mats

    @staticmethod
    def _parse_matrix_H_like(matrix_path, orbits_quantity, atom_num_orbits_cumsum, spinful):
        mats_R = {}
        dtype = np.complex128 if spinful else np.float64
        atom_pairs, bounds, shapes, entries = \
            AOMatrixObj._read_h5(matrix_path, dtype=dtype)
        assert np.array_equal(atom_pairs, atom_pairs), "The atom pairs is not the same."
        bands_quantity = orbits_quantity * (1 + spinful)
        R_quantity = len(atom_pairs)
        _matrix_shape = (R_quantity, bands_quantity, bands_quantity)
        for i_ap, ap in enumerate(atom_pairs):
            # Gen Data
            R_ijk = (ap[0], ap[1], ap[2])
            i_atom, j_atom  = ap[3], ap[4]
            if R_ijk not in mats_R:
                mats_R[R_ijk] = np.zeros(
                    (bands_quantity, bands_quantity), dtype=dtype
                )
            # Get Chunk
            _bound_slice = slice(bounds[i_ap], bounds[i_ap+1])
            _shape = shapes[i_ap]
            _mat_chunk = entries[_bound_slice].reshape(_shape)
            # Fill Values
            if spinful:
                _i_slice_up = slice(
                    atom_num_orbits_cumsum[i_atom],
                    atom_num_orbits_cumsum[i_atom+1]
                )
                _i_slice_dn = slice(
                    atom_num_orbits_cumsum[i_atom] + orbits_quantity,
                    atom_num_orbits_cumsum[i_atom+1] + orbits_quantity
                )
                _j_slice_up = slice(
                    atom_num_orbits_cumsum[j_atom],
                    atom_num_orbits_cumsum[j_atom+1]
                )
                _j_slice_dn = slice(
                    atom_num_orbits_cumsum[j_atom] + orbits_quantity,
                    atom_num_orbits_cumsum[j_atom+1] + orbits_quantity
                )
                _i_orb_num = atom_num_orbits[i_atom]
                _j_orb_num = atom_num_orbits[j_atom]
                mats_R[R_ijk][_i_slice_up, _j_slice_up] = \
                    _mat_chunk[:_i_orb_num, :_j_orb_num]
                mats_R[R_ijk][_i_slice_up, _j_slice_dn] = \
                    _mat_chunk[:_i_orb_num, _j_orb_num:]
                mats_R[R_ijk][_i_slice_dn, _j_slice_up] = \
                    _mat_chunk[_i_orb_num:, :_j_orb_num]
                mats_R[R_ijk][_i_slice_dn, _j_slice_dn] = \
                    _mat_chunk[_i_orb_num:, _j_orb_num:]
            else:
                _i_slice = slice(
                    atom_num_orbits_cumsum[i_atom],
                    atom_num_orbits_cumsum[i_atom+1]
                )
                _j_slice = slice(
                    atom_num_orbits_cumsum[j_atom],
                    atom_num_orbits_cumsum[j_atom+1]
                )
                mats_R[R_ijk][_i_slice, _j_slice] = _mat_chunk
        #
        mats = np.zeros(_matrix_shape, dtype=dtype)
        for i_R in range(R_quantity):
            R_ijk = Rijk_list[i_R]
            mats[i_R] = mats_R[tuple(R_ijk)]
        return Rijk_list, mats

    @staticmethod
    def _spinless_to_spinful(mats):
        _zeros_mats = np.zeros_like(mats)
        return np.block(
            [[mats, _zeros_mats], [_zeros_mats, mats]]
        )

    @staticmethod
    def _get_reciprocal_lattice(lattice):
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
    