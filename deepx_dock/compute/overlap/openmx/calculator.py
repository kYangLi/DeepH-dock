"""
High-level Python interface for OpenMX overlap matrix calculation.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import scipy.sparse as sp
import h5py

from .basis import ElementBasis, BasisSet
from .overlap_core import OverlapCore

BOHR_TO_ANGSTROM = 0.529177249


def _get_openmx_m_order(L: int) -> List[int]:
    """Get m values in OpenMX order: 0, 1, -1, 2, -2, ..."""
    return [0] + [m for s in range(1, L + 1) for m in [s, -s]]


class OverlapCalculator:
    """
    OpenMX-style overlap matrix calculator.
    """

    def __init__(self, basis_database_dir: str | Path, lmax_gaunt: int = 6):
        self.basis_database_dir = Path(basis_database_dir)
        if not self.basis_database_dir.is_dir():
            raise NotADirectoryError(f"Basis database directory not found: {basis_database_dir}")

        self.lmax_gaunt = lmax_gaunt
        self._core = OverlapCore(lmax_gaunt)

        self._is_structure_set = False
        self._is_basis_set = False

        self._positions: Optional[np.ndarray] = None
        self._species_ids: Optional[np.ndarray] = None
        self._cell: Optional[np.ndarray] = None

        self._element_bases: Dict[int, ElementBasis] = {}
        self._basis_names: Dict[int, str] = {}
        self._orbital_map: Dict[int, List[int]] = {}
        self._atom_basis_info: List[Dict] = []

    def set_structure(self, positions: np.ndarray, species_ids: np.ndarray, cell: Optional[np.ndarray] = None):
        positions = np.asarray(positions, dtype=np.float64)
        species_ids = np.asarray(species_ids, dtype=np.int32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N_atom, 3)")
        if species_ids.ndim != 1:
            raise ValueError("species_ids must be 1-dimensional")
        if len(species_ids) != len(positions):
            raise ValueError("positions and species_ids must have same length")

        self._positions = positions
        self._species_ids = species_ids
        self._cell = np.zeros((3, 3), dtype=np.float64) if cell is None else np.asarray(cell, dtype=np.float64)
        self._is_structure_set = True

    def set_basis(self, basis_names: Dict[int, str], orbital_map: Optional[Dict[int, List[int]]] = None):
        if not self._is_structure_set:
            raise RuntimeError("Structure not set. Call set_structure() first.")

        self._basis_names = basis_names
        self._orbital_map = orbital_map if orbital_map else {}

        unique_species = set(self._species_ids)

        for species_id in unique_species:
            if species_id not in basis_names:
                raise ValueError(f"No basis set specified for element {species_id}")

            symbol = self._get_element_symbol(species_id)
            h5_file = self.basis_database_dir / f"{symbol}.h5"

            if not h5_file.exists():
                raise FileNotFoundError(f"Basis file not found: {h5_file}")

            self._element_bases[species_id] = ElementBasis.load_h5(str(h5_file))

        self._build_atom_basis_info()
        self._is_basis_set = True

    def _build_atom_basis_info(self):
        self._atom_basis_info = []

        for i, species_id in enumerate(self._species_ids):
            basis_name = self._basis_names[species_id]
            basis = self._element_bases[species_id].get_basis_set(basis_name)

            if species_id in self._orbital_map:
                orbital_list = self._orbital_map[species_id]
            else:
                lmax = basis.metadata.lmax
                num_mu = basis.metadata.num_mu
                orbital_list = []
                for L in range(lmax + 1):
                    orbital_list.extend([L] * num_mu)

            n_basis = sum(2 * L + 1 for L in orbital_list)

            self._atom_basis_info.append(
                {
                    "atom_idx": i,
                    "species_id": species_id,
                    "basis": basis,
                    "basis_name": basis_name,
                    "n_basis": n_basis,
                    "orbital_list": orbital_list,
                }
            )

    def compute(self, cutoff: float = 15.0) -> sp.spmatrix:
        rows, cols, values = self._compute_overlap(cutoff)
        n_basis = self.total_basis_size
        return sp.csr_matrix((values, (rows, cols)), shape=(n_basis, n_basis))

    @property
    def total_basis_size(self) -> int:
        if not self._is_structure_set or not self._is_basis_set:
            return 0
        return sum(info["n_basis"] for info in self._atom_basis_info)

    def _compute_overlap(self, cutoff: float) -> Tuple[List[int], List[int], List[float]]:
        rows = []
        cols = []
        values = []

        cutoff_bohr = cutoff / BOHR_TO_ANGSTROM
        positions_bohr = np.asarray(self._positions) / BOHR_TO_ANGSTROM

        lmax = max(self._element_bases[s].get_basis_set(n).metadata.lmax for s, n in self._basis_names.items())
        self._core.precompute_gaunt(lmax)

        k_max = 20.0
        n_k = 200
        k_grid = np.linspace(0, k_max, n_k)[1:]

        atom_basis_info = self._atom_basis_info

        for i, info_i in enumerate(atom_basis_info):
            basis_i = info_i["basis"]
            pos_i = positions_bohr[i]
            offset_i = sum(atom_basis_info[j]["n_basis"] for j in range(i))
            orbital_list_i = info_i["orbital_list"]

            for j, info_j in enumerate(atom_basis_info):
                basis_j = info_j["basis"]
                pos_j = positions_bohr[j]
                offset_j = sum(atom_basis_info[k]["n_basis"] for k in range(j))
                orbital_list_j = info_j["orbital_list"]

                R_vec = pos_j - pos_i
                R = np.linalg.norm(R_vec)

                if R > cutoff_bohr:
                    continue

                if R < 1e-10:
                    S_block = self._compute_overlap_same_atom(orbital_list_i)
                else:
                    S_full = self._core.compute_atom_pair_overlap(basis_i, basis_j, R_vec, k_grid)
                    S_block = self._extract_selected_orbitals(S_full, basis_i, basis_j, orbital_list_i, orbital_list_j)

                for ii in range(S_block.shape[0]):
                    for jj in range(S_block.shape[1]):
                        val = S_block[ii, jj]
                        if abs(val) > 1e-15:
                            rows.append(offset_i + ii)
                            cols.append(offset_j + jj)
                            values.append(val)

        return rows, cols, values

    def _compute_overlap_same_atom(self, orbital_list: List[int]) -> np.ndarray:
        n_basis = sum(2 * L + 1 for L in orbital_list)
        S = np.eye(n_basis)
        return S

    def _extract_selected_orbitals(
        self,
        S_full: np.ndarray,
        basis_i: BasisSet,
        basis_j: BasisSet,
        orbital_list_i: List[int],
        orbital_list_j: List[int],
    ) -> np.ndarray:
        idx_map_i = self._build_orbital_index_map(basis_i, orbital_list_i)
        idx_map_j = self._build_orbital_index_map(basis_j, orbital_list_j)

        n_selected_i = sum(2 * L + 1 for L in orbital_list_i)
        n_selected_j = sum(2 * L + 1 for L in orbital_list_j)

        S_selected = np.zeros((n_selected_i, n_selected_j))

        for sel_i, full_i in idx_map_i.items():
            for sel_j, full_j in idx_map_j.items():
                S_selected[sel_i, sel_j] = S_full[full_i, full_j]

        return S_selected

    def _build_orbital_index_map(self, basis: BasisSet, orbital_list: List[int]) -> Dict[int, int]:
        num_mu = basis.metadata.num_mu

        def get_full_index(L: int, mu: int, m: int) -> int:
            idx = 0
            for l in range(L):
                idx += num_mu * (2 * l + 1)
            idx += mu * (2 * L + 1)
            m_list = _get_openmx_m_order(L)
            idx += m_list.index(m)
            return idx

        idx_map = {}
        sel_idx = 0
        mu_counts = {}

        for L in orbital_list:
            if L not in mu_counts:
                mu_counts[L] = 0
            mu = mu_counts[L]
            mu_counts[L] += 1

            for m in _get_openmx_m_order(L):
                full_idx = get_full_index(L, mu, m)
                idx_map[sel_idx] = full_idx
                sel_idx += 1

        return idx_map

    @staticmethod
    def _get_element_symbol(atomic_number: int) -> str:
        symbols = {
            1: "H",
            2: "He",
            3: "Li",
            4: "Be",
            5: "B",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            10: "Ne",
            11: "Na",
            12: "Mg",
            13: "Al",
            14: "Si",
            15: "P",
            16: "S",
            17: "Cl",
            18: "Ar",
            19: "K",
            20: "Ca",
            21: "Sc",
            22: "Ti",
            23: "V",
            24: "Cr",
            25: "Mn",
            26: "Fe",
            27: "Co",
            28: "Ni",
            29: "Cu",
            30: "Zn",
            31: "Ga",
            32: "Ge",
            33: "As",
            34: "Se",
            35: "Br",
            36: "Kr",
            37: "Rb",
            38: "Sr",
            39: "Y",
            40: "Zr",
            41: "Nb",
            42: "Mo",
            43: "Tc",
            44: "Ru",
            45: "Rh",
            46: "Pd",
            47: "Ag",
            48: "Cd",
            49: "In",
            50: "Sn",
            51: "Sb",
            52: "Te",
            53: "I",
            54: "Xe",
            55: "Cs",
            56: "Ba",
            57: "La",
            58: "Ce",
            59: "Pr",
            60: "Nd",
            61: "Pm",
            62: "Sm",
            63: "Eu",
            64: "Gd",
            65: "Tb",
            66: "Dy",
            67: "Ho",
            68: "Er",
            69: "Tm",
            70: "Yb",
            71: "Lu",
            72: "Hf",
            73: "Ta",
            74: "W",
            75: "Re",
            76: "Os",
            77: "Ir",
            78: "Pt",
            79: "Au",
            80: "Hg",
            81: "Tl",
            82: "Pb",
            83: "Bi",
        }
        return symbols.get(atomic_number, f"X{atomic_number}")

    def save_to_h5(self, filepath: str | Path, S: sp.spmatrix):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        S_coo = S.tocoo()
        with h5py.File(filepath, "w") as f:
            f.create_dataset("entries", data=S_coo.data, compression="gzip")
            f.create_dataset("row_indices", data=S_coo.row, compression="gzip")
            f.create_dataset("col_indices", data=S_coo.col, compression="gzip")
            f.attrs["shape"] = S.shape
