from pathlib import Path
import numpy as np
import collections
import json
import h5py
from scipy.io import FortranFile
from scipy.sparse import csr_matrix

from tqdm import tqdm

from functools import partial
from joblib import Parallel, delayed

from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.misc import get_data_dir_lister

SIESTA_LOG_FILENAME = "SIESTA.log"
SIESTA_ORB_INDX_FILE_Extension = "ORB_INDX"
SIESTA_EIG_FILE_Extension = "EIG"
SIESTA_XV_FILE_Extension = "XV"
SIESTA_HSX_FILE_Extension = "HSX"
SIESTA_DM_FILE_Extension = "DM"

HARTREE_TO_EV = 27.2113845
BOHR_TO_ANGSTROM = 0.529177249

"""
Transformation matrix from SIESTA real spherical harmonics to wikipedia real spherical harmonics:
https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

https://www.home.uni-osnabrueck.de/apostnik/Software/Spher_Harmon.pdf
"""
SIESTA_BASIS_ORDER = {
    0: [0],
    1: [0, 1, 2],
    2: [0, 1, 2, 3, 4],
    3: [0, 1, 2, 3, 4, 5, 6],
}
SIESTA_BASIS_PARITY = {
    0: [1],
    1: [-1, 1, -1],
    2: [1, -1, 1, -1, 1],
    3: [-1, 1, -1, 1, -1, 1, -1],
}


def validation_check_siesta(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if SIESTA_LOG_FILENAME in all_files:
        yield prev_dirname


class SIESTADatasetTranslator:
    def __init__(
        self,
        siesta_data_dir,
        deeph_data_dir,
        export_S=True,
        export_H=True,
        export_rho=False,
        export_r=False,
        n_jobs=1,
        n_tier=0,
    ):
        self.siesta_data_dir = Path(siesta_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.export_S = export_S
        self.export_H = export_H
        self.export_rho = export_rho
        self.export_r = export_r
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.siesta_data_dir.is_dir(), f"{siesta_data_dir} is not a directory"
        self.deeph_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_siesta_to_deeph(self):
        worker = partial(
            self.transfer_one_siesta_to_deeph,
            siesta_path=self.siesta_data_dir,
            deeph_path=self.deeph_data_dir,
            export_S=self.export_S,
            export_H=self.export_H,
            export_rho=self.export_rho,
            export_r=self.export_r,
        )
        data_dir_lister = get_data_dir_lister(
            self.siesta_data_dir, self.n_tier, validation_check_siesta
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name) for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def transfer_one_siesta_to_deeph(
        dir_name: str, siesta_path: Path, deeph_path: Path,
        export_S=True, export_H=True, export_rho=False, export_r=False,
    ):
        try:
            siesta_dir_path = siesta_path / dir_name
            if not siesta_dir_path.is_dir():
                return
            deeph_dir_path = deeph_path / dir_name
            deeph_dir_path.mkdir(parents=True, exist_ok=True)
            #
            reader = SIESTAReader(siesta_dir_path, deeph_dir_path)
            reader.dump_data(
                export_S=export_S,
                export_H=export_H,
                export_rho=export_rho,
                export_r=export_r,
            )
        except Exception as e:
            print(f"Error in {dir_name}: {e}")


class SIESTAReader:
    def __init__(self, siesta_path, deeph_path):
        self.siesta_path = Path(siesta_path)
        self.deeph_path = Path(deeph_path)

    def dump_data(self, export_S=True, export_H=True, export_rho=False, export_r=False):
        # Read necessary info.
        self._read_info()
        self._read_hsx()
        self.matrix_info = None
        # Dump data
        self._dump_info_json()
        self._dump_poscar()
        if export_S:
            self._get_matrix_info("overlaps")
            self._slice("overlaps")
            self._dump_matrix("overlaps", DEEPX_OVERLAP_FILENAME)
        if export_H:
            self._get_matrix_info("hamiltonians")
            self._slice("hamiltonians", consider_spin=True)
            self._dump_matrix(
                "hamiltonians", DEEPX_HAMILTONIAN_FILENAME, consider_spin=True
            )
        if export_rho:
            self._read_rho()
            self._get_matrix_info("density_matrixs")
            self._slice("density_matrixs", consider_spin=True)
            self._dump_matrix(
                "density_matrixs", DEEPX_DENSITY_MATRIX_FILENAME, consider_spin=True
            )
        if export_r:
            raise NotImplementedError("Position matrix export is not implemented yet.")

    def _read_info(self):
        self._read_basic_info()
        self._get_fermi_energy()
        self._get_structure()
        self._get_orb_index()

    def _read_basic_info(self):
        siesta_log_path = self.siesta_path / SIESTA_LOG_FILENAME
        self.siesta_version = None
        with open(siesta_log_path, "r") as f:
            for line in f:
                if ("Version" in line) and (self.siesta_version is None):
                    self.siesta_version = line.split()[-1]
                elif "System Label" in line:
                    self.slabel = line.split()[-1]
                elif "Number of spin components" in line:
                    self.nspin = int(line.split()[-1])
                    self.spinful = self.nspin > 1
        if self.siesta_version is not None:
            assert self.siesta_version.split(".")[0] == "5", (
                "Only Siesta version 5.x is supported."
            )

    def _get_fermi_energy(self):
        eig_name = f"{self.slabel}.{SIESTA_EIG_FILE_Extension}"
        siesta_eig_path = self.siesta_path / eig_name
        
        if siesta_eig_path.exists():
            with open(siesta_eig_path) as eig_f:
                self.fermi_energy = float(eig_f.readline().strip())
        else:
            self.fermi_energy = None
            print(f'WARN in {self.siesta_path}: Cannot find fermi energy')

    def _get_structure(self):
        xv_name = f"{self.slabel}.{SIESTA_XV_FILE_Extension}"
        siesta_xv_path = self.siesta_path / xv_name
        with open(siesta_xv_path, "r") as f:
            xv_info = f.readlines()
        self.lattice = (
            np.genfromtxt(xv_info[:3], dtype=float, usecols=(0, 1, 2))
            * BOHR_TO_ANGSTROM
        )
        xv_info = np.genfromtxt(xv_info[4:])
        self.elements = xv_info[:, 1].astype(int)
        self.cart_coords = xv_info[:, 2:5] * BOHR_TO_ANGSTROM
        self.atoms_quantity = len(self.cart_coords)
        self.atom_elem_order_dict = dict(collections.Counter(self.elements))
        self.atom_elem_order_dict = {
            PERIODIC_TABLE_INDEX_TO_SYMBOL[k]: []
            for k in self.atom_elem_order_dict.keys()
        }
        for i, elem in enumerate(self.elements):
            elem_symbol = PERIODIC_TABLE_INDEX_TO_SYMBOL[elem]
            self.atom_elem_order_dict[elem_symbol].append(i)
        self.cart_coords_ordered = []
        for cart_coords_order in self.atom_elem_order_dict.values():
            self.cart_coords_ordered.extend(cart_coords_order)

    def _get_orb_index(self):
        orb_name = f"{self.slabel}.{SIESTA_ORB_INDX_FILE_Extension}"
        siesta_orb_indx_path = self.siesta_path / orb_name
        with open(siesta_orb_indx_path, "r") as f:
            orb_info = f.readlines()
            line = orb_info[0]
            self.orbits_quantity = int(line.split()[0])
            self.supercell_orbits_quantity = int(line.split()[1])
            self.number_supercells = int(
                self.supercell_orbits_quantity / self.orbits_quantity
            )
        self.orb_R_info = np.genfromtxt(
            orb_info[3:], dtype=int, skip_footer=17, usecols=(12, 13, 14)
        )
        self.orb_R_info, first_indices = np.unique(
            self.orb_R_info, axis=0, return_index=True
        )
        self.orb_R_info = self.orb_R_info[np.argsort(first_indices)]
        self.orb_R_info = [tuple(R) for R in self.orb_R_info.tolist()]
        orb_indx = np.genfromtxt(
            orb_info[3 : 3 + self.orbits_quantity], dtype=int, filling_values=-100
        )
        self.elem_orb_map = {}
        self.elem_orb_sort = {}
        self.elem_orb_parity = {}
        unique_elements, first_indices = np.unique(self.elements, return_index=True)
        mask = np.isin(orb_indx[:, 1], first_indices + 1)

        for elem in unique_elements.tolist():
            self.elem_orb_map[elem] = []
            self.elem_orb_sort[elem] = []
            self.elem_orb_parity[elem] = []
        orb_indx = orb_indx[mask]
        n_unique_orb = len(orb_indx)
        idx = 0
        while True:
            orb = orb_indx[idx]
            ia = orb[1]
            elem = self.elements[ia - 1]
            orb_l = orb[6].item()
            self.elem_orb_map[elem].append(orb_l)
            orb_lm_order = np.array(SIESTA_BASIS_ORDER[orb_l])
            # plus 10000*l + 100*n
            orb_lm_order += orb_l * 10000 + self.elem_orb_map[elem].count(orb_l) * 100
            self.elem_orb_sort[elem].extend(orb_lm_order)
            self.elem_orb_parity[elem].extend(SIESTA_BASIS_PARITY[orb_l])
            idx += 2 * orb_l + 1
            if idx >= n_unique_orb:
                break
        for elem in self.elem_orb_sort.keys():
            self.elem_orb_sort[elem] = np.argsort(self.elem_orb_sort[elem])
            self.elem_orb_parity[elem] = np.array(self.elem_orb_parity[elem])
        site_norbits = []
        for elem in self.elements:
            orbital_types = self.elem_orb_map[elem]
            site_norbits.append(sum(orbital_types) * 2 + len(orbital_types))
        self.site_norbits_cumsum = np.concatenate(([0], np.cumsum(site_norbits)))

    def _get_matrix_info(self, item):
        if self.matrix_info:
            return
        value = getattr(self, item)[0]
        atom_pairs = {R: [] for R in self.orb_R_info}
        atom_col_starts = self.site_norbits_cumsum[:-1]
        atom_col_ends = self.site_norbits_cumsum[1:]
        atom_site_norbits = atom_col_ends - atom_col_starts
        for iR, R in enumerate(self.orb_R_info):
            start_col_global = iR * self.orbits_quantity
            end_col_global = (iR + 1) * self.orbits_quantity
            matrix_block = value[:, start_col_global:end_col_global]
            indptr = matrix_block.indptr
            indices = matrix_block.indices

            for ia in range(self.atoms_quantity):
                start_row = self.site_norbits_cumsum[ia]
                end_row = self.site_norbits_cumsum[ia + 1]
                if indptr[start_row] == indptr[end_row]:
                    continue
                start_ptr = indptr[start_row]
                end_ptr = indptr[end_row]
                nonzero_cols_in_ia_block = indices[start_ptr:end_ptr]
                cols_expanded = nonzero_cols_in_ia_block[:, np.newaxis]

                overlap_matrix = (cols_expanded >= atom_col_starts) & (
                    cols_expanded < atom_col_ends
                )

                connected_ja_indices = np.where(np.any(overlap_matrix, axis=0))[0]

                for ja in connected_ja_indices:
                    atom_pairs[R].append((ia, ja.item()))
        self.atom_pairs = atom_pairs  # not ordered
        num_keys = sum(len(v) for v in atom_pairs.values())
        atom_pairs = np.zeros((num_keys, 5), dtype=np.int64)
        chunk_shapes = np.zeros((num_keys, 2), dtype=np.int64)
        chunk_boundaries = np.zeros(num_keys + 1, dtype=np.int64)
        i = 0
        for R, pairs in self.atom_pairs.items():
            for ia, ja in pairs:
                ta = self.cart_coords_ordered.index(ia)
                tb = self.cart_coords_ordered.index(ja)
                atom_pairs[i, :] = np.array(R + (ta, tb), dtype=np.int64)
                shape = (atom_site_norbits[ia], atom_site_norbits[ja])
                chunk_shapes[i, :] = np.array(shape, dtype=np.int64)
                i += 1
        chunk_boundaries[1:] = np.cumsum(chunk_shapes[:, 0] * chunk_shapes[:, 1])
        self.matrix_info = {
            "atom_pairs": atom_pairs,
            "chunk_shapes": chunk_shapes,
            "chunk_boundaries": chunk_boundaries,
        }

    def _read_hsx(self):
        hsx_name = f"{self.slabel}.{SIESTA_HSX_FILE_Extension}"
        siesta_hsx_path = self.siesta_path / hsx_name
        f = FortranFile(siesta_hsx_path, "r")
        version = f.read_ints()[0]  # version of HSX file
        if version > 2:
            print(f"WARN in {self.siesta_path}: The HSX file version is {version}, which is not tested yet. BE CAREFUL!")
        is_dp = f.read_ints()[0]  # whether data is double precision
        if is_dp == 0:
            raise NotImplementedError(
                "Only double-precision HSX files are supported."
            )
        tmpt = f.read_ints()  # na_u, no_u, spin, species, nscx, nscy, nscz
        na_u = tmpt[0]
        assert self.nspin == tmpt[2], "The number of spin components is inconsistent."
        if self.nspin not in [1, 8]:
            raise NotImplementedError(
                "Only non-spin-polarized and fully spin-polarized (SOC) calculations are supported."
            )
        nspecies = tmpt[3]
        nsc = tmpt[4:7]
        if np.allclose(nsc, np.array([1,1,1])):
            print(f"WARN in {self.siesta_path}: The system appears to be a cluster! If you are calculating extended systems, please set `ForceAuxCell true` in your fdf file and run SIESTA again.")
        tmpt = f.read_reals()  # 0~8: lattice parameter/Bohr, 9: fermi level/au, 10: total charge, 11: electronic temperature
        tmpt = f.read_record(dtype=np.byte) # isc(nscx*nscy*nscz,3), xa(na_u,3)/Bohr in column-major, isa(na_u), lasto(na_u)
        isc = np.frombuffer(tmpt[:4*3*nsc[0]*nsc[1]*nsc[2]], dtype=np.int32).reshape(nsc[0]*nsc[1]*nsc[2], 3)
        tmpt = tmpt[4*3*nsc[0]*nsc[1]*nsc[2]:]
        xa = np.frombuffer(tmpt[:8*3*na_u], dtype=np.float64).reshape(na_u, 3) * BOHR_TO_ANGSTROM
        tmpt = tmpt[8*3*na_u:]
        tmpt = np.frombuffer(tmpt[:], dtype=np.int32)
        isa = tmpt[:na_u]
        lasto = tmpt[na_u:2*na_u]
        tmpt = f.read_reals()
        for _ in range(nspecies):
            tmpt = f.read_ints()  # orbital infos of species
        if version > 1:
            tmpt = f.read_record(*(['i4',]*9+['f8',]*3))  # k-point record: k_cell(3x3 int), k_displ(3 real)
        numh = f.read_ints()  # numh
        assert len(numh) == self.orbits_quantity, f"The number of rows read from HSX file ({len(numh)}) is inconsistent with the number of orbitals ({self.orbits_quantity})."
        numh = np.cumsum(numh)
        numh = np.append(np.array([0]), numh)
        listh = []
        for i in range(self.orbits_quantity):
            col_idx = f.read_ints()
            listh.append(col_idx)
            assert len(col_idx) == numh[i+1] - numh[i], f"The number of column indexes does not match ({len(col_idx)} != {numh[i+1] - numh[i]})."
        listh = np.concatenate(listh)
        listh -= 1  # from 1-based to 0-based
        n_listh = len(listh)
        assert n_listh == numh[-1], f"The number of matrix elements does not match ({n_listh} != {numh[-1]})."
        self.hamiltonians = np.empty(self.nspin * n_listh, dtype=np.float64)
        cur_pos = 0
        for _ in range(self.nspin):
            for _ in range(self.orbits_quantity):
                ham_tmpt = f.read_reals()
                next_pos = cur_pos + len(ham_tmpt)
                self.hamiltonians[cur_pos:next_pos] = ham_tmpt
                cur_pos = next_pos
        cur_pos = 0
        self.overlaps = np.empty(n_listh, dtype=np.float64)
        for _ in range(self.orbits_quantity):
            ovlp_tmpt = f.read_reals()
            next_pos = cur_pos + len(ovlp_tmpt)
            self.overlaps[cur_pos:next_pos] = ovlp_tmpt
            cur_pos = next_pos
        f.close()
        self.overlaps = [
            csr_matrix(
                (self.overlaps, listh, numh),
                shape=(self.orbits_quantity, self.supercell_orbits_quantity),
            )
        ]
        self.hamiltonians *= HARTREE_TO_EV / 2
        dup_hamiltonians = []  # stores nspin csr matrix of hamiltonians
        for ispin in range(self.nspin):
            dup_hamiltonians.append(
                csr_matrix(
                    (
                        self.hamiltonians[ispin * n_listh : (ispin + 1) * n_listh],
                        listh,
                        numh,
                    ),
                    shape=(self.orbits_quantity, self.supercell_orbits_quantity),
                )
            )
        self.hamiltonians = dup_hamiltonians

    def _read_rho(self):
        rho_name = f"{self.slabel}.{SIESTA_DM_FILE_Extension}"
        siesta_dm_path = self.siesta_path / rho_name
        f = FortranFile(siesta_dm_path, "r")
        tmpt = f.read_ints()  # no_u, nspin, Rx, Ry, Rz
        assert self.nspin == tmpt[1], "The number of spin components is inconsistent."
        if self.nspin not in [1, 8]:
            raise NotImplementedError(
                "Only non-spin-polarized and fully spin-polarized (SOC) calculations are supported."
            )
        numh = f.read_ints()  # numh
        numh = np.cumsum(numh)
        numh = np.append(np.array([0]), numh)
        listh = np.empty(0, dtype=int)
        for _ in range(self.orbits_quantity):
            listh = np.append(listh, f.read_ints())
        listh -= 1  # from 1-based to 0-based
        n_listh = len(listh)
        self.density_matrixs = np.empty(self.nspin * n_listh, dtype=np.float64)
        cur_pos = 0
        for _ in range(self.nspin):
            for _ in range(self.orbits_quantity):
                den_tmpt = f.read_reals()
                next_pos = cur_pos + len(den_tmpt)
                self.density_matrixs[cur_pos:next_pos] = den_tmpt
                cur_pos = next_pos
        f.close()
        dup_density_matrixs = []  # stores nspin csr matrix of density_matrixs
        for ispin in range(self.nspin):
            dup_density_matrixs.append(
                csr_matrix(
                    (
                        self.density_matrixs[ispin * n_listh : (ispin + 1) * n_listh],
                        listh,
                        numh,
                    ),
                    shape=(self.orbits_quantity, self.supercell_orbits_quantity),
                )
            )
        self.density_matrixs = dup_density_matrixs

    def _slice(self, item, consider_spin=False):
        value = getattr(self, item)
        nspin = self.nspin if consider_spin else 1
        item_tmpts = []
        # slice each of the 8 blocks
        for ispin in range(nspin):
            value_tmpt = value[ispin]
            item_tmpt = {}
            for iR, R in enumerate(self.orb_R_info):
                start_col_global = iR * self.orbits_quantity
                for (ia, jb) in self.atom_pairs[R]:
                    elem_a = self.elements[ia]
                    matrix_slice_i = slice(
                        self.site_norbits_cumsum[ia],
                        self.site_norbits_cumsum[ia + 1],
                    )
                    elem_b = self.elements[jb]
                    matrix_slice_j = slice(
                        start_col_global + self.site_norbits_cumsum[jb],
                        start_col_global + self.site_norbits_cumsum[jb + 1],
                    )
                    ta = self.cart_coords_ordered.index(ia)
                    tb = self.cart_coords_ordered.index(jb)
                    this_key = R + (ta, tb)
                    this_item = value_tmpt[matrix_slice_i, matrix_slice_j]
                    this_item = this_item.toarray()
                    # parity
                    orb_parity_a = self.elem_orb_parity[elem_a]
                    orb_parity_b = self.elem_orb_parity[elem_b]
                    this_item *= orb_parity_a[:, None]
                    this_item *= orb_parity_b[None, :]
                    # sort
                    orb_sort_a = self.elem_orb_sort[elem_a]
                    orb_sort_b = self.elem_orb_sort[elem_b]
                    item_tmpt[this_key] = this_item[np.ix_(orb_sort_a, orb_sort_b)]
            item_tmpts.append(item_tmpt)
        # collect 8 blocks into a single block
        # | H(0) + i*H(4),  H(2) - i*H(3) |
        # | H(6) + i*H(7),  H(1) + i*H(5) |
        item_out = {}
        if nspin > 1:
            for key in item_tmpts[0].keys():
                mat_dim1, mat_dim2 = np.shape(item_tmpts[0][key])
                item_spinful = np.zeros((2 * mat_dim1, 2 * mat_dim2), dtype=np.complex128)
                item_spinful[:mat_dim1, :mat_dim2] = (
                    item_tmpts[0][key] + 1j * item_tmpts[4][key]
                )
                item_spinful[mat_dim1:, :mat_dim2] = (
                    item_tmpts[6][key] + 1j * item_tmpts[7][key]
                )
                item_spinful[:mat_dim1, mat_dim2:] = (
                    item_tmpts[2][key] - 1j * item_tmpts[3][key]
                )
                item_spinful[mat_dim1:, mat_dim2:] = (
                    item_tmpts[1][key] + 1j * item_tmpts[5][key]
                )
                item_out[key] = item_spinful
        else:
            item_out = item_tmpts[0]
        setattr(self, item, item_out)

    def _dump_info_json(self):
        file_path = self.deeph_path / DEEPX_INFO_FILENAME
        info_json = {
            "atoms_quantity": self.atoms_quantity,
            "orbits_quantity": self.orbits_quantity,
            "orthogonal_basis": False,
            "spinful": self.spinful,
            "fermi_energy_eV": self.fermi_energy,
            "elements_orbital_map": {
                PERIODIC_TABLE_INDEX_TO_SYMBOL[k]: sorted(v)
                for k, v in self.elem_orb_map.items()
            },
        }
        with open(file_path, "w") as fwj:
            json.dump(info_json, fwj)

    def _dump_poscar(self):
        file_path = self.deeph_path / DEEPX_POSCAR_FILENAME
        poscar = [
            "POSCAR generated by DeepH-dock \n",
            "1.0\n",
            "  " + " ".join(map(str, self.lattice[0])) + "\n",
            "  " + " ".join(map(str, self.lattice[1])) + "\n",
            "  " + " ".join(map(str, self.lattice[2])) + "\n",
            " ".join(self.atom_elem_order_dict.keys()) + "\n",
            " "
            + " ".join(map(lambda v: str(len(v)), self.atom_elem_order_dict.values()))
            + "\n",
            "Cartesian\n",
        ] + [
            "  " + " ".join(map(str, self.cart_coords[i])) + "\n"
            for i in self.cart_coords_ordered
        ]
        with open(file_path, "w") as fwp:
            fwp.writelines(poscar)

    def _dump_matrix(self, item, file_name: str, consider_spin=False):
        spinful = self.spinful and consider_spin
        file_path = self.deeph_path / file_name
        value = getattr(self, item)
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"] * (spinful + 1),
            "chunk_boundaries": self.matrix_info["chunk_boundaries"]
            * ((spinful + 1) ** 2),
        }
        ks = np.array(list(value.keys()), dtype=int)
        if np.allclose(ks, data['atom_pairs']):
            entries = [v.reshape(-1) for k, v in value.items()]
        else:
            print(f"WARN in {self.siesta_path}: The order of atom_pairs in {item} are not consistent with the ones in matrix_info. The atom_pairs will be reordered, which may take a long time ...")
            entries = [None] * len(data["atom_pairs"])
            atom_pairs_order = data["atom_pairs"].tolist()
            for k, v in value.items():
                k = list(k)
                index = atom_pairs_order.index(k)
                entries[index] = v.reshape(-1)
        data["entries"] = np.concatenate(entries)
        with h5py.File(file_path, "w") as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)
