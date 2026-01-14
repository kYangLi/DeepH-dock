from pathlib import Path
import shutil
import json
import numpy as np
import h5py
from tqdm import tqdm

from functools import partial
from joblib import Parallel, delayed

from collections import Counter

from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME, DEEPX_VR_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX
from deepx_dock.misc import get_data_dir_lister, load_poscar_file

DEEPX_NECESSARY_FILES = {DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME}

DEEPH_LAT_FILENAME = "lat.dat"
DEEPH_RLAT_FILENAME = "rlat.dat"
DEEPH_ELEMENT_FILENAME = "element.dat"
DEEPH_SITE_POS_FILENAME = "site_positions.dat"
DEEPH_ORB_TYPE_FILENAME = "orbital_types.dat"
DEEPH_INFO_FILENAME = "info.json"
DEEPH_HAMILTONIAN_FILENAME = "hamiltonians.h5"
DEEPH_OVERLAP_FILENAME = "overlaps.h5"
DEEPH_DENSITY_MATRIX_FILENAME = "density_matrices.h5"
DEEPH_VR_FILENAME = DEEPX_VR_FILENAME
DEEPH_NECESSARY_FILES = set([
    DEEPH_LAT_FILENAME, DEEPH_ELEMENT_FILENAME, DEEPH_SITE_POS_FILENAME,
    DEEPH_ORB_TYPE_FILENAME, DEEPH_INFO_FILENAME
])
"""
Transformation matrix from OpenMX real spherical harmonics to wikipedia real spherical harmonics:
https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

https://www.openmx-square.org/openmx_man3.9/openmx3.9.pdf Table 9
"""
def BASIS_TRANS_OPENMX2WIKI(ll):
    if ll == 0:
        return np.array([0])
    elif ll == 1: # human friendly form
        return np.array([1, 2, 0])
    elif ll == 2: # human friendly form
        return np.array([2, 4, 0, 3, 1])
    elif ll == 3: # regular form
        return np.array([6, 4, 2, 0, 1, 3, 5])
    else: # similar to ll = 3 case
        return np.concatenate([np.arange(2*ll, -1, -2), np.arange(1, 2*ll, 2)])

def BASIS_TRANS_WIKI2OPENMX(ll):
    return np.argsort(BASIS_TRANS_OPENMX2WIKI(ll))


def validation_check_deeph(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if DEEPH_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname
    else:
        print(f"Skip {prev_dirname} because of missing necessary files.")


def validation_check_deepx(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if DEEPX_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname
    else:
        print(f"Skip {prev_dirname} because of missing necessary files.")


class NewDatasetTranslator:
    def __init__(self, old_data_dir, new_data_dir, n_jobs=1, n_tier=0):
        self.old_data_dir = Path(old_data_dir)
        self.new_data_dir = Path(new_data_dir)
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.old_data_dir.is_dir(), f"{old_data_dir} is not a directory"
        self.new_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_old_to_new(self):
        worker = partial(
            self.transfer_one_old_to_new,
            old_data_dir=self.old_data_dir,
            new_data_dir=self.new_data_dir,
        )
        data_dir_lister = get_data_dir_lister(
            self.old_data_dir, self.n_tier, validation_check_deeph
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def transfer_one_old_to_new(
        dir_name: str, old_data_dir: Path, new_data_dir: Path
    ):
        try:
            old_dir_path = old_data_dir / dir_name
            new_dir_path = new_data_dir / dir_name
            new_dir_path.mkdir(parents=True, exist_ok=True)
            #
            isspinful, elem_indices, orbs_save, coords_order_list = NewDatasetTranslator._transfer_old_info_to_new(
                old_dir_path, new_dir_path
            )
            NewDatasetTranslator._transfer_old_observable_to_new(
                old_dir_path, new_dir_path, isspinful, elem_indices,
                orbs_save, coords_order_list
            )
        except Exception as e:
            print(f"Error in translating {dir_name}: {e}")

    @staticmethod
    def _transfer_old_info_to_new(old_dir_path: Path, new_dir_path: Path):
        new_atomic_structure_path = new_dir_path / DEEPX_POSCAR_FILENAME
        # Old data
        lat = np.loadtxt(old_dir_path / DEEPH_LAT_FILENAME).T
        coords = np.loadtxt(old_dir_path / DEEPH_SITE_POS_FILENAME, ndmin=2).T
        elem_idx = np.loadtxt(old_dir_path / DEEPH_ELEMENT_FILENAME, ndmin=1).T
        # Process
        elem_dict = Counter(elem_idx)
        elem_dict = {int(k): int(v) for k, v in elem_dict.items()}
        # elem_dict = dict(sorted(elem_dict.items())) # Sort this dict will only make the POSCAR different. The order of one-hot vector will be decided by the key order of 'elements_orbital_map' stored in the file 'info.json'.
        elem_symbol_string = " ".join([PERIODIC_TABLE_INDEX_TO_SYMBOL[int(i)] for i in elem_dict.keys()])
        elem_num_string = " ".join([str(i) for i in elem_dict.values()])
        # Coords
        coords_save = {k: [] for k in elem_dict.keys()}
        coords_order_save = {k: [] for k in elem_dict.keys()}
        for i in range(coords.shape[0]):
            coords_save[elem_idx[i]].append(coords[i])
            coords_order_save[elem_idx[i]].append(i)
        coords_order_list = []
        for coords_order in coords_order_save.values():
            coords_order_list.extend(coords_order)
        elem_order_list = list(elem_dict.keys())
        # Transfer
        lines = []
        lines.append("POSCAR generated by DeepH-dock \n")
        lines.append("1.0\n")
        lines.append(f"  {lat[0][0]}  {lat[0][1]}  {lat[0][2]}\n")
        lines.append(f"  {lat[1][0]}  {lat[1][1]}  {lat[1][2]}\n")
        lines.append(f"  {lat[2][0]}  {lat[2][1]}  {lat[2][2]}\n")
        lines.append(f"{elem_symbol_string}\n")
        lines.append(f"{elem_num_string}\n")
        lines.append("Cartesian\n")
        for elem in elem_order_list:
            for coord in coords_save[elem]:
                lines.append(f"    {coord[0]}  {coord[1]}  {coord[2]}\n")
        # Save POSCAR
        with open(new_atomic_structure_path, "w") as fw:
            fw.writelines(lines)
        # Orbital types
        orb_types_path = old_dir_path / DEEPH_ORB_TYPE_FILENAME
        with open(orb_types_path) as f:
            orb_types_data = f.readlines()
        orb_types = [list(map(int, line.strip().split())) for line in orb_types_data]
        orbs_save = {}
        for elem_idx_i, orbs in zip(elem_idx, orb_types):
            elem = PERIODIC_TABLE_INDEX_TO_SYMBOL[int(elem_idx_i)]
            if elem not in orbs_save.keys():
                orbs_save[elem] = orbs
            else:
                if orbs_save[elem] != orbs:
                    raise ValueError("Orbital types are not consistent!")
        # Info
        new_info_path = new_dir_path / DEEPX_INFO_FILENAME
        with open(old_dir_path/DEEPH_INFO_FILENAME) as f:
            old_info = json.load(f)
        info = {
            "atoms_quantity": len(elem_idx),
            "orbits_quantity": NewDatasetTranslator.all_orb_num_of_([item for sublist in orb_types for item in sublist]),
            "orthogonal_basis": old_info.get("isorthogonal", False),
            "spinful": old_info.get("isspinful", False),
            "fermi_energy_eV": old_info.get("fermi_level", None),
            "elements_orbital_map": old_info.get("elements_orbital_map", orbs_save),
        }
        with open(new_info_path, "w") as fw:
            json.dump(info, fw)
        return old_info.get("isspinful", False), elem_idx, orbs_save, coords_order_list

    @staticmethod
    def all_orb_num_of_(orb_list):
        return sum(map(lambda x: 2*x+1, orb_list))

    @staticmethod
    def _transfer_old_observable_to_new(
        old_dir_path: Path, new_dir_path: Path, 
        isspinful, elem_indices, orbs_save, coords_order_list
    ):
        new_S_path = new_dir_path / DEEPX_OVERLAP_FILENAME
        new_H_path = new_dir_path / DEEPX_HAMILTONIAN_FILENAME
        new_Rho_path = new_dir_path / DEEPX_DENSITY_MATRIX_FILENAME
        new_potential_r_path = new_dir_path / DEEPX_VR_FILENAME
        #
        old_S_path = old_dir_path / DEEPH_OVERLAP_FILENAME
        old_H_path = old_dir_path / DEEPH_HAMILTONIAN_FILENAME
        old_Rho_path = old_dir_path / DEEPH_DENSITY_MATRIX_FILENAME
        old_potential_r_path = old_dir_path / DEEPH_VR_FILENAME
        
        #
        atom_pairs_order = None
        if old_S_path.is_file():
            atom_pairs_order = NewDatasetTranslator._transfer_old_h5_to_new(
                old_S_path,
                new_S_path,
                False,
                elem_indices,
                orbs_save,
                coords_order_list,
                atom_pairs_order=atom_pairs_order,
            )
        if old_H_path.is_file():
            atom_pairs_order = NewDatasetTranslator._transfer_old_h5_to_new(
                old_H_path,
                new_H_path,
                isspinful,
                elem_indices,
                orbs_save,
                coords_order_list,
                atom_pairs_order=atom_pairs_order,
            )
        if old_Rho_path.is_file():
            atom_pairs_order = NewDatasetTranslator._transfer_old_h5_to_new(
                old_Rho_path,
                new_Rho_path,
                isspinful,
                elem_indices,
                orbs_save,
                coords_order_list,
                atom_pairs_order=atom_pairs_order,
            )
        if old_potential_r_path.is_file():
            shutil.copy(old_potential_r_path, new_potential_r_path)

    @staticmethod
    def _transfer_old_h5_to_new(
        old_h5_path,
        new_h5_path,
        isspinful,
        elem_indices,
        orbs_save,
        coords_order_list,
        atom_pairs_order = None,
        position_order = False,
    ):
        transform_index = NewDatasetTranslator._get_transform_index(orbs_save, BASIS_TRANS_OPENMX2WIKI)
        with h5py.File(old_h5_path, "r") as f_old_h5:
            num_keys = len(f_old_h5.keys())
            key_len = len(json.loads(list(f_old_h5.keys())[0]))
            new_h5_data = {
                "atom_pairs": np.zeros((num_keys, key_len), dtype=np.int64),
                "chunk_boundaries": np.zeros(num_keys + 1, dtype=np.int64),
                "chunk_shapes": np.zeros((num_keys, 2), dtype=np.int64),
                "entries": [None] * num_keys,
            }
            i = 0
            for atom_pairs, matrix in f_old_h5.items():
                atom_pairs = json.loads(atom_pairs)
                matrix = np.array(matrix)
                old_atom1 = int(atom_pairs[3] - 1)
                old_atom2 = int(atom_pairs[4] - 1)
                atom1 = coords_order_list.index(old_atom1)
                atom2 = coords_order_list.index(old_atom2)
                atom_pairs[3] = atom1
                atom_pairs[4] = atom2
                transform_index1 = transform_index[PERIODIC_TABLE_INDEX_TO_SYMBOL[int(elem_indices[old_atom1])]]
                transform_index2 = transform_index[PERIODIC_TABLE_INDEX_TO_SYMBOL[int(elem_indices[old_atom2])]]
                matrix = NewDatasetTranslator._transform(matrix, transform_index1, transform_index2, isspinful)

                if atom_pairs_order:
                    i = atom_pairs_order.index(atom_pairs[:5])
                if position_order:
                    p = atom_pairs[5] - 1
                    index = i + num_keys // 3 * p
                    if p == 2:
                        i += 1
                else:
                    index = i
                    i += 1
                new_h5_data["atom_pairs"][index, :] = atom_pairs
                new_h5_data["chunk_shapes"][index] = matrix.shape
                new_h5_data["entries"][index] = matrix.reshape(-1)
        new_h5_data["chunk_boundaries"][1:] = np.cumsum(
            new_h5_data["chunk_shapes"][:, 0] * new_h5_data["chunk_shapes"][:, 1]
        )
        new_h5_data["entries"] = np.concatenate(new_h5_data["entries"])

        with h5py.File(new_h5_path, "w") as f_new_h5:
            for key, value in new_h5_data.items():
                f_new_h5.create_dataset(key, data=value)
        return new_h5_data["atom_pairs"].tolist()

    @staticmethod
    def _transfer_old_txt_to_new(old_txt_path, new_txt_path, coords_order_list):
        old_data = np.loadtxt(old_txt_path, ndmin=2)
        new_data = np.zeros_like(old_data)
        for i, data in enumerate(old_data):
            atom = coords_order_list.index(i)
            new_data[atom] = data
        np.savetxt(new_txt_path, new_data)

    @staticmethod
    def _get_transform_index(orbs_save, basis_trans_map):
        transform_index = {}
        for elem, orbs in orbs_save.items():
            orbital_num_list = np.array([2 * orb_l + 1 for orb_l in orbs])
            orbital_cumsum = np.concatenate((np.array([0]), np.cumsum(orbital_num_list, axis=0)), axis=0)[:-1]
            index = []
            for orb_l, orb_cum in zip(orbs, orbital_cumsum):
                index.append(basis_trans_map(orb_l) + orb_cum)
            transform_index[elem] = np.concatenate(index, axis=0)
        return transform_index

    @staticmethod
    def _transform(matrix, transform_index1, transform_index2, isspinful):
        if isspinful:
            a = matrix.shape[0] // 2
            b = matrix.shape[1] // 2
            matrix = matrix.reshape((2, a, 2, b)).transpose((0, 2, 1, 3)).reshape((4, a, b))
            matrix = matrix[:, transform_index1, :][:, :, transform_index2]
            matrix = matrix.reshape((2, 2, a, b)).transpose((0, 2, 1, 3)).reshape((2 * a, 2 * b))
            return matrix
        else:
            matrix = matrix[transform_index1, :][:, transform_index2]
            return matrix


class OldDatasetTranslator:
    def __init__(self, old_data_dir, new_data_dir, n_jobs=1, n_tier=0):
        self.old_data_dir = Path(old_data_dir)
        self.new_data_dir = Path(new_data_dir)
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.new_data_dir.is_dir(), f"{new_data_dir} is not a directory"
        self.old_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_new_to_old(self):
        worker = partial(
            self.transfer_one_new_to_old,
            old_data_dir=self.old_data_dir,
            new_data_dir=self.new_data_dir,
        )
        data_dir_lister = get_data_dir_lister(
            self.new_data_dir, self.n_tier, validation_check_deepx
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def transfer_one_new_to_old(
        dir_name: str, old_data_dir: Path, new_data_dir: Path
    ):
        try:
            old_dir_path = old_data_dir / dir_name
            old_dir_path.mkdir(parents=True, exist_ok=True)
            new_dir_path = new_data_dir / dir_name
            #
            isspinful, elem_indices, orbs_save = \
                OldDatasetTranslator._transfer_new_info_to_old(
                old_dir_path, new_dir_path
            )
            OldDatasetTranslator._transfer_new_observable_to_old(
                old_dir_path, new_dir_path, isspinful, elem_indices,
                orbs_save
            )
        except Exception as e:
            print(f"Error in translating {dir_name}: {e}")

    @staticmethod
    def _transfer_new_info_to_old(old_dir_path: Path, new_dir_path: Path):
        # Read in POSCAR
        result = load_poscar_file(new_dir_path / DEEPX_POSCAR_FILENAME)
        lat = result["lattice"].T
        elem_symbols_unique = result["elements_unique"]
        elem_counts = result["elements_counts"]
        coords = result["cart_coords"].T
        # Process POSCAR info
        elem_indices_unique = [PERIODIC_TABLE_SYMBOL_TO_INDEX[symbol] for symbol in elem_symbols_unique]
        volume = np.linalg.det(lat)
        rlat = np.zeros((3,3), dtype=float)
        rlat[:, 0] = (2*np.pi) * np.cross(lat[:, 1],lat[:, 2]) / volume
        rlat[:, 1] = (2*np.pi) * np.cross(lat[:, 2],lat[:, 0]) / volume
        rlat[:, 2] = (2*np.pi) * np.cross(lat[:, 0],lat[:, 1]) / volume
        # Read in info.json
        with open(new_dir_path / DEEPX_INFO_FILENAME, 'r') as f:
            old_info = json.load(f)
        assert "elements_orbital_map" in old_info, f"'elements_orbital_map' must exist in {new_dir_path}/{DEEPX_INFO_FILENAME}."
        orbs_save = old_info.get("elements_orbital_map")
        # Generate new info
        elem_symbols = [elem_symbol for elem_symbol, elem_count in zip(elem_symbols_unique, elem_counts) for _ in range(elem_count)]
        elem_indices = [elem_index for elem_index, elem_count in zip(elem_indices_unique, elem_counts) for _ in range(elem_count)]
        orb_types = [orbs_save[elem] for elem in elem_symbols]
        new_info = {
            "nsites": sum(elem_counts),
            "isorthogonal": False,
            "isspinful": old_info.get("spinful", False),
            "fermi_level": old_info.get("fermi_energy_eV", None),
        }
        # Dump
        np.savetxt(old_dir_path/DEEPH_LAT_FILENAME, lat, fmt="%.16f")
        np.savetxt(old_dir_path/DEEPH_RLAT_FILENAME, rlat, fmt="%.16f")
        np.savetxt(old_dir_path/DEEPH_SITE_POS_FILENAME, coords, fmt="%.16f")
        np.savetxt(old_dir_path/DEEPH_ELEMENT_FILENAME, elem_indices, fmt="%d")
        with open(old_dir_path/DEEPH_INFO_FILENAME, "w") as f:
            json.dump(new_info, f)
        with open(old_dir_path/DEEPH_ORB_TYPE_FILENAME, "w") as f:
            for orbs in orb_types:
                f.write("  ".join([str(ll) for ll in orbs])+"\n")

        return old_info.get("spinful", False), elem_indices, orbs_save

    @staticmethod
    def _transfer_new_observable_to_old(
        old_dir_path: Path, new_dir_path:Path,
        isspinful, elem_indices, orbs_save
    ):
        new_S_path = old_dir_path / DEEPH_OVERLAP_FILENAME
        new_H_path = old_dir_path / DEEPH_HAMILTONIAN_FILENAME
        new_Rho_path = old_dir_path / DEEPH_DENSITY_MATRIX_FILENAME
        new_potential_r_path = old_dir_path / DEEPH_VR_FILENAME
        #
        old_S_path = new_dir_path / DEEPX_OVERLAP_FILENAME
        old_H_path = new_dir_path / DEEPX_HAMILTONIAN_FILENAME
        old_Rho_path = new_dir_path / DEEPX_DENSITY_MATRIX_FILENAME
        old_potential_r_path = new_dir_path / DEEPX_VR_FILENAME
        #
        if old_S_path.is_file():
            OldDatasetTranslator._transfer_new_h5_to_old(
                old_S_path,
                new_S_path,
                False,
                elem_indices,
                orbs_save,
            )
        if old_H_path.is_file():
            OldDatasetTranslator._transfer_new_h5_to_old(
                old_H_path,
                new_H_path,
                isspinful,
                elem_indices,
                orbs_save,
            )
        if old_Rho_path.is_file():
            OldDatasetTranslator._transfer_new_h5_to_old(
                old_Rho_path,
                new_Rho_path,
                isspinful,
                elem_indices,
                orbs_save,
            )
        if old_potential_r_path.is_file():
            shutil.copy(old_potential_r_path, new_potential_r_path)

    @staticmethod
    def _transfer_new_h5_to_old(
        old_h5_path,
        new_h5_path,
        isspinful,
        elem_indices,
        orbs_save,
    ):
        transform_index = NewDatasetTranslator._get_transform_index(orbs_save, BASIS_TRANS_WIKI2OPENMX)
        with h5py.File(old_h5_path, "r") as f_old_h5:
            atom_pairs = np.array(f_old_h5["atom_pairs"][:])
            chunk_boundaries = np.array(f_old_h5["chunk_boundaries"][:])
            chunk_shapes = np.array(f_old_h5["chunk_shapes"][:])
            entries = np.array(f_old_h5["entries"][:])
        with h5py.File(new_h5_path, "w") as f_new_h5:
            for atom_pair, chunk_boundary, chunk_shape in zip(atom_pairs, chunk_boundaries, chunk_shapes):
                matrix = entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]].reshape(chunk_shape[0], chunk_shape[1])
                atom1 = atom_pair[3]
                atom2 = atom_pair[4]
                transform_index1 = transform_index[PERIODIC_TABLE_INDEX_TO_SYMBOL[int(elem_indices[atom1])]]
                transform_index2 = transform_index[PERIODIC_TABLE_INDEX_TO_SYMBOL[int(elem_indices[atom2])]]
                matrix = NewDatasetTranslator._transform(matrix, transform_index1, transform_index2, isspinful)
                key = f"[{atom_pair[0]}, {atom_pair[1]}, {atom_pair[2]}, {atom1+1}, {atom2+1}]"
                f_new_h5.create_dataset(key, data=matrix)
