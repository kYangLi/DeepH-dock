import numpy as np
import scipy
from itertools import accumulate
from typing import List, Dict

from pathlib import Path
import json
import h5py
import shutil

from functools import partial

from deepx_dock.parallel import parallel_map
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.misc import get_data_dir_lister

FILES_NECESSARY = set([DEEPX_HAMILTONIAN_FILENAME,])


def validation_check_H(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if FILES_NECESSARY.issubset(set(all_files)):
        yield prev_dirname


def copy_files(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for file in input_dir.iterdir():
        if (str(file) not in [DEEPX_HAMILTONIAN_FILENAME, ]) and (not (output_dir / file).is_file()):
            shutil.copyfile(input_dir/file, output_dir/file)
    for file in output_dir.iterdir():
        if (str(file) not in [DEEPX_HAMILTONIAN_FILENAME, ]) and (not (input_dir/file).is_file()):
            shutil.copyfile(output_dir/file, input_dir/file)


def merge_dict(d1, d2, tag=""):
    if len(d2) == 0:
        return d1
    for k, v in d2.items():
        tag_now = tag+"["+str(k)+"]"
        if not k in d1.keys():
            d1[k] = v
        elif not isinstance(v, dict):
            if v != d1[k]:
                raise ValueError(f"{tag_now} is not consistent with others!")
        else:
            d1[k] = merge_dict(d1[k], v, tag_now)
    return d1


def get_dataset_info(input_dir, output_dir, data_dir_lister: List[str]):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    elements_orbital_map = {}
    spinful = None
    for dir_name in data_dir_lister:
        if (input_dir / dir_name / DEEPX_INFO_FILENAME).is_file():
            info_file = input_dir / dir_name / DEEPX_INFO_FILENAME
        elif (output_dir / dir_name / DEEPX_INFO_FILENAME).is_file():
            info_file = output_dir / dir_name / DEEPX_INFO_FILENAME
        else:
            raise FileNotFoundError(f"{str(dir_name)}: No {DEEPX_INFO_FILENAME} file found in input or output directory!")
        with open(info_file, 'r') as f0:
            json_data = json.load(f0)
            elements_orbital_map = merge_dict(
                elements_orbital_map, json_data["elements_orbital_map"], 
                str(dir_name)+" [elements_orbital_map]"
            )
            if spinful is None:
                spinful = json_data["spinful"]
            elif spinful != json_data["spinful"]:
                raise ValueError(f"{str(dir_name)} [spinful] is not consistent with others!")
    return elements_orbital_map, spinful


class SingleAtomHamiltonianHandler:
    def __init__(
        self, full_dir, corrected_dir, single_atoms_dir,
        transform_offsite_blocks=False, copy_other_files=False, backward=False,
        n_jobs=-1, n_tier=0,
    ):
        if backward:
            self.input_dir, self.output_dir = corrected_dir, full_dir
        else:
            self.input_dir, self.output_dir = full_dir, corrected_dir
        self.single_atoms_dir = single_atoms_dir
        self.transform_offsite_blocks = transform_offsite_blocks
        self.copy_other_files = copy_other_files
        self.backward = backward
        self.n_jobs = n_jobs
        self.n_tier = n_tier
    
    def transfer_all(self):
        data_dir_lister = get_data_dir_lister(
            self.input_dir, self.n_tier, validation_check_H
        )
        elements_orbital_map, spinful = get_dataset_info(
            self.input_dir, self.output_dir, data_dir_lister
        )
        elements_quantity = self.get_elements_quantity(
            self.single_atoms_dir, elements_orbital_map
        )
        #
        data_dir_lister = get_data_dir_lister(
            self.input_dir, self.n_tier, validation_check_H
        )
        worker = partial(
            self.transfer_one,
            input_dir=self.input_dir, 
            output_dir=self.output_dir, 
            elements_quantities=elements_quantity,
            spinful=spinful,
            transform_offsite_blocks=self.transform_offsite_blocks, 
            copy_other_files=self.copy_other_files, 
            backward=self.backward,
        )
        parallel_map(worker, data_dir_lister, n_jobs=self.n_jobs, desc="Data")

    @staticmethod
    def get_elements_quantity(
        single_atoms_dir: str | Path, elements_orbital_map: Dict[str, List[int]]
    ):
        single_atoms_dir = Path(single_atoms_dir)
        quantities = {
            ele: {} for ele in elements_orbital_map.keys()
        }
        for ele in elements_orbital_map.keys():
            if not (single_atoms_dir / ele).exists():
                raise FileNotFoundError(f"{single_atoms_dir/ele} not found!")
            #
            with open(single_atoms_dir/ele/DEEPX_INFO_FILENAME, 'r') as f0:
                json_data = json.load(f0)
                single_atom_orbital_map = json_data["elements_orbital_map"]
                core_orbital_index = json_data.get("elements_core_orbital_index", {e: [] for e in single_atom_orbital_map.keys()})
            assert list(single_atom_orbital_map.keys()) == [ele,], f"The element in {single_atoms_dir/ele} is {single_atom_orbital_map.keys()} !"
            assert single_atom_orbital_map[ele] == elements_orbital_map[ele], f"The orbitals in {single_atoms_dir/ele} is {single_atom_orbital_map[ele]}, which is not consistent with the orbitals in dataset {elements_orbital_map[ele]} !"
            #
            split_index = [0,] + list(
                accumulate([2*ll+1 for ll in single_atom_orbital_map[ele]])
            )
            changed_index = core_orbital_index[ele]
            changed_range = np.concatenate([
                np.arange(split_index[changed_index], split_index[changed_index+1])
                for changed_index in changed_index
            ], axis=0) if len(changed_index) > 0 else []
            quantities[ele]["orbital_changed_range"] = changed_range
            #
            with h5py.File(single_atoms_dir/ele/DEEPX_HAMILTONIAN_FILENAME, 'r') as f1:
                shape = f1['chunk_shapes'][:][0]
                hamiltonian_single_atom = np.array(f1['entries'][:], dtype=np.float64).reshape(shape)
            quantities[ele]["hamiltonian"] = hamiltonian_single_atom
            with h5py.File(single_atoms_dir/ele/DEEPX_OVERLAP_FILENAME, 'r') as f2:
                shape = f2['chunk_shapes'][:][0]
                overlap_single_atom = np.array(f2['entries'][:], dtype=np.float64).reshape(shape)
            quantities[ele]["overlap"] = overlap_single_atom
            quantities[ele]["overlapinv"] = scipy.linalg.inv(overlap_single_atom)
        return quantities
    
    @staticmethod
    def transfer_one(
        dir_name, input_dir, output_dir, elements_quantities, spinful,
        transform_offsite_blocks, copy_other_files, backward, 
    ):
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            dir_name = str(dir_name)
            input_dir_path = input_dir / dir_name
            output_dir_path = output_dir / dir_name
            #
            output_dir_path.mkdir(parents=True, exist_ok=True)
            if copy_other_files:
                copy_files(input_dir_path, output_dir_path)
            #
            SingleAtomHamiltonianHandler.transfer(
                input_dir_path, output_dir_path, elements_quantities, spinful,
                transform_offsite_blocks, backward, 
            )
        except Exception as e:
            print(f"Error in {dir_name}: {e}")

    @staticmethod
    def transfer(
        input_dir, output_dir, ele_quantities, spinful,
        transform_offsite_blocks, backward, 
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        if (input_dir / DEEPX_POSCAR_FILENAME).is_file():
            POSCAR_file = input_dir / DEEPX_POSCAR_FILENAME
        elif (output_dir / DEEPX_POSCAR_FILENAME).is_file():
            POSCAR_file = output_dir / DEEPX_POSCAR_FILENAME
        else:
            raise FileNotFoundError(f"No {DEEPX_POSCAR_FILENAME} file found in input or output directory!")
        with open(POSCAR_file, "r") as f:
            lines = f.readlines()
            species = lines[5].split()
            atom_nums = [int(i) for i in lines[6].split()]
        species_list = [species[i] for i in range(len(atom_nums)) for _ in range(atom_nums[i])]
        #
        with h5py.File(input_dir/DEEPX_HAMILTONIAN_FILENAME, 'r') as f:
            atom_pairs = np.array(f['atom_pairs'][:], dtype=int)
            chunk_shapes = np.array(f['chunk_shapes'][:], dtype=int)
            chunk_boundaries = np.array(f['chunk_boundaries'][:], dtype=int)
            entries = np.array(f['entries'][:], dtype=np.float64 if not spinful else np.complex128)
        #
        if transform_offsite_blocks:
            if (input_dir / DEEPX_OVERLAP_FILENAME).is_file():
                overlap_file = input_dir / DEEPX_OVERLAP_FILENAME
            elif (output_dir / DEEPX_OVERLAP_FILENAME).is_file():
                overlap_file = output_dir / DEEPX_OVERLAP_FILENAME
            else:
                raise FileNotFoundError("No overlap file found in input or output directory.")
            with h5py.File(overlap_file, 'r') as f:
                atom_pairs_olp = np.array(f['atom_pairs'][:], dtype=int)
                chunk_shapes_olp = np.array(f['chunk_shapes'][:], dtype=int)
                chunk_boundaries_olp = np.array(f['chunk_boundaries'][:], dtype=int)
                entries_overlap = np.array(f['entries'][:], dtype=np.float64)
                assert np.allclose(atom_pairs, atom_pairs_olp)
                assert np.allclose(chunk_shapes, chunk_shapes_olp*(1+spinful))
                assert np.allclose(chunk_boundaries, chunk_boundaries_olp*(1+spinful)**2)
        #
        for i_pair, atom_pair in enumerate(atom_pairs):
            R = atom_pair[:3]
            i_atom, j_atom = atom_pair[3:5]
            specie_i, specie_j = species_list[i_atom], species_list[j_atom]
            slice_ij = slice(chunk_boundaries[i_pair], chunk_boundaries[i_pair+1])
            if R[0] == 0 and R[1] == 0 and R[2] == 0 and i_atom == j_atom:
                correction = ele_quantities[specie_i]["hamiltonian"]
            elif transform_offsite_blocks:
                slice_ij_olp = slice(chunk_boundaries_olp[i_pair], chunk_boundaries_olp[i_pair+1])
                overlap_ij = entries_overlap[slice_ij_olp].reshape(chunk_shapes_olp[i_pair])
                correction = (
                    ele_quantities[specie_i]["hamiltonian"] @ ele_quantities[specie_i]["overlapinv"] @ overlap_ij + \
                    overlap_ij @ ele_quantities[specie_j]["overlapinv"] @ ele_quantities[specie_j]["hamiltonian"]
                )
                mask = np.ones_like(correction, dtype=np.float64)
                mask[ele_quantities[specie_i]["orbital_changed_range"], :] *= 0.0
                mask[:, ele_quantities[specie_j]["orbital_changed_range"]] *= 0.0
                mask = 1.0 - mask
                correction *= mask
            else:
                continue
            if spinful:
                nz = np.zeros_like(correction)
                correction = np.block([[correction, nz], [nz, correction]])
            if backward:
                entries[slice_ij] += correction.reshape(-1)
            else:
                entries[slice_ij] -= correction.reshape(-1)
        #
        with h5py.File(output_dir/DEEPX_HAMILTONIAN_FILENAME, 'w') as f:
            f.create_dataset("atom_pairs", data=atom_pairs, dtype=int)
            f.create_dataset("chunk_shapes", data=chunk_shapes, dtype=int)
            f.create_dataset("chunk_boundaries", data=chunk_boundaries, dtype=int)
            f.create_dataset("entries", data=entries, dtype=np.float64 if not spinful else np.complex128)
