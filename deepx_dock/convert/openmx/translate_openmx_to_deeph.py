import struct
import numpy as np
import re
import collections
import json
import h5py
from pathlib import Path

from tqdm import tqdm

from functools import partial
from joblib import Parallel, delayed

from deepx_dock.convert.deeph.translate_old_dataset_to_new import BASIS_TRANS_OPENMX2WIKI
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSITION_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.misc import get_data_dir_lister

BOHR_TO_ANGSTROM = 0.529177249
HARTREE_TO_EV = 27.2113845

OPENMX_SCFOUT_FILENAME = "openmx.scfout"
OPENMX_OUT_FILENAME = "openmx.out"
OPENMX_NECESSARY_FILES = {OPENMX_SCFOUT_FILENAME, OPENMX_OUT_FILENAME}


def validation_check_openmx(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if OPENMX_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname


class BinaryFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._read_data(self.file_path)
        self.offset = 0
        
    def _read_data(self, file_path):
        with open(file_path, 'rb') as frb:
            return frb.read()
    def read(self, format_str):
        size = struct.calcsize(format_str)
        result = struct.unpack_from(format_str, self.data, self.offset)
        self.offset += size
        return result
    def skip(self, format_str):
        size = struct.calcsize(format_str)
        self.offset += size
    def reset(self):
        self.offset = 0


class TextFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lines = self._read_lines(self.file_path)
    
    def _read_lines(self, file_path):
        with open(file_path, 'r') as frp:
            return frp.readlines()
    
    def find(self, keywords, idx):
        for line in self.lines:
            if keywords in line:
                return line.strip().split()[idx]

    def find_last(self, keywords, idx):
        for line in self.lines[::-1]:
            if keywords in line:
                return line.strip().split()[idx]
    
    def find_idx(self, keywords):
        for idx, line in enumerate(self.lines):
            if keywords in line:
                return idx


def convert_orbital_string_to_list(s):
    orbital_map = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    result = []
    for match in re.findall(r'([spdfghijklmn])(\d*)', s.lower()):
        orbital, count_str = match
        count = int(count_str) if count_str else 1
        result.extend([orbital_map.index(orbital)]*count)
    return result

class OpenMXDatasetTranslator:
    """
    Translator for converting OpenMX calculation outputs to DeepH training data format.
    
    This class facilitates the conversion of electronic structure data computed by OpenMX
    into the standardized format required by DeepH for machine learning model training.
    The converter supports selective export of different components of the Hamiltonian 
    and density matrices, enabling flexible data preparation for various training scenarios.
    
    Args:
        openmx_data_dir (str): 
            Path to the directory containing OpenMX calculation outputs. The directory 
            should be organized with subdirectories for each material structure, 
            each containing the necessary OpenMX output files (e.g., openmx.scfout).
        
        deeph_data_dir (str):
            Path to the output directory where DeepH-formatted data will be stored.
        
        export_S (bool, optional):
            Whether to export overlap matrices (S). Default: True.
            Required for non-orthogonal basis set transformations.
        
        export_H (bool, optional):
            Whether to export Hamiltonian matrices (H). Default: True.
            Essential for training models that predict electronic Hamiltonian.
        
        export_rho (bool, optional):
            Whether to export density matrices (ρ). Default: False.
            Useful for training models that predict electron density.
        
        export_r (bool, optional):
            Whether to export position matrices (R). Default: False.
            Required for models that consider spatial coordinates explicitly.
        
        n_jobs (int, optional):
            Number of parallel jobs for data processing. Default: 1 (sequential).
            Use -1 to utilize all available CPU cores.
        
        n_tier (int, optional):
            Number of neighbor tiers to consider in the Hamiltonian. Default: 0.
            Higher values include longer-range interactions (0: nearest neighbors only).
    
    Examples:
        ``` python
        translator = OpenMXDatasetTranslator(
            openmx_data_dir="./openmx_outputs",
            deeph_data_dir="./deeph_training_data",
            export_S=True,
            export_H=True,
            export_rho=False,
            n_jobs=4
        )
        translator.transfer_all_openmx_to_deeph()  # Start conversion process
        ```
    
    Notes:
        - The OpenMX data directory should follow the structure convention:
            openmx_data_dir/
                structure_1/
                    openmx.scfout
                    other_output_files...
                structure_2/
                    ...
        - Conversion performance scales with n_jobs, but memory usage increases proportionally.
        - Setting n_tier > 0 requires corresponding OpenMX calculations with appropriate interaction ranges.
    
    See Also:
        DeepH documentation: https://docs.deeph-pack.com/deeph-pack/
        OpenMX documentation: https://www.openmx-square.org/
    """
    def __init__(self,
        openmx_data_dir, deeph_data_dir,
        export_S=True, export_H=True, export_rho=False, export_r=False,
        n_jobs=1, n_tier=0
    ):
        self.openmx_data_dir = Path(openmx_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.export_S = export_S
        self.export_H = export_H
        self.export_rho = export_rho
        self.export_r = export_r
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        self.deeph_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_openmx_to_deeph(self):
        worker = partial(
            self.transfer_one_openmx_to_deeph,
            openmx_path=self.openmx_data_dir,
            deeph_path=self.deeph_data_dir,
            export_S=self.export_S,
            export_H=self.export_H,
            export_rho=self.export_rho,
            export_r=self.export_r,
        )
        data_dir_lister = get_data_dir_lister(
            self.openmx_data_dir, self.n_tier, validation_check_openmx
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def transfer_one_openmx_to_deeph(
        dir_name: str, openmx_path: Path, deeph_path: Path,
        export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        openmx_dir_path = openmx_path / dir_name
        if not openmx_dir_path.is_dir():
            return
        deeph_dir_path = deeph_path / dir_name
        deeph_dir_path.mkdir(exist_ok=True, parents=True)
        #
        reader = OpenMXReader(openmx_dir_path, deeph_dir_path)
        reader.dump_data(
            export_S=export_S, export_H=export_H,
            export_rho=export_rho, export_r=export_r
        )


class OpenMXReader:
    def __init__(self, openmx_path, deeph_path):
        self.scfout_path = Path(openmx_path) / OPENMX_SCFOUT_FILENAME
        self.out_path = Path(openmx_path) / OPENMX_OUT_FILENAME
        self.scfout_reader = BinaryFileReader(self.scfout_path)
        self.out_reader = TextFileReader(self.out_path)
        self.deeph_path = Path(deeph_path)
    
    def dump_data(self,
        export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        # Read necessary info.
        self._read_scfout_info()
        self._read_out()
        self._read_scfout_matrix()
        # Dump data
        self._dump_info_json()
        self._dump_poscar()
        if export_S:
            self._dump_S()
        if export_H:
            self._dump_H()
        if export_rho:
            self._dump_rho()
        if export_r:
            self._dump_r()
    
    def _read_version_and_spinful_flag(self):
        _version_and_spinful_flag = self.scfout_reader.read('i')[0]
        _openmx_version = _version_and_spinful_flag // 4
        assert _openmx_version == 3, "You are not using the OpenMX 3.9 version!"
        return _version_and_spinful_flag % 4
    
    def _get_openmx_scfout_matrix_info(self):
        atom_pairs = []
        chunk_shapes = []
        chunk_boundaries = [0,]
        for i_atom in range(self.atoms_quantity):
            atom_i_orb_quantity = self.orbit_quantity_list[i_atom]
            for j, j_atom in enumerate(self.fnna_indices_list[i_atom]):
                atom_j_orb_quantity = self.orbit_quantity_list[j_atom]
                j_cell = self.fnna_cell_indices_list[i_atom][j]
                atom_pairs.append(list(self.R_ijk[j_cell]) + [i_atom, j_atom])
                chunk_shapes.append((atom_i_orb_quantity, atom_j_orb_quantity))
                _size = atom_i_orb_quantity * atom_j_orb_quantity
                chunk_boundaries.append(chunk_boundaries[-1] + _size)
        return {
            "atom_pairs": np.array(atom_pairs),
            "chunk_shapes": np.array(chunk_shapes),
            "chunk_boundaries": np.array(chunk_boundaries),
        }
    
    def _read_openmx_scfout_matrix(self):
        _size = self.matrix_info["chunk_boundaries"][-1]
        return np.array(self.scfout_reader.read(f'{_size}d'))
    
    def _skip_openmx_scfout_matrix(self):
        _size = self.matrix_info["chunk_boundaries"][-1]
        self.scfout_reader.skip(f'{_size}d')
    
    def _stack_spinful_matrix(self, H11, H12, H21, H22):
        """stack the spinful matrix [[H11, H12], [H21, H22]]"""
        spinful_matrix = np.zeros(H11.shape[0]*4, dtype=H11.dtype)
        bounds = self.matrix_info["chunk_boundaries"]
        shapes = self.matrix_info["chunk_shapes"]
        for i_ap, _ in enumerate(self.matrix_info["atom_pairs"]):
            _h11 = H11[bounds[i_ap]:bounds[i_ap+1]].reshape(shapes[i_ap])
            _h12 = H12[bounds[i_ap]:bounds[i_ap+1]].reshape(shapes[i_ap])
            _h21 = H21[bounds[i_ap]:bounds[i_ap+1]].reshape(shapes[i_ap])
            _h22 = H22[bounds[i_ap]:bounds[i_ap+1]].reshape(shapes[i_ap])
            _h = np.block([[_h11, _h12], [_h21, _h22]])
            spinful_matrix[bounds[i_ap]*4:bounds[i_ap+1]*4] = _h.reshape(-1)
        return spinful_matrix
    
    def _correct_r_matrix(self, rx, ry, rz):
        bounds = self.matrix_info["chunk_boundaries"]
        for i_ap, ap in enumerate(self.matrix_info["atom_pairs"]):
            i_atom = ap[3]
            ax, ay, az = self.cart_coords[i_atom]
            _s = self.S[bounds[i_ap]:bounds[i_ap+1]]
            rx[bounds[i_ap]:bounds[i_ap+1]] += _s * ax
            ry[bounds[i_ap]:bounds[i_ap+1]] += _s * ay
            rz[bounds[i_ap]:bounds[i_ap+1]] += _s * az
        return rx, ry, rz
    
    def _read_scfout_info(self):
        self.obs_down_up_sort_idx = None
        #-----------------------------------------------------------------------
        self.atoms_quantity = self.scfout_reader.read('i')[0]
        #-----------------------------------------------------------------------
        self.spin_info = self._read_version_and_spinful_flag()
        self.spinful = (0!=self.spin_info)
        #-----------------------------------------------------------------------
        self.scfout_reader.skip('3i') # CLR atom num
        #-----------------------------------------------------------------------
        self.R_quantity = self.scfout_reader.read('i')[0] + 1
        #-----------------------------------------------------------------------
        self.r_order_max = self.scfout_reader.read('i')[0]
        #-----------------------------------------------------------------------
        self.scfout_reader.skip(f'{4*self.R_quantity}d') # R_xyz
        #-----------------------------------------------------------------------
        self.R_ijk = np.array(
            self.scfout_reader.read(f'{4*self.R_quantity}i')
        ).reshape((self.R_quantity, 4))[:, 1:]
        #-----------------------------------------------------------------------
        self.orbit_quantity_list = np.array(
            self.scfout_reader.read(f'{self.atoms_quantity}i')
        )
        self.orbit_cumsum = np.insert(np.cumsum(self.orbit_quantity_list), 0, 0)
        self.orbits_quantity = int(self.orbit_cumsum[-1])
        #-----------------------------------------------------------------------
        self.fnna_quantity_list = np.array(
            self.scfout_reader.read(f'{self.atoms_quantity}i')
        )
        #-----------------------------------------------------------------------
        self.fnna_indices_list = [
            np.array(self.scfout_reader.read(f'{fnna_quantity+1}i')) - 1
            for fnna_quantity in self.fnna_quantity_list
        ]
        #-----------------------------------------------------------------------
        self.fnna_cell_indices_list = [
            np.array(self.scfout_reader.read(f'{fnna_quantity+1}i'))
            for fnna_quantity in self.fnna_quantity_list
        ]
        #-----------------------------------------------------------------------
        self.lattice = np.array(
            self.scfout_reader.read('12d')
        ).reshape(3,4)[:,1:] * BOHR_TO_ANGSTROM
        #-----------------------------------------------------------------------
        self.scfout_reader.skip('12d') # Reciprocal lattice
        #-----------------------------------------------------------------------
        self.cart_coords = np.array(
            self.scfout_reader.read(f'{4*self.atoms_quantity}d')
        ).reshape(self.atoms_quantity,4)[:,1:] * BOHR_TO_ANGSTROM
        #-----------------------------------------------------------------------
        self.matrix_info = self._get_openmx_scfout_matrix_info()
        #-----------------------------------------------------------------------
    
    def _read_scfout_matrix(self):
        if 0 == self.spin_info: # Non spinful
            self.H = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
        elif 1 == self.spin_info: # Colinear spinful
            _H_up = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H_dn = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H_zero = np.zeros_like(_H_up)
            self.H = self._stack_spinful_matrix(_H_up, _H_zero, _H_zero, _H_dn)
        elif 3 == self.spin_info: # Non-colinear spinful
            _H0_real = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H1_real = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H2_real = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H3_real = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H0_imag = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H1_imag = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            _H2_imag = self._read_openmx_scfout_matrix() * HARTREE_TO_EV
            H_up_down = _H2_real + 1j*(_H3_real + _H2_imag)
            H_down_up = self.get_obs_down_up_from_(H_up_down)
            self.H = self._stack_spinful_matrix(
                _H0_real + 1j*_H0_imag, H_up_down,
                H_down_up, _H1_real + 1j*_H1_imag
            )
        else:
            raise ValueError(f'Invalid spin info: {self.spin_info}')
        self.H = self.basis_transform_to_wiki(self.H, self.spinful)
        #-----------------------------------------------------------------------
        self.S = self._read_openmx_scfout_matrix()
        self.S = self.basis_transform_to_wiki(self.S, False)
        #-----------------------------------------------------------------------
        _rx = self._read_openmx_scfout_matrix() * BOHR_TO_ANGSTROM
        _rx = self.basis_transform_to_wiki(_rx, False)
        for _ in range(self.r_order_max-1): # Skip the r^2 r^3,... terms
            self._skip_openmx_scfout_matrix()
        _ry = self._read_openmx_scfout_matrix() * BOHR_TO_ANGSTROM
        _ry = self.basis_transform_to_wiki(_ry, False)
        for _ in range(self.r_order_max-1): # Skip the r^2 r^3,... terms
            self._skip_openmx_scfout_matrix()
        _rz = self._read_openmx_scfout_matrix() * BOHR_TO_ANGSTROM
        _rz = self.basis_transform_to_wiki(_rz, False)
        for _ in range(self.r_order_max-1): # Skip the r^2 r^3,... terms
            self._skip_openmx_scfout_matrix()
        _rx, _ry, _rz = self._correct_r_matrix(_rx, _ry, _rz)
        self.r = np.stack([_rx, _ry, _rz], axis=0)
        #-----------------------------------------------------------------------
        self._skip_openmx_scfout_matrix() # Skip the px matrix
        self._skip_openmx_scfout_matrix() # Skip the py matrix
        self._skip_openmx_scfout_matrix() # Skip the pz matrix
        #-----------------------------------------------------------------------
        if 0 == self.spin_info: # Non spinful
            self.rho = self._read_openmx_scfout_matrix()
        elif 1 == self.spin_info: # Colinear spinful
            _rho_up = self._read_openmx_scfout_matrix()
            _rho_dn = self._read_openmx_scfout_matrix()
            _rho_zero = np.zeros_like(_rho_up)
            self.rho = self._stack_spinful_matrix(
                _rho_up, _rho_zero, _rho_zero, _rho_dn
            )
        elif 3 == self.spin_info: # Non-colinear spinful
            _rho0_real = self._read_openmx_scfout_matrix()
            _rho1_real = self._read_openmx_scfout_matrix()
            _rho2_real = self._read_openmx_scfout_matrix()
            _rho3_real = self._read_openmx_scfout_matrix()
            _rho0_imag = self._read_openmx_scfout_matrix()
            _rho1_imag = self._read_openmx_scfout_matrix()
            rho_up_down = _rho2_real + 1j*_rho3_real
            rho_down_up = self.get_obs_down_up_from_(rho_up_down)
            self.rho = self._stack_spinful_matrix(
                _rho0_real + 1j*_rho0_imag, rho_up_down,
                rho_down_up, _rho1_real + 1j*_rho1_imag
            )
        else:
            raise ValueError(f'Invalid spin info: {self.spin_info}')
        self.rho = self.basis_transform_to_wiki(self.rho, self.spinful)
        #-----------------------------------------------------------------------

    def _read_out(self):
        #-----------------------------------------------------------------------
        self.species_quantity = int(self.out_reader.find("Species.Number", 1))
        #-----------------------------------------------------------------------
        _idx = self.out_reader.find_idx("<Definition.of.Atomic.Species") + 1
        self.elem_orb_map = {}
        for i_spec in range(self.species_quantity):
            elem_orb_info = self.out_reader.lines[_idx+i_spec].strip().split()
            elem, orb = elem_orb_info[0], elem_orb_info[1].split('-')[1]
            self.elem_orb_map[elem] = convert_orbital_string_to_list(orb)
        self.basis_trans_index = {}
        for elem, orbs in self.elem_orb_map.items():
            orbital_num_list = np.array([2 * orb_l + 1 for orb_l in orbs])
            orbital_cumsum = np.concatenate((np.array([0]), np.cumsum(orbital_num_list, axis=0)), axis=0)[:-1]
            index = []
            for orb_l, orb_cum in zip(orbs, orbital_cumsum):
                index.append(BASIS_TRANS_OPENMX2WIKI(orb_l) + orb_cum)
            self.basis_trans_index[elem] = np.concatenate(index, axis=0)
        #-----------------------------------------------------------------------
        self.fermi_energy = float(
            self.out_reader.find_last("Chemical potential (Hartree)", 3)
        ) * HARTREE_TO_EV
        #-----------------------------------------------------------------------
        _idx = self.out_reader.find_idx("Fractional coordinates of the") + 4
        atom_elem = []
        for i_atom in range(self.atoms_quantity):
            atom_elem.append(
                self.out_reader.lines[_idx+i_atom].strip().split()[1]
            )
        _seen, _prev = set(), atom_elem[0]
        for _curr in atom_elem[1:]:
            if _curr == _prev:
                continue
            if _curr in _seen:
                raise ValueError(
                    f'The atomic elements is not continued: {atom_elem}'
                )
            _seen.add(_prev)
            _prev = _curr
        self.atom_elem = atom_elem
        self.atom_elem_dict = dict(collections.Counter(atom_elem))
        #-----------------------------------------------------------------------

    def _dump_info_json(self):
        file_path = self.deeph_path / DEEPX_INFO_FILENAME
        info_json = {
            "atoms_quantity": self.atoms_quantity,
            "orbits_quantity": self.orbits_quantity,
            "orthogonal_basis": False,
            "spinful": self.spinful,
            "fermi_energy_eV": self.fermi_energy,
            "elements_orbital_map": self.elem_orb_map,
        }
        with open(file_path, 'w') as fwj:
            json.dump(info_json, fwj)
    
    def _dump_poscar(self):
        file_path = self.deeph_path / DEEPX_POSCAR_FILENAME
        poscar = [
            "POSCAR generated by DeepH-dock \n",
            "1.0\n",
            '  ' + ' '.join(map(str, self.lattice[0])) + '\n',
            '  ' + ' '.join(map(str, self.lattice[1])) + '\n',
            '  ' + ' '.join(map(str, self.lattice[2])) + '\n',
            ' '.join(self.atom_elem_dict.keys()) + '\n',
            ' '.join(map(str, self.atom_elem_dict.values())) + '\n',
            "Cartesian\n",
        ] + [
            '  ' + ' '.join(map(str, self.cart_coords[i])) + '\n'
            for i in range(self.atoms_quantity)
        ]
        with open(file_path, 'w') as fwp:
            fwp.writelines(poscar)
    
    def _dump_H(self):
        file_path = self.deeph_path / DEEPX_HAMILTONIAN_FILENAME
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"] * (self.spinful+1),
            "chunk_boundaries": self.matrix_info["chunk_boundaries"] * ((self.spinful+1)**2),
            "entries": self.H,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)
    
    def _dump_S(self):
        file_path = self.deeph_path / DEEPX_OVERLAP_FILENAME
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"],
            "chunk_boundaries": self.matrix_info["chunk_boundaries"],
            "entries": self.S,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)
    
    def _dump_rho(self):
        file_path = self.deeph_path / DEEPX_DENSITY_MATRIX_FILENAME
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"] * (self.spinful+1),
            "chunk_boundaries": self.matrix_info["chunk_boundaries"] * ((self.spinful+1)**2),
            "entries": self.rho,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)

    def _dump_r(self):
        file_path = self.deeph_path / DEEPX_POSITION_MATRIX_FILENAME
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"],
            "chunk_boundaries": self.matrix_info["chunk_boundaries"],
            "entries": self.r,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)

    def basis_transform_to_wiki(self, entries, spinful):
        for i_pair, atom_pair in enumerate(self.matrix_info["atom_pairs"]):
            chunk_shape = self.matrix_info["chunk_shapes"][i_pair] * (spinful+1)
            chunk_boundary = self.matrix_info["chunk_boundaries"][i_pair] * (spinful+1)**2
            block = entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]].reshape(chunk_shape)
            transform_index1 = self.basis_trans_index[self.atom_elem[atom_pair[3]]]
            transform_index2 = self.basis_trans_index[self.atom_elem[atom_pair[4]]]
            entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]] = self._transform(block, transform_index1, transform_index2, spinful).reshape(-1)
        return entries

    def _transform(self, matrix, transform_index1, transform_index2, isspinful):
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

    def get_obs_down_up_from_(self, obs_up_down):
        if self.obs_down_up_sort_idx is None:
            chunk_shapes = self.matrix_info["chunk_shapes"]
            chunk_boundaries = self.matrix_info["chunk_boundaries"]
            sizes = chunk_shapes[:, 0] * chunk_shapes[:, 1]
            ## Find the local sort index of block inverse transpose (from (cols,rows) to (rows,cols))
            starts = np.repeat(chunk_boundaries[:-1], sizes)
            global_indices = np.arange(chunk_boundaries[-1])
            local_indices = global_indices - starts
            rows = np.repeat(chunk_shapes[:, 0], sizes)
            cols = np.repeat(chunk_shapes[:, 1], sizes)
            block_transpose_local_idx = (local_indices % cols) * rows + (local_indices // cols)
            ## Find the sort index of [-Rijk, j, i]
            conj_pair_sort_index = self._get_conj_pair_sort_index()
            conjugate_pair_start_idx = np.repeat(chunk_boundaries[:-1][conj_pair_sort_index], sizes)
            ## Combine the two indexes
            self.obs_down_up_sort_idx = conjugate_pair_start_idx + block_transpose_local_idx
        ## Sort
        return np.conj(obs_up_down[self.obs_down_up_sort_idx])

    def _get_conj_pair_sort_index(self):
        atom_pairs = self.matrix_info["atom_pairs"]
        conj_atom_pairs = np.concatenate([-atom_pairs[:, :3], atom_pairs[:, [4,3]]], axis=1)
        sort_origin_idx = np.lexsort(atom_pairs.mT)
        sort_conj_idx = np.lexsort(conj_atom_pairs.mT)
        assert np.allclose(atom_pairs[sort_origin_idx, :], conj_atom_pairs[sort_conj_idx, :]), "Some Rijk do not has its inverse!"
        rev_sort_conj_idx = np.argsort(sort_conj_idx)
        sort_idx = sort_origin_idx[rev_sort_conj_idx]
        return sort_idx
