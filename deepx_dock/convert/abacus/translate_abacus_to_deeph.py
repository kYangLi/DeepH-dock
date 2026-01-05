# Script for interface from ABACUS (http://abacus.ustc.edu.cn/) to DeepH-pack
# Originally coded by ZC Tang @ Tsinghua Univ. e-mail: az_txycha@126.com
# Modified by He Li @ Tsinghua Univ. & XY Zhou @ Peking Univ.
# Integrated in deeph-dock by Boheng Zhao @ Tsinghua Univ.
# To use this script, please add 'out_mat_hs2    1' in ABACUS INPUT File
# Current version is capable of coping with f-orbitals
# 20220717: Read structure from running_scf.log
# 20220919: The suffix of the output sub-directories (OUT.suffix) can be set by ["basic"]["abacus_suffix"] keyword in preprocess.ini
# 20220920: Supporting cartesian coordinates in the log file
# 20231228: Supporting ABACUS v3.4
# 20250801: Supporting ABACUS v3.10 LTS, nspin=2

import numpy as np
from scipy.sparse import csr_matrix

from pathlib import Path
import json
import h5py
import re
import shutil
import traceback
from typing import List

from tqdm import tqdm

from functools import partial
from joblib import Parallel, delayed

from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSITION_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX
from deepx_dock.misc import get_data_dir_lister

ABACUS_LOG_SCF_FILENAME = "running_scf.log"
ABACUS_LOG_GET_S_FILENAME = "running_get_S.log"
ABACUS_OVERLAP_SCF_FILENAME = "data-SR-sparse_SPIN0.csr"
ABACUS_OVERLAP_GET_S_FILENAME = "SR.csr"
ABACUS_R_FILENAME = "data-rR-sparse.csr"
ABACUS_HAMILTONIAN_FILENAME = "data-HR-sparse_SPIN0.csr"
ABACUS_RHO_FILENAME = "data-DMR-sparse_SPIN0.csr"

'''
In abacus source/module_base/constants.h
BOHR_TO_A     = 0.529177
ANGSTROM_AU   = 1.889727
Hartree_to_eV = 27.211396
Ry_to_eV      = 13.605698
'''
BOHR_TO_ANGSTROM = 0.529177
HARTREE_TO_EV = 27.211396

'''
wiki order: m=...,-2,-1,0,+1,+2,...
l=0 1
l=1 y z x
l=2 xy yz z2 xz x2-y2
l=3 -3 -2 -1 0 +1 +2 +3

abacus order: m=0,+1,-1,+2,-2,+3,-3...
abacus basis: wiki basis * (-1)**m
l=0 1
l=1 z -x -y
l=2 z2 -xz -yz x2-y2 xy
l=3 0 -+1 --1 +2 -2 -+3 --3

deeph order: openmx order
deeph basis: wiki basis
l=0 1
l=1 x y z
l=2 z2 x2-y2 xy xz yz
l=3 0 +1 -1 +2 -2 +3 -3
'''
def BASIS_TRANS_ABACUS2WIKI(ll):
    if ll == 0:
        return np.array([0])
    elif ll == 1:
        return np.array([2, 0, 1])
    elif ll == 2:
        return np.array([4, 2, 0, 1, 3])
    elif ll == 3:
        return np.array([6, 4, 2, 0, 1, 3, 5])
    else:
        return np.concatenate([np.arange(2*ll, -1, -2), np.arange(1, 2*ll, 2)])

def BASIS_FACTOR_ABACUS2WIKI(ll):
    if ll == 0:
        return np.array([1.0])
    elif ll == 1:
        return np.array([-1.0, 1.0, -1.0])
    elif ll == 2:
        return np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    elif ll == 3:
        return np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    else:
        return (-1.0)**np.arange(-ll, ll+1, 1, dtype=np.float64)


def validation_check_abacus(root_dir: Path, prev_dirname: Path, abacus_suffix: str = ""):
    out_dirs: List[str] = []
    for out_dir in root_dir.iterdir():
        if out_dir.is_dir():
            out_dir = str(out_dir.name)
            if abacus_suffix:
                if f"OUT.{abacus_suffix}" == out_dir:
                    out_dirs.append(out_dir)
            else:
                if out_dir.startswith("OUT."):
                    out_dirs.append(out_dir)
    if len(out_dirs) > 1:
        tqdm.write(f"WARN in {prev_dirname}: Find multiple OUT.* dirs, so that skip it. Specify a target OUT.<suffix> dir by `--abacus_suffix <suffix>`.")
    elif len(out_dirs) == 1:
        out_dir = out_dirs[0]
        all_files = list(str(v.name) for v in (root_dir / out_dir).iterdir())
        if ABACUS_LOG_SCF_FILENAME in all_files \
            or ABACUS_LOG_GET_S_FILENAME in all_files:
            yield prev_dirname / out_dir


class TextFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(file_path, 'r')
    
    def __del__(self):
        self.f.close()
    
    def find(self, target):
        line = self.f.readline()
        while line:
            if target in line:
                return line
            line = self.f.readline()
        self.f.seek(0) # move to head
        return None
    
    def find_from_head(self, target):
        self.f.seek(0) # move to head
        return self.find(target)
    
    def readline(self):
        return self.f.readline()


class AbacusDatasetTranslator:
    def __init__(self,
        abacus_data_dir, deeph_data_dir, abacus_suffix="",
        export_S=True, export_H=True, export_rho=False, export_r=False,
        n_jobs=1, n_tier=0
    ):
        self.abacus_data_dir = Path(abacus_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.abacus_suffix = abacus_suffix
        self.export_S = export_S
        self.export_H = export_H
        self.export_rho = export_rho
        self.export_r = export_r
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        self.deeph_data_dir.mkdir(exist_ok=True, parents=True)

    def transfer_all_abacus_to_deeph(self):
        worker = partial(
            self.transfer_one_abacus_to_deeph,
            abacus_path=self.abacus_data_dir,
            deeph_path=self.deeph_data_dir,
            export_S=self.export_S,
            export_H=self.export_H,
            export_rho=self.export_rho,
            export_r=self.export_r,
        )
        data_dir_lister = get_data_dir_lister(
            self.abacus_data_dir, self.n_tier, 
            partial(validation_check_abacus, abacus_suffix=self.abacus_suffix)
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data", leave=True)
        )
#         info = '''
# +-----------------------------------------------------------------+
# | Thanks for using abacus-deeph interface. The original interface |
# | is coded by ZC Tang for deeph-hybrid project.                   |
# | Deeph-hybrid article https://doi.org/10.1038/s41467-024-53028-4 |
# | Original interface in https://github.com/mzjb/DeepH-pack        |
# | Additional file in https://github.com/aaaashanghai/DeepH-hybrid |
# | NOTE: The current interface supports ABACUS v3.10 LTS.          |
# | For other versions, please modify ABACUS_*_FILENAME variable in |
# | DeepH-dock/deepx_dock/io/ABACUS/translate_abacus_to_deeph.py    |
# +-----------------------------------------------------------------+
#         '''
#         print(info)

    @staticmethod
    def transfer_one_abacus_to_deeph(
        dir_name: str, abacus_path, deeph_path,
        export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        deeph_path = Path(deeph_path)
        deeph_dir = (deeph_path / dir_name).parent.resolve()
        deeph_dir.mkdir(parents=True, exist_ok=True)
        reader = AbacusReader(
            dir_name, abacus_path, deeph_path,
            export_S=export_S, export_H=export_H,
            export_rho=export_rho, export_r=export_r
        )
        try:
            reader.dump_data()
        except AssertionError as e:
            tqdm.write(f"ERROR in {e}")
            shutil.rmtree(deeph_dir)
        except Exception:
            tqdm.write(f"\nERROR in {dir_name}:")
            tqdm.write(str(traceback.format_exc()).rstrip("None"))
            shutil.rmtree(deeph_dir)


class AbacusReader:
    def __init__(
        self, dir_name: str, abacus_path, deeph_path,
        *, export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        abacus_path = Path(abacus_path)
        deeph_path = Path(deeph_path)
        abacus_dir = abacus_path / dir_name
        deeph_dir = (deeph_path / dir_name).parent.resolve()
        deeph_dir.mkdir(parents=True, exist_ok=True)
        #
        self.abacus_path = abacus_path
        self.deeph_path = deeph_path
        self.dir_name = dir_name
        self.abacus_dir = abacus_dir
        self.deeph_dir = deeph_dir
        self.export_S = export_S
        self.export_H = export_H
        self.export_rho = export_rho
        self.export_r = export_r
        if (abacus_dir / ABACUS_LOG_SCF_FILENAME).exists():
            self.calculation = "scf"
            self.log_name = ABACUS_LOG_SCF_FILENAME
        elif (abacus_dir / ABACUS_LOG_GET_S_FILENAME).exists():
            self.calculation = "get_S"
            self.log_name = ABACUS_LOG_GET_S_FILENAME
            self.export_H = False
            self.export_rho = False
        self.log_path = abacus_dir / self.log_name
        self.log_reader = TextFileReader(self.log_path)
    
    def __del__(self):
        del self.log_reader
    
    def dump_data(self):
        self._read_log()
        self._dump_info_json()
        self._dump_poscar()
        '''
        ABACUS output convention:
        nspin         | 1 2 4 |
        --------------+-------+
        S   num_files | 1 1 1 |
        S   isspinful | n n y |
        S   iscomplex | n n y |
        --------------+-------+
        r   num_files | 1 1 1 |
        r   isspinful | n n y |
        r   iscomplex | n n n |
        --------------+-------+
        H   num_files | 1 2 1 |
        H   isspinful | n n y |
        H   iscomplex | n n y |
        --------------+-------+
        rho num_files | 1 2 1 |
        rho isspinful | n n y |
        rho iscomplex | n n n |
        '''
        spin4 = (4 == self.nspin)
        spin2 = (2 == self.nspin)
        if "scf" == self.calculation:
            ABACUS_OVERLAP_FILENAME = ABACUS_OVERLAP_SCF_FILENAME
        elif "get_S" == self.calculation:
            ABACUS_OVERLAP_FILENAME = ABACUS_OVERLAP_GET_S_FILENAME
        overlap_path = Path(self.dir_name) / ABACUS_OVERLAP_FILENAME
        if not (self.abacus_path / overlap_path).exists():
            tqdm.write(f"WARN in {self.dir_name}: Can't find overlap output file. This file is necessary for determining the sparse pattern. Please check whether `basis_type` is `lcao` and `out_mat_hs2` is `true` in abacus INPUT file.")
            return
        self.matrix_info = self._get_abacus_matrix_info(
            overlap_path, spin4, spin4
        )
        self.conj_pair_sort_index = self._get_conj_pair_sort_index()
        #-----------------------------------------------------------------------
        obs_info = {
        "overlap": [
        self.export_S, ABACUS_OVERLAP_FILENAME, DEEPX_OVERLAP_FILENAME, 1, False, 1.0,
        {"spinful_in": spin4, "complex_in": spin4, "spinful_out": False, "complex_out": False}
        ],
        "position_matrix": [
        self.export_r, ABACUS_R_FILENAME, DEEPX_POSITION_MATRIX_FILENAME, 1, False, BOHR_TO_ANGSTROM,
        {"spinful_in": spin4, "complex_in": False, "spinful_out": False, "complex_out": False}
        ],
        "Hamiltonian": [
        self.export_H, ABACUS_HAMILTONIAN_FILENAME, DEEPX_HAMILTONIAN_FILENAME, 1+spin2, self.spinful, HARTREE_TO_EV/2.0, 
        {"spinful_in": spin4, "complex_in": spin4, "spinful_out": spin4, "complex_out": spin4}
        ],
        "density_matrix": [
        self.export_rho, ABACUS_RHO_FILENAME, DEEPX_DENSITY_MATRIX_FILENAME, 1+spin2, self.spinful, 1.0,
        {"spinful_in": spin4, "complex_in": False, "spinful_out": spin4, "complex_out": spin4}
        ],
        }
        #-----------------------------------------------------------------------
        for obs, info in obs_info.items():
            export, abacus_filename0, deepx_filename, nfiles, isspinful, unit, kargs = info
            if not export:
                continue
            obs_path = Path(self.dir_name) / str(abacus_filename0)
            if not (self.abacus_path / obs_path).exists():
                tqdm.write(f"WARN in {self.dir_name}: Can't find {obs} output file. Skip")
                continue
            kargs["for_r"] = "position_matrix" == obs
            entries = self._read_csr_matrix(obs_path, **kargs)
            if nfiles > 1: # colinear case, two spin files
                assert nfiles == 2, f"{self.dir_name}: More than 2 spin files is not supported."
                abacus_filename1 = str(abacus_filename0).replace("0", "1")
                obs_path = Path(self.dir_name) / abacus_filename1
                ## stack up-up and down-down to spinful matrix
                entries_up = entries
                entries_dn = self._read_csr_matrix(obs_path, **kargs)
                entries_zero = np.zeros_like(entries_up)
                entries = self._stack_spinful_matrix(
                    entries_up, entries_zero, entries_zero, entries_dn
                )
            entries *= unit
            export_path = self.deeph_dir / deepx_filename
            self._dump_h5(export_path, entries, isspinful)

    def _read_log(self):
        abacus_version = self.log_reader.find("ABACUS").split()[-1].lstrip("v")
        # tqdm.write(f"ABACUS version: {abacus_version}")
        #-----------------------------------------------------------------------
        assert self.log_reader.find("READING UNITCELL") is not None, f'{self.dir_name}: Cannot find "READING UNITCELL"'
        num_atom_type = int(self.log_reader.readline().split()[-1])
        #-----------------------------------------------------------------------
        tmp = self.log_reader.find("lattice constant (Angstrom)")
        assert tmp is not None, f"{self.dir_name}: Cannot find lattice constant"
        lattice_constant = float(tmp.split()[-1])
        #---------------------------- ATOM TYPE --------------------------------
        atom_elem_dict = {}
        site_norbits_dict = {}
        elem_orb_map = {}
        basis_trans_index = {}
        basis_trans_factor = {}
        for index_type in range(num_atom_type):
            tmp = self.log_reader.find("READING ATOM TYPE")
            assert tmp is not None, f"{self.dir_name}: Cannot find atom type {index_type}"
            assert tmp.split()[-1] == str(index_type + 1)
            #-------------------------------------------------------------------
            line = self.log_reader.readline()
            assert "atom label =" in line
            atom_label = line.split()[-1]
            assert atom_label in PERIODIC_TABLE_SYMBOL_TO_INDEX, f"{self.dir_name}: Invalid atom label {atom_label}"
            #-------------------------------------------------------------------
            current_site_norbits = 0
            current_orbital_types = []
            current_basis_trans_index = []
            current_basis_trans_factor = []
            while True:
                line = self.log_reader.readline()
                if "number of zeta" in line:
                    tmp = line.split()
                    L = int(tmp[0][2:-1])
                    num_L = int(tmp[-1])
                    offset = np.repeat(np.arange(num_L) * (2*L+1), 2*L+1)
                    index = np.tile(BASIS_TRANS_ABACUS2WIKI(L), num_L)
                    current_basis_trans_index.append(
                        current_site_norbits + offset + index
                    )
                    current_basis_trans_factor.append(
                        np.tile(BASIS_FACTOR_ABACUS2WIKI(L), num_L)
                    )
                    current_site_norbits += (2 * L + 1) * num_L
                    current_orbital_types.extend([L] * num_L)
                else:
                    break
            assert "number of atom for this type" in line
            atom_elem_dict[atom_label] = int(line.split()[-1])
            site_norbits_dict[atom_label] = current_site_norbits
            elem_orb_map[atom_label] = current_orbital_types
            basis_trans_index[atom_label] = \
                np.concatenate(current_basis_trans_index, axis=0)
            basis_trans_factor[atom_label] = \
                np.concatenate(current_basis_trans_factor, axis=0)
        #-----------------------------------------------------------------------
        line = self.log_reader.find("TOTAL ATOM NUMBER")
        assert line is not None, f'{self.dir_name}: Cannot find "TOTAL ATOM NUMBER"'
        atoms_quantity = int(line.split()[-1])
        #------------------------- ATOM COORDINATES ----------------------------
        line = self.log_reader.find(" COORDINATES")
        assert line is not None, f'{self.dir_name}: Cannot find " COORDINATES"'
        if "DIRECT" in line:
            coords_type = "direct" 
        elif "CARTESIAN" in line:
            coords_type = "cartesian" 
        else:
            raise AssertionError(f'{self.dir_name}: Cannot find "DIRECT COORDINATES" or "CARTESIAN COORDINATES"')
        #-----------------------------------------------------------------------
        assert "atom" in self.log_reader.readline()
        coords = np.zeros((atoms_quantity, 3))
        orbit_quantity_list = np.zeros(atoms_quantity, dtype=int)
        atom_elem = []
        for i_site in range(atoms_quantity):
            line = self.log_reader.readline()
            tmp = line.split()
            assert "tau" in tmp[0]
            atom_label = ''.join(re.findall(r'[A-Za-z]', tmp[0][5:]))
            assert atom_label in PERIODIC_TABLE_SYMBOL_TO_INDEX, f'{self.dir_name}: Invalid atom label {atom_label}'
            atom_elem.append(atom_label)
            orbit_quantity_list[i_site] = site_norbits_dict[atom_elem[i_site]]
            coords[i_site, :] = np.array(tmp[1:4])
        #-----------------------------------------------------------------------
        assert self.log_reader.find("Lattice vectors") is not None, f'{self.dir_name}: Cannot find lattice vectors'
        lattice = np.zeros((3, 3))
        for index_lat in range(3):
            lattice[index_lat, :] = np.array(self.log_reader.readline().split())
        lattice *= lattice_constant
        #-----------------------------------------------------------------------
        if "cartesian" == coords_type:
            cart_coords = coords * lattice_constant
        elif "direct" == coords_type:
            cart_coords = coords @ lattice
        #-----------------------------------------------------------------------
        line = self.log_reader.find("nspin")
        assert line is not None, f'{self.dir_name}: Cannot find nspin'
        nspin = int(line.split()[-1])
        #-----------------------------------------------------------------------
        fermi_energy = None
        if self.calculation == 'scf':
            line = self.log_reader.find_from_head("EFERMI")
            while (line is not None) and ("eV" not in line):
                line = self.log_reader.find("EFERMI")
            if line is not None:
                fermi_energy = float(line.split()[2])
            else:
                tqdm.write(f'WARN in {self.dir_name}: Cannot find EFERMI in eV')
        #-----------------------------------------------------------------------
        # - info
        self.nspin = nspin
        self.spinful = (nspin != 1)
        self.fermi_energy = fermi_energy # fermi_level
        # - element
        self.atom_elem_dict = atom_elem_dict # element_dict
        self.elem_orb_map = elem_orb_map # orbital_types_dict
        self.basis_trans_index = basis_trans_index
        self.basis_trans_factor = basis_trans_factor
        # site
        self.atoms_quantity = atoms_quantity # nsites
        self.orbit_quantity_list = orbit_quantity_list # site_norbits
        self.orbit_cumsum = np.insert(np.cumsum(orbit_quantity_list), 0, 0)
        self.orbits_quantity = int(self.orbit_cumsum[-1])
        self.lattice = lattice
        self.atom_elem = atom_elem # element
        self.cart_coords = cart_coords

    def _dump_info_json(self):
        file_path = self.deeph_dir / DEEPX_INFO_FILENAME
        self.info_json = {
            "atoms_quantity": self.atoms_quantity,
            "orbits_quantity": self.orbits_quantity,
            "orthogonal_basis": False,
            "spinful": self.spinful,
            "fermi_energy_eV": self.fermi_energy,
            "elements_orbital_map": self.elem_orb_map,
        }
        with open(file_path, 'w') as fwj:
            json.dump(self.info_json, fwj)
    
    def _dump_poscar(self):
        file_path = self.deeph_dir / DEEPX_POSCAR_FILENAME
        self.poscar = [
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
            fwp.writelines(self.poscar)

    def _read_csr_matrix(self, file_path: str, for_r=False, 
        spinful_in=False, complex_in=False, spinful_out=False, complex_out=False
    ):
        with open(self.abacus_path / file_path, 'r') as f:
            if not for_r:
                entries = self._read_csr_matrix_from(
                f, file_path, spinful_in, complex_in, spinful_out, complex_out
                )
            else:
                entries = self._read_csr_matrix_rR_from(
                f, file_path, spinful_in, complex_in, spinful_out, complex_out
                )
            if f.read(1):
                tqdm.write(f"WARN in {file_path}: File not read completely. Maybe you are dealing with density_matrix with nspin=4, only real part is read in!!!")
                # entries += 1j * self._read_csr_matrix_from(
                # f, file_path, spinful_in, complex_in, spinful_out, complex_out
                # )
        return entries

    def _read_csr_matrix_from(
            self, f, file_path, spinful_in, complex_in, spinful_out, complex_out
        ):
        matrix_info = self.matrix_info
        dtype = np.complex128 if complex_out else np.float64
        num_entries = matrix_info["chunk_boundaries"][-1]*(1+spinful_out)**2
        entries = np.empty(num_entries, dtype=dtype)
        orbits_quantity, num_R = \
            self._read_csr_headlines(f, file_path, spinful_in)
        for _ in range(num_R):
            line = f.readline()
            R1, R2, R3, num_nz = [int(i) for i in line.split()]
            i_R = np.where(
                np.all([R1, R2, R3]==matrix_info["R_ijk"], axis=1)
            )[0] # give index of R_now in R_ijk
            if len(i_R) == 0: # R_now not in R_ijk, continue
                if num_nz > 0:
                    f.readline(); f.readline(); f.readline()
                continue
            i_R = i_R[0]
            mat_R = self._load_one_csr_matrix(
                f, num_nz, orbits_quantity, complex_in, complex_out
            )
            entries = self._convert_one_csr_matrix(
                i_R, mat_R, entries, spinful_in, spinful_out
            )
        return entries

    def _read_csr_matrix_rR_from(self, f, file_path, 
        spinful_in=False, complex_in=False, spinful_out=False, complex_out=False
    ):
        matrix_info = self.matrix_info
        dtype = np.complex128 if complex_out else np.float64
        num_entries = matrix_info["chunk_boundaries"][-1]*(1+spinful_out)**2
        entries = [
            np.empty(num_entries, dtype=dtype), 
            np.empty(num_entries, dtype=dtype), 
            np.empty(num_entries, dtype=dtype),
        ]
        orbits_quantity, num_R = \
            self._read_csr_headlines(f, file_path, spinful_in)
        for _ in range(num_R):
            line = f.readline()
            R1, R2, R3 = [int(i) for i in line.split()]
            i_R = np.where(
                np.all([R1, R2, R3]==matrix_info["R_ijk"], axis=1)
            )[0] # give index of R_now in R_ijk
            if len(i_R) == 0: # R_now not in R_ijk, continue
                for _ in range(3): # r_matrix has 3 components
                    num_nz = int(f.readline().split()[0])
                    if num_nz > 0:
                        f.readline(); f.readline(); f.readline()
                continue
            i_R = i_R[0]
            for i in range(3): # r_matrix has 3 components
                num_nz = int(f.readline().split()[0])
                mat_R = self._load_one_csr_matrix(
                    f, num_nz, orbits_quantity, complex_in, complex_out
                )
                entries[i] = self._convert_one_csr_matrix(
                    i_R, mat_R, entries[i], spinful_in, spinful_out
                )
        entries = np.stack(entries, axis=0)
        return entries

    def _get_abacus_matrix_info(self, file_path: str, spinful_in, complex_in):
        atom_pairs = []
        chunk_shapes = []
        chunk_boundaries = [0,]
        R_ijk = []
        R_boundaries = [0,]
        with open(self.abacus_path / file_path, 'r') as f:
            orbits_quantity, num_R = \
                self._read_csr_headlines(f, file_path, spinful_in)
            for _ in range(num_R):
                line = f.readline()
                R1, R2, R3, num_nz = [int(i) for i in line.split()]
                if num_nz == 0:
                    continue
                mat_R = self._load_one_csr_matrix(
                    f, num_nz, orbits_quantity, complex_in, complex_in
                )
                num_nz_blocks = 0
                orbit_cumsum = self.orbit_cumsum*(1+spinful_in)
                orbit_quantity_list = self.orbit_quantity_list
                atom_col_starts = orbit_cumsum[:-1]
                atom_col_ends = orbit_cumsum[1:]
                for i_atom in range(self.atoms_quantity):
                    start_row = orbit_cumsum[i_atom]
                    end_row = orbit_cumsum[i_atom + 1]
                    if mat_R.indptr[start_row] == mat_R.indptr[end_row]:
                        continue
                    start_ptr = mat_R.indptr[start_row]
                    end_ptr = mat_R.indptr[end_row]
                    nonzero_cols_in_ia_block = mat_R.indices[start_ptr:end_ptr]
                    cols_expanded = nonzero_cols_in_ia_block[:, np.newaxis]
                    #
                    overlap_matrix = (cols_expanded >= atom_col_starts) & (
                        cols_expanded < atom_col_ends
                    )
                    connected_ja_indices = np.where(np.any(overlap_matrix, axis=0))[0]
                    #
                    for j_atom in connected_ja_indices:
                        num_nz_blocks += 1
                        atom_pairs.append([R1, R2, R3, i_atom, j_atom])
                        chunk_shapes.append((orbit_quantity_list[i_atom], orbit_quantity_list[j_atom]))
                        chunk_boundaries.append(chunk_boundaries[-1] + orbit_quantity_list[i_atom] * orbit_quantity_list[j_atom])
                R_ijk.append([R1, R2, R3])
                R_boundaries.append(R_boundaries[-1] + num_nz_blocks)
        return {
            "atom_pairs": np.array(atom_pairs),
            "chunk_shapes": np.array(chunk_shapes),
            "chunk_boundaries": np.array(chunk_boundaries),
            "R_ijk": np.array(R_ijk),
            "R_boundaries": np.array(R_boundaries),
        }

    def _dump_h5(self, file_path, entries, isspinful):
        mat_info = self.matrix_info
        data = {
            "atom_pairs": mat_info["atom_pairs"],
            "chunk_shapes": mat_info["chunk_shapes"]*(isspinful+1),
            "chunk_boundaries": mat_info["chunk_boundaries"]*(isspinful+1)**2,
            "entries": entries,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)

    def _read_csr_headlines(self, f, tag, isspinful):
        line = f.readline()
        if "STEP" in line or "TEP" in line: # ABACUS >= 3.0
            line = f.readline()
        #-----------------------------------------------------------------------
        assert "Matrix Dimension of" in line, f'{tag}: "Matrix Dimension of" not found'
        orbits_quantity = int(line.split()[-1])
        orbits_quantity_from_log = self.orbits_quantity*(1+isspinful)
        assert orbits_quantity == orbits_quantity_from_log, f'{tag}: Matrix dimension is {orbits_quantity}, but {self.log_name} gives {orbits_quantity_from_log}'
        #-----------------------------------------------------------------------
        line = f.readline()
        assert "Matrix number of" in line, f'{tag}: "Matrix number of" not found'
        num_R = int(line.split()[-1])
        #-----------------------------------------------------------------------
        return orbits_quantity, num_R
    
    def _load_one_csr_matrix(self, f, num_nz, dim, complex_in, complex_out):
        if num_nz == 0:
            dtype = np.complex128 if complex_out else np.float64
            return csr_matrix((dim, dim), dtype=dtype)
        vals = f.readline().split()
        cols = np.array(f.readline().split()).astype(int)
        indptr = np.array(f.readline().split()).astype(int)
        if not complex_in:
            vals = np.array(vals).astype(np.float64)
            if complex_out:
                vals = vals.astype(np.complex128)
        else:
            vals = np.char.replace(vals, '(', '')
            vals = np.char.replace(vals, ')', 'j')
            vals = np.char.replace(vals, ',', '+')
            vals = np.char.replace(vals, '+-', '-')
            vals = np.array(vals).astype(np.complex128)
            if not complex_out:
                vals = np.real(vals)
        return csr_matrix((vals, cols, indptr), shape=(dim, dim))

    def _convert_one_csr_matrix(self, i_R, mat_R, entries, spinful_in, spinful_out):
        matrix_info = self.matrix_info
        orbit_cumsum = self.orbit_cumsum*(1+spinful_in)
        chunk_boundaries = matrix_info["chunk_boundaries"]*(1+spinful_out)**2
        for i_pair in range(
            matrix_info["R_boundaries"][i_R],matrix_info["R_boundaries"][i_R+1]
        ):
            atom_pair = matrix_info["atom_pairs"][i_pair, :]
            i_atom, j_atom = atom_pair[3], atom_pair[4]
            mat_block = mat_R[
                orbit_cumsum[i_atom]:orbit_cumsum[i_atom+1],
                orbit_cumsum[j_atom]:orbit_cumsum[j_atom+1]
            ]
            mat_block = np.array(mat_block.todense())
            transform_index1 = self.basis_trans_index[self.atom_elem[i_atom]]
            transform_index2 = self.basis_trans_index[self.atom_elem[j_atom]]
            transform_factor1 = self.basis_trans_factor[self.atom_elem[i_atom]]
            transform_factor2 = self.basis_trans_factor[self.atom_elem[j_atom]]
            mat_block = self._transform(
                mat_block, transform_index1, transform_index2, 
                transform_factor1, transform_factor2, spinful_in
            )
            if spinful_in and (not spinful_out):
                orbit_quantity_i = self.orbit_quantity_list[i_atom]
                orbit_quantity_j = self.orbit_quantity_list[j_atom]
                mat_block = mat_block[:orbit_quantity_i, :orbit_quantity_j]
            entries[chunk_boundaries[i_pair]:chunk_boundaries[i_pair+1]] = mat_block.reshape(-1)
        return entries

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

    def _transform(self, matrix, transform_index1, transform_index2, transform_factor1, transform_factor2, isspinful):
        if isspinful:
            a = matrix.shape[0] // 2
            b = matrix.shape[1] // 2
            matrix = matrix.reshape((a, 2, b, 2)).transpose((0, 2, 1, 3)).reshape((a, b, 4))
            matrix = matrix[transform_index1, :, :][:, transform_index2, :]
            matrix = matrix * transform_factor1[:, None, None] * transform_factor2[None, :, None]
            matrix = matrix.reshape((a, b, 2, 2)).transpose((2, 0, 3, 1)).reshape((2 * a, 2 * b))
            return matrix
        else:
            matrix = matrix[transform_index1, :][:, transform_index2]
            matrix = matrix * transform_factor1[:, None] * transform_factor2[None, :]
            return matrix

    def _get_conj_pair_sort_index(self):
        atom_pairs = self.matrix_info["atom_pairs"]
        conj_atom_pairs = np.concatenate([-atom_pairs[:, :3], atom_pairs[:, [4,3]]], axis=1)
        sort_origin_idx = np.lexsort(atom_pairs.mT)
        sort_conj_idx = np.lexsort(conj_atom_pairs.mT)
        assert np.allclose(atom_pairs[sort_origin_idx, :], conj_atom_pairs[sort_conj_idx, :]), f"{self.dir_name}: Some Rijk do not has its inverse!"
        rev_sort_conj_idx = np.argsort(sort_conj_idx)
        sort_idx = sort_origin_idx[rev_sort_conj_idx]
        return sort_idx

