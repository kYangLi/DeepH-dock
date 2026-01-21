import numpy as np
import h5py
import json
import collections
from pathlib import Path
from ase.io import read

from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_OVERLAP_FILENAME, DEEPX_POSCAR_FILENAME
from deepx_dock.misc import get_data_dir_lister
from deepx_dock.convert.fhi_aims.parse_ctrl_in import read_from_control_in

structure_reference = '''
shape:
    im_mx: (n_hamiltonian_matrix_size)
    idx_mx: (2, n_cells_in_hamiltonian, n_basis), start and end index for each (i_cell, i_basis_row)
    col_idx_mx: (n_hamiltonian_matrix_size)
    cell_idx: (3, n_cells_in_hamiltonian)

matrix ref:
!             Example with cell_index(1,:) == 0,
!                          cell_index(2,:) == - cell_index(3,:):
!                       j=1   j=2
!             i=1 R=1 (  1     2  )   ! index_hamiltonian(:,1,1) = [ 1, 2]
!             i=2 R=1 (  2*    0  )   ! index_hamiltonian(:,1,2) = [ 0,-1]
!             i=1 R=2 (  0     3  )   ! index_hamiltonian(:,2,1) = [ 3, 3]
!             i=2 R=2 (  0     4  )   ! index_hamiltonian(:,2,2) = [ 4, 4]
!             i=1 R=3 (  0     0  )   ! index_hamiltonian(:,3,1) = [ 0,-1]
!             i=2 R=3 (  3*    4' )   ! index_hamiltoniah(:,3,2) = [ 5, 5]
!             hamiltonian              = [1, 2, 3, 4, 4']  # values
!             column_index_hamiltonian = [1, 2, 2, 2, 2]  # corresponding j
!             ! 2* and 3* are not stored because of symmetry [only 'U' stored]
!             ! 4  and 4' are equal because of symmetry but both stored.
!
!           Looping then looks like this:
!
!           do i_cell_row = 1, n_cells_in_hamiltonian-1   ! yes, "-1".
!             do i_basis_row = 1, n_basis
!               i_index_first = index_hamiltonian(1, i_cell_row, i_basis_row)
!               i_index_last = index_hamiltonian(2, i_cell_row, i_basis_row)
!               do i_index = i_index_first, i_index_last
!                 i_basis_col = column_index_hamiltonian(i_index)
!                 ! Use:
!                 !    hamiltonian(i_index, i_spin)
!                 !    density_matrix_sparse(i_index)
!                 !    and i_basis_row, i_cell_row, i_basis_col
!                 ! or any combination of
!                 !    (i_basis_row, i_loc_cell_row), &
!                 !    & (i_basis_col, i_loc_cell_col),
!                 ! with
!                 !    i_cell_row == &
!                 !    & position_in_hamiltonian(i_loc_cell_row, i_loc_cell_col)
!               end do
!             end do
!           end do
!
!
!           Just a few remarks:
!            * I (JW) would have named
!               + index_hamiltonian        -> row2index
!               + column_index_hamiltonian -> index2col
!              because the former specifies a range of indices for a given row
!              consisting of i_basis_row and i_cell_row.  The latter gives
!              the column (i_basis_col, as i_cell_col=0) for a given index.
!            * It is kind of unconventional in fortran that the fast index
!              corresponds to the column instead of the row.

basis ref:
    do i_basis = 1, n_basis, 1
      i_fn = basis_fn(i_basis)

      write(info_str,'(I5,1X,A8,1X,I3,1X,I3,1X,I3,1X,I3)') &
        i_basis, basisfn_type(i_fn), &
        basis_atom(i_basis), basisfn_n(i_fn), basis_l(i_basis), &
        basis_m(i_basis)
        !   fn.   type   at.   n   l   m
      call localorb_info(info_str,50,'(A)')

    enddo

    phase:
        (-1)**logical_and(m>0, m%2==1)
    order: from m=-l to m=+l (wikipedia real spherical harmonics: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics)
'''

HARTREE_TO_EV = 27.2113845 # 27.211386

AIMS_CONTROL_FILENAME = "control.in"
AIMS_STRUCT_FILENAME = "geometry.in"
AIMS_BASIS_FILENAME = "basis-indices.out"
FILES_MX_IDX = "rs_indices.out"
FILES_NECESSARY = set([AIMS_CONTROL_FILENAME, AIMS_STRUCT_FILENAME, AIMS_BASIS_FILENAME, FILES_MX_IDX])
FILES_LOG = 'aims.out'
FILES_IN = ["rs_overlap.out", "rs_hamiltonian.out"]
FILES_IN_SPIN = ["rs_overlap.out", "rs_hamiltonian_up.out", "rs_hamiltonian_dn.out"]  # not support spin orbit coupling yet
FILES_IN_HDF5 = ["rs_overlap.h5", "rs_hamiltonian.h5"]
FILES_IN_SPIN_HDF5 = ["rs_overlap.h5", "rs_hamiltonian_up.h5", "rs_hamiltonian_down.h5"]
FILES_OUT = [DEEPX_HAMILTONIAN_FILENAME, DEEPX_OVERLAP_FILENAME]
UNITS = [HARTREE_TO_EV, 1.0]
BASIS_TYPE_ORDER = {'ionic': 0, 'atomic': 1, 'hydro': 2}  # for sorting basis types

def validation_check_aims(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if FILES_NECESSARY.issubset(set(all_files)):
        yield prev_dirname

def _read_ctrl_in(aims_dir_path: Path):
    ctrl_in_path = Path(aims_dir_path) / AIMS_CONTROL_FILENAME
    ctrl_params = read_from_control_in(ctrl_in_path)
    spinful = False
    if 'spin' in ctrl_params.keys():
        if ctrl_params["spin"].lower() == 'collinear':
            spinful = True
    aims_data_type:str = ctrl_params.get('output_rs_matrices', None)
    if not aims_data_type:
        raise ValueError("The 'output_rs_matrices' parameter is missing in control.in!")
    #if aims_data_type.lower() == 'hdf5':
    #    raise NotImplementedError("The 'hdf5' output_rs_matrices type is not supported yet!")
    if aims_data_type.lower() not in ['plain', 'hdf5']:
        raise ValueError(f"Unsupported 'output_rs_matrices' type: {aims_data_type}!")

    return ctrl_params, spinful, aims_data_type.lower()

def _parse_struct(aims_dir_path: Path):
    atoms_path = Path(aims_dir_path) / AIMS_STRUCT_FILENAME
    atoms = read(atoms_path)
    ''' NOT change orginal atoms object '''
    is_periodic = False
    if atoms.pbc.any():
        is_periodic = True
        lat = atoms.cell.array
    if not is_periodic:
        lat = np.array([[5.2917721e8,0,0],[0,5.2917721e8,0],[0,0,5.2917721e8]])
    
    site_positions = atoms.get_positions()
    element = atoms.get_atomic_numbers()
    species = atoms.get_chemical_symbols()

    # Sort atoms to match VASP POSCAR grouping requirement
    # Sort by species to ensure grouped elements (e.g. all C, then all H)
    # Use stable sort to preserve relative order of atoms of the same species
    sort_idxs = np.argsort(species, kind='stable')
    site_positions = site_positions[sort_idxs]
    element = element[sort_idxs]
    species = [species[i] for i in sort_idxs]

    return is_periodic, lat, site_positions, element, species, sort_idxs

def _parse_basis(aims_dir_path: Path, atomic_num: int, species: list[str], sort_idxs: np.ndarray):
    basis_path = Path(aims_dir_path) / AIMS_BASIS_FILENAME

    basis_types = np.loadtxt(
            basis_path, dtype=str, usecols=(1,), skiprows=2
        ) # 'atomic', 'ionic', 'hydro', ...
    basis_indices = np.loadtxt(
            basis_path, 
            dtype=int, usecols=(0,2,3,4,5), skiprows=2
    ) # shape: [N_orb, 5], 5: i_orb, i_atom, n, l, m
    # !!!!!!!!!!!!!!! be careful with i_orb and i_atom are start from 1 !!!!!!!!!!!!!!!!!!!!!
    
    N_orb:int = basis_indices.shape[0]
    assert N_orb == basis_indices[-1,0], "Basis number not match!"
    N_atom:int = max(basis_indices[:, 1])
    assert N_atom == atomic_num, "Atom number not match!"
    basis_indices[:, 0] -= 1  # i_orb start from 0
    basis_indices[:, 1] -= 1  # i_atom start from 0
    
    # Remap atom indices according to the sorting
    old2new = np.zeros(N_atom, dtype=int)
    old2new[sort_idxs] = np.arange(N_atom)
    basis_indices[:, 1] = old2new[basis_indices[:, 1]]

    phase_factor = (-1)**(
            np.logical_and(basis_indices[:,-1] > 0, basis_indices[:,-1] % 2 == 1)
    ) # m > 0 and odd -> -1, else 1    
    # TODO: like SingleAtom, parse basis types into more detailed info if necessary

    orbit_quantity_list:list[int] = [int(0)] * N_atom   # for matrix info
    # Sort indices based on keys: l > n > atom_index > m > basis_type
    _sorted_indices = sorted(range(basis_indices.shape[0]), 
                             key=lambda k: (basis_indices[k,3], basis_indices[k,2],
                                            basis_indices[k,1], basis_indices[k,4],
                                            BASIS_TYPE_ORDER[basis_types[k]]))
    atom_elem_dict:dict[str, int] = collections.Counter(species)  # for POSCAR
    _elem_orb_map:dict[int, list[str]] = {}
    elem_orb_map:dict[str, list[int]] = {}
    basis_trans_index:dict[int, list[int]] = {}  # {idx_atom: [new_basis_indices...]}
    
    for ao in basis_indices:
        atom_index = ao[1]
        orbit_quantity_list[atom_index] += 1

    for atom_idx in range(N_atom):
        _elem_orb_map[atom_idx] = []
        basis_trans_index[atom_idx] = []

    for old_idx in _sorted_indices:
        atom_index, ll, n, m = basis_indices[old_idx,1], basis_indices[old_idx,3],\
                                basis_indices[old_idx,2], basis_indices[old_idx,4]
        basis_trans_index[atom_index].append(old_idx)
        orb_type:str = basis_types[old_idx]
        if f"{orb_type},{n},{ll}" not in _elem_orb_map[atom_index]:
                _elem_orb_map[atom_index].append(f"{orb_type},{n},{ll}")

    for atom_idx, orb_map in _elem_orb_map.items():
        elem = species[atom_idx]
        elem_orb_map[elem] = [int(inl.split(',')[2]) for inl in orb_map] # only l, for info

    _sub_idx_count = np.zeros(N_atom, dtype = int)
    sub_idx = np.zeros((N_orb, 3), dtype=int)  # row_index = old_index: (atom_index, new_index, new_orb_sub_index_in_atom)
    
    for new_idx in range(N_orb):
        old_idx = _sorted_indices[new_idx]
        atom_index = basis_indices[old_idx,1]
        sub_idx[old_idx, 0] = atom_index
        sub_idx[old_idx, 1] = new_idx
        sub_idx[old_idx, 2] = _sub_idx_count[atom_index]
        _sub_idx_count[atom_index] += 1

    # BE CAREFUL: the phase_factor is corresponding to the old basis order
    return phase_factor, orbit_quantity_list, atom_elem_dict, elem_orb_map, basis_trans_index, N_atom, N_orb, sub_idx

def _read_mx_indices(aims_dir_path: Path):
    '''
    type of rs_indices.out:
    <not a comment line>
n_hamiltonian_matrix_size:        91157
n_cells_in_hamiltonian:          106
n_basis:           90
cell_index
    <3 int each line, total n_cells_in_hamiltonian lines>
index_hamiltonian(1,:,:)  # 1=strat_idx, idx_cell, idx_orb
    <n_basis int each line, total n_cells_in_hamiltonian lines>
index_hamiltonian(2,:,:)  # 2=end_idx, idx_cell, idx_orb
    <n_basis int each line, total n_cells_in_hamiltonian lines>
column_index_hamiltonian
    <1 int each line, total n_hamiltonian_matrix_size lines>

    the last line of the cell_index is 99999999, 99999999, 99999999
    and the last of the column_index_hamiltonian is 0
    thus the last line of the (1,:,:), (2,:,:) and col_idx is useless
    '''
    mx_idx_path = Path(aims_dir_path) / FILES_MX_IDX
    with open(mx_idx_path, 'r') as f:
        lines = f.readlines()
    
    n_ham_size = int(lines[0].split()[1])
    n_cells = int(lines[1].split()[1])
    n_basis = int(lines[2].split()[1])

    idx_line = 3
    # Skip header "cell_index"
    if "cell_index" in lines[idx_line]:
        idx_line += 1
    
    cell_indices = []
    for _ in range(n_cells):
        cell_idx_line = lines[idx_line].split()
        cell_indices.append( (int(cell_idx_line[0]), int(cell_idx_line[1]), int(cell_idx_line[2])) )
        idx_line += 1
    
    # Skip header "index_hamiltonian(1,:,:)"
    if "index_hamiltonian" in lines[idx_line]:
        idx_line += 1

    start_idx_matrix = np.zeros( (n_cells, n_basis), dtype=int )
    for i in range(n_cells):
        start_idx_line = lines[idx_line].split()
        for j in range(n_basis):
            start_idx_matrix[i, j] = int(start_idx_line[j])
        idx_line += 1
    
    # Skip header "index_hamiltonian(2,:,:)"
    if "index_hamiltonian" in lines[idx_line]:
        idx_line += 1

    end_idx_matrix = np.zeros( (n_cells, n_basis), dtype=int )
    for i in range(n_cells):
        end_idx_line = lines[idx_line].split()
        for j in range(n_basis):
            end_idx_matrix[i, j] = int(end_idx_line[j])
        idx_line += 1

    # Skip header "column_index_hamiltonian"
    if "column_index" in lines[idx_line]:
        idx_line += 1

    col_idx = np.zeros( (n_ham_size,), dtype=int )
    for i in range(n_ham_size):
        col_idx_line = lines[idx_line].split()
        col_idx[i] = int(col_idx_line[0])
        idx_line += 1

    if np.asarray(cell_indices[-1], dtype=int).sum() > 99999:
        n_cells -= 1
        cell_indices = cell_indices[:-1]
        start_idx_matrix = start_idx_matrix[:-1, :]
        end_idx_matrix = end_idx_matrix[:-1, :]

    if col_idx[-1] == 0:
        col_idx = col_idx[:-1]
        n_ham_size -= 1

    # fortran to python index
    start_idx_matrix -= 1   # be careful with the index -1 is original 0
    # for end_idx_matrix, -1 is original -1
    col_idx -= 1            # not have index <0

    return n_ham_size, n_cells, n_basis, \
            np.array(cell_indices, dtype=int), \
            start_idx_matrix, end_idx_matrix, col_idx

def _read_mx_val(file_path: Path, n_ham_size: int):
    # file type: txt, n_ham_size lines, each line: value, float64
    mx_values = np.loadtxt(file_path, dtype=np.float64)
    if mx_values.shape[0] == n_ham_size + 1:
        # Check if the extra value is indeed zero-padding (common in some aims outputs)
        # But primarily rely on shape
        mx_values = mx_values[:-1]
    else:
        assert mx_values.shape[0] == n_ham_size, f"Matrix size match error: expected {n_ham_size}, got {mx_values.shape[0]}"
    return mx_values

def _read_ovlp(aims_dir_path: Path, n_ham_size: int):
    ovlp_path = Path(aims_dir_path) / FILES_IN[0]
    return _read_mx_val(ovlp_path, n_ham_size)

def _read_ham(aims_dir_path: Path, n_ham_size: int, spinful: bool):
    if spinful:
        ham_up_path = Path(aims_dir_path) / FILES_IN_SPIN[1]
        ham_dn_path = Path(aims_dir_path) / FILES_IN_SPIN[2]
        ham_up_values = _read_mx_val(ham_up_path, n_ham_size) * HARTREE_TO_EV
        ham_dn_values = _read_mx_val(ham_dn_path, n_ham_size) * HARTREE_TO_EV
        return ham_up_values, ham_dn_values
    else:
        ham_path = Path(aims_dir_path) / FILES_IN[1]
        ham_values = _read_mx_val(ham_path, n_ham_size) * HARTREE_TO_EV
        return ham_values

def _read_mx_val_hdf5(file_path: Path, n_ham_size: int):
    # file type: hdf5, dataset 'sparse_matrix', size n_ham_size
    with h5py.File(file_path, 'r') as f:
        mx_values = f['sparse_matrix'][:]
    mx_values = np.array(mx_values, dtype=np.float64)
    if mx_values.shape[0] == n_ham_size + 1:
        # Check if the extra value is indeed zero-padding (common in some aims outputs)
        # But primarily rely on shape
        mx_values = mx_values[:-1]
    else:
        assert mx_values.shape[0] == n_ham_size, f"Matrix size match error: expected {n_ham_size}, got {mx_values.shape[0]}"
    return mx_values

def _read_ovlp_hdf5(aims_dir_path: Path, n_ham_size: int):
    ovlp_path = Path(aims_dir_path) / FILES_IN_HDF5[0]
    return _read_mx_val_hdf5(ovlp_path, n_ham_size)

def _read_ham_hdf5(aims_dir_path: Path, n_ham_size: int, spinful: bool):
    if spinful:
        ham_up_path = Path(aims_dir_path) / FILES_IN_SPIN_HDF5[1]
        ham_dn_path = Path(aims_dir_path) / FILES_IN_SPIN_HDF5[2]
        ham_up_values = _read_mx_val_hdf5(ham_up_path, n_ham_size) * HARTREE_TO_EV
        ham_dn_values = _read_mx_val_hdf5(ham_dn_path, n_ham_size) * HARTREE_TO_EV
        return ham_up_values, ham_dn_values
    else:
        ham_path = Path(aims_dir_path) / FILES_IN_HDF5[1]
        ham_values = _read_mx_val_hdf5(ham_path, n_ham_size) * HARTREE_TO_EV
        return ham_values

def _read_fermi(aims_dir_path: Path):
    log_path = Path(aims_dir_path) / FILES_LOG
    fermi_energy = 0.0
    '''
    ref:
        | Chemical potential (Fermi level) in eV      :  -.53575976E+01
    '''
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "| Chemical potential (Fermi level) in eV" in line:
            fermi_energy = float(line.split(':')[-1].strip())
            break
    return fermi_energy

def _trans_mxs_to_entries(start_idx_matrix:np.ndarray, end_idx_matrix:np.ndarray, col_idx:np.ndarray, 
                          cell_indices:np.ndarray, orbit_quantity_list: list[int], phase_factor: np.ndarray,
                          mxs: list[np.ndarray], n_cells:int, n_basis:int, sub_idx:np.ndarray):
    '''
    deeph data structure ref:
        atom_pairs: (n_atom_pairs, 5), int
                    each row: [R1, R2, R3; i_atom, j_atom]
        chunk_boundaries: (n_atom_pairs+1,), int
                    each element: starting index of each chunk in atom_pairs
        chunk_shapes: (n_atom_pairs, 2), int
                    each row: [n_rows, n_cols] of the corresponding chunk
                            n_rows_i * n_cols_i = chunk_boundaries(i+1) - chunk_boundaries(i)
        entries: (M,), double if spinless or ovlp, complex double if spinful
                    sorted entries of all chunks concatenated
    '''
    num_mx = len(mxs)
    mx_R:dict[tuple, list[np.ndarray]] = {}
    entries_lst = [ [] for _ in range(num_mx) ]  # list of entries for each matrix

    for idx_R in range(n_cells):
        nR:np.ndarray = cell_indices[idx_R] # (R1, R2, R3)
        for idx_i_basis in range(n_basis):
            # old orb index is idx_i_basis
            i_atom = sub_idx[idx_i_basis, 0]   # (atom_index, new_index, new_orb_sub_index_in_atom)
            start_idx_in_val = start_idx_matrix[idx_R, idx_i_basis]
            end_idx_in_val = end_idx_matrix[idx_R, idx_i_basis]
            if start_idx_in_val < 0 or end_idx_in_val < 0 or end_idx_in_val < start_idx_in_val - 0.1:
                continue
            for idx_in_val in range(start_idx_in_val, end_idx_in_val):
                idx_j_basis = col_idx[idx_in_val]  # old orb index
                j_atom = sub_idx[idx_j_basis, 0]
                R_key = (nR[0], nR[1], nR[2], i_atom, j_atom)
                R_hermi_key = (-nR[0], -nR[1], -nR[2], j_atom, i_atom)
                if R_key not in mx_R.keys():
                    mx_R[R_key] = [np.zeros((orbit_quantity_list[i_atom], orbit_quantity_list[j_atom])
                                            , dtype=mx.dtype) for mx in mxs]
                if R_hermi_key not in mx_R.keys():
                    mx_R[R_hermi_key] = [np.zeros((orbit_quantity_list[j_atom], orbit_quantity_list[i_atom])
                                                 , dtype=mx.dtype) for mx in mxs]
                # fill in values
                idx_row_sub = sub_idx[idx_i_basis, 2]
                idx_col_sub = sub_idx[idx_j_basis, 2]
                for mx_idx in range(num_mx):
                    single_val = mxs[mx_idx][idx_in_val] * phase_factor[idx_i_basis] * phase_factor[idx_j_basis]
                    mx_R[R_key][mx_idx][idx_row_sub, idx_col_sub] = single_val
                    # Fill Hermitian Part
                    # For Spin-Orbit Coupling with Complex Hamiltonian, need conjugate here
                    # But standard FHI-aims text output usually Real (or separated). 
                    # Assuming Real for now based on dtype=float load.
                    if abs(mx_R[R_hermi_key][mx_idx][idx_col_sub, idx_row_sub]) <= 1e-10:
                        mx_R[R_hermi_key][mx_idx][idx_col_sub, idx_row_sub] = single_val  # real matrix assumed
                    else:
                        assert abs(mx_R[R_hermi_key][mx_idx][idx_col_sub, idx_row_sub] - single_val) < 1e-6, \
                            f"Hermitian check failed at R={nR}, atom pair=({i_atom},{j_atom}), orb=({idx_i_basis},{idx_j_basis})"

    num_of_atom_pairs = len(mx_R.keys())
    atom_pairs = np.zeros((num_of_atom_pairs, 5), dtype=int)
    chunk_boundaries = np.zeros((num_of_atom_pairs + 1,), dtype=int)
    chunk_shapes = np.zeros((num_of_atom_pairs, 2), dtype=int)
    idx_pair = 0
    for atom_pair_key, sub_mx_lst in mx_R.items():
        atom_pairs[idx_pair, :] = np.array(atom_pair_key, dtype=int)
        n_rows, n_cols = sub_mx_lst[0].shape
        chunk_shapes[idx_pair, 0] = n_rows
        chunk_shapes[idx_pair, 1] = n_cols
        for mx_idx in range(num_mx):
            entries_lst[mx_idx].append( sub_mx_lst[mx_idx].flatten())
        chunk_boundaries[idx_pair + 1] = chunk_boundaries[idx_pair] + n_rows * n_cols
        idx_pair += 1
    entries = [ np.concatenate(entries_sublist) for entries_sublist in entries_lst ]
    
    return atom_pairs, chunk_boundaries, chunk_shapes, entries

class PeriodicAimsDataTranslator:
    def __init__(self,
        aims_data_dir, deeph_data_dir, export_rho=False, export_r=False,
        n_jobs=1, n_tier=0
    ):
        self.aims_data_dir = Path(aims_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.export_rho = export_rho
        self.export_r = export_r
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        self.deeph_data_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_aims_to_deeph(self):
        worker = partial(self.transfer_one_aims_to_deeph,
                         aims_path=self.aims_data_dir,
                         deeph_path=self.deeph_data_dir,
                         export_rho=self.export_rho,
                         export_r=self.export_r,
        )
        data_dir_lister = get_data_dir_lister(
            self.aims_data_dir, self.n_tier, validation_check_aims
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data")
        ) # -1 if not have aims.out, 0 if have aims.out and get the fermi level
        # Count errors
        n_err = sum(1 for r in results if r is not None and r != 0)
        if n_err > 0:
            print(f"Warning: {n_err} directories can not find log file: {FILES_LOG}. \nThus the Fermi level is not extracted correctly and set to 0.0 eV.")

    @staticmethod
    def transfer_one_aims_to_deeph(dir_name: str, aims_path: Path, deeph_path: Path,
                                  export_rho=False, export_r=False):
        aims_dir_path = aims_path / dir_name
        if not aims_dir_path.is_dir():
            return
        deeph_dir_path = deeph_path / dir_name
        reader = FHIAimsReader(aims_dir_path, deeph_dir_path)
        ierr = reader.analysis_data()
        reader.dump_data(export_rho=export_rho, export_r=export_r)
        return ierr

class FHIAimsReader:
    def __init__(self, aims_path: str | Path, deeph_path: str | Path):
        self.aims_path = Path(aims_path)
        self.deeph_path = Path(deeph_path)
        self.mx_lst:list[np.ndarray] = []  # list of sparse matrix

    def analysis_data(self):
        # ------------ calculation parameters from control.in ------------
        self.ctrl_params, self.spinful, self.aims_data_type = _read_ctrl_in(self.aims_path)  # DONE: plain / hdf5
        # ------------ structure info from geometry.in ------------
        self.is_periodic, self.lat, self.site_positions, \
        self.element, self.species, sort_idxs = _parse_struct(self.aims_path)
        assert self.is_periodic, "Only periodic system is supported!"
        # ------------ basis info from basis-indices.out ------------
        self.phase_factor, self.orbit_quantity_list, self.atom_elem_dict, \
        self.elem_orb_map, self.basis_trans_index, N_atom, N_orb, self.sub_idx = _parse_basis(
            self.aims_path, len(self.element), self.species, sort_idxs
        )
            # check
        assert N_atom == len(self.element), "Atom number not match!"
        self.natom = N_atom
        # ------------ matrix indices from rs_indices.out ------------
        self.n_ham_size, self.n_cells, self.n_basis, self.cell_indices, \
        self.start_idx_matrix, self.end_idx_matrix, self.col_idx = _read_mx_indices(self.aims_path)
            # check
        assert N_orb == sum(self.orbit_quantity_list) == self.n_basis, "Basis number not match!" 
        # ------------ matrix values from rs_overlap.out and rs_hamiltonian.out ------------
        self.mx_lst = []
        if self.aims_data_type == 'plain':
            self.mx_lst.append(_read_ovlp(self.aims_path, self.n_ham_size))
            if self.spinful:
                ham_up_values, ham_dn_values = _read_ham(self.aims_path, self.n_ham_size, self.spinful)
                self.mx_lst.append(ham_up_values)
                self.mx_lst.append(ham_dn_values)
            else:
                self.mx_lst.append(_read_ham(self.aims_path, self.n_ham_size, self.spinful))
        elif self.aims_data_type == 'hdf5':
            self.mx_lst.append(_read_ovlp_hdf5(self.aims_path, self.n_ham_size))
            if self.spinful:
                ham_up_values, ham_dn_values = _read_ham_hdf5(self.aims_path, self.n_ham_size, self.spinful)
                self.mx_lst.append(ham_up_values)
                self.mx_lst.append(ham_dn_values)
            else:
                self.mx_lst.append(_read_ham_hdf5(self.aims_path, self.n_ham_size, self.spinful))
        else:
            raise ValueError(f"Unsupported aims_data_type: {self.aims_data_type}!")
        # ------------ transform to deeph data structure ------------
        self.atom_pairs, self.chunk_boundaries, self.chunk_shapes, self.entries_lst = _trans_mxs_to_entries(
            self.start_idx_matrix, self.end_idx_matrix, self.col_idx, self.cell_indices,
            self.orbit_quantity_list, self.phase_factor, self.mx_lst, self.n_cells, self.n_basis, self.sub_idx
        )
        # ------------ spin concatenate if necessary ------------
        if self.spinful:
            ham_up_entries = self.entries_lst[1]
            ham_dn_entries = self.entries_lst[2]
            # Reset entries list to keep only [0] (Overlap)
            self.entries_lst = [self.entries_lst[0]]
            
            ham_entries_lst = []
            for idx_pair, _chunk_shape in enumerate(self.chunk_shapes):
                n_rows, n_cols = _chunk_shape
                # Extract up/down blocks (flattened)
                start = self.chunk_boundaries[idx_pair]
                end = self.chunk_boundaries[idx_pair+1]
                block_up_flat = ham_up_entries[start:end]
                block_dn_flat = ham_dn_entries[start:end]
                
                # Reshape to 2D
                block_up = block_up_flat.reshape(n_rows, n_cols)
                block_dn = block_dn_flat.reshape(n_rows, n_cols)
                
                # Create 2x2 spinor block
                #   [ UU  UD ]
                #   [ DU  DD ]
                # For collinear: UD=0, DU=0
                block_spin = np.zeros((2*n_rows, 2*n_cols), dtype=np.complex128)
                block_spin[0:n_rows, 0:n_cols] = block_up
                block_spin[n_rows:, n_cols:] = block_dn

                ham_entries_lst.append(block_spin.flatten())
            
            self.entries_lst.append(np.concatenate(ham_entries_lst))
        # ------------ fermi energy from aims.out ------------
        self.fermi_level = 0.00000
            # check if file exists
        if (self.aims_path / FILES_LOG).is_file():
            self.fermi_level = _read_fermi(self.aims_path)
            return 0
        else:
            return -1

    def dump_data(self, export_rho=False, export_r=False):
        # ------------ make deeph data dir ------------
        self.deeph_path.mkdir(parents=True, exist_ok=True)
        # ------------ dump POSCAR ------------
        self._dump_poscar()
        # ------------ dump info.json ------------
        self._dump_info_json()
        # ------------ dump S.h5 ------------
        self._dump_S()
        # ------------ dump H.h5 ------------
        self._dump_H()


    def _dump_poscar(self):
        file_path = self.deeph_path / DEEPX_POSCAR_FILENAME
        
        poscar = [
            "POSCAR generated by DeepH-dock \n",
            "1.0\n",
            '  ' + ' '.join(map(str, self.lat[0])) + '\n',
            '  ' + ' '.join(map(str, self.lat[1])) + '\n',
            '  ' + ' '.join(map(str, self.lat[2])) + '\n',
            ' '.join(self.atom_elem_dict.keys()) + '\n',
            ' '.join(map(str, self.atom_elem_dict.values())) + '\n',
            "Cartesian\n",
        ] + [
            '  ' + ' '.join([f"{x:.16f}" for x in self.site_positions[i]]) + '\n'
            for i in range(self.natom)
        ]
        with open(file_path, 'w') as fwp:
            fwp.writelines(poscar)

    def _dump_info_json(self):
        file_path = self.deeph_path / DEEPX_INFO_FILENAME
        info_json = {
            "atoms_quantity": int(self.natom),
            "orbits_quantity": int(self.n_basis),
            "orthogonal_basis": False,
            "spinful": bool(self.spinful),
            "fermi_energy_eV": self.fermi_level,
            "elements_orbital_map": self.elem_orb_map,
        }
        with open(file_path, 'w') as fwj:
            json.dump(info_json, fwj)

    def _dump_S(self):
        file_path = self.deeph_path / DEEPX_OVERLAP_FILENAME
        with h5py.File(file_path, 'w') as fwh:
            fwh.create_dataset(
                'atom_pairs', data=self.atom_pairs, dtype='i4'
            )
            fwh.create_dataset(
                'chunk_boundaries', data=self.chunk_boundaries, dtype='i4'
            )
            fwh.create_dataset(
                'chunk_shapes', data=self.chunk_shapes, dtype='i4'
            )
            fwh.create_dataset(
                'entries', data=self.entries_lst[0]
            )
    
    def _dump_H(self):
        file_path = self.deeph_path / DEEPX_HAMILTONIAN_FILENAME
        # spinful: 4x entries (2N x 2N matrix), so boundaries * 4, shapes * 2
        boundary_factor = 4 if self.spinful else 1
        shape_factor = 2 if self.spinful else 1
        
        with h5py.File(file_path, 'w') as fwh:
            fwh.create_dataset(
                'atom_pairs', data=self.atom_pairs, dtype='i4'
            )
            fwh.create_dataset(
                'chunk_boundaries', data=self.chunk_boundaries * boundary_factor, dtype='i4'
            )
            fwh.create_dataset(
                'chunk_shapes', data=self.chunk_shapes * shape_factor, dtype='i4'
            )
            fwh.create_dataset(
                'entries', data=self.entries_lst[1]
            )

    # TODO: parallel HDF5 support case aims_save_type='hdf5'
    # TODO: density matrix, real-space grid V, etc
    # TODO: only dump S if we can calc S separately?
    # TODO: support non-collinear spin and SOC cases
    