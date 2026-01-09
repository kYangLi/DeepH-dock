from pyscf.pbc import gto
from pyscf import gto as molgto
from pyscf.pbc.gto.cell import Cell
import os
from ase.io import read
import numpy as np
from pymatgen.core import Lattice, Structure

Hartree2eV = 27.2113845
Bohr2Ang = 0.529177249
small_value = 1e-12
LEN_UNIT = 'Angstrom'
E_UNIT = 'eV'

def rint_array(vec:np.ndarray) -> np.ndarray:
    return np.asarray(np.rint(vec), dtype=int)

def rint(num) -> int:
    return int(np.rint(num))

def get_cell_from_POSCAR(poscar_path:str, basis:str|None=None):
    """Generate a PySCF Cell object from a POSCAR file by ase."""
    ase_cell = read(poscar_path)
    cell = Cell()
    cell.from_ase(ase_cell)
    if basis is not None:
        cell.basis = basis
    cell.unit = 'Ang'
    return cell

def get_cell_from_old_deeph(deeph_data_dir:str, basis:str|None=None):
    """Generate a PySCF Cell object from an old DeepH data directory."""
    # read structure info from structure.json
    element:list[int]|np.ndarray = rint_array(np.genfromtxt(os.path.join(deeph_data_dir, 'element.dat')))
    lat:np.ndarray = np.genfromtxt(os.path.join(deeph_data_dir, 'lat.dat')).T # each row is a lattice vector
    try:
        rlat:np.ndarray = np.genfromtxt(os.path.join(deeph_data_dir, 'rlat.dat')).T
    except:
        rlat = np.linalg.inv(lat) * 2*np.pi
    site_positions:np.ndarray = np.genfromtxt(os.path.join(deeph_data_dir, 'site_positions.dat')).T
    site_positions_frac = site_positions @ np.linalg.inv(lat) # frac pos

    element = list(element)
    lattice = Lattice(lat)
    syst = Structure(lattice, element, site_positions, coords_are_cartesian=True)

    lattice_au = syst.lattice.matrix / Bohr2Ang  # convert to Bohr
    atom_coords = syst.frac_coords @ lattice_au  # in Bohr
    cell = gto.M(a = lattice_au)
    cell.unit = 'Bohr'
    cell.atom = []
    for i in range(len(syst.atomic_numbers)):
        symbol = syst.atomic_numbers[i]
        x, y, z = atom_coords[i]
        cell.atom.append([symbol, (x, y, z)])
    if basis is not None:
        cell.basis = basis
    return cell

def get_mols_from_mat(mat_data:dict, basis:str="def2-SVP")->list[molgto.Mole]:
    ''' Generate a list of PySCF Mole objects from mat_data dictionary, such as qm7 dataset.
    Args:
        mat_data: dictionary containing 'Z' and 'R' keys
        basis: basis set to be used for the molecules
    Returns:
        mols: list of PySCF Mole objects

    Usage:
        >>> from scipy.io import loadmat
        >>> mat_data = loadmat('qm7.mat')
        >>> mols = get_mols_from_mat(mat_data, basis='def2-SVP')
    '''
    atomic_numbers = mat_data['Z']
    coordinates = mat_data['R']
    num_molecules = atomic_numbers.shape[0]
    mols = []
    for i in range(num_molecules):
        atom_nums = atomic_numbers[i].flatten()
        coords = coordinates[i]
        atom_list = []
        for j, atomic_num in enumerate(atom_nums):
            atomic_num = int(atomic_num)
            if atomic_num == 0:
                continue
            position = coords[j]
            position = np.asarray(position)
            atom_list.append([atomic_num, tuple(position)])
        mol = molgto.Mole()
        mol.atom = atom_list
        mol.basis = basis
        mol.unit = 'B'
        mol.spin = 0
        mol.build()
        mols.append(mol)
    
    return mols

def rotate(cell:gto.Cell, theta:float, phi:float)->gto.Cell:
    ''' rotate the cell and atom coordinates
    Args:
        cell: the cell to be rotated
        theta: the angle to rotate around y axis
        phi: the angle to rotate around z axis
    Returns:
        cell: the rotated cell
    '''
    # 绕y轴转theta
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    # 绕z轴转phi
    Rz = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    try:
        cell.a = np.dot(Rz, np.dot(Ry, cell.lattice_vectors().T)).T * Bohr2Ang
    except:
        print("cell.a is not defined, it is not a periodic system.")
    #print(cell.a)
    atom_coords = np.dot(Rz, np.dot(Ry, cell.atom_coords().T)).T * Bohr2Ang
    #print(atom_coords)
    atoms = cell.atom.split('\n')
    atoms_lst = [atoms[i].split() for i in range(len(atoms))]
    cell.atom = ''
    for i in range(len(atoms_lst)):
        cell.atom += atoms_lst[i][0] + ' ' + str(atom_coords[i][0]) + ' ' + str(atom_coords[i][1]) + ' ' + str(atom_coords[i][2]) + '\n'
    #print(cell.atom)
    cell.build()
    return cell