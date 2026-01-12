import numpy as np
import collections
import json
import h5py
import os, warnings

import pyscf
from pyscf.pbc import gto as pbcgto

from functools import wraps

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSITION_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL

BOHR_TO_ANGSTROM = 0.529177249
HARTREE_TO_EV = 27.2113845
LEN_UNIT = 'Angstrom'
E_UNIT = 'eV'

def BASIS_TRANS_PYSCF2WIKI(ll):
    if ll == 0:
        return np.array([0])
    elif ll == 1: # human friendly form
        return np.array([1, 2, 0]) # (px, py, pz) -> (py, pz, px)
    elif ll == 2: # human friendly form
        return np.array([0,1,2,3,4]) #(dxy, dyz, dz2, dxz, dx2-y2)
    elif ll == 3: # regular form
        return np.array([0,1,2,3,4,5,6]) # (f-3, f-2, f-1, f0, f1, f2, f3)
    else: # similar to ll = 3 case
        #TODO: to be checked for ll > 3
        return np.arange(2*ll + 1, dtype=int) # wiki

def BASIS_TRANS_WIKI2PYSCF(ll):
    return np.argsort(BASIS_TRANS_PYSCF2WIKI(ll))

def rebuild_kpt_grid(lattice, kpts):
    # rebuild kpt grid to make sure it is compatible with lattice vectors
    lat = lattice.T / (2 * np.pi)  # columns are lattice vectors
    kpts_frac = kpts @ lat  # fractional coordinates
    k_grid = []
    for i in range(3):
        unique_coords = np.unique(np.round(kpts_frac[:, i], decimals=6))
        k_grid.append(len(unique_coords))
    
    return np.asarray(k_grid, dtype=int)

class PySCFDataHooker:
    '''
    A hooker class to collect SCF data from PySCF calculation for DeepH-type data.

    Args:
        deeph_path: str, path to export DeepH data files.
        mf: 
            RKS/KRKS: Restricted closed-shell Kohn-Sham DFT, without spin polarization. \\
            UKS/KUKS, ROKS/KROKS: Unrestricted Kohn-Sham DFT, with collinear spin polarization.\\
            GKS/KGKS: Generalized Kohn-Sham DFT, with SOC and non-collinear spin polarization.

    Usage: 

        >>> import pyscf
        >>> from deepx_dock.io.PySCF import SCFDataHook
        >>> mol = pyscf.gto.Mole()
        >>> mol.build(
        >>>     atom = \'\'\'O 0 0 0; H  0 1 0; H 0 0 1\'\'\',
        >>>     basis = 'sto-3g')
        >>> mol.build()
        >>> mf = pyscf.dft.RKS(mol)
        >>> mf.xc = 'lda,vwn'
        >>> mf = PySCFDataHooker(deeph_path='./deeph_data/dft/0')(mf) # only need to add single line to hook
        >>> mf.kernel()
    '''
    def __init__(self, deeph_path:str, rcut=None,
        export_S=True, export_H=True, export_rho=False, export_r=False
    ):
        self.deeph_path = deeph_path
        # mkdir if not exist
        if not os.path.exists(self.deeph_path):
            os.makedirs(self.deeph_path)
        self.kpts = np.array([[0.0, 0.0, 0.0]])
        self._org_kpts = np.array([[0.0, 0.0, 0.0]])
        self.rcut = rcut
        self.is_kpts_reset = False
        self.export_S = export_S; self._Sk = None # (nkpts, nao, nao)
        self.export_H = export_H; self._Hk = None
        self.export_rho = export_rho; self._rhok = None
        self.export_r = export_r
        assert not export_r, "The extraction of position matrix is not supported yet!"

        self.fermi_energy = 0.0
        self.unit_trans_factor = {'length': BOHR_TO_ANGSTROM, 'energy': HARTREE_TO_EV}

        # in pyscf, the definition of spherical harmonics is string below:
        self.lm = {
            's': {'': (0, 0)},
            'p': {'x': (1, 1), 'y': (1, -1), 'z': (1, 0)},
            'd': {
                'xy': (2, -2), 'xz': (2, 1), 'yz': (2, -1),
                'x2-y2': (2, 2), 'z^2': (2, 0)
            },
            'f': {
                '-3':(3,-3), '-2':(3,-2), '-1':(3,-1), '+0':(3,0),
                '+1':(3,1), '+2':(3,2), '+3':(3,3)
            },
        }

    def __call__(self, mf, kpt=None):
        return self.hook_kernel(mf, kpt=kpt)

    def hook_kernel(self, mf, kpt=None):
        '''hook the kernel function of mf object to collect data after calculation. must be called before mf.kernel()

        Args:
            mf: 
                RKS/KRKS: Restricted closed-shell Kohn-Sham DFT, without spin polarization. \\
                UKS/KUKS, ROKS/KROKS: Unrestricted Kohn-Sham DFT, with collinear spin polarization.\\
                GKS/KGKS: Generalized Kohn-Sham DFT, with SOC and non-collinear spin polarization.
            kpt: 
                np.ndarray, shape (3,) of grid points to be calculated, or shape (nkpts, 3) of k-points list.
                Default: None, use mf.kpts attribute.
        Returns:
            mf: the modified mf object with hooked kernel function
        '''
        if hasattr(mf, 'is_hooked') and mf.is_hooked:
            return mf
        setattr(mf, 'is_hooked', False)

        # step 1. analyze cell to get structure and orbital infomation
        self._analyze_info(mf, kpt=kpt)
        # setup collector as callback function in order to collect 
        # hamiltonian, overlap, density matrices.
        mf.callback = self._collector
        
        # step 2. wrap the original kernel function
        old_kernel = mf.kernel

        if self.export_S and not (self.export_H or self.export_rho):
            raise Warning("Overlap matrix export is enabled, but neither Hamiltonian nor Density matrix export is enabled. \n" \
            "Please use PySCFDataHooker.dump_ovlp() method to dump overlap matrix separately. \n" \
            "Overlap matrix will not be dumped.")

        @wraps(old_kernel)
        def new_kernel(*args, **kwargs):
            old_kernel(*args, **kwargs) # run the original kernel function first
            # while all calc done
            if self.is_periodic:
                try:
                    self.fermi_energy = mf.get_fermi() * self.unit_trans_factor['energy']
                except:
                    self.fermi_energy = np.max(np.asarray(mf.get_fermi())) * self.unit_trans_factor['energy']
            else:
                self._Hk = [self._Hk]
                self._Sk = [self._Sk]
                self._rhok = [self._rhok]
            # ------------------------ #
            # for kpt reset case
            if self.is_kpts_reset and self.is_periodic:
                # recalc matrices at original kpts
                self._Hk, self._Sk, self._rhok = self._recalc_mx_k(mf, self.kpts)
            # ------------------------ #
            self._transfer_and_dump()
        
        mf.kernel = new_kernel
        mf.is_hooked = True
        return mf
    
    def dump(self, mf, kpt=None):
        """ dump data if mf has been calculated already.
        """
        self._analyze_info(mf, kpt=kpt)
        if self.is_periodic:
            try:
                self.fermi_energy = mf.get_fermi() * self.unit_trans_factor['energy']
            except Exception:
                self.fermi_energy = np.max(np.asarray(mf.get_fermi())) * self.unit_trans_factor['energy']
        self._Hk, self._Sk, self._rhok = self._recalc_mx_k(mf, self.kpts)
        self._transfer_and_dump()
    
    def dump_ovlp(self, mf, kpt=None, return_S = False):
        '''calc and dump overlap matrix only '''
        self._analyze_info(mf, kpt=kpt)

        if not self.is_periodic:
            _Sk = [mf.get_ovlp(self.mol)]
            _SR_dict = self._get_realspace_matrices_ift(_Sk)
            _S = self._get_entries(_SR_dict, isspinful=0)
            self._dump_S(entries=_S)
        else:
            # fix: PySCF uses Bohr unit for kpts, but self.kpts is in Angstrom unit
            _Sk = mf.get_ovlp(self.cell, self.kpts * self.unit_trans_factor['length'])
            _SR_dict = self._get_realspace_matrices_ift(_Sk)
            _S = self._get_entries(_SR_dict, isspinful=0)
            self._dump_S(entries=_S)
        if return_S:
            return _S

    def _analyze_info(self, mf, kpt=None):
        # judge pbc or molecular system, get cell and set unit transform factor
        self.is_periodic = self._judge_pbc_get_cell(mf) # kpts
        self._reset_kpts(kpt=kpt) # reset kpts if necessary
        # get atomic info
        self._get_atom_orbital_info() # atoms_quantity, orbits_quantity, spinful, elements, elem_orb_map, basis_trans_index
        # get Rijk and fnna info
        self._get_Rijk_and_fnna() # R_ijk, fnna_quantity_list
        self._judge_kpts_density() # kpts grid
        self.matrix_info = self._get_pyscf_matrix_info() # atom_pairs, chunk_shapes, chunk_boundaries

    def _transfer_and_dump(self):
        self._spin_realspace_matrices_ift()
        # ------------------------ #
        self._dump_info_json()
        self._dump_poscar()
        if self.export_H:
            self._dump_H()
        if self.export_S:
            self._dump_S()
        if self.export_rho:
            self._dump_rho()
        # print(self.__dict__)

    def _reset_kpts(self, kpt=None, origin=False):
        if kpt is None:
            return
        elif origin:
            self.kpts = kpt
            self.is_kpts_reset = False
        else:
            kpt = np.asarray(kpt)
            if kpt.ndim == 1 and kpt.shape[0] == 3:
                kpt = np.asarray(kpt, dtype=int)
                self.kpts = self.cell.make_kpts(kpt)
            elif kpt.ndim == 2 and kpt.shape[1] == 3:
                self.kpts = kpt
            else:
                raise ValueError("kpt must be of shape (3,) or (nkpts, 3)")
            self.kpts /= self.unit_trans_factor['length']
            self.is_kpts_reset = True

    def _collector(self, envs:dict):
        ''' used as callback function to collect data

        Args:
            envs: a dictionary of environment variables
        '''
        self._Hk = envs['fock']
        self._Sk = envs['s1e']
        self._rhok = envs['dm']
        #TODO: what about r?

    def _recalc_mx_k(self, mf, kpts):
        ''' recalc matrices at given kpts'''
        dm = mf.make_rdm1()
        _Hk, _Sk, _rhok = None, None, None
        if self.is_periodic:
            # fix: PySCF uses Bohr unit for kpts, but input kpts is in Angstrom unit
            kpts_bohr = kpts * self.unit_trans_factor['length']
            if self.export_H:
                _Hk = mf.get_hcore(self.cell, kpts_bohr)
                _Hk += mf.get_veff(self.cell, dm, kpts=mf.kpts, kpts_band=kpts_bohr)
            if self.export_S:
                _Sk = mf.get_ovlp(self.cell, kpts_bohr)
            if self.export_rho:
                assert self.export_H and self.export_S, "To export density matrix, Hamiltonian and Overlap matrices must be exported too."
                mo_energy, mo_coeff = mf.eig(_Hk, _Sk)
                _rhok = mf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mf.get_occ(mo_energy))
        else:
            if self.export_H:
                _Hk = mf.get_hcore(self.mol)
                _Hk += mf.get_veff(self.mol, dm)
                _Hk = [_Hk]
            if self.export_S:
                _Sk = mf.get_ovlp(self.mol)
                _Sk = [_Sk]
            if self.export_rho:
                _rhok = dm
                _rhok = [_rhok]
        return _Hk, _Sk, _rhok
    
    def _judge_pbc_get_cell(self, mf):
        # pbc or not(molecular)
        is_periodic = False

        if hasattr(mf, 'cell'):
            # periodic structure
            self.cell = mf.cell
            self.mol = self.cell.to_mol()
            assert hasattr(self.cell, 'a'), "No lattice vectors found in periodic structure mf object!!!"
            if hasattr(mf, 'kpts'):
                self.kpts = np.asarray(mf.kpts)
            else:
                warnings.warn("No kpts attribute found in periodic structure mf object!!! \n " \
                "Assuming Gamma-point calculation only.")
            self.lattice:np.ndarray = self.cell.lattice_vectors() * BOHR_TO_ANGSTROM
            self.cart_coords:np.ndarray = self.cell.atom_coords(LEN_UNIT)
            is_periodic = True

        elif hasattr(mf, 'mol'):
            self.cell = None
            self.mol = mf.mol
            self.cart_coords:np.ndarray = self.mol.atom_coords(LEN_UNIT)
            self.lattice:np.ndarray = self._pseudo_lat_for_mol(self.cart_coords)
            self.cell = pbcgto.Cell(atom=self.mol.atom, a=self.lattice, 
                                    basis=self.mol.basis, spin=self.mol.spin)
            self.cell.build()
        else:
            raise ValueError("Cannot find cell or mol attribute in mf object.")
        
        self.kpts /= self.unit_trans_factor['length']
        self._org_kpts = self.kpts.copy()

        return is_periodic
    
    def _judge_kpts_density(self):
        if not self.is_periodic or self.rcut is None:
            return
        else:
            # get k points grid from mf
            kpt_grid = rebuild_kpt_grid(self.lattice, self.kpts) # tuple or np.ndarray of 3 integers
            kpt_grid = np.array(kpt_grid, dtype=int)
            is_dense, suggest_kpt_grid = self.judge_kpt_density(self.lattice, kpt_grid, self.rcut)
            if not is_dense:
                warnings.warn(f"The k-point grid {kpt_grid} may be not dense enough for the cutoff radius {self.rcut} Angstrom. \n"
                              "Please increase the k-point density to ensure accurate Fourier transform. \n"
                              "Suggested k-point grid: {}".format(suggest_kpt_grid))

    @staticmethod
    def judge_kpt_density(lattice:np.ndarray|list, kpt_grid:np.ndarray|list, rcut:float):
        '''
        Judge whether the k-point grid is dense enough for a given cutoff radius.

        Args:
            lattice: np.ndarray or list, shape (3, 3), lattice vectors in Angstrom, each row is a lattice vector
            kpt_grid: np.ndarray or list, shape (3,), number of k-points along each lattice vector
            rcut: float, cutoff radius in Angstrom
        Returns:
            is_dense: bool, whether the k-point grid is dense enough
            suggest_kpt_grid: np.ndarray, shape (3,), suggested k-point grid if not dense enough
        '''
        lattice = np.array(lattice)
        kpt_grid = np.array(kpt_grid, dtype=int)
        rec_lattice = 2 * np.pi * np.linalg.inv(lattice) # reciprocal lattice vectors
        dk_vectors = np.array([rec_lattice[i] / kpt_grid[i] for i in range(3)]) # delta k vectors
        dk_max = np.max(np.linalg.norm(dk_vectors, axis=1))
        is_dense = dk_max < (np.pi / rcut)
        suggest_kpt_grid = np.ceil(np.linalg.norm(rec_lattice, axis=1) * rcut / np.pi).astype(int)
        
        return is_dense, suggest_kpt_grid

    @staticmethod
    def _pseudo_lat_for_mol(cart_coords, vacumm=30.0):
        '''
        Generate a pseudo lattice vector for molecular system.
        Args:
            cart_coords: np.ndarray, shape (N, 3), Cartesian coordinates of atoms
            vacumm: float, vacuum size to be added in each direction (Angstrom)
        Returns:
            pseudo_lat_vec: np.ndarray, shape (3, 3), pseudo lattice vectors
        '''
        max_range = np.max(cart_coords, axis=0) - np.min(cart_coords, axis=0)
        pseudo_lat_vec = (max_range + vacumm).reshape(3,1) * np.eye(3)
        return pseudo_lat_vec
    
    def _get_atom_orbital_info(self):
        # --------------------------------- #
        self.atoms_quantity:int = self.mol.natm
        self.spinful:bool = bool(self.mol.spin > 0.01)
        self.orbit_quantity_list:list[int] = [int(0)] * self.atoms_quantity
        self.orbits_quantity:int = self.mol.nao
        # --------------------------------- #
        elements_INDEX = np.array([pyscf.gto.mole.charge(self.mol.atom_symbol(i)) for i in range(self.atoms_quantity)], dtype=int)
        self.elements:list[str] = [PERIODIC_TABLE_INDEX_TO_SYMBOL[idx] for idx in elements_INDEX]
        # --------------------------------- #
        self.atom_elem_dict:dict[str, int] = collections.Counter(self.elements)
        _ao_labels:list[tuple[int, str, str, str]] =\
              self.mol.ao_labels(fmt=False) #[(index, name, nl, m)]
        self.ao_labels:list[tuple[int, int, int, int]] = [] #(atom_index, l, n, m)

        for i, ao in enumerate(_ao_labels):
            l, m = self.lm[ao[2][1]][ao[3]]
            n = int(ao[2][0]); atom_index = int(ao[0])
            self.orbit_quantity_list[atom_index] += 1
            self.ao_labels.append((atom_index, l, n, m))
            elem = self.elements[atom_index]
        
        # Sort indices based on keys: l > n > atom_index > m
        _sorted_indices = sorted(range(len(self.ao_labels)), key=lambda k: self.ao_labels[k])

        # --------------------------------- #
        self._elem_orb_map:dict[int, list[str]] = {}
        self.basis_trans_index:dict[int, list[int]] = {}
        for atom_idx in range(len(self.elements)):
            self._elem_orb_map[atom_idx] = []
            self.basis_trans_index[atom_idx] = []
        # --------------------------------- #
        for old_idx in _sorted_indices:
            atom_index, l, n, m = self.ao_labels[old_idx]
            # MUST append old_idx, which corresponds to the index in the original PySCF matrix
            self.basis_trans_index[atom_index].append(old_idx)
            if f"{n},{l}" not in self._elem_orb_map[atom_index]:
                self._elem_orb_map[atom_index].append(f"{n},{l}")
        # --------------------------------- #
        self.elem_orb_map:dict[str, list[int]] = {}
        for atom_idx, orb_map in self._elem_orb_map.items():
            elem = self.elements[atom_idx]
            self.elem_orb_map[elem] = [int(inl.split(',')[1]) for inl in orb_map]
            self.elem_orb_map[elem] = list(sorted(list(self.elem_orb_map[elem])))
        # --------------------------------- #
        '''
        self.basis_trans_index:dict[str, np.ndarray] = {}
        for elem, orbs in self.elem_orb_map.items():
            orbital_num_list = np.array([2 * orb_l + 1 for orb_l in orbs])
            orbital_cumsum = np.concatenate((np.array([0]), np.cumsum(orbital_num_list, axis=0)), axis=0)[:-1]
            index = []
            for orb_l, orb_cum in zip(orbs, orbital_cumsum):
                index.append(BASIS_TRANS_PYSCF2WIKI(orb_l) + orb_cum)
            self.basis_trans_index[elem] = np.concatenate(index, axis=0)
        '''

    def _get_Rijk_and_fnna(self):
        # get Rijk
        # TODO: rcut for each atom
        # but in pyscf, only support one rcut for all atoms
        if self.rcut is None:
                self.cell.use_loose_rcut = True
                self.cell.build()
                self.rcut = self.cell.rcut * BOHR_TO_ANGSTROM
        if not self.is_periodic:
            self.R_ijk = np.array([[0,0,0]])
            R_lst = np.array([[0.0, 0.0, 0.0]])
        else:
            R_lst = self.cell.get_lattice_Ls(rcut=self.rcut / BOHR_TO_ANGSTROM, discard=True)
            R_lst = np.asarray(R_lst) * BOHR_TO_ANGSTROM
            self.R_ijk = np.rint(R_lst.copy() @ np.linalg.inv(self.lattice)).astype(np.int32)
        # TODO: rcut based on ovlp matrix
        self.fnna_quantity_list = [int(0)] * self.atoms_quantity
        self.fnna_indices_list = [[] for _ in range(self.atoms_quantity)]
        self.fnna_cell_indices_list = [[] for _ in range(self.atoms_quantity)]
        for i_atom in range(self.atoms_quantity):
            pos_i = self.cart_coords[i_atom]
            for j_atom in range(self.atoms_quantity):
                pos_j = self.cart_coords[j_atom]
                for idxR, nR_vec in enumerate(self.R_ijk):
                    R_vec = R_lst[idxR]
                    dist_ij = np.linalg.norm( (pos_j + R_vec) - pos_i )
                    if dist_ij < 2*self.rcut + 1e-8:
                        self.fnna_indices_list[i_atom].append(j_atom)
                        self.fnna_cell_indices_list[i_atom].append(idxR)
                        self.fnna_quantity_list[i_atom] += 1

    def _get_Rijk_and_fnna_from_ovlp(self):
        #TODO: get fnna by overlap matrix decay
        pass

    def _get_pyscf_matrix_info(self):
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

    def _get_realspace_matrices_ift(self, mx_k_lst):
        R_lst = self.R_ijk @ self.lattice
        mx_r_dict:dict[tuple[int, int, int], np.ndarray] = {}
        for i_r, R_vec in enumerate(R_lst):
            if not self.spinful or self.spin_info == 1:
                mx_r_dict[tuple(self.R_ijk[i_r])] = HamiltonianObj._ift(R_vec, self.kpts/2/np.pi, mx_k_lst).real
            else:
                mx_r_dict[tuple(self.R_ijk[i_r])] = HamiltonianObj._ift(R_vec, self.kpts/2/np.pi, mx_k_lst)
            
        return mx_r_dict

    def _spin_realspace_matrices_ift(self):
        if not self.spinful:
            # spin = 0
            self.spin_info = 0
            self._H_R_dict = self._get_realspace_matrices_ift(self._Hk)
            self._S_R_dict = self._get_realspace_matrices_ift(self._Sk)
            self.H = self._get_entries(self._H_R_dict, isspinful=0)
            self.S = self._get_entries(self._S_R_dict, isspinful=0)
            if self.export_rho:
                self._rho_R_dict = self._get_realspace_matrices_ift(self._rhok)
                self.rho = self._get_entries(self._rho_R_dict, isspinful=0)
        else:
            if self._Hk[-1].ndim == 3:
                # spin collinear
                self.spin_info = 1
                self._S_R_dict = self._get_realspace_matrices_ift(self._Sk)
                self.S = self._get_entries(self._S_R_dict, isspinful=0)
                Hk_up = [Hk[0] for Hk in self._Hk]
                Hk_dn = [Hk[1] for Hk in self._Hk]
                H_zeros = np.zeros_like(Hk_up[0]) # zero matrix, for Hk_up_dn and Hk_dn_up
                _Hk_spin = [np.block([[Hk_up[i], H_zeros], [H_zeros, Hk_dn[i]]]) for i in range(len(Hk_up))]
                self._H_R_dict = self._get_realspace_matrices_ift(_Hk_spin)
                self.H = self._get_entries(self._H_R_dict, isspinful=1)
                if self.export_rho:
                    rho_k_up = [rho_k[0] for rho_k in self._rhok]
                    rho_k_dn = [rho_k[1] for rho_k in self._rhok]
                    rho_zeros = np.zeros_like(rho_k_up[0]) # zero matrix, for rho_k_up_dn and rho_k_dn_up
                    _rho_k_spin = [np.block([[rho_k_up[i], rho_zeros], [rho_zeros, rho_k_dn[i]]]) for i in range(len(rho_k_up))]
                    self._rho_R_dict = self._get_realspace_matrices_ift(_rho_k_spin)
                    self.rho = self._get_entries(self._rho_R_dict, isspinful=1)
                    
            elif self._Hk[-1].ndim == 2 and self._Hk[-1].shape[0] == 2 * self.orbits_quantity:
                # spin non-collinear
                self.spin_info = 2
                #TODO: to be implemented
                # because it is confused that the shape of ovlp matrix is (2*nao, 2*nao)
                # need further check 
                raise NotImplementedError(
                    "Spin non-collinear case is not implemented: "
                    "overlap matrix shape (2*nao, 2*nao) needs further handling."
                )
        self.H *= HARTREE_TO_EV

    def _get_entries(self, mx_r_dict, isspinful):
        entries = np.zeros(self.matrix_info["chunk_boundaries"][-1] * (isspinful+1)**2, dtype=mx_r_dict[tuple(self.R_ijk[0])].dtype)
        shapes = self.matrix_info["chunk_shapes"]
        bounds = self.matrix_info["chunk_boundaries"]
        for i_ap, atom_pair in enumerate(self.matrix_info["atom_pairs"]):
            nR_vec = tuple(atom_pair[:3])
            atom_i_idx = atom_pair[3]; atom_j_idx = atom_pair[4]
            transform_idx1 = self.basis_trans_index[atom_i_idx]
            transform_idx2 = self.basis_trans_index[atom_j_idx]
            _mx = mx_r_dict[nR_vec]
            _mx_block = self._transform(_mx, transform_idx1, transform_idx2, isspinful)
            entries[bounds[i_ap]*(isspinful+1)**2:bounds[i_ap+1]*(isspinful+1)**2] = _mx_block.reshape(-1)
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
    
    def _dump_info_json(self):
        file_path = os.path.join(self.deeph_path, DEEPX_INFO_FILENAME)
        info_json = {
            "atoms_quantity": self.atoms_quantity,
            "orbits_quantity": self.orbits_quantity,
            "orthogonal_basis": False,
            "spinful": self.spinful,
            "fermi_energy_eV": self.fermi_energy,
            "elements_orbital_map": self.elem_orb_map,
            "basis": self.mol.basis,
        }
        with open(file_path, 'w') as fwj:
            json.dump(info_json, fwj)

    def _dump_poscar(self):
        file_path = os.path.join(self.deeph_path, DEEPX_POSCAR_FILENAME)
        
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
        file_path = os.path.join(self.deeph_path, DEEPX_HAMILTONIAN_FILENAME)
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"] * (self.spinful+1),
            "chunk_boundaries": self.matrix_info["chunk_boundaries"] * ((self.spinful+1)**2),
            "entries": self.H,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)

    def _dump_S(self, entries: np.ndarray | None = None):
        file_path = os.path.join(self.deeph_path, DEEPX_OVERLAP_FILENAME)
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"],
            "chunk_boundaries": self.matrix_info["chunk_boundaries"],
            "entries": self.S if entries is None else entries,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)
    
    def _dump_rho(self):
        file_path = os.path.join(self.deeph_path, DEEPX_DENSITY_MATRIX_FILENAME)
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
        file_path = os.path.join(self.deeph_path, DEEPX_POSITION_MATRIX_FILENAME)
        data = {
            "atom_pairs": self.matrix_info["atom_pairs"],
            "chunk_shapes": self.matrix_info["chunk_shapes"],
            "chunk_boundaries": self.matrix_info["chunk_boundaries"],
            "entries": self.r,
        }
        with h5py.File(file_path, 'w') as fwh:
            for key, value in data.items():
                fwh.create_dataset(key, data=value)

    def _transform(self, matrix, transform_index1, transform_index2, isspinful):
        if isspinful:
            a = matrix.shape[0] // 2
            b = matrix.shape[1] // 2
            num_orb_1 = len(transform_index1)
            num_orb_2 = len(transform_index2)
            matrix = matrix.reshape((2, a, 2, b)).transpose((0, 2, 1, 3)).reshape((4, a, b))
            matrix = matrix[:, transform_index1, :][:, :, transform_index2]
            matrix = matrix.reshape((2, 2, num_orb_1, num_orb_2)).transpose((0, 2, 1, 3)).reshape((2 * num_orb_1, 2 * num_orb_2))
            return matrix
        else:
            matrix = matrix[transform_index1, :][:, transform_index2]
            return matrix

