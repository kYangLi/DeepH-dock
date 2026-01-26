from pathlib import Path
import h5py
import os
import threadpoolctl

import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel, delayed

from HPRO.utils.structure import Structure
from HPRO.io.aodata import AOData

from deepx_dock.misc import load_json_file, load_poscar_file
from deepx_dock.CONSTANT import BOHR_TO_ANGSTROM
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME

GRIDINTG_NSUBDIV_RANGE = (13, 20)

class AOWfnObj:
    """
    Atomic orbital wave function object.

    Parameters
    ----------
    info_dir_path : str | Path
        Directory path to the info file.
    basis_dir_path : str | Path
        Directory path to the basis file.
    aocode : str
        Atomic orbital code. Currently only supports "siesta".
    kpts : np.ndarray
        k-points in reduced coordinates (fractional), shape (Nk, 3).
    wfnao : np.ndarray
        Atomic orbital wave function coefficients, shape (Nk, Nband, Norb).
    el : np.ndarray
        Eigenvalues (energy levels), shape (Nband,).
    spinful : bool, optional
        If True, spinful system. Default is False.
    efermi : float, optional
        Fermi energy. Default is None.
    kgrid : tuple of int, optional
        k-point grid. Default is None. Shape (3,).
    """
    def __init__(self, info_dir_path, basis_dir_path, aocode):
        self.aocode = aocode
        self._get_necessary_data_path(info_dir_path, basis_dir_path)
        self.parse_data()
        
    def load(self, kpts, wfnao, el, spinful=False, efermi=None, kgrid=None):
        self.kpts = kpts
        self.wfnao = wfnao
        self.el = el
        self.spinful = spinful
        self.efermi = efermi
        self.kgrid = kgrid
        if kgrid is not None:
            assert kpts.shape[0] == np.prod(kgrid)

    def parse_data(self):
        self._parse_poscar()
        self._parse_basis()

    def to_h5(self, h5_path):
        assert self.kgrid is not None
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('kpts', data=self.kpts)
            f.create_dataset('kgrid', data=self.kgrid)
            f.create_dataset('wfnao', data=self.wfnao)
            f.create_dataset('el', data=self.el)
            f.create_dataset('efermi', data=self.efermi)

    def _get_necessary_data_path(self,
        info_dir_path: str | Path, basis_dir_path: str | Path
    ):
        info_dir_path = Path(info_dir_path)
        self.info_dir_path = info_dir_path
        self.poscar_path = info_dir_path / DEEPX_POSCAR_FILENAME
        self.info_json_path = info_dir_path / DEEPX_INFO_FILENAME

        self.basis_dir_path = Path(basis_dir_path)

    def _parse_poscar(self):
        raw_poscar = self._read_poscar(self.poscar_path)
        #
        self.lattice = raw_poscar["lattice"]
        self.elements = raw_poscar["elements"]
        self.atomic_numbers = raw_poscar["atomic_numbers"]
        self.frac_coords = raw_poscar["frac_coords"]
        self.reciprocal_lattice = self.get_reciprocal_lattice(self.lattice)

    def _parse_basis(self):
        structure = Structure(rprim=self.lattice / BOHR_TO_ANGSTROM, atomic_numbers=self.atomic_numbers, atomic_positions=self.frac_coords)
        self.basis_data = AOData(structure, basis_path_root=self.basis_dir_path, aocode=self.aocode)

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

    @staticmethod
    def get_reciprocal_lattice(lattice):
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

    def to_real_space(self, ik, ib, gridsize, process_num_per_grid=1):
        """
        Compute the periodic part of Bloch wavefunction u_{nk}(r) = e^{-ikr} * psi_{nk}(r).
        
        Args:
            ik: int, k point index
            ib: int, band index
            gridsize: np.ndarray (3,), real space grid size
            process_num_per_grid: int, number of processes for a single real space grid
        
        Returns:
            u_grid: np.ndarray (nx, ny, nz), complex, Bloch wavefunction values in real space
        """
        
        c_vec = self.wfnao[ik, ib, :]  # (Norb,)
        aodata = self.basis_data
        structure = aodata.structure
        kvec_frac = self.kpts[ik] 
        inv_lattice_bohr = self.reciprocal_lattice * BOHR_TO_ANGSTROM # self.reciprocal_lattice is in angstrom^{-1}.
        #
        atom_orb_ranges = []
        idx_cursor = 0
        for iat in range(structure.nat):
            spc = structure.atomic_species[iat]
            n_orb_atom = 0
            for irad in range(aodata.nradial_spc[spc]):
                l = aodata.phirgrids_spc[spc][irad].l
                n_orb_atom += (2 * l + 1)
            atom_orb_ranges.append((idx_cursor, idx_cursor + n_orb_atom))
            idx_cursor += n_orb_atom
        assert idx_cursor == c_vec.shape[0]
        #
        gridsize = np.array(gridsize)
        npoints = []
        itrials = range(*GRIDINTG_NSUBDIV_RANGE)
        for i in itrials:
            gridsizeco = (gridsize - 1) // i + 1
            npoints.append(np.prod(gridsizeco) * i**3)
        nsubdiv = itrials[np.argmin(npoints)]
        
        gridsizeco = (gridsize - 1) // nsubdiv + 1
        rprim = structure.rprim # in bohr
        rprimgridfi = rprim / gridsize[:, None]
        rprimgridco = rprimgridfi * nsubdiv
        
        grids_1d = np.arange(nsubdiv)
        mesh_idx = np.array(np.meshgrid(grids_1d, grids_1d, grids_1d, indexing='ij')).reshape(3, -1).T
        offsets = mesh_idx @ rprimgridfi

        # Pre-compute phase factors for offsets
        # Convert offsets to fractional coordinates: r_frac = r_cart @ inv_lattice.T
        offsets_frac = offsets @ inv_lattice_bohr.T # (nsubdiv**3, 3)
        # Calculate exp(-i * k * r_offset) for all points in a coarse block
        offset_phases = np.exp(-1j * (offsets_frac @ kvec_frac)) # (nsubdiv**3,)

        rvertices = np.array(np.meshgrid([0,1],[0,1],[0,1])).reshape(3,-1).T @ (rprimgridco - rprimgridfi)
        dr = np.max(np.linalg.norm(rvertices[:, None, :] - rvertices[None, :, :], axis=2)) / 2
        dxyz_center = np.sum(rprimgridfi * (nsubdiv - 1) / 2, axis=0)
        
        rmax_ao = max(aodata.cutoffs.values())
        search_radius = rmax_ao + dr

        sc_indices = [] # list of (iat_uc, R_idx)
        sc_positions = []
        
        img_range = range(-1, 2) 
        R_shifts = np.array(np.meshgrid(img_range, img_range, img_range)).reshape(3, -1).T
        
        for R_idx in R_shifts:
            R_vec = R_idx @ rprim
            curr_pos = structure.atomic_positions_cart + R_vec
            sc_positions.append(curr_pos)
            for iat in range(structure.nat):
                sc_indices.append((iat, R_idx))
                
        sc_positions = np.concatenate(sc_positions, axis=0) # (Nat * Nimage, 3)
        
        trees_spc = {}     # {spc_id: KDTree}
        map_sc_idx_spc = {} # {spc_id: array of indices in sc_indices list}
        
        sc_species = np.tile(structure.atomic_species, len(R_shifts))
        
        for spc in np.unique(structure.atomic_species):
            mask = (sc_species == spc)
            trees_spc[spc] = KDTree(sc_positions[mask])
            map_sc_idx_spc[spc] = np.where(mask)[0]

        coarse_indices = np.array(list(np.ndindex(tuple(gridsizeco))))
        
        ctx = {
            'gridsizefi': gridsize, 'nsubdiv': nsubdiv,
            'rprimgridco': rprimgridco, 'dxyz_center': dxyz_center, 'offsets': offsets,
            'offset_phases': offset_phases,
            'search_radius': search_radius,
            'trees_spc': trees_spc, 'map_sc_idx_spc': map_sc_idx_spc,
            'sc_indices': sc_indices, 'sc_positions': sc_positions,
            'aodata': aodata, 'atom_orb_ranges': atom_orb_ranges,
            'c_vec': c_vec, 
            'kvec_frac': kvec_frac,
            'inv_lattice': inv_lattice_bohr,
            'structure_species': structure.atomic_species
        }

        n_batches = min(len(coarse_indices), max(1, os.cpu_count() * 4))
        batches = np.array_split(coarse_indices, n_batches)
        
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            results = Parallel(n_jobs=process_num_per_grid, prefer="processes")(
                delayed(self._calc_u_batch)(batch_idxs, ctx) for batch_idxs in batches
            )

        u_grid_flat = np.zeros(np.prod(gridsize), dtype=np.complex128)
        
        # results: list of (linear_indices, values)
        for batch_res in results:
            if batch_res is None: continue
            lin_idxs, vals = batch_res
            u_grid_flat[lin_idxs] = vals
            
        return u_grid_flat.reshape(gridsize)

    def to_dm(self, representation='k'):
        pass

    @staticmethod
    def _calc_u_batch(coarse_idxs, ctx):
        """ Compute u(r) on fine grid points corresponding to a batch of coarse grid points """
        if len(coarse_idxs) == 0: return None
        
        # unpack context
        gridsizefi = ctx['gridsizefi']
        nsubdiv = ctx['nsubdiv']
        rprimgridco = ctx['rprimgridco']
        dxyz = ctx['dxyz_center']
        offsets = ctx['offsets']
        offset_phases = ctx['offset_phases'] # (nsubdiv**3,)
        search_radius = ctx['search_radius']
        trees_spc = ctx['trees_spc']
        map_sc_idx_spc = ctx['map_sc_idx_spc']
        sc_indices = ctx['sc_indices'] # list of (iat_uc, R_idx)
        sc_positions = ctx['sc_positions']
        aodata = ctx['aodata']
        atom_orb_ranges = ctx['atom_orb_ranges']
        c_vec = ctx['c_vec']         # (Norb,)
        kvec_frac = ctx['kvec_frac']
        inv_lattice = ctx['inv_lattice']
        
        out_indices = []
        out_values = []
        
        # Vectorized KDTree Query
        # 1. Compute centers for all coarse blocks in this batch
        coarse_idxs_arr = np.array(coarse_idxs) # (N_batch, 3)
        corners = coarse_idxs_arr @ rprimgridco # (N_batch, 3)
        centers = corners + dxyz                # (N_batch, 3)

        # 2. Query all species at once
        # batch_neighbors_dict: {spc: list_of_N_batch_results}
        batch_neighbors_dict = {}
        for spc, tree in trees_spc.items():
            # query_ball_point with vector input returns an object array of lists
            batch_neighbors_dict[spc] = tree.query_ball_point(centers, search_radius)

        # 3. Iterate through each block in the batch
        for i_batch, (ia, ib, ic) in enumerate(coarse_idxs):
            corner = corners[i_batch] # (3,)
            center = centers[i_batch] # (3,) (Already computed, just for reference if needed)
            
            # (ncoords, 3)
            ptcoords = corner[None, :] + offsets
            ncoords = len(ptcoords)
            
            slicea = slice(ia*nsubdiv, min((ia+1)*nsubdiv, gridsizefi[0]))
            sliceb = slice(ib*nsubdiv, min((ib+1)*nsubdiv, gridsizefi[1]))
            slicec = slice(ic*nsubdiv, min((ic+1)*nsubdiv, gridsizefi[2]))
            
            # generate local to global mapping
            la, lb, lc = slicea.stop-slicea.start, sliceb.stop-sliceb.start, slicec.stop-slicec.start
            if la*lb*lc == 0: continue
            
            valid_mask = np.zeros((nsubdiv, nsubdiv, nsubdiv), dtype=bool)
            valid_mask[:la, :lb, :lc] = True
            valid_mask_flat = valid_mask.flatten()
            local_grids = np.array(np.meshgrid(
                np.arange(slicea.start, slicea.stop),
                np.arange(sliceb.start, sliceb.stop),
                np.arange(slicec.start, slicec.stop), indexing='ij'
            )).reshape(3, -1).T
            
            # convert 3D indices to 1D linear indices
            # idx = x * Ny * Nz + y * Nz + z
            lin_idxs = (local_grids[:,0] * gridsizefi[1] * gridsizefi[2] + 
                        local_grids[:,1] * gridsizefi[2] + 
                        local_grids[:,2])
            
            ptcoords_valid = ptcoords[valid_mask_flat]
            n_valid = len(ptcoords_valid)
            if n_valid == 0: continue

            psi_r = np.zeros(n_valid, dtype=np.complex128)
            
            # traverse atom species using pre-queried results
            for spc in trees_spc.keys():
                found_indices = batch_neighbors_dict[spc][i_batch]
                if not found_indices: continue
                
                real_sc_indices_ptr = map_sc_idx_spc[spc][found_indices]
                
                # traverse each atom
                for ptr in real_sc_indices_ptr:
                    iat_uc, R_idx = sc_indices[ptr] # here R_idx is integer index
                    atom_pos = sc_positions[ptr]    # here it is still Cartesian coordinates, used for AO evaluation
                    
                    # calculate Bloch phase factor e^{ikR} * c_mu
                    # use fractional coordinates: k_frac * R_int * 2pi
                    phase_factor = np.exp(2j * np.pi * np.dot(kvec_frac, R_idx))
                    
                    # get the orbital coefficients slice
                    start, end = atom_orb_ranges[iat_uc]
                    coeffs_atom = c_vec[start:end] * phase_factor # (N_orb_atom,)
                    
                    # calculate AO value: phi(r - R_atom)
                    # distance vector (n_valid, 3) (keep Cartesian coordinates for physical orbital evaluation)
                    diff = ptcoords_valid - atom_pos[None, :]
                    
                    # accumulate different radial parts
                    orb_cursor = 0
                    for irad in range(aodata.nradial_spc[spc]):
                        phirgrid = aodata.phirgrids_spc[spc][irad]
                        l = phirgrid.l
                        n_m = 2*l + 1
                        
                        # (n_valid, n_m)
                        vals = phirgrid.getval3D(diff).reshape(n_valid, n_m)
                        
                        # corresponding coefficients (n_m,)
                        c_shell = coeffs_atom[orb_cursor : orb_cursor+n_m]
                        
                        # accumulate to psi_r: psi += val * c
                        psi_r += vals @ c_shell
                        
                        orb_cursor += n_m

            # u(r) = e^{-ikr} * psi(r)
            # Use pre-computed offset phases
            # 1. Compute phase for the coarse grid corner: exp(-i * k * R_corner)
            corner_frac = corner @ inv_lattice.T # (3,)
            corner_phase = np.exp(-1j * (corner_frac @ kvec_frac)) # scalar
            
            # 2. Combine with offset phases: phase = corner_phase * offset_phase
            # Extract valid phases corresponding to current block size
            offset_phases_valid = offset_phases[valid_mask_flat] # (n_valid,)
            phase_bloch_inv = corner_phase * offset_phases_valid
            
            u_r = psi_r * phase_bloch_inv
            
            out_indices.append(lin_idxs)
            out_values.append(u_r)

        if not out_indices: return None
        return np.concatenate(out_indices), np.concatenate(out_values)

"""
Aside: WfnAO object in the BSE program Pykernel (wfnao has shape (nk, nb, norb), and el has shape (nk, nb)):

class WfnAO(PyKernelBase):
    _PRINT_ITEMS = ("fname", "nk", "nb")

    def __init__(self, poscar_fname, basis_path_root, wfnao_file, nv=None, nc=None):
        with open(poscar_fname) as f:
            structure = from_poscar(f)
        aodata = AOData(structure, basis_path_root=basis_path_root, aocode='siesta')

        with h5py.File(wfnao_file, "r") as h5file:
            kpts = h5file['kpts'][:].T
            kgrid = h5file['kgrid'][:]
            wfnao = h5file['wfnao'][:]
            el = h5file['el'][:]
            efermi = h5file['efermi'][()]

        self.orb_info_dict_spc, self.num_orbitals_tot = create_orbital_list(structure, aodata)
        nspin = wfnao.shape[2] // self.num_orbitals_tot
        self.nspin = nspin
        self.dtype = {1: np.float64, 2: np.complex128}[self.nspin]
        self.wfnao = wfnao.reshape((wfnao.shape[0], wfnao.shape[1], nspin, -1), order='C')  # (nk, nb, nspin, norb)

        self.structure = structure
        self.aodata = aodata
        self.kpts = kpts
        self.kgrid = kgrid
        self.nk = kpts.shape[1]  # Number of k-points
        self.el = el
        self.efermi = efermi

        if nv is not None and nc is not None:
            idx_v = topN_less_than(self.el, self.efermi, nv)
            idx_c = lastN_greater_than(self.el, self.efermi, nc)
            band_indices = np.concatenate((idx_v, idx_c), axis=1)  # (nk, nv+nc)
            self.wfnao = self.wfnao[np.arange(self.nk)[:, None], band_indices, :, :]
            self.nb = band_indices.shape[1]
        else:
            self.nb = self.wfnao.shape[1]
            
        self.wfnao_grouped, self.positions_red_grouped = group_wfnao_all(self.wfnao, structure.atomic_species, aodata.nradial_spc, self.orb_info_dict_spc)
        self.rgrids_info_spc = {'rcut': {}, 'l': {}}
        self.nradial_spc = {}
        for spc in structure.atomic_species:
            phirgrids = aodata.phirgrids_spc[spc]
            self.nradial_spc[spc] = aodata.nradial_spc[spc]
            rcut_spc, l_spc = [], []
            for _, phirgrid in enumerate(phirgrids):
                rcut_spc.append(phirgrid.rgd.rfunc[-1])
                l_spc.append(np.zeros((phirgrid.l,), dtype=np.int32))

            self.rgrids_info_spc['rcut'][spc] = rcut_spc
            self.rgrids_info_spc['l'][spc] = l_spc

def group_wfnao_all(wfnao, atomic_species, nradial_spc, orb_info_dict_spc):
    wfnao_grouped = {}
    positions_red_grouped = {}

    for spc in atomic_species:
        wfnao_spc = []
        positions_red_spc = orb_info_dict_spc['positions_red'][spc]

        positions_red_grouped[spc] = positions_red_spc

        for irad in range(nradial_spc[spc]):
            idx_thisorbit = orb_info_dict_spc['orbital_idx'][spc][irad] # (natom_spc, 2l+1)
            wfnao_thisorbit = wfnao[:, :, :, idx_thisorbit]  # (nk, nb, nspin, natom_spc, 2l+1)
            wfnao_thisorbit = np.transpose(wfnao_thisorbit, (0, 3, 4, 1, 2))  # (nk, natom_spc, 2l+1, nband, nspin)

            # (nk, n_device, natom_padded/gpus_per_process, 2l+1, nband, nspin)
            wfnao_spc.append(wfnao_thisorbit)
        wfnao_grouped[spc] = wfnao_spc

    return wfnao_grouped, positions_red_grouped

def topN_less_than(arr: np.ndarray, val: float, N: int) -> np.ndarray:
    arr_masked = np.where(arr < val, arr, -np.inf)

    idx = np.argpartition(arr_masked, -N, axis=1)[:, -N:]  # (M,N)

    rows = np.arange(arr.shape[0])[:, None]
    vals = arr_masked[rows, idx]  # (M,N)
    order = np.argsort(vals, axis=1)

    idx_sorted = np.take_along_axis(idx, order, axis=1)
    return idx_sorted

def lastN_greater_than(arr: np.ndarray, val: float, N: int) -> np.ndarray:
    B = np.where(arr > val, arr, np.inf)

    part_idx = np.argpartition(B, N-1, axis=1)[:, :N]

    rows = np.arange(arr.shape[0])[:, None]
    vals = B[rows, part_idx]
    order = np.argsort(vals, axis=1)

    idx_sorted = np.take_along_axis(part_idx, order, axis=1)
    return idx_sorted
"""