from pathlib import Path
import h5py
import os
import threadpoolctl

import numpy as np
from scipy.spatial import KDTree

from HPRO.utils.structure import Structure
from HPRO.io.aodata import AOData
from HPRO.utils.supercell import minimum_supercell
from HPRO.utils.misc import index_traverse
from HPRO.matao.mataocsr import OrbInfo, OrbInfoSuperCell

from deepx_dock.parallel import parallel_map
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
        self.gridsizefi = None
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

    def _init_grid(self, gridsize):
        """
        iosc_ongrid(ndata), fosc_ongrid(ncoords, ndata), igdptr(ndata):
            (ncoords is the number of fine grid points inside a coarse grid volume)
        iosc_ongrid[igdptr[igdco]:igdptr[igdco+1]] are indices of atomic orbitals in supercell that have 
                                                    been computed on coarse grid point igdco
        fosc_ongrid[:, igdptr[igdco]:igdptr[igdco+1]] are values of atomic orbitals. The first dimension 
                                                        corresponds to different fine grid points within 
                                                        this coarse grid volume.
        """
        gridsize = np.array(gridsize)
        aodata = self.basis_data
        structure = aodata.structure 
        
        # Determine nsubdiv (ratio of coarse grid size to fine grid size):
        # find the smallest number of fine grid points enclosed by fine grid when i is between 13 to 19
        npoints = []
        itrials = list(range(*GRIDINTG_NSUBDIV_RANGE))
        for i in itrials:
            gridsizeco = (gridsize - 1) // i + 1
            ngridco = np.prod(gridsizeco)
            npoints.append(ngridco * i**3)
        self.nsubdiv = itrials[np.argmin(npoints)]

        self.gridsizefi = gridsize
        self.gridsizeco = (gridsize - 1) // self.nsubdiv + 1
        assert np.min(self.gridsizeco) > 0

        # Preparations of the coarse integration grid
        rprimgridfi = structure.rprim / self.gridsizefi[:, None]
        rprimgridco = rprimgridfi * self.nsubdiv
        offsets = index_traverse(np.arange(0, self.nsubdiv, 1),
                                 np.arange(0, self.nsubdiv, 1),
                                 np.arange(0, self.nsubdiv, 1)) @ rprimgridfi
        ncoords = self.nsubdiv ** 3
        # dr is 1/2 of the maximum of the distances between any pair of vertices of the parallelipiped
        rvertices = index_traverse([0, 1], [0, 1], [0, 1]) @ (rprimgridco - rprimgridfi)
        dr = np.max(np.linalg.norm(rvertices[:, None, :] - rvertices[None, :, :], axis=2)) / 2
        dxyz = np.full(3, (self.nsubdiv-1) / 2) @ rprimgridfi 
        #
        self.orbinfo1 = OrbInfo(aodata)
        rmax = max(aodata.cutoffs.values())
        rmax = 2 * (rmax + dr)
        self.supercell = minimum_supercell(structure, rmax)
        self.orbinfo2 = OrbInfoSuperCell(aodata, self.supercell, 
                                         orbinfo_uc=self.orbinfo1)
        # Create KDTree for each atomic species
        trees_spc, mapiat_spc = [], []
        for ispc in range(self.supercell.nspc):
            spc = structure.atomic_species[ispc]
            is_thisspc = (self.supercell.atomic_numbers == spc)
            trees_spc.append(KDTree(self.supercell.atomic_positions_cart[is_thisspc]))
            mapiat_spc.append(np.where(is_thisspc)[0])

        ntotcogrid = np.prod(self.gridsizeco)
        igd, igdloc = 0, 0
        igdptr = [0]
        iouc_ongrid = []
        translations_ongrid = []
        fosc_ongrid = []
        for ia_co in range(self.gridsizeco[0]):
            for ib_co in range(self.gridsizeco[1]):
                for ic_co in range(self.gridsizeco[2]):
                    # find the center of coarse grid volume
                    corner = np.array([ia_co, ib_co, ic_co]) @ rprimgridco
                    center = corner + dxyz
                    # find the points inside the coarse grid volume
                    ptcoords = corner[None, :] + offsets # (ncoords, 3)

                    nphi_thispoint = 0 # number of orbitals that take nonzero values in the coarse grid volume
                    for ispc in range(self.supercell.nspc):
                        spc = structure.atomic_species[ispc]
                        for irad in range(aodata.nradial_spc[spc]):
                            phirgrid = aodata.phirgrids_spc[spc][irad]

                            # Treat AOs within r+dr of the center of the coarse grid to be "nonzero"
                            # And compute their values in the coarse grid
                            r = phirgrid.rcut + dr
                            iat_tree = np.array(trees_spc[ispc].query_ball_point(center, r))
                            if len(iat_tree) == 0: continue
                            iatsc = mapiat_spc[ispc][iat_tree] # atoms of type ispc, whose orbital irad takes nonzero values in the coarse grid volume (count = nat)

                            # Compute orbital values
                            rdiff = ptcoords[:, None, :] - self.supercell.atomic_positions_cart[None, iatsc, :] # (ncoords, nat, 3)
                            phi = phirgrid.getval3D(rdiff).reshape(ncoords, len(iatsc)*(2*phirgrid.l+1)) # (ncoords, nat*(2l+1)); norb = nat*(2l+1)

                            # Prepare array of (iat_full, irad_full, m_full) that have the same length as phi
                            # Find the indices of the "nonzero" orbitals in the supercell
                            tmp = index_traverse(iatsc, np.arange(-phirgrid.l, phirgrid.l+1, 1))
                            iat_full, m_full = tmp[:, 0], tmp[:, 1]
                            # e.g., iat_full = [643, 643, 643, 634, 634, 634, 760, 760, 760], 
                            # m_full = [-1,  0,  1, -1,  0,  1, -1,  0,  1]
                            assert len(iat_full) == phi.shape[1]
                            irad_full = np.full(phi.shape[1], irad)
                            iorbsc = self.orbinfo2.find_orbindx3(iat_full, irad_full, m_full)
                            translations, iorbuc = self.orbinfo2._iorb_sc2uc(iorbsc, return_trans_cuc=True) # (norb, 3); (norb,)

                            iouc_ongrid.append(iorbuc)
                            translations_ongrid.append(translations)
                            fosc_ongrid.append(phi)
                            nphi_thispoint += len(iorbsc)

                    igdptr.append(igdptr[-1] + nphi_thispoint)
                    igdloc += 1

                    igd += 1

        self.iouc_ongrid = np.concatenate(iouc_ongrid) # (M,)
        self.translations_ongrid = np.concatenate(translations_ongrid) # (M, 3)
        self.fosc_ongrid = np.concatenate(fosc_ongrid, axis=1).T # (M, ncoords)
        self.igdptr = np.array(igdptr) # (ngd_co+1,)

    def to_real_space(self, ik=None, ib=None, gridsize=None, return_periodic=True):
        """
        Compute the periodic part of Bloch wavefunction u_{nk}(r) = e^{-ikr} * psi_{nk}(r).
        
        Args:
            ik: int, k point index
            ib: int, band index
            gridsize: np.ndarray (3,), real space grid size
            return_periodic: bool, whether to return the full Bloch wavefunction 
                or just its periodic part u(r) = e^{-ikr} psi(r)
        
        Returns:
            u_grid: np.ndarray (nk, nb, nspinor, nx, ny, nz), complex, Bloch wavefunction values in real space
        """
        if gridsize is not None and not np.array_equal(gridsize, self.gridsizefi):
            self._init_grid(gridsize)
        assert self.gridsizefi is not None

        aodata = self.basis_data
        structure = aodata.structure 

        nk_all = self.wfnao.shape[0]
        nb_all = self.wfnao.shape[1]
        if ik is None:
            ik = np.arange(nk_all)
        else:
            ik = np.array(ik)
        if ib is None:
            ib = np.arange(nb_all)
        else:
            ib = np.array(ib)
        nk = len(ik)
        nb = len(ib)
            
        # Fractional coordinates of the selected k-points.
        # Shape: (nk, 3)
        kpts = self.kpts[ik] 
        
        # Extract and transpose atomic orbital (AO) coefficients to prioritize the orbital dimension.
        # Transposed c_vec shape: (norb, nk, nb, nspinor)
        nspinor = 2 if self.spinful else 1
        num_orbitals_tot = self.wfnao.shape[-1] // nspinor
        wfnao = self.wfnao[np.array(ik)[:, np.newaxis], np.array(ib), :]
        c_vec = np.zeros((nk, nb, nspinor, num_orbitals_tot), dtype=self.wfnao.dtype)

        site_norbits = np.zeros(structure.natom, dtype=int)
        for iatm in range(structure.natom):
            atm_nbr = structure.atomic_numbers[iatm]
            site_norbits[iatm] = aodata.norbfull_spc[atm_nbr]
            
        if nspinor == 2:
            current_orbitals_idx = 0
            current_idx = 0
            for n_i in site_norbits:
                c_vec[:,:,0, current_orbitals_idx : current_orbitals_idx + n_i] = wfnao[:,:,current_idx : current_idx + n_i]
                c_vec[:,:,1, current_orbitals_idx : current_orbitals_idx + n_i] = wfnao[:,:,current_idx + n_i : current_idx + 2 * n_i]
                current_orbitals_idx += n_i
                current_idx += 2 * n_i
        elif nspinor == 1:
            c_vec = wfnao.reshape((nk, nb, nspinor, -1), order='C')  # (nk, nb, nspinor, norb)

        c_vec = c_vec.transpose(3, 0, 1, 2)
        
        # Initialize the global real-space wavefunction tensor.
        # Shape: (N_x, N_y, N_z, nk, nb, nspinor)
        psi_global = np.zeros(
            (self.gridsizefi[0], self.gridsizefi[1], self.gridsizefi[2], nk, nb, nspinor),
            dtype=np.complex128
        )
        
        # ncoords defines the total number of fine grid points contained within a standard coarse grid.
        # Constraint: ncoords = nsubdiv^3
        ncoords = self.nsubdiv ** 3
        
        # Total number of coarse grids generated in the system.
        ngd_co = len(self.igdptr) - 1
        
        for igd_co in range(ngd_co):
            # Unravel the 1D coarse grid index into 3D structural block indices (ia_co, ib_co, ic_co).
            ia_co = igd_co // (self.gridsizeco[1] * self.gridsizeco[2])
            rem = igd_co % (self.gridsizeco[1] * self.gridsizeco[2])
            ib_co = rem // self.gridsizeco[2]
            ic_co = rem % self.gridsizeco[2]
            
            # Calculate the global fine grid coordinate boundaries for the current coarse grid block.
            # The min() function dynamically handles non-divisible edge truncations.
            a_start = ia_co * self.nsubdiv
            a_end = min(a_start + self.nsubdiv, self.gridsizefi[0])
            b_start = ib_co * self.nsubdiv
            b_end = min(b_start + self.nsubdiv, self.gridsizefi[1])
            c_start = ic_co * self.nsubdiv
            c_end = min(c_start + self.nsubdiv, self.gridsizefi[2])
            
            # Compute the valid structural lengths along each axis, discarding the out-of-bounds region.
            la = a_end - a_start
            lb = b_end - b_start
            lc = c_end - c_start
            
            # Retrieve pointers for the pre-calculated orbital subsets mapped to this specific grid block.
            start_idx = self.igdptr[igd_co]
            end_idx = self.igdptr[igd_co+1]
            if start_idx == end_idx: 
                # Skip iteration if no atomic orbitals take nonzero values in this domain.
                continue

            # iouc_thisgd: Indices of orbitals present in the current block. Shape: (M_thisgd,)
            iouc_thisgd = self.iouc_ongrid[start_idx:end_idx]
            # fosc_thisgd: Pre-calculated real-space AO values $f_{m,r}$. Shape: (M_thisgd, ncoords)
            fosc_thisgd = self.fosc_ongrid[start_idx:end_idx, :]
            # trans_thisgd: Supercell translation vectors $\mathbf{T}_m$, in fractional coordinates. Shape: (M_thisgd, 3)
            trans_thisgd = self.translations_ongrid[start_idx:end_idx, :]

            # Compute the Bloch phase factor: $e^{i 2\pi \mathbf{T}_m \cdot \mathbf{k}}$.
            # Broadcasting matrix multiplication between translations (M_thisgd, 3) and kpts.T (3, nk).
            # Shape: (M_thisgd, nk)
            phase_thisgd = np.exp(1j * 2 * np.pi * trans_thisgd @ kpts.T) 
            
            # Select the sub-matrix of coefficients $c_{m,k,b,s}$ corresponding to the active orbitals.
            # Shape: (M_thisgd, nk, nb, nspinor)
            c_thisgd = c_vec[iouc_thisgd] 
            
            # Apply the translation phase factor to the local coefficients via broadcasting.
            # $\tilde{c} = c \cdot e^{i 2\pi \mathbf{T} \cdot \mathbf{k}}$
            # New axes align the phase array with the full tensor dimensions.
            # Shape: (M_thisgd, nk, nb, nspinor)
            c_tilde = c_thisgd * phase_thisgd[:, :, np.newaxis, np.newaxis]
            
            # Flatten all trailing physical dimensions to construct a 2D matrix for efficient computation.
            # Shape: (M_thisgd, nk * nb * nspinor)
            c_tilde_flat = c_tilde.reshape(c_tilde.shape[0], -1) 
            
            # Perform core tensor contraction via highly optimized BLAS matrix multiplication.
            # $\Phi = F^T \tilde{C}$
            # Shape: (ncoords, nk * nb * nspinor)
            psi_r_flat = fosc_thisgd.T @ c_tilde_flat 
            
            # Unflatten the matrix product to restore the physical 3D structural dimensions.
            # Shape: (nsubdiv, nsubdiv, nsubdiv, nk, nb, nspinor)
            psi_r_thisgd = psi_r_flat.reshape(self.nsubdiv, self.nsubdiv, self.nsubdiv, nk, nb, nspinor)
            
            # Map the valid inner domain of the local grid block back to the global spatial tensor.
            # Slicing [:la, :lb, :lc] exactly matches the dynamically truncated active lengths.
            psi_global[a_start:a_end, b_start:b_end, c_start:c_end] = psi_r_thisgd[:la, :lb, :lc]

        if return_periodic:
            # Generate 1D fractional coordinates for each axis
            grid_a = np.arange(self.gridsizefi[0]) / self.gridsizefi[0]
            grid_b = np.arange(self.gridsizefi[1]) / self.gridsizefi[1]
            grid_c = np.arange(self.gridsizefi[2]) / self.gridsizefi[2]
            
            # Construct a 3D grid of fractional coordinates. Shape: (N_x, N_y, N_z, 3)
            r_frac = np.stack(np.meshgrid(grid_a, grid_b, grid_c, indexing='ij'), axis=-1)
            
            # Calculate the inverse Bloch phase: exp(-i 2\pi k \cdot r_frac)
            # Use tensordot to perform dot product over the last axis (spatial dimension 3)
            # Resulting phase_inv shape: (N_x, N_y, N_z, nk)
            phase_inv = np.exp(-1j * 2 * np.pi * np.tensordot(r_frac, kpts, axes=([-1], [-1])))
            
            # Broadcast the phase factor across the band and spinor dimensions
            psi_global *= phase_inv[:, :, :, :, np.newaxis, np.newaxis]
        
        # Normalize wavefunctions so that the norm is equal to sqrt(prod(self.gridsizefi))
        psi_global *= np.sqrt(structure.cell_volume)
            
        return psi_global.transpose((3, 4, 5, 0, 1, 2)) # (nk, nb, nspinor, nx, ny, nz)

    def to_dm(self, representation='k'):
        pass