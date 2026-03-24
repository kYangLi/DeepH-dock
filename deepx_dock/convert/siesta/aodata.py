import numpy as np

from HPRO.utils.orbutils import GridFunc, LinearRGD
from HPRO.utils.structure import Structure
from HPRO.utils.misc import atom_number2name
from HPRO.utils.math import get_dmat_coeffs
from HPRO.io.aodata import AOData

class GridFunc_siesta(GridFunc):
    '''
    An HPRO GridFunc subclass for SIESTA-specific data.
    '''
    def __init__(self, rgd, func, l=0, n=1, z=1, is_polarized=False, rcut=None):
        super().__init__(rgd, func, l=l, rcut=rcut)
        self.n = n
        self.z = z
        self.is_polarized = is_polarized

class AOData_siesta(AOData):
    '''
    An HPRO AOData subclass for SIESTA-specific data.
    '''
    def __init__(self, structure: Structure, cutoffs=None, basis_path_root='./', mode='orb'):
        self.structure = structure
        self.aocode = 'siesta' if mode == 'orb' else 'siesta-projR'
        self.spinful = None
        self.magnetic = None
        assert not self.magnetic, "Magnetic calculation is not supported yet."
        
        self.ls_spc = {}
        self.ns_spc = {}
        self.zs_spc = {}
        self.ps_spc = {}
        self.phirgrids_spc = {}
        self.nradial_spc = {}
        self.Dij_spc = {}
        self.Qij_spc = {}
        self.charge_spc = {}
        
        spc_numbers = structure.atomic_species
        spc_names = atom_number2name(spc_numbers)
        for spc_nu, spc_na in zip(spc_numbers, spc_names):
            Dij, Qij = None, None
            valence_charge = None

            if mode == 'orb':
                nradial, phirgrids, valence_charge = parse_siesta_ion(f'{basis_path_root}/{spc_na}.ion', mode='orb')
            elif mode == 'projR':
                nradial, phirgrids, Dij = parse_siesta_ion(f'{basis_path_root}/{spc_na}.ion', mode='projR')
            else:
                raise ValueError(f'Unsupported mode: {mode}')

            orbitals_argsort = list(range(len(phirgrids)))
            self.phirgrids_spc[spc_nu] = [phirgrids[i] for i in orbitals_argsort]
            self.ls_spc[spc_nu] = [phirgrids[i].l for i in orbitals_argsort]
            self.ns_spc[spc_nu] = [phirgrids[i].n for i in orbitals_argsort]
            self.zs_spc[spc_nu] = [phirgrids[i].z for i in orbitals_argsort]
            self.ps_spc[spc_nu] = [phirgrids[i].is_polarized for i in orbitals_argsort]
            self.nradial_spc[spc_nu] = nradial
            if valence_charge is not None:
                self.charge_spc[spc_nu] = valence_charge
            if Dij is not None:
                self.Dij_spc[spc_nu] = Dij
        
        orbslices_spc = {}
        norbfull_spc = {}
        for spc, orbital_types in self.ls_spc.items():
            orbital_slices = [0]
            for l in orbital_types:
                orbital_slices.append(orbital_slices[-1] + 2*l+1)
            orbslices_spc[spc] = orbital_slices
            norbfull_spc[spc] = orbital_slices[-1]
        self.orbslices_spc = orbslices_spc
        self.norbfull_spc = norbfull_spc

        if cutoffs is None:
            cutoffs = {}
            cutoffs_orb = {}
            for spc_nu, spc_na in zip(spc_numbers, spc_names):
                cutoffs_orb[spc_na] = [phirgrid.rcut for phirgrid in self.phirgrids_spc[spc_nu]]
                cutoffs[spc_na] = max(cutoffs_orb[spc_na])
            self.cutoffs_orb = cutoffs_orb

        self.cutoffs = cutoffs
        self.phiQlist_spc = None
        self.phiQEcut = None

def parse_siesta_ion(filename, mode='orb'):
    phirgrids_basis = []
    phirgrids_proj = []
    l_list_proj = []
    j_list_proj = []
    Dij_list = []
    norb_basis = 0
    norb_proj = 0
    rel = None

    ionfile = open(filename, 'r')
    line = ionfile.readline()
    while line:
        if line.find('rel') > 0 and rel is None:
            rel = True
        elif line.find('nrl') > 0 and rel is None:
            rel = False

        if line.find('# Valence charge') > 0:
            sp = line.split()
            valence_charge = float(sp[0])

        # basis orbitals
        if line.find('#orbital l, n, z, is_polarized, population') > 0:
            sp = line.split()
            l = int(sp[0])
            n = int(sp[1])
            z = int(sp[2])
            is_polarized = bool(int(sp[3]))
            population = float(sp[4])
            norb_basis += 1
            
            line_sp = ionfile.readline().split()
            assert line_sp[0] == '500'
            rcut = float(line_sp[2])
            
            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            # found this from sisl/io/siesta/siesta_nc.py: ncSileSiesta.read_basis(self): 
            # sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            assert np.abs(rgd.rend - rcut) < 1e-6
            phirgrids_basis.append(GridFunc_siesta(rgd, phirgrid[1], l=l, n=n, z=z, is_polarized=is_polarized, rcut=rcut))

        # projector orbitals
        projector_header = None
        if rel:
            projector_header = '#kb l, j, n (sequence number), Reference energy'
        elif not rel:
            projector_header = '#kb l, n (sequence number), Reference energy'
        if projector_header is not None and line.find(projector_header) > 0:
            sp = line.split()
            l = int(sp[0])
            if rel:
                j = int(sp[1])
                n = int(sp[2])
                ekb = float(sp[3])
                j_list_proj.append(j)
            else:
                n = int(sp[1])
                ekb = float(sp[2])

            norb_proj += 1
            Dij_list.append(ekb / 2.) # Ry to Har

            line_sp = ionfile.readline().split()
            assert line_sp[0] == '500'
            rcut = float(line_sp[2])

            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            assert np.abs(rgd.rend - rcut) < 1e-6
            l_list_proj.append(l)
            phirgrids_proj.append(GridFunc_siesta(rgd, phirgrid[1], l=l, n=n, z=0, is_polarized=False, rcut=rcut))

        line = ionfile.readline()
    ionfile.close()

    if mode == 'orb':
        return norb_basis, phirgrids_basis, valence_charge
    elif mode == 'projR':
        Dij = np.diag(np.array(Dij_list))
        orbital_slices = np.cumsum(2 * np.array(l_list_proj) + 1)
        orbital_slices = np.insert(orbital_slices, 0, 0)
        norbfull = orbital_slices[-1]
        if rel:
            Dij_full = np.zeros((2, 2, norbfull, norbfull), dtype=np.complex128)
        else:
            Dij_full = np.zeros((norbfull, norbfull), dtype=np.float64)
        for iorb in range(len(l_list_proj)):
            l1 = l_list_proj[iorb]
            if rel: j1 = j_list_proj[iorb]
            for jorb in range(len(l_list_proj)):
                l2 = l_list_proj[jorb]
                if rel: j2 = j_list_proj[jorb]
                if (l1 == l2) and ((not rel) or (j1 == j2)):
                    if rel:
                        Dij_full[:, :, orbital_slices[iorb]:orbital_slices[iorb+1], 
                                    orbital_slices[jorb]:orbital_slices[jorb+1]] = \
                            Dij[iorb, jorb] * get_dmat_coeffs(l1, j1)
                    else:
                        np.fill_diagonal(Dij_full[orbital_slices[iorb]:orbital_slices[iorb+1],
                                                orbital_slices[jorb]:orbital_slices[jorb+1]], Dij[iorb, jorb])
                else:
                    assert np.abs(Dij[iorb, jorb]) < 1e-8

        return norb_proj, phirgrids_proj, Dij_full
    else:
        raise ValueError(f'Unsupported mode: {mode}')