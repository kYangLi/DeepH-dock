from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import h5py

from HPRO.utils.orbutils import GridFunc, ExpRGD
from HPRO.utils.structure import Structure
from HPRO.io.aodata import AOData
from HPRO.utils.misc import atom_number2name


class BasisLoader:
    """
    Loader for standardized basis.h5 format (v0.9.16).

    Provides convenient access to radial basis functions by (l, mul) indices.
    """

    def __init__(self, filepath: Path):
        """
        Load basis.h5 file.

        Parameters
        ----------
        filepath : Path
            Path to the basis.h5 file.
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "r") as f:
            self.element: str = f.attrs["element"]
            self.basis_name: str = f.attrs["basis_name"]
            self.source: str = f.attrs["source"]
            self.normalized: bool = f.attrs["normalized"]
            self.units_length: str = f.attrs["units_length"]

            self.radial_grid: np.ndarray = f["radial_grid"][:]
            self.mul_list: np.ndarray = f["mul_list"][:]
            self.radial_basis: np.ndarray = f["radial_basis"][:]

        self.rgd = ExpRGD.from_explicit_grid(self.radial_grid)
        self.lmax = len(self.mul_list) - 1
        self.total_orbitals = int(np.sum(self.mul_list))
        self.radial_cutoff = float(self.radial_grid.max())

        self._offsets = np.cumsum(np.concatenate([[0], self.mul_list]))[:-1]

    def get_orbital(self, ell: int, mul: int) -> np.ndarray:
        """
        Get radial function for specific (l, mul).

        Parameters
        ----------
        ell : int
            Angular momentum quantum number.
        mul : int
            Multiplicity index (0-indexed).

        Returns
        -------
        np.ndarray
            Radial function on the grid.
        """
        if ell < 0 or ell > self.lmax:
            raise ValueError(f"ell={ell} out of range [0, {self.lmax}]")
        if mul < 0 or mul >= self.mul_list[ell]:
            raise ValueError(f"mul={mul} out of range [0, {self.mul_list[ell]})")

        idx = self._offsets[ell] + mul
        return self.radial_basis[idx]

    def get_all_l(self, ell: int) -> np.ndarray:
        """
        Get all orbitals for a given angular momentum l.

        Parameters
        ----------
        ell : int
            Angular momentum quantum number.

        Returns
        -------
        np.ndarray
            Array of shape [mul_max_l, Nr] containing all radial functions for this l.
        """
        if ell < 0 or ell > self.lmax:
            raise ValueError(f"ell={ell} out of range [0, {self.lmax}]")

        start = self._offsets[ell]
        end = start + self.mul_list[ell]
        return self.radial_basis[start:end]

    def to_gridfuncs(self, orbital_selection: Optional[Dict[int, int]] = None) -> List[GridFunc]:
        """
        Convert to list of GridFunc objects.

        Parameters
        ----------
        orbital_selection : dict, optional
            Orbital selection rule: {L: num_mu}.
            If None, load all orbitals.

        Returns
        -------
        List[GridFunc]
            List of GridFunc objects.
        """
        gridfuncs = []

        for ell in range(self.lmax + 1):
            if orbital_selection is not None:
                if ell not in orbital_selection:
                    continue
                num_mu = min(orbital_selection[ell], self.mul_list[ell])
            else:
                num_mu = self.mul_list[ell]

            for mul in range(num_mu):
                func = self.get_orbital(ell, mul)
                gf = GridFunc(self.rgd, func, l=ell, rcut=self.radial_cutoff)
                gridfuncs.append(gf)

        return gridfuncs


def load_basis_h5(filepath: Path, orbital_selection: Optional[Dict[int, int]] = None) -> List[GridFunc]:
    """
    Load GridFunc objects from standardized basis.h5 file (v0.9.16).

    Parameters
    ----------
    filepath : Path
        Path to the basis.h5 file.
    orbital_selection : dict, optional
        Orbital selection rule: {L: num_mu}.
        If None, load all orbitals.

    Returns
    -------
    List[GridFunc]
        List of GridFunc objects.
    """
    loader = BasisLoader(filepath)
    return loader.to_gridfuncs(orbital_selection)


class AOData_openmx(AOData):
    """
    AOData subclass for OpenMX basis sets loaded from standardized basis.h5.
    """

    def __init__(
        self,
        structure: Structure,
        basis_files: Dict[int, Path],
        orbital_selections: Optional[Dict[int, Dict[int, int]]] = None,
    ):
        """
        Initialize AOData from standardized basis.h5 files.

        Parameters
        ----------
        structure : Structure
            HPRO Structure object.
        basis_files : dict
            Mapping from atomic_number to basis.h5 Path.
            {atomic_number: /path/to/basis.h5}
        orbital_selections : dict, optional
            Mapping from atomic_number to orbital selection.
            {atomic_number: {L: num_mu}}
        """
        self.structure = structure
        self.aocode = "openmx"
        self.spinful = False
        self.magnetic = False

        self.ls_spc = {}
        self.phirgrids_spc = {}
        self.nradial_spc = {}
        self.cutoffs = {}

        processed_spc = set()

        for spc_nu in structure.atomic_numbers:
            if spc_nu in processed_spc:
                continue
            processed_spc.add(spc_nu)

            basis_file = basis_files.get(spc_nu)
            if basis_file is None:
                spc_na = atom_number2name([spc_nu])[0]
                raise ValueError(f"No basis file provided for element {spc_na} (Z={spc_nu})")

            orbital_selection = None
            if orbital_selections and spc_nu in orbital_selections:
                orbital_selection = orbital_selections[spc_nu]

            gridfuncs = load_basis_h5(basis_file, orbital_selection)

            self.phirgrids_spc[spc_nu] = gridfuncs
            self.ls_spc[spc_nu] = [gf.l for gf in gridfuncs]
            self.nradial_spc[spc_nu] = len(gridfuncs)

            spc_na = atom_number2name([spc_nu])[0]
            self.cutoffs[spc_na] = max(gf.rcut for gf in gridfuncs)

        orbslices_spc = {}
        norbfull_spc = {}
        for spc, orbital_types in self.ls_spc.items():
            orbital_slices = [0]
            for angmom in orbital_types:
                orbital_slices.append(orbital_slices[-1] + 2 * angmom + 1)
            orbslices_spc[spc] = orbital_slices
            norbfull_spc[spc] = orbital_slices[-1]

        self.orbslices_spc = orbslices_spc
        self.norbfull_spc = norbfull_spc
        self.phiQlist_spc = None
        self.phiQEcut = None
