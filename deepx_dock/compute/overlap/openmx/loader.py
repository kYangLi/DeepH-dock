from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import h5py

from HPRO.utils.orbutils import GridFunc, LinearRGD
from HPRO.utils.structure import Structure
from HPRO.io.aodata import AOData
from HPRO.utils.misc import atom_number2name


def _interpolate_to_linear_grid(
    r_orig: np.ndarray,
    func_orig: np.ndarray,
    rcut: float,
    ell: int,
    rdense: float,
) -> GridFunc:
    """
    Interpolate from PAO's internal grid to uniform linear grid (OpenMX style).

    Parameters
    ----------
    r_orig : np.ndarray
        Original grid points (PAO uses logarithmic grid internally)
    func_orig : np.ndarray
        Function values on original grid
    rcut : float
        Cutoff radius
    ell : int
        Angular momentum
    rdense : float
        Linear grid density (points per Bohr)

    Returns
    -------
    GridFunc
        GridFunc on linear grid
    """
    from scipy.interpolate import CubicSpline

    npoints = max(int(rcut * rdense), 10)
    r_linear = np.linspace(0, rcut, npoints)

    spline = CubicSpline(r_orig, func_orig)
    func_linear = spline(r_linear)

    if ell == 0:
        func_linear[r_linear < r_orig[0]] = func_orig[0]
    else:
        func_linear[r_linear < r_orig[0]] = 0.0

    func_linear[r_linear > rcut] = 0.0

    rgd_linear = LinearRGD(0, rcut, npoints)

    return GridFunc(rgd_linear, func_linear, l=ell, rcut=rcut)


class BasisLoader:
    """
    Loader for standardized basis.h5 format.

    Provides convenient access to radial basis functions by (l, mul) indices.
    Supports per-orbital grids and cutoff radii for heterogeneous basis sets.

    Storage format:
    - radius_grid: [M, N_max] - 2D grid matrix (linear extrapolation padding)
    - radius_basis: [M, N_max] - radial functions (padded with 0.0)
    - grid_length: [M] - effective grid points per orbital
    - cutoff_radii: [M] - per-orbital cutoff radius
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

            self.mul_list: np.ndarray = f["mul_list"][:]
            self.cutoff_radii: np.ndarray = f["cutoff_radii"][:]
            self.grid_length: np.ndarray = f["grid_length"][:]
            self.radius_grid: np.ndarray = f["radius_grid"][:]
            self.radius_basis: np.ndarray = f["radius_basis"][:]

        self.lmax = len(self.mul_list) - 1
        self.total_orbitals = int(np.sum(self.mul_list))
        self.max_cutoff = float(self.cutoff_radii.max())

        self._offsets = np.cumsum(np.concatenate([[0], self.mul_list]))[:-1]

    def get_orbital_grid(self, ell: int, mul: int) -> np.ndarray:
        """
        Get radial grid for specific (l, mul).

        Parameters
        ----------
        ell : int
            Angular momentum quantum number.
        mul : int
            Multiplicity index (0-indexed).

        Returns
        -------
        np.ndarray
            Radial grid points (Bohr).
        """
        idx = self._get_flat_index(ell, mul)
        ni = self.grid_length[idx]
        return self.radius_grid[idx, :ni]

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
        idx = self._get_flat_index(ell, mul)
        ni = self.grid_length[idx]
        return self.radius_basis[idx, :ni]

    def get_cutoff(self, ell: int, mul: int) -> float:
        """
        Get cutoff radius for specific (l, mul).

        Parameters
        ----------
        ell : int
            Angular momentum quantum number.
        mul : int
            Multiplicity index (0-indexed).

        Returns
        -------
        float
            Cutoff radius (Bohr).
        """
        idx = self._get_flat_index(ell, mul)
        return float(self.cutoff_radii[idx])

    def get_grid_length(self, ell: int, mul: int) -> int:
        """
        Get grid length for specific (l, mul).

        Parameters
        ----------
        ell : int
            Angular momentum quantum number.
        mul : int
            Multiplicity index (0-indexed).

        Returns
        -------
        int
            Number of grid points.
        """
        idx = self._get_flat_index(ell, mul)
        return int(self.grid_length[idx])

    def _get_flat_index(self, ell: int, mul: int) -> int:
        """Get flat index from (l, mul)."""
        if ell < 0 or ell > self.lmax:
            raise ValueError(f"ell={ell} out of range [0, {self.lmax}]")
        if mul < 0 or mul >= self.mul_list[ell]:
            raise ValueError(f"mul={mul} out of range [0, {self.mul_list[ell]})")
        return self._offsets[ell] + mul

    def to_gridfuncs(
        self,
        orbital_selection: Optional[Dict[int, int]] = None,
        rdense: float = 100.0,
    ) -> List[GridFunc]:
        """
        Convert to list of GridFunc objects on uniform linear grids (OpenMX style).

        Parameters
        ----------
        orbital_selection : dict, optional
            Orbital selection rule: {L: num_mu}.
            If None, load all orbitals.
        rdense : float
            Linear grid density (points per Bohr). Default: 100.0
            OpenMX default: NumGridR=900 for typical ~9 Bohr cutoff.

        Returns
        -------
        List[GridFunc]
            List of GridFunc objects on linear grids.
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
                idx = self._offsets[ell] + mul
                ni = self.grid_length[idx]

                r_orig = self.radius_grid[idx, :ni]
                func_orig = self.radius_basis[idx, :ni]
                rcut = self.cutoff_radii[idx]

                gf = _interpolate_to_linear_grid(r_orig, func_orig, rcut, ell, rdense)
                gridfuncs.append(gf)

        return gridfuncs


def load_basis_h5(
    filepath: Path,
    orbital_selection: Optional[Dict[int, int]] = None,
    rdense: float = 100.0,
) -> List[GridFunc]:
    """
    Load GridFunc objects from standardized basis.h5 file.

    Interpolates to uniform linear grid (OpenMX style).

    Parameters
    ----------
    filepath : Path
        Path to the basis.h5 file.
    orbital_selection : dict, optional
        Orbital selection rule: {L: num_mu}.
        If None, load all orbitals.
    rdense : float
        Linear grid density (points per Bohr). Default: 100.0

    Returns
    -------
    List[GridFunc]
        List of GridFunc objects on linear grids.
    """
    loader = BasisLoader(filepath)
    return loader.to_gridfuncs(orbital_selection, rdense)


class AOData_openmx(AOData):
    """
    AOData subclass for OpenMX basis sets loaded from standardized basis.h5.

    Uses uniform linear r-space grids (OpenMX style).
    """

    def __init__(
        self,
        structure: Structure,
        basis_files: Dict[int, Path],
        orbital_selections: Optional[Dict[int, Dict[int, int]]] = None,
        rdense: float = 100.0,
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
        rdense : float
            Linear r-space grid density (points per Bohr). Default: 100.0
            OpenMX default: NumGridR=900 for typical ~9 Bohr cutoff.
        """
        self.structure = structure
        self.aocode = "openmx"
        self.spinful = False
        self.magnetic = False
        self.rdense = rdense

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

            gridfuncs = load_basis_h5(basis_file, orbital_selection, rdense)

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
