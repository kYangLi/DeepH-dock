"""Loader for unified species_{source}_{xc}.h5 format."""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import h5py

from HPRO.utils.orbutils import GridFunc, LinearRGD


@dataclass
class SpeciesMetadata:
    """Metadata for a species."""

    element: str
    species_name: str
    valence_electrons: float
    basis_source: str
    pseudo_source: str
    xc_functional: str


@dataclass
class BasisData:
    """Basis function data for a species."""

    nljz_list: np.ndarray
    cutoff_radii: np.ndarray
    grid_length: np.ndarray
    radius_grid: np.ndarray
    radius_data: np.ndarray

    @property
    def lmax(self) -> int:
        return int(self.nljz_list[:, 1].max())

    @property
    def total_orbitals(self) -> int:
        return len(self.nljz_list)

    @property
    def mul_list(self) -> np.ndarray:
        """Derive mul_list from nljz_list for backward compatibility."""
        lmax = self.lmax
        mul_list = np.zeros(lmax + 1, dtype=int)
        for ell in range(lmax + 1):
            mul_list[ell] = np.sum(self.nljz_list[:, 1] == ell)
        return mul_list


@dataclass
class PhysicalQuantityData:
    """Generic physical quantity data (density, pseudopotential)."""

    cutoff_radii: np.ndarray
    grid_length: np.ndarray
    radius_grid: np.ndarray
    radius_data: np.ndarray
    nljz_list: Optional[np.ndarray] = None


@dataclass
class NonlocalProjectorData:
    """Nonlocal projector data with j-split channels expanded."""

    nljz_list: np.ndarray
    cutoff_radii: np.ndarray
    grid_length: np.ndarray
    radius_grid: np.ndarray
    radius_data: np.ndarray

    @property
    def num_projectors(self) -> int:
        return len(self.nljz_list)

    @property
    def l_list(self) -> np.ndarray:
        """Derive l_list from nljz_list for backward compatibility."""
        return self.nljz_list[:, 1]

    @property
    def mul_list(self) -> np.ndarray:
        """Derive mul_list from nljz_list for backward compatibility."""
        l_values = sorted(set(self.nljz_list[:, 1]))
        mul_list = np.zeros(len(l_values), dtype=int)
        for i, l_val in enumerate(l_values):
            mul_list[i] = np.sum(self.nljz_list[:, 1] == l_val)
        return mul_list


class SpeciesLoader:
    """
    Loader for unified species_{source}_{xc}.h5 format.

    Provides access to basis, density, and pseudopotential data for all species.

    Storage format:
    species_openmx_pbe.h5
    ├── @xc_functional, @global_nmax, @source
    └── /{species_name}
        ├── @element, @species_name, @valence_electrons
        ├── @basis_source, @pseudo_source
        ├── /basis
        ├── /val_density
        └── /pseudopotential
            ├── /local
            ├── /nonlocal
            └── /core_density (if NLCC)
    """

    def __init__(self, filepath: str | Path):
        """
        Load species_{source}_{xc}.h5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to the species file (e.g., species_openmx_pbe.h5).
        """
        filepath = Path(filepath)
        assert filepath.exists(), f"Species file not found: {filepath}"

        self.filepath = filepath
        self._species_cache: Dict[str, dict] = {}

        with h5py.File(filepath, "r") as f:
            self.xc_functional: str = f.attrs["xc_functional"]
            self.global_nmax: int = f.attrs["global_nmax"]
            self.source: str = f.attrs["source"]
            self.units_length: str = f.attrs.get("units_length", "bohr")

            self.species_names: List[str] = list(f.keys())

            self._element_to_species: Dict[str, List[str]] = {}
            for name in self.species_names:
                elem = f[name].attrs["element"]
                if elem not in self._element_to_species:
                    self._element_to_species[elem] = []
                self._element_to_species[elem].append(name)

    def get_element_species(self, element: str) -> List[str]:
        """
        Get all species names for a given element.

        Parameters
        ----------
        element : str
            Element symbol (e.g., "Mo", "Fe").

        Returns
        -------
        List[str]
            List of species names (e.g., ["Mo7.0", "Mo7.0H", "Mo7.0S"]).
        """
        return self._element_to_species.get(element, [])

    def get_species_metadata(self, species_name: str) -> SpeciesMetadata:
        """
        Get metadata for a species.

        Parameters
        ----------
        species_name : str
            Species name (e.g., "Mo7.0", "Fe5.5H").

        Returns
        -------
        SpeciesMetadata
            Species metadata.
        """
        with h5py.File(self.filepath, "r") as f:
            grp = f[species_name]
            return SpeciesMetadata(
                element=grp.attrs["element"],
                species_name=grp.attrs.get("species_name", species_name),
                valence_electrons=grp.attrs.get("valence_electrons", 0.0),
                basis_source=grp.attrs.get("basis_source", ""),
                pseudo_source=grp.attrs.get("pseudo_source", ""),
                xc_functional=grp.attrs.get("xc_functional", self.xc_functional),
            )

    def get_basis_data(self, species_name: str) -> BasisData:
        """
        Get basis data for a species.

        Parameters
        ----------
        species_name : str
            Species name (e.g., "Mo7.0", "Fe5.5H").

        Returns
        -------
        BasisData
            Basis function data.
        """
        with h5py.File(self.filepath, "r") as f:
            grp = f[f"{species_name}/basis"]

            if "nljz_list" in grp:
                nljz_list = grp["nljz_list"][:]
            elif "mul_list" in grp:
                mul_list = grp["mul_list"][:]
                nljz_list = self._derive_nljz_from_mul_list(mul_list)
            else:
                raise KeyError(f"No nljz_list or mul_list found for {species_name}/basis")

            return BasisData(
                nljz_list=nljz_list,
                cutoff_radii=grp["cutoff_radii"][:],
                grid_length=grp["grid_length"][:],
                radius_grid=grp["radius_grid"][:],
                radius_data=grp["radius_data"][:],
            )

    def _derive_nljz_from_mul_list(self, mul_list: np.ndarray) -> np.ndarray:
        """Derive nljz_list from mul_list for backward compatibility."""
        nljz_list = []
        for ell, num_zeta in enumerate(mul_list):
            for z in range(1, int(num_zeta) + 1):
                nljz_list.append([0, ell, 0, z])
        return np.array(nljz_list, dtype=int)

    def get_valence_density(self, species_name: str) -> Optional[PhysicalQuantityData]:
        """
        Get valence density data for a species.

        Parameters
        ----------
        species_name : str
            Species name.

        Returns
        -------
        PhysicalQuantityData or None
            Valence density data, or None if not present.
        """
        with h5py.File(self.filepath, "r") as f:
            path = f"{species_name}/val_density"
            if path not in f:
                path_old = f"{species_name}/density/valence"
                if path_old not in f:
                    return None
                path = path_old
            grp = f[path]

            nljz_list = None
            if "nljz_list" in grp:
                nljz_list = grp["nljz_list"][:]
            else:
                nljz_list = np.array([[0, 0, 0, 1]], dtype=int)

            return PhysicalQuantityData(
                cutoff_radii=grp["cutoff_radii"][:],
                grid_length=grp["grid_length"][:],
                radius_grid=grp["radius_grid"][:],
                radius_data=grp["radius_data"][:],
                nljz_list=nljz_list,
            )

    def get_local_pseudopotential(self, species_name: str) -> Optional[PhysicalQuantityData]:
        """
        Get local pseudopotential data for a species.

        Parameters
        ----------
        species_name : str
            Species name.

        Returns
        -------
        PhysicalQuantityData or None
            Local pseudopotential data, or None if not present.
        """
        with h5py.File(self.filepath, "r") as f:
            path = f"{species_name}/pseudopotential/local"
            if path not in f:
                return None
            grp = f[path]

            nljz_list = None
            if "nljz_list" in grp:
                nljz_list = grp["nljz_list"][:]
            else:
                nljz_list = np.array([[0, 0, 0, 1]], dtype=int)

            return PhysicalQuantityData(
                cutoff_radii=grp["cutoff_radii"][:],
                grid_length=grp["grid_length"][:],
                radius_grid=grp["radius_grid"][:],
                radius_data=grp["radius_data"][:],
                nljz_list=nljz_list,
            )

    def get_nonlocal_projectors(self, species_name: str) -> Optional[NonlocalProjectorData]:
        """
        Get nonlocal projector data for a species.

        Parameters
        ----------
        species_name : str
            Species name.

        Returns
        -------
        NonlocalProjectorData or None
            Nonlocal projector data, or None if not present.
        """
        with h5py.File(self.filepath, "r") as f:
            path = f"{species_name}/pseudopotential/nonlocal"
            if path not in f:
                return None
            grp = f[path]

            if "nljz_list" in grp:
                nljz_list = grp["nljz_list"][:]
            elif "l_list" in grp and "mul_list" in grp:
                nljz_list = self._derive_nljz_from_l_mul_list(grp["l_list"][:], grp["mul_list"][:])
            else:
                raise KeyError(f"No nljz_list or l_list/mul_list found for {species_name}/pseudopotential/nonlocal")

            return NonlocalProjectorData(
                nljz_list=nljz_list,
                cutoff_radii=grp["cutoff_radii"][:],
                grid_length=grp["grid_length"][:],
                radius_grid=grp["radius_grid"][:],
                radius_data=grp["radius_data"][:],
            )

    def _derive_nljz_from_l_mul_list(self, l_list: np.ndarray, mul_list: np.ndarray) -> np.ndarray:
        """Derive nljz_list from old l_list/mul_list format for backward compatibility."""
        nljz_list = []
        for l_val in l_list:
            nljz_list.append([0, int(l_val), 0, 1])
        return np.array(nljz_list, dtype=int)

    def get_core_density(self, species_name: str) -> Optional[PhysicalQuantityData]:
        """
        Get core density data for a species (NLCC).

        Parameters
        ----------
        species_name : str
            Species name.

        Returns
        -------
        PhysicalQuantityData or None
            Core density data, or None if not present (no NLCC).
        """
        with h5py.File(self.filepath, "r") as f:
            path = f"{species_name}/pseudopotential/core_density"
            if path not in f:
                return None
            grp = f[path]
            return PhysicalQuantityData(
                cutoff_radii=grp["cutoff_radii"][:],
                grid_length=grp["grid_length"][:],
                radius_grid=grp["radius_grid"][:],
                radius_data=grp["radius_data"][:],
            )

    def validate_species_match(
        self,
        species_name: str,
        expected_vps: str,
    ) -> bool:
        """
        Validate that the species file matches expected pseudopotential.

        Parameters
        ----------
        species_name : str
            Species name.
        expected_vps : str
            Expected VPS filename from input file.

        Returns
        -------
        bool
            True if match, raises ValueError if mismatch.
        """
        metadata = self.get_species_metadata(species_name)
        if metadata.pseudo_source != expected_vps:
            raise ValueError(
                f"VPS mismatch for {species_name}: "
                f"expected '{expected_vps}', "
                f"but species file has '{metadata.pseudo_source}'"
            )
        return True

    def basis_to_gridfuncs(
        self,
        species_name: str,
        orbital_selection: Optional[Dict[int, int]] = None,
        rdense: float = 100.0,
    ) -> List[GridFunc]:
        """
        Convert basis data to list of GridFunc objects on uniform linear grids.

        Parameters
        ----------
        species_name : str
            Species name.
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
        basis = self.get_basis_data(species_name)
        gridfuncs = []

        for idx, nljz in enumerate(basis.nljz_list):
            ell = int(nljz[1])
            zeta = int(nljz[3])

            if orbital_selection is not None:
                if ell not in orbital_selection:
                    continue
                if zeta > orbital_selection[ell]:
                    continue

            ni = basis.grid_length[idx]

            r_orig = basis.radius_grid[idx, :ni]
            func_orig = basis.radius_data[idx, :ni]
            rcut = basis.cutoff_radii[idx]

            gf = _interpolate_to_linear_grid(r_orig, func_orig, rcut, ell, rdense)
            gridfuncs.append(gf)

        return gridfuncs

    def get_max_cutoff(self, species_name: str) -> float:
        """Get maximum cutoff radius for a species."""
        basis = self.get_basis_data(species_name)
        return float(basis.cutoff_radii.max())


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
        Original grid points.
    func_orig : np.ndarray
        Function values on original grid.
    rcut : float
        Cutoff radius.
    ell : int
        Angular momentum.
    rdense : float
        Linear grid density (points per Bohr).

    Returns
    -------
    GridFunc
        GridFunc on linear grid.
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
