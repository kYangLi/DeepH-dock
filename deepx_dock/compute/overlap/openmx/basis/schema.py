"""
Basis set data structures and HDF5 schema.

This module defines the data structures for representing basis sets
in a unified HDF5 format, compatible with OpenMX PAO files and
other DFT software.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime


class GridType(Enum):
    """Grid type for radial functions."""

    LOG = "log"  # Logarithmic grid: x = log(r)
    LINEAR = "linear"  # Linear grid: x = r


@dataclass
class RadialGrid:
    """
    Radial grid for basis functions.

    Parameters
    ----------
    grid_type : GridType
        Type of grid (log or linear)
    num_points : int
        Number of grid points
    x : np.ndarray
        Grid coordinate, either log(r) or r, shape: (N,)
    r : np.ndarray
        Radial distance in Bohr, shape: (N,)
    dr : np.ndarray
        Grid spacing, shape: (N,)
    """

    grid_type: GridType
    num_points: int
    x: np.ndarray
    r: np.ndarray
    dr: np.ndarray

    def save_h5(self, group: h5py.Group):
        """Save radial grid to HDF5 group."""
        group.attrs["grid_type"] = self.grid_type.value
        group.attrs["grid_num"] = self.num_points
        group.create_dataset("x", data=self.x, compression="gzip")
        group.create_dataset("r", data=self.r, compression="gzip")
        group.create_dataset("dr", data=self.dr, compression="gzip")

    @classmethod
    def load_h5(cls, group: h5py.Group) -> "RadialGrid":
        """Load radial grid from HDF5 group."""
        grid_type = GridType(group.attrs["grid_type"])
        num_points = int(group.attrs["grid_num"])
        x = group["x"][:]
        r = group["r"][:]
        dr = group["dr"][:]
        return cls(grid_type, num_points, x, r, dr)


@dataclass
class BasisMetadata:
    """
    Metadata for a basis set.

    Parameters
    ----------
    radial_cutoff : float
        Radial cutoff distance in Bohr
    lmax : int
        Maximum angular momentum quantum number
    num_mu : int
        Number of radial functions per angular momentum
    grid_type : GridType
        Type of radial grid
    grid_num : int
        Number of grid points
    eigenvalues : np.ndarray
        Eigenvalues for each (L, mu), shape: (lmax+1, num_mu)
    """

    radial_cutoff: float
    lmax: int
    num_mu: int
    grid_type: GridType
    grid_num: int
    eigenvalues: np.ndarray

    def save_h5(self, group: h5py.Group):
        """Save metadata to HDF5 group."""
        group.attrs["radial_cutoff"] = self.radial_cutoff
        group.attrs["lmax"] = self.lmax
        group.attrs["num_mu"] = self.num_mu
        group.attrs["grid_type"] = self.grid_type.value
        group.attrs["grid_num"] = self.grid_num
        group.create_dataset("eigenvalues", data=self.eigenvalues, compression="gzip")

    @classmethod
    def load_h5(cls, group: h5py.Group) -> "BasisMetadata":
        """Load metadata from HDF5 group."""
        return cls(
            radial_cutoff=float(group.attrs["radial_cutoff"]),
            lmax=int(group.attrs["lmax"]),
            num_mu=int(group.attrs["num_mu"]),
            grid_type=GridType(group.attrs["grid_type"]),
            grid_num=int(group.attrs["grid_num"]),
            eigenvalues=group["eigenvalues"][:],
        )


@dataclass
class KSpaceData:
    """
    k-space radial function data (precomputed).

    Parameters
    ----------
    k_grid : np.ndarray
        k-space grid, shape: (N_k,)
    wf : np.ndarray
        k-space wave functions R̃(k), shape: (lmax+1, num_mu, N_k)
    k_max : float
        Maximum k value
    num_k : int
        Number of k points
    """

    k_grid: np.ndarray
    wf: np.ndarray
    k_max: float
    num_k: int

    @property
    def lmax(self) -> int:
        """Maximum angular momentum."""
        return self.wf.shape[0] - 1

    @property
    def num_mu(self) -> int:
        """Number of radial functions per L."""
        return self.wf.shape[1]

    def save_h5(self, group: h5py.Group):
        """Save k-space data to HDF5 group."""
        group.create_dataset("k_grid", data=self.k_grid, compression="gzip")
        group.create_dataset("wf", data=self.wf, compression="gzip")
        group.attrs["k_max"] = self.k_max
        group.attrs["num_k"] = self.num_k

    @classmethod
    def load_h5(cls, group: h5py.Group) -> "KSpaceData":
        """Load k-space data from HDF5 group."""
        return cls(
            k_grid=group["k_grid"][:],
            wf=group["wf"][:],
            k_max=float(group.attrs["k_max"]),
            num_k=int(group.attrs["num_k"]),
        )


@dataclass
class BasisSet:
    """
    A single basis set (e.g., C7.0).

    Parameters
    ----------
    name : str
        Basis set name (e.g., "7.0")
    metadata : BasisMetadata
        Metadata
    radial_grid : RadialGrid
        Radial grid
    radial_wf : np.ndarray
        Radial wave functions R(r), shape: (lmax+1, num_mu, N)
    k_space : KSpaceData, optional
        Precomputed k-space data
    valence_density : np.ndarray, optional
        Valence electron density, shape: (N,)
    """

    name: str
    metadata: BasisMetadata
    radial_grid: RadialGrid
    radial_wf: np.ndarray
    k_space: Optional[KSpaceData] = None
    valence_density: Optional[np.ndarray] = None

    def save_h5(self, group: h5py.Group):
        """Save basis set to HDF5 group."""
        group.attrs["name"] = self.name

        meta_group = group.create_group("metadata")
        self.metadata.save_h5(meta_group)

        grid_group = group.create_group("radial_grid")
        self.radial_grid.save_h5(grid_group)

        group.create_dataset("radial_wf/data", data=self.radial_wf, compression="gzip")

        if self.k_space is not None:
            ks_group = group.create_group("k_space")
            self.k_space.save_h5(ks_group)

        if self.valence_density is not None:
            group.create_dataset("valence_density/data", data=self.valence_density, compression="gzip")

    @classmethod
    def load_h5(cls, group: h5py.Group) -> "BasisSet":
        """Load basis set from HDF5 group."""
        name = group.attrs["name"]

        metadata = BasisMetadata.load_h5(group["metadata"])
        radial_grid = RadialGrid.load_h5(group["radial_grid"])
        radial_wf = group["radial_wf/data"][:]

        k_space = None
        if "k_space" in group:
            k_space = KSpaceData.load_h5(group["k_space"])

        valence_density = None
        if "valence_density" in group:
            valence_density = group["valence_density/data"][:]

        return cls(
            name=name,
            metadata=metadata,
            radial_grid=radial_grid,
            radial_wf=radial_wf,
            k_space=k_space,
            valence_density=valence_density,
        )

    def get_radial_wf(self, L: int, mu: int) -> np.ndarray:
        """
        Get radial wave function for specific (L, mu).

        Parameters
        ----------
        L : int
            Angular momentum quantum number
        mu : int
            Radial function index

        Returns
        -------
        R : np.ndarray
            Radial wave function R_L,mu(r), shape: (N,)
        """
        assert 0 <= L <= self.metadata.lmax, f"L={L} out of range [0, {self.metadata.lmax}]"
        assert 0 <= mu < self.metadata.num_mu, f"mu={mu} out of range [0, {self.metadata.num_mu})"
        return self.radial_wf[L, mu, :]

    def get_k_space_wf(self, L: int, mu: int) -> np.ndarray:
        """
        Get k-space wave function for specific (L, mu).

        Parameters
        ----------
        L : int
            Angular momentum quantum number
        mu : int
            Radial function index

        Returns
        -------
        R_tilde : np.ndarray
            k-space wave function R̃_L,mu(k), shape: (N_k,)
        """
        if self.k_space is None:
            raise ValueError("k-space data not computed. Call compute_k_space() first.")
        return self.k_space.wf[L, mu, :]


@dataclass
class ElementBasis:
    """
    All basis sets for a single element.

    Parameters
    ----------
    atomic_number : int
        Atomic number Z
    symbol : str
        Element symbol (e.g., "C")
    valence_electrons : float
        Number of valence electrons
    mass : float
        Atomic mass
    basis_sets : dict
        Dictionary of basis sets, name -> BasisSet
    """

    atomic_number: int
    symbol: str
    valence_electrons: float
    mass: float
    basis_sets: Dict[str, BasisSet] = field(default_factory=dict)

    def save_h5(self, filepath: str | Path):
        """
        Save element basis to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            f.attrs["atomic_number"] = self.atomic_number
            f.attrs["symbol"] = self.symbol
            f.attrs["valence_electrons"] = self.valence_electrons
            f.attrs["mass"] = self.mass

            meta_group = f.create_group("metadata")
            meta_group.attrs["version"] = "1.0.0"
            meta_group.attrs["created"] = datetime.now().isoformat()
            meta_group.attrs["source"] = "openmx"
            meta_group.attrs["description"] = f"{self.symbol} basis sets"

            for name, basis in self.basis_sets.items():
                group = f.create_group(f"basis_sets/{name}")
                basis.save_h5(group)

    @classmethod
    def load_h5(cls, filepath: str | Path) -> "ElementBasis":
        """
        Load element basis from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Input file path

        Returns
        -------
        ElementBasis
            Loaded element basis
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "r") as f:
            atomic_number = int(f.attrs["atomic_number"])
            symbol = f.attrs["symbol"]
            valence_electrons = float(f.attrs["valence_electrons"])
            mass = float(f.attrs["mass"])

            basis_sets = {}
            if "basis_sets" in f:
                for name in f["basis_sets"].keys():
                    basis_sets[name] = BasisSet.load_h5(f[f"basis_sets/{name}"])

            return cls(
                atomic_number=atomic_number,
                symbol=symbol,
                valence_electrons=valence_electrons,
                mass=mass,
                basis_sets=basis_sets,
            )

    def get_basis_set(self, name: str) -> BasisSet:
        """
        Get a specific basis set by name.

        Parameters
        ----------
        name : str
            Basis set name (e.g., "7.0")

        Returns
        -------
        BasisSet
            Requested basis set
        """
        if name not in self.basis_sets:
            available = list(self.basis_sets.keys())
            raise KeyError(f"Basis set '{name}' not found. Available: {available}")
        return self.basis_sets[name]

    def get_default_basis_set(self) -> BasisSet:
        """
        Get the default basis set (largest cutoff radius).

        Returns
        -------
        BasisSet
            Default basis set
        """
        if not self.basis_sets:
            raise ValueError("No basis sets available")

        max_cutoff = 0.0
        default_name = None
        for name, basis in self.basis_sets.items():
            if basis.metadata.radial_cutoff > max_cutoff:
                max_cutoff = basis.metadata.radial_cutoff
                default_name = name

        return self.basis_sets[default_name]

    def list_basis_sets(self) -> list:
        """List all available basis set names."""
        return list(self.basis_sets.keys())
