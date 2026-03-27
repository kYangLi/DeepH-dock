from pathlib import Path
from typing import List, Tuple
import numpy as np
import h5py


class BasisWriter:
    """
    Generic basis set writer supporting heterogeneous grids and cutoff radii.

    Implements the NAO storage architecture (v0.9.13):
    - 2D radius_grid matrix [M, N_max] (no interpolation, 100% physical fidelity)
    - Per-orbital cutoff_radii and grid_length
    - Linear extrapolation padding for radius_grid
    - Zero padding for radius_basis

    Usage for OpenMX (uniform grid):
        writer = BasisWriter(
            element="Fe",
            basis_name="Fe6.0H",
            source="openmx",
            radius_grids=[grid] * M,  # same grid for all orbitals
            radius_funcs=funcs,
            cutoff_radii=np.full(M, 7.0),
            mul_list=[2, 2, 1],
        )

    Usage for SIESTA (heterogeneous grids):
        writer = BasisWriter(
            element="Fe",
            basis_name="Fe_DZP",
            source="siesta",
            radius_grids=grids,  # different grid per orbital
            radius_funcs=funcs,
            cutoff_radii=cutoffs,
            mul_list=[2, 2, 1],
        )
    """

    def __init__(
        self,
        element: str,
        basis_name: str,
        source: str,
        radius_grids: List[np.ndarray],
        radius_funcs: List[np.ndarray],
        cutoff_radii: np.ndarray,
        mul_list: np.ndarray,
        normalized: bool = True,
    ):
        """
        Initialize basis writer.

        Parameters
        ----------
        element : str
            Element symbol (e.g., "Fe").
        basis_name : str
            Basis set name (e.g., "Fe6.0H").
        source : str
            DFT code source (e.g., "openmx", "siesta").
        radius_grids : List[np.ndarray]
            List of radial grid arrays for each orbital [M][Ni].
        radius_funcs : List[np.ndarray]
            List of radial function arrays for each orbital [M][Ni].
        cutoff_radii : np.ndarray
            Per-orbital cutoff radii (Bohr) [M].
        mul_list : np.ndarray
            Multiplicity per angular momentum [lmax+1].
        normalized : bool
            Whether radial functions are normalized.
        """
        self.element = element
        self.basis_name = basis_name
        self.source = source
        self.normalized = normalized

        self.radius_grids = radius_grids
        self.radius_funcs = radius_funcs
        self.cutoff_radii = np.asarray(cutoff_radii, dtype=np.float64)
        self.mul_list = np.asarray(mul_list, dtype=np.int32)

        self.total_orbitals = len(radius_funcs)
        self.grid_length = np.array([len(g) for g in radius_grids], dtype=np.int32)

        assert self.total_orbitals == len(radius_grids)
        assert self.total_orbitals == len(cutoff_radii)
        assert self.total_orbitals == sum(mul_list)

    def build_2d_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 2D grid and basis matrices with padding.

        Padding rules:
        - radius_grid: extrapolate from r_max using average spacing
        - radius_basis: fill with 0.0

        Returns
        -------
        radius_grid : np.ndarray
            2D grid matrix [M, N_max].
        radius_basis : np.ndarray
            2D basis matrix [M, N_max].
        """
        n_max = int(self.grid_length.max())

        radius_grid = np.zeros((self.total_orbitals, n_max), dtype=np.float64)
        radius_basis = np.zeros((self.total_orbitals, n_max), dtype=np.float64)

        for i in range(self.total_orbitals):
            ni = self.grid_length[i]
            grid_i = self.radius_grids[i]
            func_i = self.radius_funcs[i]

            radius_grid[i, :ni] = grid_i[:ni]
            radius_basis[i, :ni] = func_i[:ni]

            if ni < n_max:
                r_max = grid_i[ni - 1]
                dr = r_max / ni
                for j in range(ni, n_max):
                    radius_grid[i, j] = r_max + dr * (j - ni + 1)

        return radius_grid, radius_basis

    def save(self, output_path: Path) -> None:
        """
        Save basis to HDF5 file.

        Parameters
        ----------
        output_path : Path
            Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        radius_grid, radius_basis = self.build_2d_matrices()

        with h5py.File(output_path, "w") as f:
            f.attrs["element"] = self.element
            f.attrs["basis_name"] = self.basis_name
            f.attrs["source"] = self.source
            f.attrs["normalized"] = self.normalized
            f.attrs["units_length"] = "bohr"

            f.create_dataset("mul_list", data=self.mul_list)
            f.create_dataset("cutoff_radii", data=self.cutoff_radii)
            f.create_dataset("grid_length", data=self.grid_length)
            f.create_dataset("radius_grid", data=radius_grid)
            f.create_dataset("radius_basis", data=radius_basis)


def write_uniform_grid_basis(
    element: str,
    basis_name: str,
    source: str,
    radius_grid: np.ndarray,
    radius_funcs: List[np.ndarray],
    cutoff: float,
    mul_list: np.ndarray,
    output_path: Path,
) -> None:
    """
    Convenience function for uniform grid (OpenMX case).

    All orbitals share the same grid.

    Parameters
    ----------
    element : str
        Element symbol.
    basis_name : str
        Basis set name.
    source : str
        DFT code source.
    radius_grid : np.ndarray
        Single radial grid shared by all orbitals.
    radius_funcs : List[np.ndarray]
        List of radial functions.
    cutoff : float
        Uniform cutoff radius for all orbitals.
    mul_list : np.ndarray
        Multiplicity per angular momentum.
    output_path : Path
        Output file path.
    """
    M = len(radius_funcs)
    radius_grids = [radius_grid] * M
    cutoff_radii = np.full(M, cutoff, dtype=np.float64)

    writer = BasisWriter(
        element=element,
        basis_name=basis_name,
        source=source,
        radius_grids=radius_grids,
        radius_funcs=radius_funcs,
        cutoff_radii=cutoff_radii,
        mul_list=mul_list,
    )
    writer.save(output_path)
