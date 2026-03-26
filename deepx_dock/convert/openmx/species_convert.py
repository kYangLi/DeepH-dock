"""Convert OpenMX PAO + VPS files to unified species_openmx_{xc}.h5 format."""

import re
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import h5py

from deepx_dock.convert.openmx.pao_parser import (
    parse_pao_file,
    find_matching_vps,
    get_pao_nmax,
)
from deepx_dock.convert.openmx.vps_parser import (
    parse_vps_file,
    get_vps_nmax,
)


def scan_all_species(
    pao_dir: str | Path,
    vps_dir: str | Path,
    xc_type: str = "PBE19",
) -> Tuple[Dict[str, Dict], int]:
    """
    Scan all PAO and VPS files to build species mapping.

    Parameters
    ----------
    pao_dir : str or Path
        Directory containing PAO files.
    vps_dir : str or Path
        Directory containing VPS files.
    xc_type : str
        XC functional type (e.g., "PBE19", "CA19").

    Returns
    -------
    species_map : dict
        {species_name: {"pao": path, "vps": path, "element": str}}
    global_nmax : int
        Global maximum grid points.
    """
    pao_dir = Path(pao_dir)
    vps_dir = Path(vps_dir)

    species_map = {}
    global_nmax = 0

    pao_pattern = re.compile(r"^([A-Z][a-z]?)(.*)\.pao$")

    for pao_file in sorted(pao_dir.glob("*.pao")):
        match = pao_pattern.match(pao_file.name)
        if not match:
            continue

        element = match.group(1)
        suffix = match.group(2)
        species_name = f"{element}{suffix}"

        vps_filename = find_matching_vps(pao_file.name, element, xc_type)
        vps_path = vps_dir / vps_filename

        if not vps_path.exists():
            print(f"⚠️ Warning: No VPS found for {species_name}, expected {vps_filename}")
            continue

        nmax_pao = get_pao_nmax(pao_file)
        nmax_vps = get_vps_nmax(vps_path)

        species_map[species_name] = {
            "pao": str(pao_file),
            "vps": str(vps_path),
            "element": element,
        }

        global_nmax = max(global_nmax, nmax_pao, nmax_vps)

    return species_map, global_nmax


def pad_to_global_nmax(
    radius_grid: np.ndarray,
    radius_data: np.ndarray,
    grid_length: int | np.ndarray,
    global_nmax: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad arrays to global_nmax.

    Parameters
    ----------
    radius_grid : np.ndarray
        Original radius grid (1D, 2D, or 3D).
    radius_data : np.ndarray
        Original radius data (1D, 2D, or 3D).
    grid_length : int or np.ndarray
        Number of valid points.
    global_nmax : int
        Target size for last dimension.

    Returns
    -------
    padded_grid : np.ndarray
        Padded radius grid.
    padded_data : np.ndarray
        Padded radius data.
    """
    if radius_grid.ndim == 1:
        ni = int(grid_length[0]) if isinstance(grid_length, np.ndarray) else int(grid_length)
        padded_grid = np.zeros((1, global_nmax))
        padded_data = np.zeros((1, global_nmax))
        padded_grid[0, :ni] = radius_grid[:ni]
        padded_data[0, :ni] = radius_data[:ni]
    elif radius_grid.ndim == 2:
        M = radius_grid.shape[0]
        padded_grid = np.zeros((M, global_nmax))
        padded_data = np.zeros((M, global_nmax))
        for i in range(M):
            ni = (
                int(grid_length[i])
                if isinstance(grid_length, np.ndarray)
                else min(int(grid_length), radius_grid.shape[1])
            )
            if ni > 0:
                padded_grid[i, :ni] = radius_grid[i, :ni]
                padded_data[i, :ni] = radius_data[i, :ni]
    elif radius_grid.ndim == 3:
        M, J = radius_grid.shape[0], radius_grid.shape[1]
        padded_grid = np.zeros((M, J, global_nmax))
        padded_data = np.zeros((M, J, global_nmax))
        for i in range(M):
            ni = (
                int(grid_length[i])
                if isinstance(grid_length, np.ndarray)
                else min(int(grid_length), radius_grid.shape[2])
            )
            if ni > 0:
                padded_grid[i, :, :ni] = radius_grid[i, :, :ni]
                padded_data[i, :, :ni] = radius_data[i, :, :ni]
    else:
        raise ValueError(f"Unsupported array dimension: {radius_grid.ndim}")

    return padded_grid, padded_data


def write_physical_group(
    group: h5py.Group,
    radius_grid: np.ndarray,
    radius_data: np.ndarray,
    grid_length: int | np.ndarray,
    cutoff_radii: float | np.ndarray,
    global_nmax: int,
    nljz_list: np.ndarray = None,
) -> None:
    """
    Write a physical group to HDF5.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to write to.
    radius_grid : np.ndarray
        Radius grid (will be padded).
    radius_data : np.ndarray
        Radius data (will be padded).
    grid_length : int or np.ndarray
        Valid point count(s).
    cutoff_radii : float or np.ndarray
        Cutoff radius (radii).
    global_nmax : int
        Global maximum grid points.
    nljz_list : np.ndarray, optional
        Quantum number list with shape [M, 4]. Columns: [n, l, j, z].
    """
    if radius_grid.ndim == 1:
        M = 1
        grid_length_arr = np.array([grid_length]) if isinstance(grid_length, int) else np.array([grid_length])
        cutoff_radii_arr = np.array([cutoff_radii]) if isinstance(cutoff_radii, float) else np.array([cutoff_radii])
    elif radius_grid.ndim == 2:
        M = radius_grid.shape[0]
        grid_length_arr = grid_length if isinstance(grid_length, np.ndarray) else np.full(M, grid_length)
        cutoff_radii_arr = cutoff_radii if isinstance(cutoff_radii, np.ndarray) else np.full(M, cutoff_radii)
    else:
        M = radius_grid.shape[0]
        grid_length_arr = grid_length if isinstance(grid_length, np.ndarray) else np.full(M, grid_length)
        cutoff_radii_arr = cutoff_radii if isinstance(cutoff_radii, np.ndarray) else np.full(M, cutoff_radii)

    if nljz_list is not None:
        group.create_dataset("nljz_list", data=nljz_list)

    group.create_dataset("cutoff_radii", data=cutoff_radii_arr)
    group.create_dataset("grid_length", data=grid_length_arr)

    padded_grid, padded_data = pad_to_global_nmax(radius_grid, radius_data, grid_length_arr, global_nmax)

    group.create_dataset("radius_grid", data=padded_grid, compression="gzip")
    group.create_dataset("radius_data", data=padded_data, compression="gzip")


def write_nonlocal_projectors(
    nonlocal_grp: h5py.Group,
    pseudo_data,
    global_nmax: int,
) -> None:
    """
    Write nonlocal projectors to HDF5 with j-split expanded.

    Parameters
    ----------
    nonlocal_grp : h5py.Group
        HDF5 group for nonlocal projectors.
    pseudo_data : PseudopotentialData
        Pseudopotential data with nonlocal projectors.
    global_nmax : int
        Global maximum grid points.
    """
    nljz_list, cutoff_radii, grid_length, radius_data = pseudo_data.get_expanded_nonlocal_data()

    if len(nljz_list) == 0:
        return

    M = len(nljz_list)
    max_grid_len = radius_data.shape[1]
    radius_grids = np.zeros((M, max_grid_len))

    idx = 0
    for proj in pseudo_data.nonlocal_projectors:
        proj_grid, proj_data = proj.get_expanded_data()
        for i in range(proj_data.shape[0]):
            n = len(proj_grid[i])
            radius_grids[idx, :n] = proj_grid[i]
            idx += 1

    write_physical_group(
        nonlocal_grp,
        radius_grids,
        radius_data,
        grid_length,
        cutoff_radii,
        global_nmax,
        nljz_list=nljz_list,
    )


def convert_to_species_h5(
    pao_dir: str | Path,
    vps_dir: str | Path,
    output_path: str | Path,
    xc_type: str = "PBE19",
) -> None:
    """
    Convert PAO + VPS files to unified species_openmx_{xc}.h5 format.

    Parameters
    ----------
    pao_dir : str or Path
        Directory containing PAO files.
    vps_dir : str or Path
        Directory containing VPS files.
    output_path : str or Path
        Output HDF5 file path (e.g., species_openmx_pbe.h5).
    xc_type : str
        XC functional type (e.g., "PBE19", "CA19").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Scanning PAO directory: {pao_dir}")
    print(f"📂 Scanning VPS directory: {vps_dir}")

    species_map, global_nmax = scan_all_species(pao_dir, vps_dir, xc_type)

    print(f"✅ Found {len(species_map)} species")
    print(f"✅ Global N_max = {global_nmax}")
    print()

    print(f"📝 Writing to {output_path}...")

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["xc_functional"] = xc_type.replace("19", "")
        h5f.attrs["source"] = "openmx"
        h5f.attrs["global_nmax"] = global_nmax
        h5f.attrs["units_length"] = "bohr"

        for species_name, paths in sorted(species_map.items()):
            element = paths["element"]

            grp = h5f.create_group(species_name)
            grp.attrs["element"] = element
            grp.attrs["species_name"] = species_name

            pao_data = parse_pao_file(paths["pao"])
            vps_data = parse_vps_file(paths["vps"])

            grp.attrs["valence_electrons"] = vps_data.metadata.valence_electrons
            grp.attrs["basis_source"] = Path(paths["pao"]).name
            grp.attrs["pseudo_source"] = vps_data.metadata.source_filename
            grp.attrs["xc_functional"] = vps_data.metadata.xc_type

            basis_grp = grp.create_group("basis")
            nljz_list = pao_data.basis.get_nljz_list()

            total_orbitals = pao_data.basis.total_orbitals
            cutoff_radii = np.full(total_orbitals, pao_data.metadata.radial_cutoff)
            grid_length = np.full(total_orbitals, pao_data.metadata.grid_num_output)

            write_physical_group(
                basis_grp,
                pao_data.basis.radius_grid,
                pao_data.basis.orbitals,
                grid_length,
                cutoff_radii,
                global_nmax,
                nljz_list=nljz_list,
            )

            val_density_grp = grp.create_group("val_density")

            if pao_data.valence_density is not None and len(pao_data.valence_density) > 0:
                cutoff = pao_data.metadata.radial_cutoff
                grid_len = len(pao_data.valence_density)

                write_physical_group(
                    val_density_grp,
                    pao_data.valence_density_grid,
                    pao_data.valence_density,
                    grid_len,
                    cutoff,
                    global_nmax,
                    nljz_list=np.array([[0, 0, 0, 1]], dtype=int),
                )

            pseudo_grp = grp.create_group("pseudopotential")

            local_grp = pseudo_grp.create_group("local")
            if (
                vps_data.pseudopotential.local_potential is not None
                and len(vps_data.pseudopotential.local_potential) > 0
            ):
                grid_len = len(vps_data.pseudopotential.local_potential)
                cutoff = vps_data.metadata.local_cutoff if vps_data.metadata.local_cutoff > 0 else 2.0

                local_part_vps = vps_data.metadata.local_part_vps
                local_component = None
                for comp in vps_data.pseudopotential.components:
                    if comp.index == local_part_vps:
                        local_component = comp
                        break

                if local_component is not None:
                    local_nljz = np.array([[local_component.n, local_component.ell, 0, 1]], dtype=int)
                else:
                    local_nljz = np.array([[0, 0, 0, 1]], dtype=int)

                write_physical_group(
                    local_grp,
                    vps_data.pseudopotential.local_potential_grid,
                    vps_data.pseudopotential.local_potential,
                    grid_len,
                    cutoff,
                    global_nmax,
                    nljz_list=local_nljz,
                )

            if vps_data.pseudopotential.nonlocal_projectors:
                nonlocal_grp = pseudo_grp.create_group("nonlocal")
                write_nonlocal_projectors(
                    nonlocal_grp,
                    vps_data.pseudopotential,
                    global_nmax,
                )

            if vps_data.core_density is not None and len(vps_data.core_density) > 0:
                core_grp = pseudo_grp.create_group("core_density")
                grid_len = len(vps_data.core_density)
                cutoff = vps_data.metadata.local_cutoff if vps_data.metadata.local_cutoff > 0 else 2.0

                write_physical_group(
                    core_grp,
                    vps_data.core_density_grid,
                    vps_data.core_density,
                    grid_len,
                    cutoff,
                    global_nmax,
                )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Done! File size: {file_size_mb:.2f} MB")
