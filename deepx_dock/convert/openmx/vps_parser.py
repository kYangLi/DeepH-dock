"""Parser for OpenMX VPS (Pseudopotential) files."""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VPSMetadata:
    """Metadata extracted from VPS file."""

    element: str
    atomic_number: int
    valence_electrons: float
    total_electrons: float
    grid_xmin: float
    grid_xmax: float
    grid_num: int
    xc_type: str
    vps_type: str
    num_vps: int
    blochl_projector_num: int
    local_type: str
    local_cutoff: float
    local_part_vps: int
    charge_pcc_calc: bool
    source_filename: str = ""


@dataclass
class VPSComponent:
    """Single pseudopotential component info from pseudo.NandL."""

    index: int
    n: int
    ell: int
    cutoff: float
    energy: float


@dataclass
class NonlocalProjector:
    """Nonlocal projector data."""

    n: int
    ell: int
    cutoff: float
    radius_grid: np.ndarray
    radius_data: np.ndarray

    def get_nljz_list(self) -> np.ndarray:
        """
        Generate nljz_list for this projector.

        Returns
        -------
        np.ndarray
            nljz_list with shape [num_j_channels, 4].
            s orbital: 1 row (j=1/2)
            p/d/f orbitals: 2 rows (j=l-1/2, j=l+1/2)
        """
        if self.ell == 0:
            j_values = [1]
        else:
            j_values = [2 * self.ell - 1, 2 * self.ell + 1]

        nljz_list = []
        for j in j_values:
            nljz_list.append([self.n, self.ell, j, 1])

        return np.array(nljz_list, dtype=int)

    def get_expanded_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get expanded radius_grid and radius_data for j-split channels.

        OpenMX VPS stores j-split as [j_plus, j_minus] = [l+1/2, l-1/2].
        We want ascending j order: [l-1/2, l+1/2] for p/d/f.

        Returns
        -------
        radius_grid_expanded : np.ndarray
            Shape [num_j_channels, N]
        radius_data_expanded : np.ndarray
            Shape [num_j_channels, N], in ascending j order
        """
        n_grid = len(self.radius_grid)
        if self.radius_data.ndim == 1:
            radius_data_2d = self.radius_data.reshape(1, -1)
        else:
            radius_data_2d = self.radius_data

        if self.ell == 0:
            radius_grid_exp = self.radius_grid.reshape(1, -1)
            radius_data_exp = radius_data_2d[0:1, :n_grid]
            return radius_grid_exp, radius_data_exp
        else:
            num_j = radius_data_2d.shape[0]
            if num_j >= 2:
                radius_grid_exp = np.tile(self.radius_grid, (2, 1))
                radius_data_exp = np.zeros((2, n_grid))
                radius_data_exp[0, :] = radius_data_2d[1, :n_grid]
                radius_data_exp[1, :] = radius_data_2d[0, :n_grid]
                return radius_grid_exp, radius_data_exp
            else:
                radius_grid_exp = np.tile(self.radius_grid, (2, 1))
                radius_data_exp = np.tile(radius_data_2d[0, :n_grid], (2, 1))
                return radius_grid_exp, radius_data_exp


@dataclass
class PseudopotentialData:
    """Parsed pseudopotential data."""

    local_potential_grid: np.ndarray
    local_potential: np.ndarray
    nonlocal_projectors: List[NonlocalProjector] = field(default_factory=list)
    components: List[VPSComponent] = field(default_factory=list)

    def get_nonlocal_nljz_list(self) -> np.ndarray:
        """
        Generate combined nljz_list for all nonlocal projectors.

        Returns
        -------
        np.ndarray
            Combined nljz_list with shape [M_nlj, 4].
        """
        if not self.nonlocal_projectors:
            return np.array([], dtype=int).reshape(0, 4)

        nljz_lists = [proj.get_nljz_list() for proj in self.nonlocal_projectors]
        return np.vstack(nljz_lists)

    def get_expanded_nonlocal_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get expanded nonlocal projector data.

        Returns
        -------
        nljz_list : np.ndarray
            Shape [M_nlj, 4]
        cutoff_radii : np.ndarray
            Shape [M_nlj]
        grid_length : np.ndarray
            Shape [M_nlj]
        radius_data : np.ndarray
            Shape [M_nlj, N_max]
        """
        if not self.nonlocal_projectors:
            return (
                np.array([], dtype=int).reshape(0, 4),
                np.array([]),
                np.array([]),
                np.array([]).reshape(0, 0),
            )

        nljz_list = self.get_nonlocal_nljz_list()
        cutoff_radii = []
        grid_lengths = []
        radius_data_list = []
        max_grid_len = 0

        for proj in self.nonlocal_projectors:
            proj_data = proj.get_expanded_data()[1]
            for i in range(proj_data.shape[0]):
                cutoff_radii.append(proj.cutoff)
                n = proj.radius_grid.shape[0] if proj.radius_grid.ndim == 1 else proj.radius_grid.shape[-1]
                grid_lengths.append(n)
                radius_data_list.append(proj_data[i])
                max_grid_len = max(max_grid_len, len(proj_data[i]))

        M = len(cutoff_radii)
        radius_data_padded = np.zeros((M, max_grid_len))
        for i, data in enumerate(radius_data_list):
            radius_data_padded[i, : len(data)] = data

        return (
            nljz_list,
            np.array(cutoff_radii),
            np.array(grid_lengths),
            radius_data_padded,
        )


@dataclass
class VPSData:
    """Complete parsed VPS file data."""

    metadata: VPSMetadata
    pseudopotential: PseudopotentialData
    core_density_grid: Optional[np.ndarray]
    core_density: Optional[np.ndarray]


def parse_vps_file(filepath: str | Path) -> VPSData:
    """
    Parse OpenMX VPS file.

    Parameters
    ----------
    filepath : str or Path
        Path to VPS file.

    Returns
    -------
    VPSData
        Parsed data structure.
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    metadata = _parse_vps_metadata(lines, filepath.name)
    pseudopotential = _parse_pseudopotential(lines, metadata)
    core_density_grid, core_density = _parse_core_density(lines)

    return VPSData(
        metadata=metadata,
        pseudopotential=pseudopotential,
        core_density_grid=core_density_grid,
        core_density=core_density,
    )


def _parse_vps_metadata(lines: List[str], filename: str) -> VPSMetadata:
    """Parse metadata from VPS file."""
    data = {"source_filename": filename}

    for line in lines:
        line = line.strip()
        if line.startswith("AtomSpecies"):
            data["atomic_number"] = int(line.split()[1])
        elif line.startswith("valence.electron"):
            data["valence_electrons"] = float(line.split()[1])
        elif line.startswith("total.electron"):
            data["total_electrons"] = float(line.split()[1])
        elif line.startswith("grid.xmin"):
            data["grid_xmin"] = float(line.split()[1])
        elif line.startswith("grid.xmax"):
            data["grid_xmax"] = float(line.split()[1])
        elif line.startswith("grid.num") and "output" not in line:
            data["grid_num"] = int(line.split()[1])
        elif line.startswith("xc.type"):
            data["xc_type"] = line.split()[1]
        elif line.startswith("vps.type"):
            data["vps_type"] = line.split()[1]
        elif line.startswith("number.vps"):
            data["num_vps"] = int(line.split()[1])
        elif line.startswith("Blochl.projector.num"):
            data["blochl_projector_num"] = int(line.split()[1])
        elif line.startswith("local.type"):
            data["local_type"] = line.split()[1]
        elif line.startswith("local.cutoff"):
            data["local_cutoff"] = float(line.split()[1])
        elif line.startswith("local.part.vps"):
            data["local_part_vps"] = int(line.split()[1])
        elif line.startswith("charge.pcc.calc"):
            val = line.split()[1].lower()
            data["charge_pcc_calc"] = val in ["on", "yes", "true"]

    element = _extract_element_from_filename(filename)
    data["element"] = element

    if "xc_type" not in data:
        data["xc_type"] = "PBE"
    if "vps_type" not in data:
        data["vps_type"] = "MBK"
    if "num_vps" not in data:
        data["num_vps"] = 0
    if "blochl_projector_num" not in data:
        data["blochl_projector_num"] = 1
    if "local_type" not in data:
        data["local_type"] = "Polynomial"
    if "local_cutoff" not in data:
        data["local_cutoff"] = 0.0
    if "local_part_vps" not in data:
        data["local_part_vps"] = 0
    if "charge_pcc_calc" not in data:
        data["charge_pcc_calc"] = False

    return VPSMetadata(**data)


def _extract_element_from_filename(filename: str) -> str:
    """Extract element from VPS filename like 'Mo_PBE19' or 'Fe_PBE19_pv'."""
    match = re.match(r"^([A-Z][a-z]?)", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract element from filename: {filename}")


def _parse_pseudo_nandl(lines: List[str]) -> List[VPSComponent]:
    """Parse <pseudo.NandL> block."""
    components = []
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "<pseudo.NandL" in line:
            start_idx = i + 1
        elif start_idx is not None and "pseudo.NandL>" in line:
            end_idx = i
            break

    if start_idx is None:
        return components

    if end_idx is None:
        end_idx = len(lines)

    for line in lines[start_idx:end_idx]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                comp = VPSComponent(
                    index=int(parts[0]),
                    n=int(parts[1]),
                    ell=int(parts[2]),
                    cutoff=float(parts[3]),
                    energy=float(parts[4]),
                )
                components.append(comp)
            except ValueError:
                continue

    return components


def _parse_pseudopotential(lines: List[str], metadata: VPSMetadata) -> PseudopotentialData:
    """Parse pseudopotential data from VPS file."""
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "<Pseudo.Potentials" in line:
            start_idx = i + 1
        elif start_idx is not None and "Pseudo.Potentials>" in line:
            end_idx = i
            break

    if start_idx is None:
        return PseudopotentialData(
            local_potential_grid=np.array([]),
            local_potential=np.array([]),
            nonlocal_projectors=[],
            components=[],
        )

    if end_idx is None:
        end_idx = len(lines)

    components = _parse_pseudo_nandl(lines)

    block_lines = lines[start_idx:end_idx]
    data_rows = []
    for line in block_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                row = [float(x) for x in parts]
                data_rows.append(row)
            except ValueError:
                continue

    if not data_rows:
        return PseudopotentialData(
            local_potential_grid=np.array([]),
            local_potential=np.array([]),
            nonlocal_projectors=[],
            components=components,
        )

    data_array = np.array(data_rows)

    r = data_array[:, 1]

    local_idx = metadata.local_part_vps
    col_per_component = 2
    col_offset = 3

    local_col_start = col_offset + local_idx * col_per_component

    if local_col_start + 2 <= data_array.shape[1]:
        local_pot = (data_array[:, local_col_start] + data_array[:, local_col_start + 1]) / 2
    elif local_col_start + 1 <= data_array.shape[1]:
        local_pot = data_array[:, local_col_start]
    else:
        local_pot = np.zeros_like(r)

    nonlocal_projectors = []

    for comp in components:
        if comp.index == local_idx:
            continue

        col_start = col_offset + comp.index * col_per_component

        if col_start + 2 <= data_array.shape[1]:
            proj_data_j_plus = data_array[:, col_start]
            proj_data_j_minus = data_array[:, col_start + 1]

            proj_data = np.stack([proj_data_j_plus, proj_data_j_minus], axis=0)

            projector = NonlocalProjector(
                n=comp.n,
                ell=comp.ell,
                cutoff=comp.cutoff,
                radius_grid=r,
                radius_data=proj_data,
            )
            nonlocal_projectors.append(projector)

    return PseudopotentialData(
        local_potential_grid=r,
        local_potential=local_pot,
        nonlocal_projectors=nonlocal_projectors,
        components=components,
    )


def _parse_core_density(lines: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse core density (NLCC) from VPS file."""
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "<density.PCC" in line:
            start_idx = i + 1
        elif start_idx is not None and "density.PCC>" in line:
            end_idx = i
            break

    if start_idx is None:
        return None, None

    if end_idx is None:
        end_idx = len(lines)

    block_lines = lines[start_idx:end_idx]
    data_rows = []
    for line in block_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                log_r = float(parts[0])
                r = float(parts[1])
                rho = float(parts[2])
                data_rows.append([log_r, r, rho])
            except ValueError:
                continue

    if not data_rows:
        return None, None

    data_array = np.array(data_rows)
    radius_grid = data_array[:, 1]
    density = data_array[:, 2]

    return radius_grid, density


def get_vps_nmax(filepath: str | Path) -> int:
    """
    Get the number of output grid points in VPS file.

    Note: This returns grid.num.output if available, otherwise grid.num.
    The output grid is what's actually stored in the data blocks.
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        grid_num_output = None
        grid_num = None
        for line in f:
            if line.strip().startswith("grid.num.output"):
                grid_num_output = int(line.split()[1])
            elif line.strip().startswith("grid.num") and "output" not in line.strip():
                grid_num = int(line.split()[1])

        return grid_num_output if grid_num_output else (grid_num if grid_num else 500)
