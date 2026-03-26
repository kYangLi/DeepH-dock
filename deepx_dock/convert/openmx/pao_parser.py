"""Parser for OpenMX PAO (Pseudo Atomic Orbital) files."""

import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PAOMetadata:
    """Metadata extracted from PAO file."""

    element: str
    atomic_number: int
    valence_electrons: float
    total_electrons: float
    grid_xmin: float
    grid_xmax: float
    grid_num: int
    grid_num_output: int
    radial_cutoff: float
    max_l: int
    num_pao: int
    species_name: str
    source_filename: str = ""


@dataclass
class PAOOrbitalData:
    """Orbital data for one L channel."""

    ell: int
    mul_list: List[int]
    radius_grid: np.ndarray
    orbitals: np.ndarray

    @property
    def total_orbitals(self) -> int:
        """Total number of orbitals."""
        return sum(self.mul_list)

    def get_nljz_list(self) -> np.ndarray:
        """
        Generate nljz_list for basis orbitals.

        For OpenMX PAO:
        - n = 0 (no principal quantum number concept)
        - l = angular momentum
        - j = 0 (no SOC)
        - z = zeta index within l channel (starting from 1)

        Returns
        -------
        np.ndarray
            nljz_list with shape [total_orbitals, 4].
            Order: all s orbitals, then p, then d, etc.
        """
        nljz_list = []
        for ell, num_zeta in enumerate(self.mul_list):
            for z in range(1, num_zeta + 1):
                nljz_list.append([0, ell, 0, z])
        return np.array(nljz_list, dtype=int)


@dataclass
class PAOData:
    """Complete parsed PAO file data."""

    metadata: PAOMetadata
    basis: PAOOrbitalData
    valence_density_grid: np.ndarray
    valence_density: np.ndarray


def parse_pao_file(filepath: str | Path) -> PAOData:
    """
    Parse OpenMX PAO file.

    Parameters
    ----------
    filepath : str or Path
        Path to PAO file.

    Returns
    -------
    PAOData
        Parsed data structure.
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    metadata = _parse_pao_metadata(lines, filepath.stem, filepath.name)
    basis = _parse_pao_basis(lines, metadata)
    valence_density_grid, valence_density = _parse_valence_density(lines)

    return PAOData(
        metadata=metadata,
        basis=basis,
        valence_density_grid=valence_density_grid,
        valence_density=valence_density,
    )


def _parse_pao_metadata(lines: List[str], species_name: str, source_filename: str) -> PAOMetadata:
    """Parse metadata from PAO file."""
    data = {
        "atomic_number": 0,
        "valence_electrons": 0.0,
        "total_electrons": 0.0,
        "grid_xmin": -9.0,
        "grid_xmax": 3.2,
        "grid_num": 12000,
        "grid_num_output": 500,
        "radial_cutoff": 7.0,
        "max_l": 3,
        "num_pao": 7,
        "source_filename": source_filename,
    }

    for line in lines:
        line = line.strip()
        if line.startswith("AtomSpecies"):
            data["atomic_number"] = int(float(line.split()[1]))
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
        elif line.startswith("grid.num.output"):
            data["grid_num_output"] = int(line.split()[1])
        elif line.startswith("radial.cutoff.pao"):
            data["radial_cutoff"] = float(line.split()[1])
        elif line.startswith("maxL.pao"):
            data["max_l"] = int(line.split()[1])
        elif line.startswith("num.pao"):
            data["num_pao"] = int(line.split()[1])

    data["element"] = _extract_element_from_species(species_name)
    data["species_name"] = species_name

    return PAOMetadata(**data)


def _extract_element_from_species(species_name: str) -> str:
    """Extract element symbol from species name like 'Mo7.0' or 'Fe5.5H_pv'."""
    match = re.match(r"^([A-Z][a-z]?)", species_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract element from species name: {species_name}")


def _parse_pao_basis(lines: List[str], metadata: PAOMetadata) -> PAOOrbitalData:
    """Parse basis orbitals from PAO file."""
    max_l = metadata.max_l
    grid_num = metadata.grid_num_output

    orbitals_by_L = {}
    radius_by_L = {}
    mul_list = []

    for ell in range(max_l + 1):
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if f"<pseudo.atomic.orbitals.L={ell}" in line:
                start_idx = i + 1
            elif start_idx is not None and line.strip().startswith("pseudo.atomic.orbitals.L="):
                end_idx = i
                break
            elif start_idx is not None and ">" in line and line.strip().endswith(">"):
                end_idx = i
                break

        if start_idx is None:
            continue

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
                    row = [float(x) for x in parts]
                    data_rows.append(row)
                except ValueError:
                    continue

        if not data_rows:
            continue

        data_array = np.array(data_rows)

        num_cols = data_array.shape[1]
        num_mul = (num_cols - 2) // 2

        if num_mul == 0:
            continue

        mul_list.append(num_mul)

        log_r = data_array[:, 0]
        r = np.exp(log_r)
        radius_by_L[ell] = r
        orbitals_by_L[ell] = data_array[:, 2 : 2 + num_mul]

    total_orbitals = sum(mul_list)
    radius_grid = np.zeros((total_orbitals, grid_num))
    orbitals = np.zeros((total_orbitals, grid_num))

    idx = 0
    for ell in range(len(mul_list)):
        num_mul = mul_list[ell]
        if ell not in orbitals_by_L:
            idx += num_mul
            continue

        orbital_data = orbitals_by_L[ell]
        r = radius_by_L[ell]
        n_points = min(len(r), grid_num)

        for mul in range(num_mul):
            radius_grid[idx, :n_points] = r[:n_points]

            r_phi = orbital_data[:n_points, mul]
            orbitals[idx, :n_points] = r_phi

            idx += 1

    return PAOOrbitalData(
        ell=max_l,
        mul_list=mul_list,
        radius_grid=radius_grid,
        orbitals=orbitals,
    )


def _parse_valence_density(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse valence charge density from PAO file."""
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "<valence.charge.density" in line:
            start_idx = i + 1
        elif start_idx is not None and "valence.charge.density>" in line:
            end_idx = i
            break

    if start_idx is None:
        return np.array([]), np.array([])

    if end_idx is None:
        end_idx = len(lines)

    block_lines = lines[start_idx:end_idx]
    data_rows = []
    for line in block_lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
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
        return np.array([]), np.array([])

    data_array = np.array(data_rows)
    radius_grid = data_array[:, 1]
    density = data_array[:, 2]

    return radius_grid, density


def get_pao_nmax(filepath: str | Path) -> int:
    """Get the number of grid points in PAO file."""
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("grid.num.output"):
                return int(line.split()[1])

    return 500


def find_matching_vps(pao_filename: str, element: str, xc_type: str = "PBE19") -> str:
    """
    Find matching VPS file for a PAO file.

    Matching rules:
    - Fe5.5H.pao → Fe_PBE19H.vps (H = Hard)
    - Fe5.5S.pao → Fe_PBE19S.vps (S = Soft)
    - W_pv8.0.pao → W_PBE19_pv.vps (_pv = semi-core)
    - Mo7.0.pao → Mo_PBE19.vps (no suffix)

    Parameters
    ----------
    pao_filename : str
        PAO filename (e.g., "Fe5.5H.pao", "W_pv8.0.pao")
    element : str
        Element symbol (e.g., "Fe", "W")
    xc_type : str
        XC functional type (e.g., "PBE19", "CA19")

    Returns
    -------
    str
        Expected VPS filename.
    """
    base_name = pao_filename.replace(".pao", "")

    suffix_after_xc = ""
    suffix_before_xc = ""

    for suffix in ["H", "S"]:
        if base_name.endswith(suffix):
            suffix_after_xc = suffix
            base_name = base_name[: -len(suffix)]
            break

    for tag in ["_pv", "_sv", "_sc", "_OC"]:
        if tag in base_name:
            suffix_before_xc = tag
            break

    if suffix_before_xc and suffix_after_xc:
        return f"{element}_{xc_type}{suffix_before_xc}{suffix_after_xc}.vps"
    elif suffix_before_xc:
        return f"{element}_{xc_type}{suffix_before_xc}.vps"
    elif suffix_after_xc:
        return f"{element}_{xc_type}{suffix_after_xc}.vps"
    else:
        return f"{element}_{xc_type}.vps"
