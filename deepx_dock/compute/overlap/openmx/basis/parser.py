"""
OpenMX PAO file parser.

This module provides functions to parse OpenMX's .pao (Pseudo Atomic Orbital)
files and convert them to the unified BasisSet format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re
import numpy as np

from .schema import (
    BasisSet,
    BasisMetadata,
    RadialGrid,
    GridType,
)


@dataclass
class PAORawData:
    """
    Raw data parsed from a PAO file.

    Attributes
    ----------
    atom_species : int
        Atomic number Z
    total_electrons : float
        Total number of electrons
    valence_electrons : float
        Number of valence electrons
    grid_xmin : float
        Minimum value of logarithmic grid x = log(r)
    grid_xmax : float
        Maximum value of logarithmic grid x = log(r)
    grid_num_total : int
        Total number of grid points
    grid_num_output : int
        Number of grid points in output
    lmax : int
        Maximum angular momentum quantum number
    num_mu : int
        Number of radial functions per angular momentum
    radial_cutoff : float
        Radial cutoff distance in Bohr
    xv : np.ndarray
        Logarithmic grid coordinates x = log(r), shape: (N,)
    rv : np.ndarray
        Radial distances r, shape: (N,)
    radial_wf : dict
        Radial wave functions, L -> array of shape (num_mu, N)
    eigenvalues : dict
        Eigenvalues, L -> array of shape (num_mu,)
    valence_density : np.ndarray
        Valence electron density, shape: (N,)
    """

    atom_species: int
    total_electrons: float
    valence_electrons: float
    grid_xmin: float
    grid_xmax: float
    grid_num_total: int
    grid_num_output: int
    lmax: int
    num_mu: int
    radial_cutoff: float
    xv: np.ndarray
    rv: np.ndarray
    radial_wf: Dict[int, np.ndarray]
    eigenvalues: Dict[int, np.ndarray]
    valence_density: np.ndarray


def parse_pao_file(filepath: str | Path) -> PAORawData:
    """
    Parse an OpenMX PAO file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .pao file

    Returns
    -------
    PAORawData
        Parsed data

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is invalid

    Examples
    --------
    >>> from deepx_dock.compute.overlap.openmx.basis import parse_pao_file
    >>> pao_data = parse_pao_file("C7.0.pao")
    >>> print(f"Element: Z={pao_data.atom_species}, lmax={pao_data.lmax}")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PAO file not found: {filepath}")

    data = {
        "atom_species": None,
        "total_electrons": None,
        "valence_electrons": None,
        "grid_xmin": None,
        "grid_xmax": None,
        "grid_num_total": None,
        "grid_num_output": None,
        "lmax": None,
        "num_mu": None,
        "radial_cutoff": None,
        "xv": [],
        "rv": [],
        "radial_wf": {},
        "eigenvalues": {},
        "valence_density": [],
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("*"):
            i += 1
            continue

        if line.startswith("AtomSpecies"):
            data["atom_species"] = int(line.split()[1])

        elif line.startswith("total.electron"):
            data["total_electrons"] = float(line.split()[1])

        elif line.startswith("valence.electron"):
            data["valence_electrons"] = float(line.split()[1])

        elif line.startswith("grid.xmin"):
            data["grid_xmin"] = float(line.split()[1])

        elif line.startswith("grid.xmax"):
            data["grid_xmax"] = float(line.split()[1])

        elif line.startswith("grid.num"):
            if "output" in line:
                data["grid_num_output"] = int(line.split()[1])
            else:
                data["grid_num_total"] = int(line.split()[1])

        elif line.startswith("PAO.Lmax"):
            data["lmax"] = int(line.split()[1])

        elif line.startswith("PAO.Mul"):
            data["num_mu"] = int(line.split()[1])

        elif line.startswith("radial.cutoff.pao"):
            data["radial_cutoff"] = float(line.split()[1])

        elif line.startswith("<valence.charge.density"):
            i += 1
            if i < len(lines):
                header = lines[i].strip()
                if header.startswith("XV"):
                    i += 1

            for _ in range(data["grid_num_output"] if data["grid_num_output"] else 0):
                if i >= len(lines):
                    break
                vals = lines[i].split()
                if len(vals) >= 3:
                    try:
                        data["xv"].append(float(vals[0]))
                        data["rv"].append(float(vals[1]))
                        data["valence_density"].append(float(vals[2]))
                    except ValueError:
                        break
                i += 1
            continue

        elif line.startswith("Eigenvalues"):
            i += 1
            if i < len(lines):
                header = lines[i].strip()
                if "Lmax" in header:
                    i += 1

            for _ in range(
                (data["lmax"] + 1) * (data["num_mu"] if data["num_mu"] else 0) if data["lmax"] is not None else 0
            ):
                if i >= len(lines):
                    break
                vals = lines[i].split()
                if len(vals) >= 5:
                    try:
                        L = int(vals[2])
                        mu = int(vals[3])
                        eigenval = float(vals[4])
                        if L not in data["eigenvalues"]:
                            data["eigenvalues"][L] = []
                        data["eigenvalues"][L].append(eigenval)
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue

        elif "pseudo.atomic.orbitals.L=" in line and line.startswith("<"):
            match = re.search(r"L=(\d+)", line)
            if match:
                current_L = int(match.group(1))
                data["radial_wf"][current_L] = []
                i += 1

                if i < len(lines):
                    header = lines[i].strip()
                    i += 1

                num_mu = data["num_mu"] if data["num_mu"] else 0
                grid_num = data["grid_num_output"] if data["grid_num_output"] else 0

                rwf_columns = []
                for _ in range(grid_num):
                    if i >= len(lines):
                        break
                    vals = lines[i].split()
                    if len(vals) >= 2 + num_mu:
                        try:
                            rwf_row = [float(v) for v in vals[2 : 2 + num_mu]]
                            rwf_columns.append(rwf_row)
                        except ValueError:
                            break
                    i += 1

                if rwf_columns:
                    data["radial_wf"][current_L] = np.array(rwf_columns).T
                continue

        i += 1

    data["xv"] = np.array(data["xv"])
    data["rv"] = np.array(data["rv"])
    data["valence_density"] = np.array(data["valence_density"])

    for L in data["eigenvalues"]:
        data["eigenvalues"][L] = np.array(data["eigenvalues"][L])

    required_fields = ["atom_species", "lmax", "num_mu", "radial_cutoff", "grid_num_output"]
    for field in required_fields:
        if data[field] is None:
            raise ValueError(f"Required field '{field}' not found in PAO file")

    return PAORawData(
        atom_species=data["atom_species"],
        total_electrons=data["total_electrons"] if data["total_electrons"] else 0.0,
        valence_electrons=data["valence_electrons"] if data["valence_electrons"] else 0.0,
        grid_xmin=data["grid_xmin"] if data["grid_xmin"] else 0.0,
        grid_xmax=data["grid_xmax"] if data["grid_xmax"] else 0.0,
        grid_num_total=data["grid_num_total"] if data["grid_num_total"] else 0,
        grid_num_output=data["grid_num_output"],
        lmax=data["lmax"],
        num_mu=data["num_mu"],
        radial_cutoff=data["radial_cutoff"],
        xv=data["xv"],
        rv=data["rv"],
        radial_wf=data["radial_wf"],
        eigenvalues=data["eigenvalues"],
        valence_density=data["valence_density"],
    )


def convert_pao_to_basis_set(pao_data: PAORawData) -> BasisSet:
    """
    Convert parsed PAO data to BasisSet format.

    Parameters
    ----------
    pao_data : PAORawData
        Parsed PAO data

    Returns
    -------
    BasisSet
        Unified basis set format

    Examples
    --------
    >>> pao_data = parse_pao_file("C7.0.pao")
    >>> basis = convert_pao_to_basis_set(pao_data)
    >>> print(f"Cutoff: {basis.metadata.radial_cutoff} Bohr")
    """
    if len(pao_data.rv) == 0:
        raise ValueError("No radial grid data found")

    dr = np.gradient(pao_data.rv)

    grid = RadialGrid(grid_type=GridType.LOG, num_points=pao_data.grid_num_output, x=pao_data.xv, r=pao_data.rv, dr=dr)

    eigenvalues_list = []
    for L in range(pao_data.lmax + 1):
        if L in pao_data.eigenvalues:
            eigenvalues_list.append(pao_data.eigenvalues[L])
        else:
            eigenvalues_list.append(np.zeros(pao_data.num_mu))
    eigenvalues = np.array(eigenvalues_list)

    metadata = BasisMetadata(
        radial_cutoff=pao_data.radial_cutoff,
        lmax=pao_data.lmax,
        num_mu=pao_data.num_mu,
        grid_type=GridType.LOG,
        grid_num=pao_data.grid_num_output,
        eigenvalues=eigenvalues,
    )

    radial_wf_list = []
    for L in range(pao_data.lmax + 1):
        if L in pao_data.radial_wf:
            wf_L = pao_data.radial_wf[L]
            if wf_L.shape[1] < pao_data.grid_num_output:
                wf_L = np.pad(wf_L, ((0, 0), (0, pao_data.grid_num_output - wf_L.shape[1])), mode="constant")
            radial_wf_list.append(wf_L)
        else:
            radial_wf_list.append(np.zeros((pao_data.num_mu, pao_data.grid_num_output)))
    radial_wf = np.array(radial_wf_list)

    name = f"{pao_data.radial_cutoff:.1f}"

    return BasisSet(
        name=name,
        metadata=metadata,
        radial_grid=grid,
        radial_wf=radial_wf,
        k_space=None,
        valence_density=pao_data.valence_density if len(pao_data.valence_density) > 0 else None,
    )
