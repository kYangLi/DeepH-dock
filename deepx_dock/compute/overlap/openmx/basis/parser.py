"""
OpenMX PAO file parser.

This module provides functions to parse OpenMX's .pao (Pseudo Atomic Orbital)
files and convert them to the unified BasisSet format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import re
import numpy as np

from .schema import (
    BasisSet,
    BasisMetadata,
    RadialGrid,
    GridType,
)


@dataclass
class ContractionSet:
    """
    Contraction coefficients for one optimization set.

    OpenMX PAO files can contain multiple contraction sets (typically 3),
    each optimized for different molecular environments.

    Attributes
    ----------
    set_id : int
        Contraction set identifier (1, 2, or 3)
    coefficients : dict
        (L, mul) -> array of contraction coefficients for primitives 0..num_mu-1
    """

    set_id: int
    coefficients: Dict[tuple, np.ndarray]


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
        Number of primitive radial functions per angular momentum
    radial_cutoff : float
        Radial cutoff distance in Bohr
    xv : np.ndarray
        Logarithmic grid coordinates x = log(r), shape: (N,)
    rv : np.ndarray
        Radial distances r, shape: (N,)
    radial_wf : dict
        Primitive radial wave functions, L -> array of shape (num_mu, N)
    eigenvalues : dict
        Eigenvalues, L -> array of shape (num_mu,)
    valence_density : np.ndarray
        Valence electron density, shape: (N,)
    contractions : list
        List of ContractionSet objects
    num_opt_pao : int
        Number of contraction sets (number.optpao)
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
    contractions: List[ContractionSet]
    num_opt_pao: int


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
        "num_opt_pao": 0,
        "xv": [],
        "rv": [],
        "radial_wf": {},
        "eigenvalues": {},
        "valence_density": [],
        "contractions": [],
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

        elif line.startswith("number.optpao"):
            data["num_opt_pao"] = int(line.split()[1])

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

        elif re.match(r"<Contraction\.coefficients(\d+)", line):
            match = re.match(r"<Contraction\.coefficients(\d+)", line)
            set_id = int(match.group(1))
            i += 1

            num_coeff_line = lines[i].strip() if i < len(lines) else "0"
            num_coeffs = int(num_coeff_line.split()[0]) if num_coeff_line else 0
            i += 1

            coefficients = {}
            for _ in range(num_coeffs):
                if i >= len(lines):
                    break
                coeff_line = lines[i].strip()
                if coeff_line.startswith("Contraction.coefficients"):
                    i += 1
                    continue

                match = re.search(r"L\s*=\s*(\d+)\s+Mul\s*=\s*(\d+)\s+p\s*=\s*(\d+)\s+([\d.\-+eE]+)", coeff_line)
                if match:
                    L = int(match.group(1))
                    mul = int(match.group(2))
                    p = int(match.group(3))
                    c = float(match.group(4))

                    key = (L, mul)
                    if key not in coefficients:
                        num_mu = data["num_mu"] if data["num_mu"] else 15
                        coefficients[key] = np.zeros(num_mu)
                    coefficients[key][p] = c
                i += 1

            if coefficients:
                data["contractions"].append(ContractionSet(set_id=set_id, coefficients=coefficients))
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
        contractions=data["contractions"],
        num_opt_pao=data["num_opt_pao"],
    )


def apply_contraction(
    primitive_wf: Dict[int, np.ndarray], contraction: ContractionSet, num_grid: int
) -> Dict[tuple, np.ndarray]:
    """
    Apply contraction coefficients to primitive wavefunctions.

    Parameters
    ----------
    primitive_wf : dict
        L -> array of shape (num_mu, N) for primitive wavefunctions
    contraction : ContractionSet
        Contraction coefficients
    num_grid : int
        Number of grid points (expected, may differ from actual)

    Returns
    -------
    dict
        (L, mul) -> contracted wavefunction of shape (N,)
    """
    contracted = {}
    num_mu = primitive_wf[list(primitive_wf.keys())[0]].shape[0] if primitive_wf else 15
    actual_grid = primitive_wf[list(primitive_wf.keys())[0]].shape[1] if primitive_wf else num_grid

    for (L, mul), coeffs in contraction.coefficients.items():
        if L not in primitive_wf:
            continue

        wf = np.zeros(actual_grid)
        for p in range(min(len(coeffs), num_mu)):
            if abs(coeffs[p]) > 1e-15 and p < primitive_wf[L].shape[0]:
                wf += coeffs[p] * primitive_wf[L][p, :]

        contracted[(L, mul)] = wf

    return contracted


def convert_pao_to_basis_set(pao_data: PAORawData, use_contractions: bool = False) -> BasisSet:
    """
    Convert parsed PAO data to BasisSet format.

    Parameters
    ----------
    pao_data : PAORawData
        Parsed PAO data
    use_contractions : bool
        If True and contractions exist, use contracted orbitals.
        If False, use primitive orbitals (default).

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
    num_grid = pao_data.grid_num_output

    grid = RadialGrid(grid_type=GridType.LOG, num_points=num_grid, x=pao_data.xv, r=pao_data.rv, dr=dr)

    if use_contractions and pao_data.contractions:
        lmax = pao_data.lmax
        contracted_orbitals = {}
        l_mul_counter = {}

        for contraction in pao_data.contractions:
            contracted = apply_contraction(pao_data.radial_wf, contraction, num_grid)
            for (L, orig_mul), wf in contracted.items():
                if L not in l_mul_counter:
                    l_mul_counter[L] = 0
                new_mul = l_mul_counter[L]
                l_mul_counter[L] += 1
                contracted_orbitals[(L, new_mul)] = wf

        l_counts = {}
        for L, mul in contracted_orbitals.keys():
            l_counts[L] = l_counts.get(L, 0) + 1

        max_mu_per_l = max(l_counts.values()) if l_counts else pao_data.num_mu

        actual_grid = list(contracted_orbitals.values())[0].shape[0] if contracted_orbitals else num_grid

        eigenvalues_list = []
        for L in range(lmax + 1):
            num_mu_L = l_counts.get(L, 0)
            if num_mu_L > 0:
                eig_L = np.zeros(max_mu_per_l)
                if L in pao_data.eigenvalues:
                    for i, e in enumerate(pao_data.eigenvalues[L][:num_mu_L]):
                        eig_L[i] = e
                eigenvalues_list.append(eig_L)
            else:
                eigenvalues_list.append(np.zeros(max_mu_per_l))
        eigenvalues = np.array(eigenvalues_list)

        actual_rv = pao_data.rv[:actual_grid] if len(pao_data.rv) >= actual_grid else pao_data.rv
        actual_xv = pao_data.xv[:actual_grid] if len(pao_data.xv) >= actual_grid else pao_data.xv
        actual_dr = np.gradient(actual_rv)

        grid = RadialGrid(grid_type=GridType.LOG, num_points=actual_grid, x=actual_xv, r=actual_rv, dr=actual_dr)

        metadata = BasisMetadata(
            radial_cutoff=pao_data.radial_cutoff,
            lmax=lmax,
            num_mu=max_mu_per_l,
            grid_type=GridType.LOG,
            grid_num=actual_grid,
            eigenvalues=eigenvalues,
        )

        radial_wf_list = []
        for L in range(lmax + 1):
            wf_L = np.zeros((max_mu_per_l, actual_grid))
            mu_idx = 0
            for (l_key, mul_key), wf in sorted(contracted_orbitals.items()):
                if l_key == L and mu_idx < max_mu_per_l:
                    wf_L[mu_idx, : min(len(wf), actual_grid)] = wf[: min(len(wf), actual_grid)]
                    mu_idx += 1
            radial_wf_list.append(wf_L)
        radial_wf = np.array(radial_wf_list)

    else:
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
            grid_num=num_grid,
            eigenvalues=eigenvalues,
        )

        radial_wf_list = []
        for L in range(pao_data.lmax + 1):
            if L in pao_data.radial_wf:
                wf_L = pao_data.radial_wf[L]
                if wf_L.shape[1] < num_grid:
                    wf_L = np.pad(wf_L, ((0, 0), (0, num_grid - wf_L.shape[1])), mode="constant")
                radial_wf_list.append(wf_L)
            else:
                radial_wf_list.append(np.zeros((pao_data.num_mu, num_grid)))
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


def get_contracted_orbital_count(pao_data: PAORawData) -> Dict[int, int]:
    """
    Get the number of contracted orbitals per angular momentum.

    Parameters
    ----------
    pao_data : PAORawData
        Parsed PAO data

    Returns
    -------
    dict
        L -> number of contracted orbitals
    """
    if not pao_data.contractions:
        return {L: pao_data.num_mu for L in range(pao_data.lmax + 1)}

    l_counts = {}
    for contraction in pao_data.contractions:
        for L, mul in contraction.coefficients.keys():
            l_counts[L] = l_counts.get(L, 0) + 1

    for L in range(pao_data.lmax + 1):
        if L not in l_counts:
            l_counts[L] = 0

    return l_counts
