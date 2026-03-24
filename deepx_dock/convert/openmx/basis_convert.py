from pathlib import Path
import numpy as np
import h5py

from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL


def parse_openmx_pao(filepath: Path) -> dict:
    """
    Parse OpenMX PAO file.

    Parameters
    ----------
    filepath : Path
        Path to the .pao file.

    Returns
    -------
    dict
        Parsed data with structure:
        {
            'element': str,
            'atomic_number': int,
            'source': 'openmx',
            'radial_cutoff': float,
            'lmax': int,
            'mul_max': int,
            'grid_type': 'logarithmic',
            'grid_num': int,
            'r': np.ndarray,
            'x': np.ndarray,
            'orbitals': {
                L: {
                    mu: {
                        'func': np.ndarray,
                        'eigenvalue': float (optional)
                    }
                }
            }
        }
    """
    data = {
        "element": None,
        "atomic_number": None,
        "source": "openmx",
        "radial_cutoff": None,
        "lmax": None,
        "mul_max": None,
        "grid_type": "logarithmic",
        "grid_num": None,
        "r": None,
        "x": None,
        "orbitals": {},
        "eigenvalues": {},
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("AtomSpecies"):
            data["atomic_number"] = int(line_stripped.split()[1])

        elif line_stripped.startswith("grid.num.output"):
            data["grid_num"] = int(line_stripped.split()[1])

        elif line_stripped.startswith("radial.cutoff.pao"):
            data["radial_cutoff"] = float(line_stripped.split()[1])

        elif line_stripped.startswith("PAO.Lmax"):
            data["lmax"] = int(line_stripped.split()[1])

        elif line_stripped.startswith("PAO.Mul"):
            data["mul_max"] = int(line_stripped.split()[1])

        elif "pseudo.atomic.orbitals.L=" in line and line.strip().startswith("<"):
            L = int(line.split("L=")[1].split()[0])
            if L not in data["orbitals"]:
                data["orbitals"][L] = {}

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if "pseudo.atomic.orbitals.L=" in line and line.strip().startswith("<"):
            L = int(line.split("L=")[1].split()[0])

            header_idx = i + 1

            r_list = []
            func_dict = {mu: [] for mu in range(data["mul_max"])}

            for j in range(data["grid_num"]):
                data_line_idx = header_idx + j
                if data_line_idx >= len(lines):
                    break
                vals = lines[data_line_idx].split()

                if len(vals) < 2 + data["mul_max"]:
                    continue

                if data["r"] is None:
                    r_list.append(float(vals[1]))

                for mu in range(data["mul_max"]):
                    func_dict[mu].append(float(vals[2 + mu]))

            if data["r"] is None:
                data["r"] = np.array(r_list, dtype=np.float64)
                data["x"] = np.log(data["r"])

            for mu in range(data["mul_max"]):
                data["orbitals"][L][mu] = {"func": np.array(func_dict[mu], dtype=np.float64)}

        elif line_stripped.startswith("Eigenvalues"):
            eigenvalues = {}
            for j in range(i + 2, len(lines)):
                parts = lines[j].strip().split()
                if len(parts) >= 4 and parts[0] == "l" and parts[1] == "mu":
                    L = int(parts[2])
                    mu = int(parts[3])
                    eigenvalue = float(parts[4])
                    if L not in eigenvalues:
                        eigenvalues[L] = {}
                    eigenvalues[L][mu] = eigenvalue
                elif "pseudo.atomic.orbitals" in lines[j]:
                    break

            for L, mu_dict in eigenvalues.items():
                if L in data["orbitals"]:
                    for mu, eigenvalue in mu_dict.items():
                        if mu in data["orbitals"][L]:
                            data["orbitals"][L][mu]["eigenvalue"] = eigenvalue

    if data["atomic_number"] is not None:
        data["element"] = PERIODIC_TABLE_INDEX_TO_SYMBOL.get(data["atomic_number"], "Unknown")

    data.pop("eigenvalues", None)

    return data


def save_basis_to_hdf5(data: dict, output_path: Path) -> None:
    """
    Save parsed basis data to standardized HDF5 format (v0.9.16).

    Flat structure with all radial functions in a single matrix.
    Radial functions are stored in their original form (normalized from PAO).

    Parameters
    ----------
    data : dict
        Parsed basis data from parse_openmx_pao.
    output_path : Path
        Output path for the .h5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lmax = data["lmax"]

    mul_list = []
    radial_basis_list = []

    for L in range(lmax + 1):
        if L in data["orbitals"]:
            orbitals_L = data["orbitals"][L]
            mul_count = len(orbitals_L)
            mul_list.append(mul_count)
            for mu in range(mul_count):
                if mu in orbitals_L:
                    radial_basis_list.append(orbitals_L[mu]["func"])
        else:
            mul_list.append(0)

    with h5py.File(output_path, "w") as f:
        f.attrs["element"] = data["element"]
        f.attrs["basis_name"] = output_path.stem
        f.attrs["source"] = data["source"]
        f.attrs["normalized"] = True
        f.attrs["units_length"] = "bohr"

        f.create_dataset("radial_grid", data=data["r"])
        f.create_dataset("mul_list", data=np.array(mul_list, dtype=np.int32))
        f.create_dataset("radial_basis", data=np.array(radial_basis_list, dtype=np.float64))


def convert_pao_to_h5(pao_path: Path, h5_path: Path) -> None:
    """
    Convert OpenMX PAO file to standardized HDF5 format.

    Parameters
    ----------
    pao_path : Path
        Path to the .pao file.
    h5_path : Path
        Output path for the .h5 file.
    """
    data = parse_openmx_pao(pao_path)
    save_basis_to_hdf5(data, h5_path)


def parse_basis_definition(basis_def: str) -> tuple:
    """
    Parse basis definition string from OpenMX input.

    Example: 'Fe6.0H-s2p2d2' -> ('Fe6.0H', {0: 2, 1: 2, 2: 2})

    Parameters
    ----------
    basis_def : str
        Basis definition string (e.g., 'Fe6.0H-s2p2d2').

    Returns
    -------
    tuple
        (basis_name, orbital_selection)
        basis_name: str (e.g., 'Fe6.0H')
        orbital_selection: dict (e.g., {0: 2, 1: 2, 2: 2})
    """
    import re

    if "-" not in basis_def:
        return basis_def, {}

    parts = basis_def.split("-", 1)
    basis_name = parts[0]
    orbital_str = parts[1] if len(parts) > 1 else ""

    orbital_selection = {}
    for match in re.finditer(r"([spdfghijklmn])(\d+)", orbital_str.lower()):
        angmom = "spdfghijklmn".index(match.group(1))
        orbital_selection[angmom] = int(match.group(2))

    return basis_name, orbital_selection
