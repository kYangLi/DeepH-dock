from pathlib import Path
import numpy as np

from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX, BOHR_TO_ANGSTROM


def parse_openmx_input(filepath: Path) -> dict:
    """
    Parse OpenMX input file (.dat or openmx.in).

    Parameters
    ----------
    filepath : Path
        Path to the OpenMX input file.

    Returns
    -------
    dict
        Parsed data with structure:
        {
            'species_definition': dict,
            'atoms': list,
            'lattice': np.ndarray,
            'coordinate_unit': str
        }
    """
    filepath = Path(filepath)

    data = {"species_definition": {}, "atoms": [], "lattice": None, "coordinate_unit": "Ang"}

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if "<Definition.of.Atomic.Species" in line:
            i += 1
            while i < len(lines) and "Definition.of.Atomic.Species>" not in lines[i]:
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    species = parts[0]
                    basis_def = parts[1]
                    pseudo_pot = parts[2] if len(parts) > 2 else ""

                    from deepx_dock.convert.openmx.basis_convert import parse_basis_definition

                    basis_name, orbital_selection = parse_basis_definition(basis_def)

                    data["species_definition"][species] = {
                        "basis_name": basis_name,
                        "orbital_selection": orbital_selection,
                        "pseudo_potential": pseudo_pot,
                    }
                i += 1

        elif "Atoms.SpeciesAndCoordinates.Unit" in line:
            parts = line.split()
            if len(parts) >= 2:
                data["coordinate_unit"] = parts[1]

        elif "<Atoms.SpeciesAndCoordinates" in line:
            i += 1
            while i < len(lines) and "Atoms.SpeciesAndCoordinates>" not in lines[i]:
                parts = lines[i].strip().split()
                if len(parts) >= 5:
                    atom_id = int(parts[0])
                    species = parts[1]
                    coords = [float(parts[2]), float(parts[3]), float(parts[4])]
                    data["atoms"].append({"id": atom_id, "species": species, "coords": coords})
                i += 1

        elif "<Atoms.UnitVectors" in line:
            i += 1
            lattice = []
            while i < len(lines) and "Atoms.UnitVectors>" not in lines[i]:
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            data["lattice"] = np.array(lattice, dtype=np.float64)

        i += 1

    return data


def openmx_input_to_structure(input_data: dict) -> dict:
    """
    Convert parsed OpenMX input to structure data for DeepH format.

    Parameters
    ----------
    input_data : dict
        Parsed data from parse_openmx_input.

    Returns
    -------
    dict
        Structure data with:
        {
            'atomic_numbers': List[int],
            'positions_cart': np.ndarray (N x 3),
            'lattice': np.ndarray (3 x 3),
            'species_definition': dict
        }
    """
    lattice = input_data["lattice"].copy()
    positions = []
    atomic_numbers = []
    species_list = []

    for atom in input_data["atoms"]:
        species = atom["species"]
        coords = atom["coords"]

        atomic_numbers.append(PERIODIC_TABLE_SYMBOL_TO_INDEX[species])
        positions.append(coords)
        species_list.append(species)

    positions = np.array(positions, dtype=np.float64)

    coord_unit = input_data.get("coordinate_unit", "Ang")

    if coord_unit == "FRAC":
        positions = positions @ lattice
    elif coord_unit == "Bohr" or coord_unit == "AU":
        lattice = lattice * BOHR_TO_ANGSTROM
        positions = positions * BOHR_TO_ANGSTROM

    species_definition = {}
    for species, info in input_data["species_definition"].items():
        spc_num = PERIODIC_TABLE_SYMBOL_TO_INDEX[species]
        species_definition[spc_num] = info

    return {
        "atomic_numbers": atomic_numbers,
        "positions_cart": positions,
        "lattice": lattice,
        "species_definition": species_definition,
    }
