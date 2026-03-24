from pathlib import Path
import numpy as np

from deepx_dock.compute.overlap.openmx.parse_input import parse_openmx_input, openmx_input_to_structure
from deepx_dock.convert.openmx.basis_convert import convert_pao_to_h5
from deepx_dock.compute.overlap.openmx.loader import AOData_openmx
from deepx_dock.compute.overlap.overlap import calc_overlap as calc_olp
from deepx_dock.misc import dump_poscar_file
from deepx_dock.CONSTANT import (
    PERIODIC_TABLE_SYMBOL_TO_INDEX,
    PERIODIC_TABLE_INDEX_TO_SYMBOL,
    DEEPX_POSCAR_FILENAME,
    ANGSTROM_TO_BOHR,
)

from HPRO.utils.structure import Structure
from HPRO.io.deephio import save_mat_deeph


def build_atom_reorder_mapping(atomic_numbers: list) -> tuple:
    """
    Build mapping for reordering atoms by species (continuous grouping).

    DeepH format requires atoms of the same species to be continuous.
    This function creates a mapping from original indices to reordered indices.

    Parameters
    ----------
    atomic_numbers : list
        Original atomic numbers in input order.

    Returns
    -------
    tuple
        (old_to_new, new_to_old, sorted_atomic_numbers)
        - old_to_new: dict mapping original index -> new index
        - new_to_old: dict mapping new index -> original index
        - sorted_atomic_numbers: list of atomic numbers in new order
    """
    old_to_new = {}
    new_to_old = {}

    sorted_indices = sorted(range(len(atomic_numbers)), key=lambda i: atomic_numbers[i])

    for new_idx, old_idx in enumerate(sorted_indices):
        old_to_new[old_idx] = new_idx
        new_to_old[new_idx] = old_idx

    sorted_atomic_numbers = [atomic_numbers[new_to_old[i]] for i in range(len(atomic_numbers))]

    return old_to_new, new_to_old, sorted_atomic_numbers


class OpenMXOverlapCalculator:
    """
    Calculate overlap matrix from OpenMX input files.

    This class handles:
    1. Parsing OpenMX input files
    2. Converting PAO files to standardized basis.h5
    3. Computing overlap matrix using HPRO
    4. Saving results in DeepH format

    Note: Output atoms are reordered by species (continuous grouping) to comply
    with DeepH format requirements. The overlap.h5 atom indices are consistent
    with the output POSCAR.
    """

    def __init__(
        self, openmx_input: Path, basis_dir: Path, raw_basis_dir: Path = None, ecut: float = 50.0, force: bool = False
    ):
        """
        Initialize calculator.

        Parameters
        ----------
        openmx_input : Path
            Path to OpenMX input file
        basis_dir : Path
            Directory for standardized basis.h5 files
        raw_basis_dir : Path, optional
            Directory containing original PAO files
        ecut : float
            Energy cutoff for Fourier transform (Hartree)
        force : bool
            Force re-conversion of PAO files
        """
        self.openmx_input = Path(openmx_input)
        self.basis_dir = Path(basis_dir)
        self.raw_basis_dir = Path(raw_basis_dir) if raw_basis_dir else None
        self.ecut = ecut
        self.force = force

        self.output_dir = self.openmx_input.parent
        self.basis_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Execute the full workflow."""
        input_data = self._parse_input()
        basis_files, orbital_selections = self._prepare_basis_files(input_data)
        structure, aodata = self._create_structure_and_aodata(input_data, basis_files, orbital_selections)
        overlaps = self._compute_overlap(aodata)
        self._save_results(input_data, structure, overlaps)

    def _parse_input(self) -> dict:
        """Parse OpenMX input file."""
        return parse_openmx_input(self.openmx_input)

    def _prepare_basis_files(self, input_data: dict) -> tuple:
        """
        Prepare basis.h5 files (convert if needed).

        Returns
        -------
        tuple
            (basis_files, orbital_selections)
        """
        basis_files: dict[int, Path] = {}
        orbital_selections: dict[int, dict[int, int]] = {}

        for spc_na, spc_info in input_data["species_definition"].items():
            spc_nu = PERIODIC_TABLE_SYMBOL_TO_INDEX[spc_na]
            basis_name = spc_info["basis_name"]
            orbital_selection = spc_info["orbital_selection"]

            h5_file = self.basis_dir / f"{basis_name}.h5"

            if not h5_file.exists() or self.force:
                self._convert_basis_file(basis_name, h5_file)

            basis_files[spc_nu] = h5_file
            if orbital_selection:
                orbital_selections[spc_nu] = orbital_selection

        return basis_files, orbital_selections

    def _convert_basis_file(self, basis_name: str, h5_file: Path) -> None:
        """Convert PAO file to HDF5 format."""
        if self.raw_basis_dir is None:
            raise FileNotFoundError(
                f"Basis file '{h5_file.name}' not found and no PAO source provided. "
                f"Use --raw-basis-dir to specify PAO file location."
            )

        pao_file = self.raw_basis_dir / f"{basis_name}.pao"

        if not pao_file.exists():
            raise FileNotFoundError(f"PAO file not found: {pao_file}")

        convert_pao_to_h5(pao_file, h5_file)

    def _create_structure_and_aodata(
        self, input_data: dict, basis_files: dict[int, Path], orbital_selections: dict[int, dict[int, int]]
    ) -> tuple:
        """
        Create Structure and AOData objects.

        Returns
        -------
        tuple
            (Structure, AOData_openmx)
        """
        structure_data = openmx_input_to_structure(input_data)

        rprim = structure_data["lattice"] * ANGSTROM_TO_BOHR
        structure = Structure(
            rprim=rprim,
            atomic_numbers=structure_data["atomic_numbers"],
            atomic_positions=structure_data["positions_cart"] * ANGSTROM_TO_BOHR,
            atomic_positions_is_cart=True,
        )

        aodata = AOData_openmx(structure, basis_files=basis_files, orbital_selections=orbital_selections)

        return structure, aodata

    def _compute_overlap(self, aodata):
        """Compute overlap matrix using HPRO."""
        return calc_olp(aodata, Ecut=self.ecut)

    def _remap_overlap(self, overlaps, old_to_new: dict, natoms: int) -> None:
        """
        Remap atom indices in overlap object.

        Modifies overlaps.atom_pairs and overlaps.translations in-place.
        """
        for i in range(len(overlaps.atom_pairs)):
            old_i = int(overlaps.atom_pairs[i, 0])
            old_j = int(overlaps.atom_pairs[i, 1])
            overlaps.atom_pairs[i, 0] = old_to_new[old_i]
            overlaps.atom_pairs[i, 1] = old_to_new[old_j]

    def _save_results(self, input_data: dict, structure, overlaps) -> None:
        """Save overlap matrix and structure files."""
        structure_data = openmx_input_to_structure(input_data)
        original_atomic_numbers = structure_data["atomic_numbers"]
        original_positions = structure_data["positions_cart"]

        old_to_new, new_to_old, sorted_atomic_numbers = build_atom_reorder_mapping(original_atomic_numbers)

        remapped = old_to_new != {i: i for i in range(len(original_atomic_numbers))}

        sorted_positions = original_positions[[new_to_old[i] for i in range(len(original_atomic_numbers))]]

        elements_unique = []
        elements_counts = []
        current_elem = None
        for z in sorted_atomic_numbers:
            elem = PERIODIC_TABLE_INDEX_TO_SYMBOL[z]
            if elem != current_elem:
                elements_unique.append(elem)
                elements_counts.append(1)
                current_elem = elem
            else:
                elements_counts[-1] += 1

        poscar_structure = {
            "lattice": structure_data["lattice"],
            "elements_unique": elements_unique,
            "elements_counts": elements_counts,
            "atomic_numbers": sorted_atomic_numbers,
            "cart_coords": sorted_positions,
            "frac_coords": sorted_positions @ np.linalg.inv(structure_data["lattice"]),
        }
        dump_poscar_file(self.output_dir / DEEPX_POSCAR_FILENAME, poscar_structure)

        if remapped:
            self._remap_overlap(overlaps, old_to_new, len(original_atomic_numbers))

        save_mat_deeph(str(self.output_dir), overlaps, "o")
