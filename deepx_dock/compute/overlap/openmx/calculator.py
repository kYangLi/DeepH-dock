from pathlib import Path
import warnings
import json
import numpy as np
import h5py

from deepx_dock.compute.overlap.openmx.parse_input import parse_openmx_input, openmx_input_to_structure
from deepx_dock.compute.overlap.openmx.loader import AOData_from_species
from deepx_dock.compute.overlap.overlap import calc_overlap as calc_olp
from deepx_dock.misc import dump_poscar_file
from deepx_dock.CONSTANT import (
    PERIODIC_TABLE_SYMBOL_TO_INDEX,
    PERIODIC_TABLE_INDEX_TO_SYMBOL,
    DEEPX_POSCAR_FILENAME,
    ANGSTROM_TO_BOHR,
)

from HPRO.utils.structure import Structure
from HPRO.utils.misc import atom_number2name, unique_nosort

OPENMX_DEFAULT_ECUT = 1800.0
OPENMX_DEFAULT_KDENSE = 15.0
OPENMX_DEFAULT_RDENSE = 100.0


def build_atom_reorder_mapping(structure: Structure) -> tuple:
    """
    Build atom reordering mapping for POSCAR format.

    POSCAR requires all atoms of the same element to be adjacent.
    This function generates indices to reorder atoms accordingly,
    preserving the order of first occurrence.

    Parameters
    ----------
    structure : Structure
        HPRO Structure object.

    Returns
    -------
    iatm_argsort : np.ndarray
        Indices such that atomic_numbers[iatm_argsort] is properly grouped.
        sorted_atoms = original_atoms[iatm_argsort]
    species_uniq : np.ndarray
        Unique atomic numbers in order of first occurrence.
    """
    species_uniq = unique_nosort(structure.atomic_numbers)
    indices = []
    for spc in species_uniq:
        eq = np.where(structure.atomic_numbers == spc)[0]
        indices.extend(eq)

    return np.array(indices, dtype=np.int64), species_uniq


def save_overlap_deeph(
    savedir: str,
    matao,
    iatm_argsort: np.ndarray,
    spinful: bool = False,
) -> None:
    """
    Save overlap matrix in DeepH format with correct atom ordering.

    This function replaces HPRO's save_mat_deeph to fix the sort_atoms bug
    that doesn't properly group atoms by species.

    Parameters
    ----------
    savedir : str
        Output directory.
    matao : MatAO
        Overlap matrix object from calc_overlap.
    iatm_argsort : np.ndarray
        Indices for reordering atoms (from build_atom_reorder_mapping).
    spinful : bool
        Whether the calculation includes spin degree of freedom.
        This is determined from OpenMX input file (SOC or spin polarization).
    """
    import os

    os.makedirs(savedir, exist_ok=True)

    stru = matao.structure
    aodata = matao.aodata1

    info_dict = {}
    info_dict["atoms_quantity"] = stru.natom
    info_dict["orbits_quantity"] = sum(aodata.norbfull_spc[spc] for spc in stru.atomic_numbers)
    info_dict["orthogonal_basis"] = False
    info_dict["spinful"] = spinful
    info_dict["fermi_energy_eV"] = stru.efermi * 0.0
    info_dict["elements_orbital_map"] = {atom_number2name([number])[0]: ls for number, ls in aodata.ls_spc.items()}

    with open(f"{savedir}/info.json", "w") as f:
        json.dump(info_dict, f)

    mapatm = np.argsort(iatm_argsort)

    atom_pairs = np.concatenate(
        (matao.translations, mapatm[matao.atom_pairs]),
        axis=1,
        dtype="i8",
    )
    displ = np.empty(matao.npairs + 1, dtype="i8")
    shapes = np.empty((matao.npairs, 2), dtype="i8")
    displ[0] = 0
    flatmat = []
    for ipair in range(matao.npairs):
        mat = matao.mats[ipair]
        displ[ipair + 1] = displ[ipair] + mat.size
        shapes[ipair, :] = mat.shape
        flatmat.append(mat.reshape(-1))
    flatmat = np.concatenate(flatmat)

    with h5py.File(f"{savedir}/overlap.h5", "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_boundaries", data=displ)
        f.create_dataset("chunk_shapes", data=shapes)
        f.create_dataset("entries", data=flatmat)


class OpenMXOverlapCalculator:
    """
    Calculate overlap matrix from OpenMX input files.

    This class handles:
    1. Parsing OpenMX input files
    2. Preparing species_openmx_{xc}.h5 (generate if needed)
    3. Computing overlap matrix using HPRO
    4. Saving results in DeepH format

    Note: Atoms are reordered to ensure same-element atoms are adjacent,
    which is required by POSCAR format. The order of first occurrence
    is preserved.
    """

    def __init__(
        self,
        openmx_input: Path,
        species_file: Path,
        raw_species_dir: Path = None,
        ecut: float = OPENMX_DEFAULT_ECUT,
        kdense: float = OPENMX_DEFAULT_KDENSE,
        rdense: float = OPENMX_DEFAULT_RDENSE,
        force: bool = False,
    ):
        """
        Initialize the overlap calculator.

        Parameters
        ----------
        openmx_input : Path
            Path to OpenMX input file (e.g., openmx_in.dat).
        species_file : Path
            Path to species_openmx_{xc}.h5 file (required).
            If the file does not exist, raw_species_dir must be provided.
        raw_species_dir : Path, optional
            Directory containing PAO/ and VPS/ subdirectories.
            Used to generate species file if it doesn't exist.
            Expected structure:
                raw_species_dir/
                ├── PAO/
                │   ├── Mo7.0.pao
                │   └── Te7.0.pao
                └── VPS/
                    ├── Mo_PBE19.vps
                    └── Te_PBE19.vps
        ecut : float
            Energy cutoff (Hartree). Default: 1800.0
        kdense : float
            k-space grid density. Default: 15.0
        rdense : float
            r-space grid density. Default: 100.0
        force : bool
            Force regenerate species file. Default: False
        """
        self.openmx_input = Path(openmx_input)
        self.species_file = Path(species_file)
        self.raw_species_dir = Path(raw_species_dir) if raw_species_dir else None
        self.ecut = ecut
        self.kdense = kdense
        self.rdense = rdense
        self.force = force

        self.output_dir = self.openmx_input.parent

        self._validate_grid_params()

    def _validate_grid_params(self) -> None:
        """Validate grid parameters and warn if k/r point counts mismatch."""
        kmax = np.sqrt(2 * self.ecut)
        expected_grid_nq = kmax * self.kdense

        typical_rcut = 9.0
        expected_grid_nr = typical_rcut * self.rdense

        ratio = expected_grid_nr / expected_grid_nq

        if ratio > 2.0 or ratio < 0.5:
            warnings.warn(
                f"Grid size mismatch: expected ~{expected_grid_nr:.0f} r-points "
                f"vs ~{expected_grid_nq:.0f} k-points (ratio={ratio:.2f}). "
                f"Consider adjusting kdense/rdense for similar grid sizes. "
                f"OpenMX uses equal NumGridK and NumGridR (typically 900).",
                UserWarning,
            )

    def run(self) -> None:
        """Execute the full workflow."""
        input_data = self._parse_input()
        self._prepare_species_file()
        species_names, orbital_selections = self._extract_species_info(input_data)
        structure, aodata, spinful = self._create_structure_and_aodata(input_data, species_names, orbital_selections)
        overlaps = self._compute_overlap(aodata)
        self._save_results(input_data, structure, overlaps, spinful)

    def _parse_input(self) -> dict:
        """Parse OpenMX input file."""
        return parse_openmx_input(self.openmx_input)

    def _prepare_species_file(self) -> None:
        """Prepare species_openmx_{xc}.h5 (generate if needed)."""
        if self.species_file.exists() and not self.force:
            return

        if self.raw_species_dir is None:
            raise FileNotFoundError(
                f"Species file '{self.species_file}' not found. "
                f"Use --raw-species-dir to specify PAO/VPS source directories for auto-generation."
            )

        pao_dir = self.raw_species_dir / "PAO"
        vps_dir = self.raw_species_dir / "VPS"

        if not pao_dir.exists():
            raise FileNotFoundError(f"PAO directory not found: {pao_dir}")
        if not vps_dir.exists():
            raise FileNotFoundError(f"VPS directory not found: {vps_dir}")

        from deepx_dock.convert.openmx.species_convert import convert_to_species_h5

        self.species_file.parent.mkdir(parents=True, exist_ok=True)
        convert_to_species_h5(pao_dir, vps_dir, self.species_file)

    def _extract_species_info(self, input_data: dict) -> tuple:
        """
        Extract species names and orbital selections from input data.

        Returns
        -------
        tuple
            (species_names, orbital_selections)
            species_names: {atomic_number: "Mo7.0"}
            orbital_selections: {atomic_number: {L: num_mu}}
        """
        species_names: dict[int, str] = {}
        orbital_selections: dict[int, dict[int, int]] = {}

        for spc_na, spc_info in input_data["species_definition"].items():
            spc_nu = PERIODIC_TABLE_SYMBOL_TO_INDEX[spc_na]
            basis_name = spc_info["basis_name"]
            orbital_selection = spc_info["orbital_selection"]

            species_names[spc_nu] = basis_name
            if orbital_selection:
                orbital_selections[spc_nu] = orbital_selection

        return species_names, orbital_selections

    def _create_structure_and_aodata(
        self, input_data: dict, species_names: dict[int, str], orbital_selections: dict[int, dict[int, int]]
    ) -> tuple:
        """
        Create Structure and AOData objects.

        Returns
        -------
        tuple
            (Structure, AOData_from_species, spinful)
        """
        structure_data = openmx_input_to_structure(input_data)

        rprim = structure_data["lattice"] * ANGSTROM_TO_BOHR
        structure = Structure(
            rprim=rprim,
            atomic_numbers=structure_data["atomic_numbers"],
            atomic_positions=structure_data["positions_cart"] * ANGSTROM_TO_BOHR,
            atomic_positions_is_cart=True,
        )

        soc = input_data.get("spin_orbit_coupling", False)
        spin_pol = input_data.get("spin_polarization", "Off")
        spinful = soc or (spin_pol in ["On", "NC"])

        aodata = AOData_from_species(
            structure,
            species_file=self.species_file,
            species_names=species_names,
            orbital_selections=orbital_selections,
            rdense=self.rdense,
            spinful=False,
        )

        return structure, aodata, spinful

    def _compute_overlap(self, aodata):
        """Compute overlap matrix using HPRO."""
        return calc_olp(aodata, Ecut=self.ecut, kdense=self.kdense)

    def _save_results(self, input_data: dict, structure, overlaps, spinful: bool) -> None:
        """
        Save overlap matrix and structure files.

        Atoms are reordered to ensure same-element atoms are adjacent,
        which is required by POSCAR format.
        """
        structure_data = openmx_input_to_structure(input_data)
        original_atomic_numbers = np.array(structure_data["atomic_numbers"])
        original_positions = structure_data["positions_cart"]

        iatm_argsort, species_uniq = build_atom_reorder_mapping(structure)

        sorted_atomic_numbers = original_atomic_numbers[iatm_argsort]
        sorted_positions = original_positions[iatm_argsort]

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
            "atomic_numbers": sorted_atomic_numbers.tolist(),
            "cart_coords": sorted_positions,
            "frac_coords": sorted_positions @ np.linalg.inv(structure_data["lattice"]),
        }
        dump_poscar_file(self.output_dir / DEEPX_POSCAR_FILENAME, poscar_structure)

        save_overlap_deeph(str(self.output_dir), overlaps, iatm_argsort, spinful)
