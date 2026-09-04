"""Overlap calculation through HPRO."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


OPENMX_DEFAULT_ECUT = 1800.0
OPENMX_DEFAULT_KDENSE = 15.0
SIESTA_DEFAULT_ECUT = 100.0


def normalize_aocode(aocode: str) -> str:
    """Normalize the basis-code spelling accepted by the overlap interface."""
    normalized = aocode.strip().lower()
    if normalized not in {"siesta", "openmx"}:
        raise ValueError(f"Unsupported overlap basis code: {aocode}. Use 'siesta' or 'openmx'.")
    return normalized


def default_ecut(aocode: str) -> float:
    """Return the default HPRO Fourier cutoff for a basis interface."""
    if normalize_aocode(aocode) == "openmx":
        return OPENMX_DEFAULT_ECUT
    return SIESTA_DEFAULT_ECUT


def default_kdense(aocode: str) -> Optional[float]:
    """Return the default reciprocal radial-grid density for a basis interface."""
    if normalize_aocode(aocode) == "openmx":
        return OPENMX_DEFAULT_KDENSE
    return None


def load_aodata_from_files(poscar_path: str | Path, basis_path: str | Path, aocode: str):
    """
    Load a POSCAR structure and AO basis files through HPRO.

    For OpenMX, ``basis_path`` must contain the OpenMX 3.9 PAO files and
    ``basis_info.json``. For SIESTA, it must contain one ``.ion`` file per
    element in the POSCAR.
    """
    from HPRO.io.aodata import AOData
    from HPRO.io.struio import from_poscar

    poscar_path = Path(poscar_path)
    basis_path = Path(basis_path)
    aocode = normalize_aocode(aocode)

    with poscar_path.open() as poscar_file:
        structure = from_poscar(poscar_file)
    return AOData(structure, basis_path_root=str(basis_path), aocode=aocode)


def calc_overlap(aodata1, aodata2=None, Ecut=OPENMX_DEFAULT_ECUT, kdense=OPENMX_DEFAULT_KDENSE, **kwargs):
    """
    Calculate an AO overlap matrix using HPRO.

    This wrapper preserves the previous DeepH-dock function name while
    delegating the numerical implementation to HPRO.
    """
    from HPRO.v2h.twocenter import calc_overlap as hpro_calc_overlap

    return hpro_calc_overlap(aodata1, aodata2=aodata2, Ecut=Ecut, kdense=kdense, **kwargs)


def calc_overlap_from_files(
    poscar_path: str | Path,
    basis_path: str | Path,
    aocode: str,
    *,
    spinful: bool = False,
    ecut: Optional[float] = None,
    kdense: Optional[float] = None,
):
    """Load files and return the HPRO ``MatAO`` overlap object."""
    aocode = normalize_aocode(aocode)
    if ecut is None:
        ecut = default_ecut(aocode)
    if kdense is None:
        kdense = default_kdense(aocode)

    aodata = load_aodata_from_files(poscar_path, basis_path, aocode)
    overlaps = calc_overlap(aodata, Ecut=ecut, kdense=kdense)
    if spinful:
        overlaps.spinless_to_spinful()
    return overlaps


def save_overlap_from_files(
    poscar_path: str | Path,
    basis_path: str | Path,
    aocode: str,
    *,
    output_dir: str | Path | None = None,
    spinful: bool = False,
    ecut: Optional[float] = None,
    kdense: Optional[float] = None,
    force: bool = False,
):
    """
    Calculate and save ``POSCAR``, ``info.json``, and ``overlap.h5``.

    The matrix and metadata are written with HPRO's DeepH-format writers.
    """
    from HPRO.io.deephio import save_mat_deeph, save_structure_deeph
    from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME

    poscar_path = Path(poscar_path)
    output_dir = Path(output_dir) if output_dir is not None else poscar_path.parent
    overlap_file = output_dir / DEEPX_OVERLAP_FILENAME

    if overlap_file.exists() and not force:
        raise FileExistsError(f"{overlap_file} already exists; use force=True to overwrite it.")

    overlaps = calc_overlap_from_files(
        poscar_path,
        basis_path,
        aocode,
        spinful=spinful,
        ecut=ecut,
        kdense=kdense,
    )
    save_structure_deeph(overlaps.structure, str(output_dir), spinful=spinful)
    save_mat_deeph(str(output_dir), overlaps, "o")
    return overlaps


def calc_overlap_in_memory(
    structure_dict: dict,
    basis_path: str | Path,
    aocode: str,
    *,
    spinful: bool = False,
    ecut: Optional[float] = None,
    kdense: Optional[float] = None,
    splines_for_overlap: dict | None = None,
    num_workers: int = 1,
):
    """
    Calculate the overlap matrix from an in-memory atomic structure.

    Returns info and overlap in DeepH-format.

    Parameters
    ----------
    structure_dict : dict
        Must contain ``lattice`` (3, 3), ``atomic_numbers`` (Natom,) and
        ``frac_coords`` (Natom, 3).
    basis_path : str or Path
        Directory containing the basis files (OpenMX PAO files +
        ``basis_info.json``, or one SIESTA ``.ion`` file per element).
    aocode : str
        Basis code: "siesta" or "openmx".
    spinful : bool, optional
        If True, return the expanded overlap matrix as [[S, 0], [0, S]].
        Default: False.
    ecut : float, optional
        Energy cutoff of Fourier transforms. Default: interface-specific (1800
        for OpenMX, 100 for SIESTA).
    kdense : float, optional
        K point density of Fourier transforms. Default: 15.0 for OpenMX, None
        for SIESTA.
    splines_for_overlap : dict, optional
        Two-center spline pool for the overlap calculation. The dict is filled
        in place; passing the same dict to later calls with the same basis
        files, Ecut and kdense reuses the cached splines. Default: None (a
        fresh throwaway pool is created internally).
    num_workers : int, optional
        Number of worker threads for spline construction. Default: 1.

    Returns
    -------
    info_dict : dict
        DeepH info metadata (spinful, elements_orbital_map, ...).
    overlap_data : dict
        DeepH-format overlap (atom_pairs, chunk_boundaries, chunk_shapes,
        entries).
    """
    import numpy as np
    from HPRO.utils.structure import Structure
    from HPRO.io.aodata import AOData
    from deepx_dock.CONSTANT import BOHR_TO_ANGSTROM

    if splines_for_overlap is None:
        splines_for_overlap = {}
    elif splines_for_overlap:
        print("[info] Using cached TwoCenterIntgSplines for overlap if possible ...")

    aocode = normalize_aocode(aocode)
    if ecut is None:
        ecut = default_ecut(aocode)
    if kdense is None:
        kdense = default_kdense(aocode)

    structure = Structure(
        np.array(structure_dict["lattice"]) / BOHR_TO_ANGSTROM,
        np.array(structure_dict["atomic_numbers"]),
        np.array(structure_dict["frac_coords"]),
        atomic_positions_is_cart=False,
        efermi=None,
    )
    aodata = AOData(structure, basis_path_root=str(basis_path), aocode=aocode)
    overlaps = calc_overlap(
        aodata, Ecut=ecut, kdense=kdense, splines=splines_for_overlap, 
        num_workers=num_workers,
    )

    if spinful:
        overlaps.spinless_to_spinful()

    return matao2deepx(overlaps)


def matao2deepx(matao):
    """
    Convert an HPRO MatAO object to DeepH-format dicts.
    The order of atoms are not sorted so that the atoms of the same species may
    not be continuous.

    Parameters
    ----------
    matao : HPRO MatAO object.

    Returns
    -------
    info_dict : dict
        DeepH info metadata.
    obs_dict : dict
        DeepH-format observables (atom_pairs, chunk_boundaries, chunk_shapes,
        entries).
    """
    import numpy as np
    from HPRO.constants import hartree2ev
    from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL

    stru = matao.structure
    aodata = matao.aodata1

    # save info dict
    info_dict = {}
    info_dict["atoms_quantity"] = stru.natom
    info_dict["orbits_quantity"] = sum(aodata.norbfull_spc[spc] for spc in stru.atomic_numbers)
    info_dict["orthogonal_basis"] = False
    info_dict["spinful"] = matao.spinful
    info_dict["fermi_energy_eV"] = stru.efermi * hartree2ev if stru.efermi is not None else None
    info_dict["elements_orbital_map"] = {
        PERIODIC_TABLE_INDEX_TO_SYMBOL[number]: ls for number, ls in aodata.ls_spc.items()
    }

    atom_pairs = np.concatenate((matao.translations, matao.atom_pairs), axis=1, dtype="i8")
    shapes = np.array([mat.shape for mat in matao.mats])
    sizes = np.array([mat.size for mat in matao.mats])
    displ = np.concatenate([[0], np.cumsum(sizes)])
    flatmat = np.concatenate([mat.reshape(-1) for mat in matao.mats])

    obs_dict = {
        "atom_pairs": atom_pairs,
        "chunk_boundaries": displ,
        "chunk_shapes": shapes,
        "entries": flatmat,
    }
    return info_dict, obs_dict
