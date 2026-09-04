"""Position matrix calculation through HPRO."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from deepx_dock.compute.overlap.overlap import (
    OPENMX_DEFAULT_ECUT, OPENMX_DEFAULT_KDENSE,
    normalize_aocode, default_ecut, default_kdense, load_aodata_from_files,
    matao2deepx, calc_overlap,
)


def calc_position(
    aodata1,
    aodata2=None,
    Ecut=OPENMX_DEFAULT_ECUT,
    kdense=OPENMX_DEFAULT_KDENSE,
    overlaps=None,
    **kwargs,
):
    """
    Calculate an AO position matrix using HPRO.
    
    Returns:
        [MatAO_x, MatAO_y, MatAO_z]: r_x, r_y and r_z components.
    """
    from HPRO.v2h.twocenter import calc_position as hpro_calc_position

    return hpro_calc_position(aodata1, aodata2=aodata2, Ecut=Ecut, kdense=kdense, overlaps=overlaps, **kwargs)


def calc_overlap_and_position_from_files(
    poscar_path: str | Path,
    basis_path: str | Path,
    aocode: str,
    *,
    spinful: bool = False,
    ecut: Optional[float] = None,
    kdense: Optional[float] = None,
):
    """Load structure files and return the HPRO overlap (``MatAO``) and position matrix (``List[MatAO]``) object."""
    aocode = normalize_aocode(aocode)
    if ecut is None:
        ecut = default_ecut(aocode)
    if kdense is None:
        kdense = default_kdense(aocode)

    aodata = load_aodata_from_files(poscar_path, basis_path, aocode)
    overlaps = calc_overlap(aodata, Ecut=ecut, kdense=kdense)
    positions = calc_position(aodata, Ecut=ecut, kdense=kdense, overlaps=overlaps)
    if spinful:
        overlaps.spinless_to_spinful()
        for mat in positions:
            mat.spinless_to_spinful()
    return overlaps, positions


def save_overlap_and_position_from_files(
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
    Calculate and save ``POSCAR``, ``info.json``, ``overlap.h5`` and ``position_matrix.h5``.

    The matrix and metadata are written with HPRO's DeepH-format writers.
    """
    from HPRO.io.deephio import save_mat_deeph, save_mats_deeph, save_structure_deeph
    from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME, DEEPX_POSITION_MATRIX_FILENAME

    poscar_path = Path(poscar_path)
    output_dir = Path(output_dir) if output_dir is not None else poscar_path.parent
    overlap_file = output_dir / DEEPX_OVERLAP_FILENAME
    position_file = output_dir / DEEPX_POSITION_MATRIX_FILENAME

    if overlap_file.exists() and not force:
        raise FileExistsError(f"{overlap_file} already exists; use force=True to overwrite it.")
    if position_file.exists() and not force:
        raise FileExistsError(f"{position_file} already exists; use force=True to overwrite it.")

    overlaps, positions = calc_overlap_and_position_from_files(
        poscar_path,
        basis_path,
        aocode,
        spinful=spinful,
        ecut=ecut,
        kdense=kdense,
    )
    save_structure_deeph(overlaps.structure, str(output_dir), spinful=spinful)
    save_mat_deeph(str(output_dir), overlaps, "o")
    save_mats_deeph(str(output_dir), positions, "p")
    return overlaps, positions


def calc_overlap_and_position_in_memory(
    structure_dict: dict,
    basis_path: str | Path,
    aocode: str,
    *,
    spinful: bool = False,
    ecut: Optional[float] = None,
    kdense: Optional[float] = None,
    splines_for_overlap: dict | None = None,
    splines_for_position: dict | None = None,
    num_workers: int = 1,
):
    """
    Calculate the overlap and position matrix from an in-memory atomic structure.

    Returns info, overlap, and position matrix in DeepH-format.

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
        If True, return the expanded overlap and position matrix as [[M, 0], [0, M]].
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
        files, Ecut and kdense reuses the cached splines. Default: None.
    splines_for_position : dict, optional
        Two-center spline pool for the position calculation, filled in place
        analogously. Must not share the dict with ``splines_for_overlap``.
        Default: None.
    num_workers : int, optional
        Number of worker threads for spline construction. Default: 1.

    Returns
    -------
    info_dict : dict
        DeepH info metadata (spinful, elements_orbital_map, ...).
    overlap_data : dict
        DeepH-format overlap (atom_pairs, chunk_boundaries, chunk_shapes,
        entries: shape (E,)).
    position_data : dict
        DeepH-format position matrix (atom_pairs, chunk_boundaries, chunk_shapes,
        entries: shape (3, E)).
    """
    import numpy as np
    from HPRO.utils.structure import Structure
    from HPRO.io.aodata import AOData
    from deepx_dock.CONSTANT import BOHR_TO_ANGSTROM

    if splines_for_overlap is None:
        splines_for_overlap = {}
    elif splines_for_overlap:
        print("[info] Using cached TwoCenterIntgSplines for overlap if possible ...")

    if splines_for_position is None:
        splines_for_position = {}
    elif splines_for_position:
        print("[info] Using cached PositionTwoCenter for position if possible ...")

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
    positions = calc_position(
        aodata, Ecut=ecut, kdense=kdense, splines=splines_for_position,
        num_workers=num_workers, overlaps=overlaps,
    )

    if spinful:
        overlaps.spinless_to_spinful()
        for mat in positions:
            mat.spinless_to_spinful()

    info_dict, overlap_dict = matao2deepx(overlaps)
    position_entries = []
    for mat in positions:
        _, pos_i = matao2deepx(mat)
        same_atom_pairs = np.allclose(pos_i["atom_pairs"], overlap_dict["atom_pairs"])
        assert same_atom_pairs, "the atom pairs of overlap and three components of position matrix are not the same!"
        position_entries.append(pos_i["entries"] * BOHR_TO_ANGSTROM)
    position_dict = {
        "atom_pairs": overlap_dict["atom_pairs"],
        "chunk_boundaries": overlap_dict["chunk_boundaries"],
        "chunk_shapes": overlap_dict["chunk_shapes"],
        "entries": np.stack(position_entries, axis=0),
    }

    return info_dict, overlap_dict, position_dict
