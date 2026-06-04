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


def calc_overlap(aodata1, aodata2=None, Ecut=OPENMX_DEFAULT_ECUT, kdense=OPENMX_DEFAULT_KDENSE):
    """
    Calculate an AO overlap matrix using HPRO.

    This wrapper preserves the previous DeepH-dock function name while
    delegating the numerical implementation to HPRO.
    """
    from HPRO.v2h.twocenter import calc_overlap as hpro_calc_overlap

    return hpro_calc_overlap(aodata1, aodata2=aodata2, Ecut=Ecut, kdense=kdense)


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

    poscar_path = Path(poscar_path)
    output_dir = Path(output_dir) if output_dir is not None else poscar_path.parent
    overlap_file = output_dir / "overlap.h5"

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
