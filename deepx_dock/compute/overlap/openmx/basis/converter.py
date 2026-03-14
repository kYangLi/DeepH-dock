"""
PAO to HDF5 converter.

This module provides functions to convert OpenMX PAO files to the unified
HDF5 format for use in overlap matrix calculations.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from .parser import parse_pao_file, PAORawData
from .schema import BasisSet, ElementBasis


def convert_pao_to_h5(
    pao_file: str | Path, output_file: str | Path, compute_k_space: bool = False, k_max: float = 20.0, num_k: int = 500
) -> None:
    """
    Convert a PAO file to HDF5 format.

    Parameters
    ----------
    pao_file : str or Path
        Path to the input PAO file
    output_file : str or Path
        Path to the output HDF5 file
    compute_k_space : bool, optional
        Whether to precompute k-space data, default False
    k_max : float, optional
        Maximum k value for k-space grid, default 20.0 (a.u.^-1)
    num_k : int, optional
        Number of k-space grid points, default 500

    Examples
    --------
    >>> from deepx_dock.compute.overlap.openmx.basis import convert_pao_to_h5
    >>> convert_pao_to_h5("C7.0.pao", "C.h5")
    >>>
    >>> # With k-space precomputation
    >>> convert_pao_to_h5("C7.0.pao", "C.h5", compute_k_space=True)
    """
    pao_file = Path(pao_file)
    output_file = Path(output_file)

    pao_data = parse_pao_file(pao_file)

    basis_set = _convert_pao_data_to_basis_set(pao_data)

    if compute_k_space:
        k_space = _compute_k_space(basis_set, k_max, num_k)
        basis_set.k_space = k_space

    element = ElementBasis(
        atomic_number=pao_data.atom_species,
        symbol=_get_element_symbol(pao_data.atom_species),
        valence_electrons=pao_data.valence_electrons,
        mass=_get_atomic_mass(pao_data.atom_species),
        basis_sets={basis_set.name: basis_set},
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    element.save_h5(output_file)


def convert_pao_to_element_basis(
    pao_file: str | Path, compute_k_space: bool = False, k_max: float = 20.0, num_k: int = 500
) -> ElementBasis:
    """
    Convert a PAO file to ElementBasis object.

    Parameters
    ----------
    pao_file : str or Path
        Path to the input PAO file
    compute_k_space : bool, optional
        Whether to precompute k-space data
    k_max : float, optional
        Maximum k value
    num_k : int, optional
        Number of k-space grid points

    Returns
    -------
    ElementBasis
        Converted element basis
    """
    pao_file = Path(pao_file)
    pao_data = parse_pao_file(pao_file)

    basis_set = _convert_pao_data_to_basis_set(pao_data)

    if compute_k_space:
        k_space = _compute_k_space(basis_set, k_max, num_k)
        basis_set.k_space = k_space

    return ElementBasis(
        atomic_number=pao_data.atom_species,
        symbol=_get_element_symbol(pao_data.atom_species),
        valence_electrons=pao_data.valence_electrons,
        mass=_get_atomic_mass(pao_data.atom_species),
        basis_sets={basis_set.name: basis_set},
    )


def batch_convert_pao_dir(
    pao_dir: str | Path,
    output_dir: str | Path,
    compute_k_space: bool = False,
    k_max: float = 20.0,
    num_k: int = 500,
    pattern: str = "*.pao",
) -> None:
    """
    Batch convert all PAO files in a directory.

    Parameters
    ----------
    pao_dir : str or Path
        Directory containing PAO files
    output_dir : str or Path
        Output directory for HDF5 files
    compute_k_space : bool, optional
        Whether to precompute k-space data
    k_max : float, optional
        Maximum k value
    num_k : int, optional
        Number of k-space grid points
    pattern : str, optional
        Glob pattern for PAO files, default "*.pao"

    Examples
    --------
    >>> batch_convert_pao_dir("./pao_files", "./basis", compute_k_space=True)
    """
    pao_dir = Path(pao_dir)
    output_dir = Path(output_dir)

    if not pao_dir.is_dir():
        raise NotADirectoryError(f"PAO directory not found: {pao_dir}")

    pao_files = list(pao_dir.glob(pattern))

    if not pao_files:
        raise FileNotFoundError(f"No PAO files found in {pao_dir} with pattern {pattern}")

    element_bases = {}

    for pao_file in tqdm(pao_files, desc="Converting PAO files"):
        try:
            pao_data = parse_pao_file(pao_file)
            basis_set = _convert_pao_data_to_basis_set(pao_data)

            if compute_k_space:
                k_space = _compute_k_space(basis_set, k_max, num_k)
                basis_set.k_space = k_space

            element_key = pao_data.atom_species

            if element_key not in element_bases:
                element_bases[element_key] = ElementBasis(
                    atomic_number=pao_data.atom_species,
                    symbol=_get_element_symbol(pao_data.atom_species),
                    valence_electrons=pao_data.valence_electrons,
                    mass=_get_atomic_mass(pao_data.atom_species),
                    basis_sets={},
                )

            element_bases[element_key].basis_sets[basis_set.name] = basis_set

        except Exception as e:
            print(f"Warning: Failed to convert {pao_file}: {e}")
            continue

    output_dir.mkdir(parents=True, exist_ok=True)

    for atomic_number, element in element_bases.items():
        symbol = element.symbol
        output_file = output_dir / f"{symbol}.h5"
        element.save_h5(output_file)
        print(f"Saved {symbol} with {len(element.basis_sets)} basis sets to {output_file}")


def _convert_pao_data_to_basis_set(pao_data: PAORawData) -> BasisSet:
    """Convert PAORawData to BasisSet."""
    from .parser import convert_pao_to_basis_set

    return convert_pao_to_basis_set(pao_data)


def _compute_k_space(basis_set: BasisSet, k_max: float, num_k: int):
    """
    Compute k-space radial functions using Fourier transform.

    R̃_L(k) = ∫ j_L(kr) R_L(r) r² dr

    This is a placeholder implementation. The actual Fourier transform
    will be implemented in the C++ core library for performance.
    """
    from .schema import KSpaceData

    k_grid = np.linspace(0, k_max, num_k)

    lmax = basis_set.metadata.lmax
    num_mu = basis_set.metadata.num_mu

    wf = np.zeros((lmax + 1, num_mu, num_k), dtype=np.complex128)

    r = basis_set.radial_grid.r
    dr = basis_set.radial_grid.dr

    for L in range(lmax + 1):
        for mu in range(num_mu):
            R_r = basis_set.get_radial_wf(L, mu)

            for ik, k in enumerate(k_grid):
                j_L = _spherical_bessel(L, k * r)

                integrand = R_r * j_L * r**2
                R_tilde = np.trapz(integrand * dr, r)
                wf[L, mu, ik] = R_tilde

    return KSpaceData(k_grid=k_grid, wf=wf, k_max=k_max, num_k=num_k)


def _spherical_bessel(l: int, x: np.ndarray) -> np.ndarray:
    """
    Compute spherical Bessel function j_l(x).

    This is a simple Python implementation. The C++ version will be
    more efficient and numerically stable.
    """
    from scipy.special import spherical_jn

    return spherical_jn(l, x)


def _get_element_symbol(atomic_number: int) -> str:
    """Get element symbol from atomic number."""
    symbols = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
    }
    return symbols.get(atomic_number, f"X{atomic_number}")


def _get_atomic_mass(atomic_number: int) -> float:
    """Get atomic mass from atomic number (approximate)."""
    masses = {
        1: 1.008,
        6: 12.011,
        7: 14.007,
        8: 15.999,
        11: 22.990,
        12: 24.305,
        13: 26.982,
        14: 28.086,
        15: 30.974,
        16: 32.065,
        17: 35.453,
        19: 39.098,
        20: 40.078,
        26: 55.845,
        28: 58.693,
        29: 63.546,
        30: 65.38,
        47: 107.87,
        79: 196.97,
    }
    return masses.get(atomic_number, float(atomic_number * 2))
