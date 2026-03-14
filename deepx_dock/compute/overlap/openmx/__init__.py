"""
OpenMX-style overlap matrix calculation module.

This module provides a high-performance implementation of overlap matrix calculation
using the k-space method, compatible with OpenMX's PAO basis sets.

Key Components
--------------
- basis: Basis set management and PAO file parsing
- calculator: High-level Python interface for overlap calculation
- overlap_core: Core k-space overlap calculation algorithm

Examples
--------
>>> from deepx_dock.compute.overlap.openmx import OverlapCalculator
>>>
>>> # Initialize calculator
>>> calc = OverlapCalculator(basis_database_dir="./basis/data")
>>>
>>> # Set structure
>>> positions = np.array([[0, 0, 0], [1.42, 0, 0]])  # Angstrom
>>> species = np.array([6, 6])  # Two carbon atoms
>>> calc.set_structure(positions, species)
>>>
>>> # Set basis
>>> calc.set_basis({6: "7.0"})  # Use C7.0.pao basis
>>>
>>> # Calculate overlap matrix
>>> S = calc.compute(cutoff=10.0)
"""

__version__ = "0.1.0"

from .basis import BasisSet, ElementBasis, GridType
from .basis import parse_pao_file, convert_pao_to_h5
from .calculator import OverlapCalculator

__all__ = [
    "OverlapCalculator",
    "BasisSet",
    "ElementBasis",
    "GridType",
    "parse_pao_file",
    "convert_pao_to_h5",
]
