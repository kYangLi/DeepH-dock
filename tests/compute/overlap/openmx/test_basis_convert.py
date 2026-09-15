"""Portable tests for OpenMX PAO parsing and HDF5 basis serialization."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from deepx_dock.convert.openmx.basis_convert import parse_basis_definition, parse_openmx_pao, save_basis_to_hdf5


@pytest.fixture
def pao_file(tmp_path: Path) -> Path:
    """Provide two angular channels with distinct, exactly known radial samples."""
    path = tmp_path / "Fe5.5H.pao"
    path.write_text(
        "AtomSpecies 26\n"
        "grid.num.output 3\n"
        "radial.cutoff.pao 5.5\n"
        "PAO.Lmax 1\n"
        "PAO.Mul 2\n"
        "<pseudo.atomic.orbitals.L=0\n"
        "-2.302585093 0.1 1.0 2.0\n"
        "-1.609437912 0.2 3.0 4.0\n"
        "-0.916290732 0.4 5.0 6.0\n"
        "pseudo.atomic.orbitals.L=0>\n"
        "<pseudo.atomic.orbitals.L=1\n"
        "-2.302585093 0.1 7.0 8.0\n"
        "-1.609437912 0.2 9.0 10.0\n"
        "-0.916290732 0.4 11.0 12.0\n"
        "pseudo.atomic.orbitals.L=1>\n"
    )
    return path


def test_basis_conversion(pao_file: Path) -> None:
    data = parse_openmx_pao(pao_file)
    assert data["element"] == "Fe"
    assert data["atomic_number"] == 26
    assert data["radial_cutoff"] == 5.5
    assert data["lmax"] == 1
    assert data["mul_max"] == 2
    np.testing.assert_allclose(data["r"], [0.1, 0.2, 0.4])
    np.testing.assert_allclose(data["x"], np.log([0.1, 0.2, 0.4]))
    for ell, functions in enumerate((([1, 3, 5], [2, 4, 6]), ([7, 9, 11], [8, 10, 12]))):
        for zeta, expected in enumerate(functions):
            np.testing.assert_array_equal(data["orbitals"][ell][zeta]["func"], expected)


@pytest.mark.parametrize(
    "basis, expected_name, expected_selection",
    [
        ("Fe6.0H-s2p2d2", "Fe6.0H", {0: 2, 1: 2, 2: 2}),
        ("C7.0-s2p1d1", "C7.0", {0: 2, 1: 1, 2: 1}),
        ("Fe6.0H", "Fe6.0H", {}),
    ],
)
def test_basis_definition_parsing(basis: str, expected_name: str, expected_selection: dict) -> None:
    assert parse_basis_definition(basis) == (expected_name, expected_selection)


def test_hdf5_loading(pao_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "basis" / "Fe5.5H.h5"
    save_basis_to_hdf5(parse_openmx_pao(pao_file), output)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["element"] == "Fe"
        assert handle.attrs["units_length"] == "bohr"
        np.testing.assert_array_equal(handle["mul_list"][:], [2, 2])
        np.testing.assert_array_equal(handle["grid_length"][:], [3, 3, 3, 3])
        np.testing.assert_array_equal(handle["cutoff_radii"][:], [5.5, 5.5, 5.5, 5.5])
        np.testing.assert_allclose(handle["radius_grid"][:], np.tile([0.1, 0.2, 0.4], (4, 1)))
        np.testing.assert_array_equal(handle["radius_basis"][:], [[1, 3, 5], [2, 4, 6], [7, 9, 11], [8, 10, 12]])
