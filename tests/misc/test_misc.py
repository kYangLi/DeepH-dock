"""
Unit tests for utility modules (misc.py, CONSTANT.py, etc.)

These tests cover non-computational functionality such as:
- File I/O utilities
- Data format conversion
- Physical constants
- Helper functions
"""

import pytest
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB
from deepx_dock.misc import (
    dump_json_file,
    load_json_file,
    parse_hamiltonian_storage,
    require_full_hamiltonian_storage,
)


@pytest.mark.unit
def test_load_json_file(temp_output_dir):
    """Test JSON file loading"""
    json_file = temp_output_dir / "test.json"
    json_file.write_text('{"key": "value", "number": 42}')

    data = load_json_file(json_file)
    assert data == {"key": "value", "number": 42}


@pytest.mark.unit
def test_dump_json_file(temp_output_dir):
    """Test JSON file saving"""
    json_file = temp_output_dir / "test.json"
    data = {"key": "value", "number": 42}

    dump_json_file(json_file, data)
    assert json_file.exists()

    loaded = load_json_file(json_file)
    assert loaded == data


@pytest.mark.unit
@pytest.mark.parametrize(
    ("info_dict", "expected"),
    [
        ({}, None),
        ({"spinful": True, "split_hkb": False}, None),
        (
            {"spinful": True, "split_hkb": True},
            DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB,
        ),
        ({"hamiltonian_storage": "full"}, "full"),
        (
            {"hamiltonian_storage": DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB, "spinful": True},
            DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB,
        ),
    ],
)
def test_parse_hamiltonian_storage_valid(info_dict, expected):
    assert parse_hamiltonian_storage(info_dict) == expected


@pytest.mark.unit
def test_parse_hamiltonian_storage_rejects_invalid_schema():
    with pytest.raises(ValueError, match="Unknown Hamiltonian storage: invalid"):
        parse_hamiltonian_storage({"hamiltonian_storage": "invalid"})
    with pytest.raises(ValueError, match="split_hkb=true requires info.json spinful=true"):
        parse_hamiltonian_storage(
            {"hamiltonian_storage": DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB, "spinful": False}
        )
    with pytest.raises(ValueError, match="split_hkb must be true or false"):
        parse_hamiltonian_storage({"spinful": True, "split_hkb": 1})
    with pytest.raises(ValueError, match="Conflicting split-HKB metadata"):
        parse_hamiltonian_storage(
            {
                "spinful": True,
                "split_hkb": False,
                "hamiltonian_storage": DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB,
            }
        )
    assert parse_hamiltonian_storage(
        {
            "spinful": True,
            "split_hkb": True,
            "hamiltonian_storage": DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB,
        }
    ) == DEEPX_HAMILTONIAN_STORAGE_SPINLESS_PLUS_HKB


@pytest.mark.unit
def test_require_full_hamiltonian_storage(temp_output_dir):
    info_path = temp_output_dir / "info.json"
    for info_dict in ({}, {"split_hkb": False}, {"hamiltonian_storage": "full"}):
        dump_json_file(info_path, info_dict)
        require_full_hamiltonian_storage(temp_output_dir, "test operation")

    dump_json_file(
        info_path,
        {"split_hkb": True, "spinful": True},
    )
    with pytest.raises(ValueError) as error:
        require_full_hamiltonian_storage(temp_output_dir, "test operation")
    message = str(error.value)
    assert "test operation" in message
    assert "Reading hamiltonian.h5 alone would omit hkb.h5" in message


@pytest.mark.unit
def test_constant_values():
    """Test physical constants"""
    from deepx_dock.CONSTANT import HARTREE_TO_EV, BOHR_TO_ANGSTROM

    assert abs(HARTREE_TO_EV - 27.2113845) < 1e-6
    assert abs(BOHR_TO_ANGSTROM - 0.529177249) < 1e-9
