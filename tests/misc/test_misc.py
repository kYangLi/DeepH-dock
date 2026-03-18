"""
Unit tests for utility modules (misc.py, CONSTANT.py, etc.)

These tests cover non-computational functionality such as:
- File I/O utilities
- Data format conversion
- Physical constants
- Helper functions
"""

import pytest
from pathlib import Path
from deepx_dock.misc import load_json_file, dump_json_file


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
def test_constant_values():
    """Test physical constants"""
    from deepx_dock.CONSTANT import HARTREE_TO_EV, BOHR_TO_ANGSTROM

    assert abs(HARTREE_TO_EV - 27.2113845) < 1e-6
    assert abs(BOHR_TO_ANGSTROM - 0.529177249) < 1e-9
