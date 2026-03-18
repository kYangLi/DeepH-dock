"""
Tests for FHI-aims to DeepH format conversion
"""

import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fixtures.comparison import compare_directories


@pytest.fixture
def fhi_aims_data():
    """FHI-aims test data"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "single_atoms_aims.bak",
        "reference": test_dir / "single_atoms_deeph.bak",
    }


def test_fhi_aims_single_atom_to_deeph(fhi_aims_data, tmp_path):
    """Test FHI-aims single atom to DeepH format conversion"""
    input_dir = fhi_aims_data["input"]
    output_dir = tmp_path / "deeph_output"
    reference_dir = fhi_aims_data["reference"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    result = subprocess.run(
        ["dock", "convert", "fhi-aims", "single-atom-to-deeph", str(input_dir), str(output_dir), "-t", "0"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"
    assert output_dir.exists(), "Output directory not created"

    is_equal, errors = compare_directories(output_dir, reference_dir, threshold=1e-10)

    if not is_equal:
        error_msg = "\n".join(errors)
        pytest.fail(f"Output does not match reference:\n{error_msg}")

    print(f"[tmp] Cleaned: {tmp_path}")


def test_fhi_aims_cli_help():
    """Test FHI-aims CLI help command"""
    result = subprocess.run(
        ["dock", "convert", "fhi-aims", "single-atom-to-deeph", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "single-atom-to-deeph" in result.stdout
