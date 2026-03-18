"""
Tests for OpenMX to DeepH format conversion
"""

import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fixtures.comparison import compare_directories


@pytest.fixture
def openmx_data():
    """OpenMX test data"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "openmx.bak",
        "reference": test_dir / "deeph.bak",
    }


def test_openmx_to_deeph_conversion(openmx_data, temp_output_dir):
    """Test OpenMX to DeepH format conversion"""
    input_dir = openmx_data["input"]
    output_dir = temp_output_dir / "deeph_output"
    reference_dir = openmx_data["reference"]

    print(f"\n[tmp] Working directory: {temp_output_dir}")

    result = subprocess.run(
        ["dock", "convert", "openmx", "to-deeph", str(input_dir), str(output_dir), "-t", "0", "-j", "1"],
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

    print(f"[tmp] Cleaned: {temp_output_dir}")


def test_openmx_cli_help():
    """Test OpenMX CLI help command"""
    result = subprocess.run(
        ["dock", "convert", "openmx", "to-deeph", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "to-deeph" in result.stdout
