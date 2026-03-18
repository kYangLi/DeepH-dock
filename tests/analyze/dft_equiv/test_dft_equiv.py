"""
Tests for DFT equivariance testing
"""

import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def dft_equiv_data():
    """DFT equivariance test data"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "poscars.clean",
        "reference": test_dir / "dft_calc.bak",
    }


def test_dft_equiv_analysis(dft_equiv_data, temp_output_dir):
    """Test DFT equivariance analysis"""
    input_dir = dft_equiv_data["input"]
    output_dir = temp_output_dir / "output"
    output_dir.mkdir()

    print(f"\n[tmp] Working directory: {temp_output_dir}")

    result = subprocess.run(
        ["dock", "analyze", "dft-equiv", "test", str(input_dir), str(output_dir), "-j", "1"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(f"[tmp] Cleaned: {temp_output_dir}")


def test_dft_equiv_cli_help():
    """Test DFT equivariance CLI help command"""
    result = subprocess.run(
        ["dock", "analyze", "dft-equiv", "test", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "test" in result.stdout
