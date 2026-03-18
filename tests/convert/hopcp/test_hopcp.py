"""
Tests for HOPCP (PETSc format) conversion
"""

import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fixtures.comparison import compare_directories


def is_petsc4py_available():
    """Check if petsc4py is installed"""
    try:
        import petsc4py

        return True
    except ImportError:
        return False


@pytest.fixture
def hopcp_data():
    """HOPCP test data"""
    test_dir = Path(__file__).parent
    return {
        "deeph_input": test_dir / "deeph.bak",
        "petsc_reference": test_dir / "petsc.bak",
    }


@pytest.mark.skipif(not is_petsc4py_available(), reason="petsc4py not installed")
def test_hopcp_from_deeph(hopcp_data, tmp_path):
    """Test DeepH to PETSc format conversion"""
    input_dir = hopcp_data["deeph_input"]
    output_dir = tmp_path / "petsc"
    reference_dir = hopcp_data["petsc_reference"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    result = subprocess.run(
        ["dock", "convert", "hopcp", "from-deeph", str(input_dir), str(output_dir), "-t", "0", "-j", "1"],
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


def test_hopcp_cli_help():
    """Test HOPCP CLI help command"""
    result = subprocess.run(
        ["dock", "convert", "hopcp", "from-deeph", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "from-deeph" in result.stdout
