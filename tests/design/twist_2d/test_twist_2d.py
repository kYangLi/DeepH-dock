"""
Tests for twist 2D heterostructure generation
"""

import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def twist_2d_data():
    """Twist 2D test data"""
    test_dir = Path(__file__).parent
    return {
        "poscar_c": test_dir / "POSCAR-C",
        "poscar_bn": test_dir / "POSCAR-BN",
        "reference": test_dir / "POSCAR-Twisted.bak",
    }


def test_twist_2d_stack(twist_2d_data, tmp_path):
    """Test twisted 2D heterostructure generation"""
    poscar_c = twist_2d_data["poscar_c"]
    poscar_bn = twist_2d_data["poscar_bn"]
    reference = twist_2d_data["reference"]

    work_dir = tmp_path / "twist_work"
    work_dir.mkdir()

    print(f"\n[tmp] Working directory: {tmp_path}")

    result = subprocess.run(
        [
            "dock",
            "design",
            "twist-2d",
            "stack",
            str(poscar_c),
            "7,8,-8,15",
            str(poscar_bn),
            "8,7,-7,15",
            "-d",
            "3.0",
            "-z",
            "0.1",
        ],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )

    print(result.stdout)
    output_file = work_dir / "POSCAR-Twisted"

    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"
    assert output_file.exists(), f"Output file not created at {output_file}"

    print(f"[tmp] Output file: {output_file}")

    with open(output_file) as f:
        output_content = f.read()
    with open(reference) as f:
        reference_content = f.read()

    assert output_content == reference_content, "Output does not match reference"

    print(f"[tmp] Cleaned: {tmp_path}")


def test_twist_2d_cli_help():
    """Test twist-2d CLI help command"""
    result = subprocess.run(
        ["dock", "design", "twist-2d", "stack", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "stack" in result.stdout
