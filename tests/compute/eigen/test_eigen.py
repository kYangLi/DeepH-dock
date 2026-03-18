"""
Tests for eigenvalue calculation (band structure, DOS, Fermi energy)
"""

import pytest
import subprocess
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def eigen_data():
    """Eigenvalue test data"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "eigen.clean",
        "reference": test_dir / "eigen.bak",
    }


def test_eigen_find_fermi(eigen_data, tmp_path):
    """Test Fermi energy calculation"""
    input_dir = eigen_data["input"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    test_dir = tmp_path / "eigen"
    shutil.copytree(input_dir, test_dir)

    for structure_dir in test_dir.iterdir():
        if structure_dir.is_dir():
            result = subprocess.run(
                ["dock", "compute", "eigen", "find-fermi", str(structure_dir), "-d", "0.1", "-j", "1"],
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            assert result.returncode == 0, f"Command failed for {structure_dir.name}:\n{result.stderr}"

            fermi_file = structure_dir / "fermi_energy.json"
            assert fermi_file.exists(), f"fermi_energy.json not found in {structure_dir.name}"

    print(f"[tmp] Cleaned: {tmp_path}")


def test_eigen_calc_band(eigen_data, tmp_path):
    """Test band structure calculation"""
    input_dir = eigen_data["input"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    test_dir = tmp_path / "eigen"
    shutil.copytree(input_dir, test_dir)

    for structure_dir in test_dir.iterdir():
        if structure_dir.is_dir():
            result = subprocess.run(
                ["dock", "compute", "eigen", "calc-band", str(structure_dir), "-j", "1"],
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            assert result.returncode == 0, f"Command failed for {structure_dir.name}:\n{result.stderr}"

            band_file = structure_dir / "band.h5"
            assert band_file.exists(), f"band.h5 not found in {structure_dir.name}"

    print(f"[tmp] Cleaned: {tmp_path}")


def test_eigen_calc_dos(eigen_data, tmp_path):
    """Test DOS calculation"""
    input_dir = eigen_data["input"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    test_dir = tmp_path / "eigen"
    shutil.copytree(input_dir, test_dir)

    for structure_dir in test_dir.iterdir():
        if structure_dir.is_dir():
            result = subprocess.run(
                [
                    "dock",
                    "compute",
                    "eigen",
                    "calc-dos",
                    str(structure_dir),
                    "-d",
                    "0.03",
                    "--energy-window",
                    "-2.0",
                    "2.0",
                    "--energy-num",
                    "1000",
                    "-s",
                    "0.04",
                    "-j",
                    "1",
                ],
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            assert result.returncode == 0, f"Command failed for {structure_dir.name}:\n{result.stderr}"

            dos_file = structure_dir / "dos.h5"
            assert dos_file.exists(), f"dos.h5 not found in {structure_dir.name}"

    print(f"[tmp] Cleaned: {tmp_path}")


def test_eigen_cli_help():
    """Test eigen CLI help command"""
    result = subprocess.run(
        ["dock", "compute", "eigen", "calc-band", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "calc-band" in result.stdout
