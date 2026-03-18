"""
Tests for DeepH format standardization and conversion
"""

import pytest
import subprocess
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fixtures.comparison import compare_directories


@pytest.fixture
def deeph_data():
    """DeepH test data"""
    test_dir = Path(__file__).parent
    return {
        "legacy": test_dir / "legacy.bak",
        "updated_ref": test_dir / "updated.bak",
        "standardize_ref": test_dir / "standardize.bak",
    }


def test_deeph_upgrade(deeph_data, tmp_path):
    """Test DeepH upgrade from legacy to updated format"""
    legacy_dir = deeph_data["legacy"]
    updated_ref = deeph_data["updated_ref"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    output_dir = tmp_path / "updated"

    result = subprocess.run(
        ["dock", "convert", "deeph", "upgrade", str(legacy_dir), str(output_dir), "-t", "0", "-j", "1"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"
    assert output_dir.exists(), "Output directory not created"

    is_equal, errors = compare_directories(output_dir, updated_ref, threshold=1e-10)

    if not is_equal:
        error_msg = "\n".join(errors)
        pytest.fail(f"Output does not match reference:\n{error_msg}")

    print(f"[tmp] Cleaned: {tmp_path}")


def test_deeph_standardize(deeph_data, tmp_path):
    """Test DeepH format standardization (in-place modification)"""
    standardize_ref = deeph_data["standardize_ref"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    work_dir = tmp_path / "standardize"
    shutil.copytree(deeph_data["updated_ref"], work_dir)

    result = subprocess.run(
        ["dock", "convert", "deeph", "standardize", str(work_dir), "-t", "0", "-j", "1", "--overwrite"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    is_equal, errors = compare_directories(work_dir, standardize_ref, threshold=1e-10)

    if not is_equal:
        error_msg = "\n".join(errors)
        pytest.fail(f"Output does not match reference:\n{error_msg}")

    print(f"[tmp] Cleaned: {tmp_path}")


def test_deeph_cli_help():
    """Test DeepH CLI help command"""
    result = subprocess.run(
        ["dock", "convert", "deeph", "standardize", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "standardize" in result.stdout
