"""
Tests for dataset analysis
"""

import pytest
import subprocess
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def dataset_data():
    """Dataset analysis test data"""
    test_dir = Path(__file__).parent
    return {
        "input_clean": test_dir / "inputs.clean",
        "reference": test_dir / "inputs.bak",
    }


def test_dataset_edge_analysis(dataset_data, tmp_path):
    """Test edge statistic analysis"""
    input_clean = dataset_data["input_clean"]
    reference = dataset_data["reference"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    work_dir = tmp_path / "inputs"
    shutil.copytree(input_clean, work_dir)

    result = subprocess.run(
        ["dock", "analyze", "dataset", "edge", str(work_dir), "-t", "0", "-j", "1"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    assert (work_dir / "edge_statistic.h5").exists()
    assert (work_dir / "edge_statistic.png").exists()

    print(f"[tmp] Cleaned: {tmp_path}")


def test_dataset_split(dataset_data, tmp_path):
    """Test dataset split"""
    input_clean = dataset_data["input_clean"]

    print(f"\n[tmp] Working directory: {tmp_path}")

    work_dir = tmp_path / "inputs"
    shutil.copytree(input_clean, work_dir)

    result = subprocess.run(
        ["dock", "analyze", "dataset", "split", str(work_dir), "-t", "0", "-j", "1"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    split_file = tmp_path / "dataset_split.json"
    assert split_file.exists(), "dataset_split.json not created"

    print(f"[tmp] Cleaned: {tmp_path}")


def test_dataset_cli_help():
    """Test dataset analysis CLI help command"""
    result = subprocess.run(
        ["dock", "analyze", "dataset", "edge", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "edge" in result.stdout
