"""
Tests for error analysis
"""

import pytest
import subprocess
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def error_data():
    """Error analysis test data"""
    test_dir = Path(__file__).parent
    return {
        "benchmark": test_dir / "benchmark.bak",
        "input": test_dir / "infer.clean",
        "reference": test_dir / "infer.bak",
    }


def test_error_entries_analysis(error_data, temp_output_dir):
    """Test entries error analysis"""
    input_dir = error_data["input"]
    benchmark_dir = error_data["benchmark"]

    print(f"\n[tmp] Working directory: {temp_output_dir}")

    test_dir = temp_output_dir / "infer"
    shutil.copytree(input_dir, test_dir)

    result = subprocess.run(
        [
            "dock",
            "analyze",
            "error",
            "entries",
            str(test_dir / "dft"),
            "-b",
            str(benchmark_dir / "dft"),
            "-t",
            "0",
            "-j",
            "1",
            "--cache-res",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    print(f"[tmp] Cleaned: {temp_output_dir}")


def test_error_orbital_analysis(error_data, temp_output_dir):
    """Test orbital-resolved error analysis"""
    input_dir = error_data["input"]
    benchmark_dir = error_data["benchmark"]

    print(f"\n[tmp] Working directory: {temp_output_dir}")

    test_dir = temp_output_dir / "infer"
    shutil.copytree(input_dir, test_dir)

    result = subprocess.run(
        [
            "dock",
            "analyze",
            "error",
            "orbital",
            str(test_dir / "dft"),
            "-b",
            str(benchmark_dir / "dft"),
            "-t",
            "0",
            "-j",
            "1",
            "--cache-res",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    print(f"[tmp] Cleaned: {temp_output_dir}")


def test_error_element_analysis(error_data, temp_output_dir):
    """Test element-resolved error analysis"""
    input_dir = error_data["input"]
    benchmark_dir = error_data["benchmark"]

    print(f"\n[tmp] Working directory: {temp_output_dir}")

    test_dir = temp_output_dir / "infer"
    shutil.copytree(input_dir, test_dir)

    result = subprocess.run(
        [
            "dock",
            "analyze",
            "error",
            "element",
            str(test_dir / "dft"),
            "-b",
            str(benchmark_dir / "dft"),
            "-t",
            "0",
            "-j",
            "1",
            "--cache-res",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

    print(f"[tmp] Cleaned: {temp_output_dir}")


def test_error_cli_help():
    """Test error analysis CLI help command"""
    result = subprocess.run(
        ["dock", "analyze", "error", "entries", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "entries" in result.stdout
