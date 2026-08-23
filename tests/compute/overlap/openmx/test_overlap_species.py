"""
Tests for OpenMX overlap calculation using species file
"""

import pytest
import subprocess
from pathlib import Path
import tempfile
import shutil
import h5py
import numpy as np


@pytest.fixture
def raw_species_dir():
    """OpenMX DFT_DATA19 directory."""
    dft_data = Path("/home/deeph/software/calc/OpenMX/build/openmx3.9/DFT_DATA19")
    if not dft_data.exists():
        pytest.skip("OpenMX DFT_DATA19 not found")
    return dft_data


@pytest.fixture
def openmx_test_data():
    """OpenMX test data directory."""
    test_dir = Path(__file__).parent.parent.parent.parent / "convert" / "openmx"
    openmx_bak = test_dir / "openmx.bak"
    deeph_bak = test_dir / "deeph.bak"
    if not openmx_bak.exists():
        pytest.skip("OpenMX test data not found")
    return {
        "input": openmx_bak,
        "reference": deeph_bak,
    }


def compare_deeph_h5(file1: Path, file2: Path, threshold: float = 1e-4) -> tuple:
    """Compare DeepH format HDF5 files with atom_pairs matching."""
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        if set(f1.keys()) != set(f2.keys()):
            return False, f"Keys differ: {set(f1.keys())} vs {set(f2.keys())}"

        if "atom_pairs" not in f1:
            for key in f1.keys():
                d1, d2 = f1[key][:], f2[key][:]
                if d1.shape != d2.shape:
                    return False, f"Shape mismatch for {key}"
                if np.issubdtype(d1.dtype, np.number):
                    diff = np.abs(d1 - d2).max()
                    if diff > threshold:
                        return False, f"Value diff for {key}: {diff:.2e}"
            return True, ""

        ref_pairs = f1["atom_pairs"][:]
        calc_pairs = f2["atom_pairs"][:]
        ref_entries = f1["entries"][:]
        calc_entries = f2["entries"][:]
        ref_bounds = f1["chunk_boundaries"][:]
        calc_bounds = f2["chunk_boundaries"][:]
        ref_shapes = f1["chunk_shapes"][:]
        calc_shapes = f2["chunk_shapes"][:]

        if len(ref_pairs) != len(calc_pairs):
            return False, f"Different number of pairs: {len(ref_pairs)} vs {len(calc_pairs)}"

        ref_lookup = {}
        for i, pair in enumerate(ref_pairs):
            key = tuple(pair)
            start, end = ref_bounds[i], ref_bounds[i + 1]
            ref_lookup[key] = (ref_entries[start:end], tuple(ref_shapes[i]))

        max_diff = 0.0
        for i, pair in enumerate(calc_pairs):
            key = tuple(pair)
            if key not in ref_lookup:
                return False, f"Pair {key} not found in reference"

            start, end = calc_bounds[i], calc_bounds[i + 1]
            calc_mat = calc_entries[start:end]
            ref_mat, ref_shape = ref_lookup[key]
            calc_shape = tuple(calc_shapes[i])

            if ref_shape != calc_shape:
                return False, f"Shape mismatch for {key}: {ref_shape} vs {calc_shape}"

            diff = np.abs(ref_mat - calc_mat).max()
            max_diff = max(max_diff, diff)
            if diff > threshold:
                return False, f"Entry diff for {key}: {diff:.2e} > {threshold:.2e}"

        return True, f"max_diff={max_diff:.2e}"


def test_overlap_single_file(raw_species_dir, openmx_test_data):
    """Test overlap calculation for single file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        input_file = output_dir / "openmx_in.dat"
        shutil.copy(openmx_test_data["input"] / "MoTe2" / "openmx_in.dat", input_file)

        species_file = tmpdir / "species_openmx_pbe.h5"

        result = subprocess.run(
            [
                "dock",
                "compute",
                "overlap",
                "openmx",
                str(input_file),
                str(species_file),
                "--raw-species-dir",
                str(raw_species_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

        ref_overlap = openmx_test_data["reference"] / "MoTe2" / "overlap.h5"
        calc_overlap = output_dir / "overlap.h5"

        assert calc_overlap.exists(), "overlap.h5 not created"

        is_equal, msg = compare_deeph_h5(ref_overlap, calc_overlap)
        assert is_equal, f"Overlap mismatch: {msg}"


def test_overlap_batch_mode(raw_species_dir, openmx_test_data):
    """Test overlap calculation in batch mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        data_dir = tmpdir / "dataset"
        data1 = data_dir / "data1"
        data2 = data_dir / "data2"
        data1.mkdir(parents=True)
        data2.mkdir(parents=True)

        shutil.copy(openmx_test_data["input"] / "MoTe2" / "openmx_in.dat", data1 / "openmx_in.dat")
        shutil.copy(openmx_test_data["input"] / "MoTe2" / "openmx_in.dat", data2 / "openmx_in.dat")

        species_file = tmpdir / "species_openmx_pbe.h5"

        result = subprocess.run(
            [
                "dock",
                "compute",
                "overlap",
                "openmx",
                str(data_dir),
                str(species_file),
                "-t",
                "0",
                "--raw-species-dir",
                str(raw_species_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed with stderr:\n{result.stderr}"

        assert (data1 / "overlap.h5").exists(), "data1/overlap.h5 not created"
        assert (data2 / "overlap.h5").exists(), "data2/overlap.h5 not created"


def test_overlap_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        ["dock", "compute", "overlap", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "BASIS_PATH" in result.stdout
    assert "--tier-num" in result.stdout
    assert "--ecut" in result.stdout
    assert "--kdense" in result.stdout
