"""Numerical tests for the HPRO-backed OpenMX overlap interface."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def raw_species_dir() -> Path:
    """Allow an external OpenMX data installation without fixing a machine path."""
    dft_data = Path(os.environ.get("OPENMX_DFT_DATA19", "/home/deeph/software/calc/OpenMX/build/openmx3.9/DFT_DATA19"))
    if not dft_data.is_dir():
        dft_data = Path(__file__).resolve().parent
    if not dft_data.is_dir():
        pytest.skip("Set OPENMX_DFT_DATA19 to an OpenMX DFT_DATA19 directory")
    return dft_data


@pytest.fixture
def openmx_test_data() -> dict:
    test_dir = Path(__file__).parents[3] / "convert" / "openmx"
    return {"input": test_dir / "openmx.bak", "reference": test_dir / "deeph.bak"}


@pytest.fixture
def openmx_basis_dir(raw_species_dir: Path, openmx_test_data: dict, tmp_path: Path) -> Path:
    input_text = (openmx_test_data["input"] / "MoTe2" / "openmx_in.dat").read_text()
    declarations = input_text.split("<Definition.of.Atomic.Species", 1)[1].split("Definition.of.Atomic.Species>", 1)[0]
    basis_specs = {}
    basis_dir = tmp_path / "basis"
    basis_dir.mkdir()
    for line in declarations.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        element, basis = fields[:2]
        basis_specs[element] = basis
        filename = basis.split("-", 1)[0] + ".pao"
        source = raw_species_dir / "PAO" / filename
        assert source.is_file(), f"Missing OpenMX basis fixture: {source}"
        (basis_dir / filename).symlink_to(source)
    (basis_dir / "basis_info.json").write_text(json.dumps(basis_specs))
    return basis_dir


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


def test_overlap_single_file(openmx_basis_dir: Path, openmx_test_data: dict, tmp_path: Path) -> None:
    reference = openmx_test_data["reference"] / "MoTe2"
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "dock",
            "compute",
            "overlap",
            str(reference / "POSCAR"),
            str(openmx_basis_dir),
            "openmx",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
    is_equal, message = compare_deeph_h5(reference / "overlap.h5", output / "overlap.h5")
    assert is_equal, message


def test_overlap_batch_mode(openmx_basis_dir: Path, openmx_test_data: dict, tmp_path: Path) -> None:
    reference = openmx_test_data["reference"] / "MoTe2"
    dataset = tmp_path / "dataset"
    for name in ("data1", "data2"):
        data_dir = dataset / name
        data_dir.mkdir(parents=True)
        shutil.copy2(reference / "POSCAR", data_dir / "POSCAR")
    result = subprocess.run(
        ["dock", "compute", "overlap", str(dataset), str(openmx_basis_dir), "openmx", "-t", "0", "-j", "1"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
    for name in ("data1", "data2"):
        is_equal, message = compare_deeph_h5(reference / "overlap.h5", dataset / name / "overlap.h5")
        assert is_equal, message


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
