"""
Tests for eigenvalue calculation (band structure, DOS, Fermi energy)
"""

import json
import pytest
import subprocess
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.linalg import eigvalsh

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj
from split_hkb_test_utils import expand_overlap_to_spinful, make_split_hkb, read_matrix_data


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


def test_spinless_plus_hkb_reconstruction(eigen_data, tmp_path):
    source_dir = eigen_data["input"] / "Bi2Se3_SOC"
    split_dir = tmp_path / "Bi2Se3_SOC_split"
    base_data, hkb_data, info = make_split_hkb(source_dir, split_dir)

    legacy = HamiltonianObj(source_dir)
    split = HamiltonianObj(split_dir)
    np.testing.assert_allclose(split.HR, legacy.HR, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(split.SR, legacy.SR, rtol=0.0, atol=1e-14)

    base_obj = AOMatrixObj(split_dir)
    hkb_obj = AOMatrixObj(split_dir, matrix_type="hkb")
    assert not base_obj.stored_spinful
    assert hkb_obj.stored_spinful
    n_orbit = base_obj.orbits_quantity
    reconstructed = hkb_obj.mats.copy()
    reconstructed[:, :n_orbit, :n_orbit] += base_obj.mats
    reconstructed[:, n_orbit:, n_orbit:] += base_obj.mats
    np.testing.assert_allclose(reconstructed, legacy.HR, rtol=0.0, atol=1e-12)

    supplied_base = AOMatrixObj(split_dir, mats=base_obj.mats)
    supplied_hkb = AOMatrixObj(split_dir, matrix_type="hkb", mats=hkb_obj.mats)
    assert supplied_base.stored_spinful is False
    assert supplied_hkb.stored_spinful is True

    for kpoint in (np.zeros(3), np.array([0.137, 0.231, 0.0])):
        legacy_sk, legacy_hk = legacy.Sk_and_Hk(kpoint)
        split_sk, split_hk = split.Sk_and_Hk(kpoint)
        np.testing.assert_allclose(split_hk, legacy_hk, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(split_sk, legacy_sk, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(
            eigvalsh(split_hk, split_sk), eigvalsh(legacy_hk, legacy_sk), rtol=0.0, atol=1e-10
        )

    structure_dict = {
        "lattice": split.lattice,
        "atomic_symbols": split.elements,
        "frac_coords": split.frac_coords,
    }
    overlap_data = read_matrix_data(split_dir / "overlap.h5")
    in_memory = HamiltonianObj.from_data(
        structure_dict, info, base_data, overlap_data, hkb_data=hkb_data
    )
    np.testing.assert_allclose(in_memory.HR, legacy.HR, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(in_memory.SR, legacy.SR, rtol=0.0, atol=1e-14)

    legacy_alias_info = dict(info)
    legacy_alias_info.pop("split_hkb")
    legacy_alias_info["hamiltonian_storage"] = "spinless_plus_hkb"
    with (split_dir / "info.json").open("w") as file:
        json.dump(legacy_alias_info, file)
    legacy_alias = HamiltonianObj(split_dir)
    np.testing.assert_allclose(legacy_alias.HR, legacy.HR, rtol=0.0, atol=1e-12)


def test_spinless_plus_hkb_requires_explicit_valid_metadata(eigen_data, tmp_path):
    source_dir = eigen_data["input"] / "Bi2Se3_SOC"
    legacy_with_hkb = tmp_path / "legacy_with_hkb"
    shutil.copytree(source_dir, legacy_with_hkb)
    shutil.copy2(source_dir / "hamiltonian.h5", legacy_with_hkb / "hkb.h5")
    np.testing.assert_allclose(HamiltonianObj(legacy_with_hkb).HR, HamiltonianObj(source_dir).HR)

    legacy = HamiltonianObj(source_dir)
    with (source_dir / "info.json").open() as file:
        legacy_info = json.load(file)
    legacy_structure = {
        "lattice": legacy.lattice,
        "atomic_symbols": legacy.elements,
        "frac_coords": legacy.frac_coords,
    }
    legacy_hamiltonian = read_matrix_data(source_dir / "hamiltonian.h5")
    legacy_overlap = read_matrix_data(source_dir / "overlap.h5")
    with pytest.raises(ValueError, match="hkb_data requires split_hkb=true"):
        HamiltonianObj.from_data(
            legacy_structure,
            legacy_info,
            legacy_hamiltonian,
            legacy_overlap,
            hkb_data=legacy_hamiltonian,
        )

    split_dir = tmp_path / "invalid_split"
    make_split_hkb(source_dir, split_dir)
    (split_dir / "hkb.h5").unlink()
    with pytest.raises(FileNotFoundError):
        HamiltonianObj(split_dir)

    with (split_dir / "info.json").open() as file:
        info = json.load(file)
    info["hamiltonian_storage"] = "unknown"
    with (split_dir / "info.json").open("w") as file:
        json.dump(info, file)
    with pytest.raises(ValueError, match="Unknown Hamiltonian storage"):
        HamiltonianObj(split_dir)


def test_legacy_full_accepts_scalar_and_spin_expanded_overlap(eigen_data, tmp_path):
    source_dir = eigen_data["input"] / "Bi2Se3_SOC"
    expanded_dir = tmp_path / "Bi2Se3_SOC_expanded_overlap"
    shutil.copytree(source_dir, expanded_dir)
    expanded_overlap_data = expand_overlap_to_spinful(expanded_dir)

    scalar_overlap = AOMatrixObj(source_dir, matrix_type="overlap")
    expanded_overlap = AOMatrixObj(expanded_dir, matrix_type="overlap")
    assert scalar_overlap.stored_spinful is False
    assert expanded_overlap.stored_spinful is True
    np.testing.assert_allclose(expanded_overlap.mats, scalar_overlap.mats, rtol=0.0, atol=1e-14)

    scalar_hamiltonian = HamiltonianObj(source_dir)
    expanded_hamiltonian = HamiltonianObj(expanded_dir)
    np.testing.assert_allclose(expanded_hamiltonian.HR, scalar_hamiltonian.HR, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(expanded_hamiltonian.SR, scalar_hamiltonian.SR, rtol=0.0, atol=1e-14)

    supplied_hamiltonian = AOMatrixObj(source_dir, mats=scalar_hamiltonian.HR)
    assert supplied_hamiltonian.stored_spinful is True

    with (source_dir / "info.json").open() as file:
        info = json.load(file)
    structure = {
        "lattice": scalar_hamiltonian.lattice,
        "atomic_symbols": scalar_hamiltonian.elements,
        "frac_coords": scalar_hamiltonian.frac_coords,
    }
    in_memory = HamiltonianObj.from_data(
        structure,
        info,
        read_matrix_data(source_dir / "hamiltonian.h5"),
        expanded_overlap_data,
    )
    np.testing.assert_allclose(in_memory.HR, scalar_hamiltonian.HR, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(in_memory.SR, scalar_hamiltonian.SR, rtol=0.0, atol=1e-14)


def test_split_hkb_rejects_spin_expanded_overlap(eigen_data, tmp_path):
    source_dir = eigen_data["input"] / "Bi2Se3_SOC"
    split_dir = tmp_path / "Bi2Se3_SOC_split_expanded_overlap"
    make_split_hkb(source_dir, split_dir)
    hkb = AOMatrixObj(split_dir, matrix_type="hkb")
    expand_overlap_to_spinful(split_dir)
    with pytest.raises(ValueError, match="scalar"):
        HamiltonianObj(split_dir)
    with pytest.raises(ValueError, match="spinless"):
        AOMatrixObj(split_dir, matrix_type="hkb", mats=hkb.mats)


def test_overlap_storage_inference_rejects_mixed_or_nonphysical_shapes():
    atom_pairs = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="consistently"):
        AOMatrixObj._infer_overlap_storage(atom_pairs, np.array([[1, 1], [2, 2]]), [1], True)
    with pytest.raises(ValueError, match="consistently"):
        AOMatrixObj._infer_overlap_storage(atom_pairs[:1], np.array([[2, 2]]), [1], False)
