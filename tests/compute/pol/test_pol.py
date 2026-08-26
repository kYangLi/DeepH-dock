"""
Tests for electric polarization (Berry phase) via :class:`PolCalc`.

The test data (GaAs zincblende) lives under ``tests/compute/pol``:

- ``dft/*`` : DeepH-format inputs (POSCAR, info.json, overlap.h5,
  hamiltonian.h5, position_matrix.h5) converted from the OpenMX SCF runs.
- ``openmx_calc/*`` : reference ``polB.std`` from OpenMX's ``polB`` utility,
  used as the benchmark.

The electronic polarization is compared against the "Electron" column of
``polB.std`` (Berry phase, ``mod`` the polarization quantum).  The ionic
part is compared against the "Core" column using ``element_ion_charge``
``{Ga: 13, As: 15}`` (the pseudo valence charges of the PBE19 potentials,
which include the 3d electrons in valence for both Ga and As).
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.compute.eigen.polarization import PolCalc, assemble_position_matrix

POL_DIR = Path(__file__).parent
DFT_DIR = POL_DIR / "pol.bak"

# Pseudo valence charges (from OpenMX PBE19 potentials, 3d in valence).
ION_CHARGE = {"Ga": 13.0, "As": 15.0}

# polB benchmark used Nk = 9 x 9 x 9 (Gamma centered).
K_MESH = (9, 9, 9)

# GaAs: 28 electrons -> 14 occupied bands (spinless).
OCCUPATION = 28

# Electron dipole from polB.std "Electron" column (Debye).
BENCHMARK_ELEC_DEBYE = {
    "0.original": (-13.56905166, -13.56905144, -13.56905101),
    "1.Ga_x": (-16.19904321, -13.56874253, -13.56874212),
    "2.As_x": (-17.66258211, -13.56851714, -13.56851665),
}

# Core (ionic) dipole from polB.std "Core" column (Debye).
BENCHMARK_CORE_DEBYE = {
    "0.original": (101.76794404, 278.16571372, 278.16571372),
    "1.Ga_x": (104.89002846, 278.16571372, 278.16571372),
    "2.As_x": (105.37034914, 101.76794404, 101.76794404),
}

# --- Spinful (non-collinear) benchmark: Bi2Se3_SOC ---
# Pseudo valence charges: Bi (5d in valence) = 15, Se (3d in core) = 6.
SOC_ION_CHARGE = {"Bi": 15.0, "Se": 6.0}
SOC_K_MESH = (9, 9, 1)  # polB used Nk = 9 9 1
SOC_OCCUPATION = 48

BENCHMARK_SOC_ELEC_DEBYE = {
    "Bi2Se3_SOC": (-0.00000000, 0.00001169, -0.00044602),
}
BENCHMARK_SOC_CORE_DEBYE = {
    "Bi2Se3_SOC": (271.78200259, -87.17411798, 4600.11756286),
}


def load_position_matrix(data_path):
    with h5py.File(str(Path(data_path) / "position_matrix.h5"), "r") as f:
        return {k: f[k][:] for k in f.keys()}


def make_polcalc(name):
    data_path = DFT_DIR / name
    obj_H = HamiltonianObj(str(data_path))
    pos = load_position_matrix(data_path)
    return PolCalc(obj_H, pos, ION_CHARGE, occupation=OCCUPATION)


def test_assemble_position_matrix():
    """Position matrix assembles to (N_R, 3, Norb, Norb), absolute coords (Angstrom)."""
    data_path = DFT_DIR / "0.original"
    obj_H = HamiltonianObj(str(data_path))
    pos = load_position_matrix(data_path)
    r_abs = assemble_position_matrix(pos, obj_H)

    assert r_abs.shape == (len(obj_H.Rijk_list), 3, obj_H.orbits_quantity, obj_H.orbits_quantity)

    # On-site (R = [0,0,0]) block of atom 0 (Ga at (0, 2.825, 2.825) Ang).
    # x-component diagonal ~ 0, y/z-component diagonal ~ 2.825.
    i_R0 = np.where((obj_H.Rijk_list == [0, 0, 0]).all(axis=1))[0][0]
    assert np.isclose(r_abs[i_R0, 0, 0, 0], 0.0, atol=1e-6)
    assert np.isclose(r_abs[i_R0, 1, 0, 0], 2.825, atol=1e-3)
    assert np.isclose(r_abs[i_R0, 2, 0, 0], 2.825, atol=1e-3)


@pytest.mark.parametrize("name", ["0.original", "1.Ga_x", "2.As_x"])
def test_electronic_polarization_benchmark(name):
    """Electronic (Berry phase) dipole matches polB.std 'Electron' column."""
    pc = make_polcalc(name)
    res = pc.calc(k_mesh=K_MESH, n_jobs=1, parallel_k=False)
    debye = res["dipole_elec_debye"]
    ref = np.array(BENCHMARK_ELEC_DEBYE[name])
    assert np.allclose(debye, ref, atol=0.05), f"{name}: electron dipole {debye} != {ref}"


@pytest.mark.parametrize("name", ["0.original", "1.Ga_x", "2.As_x"])
def test_ionic_polarization_benchmark(name):
    """Ionic dipole matches polB.std 'Core' column (with Ga=13, As=15)."""
    pc = make_polcalc(name)
    res = pc.calc(k_mesh=K_MESH, n_jobs=1, parallel_k=False)
    debye = res["dipole_ion_debye"]
    ref = np.array(BENCHMARK_CORE_DEBYE[name])
    assert np.allclose(debye, ref, atol=0.05), f"{name}: ionic dipole {debye} != {ref}"


@pytest.mark.parametrize("name", ["Bi2Se3_SOC"])
def test_spinful_electronic_polarization_benchmark(name):
    """Spinful (non-collinear) electronic polarization matches polB.std."""
    data_path = DFT_DIR / name
    obj_H = HamiltonianObj(str(data_path))
    assert obj_H.spinful, f"{name} should be spinful"
    pos = load_position_matrix(data_path)
    pc = PolCalc(obj_H, pos, SOC_ION_CHARGE, occupation=SOC_OCCUPATION)
    res = pc.calc(k_mesh=SOC_K_MESH, n_jobs=1, parallel_k=False)
    debye = res["dipole_elec_debye"]
    ref = np.array(BENCHMARK_SOC_ELEC_DEBYE[name])
    assert np.allclose(debye, ref, atol=0.05), f"{name}: electron dipole {debye} != {ref}"


@pytest.mark.parametrize("name", ["Bi2Se3_SOC"])
def test_spinful_ionic_polarization_benchmark(name):
    """Spinful ionic dipole matches polB.std 'Core' column (Bi=15, Se=6)."""
    data_path = DFT_DIR / name
    obj_H = HamiltonianObj(str(data_path))
    pos = load_position_matrix(data_path)
    pc = PolCalc(obj_H, pos, SOC_ION_CHARGE, occupation=SOC_OCCUPATION)
    res = pc.calc(k_mesh=SOC_K_MESH, n_jobs=1, parallel_k=False)
    debye = res["dipole_ion_debye"]
    ref = np.array(BENCHMARK_SOC_CORE_DEBYE[name])
    assert np.allclose(debye, ref, atol=0.05), f"{name}: ionic dipole {debye} != {ref}"


def test_hermiticity_identity():
    """T(k1,k2; +dk) == T(k2,k1; -dk)^dagger (symmetrization property)."""
    pc = make_polcalc("0.original")
    obj_H = pc.obj_H

    tR_plus = pc._build_tR(K_MESH, 0, sign=+1.0)
    tR_minus = pc._build_tR(K_MESH, 0, sign=-1.0)

    def Hp(k, t):
        return np.tensordot(np.exp(2j * np.pi * (k @ obj_H.Rijk_list.T)), t, axes=1)

    rng = np.random.default_rng(0)
    for _ in range(10):
        k1 = np.array([rng.uniform(0, 1), 0.0, 0.0])
        k2 = np.array([rng.uniform(0, 1), 0.0, 0.0])
        T12 = 0.5 * (Hp(k2, tR_plus) + Hp(-k1, tR_plus).T)
        T21 = 0.5 * (Hp(k1, tR_minus) + Hp(-k2, tR_minus).T)
        assert np.allclose(T12, T21.conj().T, atol=1e-10), "Hermiticity identity violated"


def test_kmesh_convergence():
    """Electronic polarization converges as the k-mesh is refined."""
    obj_H = HamiltonianObj(str(DFT_DIR / "0.original"))
    pos = load_position_matrix(DFT_DIR / "0.original")

    results = []
    for mesh in [(3, 3, 3), (5, 5, 5), (9, 9, 9)]:
        pc = PolCalc(obj_H, pos, ION_CHARGE, occupation=OCCUPATION)
        res = pc.calc(k_mesh=mesh, n_jobs=1, parallel_k=False)
        results.append(res["dipole_elec_debye"])

    # The converged value (9x9x9) should be closest to the benchmark.
    ref = np.array(BENCHMARK_ELEC_DEBYE["0.original"])
    assert np.allclose(results[-1], ref, atol=0.05)
    # Coarse meshes should already be in the right branch (mod quantum).
    for r in results:
        assert np.allclose(r, ref, atol=5.0), f"coarse-mesh result drifted: {r} vs {ref}"


# --------------------------------------------------------------------------
# Synthetic unit tests for _align_berry_phases (no external data needed).
# --------------------------------------------------------------------------

def _wrap(x):
    return (x + 0.5) % 1.0 - 0.5


def test_align_berry_phases_unwraps_jumps():
    """A smooth phase crossing the +/-0.5 branch cut is recovered continuously."""
    na, nb = 16, 16
    ia = np.arange(na)[:, None]
    ib = np.arange(nb)[None, :]
    Phi = 0.2 + 0.03 * ia + 0.02 * ib  # smooth, ranges past 0.5
    phase = _wrap(Phi)

    aligned, winding = PolCalc._align_berry_phases(phase)

    assert winding == (0, 0)
    # aligned - phase must be integer-valued (aligned == phase + n)
    diff = aligned - phase
    assert np.allclose(diff, np.round(diff))
    # aligned recovers the continuous function (same branch, anchored at phase[0,0])
    assert np.allclose(aligned, Phi, atol=1e-12)
    # the average is the correct continuous average (no branch jumps)
    assert np.isclose(aligned.mean(), Phi.mean(), atol=1e-12)


def test_align_berry_phases_detects_winding():
    """A winding phase (nonzero Chern) is detected and left unwrapped."""
    na, nb = 8, 4
    ia = np.arange(na)[:, None]
    Phi = 2.0 * ia / na  # winds twice along ia
    phase = _wrap(Phi)

    aligned, winding = PolCalc._align_berry_phases(phase)

    assert winding == (2, 0)
    assert np.allclose(aligned, phase)  # falls back to the wrapped phase


def test_align_berry_phases_residue_warns():
    """A phase field with a residue (coarse mesh) warns and is left unwrapped."""
    import warnings

    phase = np.array([[0.0, 0.6], [0.2, 0.3]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aligned, winding = PolCalc._align_berry_phases(phase)

    assert winding == (0, 0)
    assert np.allclose(aligned, phase)  # unchanged
    assert len(w) >= 1
    assert "residue" in str(w[0].message)


def test_align_berry_phases_1d():
    """Degenerate 1D case (nb=1) unwraps along the single axis."""
    na = 16
    # Periodic single-valued function that still crosses the +/-0.5 branch cut.
    Phi = 0.2 + 0.4 * np.sin(2 * np.pi * np.arange(na) / na)
    phase = _wrap(Phi)[:, None]

    aligned, winding = PolCalc._align_berry_phases(phase)

    assert winding == (0, 0)
    assert np.allclose(aligned[:, 0], Phi, atol=1e-12)
