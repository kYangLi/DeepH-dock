"""
Tests for the MPI-parallel PETSc/SLEPc eigensolver (calc-band-petsc).
"""

import os
import shutil
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MPI_ENV = {**os.environ, "FI_PROVIDER": "shm", "I_MPI_FABRICS": "shm"}


@lru_cache(maxsize=1)
def _petsc_stack_ok() -> bool:
    """Probe the mpi4py/petsc4py/slepc4py stack (complex scalar PETSc) in a subprocess."""
    probe = (
        "from mpi4py import MPI; "
        "from petsc4py import PETSc; "
        "from slepc4py import SLEPc; "
        "import numpy as np; "
        "assert np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], env=MPI_ENV, capture_output=True, timeout=120, check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(not _petsc_stack_ok(), reason="PETSc/SLEPc stack (complex scalar) is not available")


@pytest.fixture
def petsc_data():
    """PETSc eigen test data (spinless)"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "eigen.clean" / "MoTe2",
        "reference": test_dir / "eigen.bak" / "MoTe2",
    }


@pytest.fixture
def petsc_data_soc():
    """PETSc eigen test data (spinful, SOC)"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "eigen.clean" / "Bi2Se3_SOC",
        "reference": test_dir / "eigen.bak" / "Bi2Se3_SOC",
    }


def _run_calc_band(tmp_path, input_dir, extra_args=(), mpirun_np=None):
    test_dir = tmp_path / "petsc"
    shutil.copytree(input_dir, test_dir)
    cmd = []
    if mpirun_np is not None:
        mpirun = shutil.which("mpirun")
        if mpirun is None:
            pytest.skip("mpirun is not available")
        cmd += [mpirun, "-np", str(mpirun_np)]
    cmd += ["dock", "compute", "eigen", "calc-band-petsc", str(test_dir), "--nb", "10"] + list(extra_args)
    result = subprocess.run(cmd, env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
    band_file = test_dir / "band.h5"
    assert band_file.exists(), "band.h5 not found"
    return band_file


def _compare_with_reference(band_file, reference_file, atol=1e-3):
    with h5py.File(band_file, "r") as hf:
        petsc_data = hf["band_data"][()]
        petsc_shift = hf["fermi_energy_before_shift_eV"][()]
    with h5py.File(reference_file, "r") as hf:
        ref_data = hf["band_data"][()]
        ref_shift = hf["fermi_energy_before_shift_eV"][()]
    assert petsc_data.shape[1] == ref_data.shape[1], "k-point quantity mismatch"
    petsc_absolute = petsc_data + petsc_shift
    ref_absolute = ref_data + ref_shift
    distance = np.abs(petsc_absolute.T[:, :, None] - ref_absolute.T[:, None, :]).min(axis=2)
    assert distance.max() < atol, f"Max eigenvalue deviation from reference: {distance.max()} eV"


def test_band_petsc_serial(petsc_data, tmp_path):
    """Test PETSc band calculation, single process"""
    band_file = _run_calc_band(tmp_path, petsc_data["input"])
    _compare_with_reference(band_file, petsc_data["reference"] / "band.h5")


def test_band_petsc_mpi2(petsc_data, tmp_path):
    """Test PETSc band calculation with 2 MPI processes"""
    band_file = _run_calc_band(tmp_path, petsc_data["input"], mpirun_np=2)
    _compare_with_reference(band_file, petsc_data["reference"] / "band.h5")


def test_band_petsc_spinful(petsc_data_soc, tmp_path):
    """Test PETSc band calculation for a spinful (SOC) system"""
    band_file = _run_calc_band(tmp_path, petsc_data_soc["input"])
    _compare_with_reference(band_file, petsc_data_soc["reference"] / "band.h5")


def test_eigenvector_gather(petsc_data, tmp_path):
    """Test the bands_only=False path: rank-local blocks, gather_vec_to_rank0, residual and S-orthonormality"""
    script = textwrap.dedent("""
        import numpy as np
        from mpi4py import MPI
        from deepx_dock.compute.eigen.hamiltonian_petsc import PETScHamiltonianObj

        data_path = r"{data_path}"
        obj = PETScHamiltonianObj(data_path)
        ks = [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])]
        eigvals, eigvecs = obj.diag(ks, bands_only=False, num_band=8, maxiter=500)
        assert eigvals.shape == (8, 2), f"eigvals shape {{eigvals.shape}}"
        ## rank-local blocks: shape and row ownership consistency
        rstart, rend = obj.vecs_empty[0].getOwnershipRange() if obj.vecs_empty is not None else (0, obj.nrows)
        assert eigvecs.shape == (rend - rstart, 8, 2), f"local shape {{eigvecs.shape}}, ownership [{{rstart}}, {{rend}})"
        total_rows = obj.comm.allreduce(eigvecs.shape[0], op=MPI.SUM)
        assert total_rows == obj.nrows, f"sum of local rows {{total_rows}} != nrows {{obj.nrows}}"
        ## gather to rank 0
        gathered = obj.gather_vec_to_rank0(eigvecs)
        if obj.rank == 0:
            assert gathered.shape == (obj.nrows, 8, 2), f"gathered shape {{gathered.shape}}"
            for ik, k in enumerate(ks):
                Sk, Hk = obj.Sk_and_Hk(k)
                Sk = Sk.toarray()
                Hk = Hk.toarray()
                V = gathered[:, :, ik]
                residual = np.abs(Hk @ V - (Sk @ V) * eigvals[:, ik][None, :]).max()
                assert residual < 1e-4, f"residual {{residual}}"
                overlap = V.conj().T @ Sk @ V
                orthonormality_error = np.abs(overlap - np.eye(eigvals.shape[0])).max()
                assert orthonormality_error < 1e-4, f"orthonormality error {{orthonormality_error}}"
        else:
            assert gathered is None, "non-zero ranks must receive None"
        print("EIGVEC_TEST_OK")
        """).format(data_path=petsc_data["input"])
    result = subprocess.run([sys.executable, "-c", script], env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    print(result.stderr)
    assert "EIGVEC_TEST_OK" in result.stdout, f"Eigenvector test failed:\n{result.stdout}\n{result.stderr}"


def test_gather_vec_to_rank0(petsc_data, tmp_path):
    """Test gather_vec_to_rank0 on small blocks: correct concatenation and None on non-root ranks"""
    script = textwrap.dedent("""
        import numpy as np
        from mpi4py import MPI
        from deepx_dock.compute.eigen.hamiltonian_petsc import PETScHamiltonianObj

        obj = PETScHamiltonianObj(r"{data_path}")
        rstart, rend = (0, 0)
        local = np.full((3, 2, 1), obj.rank, dtype=float)
        gathered = obj.gather_vec_to_rank0(local)
        size = obj.comm.Get_size()
        if obj.rank == 0:
            expected = np.concatenate([np.full((3, 2, 1), r, dtype=float) for r in range(size)], axis=0)
            assert gathered.shape == expected.shape, f"shape {{gathered.shape}} != {{expected.shape}}"
            assert np.array_equal(gathered, expected), "concatenation order is wrong"
        else:
            assert gathered is None, "non-zero ranks must receive None"
        print("GATHER_TEST_OK")
        """).format(data_path=petsc_data["input"])
    result = subprocess.run([sys.executable, "-c", script], env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    print(result.stderr)
    assert "GATHER_TEST_OK" in result.stdout, f"Gather test failed:\n{result.stdout}\n{result.stderr}"


def test_dos_petsc(petsc_data, tmp_path):
    """Test PETSc DOS (fermi cache reuse + partial spectrum) against the scipy full-spectrum DOS"""
    test_dir = tmp_path / "petsc_dos"
    shutil.copytree(petsc_data["input"], test_dir)

    scipy_cmd = [
        "dock",
        "compute",
        "eigen",
        "calc-dos",
        str(test_dir),
        "-d",
        "0.3",
        "--E-win",
        "-2",
        "2",
        "--num",
        "400",
        "-s",
        "0.04",
        "-j",
        "1",
    ]
    result = subprocess.run(scipy_cmd, env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    assert result.returncode == 0, f"scipy calc-dos failed:\n{result.stdout}\n{result.stderr}"
    with h5py.File(test_dir / "dos.h5", "r") as hf:
        ref_egrid, ref_dos = hf["energy"][()], hf["dos_data"][()]

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available")
    petsc_cmd = [
        mpirun,
        "-np",
        "2",
        "dock",
        "compute",
        "eigen",
        "calc-dos-petsc",
        str(test_dir),
        "-d",
        "0.3",
        "--nb",
        "40",
        "--E-win",
        "-2",
        "2",
        "--num",
        "400",
        "-s",
        "0.04",
    ]
    result = subprocess.run(petsc_cmd, env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    assert result.returncode == 0, f"calc-dos-petsc failed:\n{result.stdout}\n{result.stderr}"
    assert "Use cached fermi energy" in result.stdout, "The cached fermi energy was not reused"
    for fname in ("eigval.h5", "dos.h5", "dos.png"):
        assert (test_dir / fname).exists(), f"{fname} not found"

    with h5py.File(test_dir / "dos.h5", "r") as hf:
        petsc_egrid, petsc_dos = hf["energy"][()], hf["dos_data"][()]
    assert np.allclose(ref_egrid, petsc_egrid), "DOS energy grids differ"
    interior = np.abs(petsc_egrid) <= 1.0
    dos_scale = ref_dos[interior].max()
    max_dev = np.abs(ref_dos[interior] - petsc_dos[interior]).max()
    assert max_dev < 1e-3 * dos_scale, f"Max interior DOS deviation {max_dev} exceeds tolerance"


def test_dos_petsc_fermi_fallback(petsc_data, tmp_path):
    """Test the info.json fermi fallback when no fermi_energy.json cache exists"""
    test_dir = tmp_path / "petsc_dos_fallback"
    shutil.copytree(petsc_data["input"], test_dir)
    result = subprocess.run(
        ["dock", "compute", "eigen", "calc-dos-petsc", str(test_dir), "-d", "0.5", "--nb", "10", "--E-win", "-2", "2"],
        env=MPI_ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    assert result.returncode == 0, f"calc-dos-petsc failed:\n{result.stdout}\n{result.stderr}"
    assert "info.json" in result.stdout, "The info.json fermi fallback was not used"
    assert (test_dir / "dos.h5").exists(), "dos.h5 not found"


def test_fA_mul_b(petsc_data, tmp_path):
    """Test fA_mul_b: distributed sqrt(S) @ b vs the dense eigendecomposition reference"""
    script = textwrap.dedent("""
        import numpy as np
        from scipy.linalg import eigh
        from deepx_dock.compute.eigen.hamiltonian_petsc import PETSc, PETScHamiltonianObj

        obj = PETScHamiltonianObj(r"{data_path}")
        k = np.array([0.1, 0.0, 0.0])
        Sk_csr, _ = obj.Sk_and_Hk(k)  # rank 0 only
        Sk = obj._translate_scipy_to_petsc(Sk_csr)
        rstart, rend = Sk.getOwnershipRange()

        rng = np.random.default_rng(42)
        b_full = rng.standard_normal(obj.nrows) + 1j * rng.standard_normal(obj.nrows)
        b = PETSc.Vec().createWithArray(b_full[rstart:rend].copy(), comm=obj.comm)
        sqrt_clip = lambda lam: np.sqrt(np.clip(lam, 0.0, None))
        y = obj.fA_mul_b(sqrt_clip, Sk, b, m=60)
        b.destroy()

        y_gathered = obj.gather_vec_to_rank0(y.getArray(readonly=True).copy())
        y.destroy()
        if obj.rank == 0:
            S = Sk_csr.toarray()
            S = (S + S.conj().T) / 2.0
            lam, V = eigh(S)
            y_ref = (V * sqrt_clip(lam)) @ V.conj().T @ b_full
            rel_err = np.linalg.norm(y_gathered - y_ref) / np.linalg.norm(y_ref)
            assert rel_err < 1e-8, f"sqrt(S) b rel_err {{rel_err}}"
            print(f"rel_err = {{rel_err:.3e}}")
        else:
            assert y_gathered is None
        print("FA_MUL_B_TEST_OK")
        """).format(data_path=petsc_data["input"])
    result = subprocess.run([sys.executable, "-c", script], env=MPI_ENV, capture_output=True, text=True, check=False)
    print(result.stdout)
    print(result.stderr)
    assert "FA_MUL_B_TEST_OK" in result.stdout, f"fA_mul_b test failed:\n{result.stdout}\n{result.stderr}"
