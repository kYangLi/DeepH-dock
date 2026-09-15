import json

import pytest

from deepx_dock.analyze.error.with_infer_res import BaseAnalyzer
from deepx_dock.convert.deeph import minus_core_hamiltonian
from deepx_dock.convert.deeph import standardize_hamiltonian
from deepx_dock.convert.deeph import translate_old_dataset_to_new
from deepx_dock.convert.hopcp.translate_deeph_to_petsc import (
    DeepHtoPETScTranslator,
    PETScWriter,
)
from deepx_dock.convert.openmx.translate_deeph_to_openmx import (
    DeepHToOpenMXTranslator,
    OPENMX_SCFOUT_FILENAME,
    OpenMXWriter,
)
from deepx_dock.convert.siesta import translate_deeph_to_siesta


def _write_info(data_dir, split=True):
    data_dir.mkdir(parents=True)
    info = {"spinful": split, "split_hkb": split}
    with (data_dir / "info.json").open("w") as file:
        json.dump(info, file)


def test_downgrade_rejects_split_before_output_creation(tmp_path):
    new_data_dir = tmp_path / "new"
    _write_info(new_data_dir / "sample")
    (new_data_dir / "sample" / "POSCAR").touch()
    old_data_dir = tmp_path / "old"
    translator = translate_old_dataset_to_new.OldDatasetTranslator(
        old_data_dir, new_data_dir, n_jobs=1
    )

    with pytest.raises(ValueError, match="split_hkb=true"):
        translator.transfer_all_new_to_old()

    assert not old_data_dir.exists()


def test_standardize_rejects_split_before_mutation(tmp_path):
    data_dir = tmp_path / "data"
    sample_dir = data_dir / "sample"
    _write_info(sample_dir)
    (sample_dir / "POSCAR").touch()
    standardizer = standardize_hamiltonian.DatasetHStandardize(
        data_dir, n_jobs=1
    )

    with pytest.raises(ValueError, match="split_hkb=true"):
        standardizer.standardize_all()

    assert not (sample_dir / standardize_hamiltonian.DEEPX_RAW_HAMILTONIAN_FILENAME).exists()


def test_minus_core_rejects_split_input_before_output_creation(tmp_path):
    input_dir = tmp_path / "input"
    sample_dir = input_dir / "sample"
    _write_info(sample_dir)
    (sample_dir / "hamiltonian.h5").touch()
    output_dir = tmp_path / "output"
    handler = minus_core_hamiltonian.SingleAtomHamiltonianHandler(
        input_dir, output_dir, tmp_path / "single_atoms", n_jobs=1
    )

    with pytest.raises(ValueError, match="split_hkb=true"):
        handler.transfer_all()

    assert not output_dir.exists()


def test_minus_core_rejects_split_reference_before_output_creation(tmp_path):
    input_dir = tmp_path / "input"
    sample_dir = input_dir / "sample"
    _write_info(sample_dir, split=False)
    (sample_dir / "hamiltonian.h5").touch()
    single_atoms_dir = tmp_path / "single_atoms"
    _write_info(single_atoms_dir / "Mo")
    output_dir = tmp_path / "output"
    handler = minus_core_hamiltonian.SingleAtomHamiltonianHandler(
        input_dir, output_dir, single_atoms_dir, n_jobs=1
    )

    with pytest.raises(ValueError, match="split_hkb=true"):
        handler.transfer_all()

    assert not output_dir.exists()


def test_deeph_to_siesta_rejects_split_before_output_creation(tmp_path):
    deeph_path = tmp_path / "deeph"
    _write_info(deeph_path / "sample")
    siesta_path = tmp_path / "siesta"

    with pytest.raises(ValueError, match="split_hkb=true"):
        translate_deeph_to_siesta.transfer_one_deeph_to_siesta(
            "sample", siesta_path, deeph_path, tmp_path / "basis"
        )

    assert not siesta_path.exists()


def test_hopcp_rejects_split_before_output_creation(tmp_path):
    deeph_dir = tmp_path / "deeph"
    _write_info(deeph_dir)
    (deeph_dir / "POSCAR").touch()
    output_dir = tmp_path / "petsc"

    translator = DeepHtoPETScTranslator(deeph_dir, output_dir, n_jobs=1, n_tier=-1)
    with pytest.raises(ValueError, match="split_hkb=true"):
        translator.transfer_all_deeph_to_petsc()

    assert not output_dir.exists()


def test_hopcp_single_worker_rejects_split_before_output_creation(tmp_path):
    deeph_root = tmp_path / "deeph"
    _write_info(deeph_root / "sample")
    output_root = tmp_path / "petsc"

    DeepHtoPETScTranslator.transfer_one_deeph_to_petsc(
        "sample", deeph_root, output_root, export_H=True
    )

    assert not output_root.exists()


def test_hopcp_writer_rejects_split_before_reading(tmp_path):
    deeph_dir = tmp_path / "deeph"
    _write_info(deeph_dir)
    output_dir = tmp_path / "petsc"

    with pytest.raises(ValueError, match="split_hkb=true"):
        PETScWriter(deeph_dir, output_dir).dump_data(export_H=True)

    assert not output_dir.exists()


def test_openmx_rejects_split_before_output_creation(tmp_path):
    openmx_dir = tmp_path / "openmx"
    openmx_dir.mkdir()
    (openmx_dir / OPENMX_SCFOUT_FILENAME).touch()
    deeph_dir = tmp_path / "deeph"
    _write_info(deeph_dir)
    output_dir = tmp_path / "output"

    translator = DeepHToOpenMXTranslator(openmx_dir, deeph_dir, output_dir, n_jobs=1, n_tier=-1)
    with pytest.raises(ValueError, match="split_hkb=true"):
        translator.transfer_all_deeph_to_openmx()

    assert not output_dir.exists()


def test_openmx_single_worker_rejects_split_before_output_creation(tmp_path):
    openmx_root = tmp_path / "openmx"
    openmx_sample = openmx_root / "sample"
    openmx_sample.mkdir(parents=True)
    deeph_root = tmp_path / "deeph"
    _write_info(deeph_root / "sample")
    output_root = tmp_path / "output"

    DeepHToOpenMXTranslator._transfer_one(
        "sample", openmx_root, deeph_root, output_root
    )

    assert not output_root.exists()


def test_openmx_writer_rejects_split_before_output_creation(tmp_path):
    openmx_dir = tmp_path / "openmx"
    openmx_dir.mkdir()
    deeph_dir = tmp_path / "deeph"
    _write_info(deeph_dir)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="split_hkb=true"):
        OpenMXWriter(openmx_dir, deeph_dir, output_dir)

    assert not output_dir.exists()


def test_error_analysis_checks_only_hamiltonian_target(tmp_path):
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "hamiltonian_pred.h5").touch()
    benchmark_dir = tmp_path / "benchmark"
    _write_info(benchmark_dir)

    h_analyzer = BaseAnalyzer(pred_dir, benchmark_dir, target_name="H", n_tier=-1)
    h_analyzer.cached_result_path.touch()
    with pytest.raises(ValueError, match="split_hkb=true"):
        h_analyzer.analyze_all()

    rho_analyzer = BaseAnalyzer(pred_dir, benchmark_dir, target_name="Rho", n_tier=-1)
    loaded = []
    rho_analyzer._load_cached_result = lambda: loaded.append(True)
    rho_analyzer.analyze_all()
    assert loaded == [True]
