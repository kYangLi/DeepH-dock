"""Batch converter for FHI-aims output-basis to species HDF5.

This module scans multiple run directories, parses each run in parallel,
merges compatible species definitions, and exports one consolidated
species_aims_{xc}.h5 file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

from deepx_dock.misc import get_data_dir_lister
from deepx_dock.parallel import parallel_map
from deepx_dock.convert.fhi_aims.species_h5_single import (
    SpeciesModel,
    build_species_models,
    pack_species_models,
    parse_control_species,
    parse_output_basis_run,
    validate_species_h5,
    write_species_h5,
)


@dataclass
class RunSpeciesResult:
    run_dir: Path
    xc_functional: str
    models: Dict[str, SpeciesModel]


@dataclass
class MultiExportSummary:
    output_h5: Path
    selected_run_dirs: List[Path]
    used_run_dirs: List[Path]
    skipped_run_dirs: Dict[Path, str]
    conflict_warnings: List[str]
    xc_functional: str
    n_species: int


@dataclass
class MultiExportConfig:
    runs_root: Path
    output_h5: Path
    tier_num: int = 0
    jobs_num: int = -1
    xc_functional: str | None = None
    tol: float = 5e-7
    conflict_policy: str = "error"
    fail_fast: bool = True


EchoFunc = Callable[[str], None]


def _emit(echo: EchoFunc | None, msg: str) -> None:
    if echo is not None:
        echo(msg)


def _make_validation_check(echo: EchoFunc | None = None):
    def validation_check(root_dir: Path, prev_dirname: Path):
        all_files = [v.name for v in root_dir.iterdir()]
        has_control = "control.in" in all_files
        has_dat = any(name.endswith(".dat") for name in all_files)

        if has_control and has_dat:
            yield prev_dirname
        else:
            if has_control and (not has_dat):
                _emit(echo, f"Skip {prev_dirname} (no .dat files found)")
            elif (not has_control) and has_dat:
                _emit(echo, f"Skip {prev_dirname} (no control.in found)")
            else:
                _emit(echo, f"Skip {prev_dirname} (missing control.in and .dat files)")

    return validation_check


def collect_run_dirs_from_root(
    runs_root: str | Path,
    tier_num: int = 0,
    echo: EchoFunc | None = None,
) -> List[Path]:
    """Collect run directories using DeepX tier scanning semantics."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs_root is not a directory: {runs_root}")

    rel_dirs = list(get_data_dir_lister(runs_root, tier_num, _make_validation_check(echo=echo)))
    rel_dirs = sorted(rel_dirs, key=lambda p: str(p))
    return [runs_root / rel for rel in rel_dirs]


def _extract_single_run(run_dir: Path, tol: float) -> RunSpeciesResult:
    try:
        control = parse_control_species(run_dir / "control.in")
        records = parse_output_basis_run(run_dir)
        models = build_species_models(control, records, run_dir=run_dir, tol=tol)
    except Exception as exc:
        raise RuntimeError(f"{run_dir}: {exc}") from exc

    xc = str(control["xc_functional"]).lower().strip()
    if not xc:
        xc = "unknown"

    return RunSpeciesResult(run_dir=run_dir, xc_functional=xc, models=models)


def _rows_close(rows_a: List[np.ndarray], rows_b: List[np.ndarray], atol: float, rtol: float) -> bool:
    if len(rows_a) != len(rows_b):
        return False
    for row_a, row_b in zip(rows_a, rows_b):
        if row_a.shape != row_b.shape:
            return False
        if not np.allclose(row_a, row_b, atol=atol, rtol=rtol):
            return False
    return True


def _species_models_equal(a: SpeciesModel, b: SpeciesModel, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    if a.element != b.element:
        return False
    if a.species_name != b.species_name:
        return False
    if not np.array_equal(a.nljz, b.nljz):
        return False
    if not np.array_equal(a.grid_length, b.grid_length):
        return False
    if not np.allclose(a.cutoff_radii, b.cutoff_radii, atol=atol, rtol=rtol):
        return False
    if not np.array_equal(a.fn_type_codes, b.fn_type_codes):
        return False
    if not np.array_equal(a.orbital_index_local, b.orbital_index_local):
        return False
    if not _rows_close(a.radius_grid_raw, b.radius_grid_raw, atol=atol, rtol=rtol):
        return False
    if not _rows_close(a.radius_data_raw, b.radius_data_raw, atol=atol, rtol=rtol):
        return False
    if not _rows_close(a.kinetic_data_raw, b.kinetic_data_raw, atol=atol, rtol=rtol):
        return False
    if not np.array_equal(a.val_density_nljz, b.val_density_nljz):
        return False
    if not np.array_equal(a.val_density_grid_length, b.val_density_grid_length):
        return False
    if not np.allclose(a.val_density_cutoff_radii, b.val_density_cutoff_radii, atol=atol, rtol=rtol):
        return False
    if not _rows_close(a.val_density_radius_grid_raw, b.val_density_radius_grid_raw, atol=atol, rtol=rtol):
        return False
    if not _rows_close(a.val_density_radius_data_raw, b.val_density_radius_data_raw, atol=atol, rtol=rtol):
        return False
    return True


def _merge_species_models(
    results: List[RunSpeciesResult],
    conflict_policy: str = "error",
) -> tuple[Dict[str, SpeciesModel], Dict[str, Path], List[str]]:
    merged: Dict[str, SpeciesModel] = {}
    source_run: Dict[str, Path] = {}
    warnings: List[str] = []

    for result in sorted(results, key=lambda x: str(x.run_dir)):
        for species_name in sorted(result.models.keys()):
            model = result.models[species_name]
            if species_name not in merged:
                merged[species_name] = model
                source_run[species_name] = result.run_dir
                continue

            if _species_models_equal(merged[species_name], model):
                continue

            msg = (
                f"species={species_name} conflict: keep={source_run[species_name]}, "
                f"incoming={result.run_dir}"
            )
            if conflict_policy == "error":
                raise ValueError(f"E_SPECIES_CONFLICT: {msg}")
            if conflict_policy == "first":
                warnings.append(msg)
                continue

            raise ValueError(f"Unknown conflict policy: {conflict_policy}")

    return merged, source_run, warnings


def _resolve_xc(results: List[RunSpeciesResult], xc_functional: str | None) -> str:
    if xc_functional is not None:
        xc = xc_functional.lower().strip()
        return xc if xc else "unknown"

    xcs = sorted({r.xc_functional for r in results if r.xc_functional and r.xc_functional != "unknown"})
    if len(xcs) == 0:
        return "unknown"
    if len(xcs) == 1:
        return xcs[0]
    raise ValueError(f"Inconsistent xc functionals from runs: {xcs}")


def convert_multi_runs_to_species_h5(
    run_dirs: Iterable[str | Path],
    output_h5: str | Path,
    xc_functional: str | None = None,
    tol: float = 5e-7,
    n_jobs: int = -1,
    conflict_policy: str = "error",
    fail_fast: bool = True,
) -> MultiExportSummary:
    """Parse multiple runs in parallel and export consolidated species HDF5."""
    selected = [Path(p) for p in run_dirs]
    selected = sorted(set(selected), key=lambda p: str(p))
    if len(selected) == 0:
        raise ValueError("No run directories provided")

    for run_dir in selected:
        if not run_dir.is_dir():
            raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    skipped: Dict[Path, str] = {}
    results: List[RunSpeciesResult] = []

    if fail_fast:
        def worker_strict(path: Path):
            return _extract_single_run(path, tol=tol)

        results = parallel_map(worker_strict, selected, n_jobs=n_jobs, desc="Run dirs")
    else:
        def worker_best_effort(path: Path):
            try:
                return _extract_single_run(path, tol=tol), None
            except Exception as exc:
                return None, str(exc)

        parsed = parallel_map(worker_best_effort, selected, n_jobs=n_jobs, desc="Run dirs")
        for run_dir, (result, err) in zip(selected, parsed):
            if result is None:
                skipped[run_dir] = err if err else "unknown parse failure"
            else:
                results.append(result)

        if len(results) == 0:
            raise RuntimeError("All run directories failed to parse")

    merged_models, source_run, conflict_warnings = _merge_species_models(
        results,
        conflict_policy=conflict_policy,
    )

    xc = _resolve_xc(results, xc_functional=xc_functional)
    export_model = pack_species_models(merged_models, xc_functional=xc)

    output_h5 = Path(output_h5)
    write_species_h5(output_h5, export_model)
    validate_species_h5(output_h5)

    return MultiExportSummary(
        output_h5=output_h5,
        selected_run_dirs=selected,
        used_run_dirs=sorted(set(source_run.values()), key=lambda p: str(p)),
        skipped_run_dirs=skipped,
        conflict_warnings=conflict_warnings,
        xc_functional=xc,
        n_species=len(merged_models),
    )


class SpeciesH5MultiExporter:
    """High-level multi-run species exporter with validation and user-facing logs."""

    def __init__(self, config: MultiExportConfig, echo: EchoFunc | None = None):
        self.config = config
        self.echo = echo

    def _validate_config(self) -> None:
        if not self.config.runs_root.is_dir():
            raise NotADirectoryError(f"runs_root is not a directory: {self.config.runs_root}")
        policy = self.config.conflict_policy.lower()
        if policy not in {"error", "first"}:
            raise ValueError(f"Unknown conflict policy: {self.config.conflict_policy}")

    def _print_header(self) -> None:
        _emit(self.echo, "=" * 60)
        _emit(self.echo, "[info] Exporting multi-run FHI-aims species H5")
        _emit(self.echo, "=" * 60)
        _emit(self.echo, f"[info] runs_root: {self.config.runs_root}")
        _emit(self.echo, f"[info] output_h5: {self.config.output_h5}")
        _emit(self.echo, f"[info] tier_num: {self.config.tier_num}")
        _emit(self.echo, f"[info] jobs_num: {self.config.jobs_num}")
        _emit(self.echo, f"[info] tol: {self.config.tol}")
        _emit(self.echo, f"[info] conflict_policy: {self.config.conflict_policy.lower()}")
        _emit(self.echo, f"[info] fail_fast: {self.config.fail_fast}")
        if self.config.xc_functional:
            _emit(self.echo, f"[info] xc override: {self.config.xc_functional}")
        _emit(self.echo, "")

    def _print_summary(self, summary: MultiExportSummary) -> None:
        _emit(self.echo, "[done] Export completed and validated")
        _emit(self.echo, f"[done] Output file: {summary.output_h5}")
        _emit(self.echo, f"[done] species count: {summary.n_species}")
        _emit(self.echo, f"[done] xc functional: {summary.xc_functional}")
        _emit(self.echo, f"[done] source runs used: {len(summary.used_run_dirs)}")

        if summary.skipped_run_dirs:
            _emit(self.echo, "")
            _emit(self.echo, f"[warning] skipped runs: {len(summary.skipped_run_dirs)}")
            for run_dir, reason in summary.skipped_run_dirs.items():
                _emit(self.echo, f"  - {run_dir}: {reason}")

        if summary.conflict_warnings:
            _emit(self.echo, "")
            _emit(self.echo, f"[warning] species conflicts ignored: {len(summary.conflict_warnings)}")
            for msg in summary.conflict_warnings:
                _emit(self.echo, f"  - {msg}")

    def run(self) -> MultiExportSummary:
        self._validate_config()
        self._print_header()

        run_dirs = collect_run_dirs_from_root(
            runs_root=self.config.runs_root,
            tier_num=self.config.tier_num,
            echo=self.echo,
        )
        if len(run_dirs) == 0:
            raise ValueError("No valid run directories found (need control.in + *.dat)")

        _emit(self.echo, f"[info] selected run dirs: {len(run_dirs)}")
        _emit(self.echo, "")

        summary = convert_multi_runs_to_species_h5(
            run_dirs=run_dirs,
            output_h5=self.config.output_h5,
            xc_functional=self.config.xc_functional,
            tol=self.config.tol,
            n_jobs=self.config.jobs_num,
            conflict_policy=self.config.conflict_policy.lower(),
            fail_fast=self.config.fail_fast,
        )
        self._print_summary(summary)
        return summary


def run_species_h5_multi(
    runs_root: str | Path,
    output_h5: str | Path,
    tier_num: int = 0,
    jobs_num: int = -1,
    xc_functional: str | None = None,
    tol: float = 5e-7,
    conflict_policy: str = "error",
    fail_fast: bool = True,
    echo: EchoFunc | None = None,
) -> MultiExportSummary:
    """One-shot entry for CLI and scripts."""
    config = MultiExportConfig(
        runs_root=Path(runs_root),
        output_h5=Path(output_h5),
        tier_num=tier_num,
        jobs_num=jobs_num,
        xc_functional=xc_functional,
        tol=tol,
        conflict_policy=conflict_policy,
        fail_fast=fail_fast,
    )
    return SpeciesH5MultiExporter(config=config, echo=echo).run()
