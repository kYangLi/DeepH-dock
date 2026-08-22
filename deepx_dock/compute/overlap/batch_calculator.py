"""Batch overlap calculation for DeepH data directories."""

from pathlib import Path

import click

from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME, DEEPX_POSCAR_FILENAME
from deepx_dock.compute.overlap.overlap import save_overlap_from_files
from deepx_dock.misc import get_data_dir_lister
from deepx_dock.parallel import parallel_map


class BatchOverlapCalculator:
    """Run HPRO overlap calculations over a tiered DeepH data directory."""

    def __init__(
        self,
        data_dir: Path,
        basis_path: Path,
        aocode: str,
        tier_num: int = 0,
        n_jobs: int = -1,
        spinful: bool = False,
        ecut: float | None = None,
        kdense: float | None = None,
        force: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.basis_path = Path(basis_path)
        self.aocode = aocode
        self.tier_num = tier_num
        self.n_jobs = n_jobs
        self.spinful = spinful
        self.ecut = ecut
        self.kdense = kdense
        self.force = force

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.basis_path.is_dir():
            raise FileNotFoundError(f"Basis directory not found: {self.basis_path}")

    def run(self) -> None:
        """Execute batch processing."""
        data_dirs = list(self._scan_data_dirs())
        if not data_dirs:
            click.echo("[warning] No valid data directories found")
            return

        click.echo(f"[info] Found {len(data_dirs)} data directories")
        self._process_all(data_dirs)

    def _scan_data_dirs(self):
        validation_check = self._make_validation_check()
        return get_data_dir_lister(self.data_dir, self.tier_num, validation_check)

    def _make_validation_check(self):
        def validation_check(root_dir: Path, prev_dirname: Path):
            all_files = {v.name for v in root_dir.iterdir()}
            has_poscar = DEEPX_POSCAR_FILENAME in all_files
            has_overlap = DEEPX_OVERLAP_FILENAME in all_files

            if not has_poscar:
                print(f"Skip {prev_dirname} (no {DEEPX_POSCAR_FILENAME} found)")
                return
            if has_overlap and not self.force:
                print(f"Skip {prev_dirname} ({DEEPX_OVERLAP_FILENAME} exists, use --force to overwrite)")
                return
            yield prev_dirname

        return validation_check

    def _process_all(self, data_dirs: list[Path]) -> None:
        def process_single(relative_dir: Path):
            data_path = self.data_dir / relative_dir
            try:
                save_overlap_from_files(
                    data_path / DEEPX_POSCAR_FILENAME,
                    self.basis_path,
                    self.aocode,
                    output_dir=data_path,
                    spinful=self.spinful,
                    ecut=self.ecut,
                    kdense=self.kdense,
                    force=True,
                )
                return f"[done] {relative_dir}"
            except Exception as exc:
                return f"[error] {relative_dir}: {exc}"

        results = parallel_map(process_single, data_dirs, n_jobs=self.n_jobs, desc="Processing")
        for result in results:
            if result:
                click.echo(f"  {result}")
