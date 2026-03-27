"""Batch overlap calculation for DeepH dataset."""

from pathlib import Path
from typing import Optional
import click

from deepx_dock.misc import get_data_dir_lister
from deepx_dock.parallel import parallel_map
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME


OPENMX_INPUT_FILENAME = "openmx_in.dat"


class BatchOverlapCalculator:
    """
    Batch overlap calculation for DeepH dataset (OpenMX).

    This class handles:
    1. Scanning data directories
    2. Finding openmx_in.dat files
    3. Calling OpenMXOverlapCalculator
    4. Skipping already processed data points
    """

    def __init__(
        self,
        data_dir: Path,
        species_file: Path,
        tier_num: int = 0,
        n_jobs: int = -1,
        force: bool = False,
        raw_species_dir: Optional[Path] = None,
    ):
        """
        Initialize the batch overlap calculator.

        Parameters
        ----------
        data_dir : Path
            Root directory containing the dataset.
        species_file : Path
            Path to species_openmx_{xc}.h5 file.
        tier_num : int
            Tier number for data directory structure.
            -1 for [data_dir] itself, 0 for subdirectories, etc.
        n_jobs : int
            Number of parallel jobs. -1 for all cores.
        force : bool
            Force overwrite existing overlap.h5 files.
        raw_species_dir : Path, optional
            Directory containing raw PAO/VPS files for auto-generation.
        """
        self.data_dir = Path(data_dir)
        self.species_file = Path(species_file)
        self.tier_num = tier_num
        self.n_jobs = n_jobs
        self.force = force
        self.raw_species_dir = Path(raw_species_dir) if raw_species_dir else None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        assert self.data_dir.is_dir(), f"Data directory not found: {self.data_dir}"

        if not self.species_file.exists() and self.raw_species_dir is None:
            raise FileNotFoundError(
                f"Species file '{self.species_file}' not found. "
                f"Use --raw-species-dir to specify PAO/VPS source directories."
            )

    def run(self) -> None:
        """Execute batch processing."""
        self._ensure_species_file()

        data_dirs = list(self._scan_data_dirs())

        if not data_dirs:
            click.echo("[warning] No valid data directories found")
            return

        click.echo(f"[info] Found {len(data_dirs)} data directories")
        click.echo()

        self._process_all(data_dirs)

    def _ensure_species_file(self) -> None:
        """Ensure species file exists before batch processing."""
        if self.species_file.exists():
            return

        if self.raw_species_dir is None:
            raise FileNotFoundError(
                f"Species file '{self.species_file}' not found. "
                f"Use --raw-species-dir to specify PAO/VPS source directories."
            )

        click.echo("[info] Generating species file...")
        from deepx_dock.convert.openmx.species_convert import convert_to_species_h5

        pao_dir = self.raw_species_dir / "PAO"
        vps_dir = self.raw_species_dir / "VPS"

        if not pao_dir.exists():
            raise FileNotFoundError(f"PAO directory not found: {pao_dir}")
        if not vps_dir.exists():
            raise FileNotFoundError(f"VPS directory not found: {vps_dir}")

        self.species_file.parent.mkdir(parents=True, exist_ok=True)
        convert_to_species_h5(pao_dir, vps_dir, self.species_file)
        click.echo()

    def _scan_data_dirs(self):
        """Scan data directories with validation check."""
        validation_check = self._make_validation_check()
        return get_data_dir_lister(self.data_dir, self.tier_num, validation_check)

    def _make_validation_check(self):
        """Create validation check function for data directories."""

        def validation_check(root_dir: Path, prev_dirname: Path):
            all_files = [v.name for v in root_dir.iterdir()]

            has_input = OPENMX_INPUT_FILENAME in all_files
            has_overlap = DEEPX_OVERLAP_FILENAME in all_files

            if has_input:
                if has_overlap and not self.force:
                    print(f"Skip {prev_dirname} (overlap.h5 exists, use --force to overwrite)")
                else:
                    yield prev_dirname
            else:
                print(f"Skip {prev_dirname} (no {OPENMX_INPUT_FILENAME} found)")

        return validation_check

    def _process_all(self, data_dirs: list) -> None:
        """Process all data directories."""
        from deepx_dock.compute.overlap.openmx.calculator import OpenMXOverlapCalculator

        def process_single(relative_dir: Path):
            data_path = self.data_dir / relative_dir
            input_file = data_path / OPENMX_INPUT_FILENAME

            if not input_file.exists():
                return f"[skip] {relative_dir}: no input file"

            try:
                calculator = OpenMXOverlapCalculator(
                    openmx_input=input_file,
                    species_file=self.species_file,
                    raw_species_dir=self.raw_species_dir,
                    force=self.force,
                )
                calculator.run()
                return f"[done] {relative_dir}"
            except Exception as e:
                return f"[error] {relative_dir}: {e}"

        results = parallel_map(
            process_single,
            data_dirs,
            n_jobs=self.n_jobs,
            desc="Processing",
        )

        for result in results:
            if result:
                click.echo(f"  {result}")
