import click
from pathlib import Path
from typing import Optional

from deepx_dock._cli.registry import register
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.compute.overlap.openmx.calculator import (
    OPENMX_DEFAULT_ECUT,
    OPENMX_DEFAULT_KDENSE,
    OPENMX_DEFAULT_RDENSE,
)


OPENMX_INPUT_FILENAME = "openmx_in.dat"


@register(
    cli_name="openmx",
    cli_help="Calculate overlap matrix from OpenMX input(s) using species_openmx_{xc}.h5",
    cli_args=[
        click.argument("path", type=click.Path(exists=True)),
        click.argument("species_file", type=click.Path()),
        click.option(
            "--tier-num",
            "-t",
            type=int,
            default=None,
            help="Tier number for batch processing. If provided, PATH is treated as data directory. "
            "-1 for [path] itself, 0 for <path>/<data_dirs>, 1 for <path>/<tier1>/<data_dirs>, etc.",
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="Number of parallel jobs for batch processing. -1 for all cores.",
        ),
        click.option(
            "--raw-species-dir",
            type=click.Path(exists=True),
            default=None,
            help="Directory containing PAO/ and VPS/ subdirectories for auto-generation of species file",
        ),
        click.option(
            "--ecut",
            type=float,
            default=OPENMX_DEFAULT_ECUT,
            help=f"Energy cutoff (Hartree). kmax = sqrt(2*Ecut). Default: {OPENMX_DEFAULT_ECUT} Ha (OpenMX: 3600 Ry)",
        ),
        click.option(
            "--kdense",
            type=float,
            default=OPENMX_DEFAULT_KDENSE,
            help=f"k-space grid density. grid_nq = kmax * kdense. Default: {OPENMX_DEFAULT_KDENSE} (OpenMX: NumGridK=900 at kmax=60)",
        ),
        click.option(
            "--rdense",
            type=float,
            default=OPENMX_DEFAULT_RDENSE,
            help=f"r-space grid density (points/Bohr). Default: {OPENMX_DEFAULT_RDENSE} (OpenMX: NumGridR=900 for ~9 Bohr cutoff)",
        ),
        click.option("--force", is_flag=True, help="Force regenerate/overwrite"),
    ],
)
def overlap_openmx(
    path: str,
    species_file: str,
    tier_num: Optional[int],
    jobs_num: int,
    raw_species_dir: Optional[str],
    ecut: float,
    kdense: float,
    rdense: float,
    force: bool,
):
    """
    Calculate overlap matrix from OpenMX input file(s).

    This command supports two modes:

    \b
    1. Single file mode (default, no --tier-num):
       PATH is the OpenMX input file (e.g., openmx_in.dat).

       Example:
         dock compute overlap openmx ./openmx_in.dat ./species_openmx_pbe.h5

    \b
    2. Batch mode (with --tier-num):
       PATH is a data directory. The command scans subdirectories
       and processes each one containing openmx_in.dat.

       Examples:
         # Process subdirectories
         dock compute overlap openmx ./data ./species_openmx_pbe.h5 -t 0

         # Process current directory
         dock compute overlap openmx ./data ./species_openmx_pbe.h5 -t -1

         # With parallel jobs
         dock compute overlap openmx ./data ./species_openmx_pbe.h5 -t 0 -j 4

    \b
    Species file:
       Requires species_openmx_{xc}.h5 containing basis and pseudopotential data.
       Use --raw-species-dir to auto-generate from PAO/VPS files.

    \b
    Output files (per data point):
       - overlap.h5: Overlap matrix
       - info.json: Metadata
       - POSCAR: Crystal structure

    \b
    Grid parameters (OpenMX defaults):
       --ecut 1800 Ha (OpenMX: 1DFFT.EnergyCutoff=3600 Ry)
       --kdense 15 (OpenMX: NumGridK=900 at kmax=60)
       --rdense 100 (OpenMX: NumGridR=900 for ~9 Bohr cutoff)
    """
    path = Path(path)
    species_file = Path(species_file)
    raw_species_dir = Path(raw_species_dir) if raw_species_dir else None

    if tier_num is None:
        _run_single(path, species_file, raw_species_dir, ecut, kdense, rdense, force)
    else:
        _run_batch(path, species_file, tier_num, jobs_num, raw_species_dir, force)


def _run_single(
    openmx_input: Path,
    species_file: Path,
    raw_species_dir: Optional[Path],
    ecut: float,
    kdense: float,
    rdense: float,
    force: bool,
) -> None:
    """Run single file processing."""
    from deepx_dock.compute.overlap.openmx.calculator import OpenMXOverlapCalculator

    output_dir = openmx_input.parent

    import numpy as np

    kmax = np.sqrt(2 * ecut)
    grid_nq = int(kmax * kdense)

    click.echo("=" * 60)
    click.echo("[info] OpenMX Overlap Calculation (single file)")
    click.echo("=" * 60)
    click.echo(f"[info] Input file: {openmx_input}")
    click.echo(f"[info] Species file: {species_file}")
    click.echo()

    click.echo("[info] Grid Parameters (OpenMX defaults):")
    click.echo("-" * 40)
    click.echo(f"  k-space: Ecut={ecut:.0f} Ha, kmax={kmax:.0f} Bohr^-1")
    click.echo(f"           kdense={kdense:.0f}, grid_nq={grid_nq}")
    click.echo(f"  r-space: rdense={rdense:.0f} Bohr^-1 (linear grid)")
    click.echo()

    calculator = OpenMXOverlapCalculator(
        openmx_input=openmx_input,
        species_file=species_file,
        raw_species_dir=raw_species_dir,
        ecut=ecut,
        kdense=kdense,
        rdense=rdense,
        force=force,
    )

    click.echo("[info] Running calculation...")
    click.echo()

    calculator.run()

    click.echo("=" * 60)
    click.echo(f"[done] Results saved to {output_dir}")
    click.echo(f"       - {DEEPX_OVERLAP_FILENAME}")
    click.echo("       - info.json")
    click.echo("       - POSCAR")
    click.echo("=" * 60)


def _run_batch(
    data_dir: Path,
    species_file: Path,
    tier_num: int,
    n_jobs: int,
    raw_species_dir: Optional[Path],
    force: bool,
) -> None:
    """Run batch processing."""
    from deepx_dock.compute.overlap.batch_calculator import BatchOverlapCalculator

    click.echo("=" * 60)
    click.echo("[info] OpenMX Overlap Calculation (batch)")
    click.echo("=" * 60)
    click.echo(f"[info] Data directory: {data_dir}")
    click.echo(f"[info] Species file: {species_file}")
    click.echo(f"[info] Tier: {tier_num}")
    click.echo(f"[info] Jobs: {n_jobs}")
    click.echo()

    calculator = BatchOverlapCalculator(
        data_dir=data_dir,
        species_file=species_file,
        tier_num=tier_num,
        n_jobs=n_jobs,
        force=force,
        raw_species_dir=raw_species_dir,
    )

    calculator.run()

    click.echo("=" * 60)
    click.echo("[done] Batch processing completed")
    click.echo("=" * 60)
