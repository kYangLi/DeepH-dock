import click
from pathlib import Path

from deepx_dock._cli.registry import register
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.compute.overlap.openmx.calculator import (
    OPENMX_DEFAULT_ECUT,
    OPENMX_DEFAULT_KDENSE,
    OPENMX_DEFAULT_RDENSE,
)


@register(
    cli_name="calc",
    cli_help="Calculate overlap matrix from OpenMX input file with automatic basis conversion",
    cli_args=[
        click.argument("openmx_input", type=click.Path(exists=True)),
        click.argument("basis_dir", type=click.Path()),
        click.option(
            "--raw-basis-dir",
            type=click.Path(exists=True),
            default=None,
            help="Directory containing original PAO files for auto-conversion",
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
        click.option("--force", is_flag=True, help="Force re-conversion of PAO files even if .h5 exists"),
    ],
)
def calc_overlap(
    openmx_input: str,
    basis_dir: str,
    raw_basis_dir: str,
    ecut: float,
    kdense: float,
    rdense: float,
    force: bool,
):
    """
    Calculate overlap matrix from OpenMX input file.

    This command automatically:
    1. Parses openmx.in to extract structure and basis definitions
    2. Converts PAO files to standardized basis.h5 (if needed)
    3. Interpolates PAO to uniform linear grid (OpenMX style)
    4. Computes overlap matrix using HPRO

    Output files (saved in openmx.in directory):
        - overlap.h5: Overlap matrix
        - info.json: Metadata
        - POSCAR: Crystal structure

    Defaults match OpenMX:
        --ecut 1800 Ha (OpenMX: 1DFFT.EnergyCutoff=3600 Ry)
        --kdense 15 (OpenMX: NumGridK=900 at kmax=60)
        --rdense 100 (OpenMX: NumGridR=900 for ~9 Bohr cutoff)
    """
    from deepx_dock.compute.overlap.openmx.calculator import OpenMXOverlapCalculator

    openmx_input = Path(openmx_input)
    basis_dir = Path(basis_dir)
    raw_basis_dir = Path(raw_basis_dir) if raw_basis_dir else None
    output_dir = openmx_input.parent

    import numpy as np

    kmax = np.sqrt(2 * ecut)
    grid_nq = int(kmax * kdense)

    click.echo("=" * 60)
    click.echo("[info] OpenMX Overlap Matrix Calculation")
    click.echo("=" * 60)
    click.echo(f"[info] Input file: {openmx_input.name}")
    click.echo()

    click.echo("[info] Grid Parameters (OpenMX defaults):")
    click.echo("-" * 40)
    click.echo(f"  k-space: Ecut={ecut:.0f} Ha, kmax={kmax:.0f} Bohr^-1")
    click.echo(f"           kdense={kdense:.0f}, grid_nq={grid_nq}")
    click.echo(f"  r-space: rdense={rdense:.0f} Bohr^-1 (linear grid)")
    click.echo()

    calculator = OpenMXOverlapCalculator(
        openmx_input=openmx_input,
        basis_dir=basis_dir,
        raw_basis_dir=raw_basis_dir,
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
