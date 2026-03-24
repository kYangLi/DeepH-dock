import click
from pathlib import Path

from deepx_dock._cli.registry import register
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME


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
            "--ecut", type=float, default=50.0, help="Energy cutoff for Fourier transform (Hartree), default: 50.0"
        ),
        click.option("--force", is_flag=True, help="Force re-conversion of PAO files even if .h5 exists"),
    ],
)
def calc_overlap(openmx_input: str, basis_dir: str, raw_basis_dir: str, ecut: float, force: bool):
    """
    Calculate overlap matrix from OpenMX input file.

    This command automatically:
    1. Parses openmx.in to extract structure and basis definitions
    2. Converts PAO files to standardized basis.h5 (if needed)
    3. Computes overlap matrix using HPRO

    Output files (saved in openmx.in directory):
        - overlap.h5: Overlap matrix
        - info.json: Metadata
        - POSCAR: Crystal structure

    Example:
        # First time: provide PAO files
        dock compute overlap openmx calc openmx.in basis/ --raw-basis-dir pao/

        # Subsequent: use existing basis.h5
        dock compute overlap openmx calc openmx.in basis/
    """
    from deepx_dock.compute.overlap.openmx.calculator import OpenMXOverlapCalculator

    openmx_input = Path(openmx_input)
    basis_dir = Path(basis_dir)
    raw_basis_dir = Path(raw_basis_dir) if raw_basis_dir else None
    output_dir = openmx_input.parent

    click.echo(f"[info] Computing overlap matrix from {openmx_input.name}")
    click.echo(f"[info] Energy cutoff: {ecut} Hartree")

    calculator = OpenMXOverlapCalculator(
        openmx_input=openmx_input, basis_dir=basis_dir, raw_basis_dir=raw_basis_dir, ecut=ecut, force=force
    )

    calculator.run()

    click.echo(f"[done] Results saved to {output_dir}")
    click.echo(f"       - {DEEPX_OVERLAP_FILENAME}")
    click.echo("       - info.json")
    click.echo("       - POSCAR")
