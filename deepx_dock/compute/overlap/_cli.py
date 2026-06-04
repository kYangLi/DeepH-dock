from pathlib import Path
from typing import Optional

import click

from deepx_dock._cli.registry import register
from deepx_dock.compute.overlap.overlap import (
    OPENMX_DEFAULT_ECUT,
    OPENMX_DEFAULT_KDENSE,
    SIESTA_DEFAULT_ECUT,
    default_ecut,
    default_kdense,
    save_overlap_from_files,
)


@register(
    cli_name="overlap",
    cli_help="Calculate AO overlap matrices from POSCAR and SIESTA/OpenMX basis files using HPRO.",
    cli_default=True,
    cli_args=[
        click.argument("path", type=click.Path(exists=True)),
        click.argument("basis_path", type=click.Path(exists=True, file_okay=False)),
        click.argument("aocode", type=click.Choice(["siesta", "openmx"], case_sensitive=False)),
        click.option(
            "--spinful",
            "--soc",
            "-s",
            is_flag=True,
            help="Write spinful SOC-compatible overlap blocks.",
        ),
        click.option(
            "--output-dir",
            "-o",
            type=click.Path(file_okay=False),
            default=None,
            help="Output directory for single-POSCAR mode. Defaults to the POSCAR directory.",
        ),
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
            "--ecut",
            type=float,
            default=None,
            help=f"Fourier cutoff in Hartree. Defaults: SIESTA={SIESTA_DEFAULT_ECUT}, "
            f"OpenMX={OPENMX_DEFAULT_ECUT}.",
        ),
        click.option(
            "--kdense",
            type=float,
            default=None,
            help=f"Reciprocal radial-grid density. Defaults: SIESTA=HPRO default, OpenMX={OPENMX_DEFAULT_KDENSE}.",
        ),
        click.option("--force", is_flag=True, help="Overwrite an existing overlap.h5 file."),
    ],
)
def overlap(
    path: str,
    basis_path: str,
    aocode: str,
    spinful: bool,
    output_dir: Optional[str],
    tier_num: Optional[int],
    jobs_num: int,
    ecut: Optional[float],
    kdense: Optional[float],
    force: bool,
):
    """
    Calculate overlap matrix files.

    Examples:

    \b
      dock compute overlap path_to_poscar path_to_basis_files siesta
      dock compute overlap path_to_poscar path_to_basis_files siesta -s
      dock compute overlap path_to_poscar path_to_basis_files openmx
      dock compute overlap path_to_poscar path_to_basis_files openmx -s
    """
    path_obj = Path(path)
    basis_path_obj = Path(basis_path)
    aocode = aocode.lower()
    ecut_to_use = default_ecut(aocode) if ecut is None else ecut
    kdense_to_use = default_kdense(aocode) if kdense is None else kdense

    if tier_num is None:
        _run_single(
            path_obj,
            basis_path_obj,
            aocode,
            spinful=spinful,
            output_dir=Path(output_dir) if output_dir is not None else None,
            ecut=ecut_to_use,
            kdense=kdense_to_use,
            force=force,
        )
    else:
        _run_batch(
            path_obj,
            basis_path_obj,
            aocode,
            tier_num=tier_num,
            jobs_num=jobs_num,
            spinful=spinful,
            ecut=ecut_to_use,
            kdense=kdense_to_use,
            force=force,
        )


def _run_single(
    poscar_path: Path,
    basis_path: Path,
    aocode: str,
    *,
    spinful: bool,
    output_dir: Optional[Path],
    ecut: float,
    kdense: Optional[float],
    force: bool,
) -> None:
    if not poscar_path.is_file():
        raise click.ClickException(f"Single-POSCAR mode expects PATH to be a POSCAR file: {poscar_path}")

    actual_output_dir = output_dir if output_dir is not None else poscar_path.parent

    click.echo("[info] HPRO overlap calculation")
    click.echo(f"[info] POSCAR: {poscar_path}")
    click.echo(f"[info] Basis directory: {basis_path}")
    click.echo(f"[info] AO code: {aocode}")
    click.echo(f"[info] Spinful: {spinful}")
    click.echo(f"[info] Ecut: {ecut}")
    click.echo(f"[info] kdense: {kdense if kdense is not None else 'HPRO default'}")

    save_overlap_from_files(
        poscar_path,
        basis_path,
        aocode,
        output_dir=actual_output_dir,
        spinful=spinful,
        ecut=ecut,
        kdense=kdense,
        force=force,
    )

    click.echo(f"[done] Results saved to {actual_output_dir}")
    click.echo("       - POSCAR")
    click.echo("       - info.json")
    click.echo("       - overlap.h5")


def _run_batch(
    data_dir: Path,
    basis_path: Path,
    aocode: str,
    *,
    tier_num: int,
    jobs_num: int,
    spinful: bool,
    ecut: float,
    kdense: Optional[float],
    force: bool,
) -> None:
    from deepx_dock.compute.overlap.batch_calculator import BatchOverlapCalculator

    click.echo("[info] HPRO overlap batch calculation")
    click.echo(f"[info] Data directory: {data_dir}")
    click.echo(f"[info] Basis directory: {basis_path}")
    click.echo(f"[info] AO code: {aocode}")
    click.echo(f"[info] Tier: {tier_num}")
    click.echo(f"[info] Jobs: {jobs_num}")
    click.echo(f"[info] Spinful: {spinful}")
    click.echo(f"[info] Ecut: {ecut}")
    click.echo(f"[info] kdense: {kdense if kdense is not None else 'HPRO default'}")

    calculator = BatchOverlapCalculator(
        data_dir=data_dir,
        basis_path=basis_path,
        aocode=aocode,
        tier_num=tier_num,
        n_jobs=jobs_num,
        spinful=spinful,
        ecut=ecut,
        kdense=kdense,
        force=force,
    )
    calculator.run()
    click.echo("[done] Batch processing completed")
