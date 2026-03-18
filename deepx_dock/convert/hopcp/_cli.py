import click
from pathlib import Path
from deepx_dock._cli.registry import register


# ------------------------------------------------------------------------------
@register(
    cli_name="how-to-install-petsc",
    cli_help="Create PETSc installation template directory",
    cli_args=[
        click.argument('target_dir', type=click.Path(path_type=Path)),
        click.option(
            '--force', is_flag=True,
            help='Force overwrite if target directory already contains files.'
        ),
    ],
)
def establish_petsc_install_folder(target_dir: Path, force: bool,):
    target_dir = target_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            click.confirm(
                f"Target directory '{target_dir}' is not empty. Continue and overwrite?",
                abort=True
            )
    #
    from deepx_dock.convert.hopcp.establish_petsc_install_dir import copy_petsc_install_folder
    copy_petsc_install_folder(target_path=target_dir)
    click.echo("[done] PETSc installation template created successfully!")


# ------------------------------------------------------------------------------
@register(
    cli_name="from-deeph",
    cli_help="Convert DeepH h5 data files to PETSc format",
    cli_args=[
        click.argument('deeph_dir', type=click.Path()),
        click.argument('petsc_dir', type=click.Path()),
        click.option('--ignore-S', is_flag=True, help="Do not export SR.petsc"),
        click.option('--ignore-H', is_flag=True, help="Do not export HR.petsc"),
        click.option(
            "--jobs-num", "-j", type=int, default=-1,
            help="The parallel processing number, -1 for using all of the cores."
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help="The tier number of the source data, -1 for [deeph_dir], 0 for <deeph_dir>/[data_dirs], 1 for <deeph_dir>/<tier1>/[data_dirs], etc."
        ),
        click.option(
            '--force', is_flag=True, help="Force to overwrite the existing files."
        ),
    ],
)
def translate_deeph_to_petsc(
    deeph_dir: Path,
    petsc_dir: Path,
    ignore_s: bool,
    ignore_h: bool,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    deeph_dir = Path(deeph_dir)
    petsc_dir = Path(petsc_dir)
    if not deeph_dir.is_dir():
        raise click.ClickException(f"DeepH data directory '{deeph_dir}' does not exist or is not a directory!")
    if (not force) and petsc_dir.exists():
        click.confirm(
            f"The PETSc output path '{petsc_dir}' already exists. Continue?",
            abort=True
        )
    else:
        petsc_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.hopcp.translate_deeph_to_petsc import DeepHtoPETScTranslator
    transfer = DeepHtoPETScTranslator(
        deeph_dir=deeph_dir,
        petsc_dir=petsc_dir,
        export_S=not ignore_s,
        export_H=not ignore_h,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    transfer.transfer_all_deeph_to_petsc()
    click.echo("[done] DeepH to PETSc conversion completed successfully!")

