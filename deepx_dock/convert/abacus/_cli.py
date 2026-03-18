import click
from pathlib import Path
from typing import Optional
from deepx_dock._cli.registry import register


# ------------------------------------------------------------------------------
@register(
    cli_name="to-deeph",
    cli_help="Translate ABACUS output data to DeepH DFT data training set forma",
    cli_args=[
        click.argument(
            'abacus_dir', type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False),
        ),
        click.option(
            '--abacus-suffix', '-s', '--suffix', type=str, default=None,
            help="Only look for OUT.suffix in abacus_output_dir."
        ),
        click.option(
            '--ignore-S', is_flag=True,
            help="Do not export overlap.h5"
        ),
        click.option(
            '--ignore-H', is_flag=True,
            help="Do not export hamiltonian.h5"
        ),
        click.option(
            '--export-rho', is_flag=True,
            help="Export density_matrix.h5"
        ),
        click.option(
            '--export-r', is_flag=True,
            help="Export position_matrix.h5"
        ),
        click.option(
            "--jobs-num", "-j", type=int, default=-1,
            help="The parallel processing number, -1 for using all of the cores."
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help="The tier number of the ABACUS source data, -1 for [abacus_dir], 0 for <abacus_dir>/<data_dirs>, 1 for <abacus_dir>/<tier1>/<data_dirs>, etc."
        ),
        click.option(
            '--force', is_flag=True, help="Force to overwrite the existing files."
        ),
    ],
)
def translate_abacus_to_deeph(
    abacus_dir: Path, deeph_dir: Path, abacus_suffix: Optional[str],
    ignore_s: bool, ignore_h: bool, export_rho: bool, export_r: bool,
    jobs_num: int, tier_num: int, force: bool,
):
    abacus_dir = Path(abacus_dir)
    deeph_dir = Path(deeph_dir)
    if not abacus_dir.is_dir():
        raise click.ClickException(f"ABACUS data path '{abacus_dir}' is not a directory!")
    if (not force) and deeph_dir.is_dir():
        click.confirm(
            f"The DeepH data path '{deeph_dir}' already exists. Continue?",
            abort=True
        )
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.abacus.translate_abacus_to_deeph import AbacusDatasetTranslator
    translator = AbacusDatasetTranslator(
        abacus_data_dir=abacus_dir,
        deeph_data_dir=deeph_dir,
        abacus_suffix=abacus_suffix,
        export_S=not ignore_s,
        export_H=not ignore_h,
        export_rho=export_rho,
        export_r=export_r,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    translator.transfer_all_abacus_to_deeph()
    click.echo("[done] Translation completed successfully!")

