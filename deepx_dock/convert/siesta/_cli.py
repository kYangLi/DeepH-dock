import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="to-deeph",
    cli_help="Translate SIESTA output data to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            'siesta_dir', type=click.Path(exists=True, file_okay=False)
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False)
        ),
        click.option(
            '--ignore-S', is_flag=True, help="Do not export overlap.h5"
        ),
        click.option(
            '--ignore-H', is_flag=True, help="Do not export hamiltonian.h5"
        ),
        click.option(
            '--export-rho', is_flag=True, help="Export density_matrix.h5"
        ),
        click.option(
            '--export-r', is_flag=True, help="Export position_matrix.h5"
        ),
        click.option(
            "--jobs-num", "-j", type=int, default=-1,
            help="The parallel processing number, -1 for using all of the cores."
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help="The tier number of the SIESTA source data, -1 for [siesta_dir], 0 for <siesta_dir>/[data_dirs], 1 for <siesta_dir>/<tier1>/[data_dirs], etc."
        ),
        click.option(
            '--force', is_flag=True, help="Force to overwrite the existing files."
        ),
    ],
)
def translate_siesta_to_deeph(
    siesta_dir: Path,
    deeph_dir: Path,
    ignore_s: bool,
    ignore_h: bool,
    export_rho: bool,
    export_r: bool,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    siesta_dir = Path(siesta_dir)
    deeph_dir = Path(deeph_dir)
    if not siesta_dir.is_dir():
        raise click.ClickException(
            f"SIESTA data directory '{siesta_dir}' does not exist or is not a directory!"
        )
    if (not force) and deeph_dir.exists():
        click.confirm(
            f"The DeepH data path '{deeph_dir}' already exists. Continue?",
            abort=True
        )
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.siesta.translate_siesta_to_deeph import SIESTADatasetTranslator
    translator = SIESTADatasetTranslator(
        siesta_data_dir=siesta_dir,
        deeph_data_dir=deeph_dir,
        export_S=not ignore_s,
        export_H=not ignore_h,
        export_rho=export_rho,
        export_r=export_r,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    translator.transfer_all_siesta_to_deeph()
    click.echo("[done] Translation completed successfully!")

