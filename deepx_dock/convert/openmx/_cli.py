import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="to-deeph",
    cli_help="Translate the OpenMX output data to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            'openmx_dir', type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False),
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
            help="The tier number of the OpenMX source data, -1 for [openmx_dir], 0 for <openmx_dir>/<data_dirs>, 1 for <openmx_dir>/<tier1>/<data_dirs>, etc."
        ),
        click.option(
            '--force', is_flag=True, help="Force to overwrite the existing files."
        ),
    ],
)
def translate_openmx_to_deeph(
    openmx_dir: Path, deeph_dir: Path,
    ignore_s: bool, ignore_h: bool, export_rho: bool, export_r: bool,
    jobs_num: int, tier_num: int, force: bool,
):
    openmx_dir = Path(openmx_dir)
    deeph_dir = Path(deeph_dir)
    if not openmx_dir.is_dir():
        raise click.ClickException(f"OpenMX data path '{openmx_dir}' is not a directory!")
    if (not force) and deeph_dir.is_dir():
        click.confirm(
            f"The DeepH data path '{deeph_dir}' already exists. Continue?",
            abort=True
        )
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.openmx.translate_openmx_to_deeph import OpenMXDatasetTranslator
    translator = OpenMXDatasetTranslator(
        openmx_data_dir=openmx_dir,
        deeph_data_dir=deeph_dir,
        export_S=not ignore_s,
        export_H=not ignore_h,
        export_rho=export_rho,
        export_r=export_r,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    translator.transfer_all_openmx_to_deeph()
    click.echo("[done] Translation completed successfully!")

