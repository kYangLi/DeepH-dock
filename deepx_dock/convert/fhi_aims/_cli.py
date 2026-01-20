import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="single-atom-to-deeph",
    cli_help="Translate the FHI-aims output data of single atom calculation to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            'aims_dir', type=click.Path(file_okay=False),
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False),
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help='The tier number of the aims source data, -1 for [aims_dir], 0 for <aims_dir>/<aims_dir>, 1 for <aims_dir>/<tier1>/<data_dirs>, etc.'
        ),
    ],
)
def translate_vasp_to_deeph(aims_dir: Path, deeph_dir: Path, tier_num: int):
    aims_dir = Path(aims_dir)
    deeph_dir = Path(deeph_dir)
    #
    from deepx_dock.convert.fhi_aims.single_atom_aims_to_deeph import SingleAtomDataTranslatorToDeepH
    translator = SingleAtomDataTranslatorToDeepH(
        aims_dir, deeph_dir, tier_num
    )
    translator.transfer_all_aims_to_deeph()
    click.echo("[done] Translation completed successfully!")

@register(
    cli_name="periodic-to-deeph",
    cli_help="Translate the FHI-aims output data of periodic structure calculation to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            'aims_dir', type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False),
        ),
        click.option(
            '--parallel-num', '-p', type=int, default=-1,
            help="The parallel processing number, -1 for using all of the cores."
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help='The tier number of the aims source data, -1 for [aims_dir], 0 for <aims_dir>/<aims_dir>, 1 for <aims_dir>/<tier1>/<data_dirs>, etc.'
        ),
        click.option(
            '--force', is_flag=True, help="Force to overwrite the existing files."
        ),
    ],
)
def translate_aims_to_deeph(
    aims_dir: Path, deeph_dir: Path,
    parallel_num: int, tier_num: int, force: bool,
):
    aims_dir = Path(aims_dir)
    deeph_dir = Path(deeph_dir)
    if not aims_dir.is_dir():
        raise click.ClickException(f"AIMS data path '{aims_dir}' is not a directory!")
    if (not force) and deeph_dir.is_dir():
        click.confirm(
            f"The DeepH data path '{deeph_dir}' already exists. Continue?",
            abort=True
        )
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.fhi_aims.aims_to_deeph import PeriodicAimsDataTranslator
    translator = PeriodicAimsDataTranslator(
        aims_dir, deeph_dir, export_rho=False, export_r=False,
        n_jobs=parallel_num, n_tier=tier_num
    )
    translator.transfer_all_aims_to_deeph()
    click.echo("[done] Translation completed successfully!")
