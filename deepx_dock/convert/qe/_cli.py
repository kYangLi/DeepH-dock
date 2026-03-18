import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="to-deeph",
    cli_help="Convert Quantum ESPRESSO output files to DeepH training data format",
    cli_args=[
        click.argument(
            'espresso_dir', type=click.Path(exists=True, file_okay=False)
        ),
        click.argument(
            'deeph_dir', type=click.Path(file_okay=False)
        ),
        click.option(
            '--basis-dir', type=click.Path(exists=True, file_okay=False), default='',
            help="Path of Quantum ESPRESSO basis files (*.upf) and atomic-orbital basis files (*.ion)."
        ),
        click.option(
            '--ignore-S', is_flag=True,help="Not export overlap.h5"
        ),
        click.option(
            '--ignore-H', is_flag=True,help="Not export hamiltonian.h5"
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
            help="The tier number of the source data, -1 for [source], 0 for <source>/[data_dirs], 1 for <source>/<tier1>/[data_dirs], etc."
        ),
    ],
)
def translate_espresso_to_deeph(
    espresso_dir: Path, deeph_dir: Path, basis_dir: Path,
    ignore_s: bool, ignore_h: bool, export_rho: bool, export_r: bool,
    jobs_num: int, tier_num: int,
):
    if export_rho or export_r:
        raise click.ClickException(
            "The export_rho and export_r options are not supported in the current version of DeepH-dock."
        )
    if not espresso_dir.is_dir():
        raise click.ClickException(
            f"Quantum ESPRESSO data directory '{espresso_dir}' does not exist or is not a directory!"
        )
    if not basis_dir.is_dir():
        raise click.ClickException(
            f"Basis directory '{basis_dir}' does not exist or is not a directory!"
        )
    if deeph_dir.exists():
        click.confirm(
            f"The DeepH data path '{deeph_dir}' already exists. Continue?",
            abort=True
        )
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.qe.translate_espresso_to_deeph import EspressoDatasetTranslator
    translator = EspressoDatasetTranslator(
        espresso_data_dir=espresso_dir,
        deeph_data_dir=deeph_dir,
        basis_dir=basis_dir, # Basis directory is not used in this context
        export_S=not ignore_s,
        export_H=not ignore_h,
        export_rho=export_rho,
        export_r=export_r,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    translator.transfer_all_espresso_to_deeph()
    click.echo("[done] Translation completed successfully!")

