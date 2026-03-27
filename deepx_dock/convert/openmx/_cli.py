import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="to-deeph",
    cli_help="Translate the OpenMX output data to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            "openmx_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            "deeph_dir",
            type=click.Path(file_okay=False),
        ),
        click.option("--ignore-S", is_flag=True, help="Do not export overlap.h5"),
        click.option("--ignore-H", is_flag=True, help="Do not export hamiltonian.h5"),
        click.option("--export-rho", is_flag=True, help="Export density_matrix.h5"),
        click.option("--export-r", is_flag=True, help="Export position_matrix.h5"),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="The parallel processing number, -1 for using all of the cores.",
        ),
        click.option(
            "--tier-num",
            "-t",
            type=int,
            default=0,
            help="The tier number of the OpenMX source data, -1 for [openmx_dir], 0 for <openmx_dir>/<data_dirs>, 1 for <openmx_dir>/<tier1>/<data_dirs>, etc.",
        ),
        click.option("--force", is_flag=True, help="Force to overwrite the existing files."),
    ],
)
def translate_openmx_to_deeph(
    openmx_dir: Path,
    deeph_dir: Path,
    ignore_s: bool,
    ignore_h: bool,
    export_rho: bool,
    export_r: bool,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    openmx_dir = Path(openmx_dir)
    deeph_dir = Path(deeph_dir)
    if not openmx_dir.is_dir():
        raise click.ClickException(f"OpenMX data path '{openmx_dir}' is not a directory!")
    if (not force) and deeph_dir.is_dir():
        click.confirm(f"The DeepH data path '{deeph_dir}' already exists. Continue?", abort=True)
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
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


@register(
    cli_name="from-deeph",
    cli_help="Inject the DeepH predicted Hamiltonian into OpenMX scfout file.",
    cli_args=[
        click.argument(
            "openmx_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            "deeph_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            "output_dir",
            type=click.Path(file_okay=False),
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="The parallel processing number, -1 for using all of the cores.",
        ),
        click.option(
            "--tier-num",
            "-t",
            type=int,
            default=0,
            help="The tier number of the dataset, -1 for [deeph_dir], 0 for <deeph_dir>/<data_dirs>, 1 for <deeph_dir>/<tier1>/<data_dirs>, etc.",
        ),
        click.option("--force", is_flag=True, help="Force to overwrite the existing files."),
    ],
)
def translate_deeph_to_openmx(
    openmx_dir: Path,
    deeph_dir: Path,
    output_dir: Path,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    openmx_dir = Path(openmx_dir)
    deeph_dir = Path(deeph_dir)
    output_dir = Path(output_dir)
    if not openmx_dir.is_dir():
        raise click.ClickException(f"OpenMX data path '{openmx_dir}' is not a directory!")
    if not deeph_dir.is_dir():
        raise click.ClickException(f"DeepH data path '{deeph_dir}' is not a directory!")
    if (not force) and output_dir.is_dir():
        click.confirm(f"The output data path '{output_dir}' already exists. Continue?", abort=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    from deepx_dock.convert.openmx.translate_deeph_to_openmx import DeepHToOpenMXTranslator

    translator = DeepHToOpenMXTranslator(
        openmx_data_dir=openmx_dir,
        deeph_data_dir=deeph_dir,
        output_dir=output_dir,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    translator.transfer_all_deeph_to_openmx()
    click.echo("[done] Translation completed successfully!")


@register(
    cli_name="species-h5",
    cli_help="Convert OpenMX PAO+VPS files to species_openmx_{xc}.h5 format",
    cli_args=[
        click.argument("pao_dir", type=click.Path(exists=True)),
        click.argument("vps_dir", type=click.Path(exists=True)),
        click.argument("output_file", type=click.Path()),
        click.option("--xc-type", type=str, default="PBE19", help="XC functional type (default: PBE19)"),
    ],
)
def convert_species(pao_dir: str, vps_dir: str, output_file: str, xc_type: str):
    """
    Convert OpenMX PAO and VPS files to species_openmx_{xc}.h5 format.

    Creates a single HDF5 file containing all species data:
    - Basis functions (from PAO)
    - Valence density (from PAO)
    - Pseudopotentials (from VPS)

    Example:
        dock convert openmx convert-species \\
            /path/to/DFT_DATA19/PAO \\
            /path/to/DFT_DATA19/VPS \\
            species_openmx_pbe.h5
    """
    from deepx_dock.convert.openmx.species_convert import convert_to_species_h5

    pao_dir = Path(pao_dir)
    vps_dir = Path(vps_dir)
    output_file = Path(output_file)

    xc_short = xc_type.replace("19", "")
    click.echo("=" * 60)
    click.echo(f"[info] Converting OpenMX PAO+VPS to species_openmx_{xc_short.lower()}.h5")
    click.echo("=" * 60)
    click.echo(f"[info] PAO directory: {pao_dir}")
    click.echo(f"[info] VPS directory: {vps_dir}")
    click.echo(f"[info] XC type: {xc_type}")
    click.echo(f"[info] Output file: {output_file}")
    click.echo()

    convert_to_species_h5(pao_dir, vps_dir, output_file, xc_type)

    click.echo("=" * 60)
    click.echo("[done] Species file generated successfully!")
    click.echo("=" * 60)
