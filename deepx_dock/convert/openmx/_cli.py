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


@register(
    cli_name="to-openmx",
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
    #
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
    cli_name="convert-basis",
    cli_help="Convert OpenMX PAO files to standardized HDF5 format",
    cli_args=[
        click.argument("pao_dir", type=click.Path(exists=True)),
        click.argument("output_dir", type=click.Path()),
        click.option("--pattern", type=str, default="*.pao", help="File pattern to match PAO files, default: *.pao"),
        click.option("--force", is_flag=True, help="Force re-conversion even if .h5 exists"),
    ],
)
def convert_basis(pao_dir: str, output_dir: str, pattern: str, force: bool):
    """
    Batch convert OpenMX PAO files to standardized HDF5 format.

    Example:
        dock convert openmx convert-basis pao_folder/ basis/
    """
    from deepx_dock.convert.openmx.basis_convert import convert_pao_to_h5
    from tqdm import tqdm

    pao_dir = Path(pao_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pao_files = sorted(pao_dir.glob(pattern))

    if not pao_files:
        click.echo(f"[error] No PAO files found matching pattern '{pattern}' in {pao_dir}")
        return

    click.echo(f"Found {len(pao_files)} PAO file(s) to convert")

    converted = 0
    skipped = 0

    for pao_file in tqdm(pao_files, desc="Converting"):
        h5_file = output_dir / f"{pao_file.stem}.h5"

        if h5_file.exists() and not force:
            skipped += 1
            continue

        try:
            convert_pao_to_h5(pao_file, h5_file)
            converted += 1
        except Exception as e:
            click.echo(f"\n[error] Failed to convert {pao_file.name}: {e}")

    click.echo(f"\n[done] Converted: {converted}, Skipped: {skipped}")
    click.echo(f"Output directory: {output_dir}")


@register(
    cli_name="convert-single",
    cli_help="Convert a single OpenMX PAO file to HDF5",
    cli_args=[
        click.argument("pao_file", type=click.Path(exists=True)),
        click.argument("output_file", type=click.Path()),
    ],
)
def convert_single(pao_file: str, output_file: str):
    """
    Convert a single OpenMX PAO file to standardized HDF5 format.

    Example:
        dock convert openmx convert-single Fe6.0H.pao basis/Fe6.0H.h5
    """
    from deepx_dock.convert.openmx.basis_convert import convert_pao_to_h5

    pao_file = Path(pao_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Converting: {pao_file.name}")
    convert_pao_to_h5(pao_file, output_file)
    click.echo(f"[done] Saved to: {output_file}")
