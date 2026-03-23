import click
from pathlib import Path
from deepx_dock._cli.registry import register


def _get_supported_versions_str() -> str:
    """Get formatted list of supported FHI-aims versions."""
    from deepx_dock.convert.fhi_aims.patch_aims import AimsPatcher

    versions = AimsPatcher.get_supported_versions()
    if not versions:
        return "  - none"
    return "\n".join(f"  - {v}" for v in versions)


_SUPPORTED_VERSIONS = _get_supported_versions_str()


@register(
    cli_name="patch-aims",
    cli_help="""
Apply a patch to FHI-aims source for DeepX warmstart support (periodic mode only).

\b
  1. Download FHI-aims from https://fhi-aims.org (registration required)
  2. Extract: tar -xzf FHIaims.YYMMDD_N.tar.gz (or *.zip or whatever)
  3. Run: dock convert fhi-aims patch-aims FHIaims.YYMMDD_N
  4. Read DEEPH_WARMSTART_USAGE.md inside FHIaims.YYMMDD_N for build/usage instructions

\b
Supported FHI-aims versions:
{versions}
""".format(versions=_SUPPORTED_VERSIONS),
    cli_args=[
        click.argument(
            "aims_src_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.option(
            "--dry-run", is_flag=True, help="Only validate and show what would be done, without applying patch."
        ),
        click.option("--force", is_flag=True, help="Force re-patch even if already patched."),
    ],
)
def patch_aims(aims_src_dir: Path, dry_run: bool, force: bool):
    aims_src_dir = Path(aims_src_dir)
    from deepx_dock.convert.fhi_aims.patch_aims import AimsPatcher

    patcher = AimsPatcher(aims_src_dir)

    detected_version = patcher._detect_aims_version()
    if detected_version is None:
        raise click.ClickException(f"Cannot detect FHI-aims version from {aims_src_dir}/README.md")

    click.echo(f"[info] Detected FHI-aims version: {detected_version}")

    supported_versions = AimsPatcher.get_supported_versions()
    if detected_version not in supported_versions:
        click.echo(f"[error] Version {detected_version} is not supported")
        if supported_versions:
            click.echo(f"[error] Supported versions: {', '.join(supported_versions)}")
        raise click.ClickException(f"FHI-aims version {detected_version} is not supported")

    patch_file = patcher._find_patch_for_version(detected_version)
    assert patch_file is not None
    click.echo(f"[info] Patch file: {patch_file.name}")
    click.echo(f"[info] Version {detected_version} is supported ✓")

    if dry_run:
        click.echo("[dry-run] Validation passed. Patch would be applied.")
        return

    click.echo("[info] Applying patch...")
    try:
        patcher.apply_patch(force=force)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo("[done] Patch applied successfully!")
    click.echo(f"[info] Patched source: {aims_src_dir}")
    click.echo(f"[info] Usage guide: {patcher.usage_file_path}")


@register(
    cli_name="cluster-to-deeph",
    cli_help="Translate the FHI-aims output data of cluster (currently only supports single atom) calculation to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            "aims_dir",
            type=click.Path(file_okay=False),
        ),
        click.argument(
            "deeph_dir",
            type=click.Path(file_okay=False),
        ),
        click.option(
            "--tier-num",
            "-t",
            type=int,
            default=0,
            help="The tier number of the aims source data, -1 for [aims_dir], 0 for <aims_dir>/<aims_dir>, 1 for <aims_dir>/<tier1>/<data_dirs>, etc.",
        ),
    ],
)
def translate_cluster_aims_to_deeph(aims_dir: Path, deeph_dir: Path, tier_num: int):
    aims_dir = Path(aims_dir)
    deeph_dir = Path(deeph_dir)
    from deepx_dock.convert.fhi_aims.single_atom_aims_to_deeph import SingleAtomDataTranslatorToDeepH

    translator = SingleAtomDataTranslatorToDeepH(aims_dir, deeph_dir, tier_num)
    translator.transfer_all_aims_to_deeph()
    click.echo("[done] Translation completed successfully!")


@register(
    cli_name="periodic-to-deeph",
    cli_help="Translate the FHI-aims output data of periodic structure calculation to DeepH DFT data training set format.",
    cli_args=[
        click.argument(
            "aims_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.argument(
            "deeph_dir",
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
            help="The tier number of the aims source data, -1 for [aims_dir], 0 for <aims_dir>/<aims_dir>, 1 for <aims_dir>/<tier1>/<data_dirs>, etc.",
        ),
        click.option("--force", is_flag=True, help="Force to overwrite the existing files."),
    ],
)
def translate_periodic_aims_to_deeph(
    aims_dir: Path,
    deeph_dir: Path,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    aims_dir = Path(aims_dir)
    deeph_dir = Path(deeph_dir)
    if not aims_dir.is_dir():
        raise click.ClickException(f"AIMS data path '{aims_dir}' is not a directory!")
    if (not force) and deeph_dir.is_dir():
        click.confirm(f"The DeepH data path '{deeph_dir}' already exists. Continue?", abort=True)
    else:
        deeph_dir.mkdir(parents=True, exist_ok=True)
    from deepx_dock.convert.fhi_aims.aims_to_deeph import PeriodicAimsDataTranslator

    translator = PeriodicAimsDataTranslator(
        aims_dir, deeph_dir, export_rho=False, export_r=False, n_jobs=jobs_num, n_tier=tier_num
    )
    translator.transfer_all_aims_to_deeph()
    click.echo("[done] Translation completed successfully!")
