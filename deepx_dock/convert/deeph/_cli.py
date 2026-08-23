import click
from pathlib import Path
from deepx_dock._cli.registry import register


# ------------------------------------------------------------------------------
@register(
    cli_name="standardize",
    cli_help="Standardize DeepH Hamiltonian, so that it can eliminate the mu gauge.",
    cli_args=[
        click.argument(
            "deeph_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="The parallel processing number, -1 for using all of the cores.",
        ),
        click.option("--overwrite", is_flag=True, help="Overwrite the existing Hamiltonian file."),
        click.option(
            "--tier-num",
            "-t",
            type=int,
            default=0,
            help="The tier number of the source data, -1 for [source], 0 for <source>/<data_dirs>, 1 for <source>/<tier1>/<data_dirs>, etc.",
        ),
    ],
)
def standardize_hamiltonian(deeph_dir, jobs_num, overwrite, tier_num):
    deeph_dir = Path(deeph_dir).resolve()
    if not deeph_dir.is_dir():
        raise FileNotFoundError(f"The old data path `{deeph_dir}` dose not exist!")
    from deepx_dock.convert.deeph.standardize_hamiltonian import DatasetHStandardize

    std_obj = DatasetHStandardize(data_dir=deeph_dir, h5_overwrite=overwrite, n_jobs=jobs_num, n_tier=tier_num)
    std_obj.standardize_all()
    click.echo("[done] Translation completed successfully!")


# ------------------------------------------------------------------------------
@register(
    cli_name="minus-core",
    cli_help="Remove or add back the single atomic Hamiltonian, which significantly reduces the range of Hamiltonian values.",
    cli_args=[
        click.argument(
            "full_dir",
            type=click.Path(file_okay=False),
        ),
        click.argument(
            "corrected_dir",
            type=click.Path(file_okay=False),
        ),
        click.argument(
            "single_atoms_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.option(
            "--transform-offsite-blocks",
            is_flag=True,
            help="Estimate and remove the offsite Hamiltonian blocks by single atomic Hamiltonians and offsite overlaps.",
        ),
        click.option("--copy-other-files", is_flag=True, help="Copy other files to the input/output dir."),
        click.option("--backward", is_flag=True, help="Transform back."),
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
            help="The tier number of the source data, -1 for [source], 0 for <source>/<data_dirs>, 1 for <source>/<tier1>/<data_dirs>, etc.",
        ),
    ],
)
def minus_core_hamiltonian(
    full_dir, corrected_dir, single_atoms_dir, transform_offsite_blocks, copy_other_files, backward, jobs_num, tier_num
):
    full_dir = Path(full_dir).resolve()
    corrected_dir = Path(corrected_dir).resolve()
    single_atoms_dir = Path(single_atoms_dir).resolve()
    from deepx_dock.convert.deeph.minus_core_hamiltonian import SingleAtomHamiltonianHandler

    handler = SingleAtomHamiltonianHandler(
        full_dir,
        corrected_dir,
        single_atoms_dir,
        transform_offsite_blocks,
        copy_other_files,
        backward,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    handler.transfer_all()
    click.echo("[done] Translation completed successfully!")


# ------------------------------------------------------------------------------
@register(
    cli_name="upgrade",
    cli_help="Convert data from legacy DeepH-E3/DeepH-2 formats to the updated DeepH-pack specification",
    cli_args=[
        click.argument("legacy_dir", type=click.Path(exists=True, file_okay=False)),
        click.argument("updated_dir", type=click.Path(file_okay=False)),
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
            help="The tier number of the source data, -1 for [legacy], 0 for <legacy>/<data_dirs>, 1 for <legacy>/<tier1>/<data_dirs>, etc.",
        ),
        click.option("--force", is_flag=True, help="Force to overwrite the existing files."),
    ],
)
def translate_old_to_new(
    legacy_dir: str,
    updated_dir: str,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    legacy_dir = Path(legacy_dir).resolve()
    updated_dir = Path(updated_dir).resolve()
    if not legacy_dir.is_dir():
        raise click.ClickException(f"The legacy data path '{legacy_dir}' does not exist or is not a directory!")
    if (not force) and updated_dir.exists():
        click.confirm(f"The updated data path '{updated_dir}' already exists. Continue?", abort=True)
    else:
        updated_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.deeph.translate_old_dataset_to_new import NewDatasetTranslator

    transfer = NewDatasetTranslator(
        old_data_dir=legacy_dir,
        new_data_dir=updated_dir,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    transfer.transfer_all_old_to_new()
    click.echo("[done] Translation completed successfully!")


# ------------------------------------------------------------------------------
@register(
    cli_name="downgrade",
    cli_help="Convert data from updated DeepH-pack format to legacy DeepH-E3/DeepH-2 formats",
    cli_args=[
        click.argument(
            "updated_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.argument("legacy_dir", type=click.Path(file_okay=False)),
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
            help="The tier number of the updated source data, -1 for [updated], 0 for <updated>/<data_dirs>, 1 for <updated>/<tier1>/<data_dirs>, etc.",
        ),
        click.option("--force", is_flag=True, help="Force to overwrite the existing files."),
    ],
)
def translate_new_to_old(
    updated_dir: str,
    legacy_dir: str,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    updated_dir = Path(updated_dir).resolve()
    legacy_dir = Path(legacy_dir).resolve()
    if not updated_dir.is_dir():
        raise click.ClickException(f"The updated data path '{updated_dir}' does not exist or is not a directory!")
    if (not force) and legacy_dir.exists():
        click.confirm(f"The legacy data path '{legacy_dir}' already exists. Continue?", abort=True)
    else:
        legacy_dir.mkdir(parents=True, exist_ok=True)
    #
    from deepx_dock.convert.deeph.translate_old_dataset_to_new import OldDatasetTranslator

    transfer = OldDatasetTranslator(
        old_data_dir=legacy_dir,
        new_data_dir=updated_dir,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    transfer.transfer_all_new_to_old()
    click.echo("[done] Translation completed successfully!")


# ------------------------------------------------------------------------------
@register(
    cli_name="standardize-force",
    cli_help="Shift/unshift per-element reference energies (E0s) for force field data.",
    cli_args=[
        click.argument(
            "data_dir",
            type=click.Path(exists=True, file_okay=False),
        ),
        click.option(
            "--backward",
            is_flag=True,
            help="Unshift: add back E0s to the target file. Requires --e0s-file.",
        ),
        click.option(
            "--filename",
            type=str,
            default="force.h5",
            show_default=True,
            help="Target HDF5 filename within each data directory.",
        ),
        click.option(
            "--e0s-file",
            type=click.Path(exists=True, dir_okay=False),
            default=None,
            help="Path to e0s.json for loading E0s. Required for --backward. "
            "If omitted in forward mode, E0s are computed from the dataset.",
        ),
        click.option(
            "--e0s-output",
            type=click.Path(dir_okay=False),
            default=None,
            help="Path to save e0s.json in forward-compute mode. "
            "Defaults to ./e0s.json in the current working directory.",
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
            help="The tier number of the source data, -1 for [source], 0 for <source>/<data_dirs>, 1 for <source>/<tier1>/<data_dirs>, etc.",
        ),
    ],
)
def standardize_force(data_dir, backward, filename, e0s_file, e0s_output, jobs_num, tier_num):
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"The data path `{data_dir}` does not exist!")
    from deepx_dock.convert.deeph.standardize_force import DatasetForceStandardize

    std_obj = DatasetForceStandardize(
        data_dir=data_dir,
        backward=backward,
        filename=filename,
        e0s_file=e0s_file,
        e0s_output=e0s_output,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    std_obj.standardize_all()
    click.echo("[done] Force standardization completed successfully!")
