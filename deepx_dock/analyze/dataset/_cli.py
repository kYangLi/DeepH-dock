import click
from typing import Optional, Tuple
from pathlib import Path
from deepx_dock._cli.registry import register


COMMON_OPT = [
    click.argument(
        'data_path',
        type=click.Path(exists=True, file_okay=False, readable=True),
    ),
    click.option(
        '--parallel-num', '-p', type=int, default=-1,
        help='The parallel processing number, -1 for using all of the cores.'
    ),
    click.option(
        '--tier-num', '-t', type=int, default=0,
        help='The tier number of the dft data directory, -1 for <data_path>/dft, 0 for <data_path>/dft/<data_dirs>, 1 for <data_path>/dft/<tier1>/<data_dirs>, etc.'
    ),
]


# ------------------------------------------------------------------------------
@register(
    cli_name="features",
    cli_help="Given an input data path containing a `dft/` subdirectory, examine the features of the raw DFT dataset within it.",
    cli_args=COMMON_OPT + [
        click.option(
            '--common-orbital-types', type=str, default=None,
            help='User given common orbital types.'
        ),
        click.option(
            '--consider-parity', is_flag=True,
            help='Consider parity when analysis DFT data features.'
        ),
    ],
)
def analyze_dataset_features(
    data_path: str | Path,
    parallel_num: int,
    tier_num: int,
    common_orbital_types: str,
    consider_parity: bool,
):
    from deepx_dock.analyze.dataset.analyze_dataset import DatasetAnalyzer
    inspector = DatasetAnalyzer(
        data_path=Path(data_path).resolve(),
        n_jobs=parallel_num,
        n_tier=tier_num,
    )
    inspector.analysis_dft_features(
        common_orbital_types=common_orbital_types,
        consider_parity=consider_parity,
    )

# ------------------------------------------------------------------------------
@register(
    cli_name="edge",
    cli_help="Statistic and show edge related information.",
    cli_args=COMMON_OPT + [
        click.option(
            '--edge-bins', type=int, default=None,
            help='The bins count width, default is auto decided.'
        ),
        click.option(
            '--plot-dpi', type=int,
            default=300,
            show_default=True,
            help='The bin count plot DPI.'
        ),
    ],
)
def analyze_dataset_edge(
    data_path: str | Path,
    parallel_num: int,
    tier_num: int,
    edge_bins: Optional[int],
    plot_dpi: int,
):
    from deepx_dock.analyze.dataset.analyze_dataset import DatasetAnalyzer
    inspector = DatasetAnalyzer(
        data_path=Path(data_path).resolve(),
        n_jobs=parallel_num,
        n_tier=tier_num,
    )
    inspector.statistic_edge_quantity(bins=edge_bins)
    inspector.plot_edge_quantity(dpi=plot_dpi)


# ------------------------------------------------------------------------------
@register(
    cli_name="split",
    cli_help="Generate train, validate, and test data split json file.",
    cli_args=COMMON_OPT + [
        click.option(
            '--split-ratio', type=(float,float,float), default=(0.6, 0.2, 0.2),
            help='The train_ratio, validate_ratio and test_ratio of the split data.'
        ),
        click.option(
            '--split-max-edge-num', type=int, default=-1,
            help='The max edge number of the split data, -1 for no constraint.'
        ),
        click.option(
            '--split-rng-seed', type=int, default=137,
            help='The random seed for processing dataset.'
        ),
    ],
)
def analyze_dataset_split(
    data_path: str,
    parallel_num: int,
    tier_num: int,
    split_ratio: Tuple[float, float, float],
    split_max_edge_num: int,
    split_rng_seed: int,
):
    from deepx_dock.analyze.dataset.analyze_dataset import DatasetAnalyzer
    inspector = DatasetAnalyzer(
        data_path=Path(data_path).resolve(),
        n_jobs=parallel_num,
        n_tier=tier_num,
    )
    inspector.generate_data_split_json(
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
        max_edge_num=split_max_edge_num,
        rng_seed=split_rng_seed,
    )

