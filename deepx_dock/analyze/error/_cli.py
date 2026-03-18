import click
from typing import Optional, Tuple, List
from deepx_dock._cli.registry import register


COMMON_OPT = [
    click.argument(
        'predicted_dft_dir',
        type=click.Path(exists=True, file_okay=False, readable=True),
    ),
    click.option(
        '--benchmark-dft-dir', '-b', '--bm',
        type=click.Path(file_okay=False, readable=True),
        default='',
        help='Path of the benchmark DFT data directory. If not provided, it will be set as `predicted_dft_dir`.'
    ),
    click.option(
        '--target', 
        type=click.Choice(['H', 'Rho']),
        default='H',
        help='The target in the benchmark DFT data directory: "H" for hamiltonian.h5. "Rho" for density_matrix.h5.'
    ),
    click.option(
        '--not-standardize-gauge', is_flag=True,
        help='Whether to use overlap to correct the chemical potential.'
    ),
    click.option(
        '--ignore-overlap-mask', is_flag=True,
        help='Whether to ignore the overlaps value mask.',
    ),
    click.option(
        '--cache-res', is_flag=True, help='Cache the analysis results.'
    ),
    click.option(
        "--jobs-num", "-j", type=int, default=1,
        help='The number of parallel processes.'
    ),
    click.option(
        '--tier-num', '-t', type=int, default=0,
        help='The tier number of the source data, -1 for <source>, 0 for <source>/<data_dirs>, 1 for <source>/<tier1>/<data_dirs>, etc.'
    ),
    click.option(
        '--plot-dpi', '--dpi', type=int, default=300,
        help='The plot figure DPI.'
    ),
    click.option(
        '--data-split-json', type=click.Path(exists=True), default=None,
        help='The data split json file that indicate which belongs to train, validate, and test.'
    ),
    click.option(
        '--data-split-tags', type=str, default="train,validate,test",
        help='The data split tags that indicate use which data set to analysis.'
    ),
]


# ------------------------------------------------------------------------------
@register(
    cli_name="entries",
    cli_help="Error distribution for each entries with scatter figure.",
    cli_args=COMMON_OPT + [
        click.option(
            '--entries-range', '--xlim', type=(float, float), default=None,
            help='The range of entries to analysis and show.'
        ),
        click.option(
            '--entries-abs-err-range', '--y1lim',
            type=(float, float), default=None,
            help='The range of absolute error to analysis and show.'
        ),
        click.option(
            '--entries-rel-err-range', '--y2lim',
            type=(float, float), default=None,
            help='The range of relative error to analysis and show.'
        ),
        click.option(
            '--plot-heatmap', '--heatmap', is_flag=True,
            help='Enable the heatmap mode for error distribution.'
        ),
        click.option(
            '--heatmap-bucket-size', type=int, default=500,
            help='The bin (bucket) size for heatmap error distribution.'
        ),
    ],
)
def analyze_error_entries(
    predicted_dft_dir: str, benchmark_dft_dir: str, target: str,
    cache_res: bool, jobs_num: int, tier_num: int, plot_dpi: int,
    data_split_json: Optional[str], data_split_tags: str,
    not_standardize_gauge: bool, ignore_overlap_mask: bool,
    entries_range: Optional[Tuple[float, float]],
    entries_abs_err_range: Optional[Tuple[float, float]],
    entries_rel_err_range: Optional[Tuple[float, float]],
    plot_heatmap: bool, heatmap_bucket_size: int,
):
    from deepx_dock.analyze.error.with_infer_res import ErrorEachEntriesDistributionAnalyzer
    #
    dist = ErrorEachEntriesDistributionAnalyzer(
        pred_dft_dir=predicted_dft_dir,
        bm_dft_dir=benchmark_dft_dir,
        target_name=target,
        standardize_gauge=(not not_standardize_gauge),
        consider_overlap_mask=(not ignore_overlap_mask),
        data_split_json=data_split_json,
        data_split_tags=data_split_tags,
        cache_res=cache_res,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(eV)' if target == 'H' else ''
    #
    if plot_heatmap:
        dist.heatmap_statistic(
            entries_bucket_size=heatmap_bucket_size,
            errs_bucket_size=heatmap_bucket_size,
            entries_range=entries_range,
            abs_errs_range=entries_abs_err_range,
            rel_errs_range=entries_rel_err_range,
        )
        dist.plot_heatmap(plot_dpi=plot_dpi, unit=unit)
        return
    dist.plot_scatter(
        plot_dpi=plot_dpi,
        x_lim=entries_range,
        y_abs_lim=entries_abs_err_range,
        y_rel_lim=entries_rel_err_range,
        unit=unit,
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="orbital",
    cli_help="Error distribution for each elements orbital pair.",
    cli_args=COMMON_OPT + [
        click.option(
            '--pred-only', is_flag=True,
            help='Only analysis the predicted Hamiltonian, which means we will show the value distribution of `abs(H_pred)` instead `abs(H_pred - H_bm)`.'
        ),
        click.option(
            '--onsite-only', is_flag=True,
            help='Only analysis the onsite (self atom loop) values.'
        ),
        click.option(
            '--plot-z-range', '--zrange',type=(float, float), default=None,
            help='The lower and upper limit for plotting color map.'
        ),
        click.option(
            '--log-scale', is_flag=True, help='Plot using the log scale.'
        ),
    ],
)
def analyze_error_orbital(
    predicted_dft_dir: str, benchmark_dft_dir: str, target: str,
    cache_res: bool, jobs_num: int, tier_num: int, plot_dpi: int,
    data_split_json: Optional[str], data_split_tags: str,
    not_standardize_gauge: bool, ignore_overlap_mask: bool,
    pred_only: bool, onsite_only: bool, plot_z_range: List[float],
    log_scale: bool,
):
    from deepx_dock.analyze.error.with_infer_res import ErrorOrbitalResoluteDistributionAnalyzer
    #
    dist = ErrorOrbitalResoluteDistributionAnalyzer(
        pred_dft_dir=predicted_dft_dir,
        bm_dft_dir=benchmark_dft_dir,
        target_name=target,
        standardize_gauge=(not not_standardize_gauge),
        consider_overlap_mask=(not ignore_overlap_mask),
        data_split_json=data_split_json,
        data_split_tags=data_split_tags,
        cache_res=cache_res,
        pred_only=pred_only,
        onsite_only=onsite_only,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_vmax=plot_z_range[0] if plot_z_range is not None else None,
        plot_vmin=plot_z_range[1] if plot_z_range is not None else None,
        plot_with_log_scale=log_scale,
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="element",
    cli_help="Element distribution analysis (from inference results).",
    cli_args=COMMON_OPT + [
        click.option(
            '--plot-elem-range', '--E-range',
            type=(float, float), default=(0.2, 1.0),
            help='The range of energy error to analysis and show.'
        ),
    ],
)
def analyze_error_element(
    predicted_dft_dir: str, benchmark_dft_dir: str, target: str,
    cache_res: bool, jobs_num: int, tier_num: int, plot_dpi: int,
    data_split_json: Optional[str], data_split_tags: str,
    not_standardize_gauge: bool, ignore_overlap_mask: bool,
    plot_elem_range: Tuple[float, float],
):
    from deepx_dock.analyze.error.with_infer_res import ErrorElementsDistributionAnalyzer
    #
    dist = ErrorElementsDistributionAnalyzer(
        pred_dft_dir=predicted_dft_dir,
        bm_dft_dir=benchmark_dft_dir,
        target_name=target,
        standardize_gauge=(not not_standardize_gauge),
        consider_overlap_mask=(not ignore_overlap_mask),
        data_split_json=data_split_json,
        data_split_tags=data_split_tags,
        cache_res=cache_res,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
        E_min=plot_elem_range[0],
        E_max=plot_elem_range[1],
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="element-logfile",
    cli_help="Element distribution analysis (from training log file).",
    cli_args=[
        click.argument(
            'log_file_path',
            type=click.Path(exists=True, dir_okay=False, readable=True),
        ),
        click.option(
            '--dft-dir',
            type=click.Path(exists=True, file_okay=False, readable=True),
            default="",
            help='Path of the DFT data directory for reference.'
        ),
        click.option(
            '--target', 
            type=click.Choice(['H', 'Rho']),
            default='H',
            help='The target in the benchmark DFT data directory: "H" for hamiltonian.h5. "Rho" for density_matrix.h5.'
        ),
        click.option(
            '--cache-res', is_flag=True, help='Cache the analysis results.'
        ),
        click.option(
            "--jobs-num", "-j", type=int, default=1,
            help='The number of parallel processes.'
        ),
        click.option(
            '--plot-dpi', '--dpi', type=int, default=300,
            help='The plot figure DPI.'
        ),
        click.option(
            '--plot-elem-range', '--E-range',
            type=(float, float), default=(0.2, 1.0),
            help='The range of energy error to analysis and show.'
        ),
    ],
)
def analyze_error_element_logfile(
    log_file_path: str,
    dft_dir: str,
    target: str,
    cache_res: bool,
    jobs_num: int,
    plot_dpi: int,
    plot_elem_range: Tuple[float, float],
):
    from deepx_dock.analyze.error.with_train_log import ErrorElementsDistAnalyzerWithLog
    #
    dist = ErrorElementsDistAnalyzerWithLog(
        log_file_path=log_file_path,
        dft_dir=dft_dir if dft_dir else None,
        cache_res=cache_res,
        n_jobs=jobs_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
        E_min=plot_elem_range[0],
        E_max=plot_elem_range[1],
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="element-pair",
    cli_help="Element pair distribution analysis.",
    cli_args=COMMON_OPT + [
        click.option(
            '--pred-only', is_flag=True,
            help='Only analysis the predicted Hamiltonian.'
        ),
        click.option(
            '--onsite-only', is_flag=True,
            help='Only analysis the onsite (self atom loop) values.'
        ),
        click.option(
            '--plot-z-range', '--zrange',type=(float, float), default=None,
            help='The upper limit for plotting color map.'
        ),
        click.option(
            '--log-scale', is_flag=True, help='Plot using the log scale.'
        ),
    ],
)
def analyze_error_element_pair(
    predicted_dft_dir: str, benchmark_dft_dir: str, target: str,
    cache_res: bool, jobs_num: int, tier_num: int,
    plot_dpi: int, data_split_json: Optional[str], data_split_tags: str,
    not_standardize_gauge: bool, ignore_overlap_mask: bool,
    pred_only: bool, onsite_only: bool, plot_z_range: List[float],
    log_scale: bool,
):
    from deepx_dock.analyze.error.with_infer_res import ErrorElementsPairDistributionAnalyzer
    #
    dist = ErrorElementsPairDistributionAnalyzer(
        pred_dft_dir=predicted_dft_dir,
        bm_dft_dir=benchmark_dft_dir,
        target_name=target,
        standardize_gauge=(not not_standardize_gauge),
        consider_overlap_mask=(not ignore_overlap_mask),
        data_split_json=data_split_json,
        data_split_tags=data_split_tags,
        cache_res=cache_res,
        pred_only=pred_only,
        onsite_only=onsite_only,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_vmax=plot_z_range[0] if plot_z_range is not None else None,
        plot_vmin=plot_z_range[1] if plot_z_range is not None else None,
        plot_with_log_scale=log_scale,
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="structure",
    cli_help="Structure distribution analysis (from inference results).",
    cli_args=COMMON_OPT + [
        click.option(
            '--xlims',
            type=(float, float),
            default=None,
            help='The range of energy error to analysis and show.'
        ),
        click.option(
            '--ylims',
            type=(float, float),
            default=None,
            help='The range of density to analysis and show.'
        ),
    ],
)
def analyze_error_structure(
    predicted_dft_dir: str, benchmark_dft_dir: str,
    target: str, cache_res: bool, jobs_num: int, tier_num: int,
    plot_dpi: int, data_split_json: Optional[str], data_split_tags: str,
    not_standardize_gauge: bool, ignore_overlap_mask: bool,
    xlims: Optional[Tuple[float, float]], ylims: Optional[Tuple[float, float]],
):
    from deepx_dock.analyze.error.with_infer_res import ErrorStructureDistributionAnalyzer
    #
    dist = ErrorStructureDistributionAnalyzer(
        pred_dft_dir=predicted_dft_dir,
        bm_dft_dir=benchmark_dft_dir,
        target_name=target,
        standardize_gauge=(not not_standardize_gauge),
        consider_overlap_mask=(not ignore_overlap_mask),
        data_split_json=data_split_json,
        data_split_tags=data_split_tags,
        cache_res=cache_res,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
        xlims=xlims,
        ylims=ylims,
    )


# ------------------------------------------------------------------------------
@register(
    cli_name="structure-logfile",
    cli_help="Structure distribution analysis (from training log file).",
    cli_args=[
        click.argument(
            'log_file_path',
            type=click.Path(exists=True, dir_okay=False, readable=True),
        ),
        click.option(
            '--target', 
            type=click.Choice(['H', 'Rho']),
            default='H',
            help='The target in the benchmark DFT data directory: "H" for hamiltonian.h5. "Rho" for density_matrix.h5.'
        ),
        click.option(
            '--cache-res', is_flag=True, help='Cache the analysis results.'
        ),
        click.option(
            '--plot-dpi', '--dpi', type=int, default=300,
            help='The plot figure DPI.'
        ),
        click.option(
            '--xlims',
            type=(float, float),
            default=None,
            help='The range of energy error to analysis and show.'
        ),
        click.option(
            '--ylims',
            type=(float, float),
            default=None,
            help='The range of density to analysis and show.'
        ),
    ],
)
def analyze_error_structure_logfile(
    log_file_path: str,
    target: str,
    cache_res: bool,
    plot_dpi: int,
    xlims: Optional[Tuple[float, float]],
    ylims: Optional[Tuple[float, float]],
):
    from deepx_dock.analyze.error.with_train_log import ErrorStructureDistributionAnalyzerWithLog
    #
    dist = ErrorStructureDistributionAnalyzerWithLog(
        log_file_path=log_file_path,
        cache_res=cache_res,
    )
    #
    dist.analyze_all()
    #
    unit = '(meV)' if target == 'H' else ''
    scale = 1000.0 if target == 'H' else 1.0
    #
    dist.plot(
        plot_dpi=plot_dpi,
        unit=unit,
        scale=scale,
        xlims=xlims,
        ylims=ylims,
    )

