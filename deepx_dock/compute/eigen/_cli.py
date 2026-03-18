from pathlib import Path
import click
from deepx_dock._cli.registry import register

from deepx_dock.CONSTANT import DEEPX_BAND_FILENAME, DEEPX_K_PATH_FILENAME


@register(
    cli_name="calc-band",
    cli_help="Calculate the energy band and save it into h5 file.",
    cli_args=[
        click.argument(
            "data_path",
            type=click.Path(exists=True, file_okay=False, readable=True),
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="Number of parallel workers. Default: -1 (auto-detect CPU cores).",
        ),
        click.option(
            "--blas-parallel-mode",
            is_flag=True,
            help="Use BLAS multi-threaded mode instead of parallel k-point mode (default).",
        ),
        click.option("--sparse-calc", is_flag=True, help="Use sparse diagonalization."),
        click.option("--num-band", type=int, default=50, help="Number of bands when using sparse diagonalization."),
        click.option(
            "--E-min",
            "--min",
            type=float,
            default=-0.5,
            help="Lowest band energy (from the fermi level) when using sparse diagonalization.",
        ),
        click.option(
            "--maxiter", type=int, default=300, help="Max number of iterations when using sparse diagonalization."
        ),
        click.option(
            "--ill-method",
            type=click.Choice(["none", "window", "orbital"]),
            default="none",
            help="Method to handle ill-conditioned eigenvalues: 'none', 'window' (window regularization), or 'orbital' (orbital removal).",
        ),
        click.option(
            "--ill-threshold", type=float, default=1e-3, help="Threshold for ill-conditioned eigenvalue detection."
        ),
        click.option(
            "--window-emin",
            type=float,
            default=-1000.0,
            help="Minimum energy for window regularization (eV, relative to Fermi).",
        ),
        click.option(
            "--window-emax",
            type=float,
            default=6.0,
            help="Maximum energy for window regularization (eV, relative to Fermi).",
        ),
    ],
)
def calc_band(
    data_path,
    jobs_num,
    blas_parallel_mode,
    num_band,
    e_min,
    maxiter,
    sparse_calc,
    ill_method,
    ill_threshold,
    window_emin,
    window_emax,
):
    data_path = Path(data_path).resolve()
    band_data_path = data_path / DEEPX_BAND_FILENAME
    k_path_path = data_path / DEEPX_K_PATH_FILENAME
    with open(k_path_path, "r") as f:
        k_list_spell = f.read()
    band_conf = {
        "k_list_spell": k_list_spell,
        "num_band": num_band,
        "lowest_band_energy": e_min,
        "maxiter": maxiter,
    }
    from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
    from deepx_dock.compute.eigen.band import BandDataGenerator

    obj_H = HamiltonianObj(data_path)
    bd_gen = BandDataGenerator(obj_H, band_conf)

    ill_method_arg = None if ill_method == "none" else ill_method
    if ill_method_arg == "window":
        ill_method_arg = "window_regularization"
    elif ill_method_arg == "orbital":
        ill_method_arg = "orbital_removal"

    parallel_k = not blas_parallel_mode
    bd_gen.calc_band_data(
        n_jobs=jobs_num,
        parallel_k=parallel_k,
        sparse_calc=sparse_calc,
        ill_method=ill_method_arg,
        ill_threshold=ill_threshold,
        window_emin=window_emin,
        window_emax=window_emax,
    )
    bd_gen.dump_band_data(band_data_path)


@register(
    cli_name="plot-band",
    cli_help="Plot energy band with the h5 file that is calculated already.",
    cli_args=[
        click.argument(
            "data_path",
            type=click.Path(exists=True, file_okay=False, readable=True),
        ),
        click.option(
            "--energy-window",
            "--E-win",
            type=(float, float),
            default=(-5, 5),
            help="Plot band energy window (respect to fermi energy).",
        ),
    ],
)
def plot_band_data(data_path, energy_window):
    data_path = Path(data_path).resolve()
    band_data_path = data_path / DEEPX_BAND_FILENAME
    from deepx_dock.compute.eigen.band import BandPlotter

    bd_plotter = BandPlotter(band_data_path)
    bd_plotter.plot(Emin=energy_window[0], Emax=energy_window[1])


@register(
    cli_name="find-fermi",
    cli_help="Find the Fermi energy using the number of occupied electrons.",
    cli_args=[
        click.argument(
            "data_path",
            type=click.Path(exists=True, file_okay=False, readable=True),
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="Number of parallel workers. Default: -1 (auto-detect CPU cores).",
        ),
        click.option(
            "--blas-parallel-mode",
            is_flag=True,
            help="Use BLAS multi-threaded mode instead of parallel k-point mode (default).",
        ),
        click.option(
            "--method",
            type=click.Choice(["counting", "tetrahedron"]),
            default="counting",
            help="Calculating method that is used for obtaining DOS.",
        ),
        click.option("--kp-density", "-d", type=float, default=0.1, help="The density of the k points."),
        click.option(
            "--cache-res",
            is_flag=True,
            help="Cache the eigenvalues so that you can save time in the subsequent DOS calculation.",
        ),
        click.option(
            "--ill-method",
            type=click.Choice(["none", "window", "orbital"]),
            default="none",
            help="Method to handle ill-conditioned eigenvalues.",
        ),
        click.option(
            "--ill-threshold", type=float, default=1e-3, help="Threshold for ill-conditioned eigenvalue detection."
        ),
        click.option(
            "--window-emin",
            type=float,
            default=-1000.0,
            help="Minimum energy for window regularization (eV, relative to Fermi).",
        ),
        click.option(
            "--window-emax",
            type=float,
            default=6.0,
            help="Maximum energy for window regularization (eV, relative to Fermi).",
        ),
    ],
)
def find_fermi_energy(
    data_path,
    jobs_num,
    blas_parallel_mode,
    method,
    kp_density,
    cache_res,
    ill_method,
    ill_threshold,
    window_emin,
    window_emax,
):
    data_path = Path(data_path).resolve()
    from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
    from deepx_dock.compute.eigen.fermi_dos import FermiEnergyAndDOSGenerator
    from deepx_dock.compute.eigen.ill_conditioned import IllConditionedHandler

    obj_H = HamiltonianObj(data_path)

    ill_handler = None
    if ill_method != "none":
        ill_method_arg = None
        if ill_method == "window":
            ill_method_arg = "window_regularization"
        elif ill_method == "orbital":
            ill_method_arg = "orbital_removal"

        ill_handler = IllConditionedHandler(
            method=ill_method_arg,
            ill_threshold=ill_threshold,
            window_emin=window_emin,
            window_emax=window_emax,
            fermi_energy=obj_H.fermi_energy,
        )

    parallel_k = not blas_parallel_mode
    fd_fermi = FermiEnergyAndDOSGenerator(data_path, obj_H, ill_handler=ill_handler)
    fd_fermi.find_fermi_energy(dk=kp_density, n_jobs=jobs_num, parallel_k=parallel_k, method=method)
    fd_fermi.dump_fermi_energy()
    if cache_res and fd_fermi.eigvals is not None:
        fd_fermi.dump_eigval_data()


@register(
    cli_name="calc-dos",
    cli_help="Calc and plot the density of states.",
    cli_args=[
        click.argument(
            "data_path",
            type=click.Path(exists=True, file_okay=False, readable=True),
        ),
        click.option(
            "--jobs-num",
            "-j",
            type=int,
            default=-1,
            help="Number of parallel workers. Default: -1 (auto-detect CPU cores).",
        ),
        click.option(
            "--blas-parallel-mode",
            is_flag=True,
            help="Use BLAS multi-threaded mode instead of parallel k-point mode (default).",
        ),
        click.option(
            "--method",
            type=click.Choice(["gaussian", "tetrahedron"]),
            default="gaussian",
            help="Calculating method that is used for obtaining DOS.",
        ),
        click.option("--kp-density", "-d", type=float, default=0.1, help="The density of the k points."),
        click.option(
            "--energy-window",
            "--E-win",
            type=(float, float),
            default=(-5, 5),
            help="Plot band energy window (respect to fermi energy).",
        ),
        click.option("--smearing", "-s", type=float, default=-1.0, help="The smearing width (eV) in gaussian method."),
        click.option("--energy-num", "--num", type=int, default=201, help="Number of energy points."),
        click.option(
            "--cache-res",
            is_flag=True,
            help="Cache the eigenvalues so that you can save time in the next same task.",
        ),
        click.option(
            "--ill-method",
            type=click.Choice(["none", "window", "orbital"]),
            default="none",
            help="Method to handle ill-conditioned eigenvalues.",
        ),
        click.option(
            "--ill-threshold", type=float, default=1e-3, help="Threshold for ill-conditioned eigenvalue detection."
        ),
        click.option(
            "--window-emin",
            type=float,
            default=-1000.0,
            help="Minimum energy for window regularization (eV, relative to Fermi).",
        ),
        click.option(
            "--window-emax",
            type=float,
            default=6.0,
            help="Maximum energy for window regularization (eV, relative to Fermi).",
        ),
    ],
)
def calc_dos_from_H(
    data_path,
    jobs_num,
    blas_parallel_mode,
    method,
    kp_density,
    energy_window,
    energy_num,
    smearing,
    cache_res,
    ill_method,
    ill_threshold,
    window_emin,
    window_emax,
):
    data_path = Path(data_path).resolve()
    from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
    from deepx_dock.compute.eigen.fermi_dos import FermiEnergyAndDOSGenerator
    from deepx_dock.compute.eigen.ill_conditioned import IllConditionedHandler

    obj_H = HamiltonianObj(data_path)

    ill_handler = None
    if ill_method != "none":
        ill_method_arg = None
        if ill_method == "window":
            ill_method_arg = "window_regularization"
        elif ill_method == "orbital":
            ill_method_arg = "orbital_removal"

        ill_handler = IllConditionedHandler(
            method=ill_method_arg,
            ill_threshold=ill_threshold,
            window_emin=window_emin,
            window_emax=window_emax,
            fermi_energy=obj_H.fermi_energy,
        )

    parallel_k = not blas_parallel_mode
    fd_fermi = FermiEnergyAndDOSGenerator(data_path, obj_H, ill_handler=ill_handler)
    fermi_method = "counting" if method == "gaussian" else method
    fd_fermi.find_fermi_energy(dk=kp_density, n_jobs=jobs_num, parallel_k=parallel_k, method=fermi_method)
    fd_fermi.dump_fermi_energy()
    fd_fermi.calc_dos(
        dk=kp_density,
        n_jobs=jobs_num,
        parallel_k=parallel_k,
        emin=energy_window[0],
        emax=energy_window[1],
        enum=energy_num,
        method=method,
        sigma=smearing,
    )
    if cache_res:
        fd_fermi.dump_eigval_data()
    fd_fermi.dump_dos_data()
    fd_fermi.plot_dos_data(plot_format="png", dpi=300)
