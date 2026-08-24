import numpy as np
from pathlib import Path
import json
import h5py
import matplotlib.pyplot as plt
import libtetrabz
from collections import Counter

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.misc import dump_json_file
from deepx_dock.CONSTANT import FERMI_ENERGY_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_EIGVAL_FILENAME, DEEPX_DOS_FILENAME
from deepx_dock.CONSTANT import KELVIN_TO_EV


GAUSSIAN_NEGLECT_FACTOR = 9.0
FERMI_NEGLECT_FACTOR = 50.0
DELTA_ENERGY = 0.1


def gaussian_smearing(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)


libtetrabz_info = """
+----------------------------------------------------+
| The tetrahedron method calls libtetrabz package    |
| https://github.com/mitsuaki1987/libtetrabz         |
| https://doi.org/10.1103/PhysRevB.89.094515         |
+----------------------------------------------------+
"""


class FermiEnergyAndDOSGenerator:
    """
    Fermi energy and density of states (DOS) generator.

    Parameters
    ----------
    data_path : str or Path
        Operating directory.
    obj_H : HamiltonianObj
        The Hamiltonian object to diagonalize.
    ill_handler : IllConditionedHandler, optional
        Handler for ill-conditioned eigenvalues. Default: None.
    """

    def __init__(
        self,
        data_path: str | Path,
        obj_H: HamiltonianObj,
        ill_handler=None,
    ):
        self.data_path = Path(data_path)
        self.obj_H = obj_H
        self.ill_handler = ill_handler
        self._parse_obj_H_info()
        self.eigvals = None
        self.fermi_energy = None
        self.partial_spectrum = False

    def _parse_obj_H_info(self):
        self.reciprocal_lattice = self.obj_H.reciprocal_lattice
        self.band_quantity = self.obj_H.orbits_quantity * (1 + self.obj_H.spinful)
        self.occupation = self.obj_H.occupation
        self.spinful = self.obj_H.spinful

    @property
    def _is_metal(self) -> bool:
        if self.eigvals is None or self.fermi_energy is None:
            raise ValueError("Eigenvalues or Fermi energy not calculated yet.")

        eigvals_flat = self.eigvals.reshape(self.band_quantity, -1)
        k_occupy_counts = np.sum(eigvals_flat < self.fermi_energy, axis=0)

        stats = Counter(k_occupy_counts)
        if len(stats) == 1:
            return False

        THRESHOLD = 0.05
        min_count = max(1, int(THRESHOLD * self.nktot / len(stats)))
        significant_cate = sum(count > min_count for count in stats.values())
        if significant_cate == 0:
            raise ValueError("Band occupation statistics show inconsistent patterns")
        return significant_cate > 1

    def find_fermi_energy(
        self, k_mesh=(-1,-1,-1), dk=0.02, n_jobs=-1, parallel_k=True, method="counting", force_recalc=False, **kwargs
    ):
        """
        Find the Fermi energy on a uniform k-mesh.

        The cached ``fermi_energy.json`` is reused unless ``force_recalc`` is
        set. If occupation is unknown, the Fermi energy from info.json is used
        as a fallback. Otherwise the eigenvalues are computed on the k-mesh and
        the Fermi energy is determined from the occupation.

        Parameters
        ----------
        k_mesh : tuple of int, optional
            Number of k points along three reciprocal axes. Values <= 0 switch
            to ``dk``. Default: (-1, -1, -1).
        dk : float, optional
            k-point spacing used when ``k_mesh`` is not given. Default: 0.02.
        n_jobs : int, optional
            Number of parallel workers. Default: -1 (auto-detect).
        parallel_k : bool, optional
            Parallelization across k points, see ``HamiltonianObj.diag``.
            Default: True.
        method : str, optional
            - 'counting': count energy levels (default),
            - 'fermi': Fermi-Dirac smearing,
            - 'tetrahedron': tetrahedron method from libtetrabz.
        force_recalc : bool, optional
            Ignore the ``fermi_energy.json`` cache. Default: False.
        **kwargs
            Extra options for Hamiltonian diagonalization and find-fermi.
            - For method 'fermi': ``temperature`` (K), ``n_elect_thres``,
            ``fermi_thres``, ``max_iter``.
        """
        fermi_json = self.data_path / FERMI_ENERGY_INFO_FILENAME
        if fermi_json.exists() and (not force_recalc):
            with open(fermi_json, "r") as f:
                self.fermi_energy = json.load(f).get("fermi_energy_eV", None)
                print(f"Use cached fermi energy from {FERMI_ENERGY_INFO_FILENAME}")
        elif self.occupation is None:
            if self.obj_H.fermi_energy is not None:
                print("Fermi energy can't be determined because occupation is None, use the original one in info.json")
                self.fermi_energy = self.obj_H.fermi_energy
            else:
                raise ValueError("Can't determine fermi energy because both occupation and original fermi energy are None")
        if self.fermi_energy is not None:
            return
        if self.band_quantity * (2.0 - self.spinful) < self.occupation:
            raise ValueError(
                f"occupation ({self.occupation}) is larger than the number of bands ({self.band_quantity * (2.0 - self.spinful)})"
            )
        self._calc_eigvals_on_mesh(k_mesh=k_mesh, dk=dk, n_jobs=n_jobs, parallel_k=parallel_k, **kwargs)
        self._find_fermi_energy(self.eigvals, method, **kwargs)

    def _find_fermi_energy(self, eigvals, method, **kwargs):
        print(f"Determining fermi energy with method={method} ...")
        if "counting" == method:
            eigvals_flattened = eigvals.flatten()
            ifermi = round((self.nktot * self.occupation) / (2 - self.spinful))
            val_N = np.partition(eigvals_flattened, ifermi - 1)[ifermi - 1]
            val_Np1 = np.partition(eigvals_flattened, ifermi)[ifermi]
            self.fermi_energy = (val_N + val_Np1) / 2.0
        elif "fermi" == method:
            temperature = kwargs.get("temperature", 300.0)
            n_elect_thres = kwargs.get("n_elect_thres", 1e-6)
            fermi_thres = kwargs.get("fermi_thres", 1e-6)
            max_iter = kwargs.get("max_iter", 1000)
            print(f"  Temperature={temperature}K")
            #
            n_elect = (self.nktot * self.occupation) / (2.0 - self.spinful)
            temp_eV = temperature * KELVIN_TO_EV
            eigvals_flattened = eigvals.flatten()
            ifermi = round(n_elect)
            val_N = np.partition(eigvals_flattened, ifermi - 1)[ifermi - 1]
            val_Np1 = np.partition(eigvals_flattened, ifermi)[ifermi]
            E_fermi_0 = (val_N + val_Np1) / 2.0
            #
            E_min = E_fermi_0 - FERMI_NEGLECT_FACTOR * temp_eV
            E_max = E_fermi_0 + FERMI_NEGLECT_FACTOR * temp_eV
            full_occ = np.sum(eigvals_flattened <= E_min)
            eigvals_flattened = eigvals_flattened[np.logical_and(
                eigvals_flattened > E_min, eigvals_flattened < E_max
            )]
            #
            step = 1
            for _ in range(max_iter):
                E_mid = (E_min + E_max) / 2.0
                delta_n_elect = np.sum(
                    1.0 / (1.0 + np.exp((eigvals_flattened - E_mid) / temp_eV))
                ) + full_occ - n_elect
                if delta_n_elect < 0.0:
                    E_min = E_mid
                else:
                    E_max = E_mid
                if (abs(delta_n_elect) < n_elect_thres) and (E_max - E_min < fermi_thres):
                    break
                step += 1
            if step == max_iter + 1:
                message = f"  Did NOT find Fermi energy after {max_iter} bisection steps.\n  E_min = {E_min} eV, E_max = {E_max} eV, dN_ele = {delta_n_elect}"
                print(message)
                raise ValueError(message)
            else:
                print(f"  Find Fermi energy after {step} bisection steps.\n  E_min = {E_min} eV, E_max = {E_max} eV, dN_ele = {delta_n_elect}")
            self.fermi_energy = E_mid
        elif "tetrahedron" == method:
            eigvals_reshaped = eigvals.T.reshape(
                *self.kpoint_density, self.band_quantity
            )  # [nk0, nk1, nk2, nband]
            n_elect = self.occupation / (2.0 - self.spinful)
            self.fermi_energy, _, _ = libtetrabz.fermieng(self.reciprocal_lattice, eigvals_reshaped, n_elect)
            print(libtetrabz_info)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calc_dos(
        self, k_mesh=(-1,-1,-1), dk=0.02, emin=None, emax=None, enum=None, n_jobs=-1, parallel_k=True, method="gaussian", sigma=0.0, **kwargs
    ):
        """
        Calculate the density of states (DOS) on a uniform k-mesh.

        The eigenvalues are computed on the k-mesh (or loaded from a matching
        ``eigval.h5`` cache) and smeared with the chosen method. The resulting
        energy grid is stored in ``self.egrid`` and the DOS in ``self.dos``.

        Parameters
        ----------
        k_mesh : tuple of int, optional
            Number of k points along three reciprocal axes. Values <= 0 switch
            to ``dk``. Default: (-1, -1, -1).
        dk : float, optional
            k-point spacing used when ``k_mesh`` is not given. Default: 0.02.
        emin, emax : float, optional
            Energy window in eV relative to the Fermi energy. If None, the
            window is derived from the eigenvalue range. Default: None.
        enum : int, optional
            Number of energy grid points. Default: None (101).
        n_jobs : int, optional
            Number of parallel workers. Default: -1 (auto-detect).
        parallel_k : bool, optional
            Parallelization across k points, see ``HamiltonianObj.diag``.
            Default: True.
        method : str, optional
            - 'gaussian': Gaussian smearing (default),
            - 'tetrahedron': tetrahedron method from libtetrabz.
        sigma : float, optional
            Smearing width of method 'gaussian' in eV. Values <= 0 select an
            automatic width. Default: 0.0 (auto).
        **kwargs
            Extra options forwarded to the diagonalization (e.g.
            ``sparse_calc``), see ``HamiltonianObj.diag``.
        """
        self._calc_eigvals_on_mesh(k_mesh=k_mesh, dk=dk, n_jobs=n_jobs, parallel_k=parallel_k, **kwargs)
        self._calc_dos(self.eigvals - self.fermi_energy, method, emin, emax, enum, sigma)

    def _calc_dos(self, eigvals, method, emin, emax, enum, sigma):
        self._print(f"Calculating DOS with emin={emin} eV, emax={emax} eV, enum={enum}, method={method} ...")
        if emin is None:
            emin = np.min(eigvals - GAUSSIAN_NEGLECT_FACTOR * sigma)
        if emax is None:
            self._print("  E_max is not specified, calculation maybe long ...")
            emax = np.max(eigvals + GAUSSIAN_NEGLECT_FACTOR * sigma)
        if enum is None:
            enum = 101
        self.egrid = np.linspace(emin, emax, enum)
        #
        if "gaussian" == method:
            tmp_num = np.sum(np.logical_and(eigvals > emin, eigvals < emax)) / self.nktot**0.5
            sigma = sigma if sigma > 0.0 else (emax - emin) / tmp_num * 3.0
            self._print(f"  Using sigma={sigma} eV for gaussian smearing")
            eigvals_flat = eigvals[
                np.logical_and(
                    eigvals > emin - GAUSSIAN_NEGLECT_FACTOR * sigma, eigvals < emax + GAUSSIAN_NEGLECT_FACTOR * sigma
                )
            ]
            self.dos = (
                np.sum(gaussian_smearing(np.expand_dims(eigvals_flat, 0), np.expand_dims(self.egrid, 1), sigma), axis=1)
                / self.nktot
                * (2.0 - self.spinful)
            )
        elif "tetrahedron" == method:
            if self.nktot <= 27:
                self._print("  K points are too few for tetrahedron method, result maybe not accurate ...")
            eigvals = eigvals.T  # [nktot, nband]
            band_min = np.min(eigvals, axis=0)
            band_max = np.max(eigvals, axis=0)
            eigvals = eigvals[:, np.logical_and(band_max > emin - DELTA_ENERGY, band_min < emax + DELTA_ENERGY)]
            eigvals_reshaped = eigvals.reshape(*self.kpoint_density, -1)
            weight_kbe = libtetrabz.dos(
                self.reciprocal_lattice, eigvals_reshaped, self.egrid
            )  # [nk0, nk1, nk2, nband, enum]
            self.dos = np.sum(weight_kbe, axis=(0, 1, 2, 3)) * (2.0 - self.spinful)
            self._print(libtetrabz_info)
        else:
            raise ValueError(f"Unknown method: {method}")

    def plot_dos_data(self, plot_format="png", dpi=300):
        """
        Plot the calculated DOS and save the figure into the data directory.

        Parameters
        ----------
        plot_format : str, optional
            Output figure format, e.g. 'png' or 'pdf'. Default: 'png'.
        dpi : int, optional
            Figure resolution. Default: 300.
        """
        self._print("Plotting DOS ...")
        self._setup_plot_style()
        fig, ax = plt.subplots()
        ax.plot(self.egrid, self.dos, color="black")
        ax.set_ylim([0.0, None])
        ax.set_xlim([self.egrid[0], self.egrid[-1]])
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS (states/eV)")
        plt.tight_layout()
        fig_save_path = self.data_path / f"dos.{plot_format}"
        plt.savefig(fig_save_path, format=plot_format, dpi=dpi)

    def _decide_k_mesh(self, k_mesh, dk):
        if np.all(np.array(k_mesh) > 0):
            k_mesh = np.array(k_mesh)
            self._print(f"Use k_mesh={k_mesh}")
        else:
            b_lengths = np.linalg.norm(self.reciprocal_lattice, axis=1)
            k_mesh = np.ceil(b_lengths / dk).astype(int)
            self._print(f"Use dk={dk} and k_mesh={k_mesh}")
        return k_mesh

    def _calc_eigvals_on_mesh(self, k_mesh=(-1, -1, -1), dk=0.02, n_jobs=-1, parallel_k=True, **kwargs):
        """
        Compute the eigenvalues on a uniform k-mesh (cached or fresh).

        The k-points are decided first, then cached eigenvalues are reused if
        available (:meth:`_load_eigval_data_on_mesh`). If cached results is
        missing, eigenvalues are diagonalized via :meth:`_diag`.

        Parameters
        ----------
        k_mesh : tuple of int, optional
            Number of k points along three reciprocal axes. Values <= 0 switch
            to ``dk``. Default: (-1, -1, -1).
        dk : float, optional
            k-point spacing used when ``k_mesh`` is not given. Default: 0.02.
        n_jobs : int, optional
            Number of parallel workers. Default: -1 (auto-detect).
        parallel_k : bool, optional
            Parallelization across k points, see ``HamiltonianObj.diag``.
            Default: True.
        **kwargs : dict
            Extra options forwarded to :meth:`_diag` (e.g. ``sparse_calc``
            for a partial spectrum).
        """
        if self.eigvals is not None:
            return
        self.kpoint_density = self._decide_k_mesh(k_mesh, dk)
        self.nktot = np.prod(self.kpoint_density)
        self.ks = np.stack(
            [
                ki.reshape(-1)
                for ki in np.meshgrid(*[np.arange(nk) * 1.0 / nk for nk in self.kpoint_density], indexing="ij")
            ],
            axis=1,
        )  # [nktot, 3], k0 goes slowest and k2 goes fastest
        self._load_eigval_data_on_mesh()
        if self.eigvals is not None:
            return
        if self.ill_handler is not None and self.ill_handler.method == "orbital_removal":
            self._print("Preparing orbital truncation for ill-conditioned handling...")
            Sk_list = self.obj_H.get_all_Sk(self.ks, n_jobs=n_jobs, parallel_k=parallel_k)
            self.ill_handler.prepare_orbital_truncation(Sk_list)
        self.eigvals = self._diag(n_jobs=n_jobs, parallel_k=parallel_k, **kwargs)

    def _diag(self, n_jobs, parallel_k, sparse_calc=False, **kwargs):
        self._print(f"Calculating eigenvalues with k_mesh={self.kpoint_density}, n_jobs={n_jobs} ...")
        self.partial_spectrum = sparse_calc
        return self.obj_H.diag(
            self.ks,
            n_jobs=n_jobs,
            parallel_k=parallel_k,
            sparse_calc=sparse_calc,
            bands_only=True,
            ill_handler=self.ill_handler,
            **kwargs,
        )

    def _load_eigval_data_on_mesh(self):
        eigval_h5file = self.data_path / DEEPX_EIGVAL_FILENAME
        if eigval_h5file.exists() and self.ill_handler is None:
            with h5py.File(eigval_h5file, "r") as h5file:
                if "partial_spectrum" in h5file and bool(h5file["partial_spectrum"][()]):
                    self._print(
                        f"Skip cached {DEEPX_EIGVAL_FILENAME}: it contains a partial spectrum "
                        "that cannot be reused for full-spectrum quantities."
                    )
                else:
                    kpoint_density = np.array(h5file["kmesh"][:], dtype=int)
                    if np.allclose(kpoint_density, self.kpoint_density):
                        self._print(f"Use cached eigenvalues from {DEEPX_EIGVAL_FILENAME}")
                        self.eigvals = np.array(h5file["eigval_data"][:], dtype=float).T

    def _print(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)

    def dump_fermi_energy(self):
        """Dump the determined Fermi energy into fermi_energy.json in the data directory."""
        json_path = self.data_path / FERMI_ENERGY_INFO_FILENAME
        dump_json_file(json_path, {"fermi_energy_eV": self.fermi_energy})

    def dump_eigval_data(self):
        """
        Dump the k-mesh eigenvalues into ``eigval.h5`` in the data directory.

        The file stores the k-mesh, the k-points, the eigenvalues and a
        ``partial_spectrum`` flag telling whether only a partial spectrum was
        computed.
        """
        h5file_path = self.data_path / DEEPX_EIGVAL_FILENAME
        formatted_eigval_data = {
            "kmesh": self.kpoint_density,
            "kpoints": self.ks,
            "eigval_data": self.eigvals.T,
            "partial_spectrum": self.partial_spectrum,
        }
        with h5py.File(h5file_path, "w") as hf:
            for key, value in formatted_eigval_data.items():
                hf.create_dataset(key, data=value)

    def dump_dos_data(self):
        """Dump the calculated DOS (energy grid and dos_data) into ``dos.h5`` in the data directory."""
        h5file_path = self.data_path / DEEPX_DOS_FILENAME
        formatted_dos_data = {
            "energy": self.egrid,
            "dos_data": self.dos,
            "fermi_energy_before_shift": self.fermi_energy,
        }
        with h5py.File(h5file_path, "w") as hf:
            for key, value in formatted_dos_data.items():
                hf.create_dataset(key, data=value)

    def _setup_plot_style(self):
        # Set the Fonts
        plt.rcParams.update(
            {
                "font.size": 14,
                "font.family": "sans-serif",
            }
        )
        # Set the spacing between the axis and labels
        plt.rcParams["xtick.major.pad"] = "6"
        plt.rcParams["ytick.major.pad"] = "6"
        # Set the ticks 'inside' the axis
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"


class PETScDOSGenerator(FermiEnergyAndDOSGenerator):
    """
    DOS generator backed by the MPI-parallel PETSc/SLEPc eigensolver.

    Only a partial spectrum around a target energy is computed, so the Fermi
    energy cannot be determined by electron counting
    (:meth:`find_fermi_energy` is not implemented); :meth:`calc_dos` resolves
    it from the fermi_energy.json cache or info.json instead. The DOS itself
    is evaluated on rank 0 only, from the broadcast eigenvalues.

    Note that the tetrahedron method requires a full spectrum, which the
    partial PETSc spectrum cannot provide; use the gaussian method.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the DeepH format files.
    obj_H : PETScHamiltonianObj
        The MPI-parallel Hamiltonian object (provides ``comm``).
    """

    def __init__(self, data_path: str | Path, obj_H):
        self.comm = obj_H.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        super().__init__(data_path, obj_H)
        self.partial_spectrum = True

    def _print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs, flush=True)

    def calc_dos(
        self, k_mesh=(-1, -1, -1), dk=0.02, emin=None, emax=None, enum=None, n_jobs=None, parallel_k=None, method="gaussian", sigma=0.0, **kwargs,
    ):
        """
        Calculate the DOS with the MPI-parallel PETSc/SLEPc eigensolver.

        The Fermi energy is resolved first (:meth:`_resolve_fermi_energy`),
        then the eigenvalues are computed (or loaded from cache) and the DOS
        is evaluated on rank 0 only. ``n_jobs`` and ``parallel_k`` are
        ignored: parallelism comes from MPI.

        Parameters
        ----------
        **kwargs
            PETSc solver options, see ``PETScHamiltonianObj.parse_diag_kwargs``
            (``num_band``, ``target_band_energy``, ``maxiter``, ...). The
            remaining DOS parameters are shared with the parent method.
        """
        self._resolve_fermi_energy()
        super().calc_dos(
            k_mesh=k_mesh, dk=dk, emin=emin, emax=emax, enum=enum, n_jobs=n_jobs, parallel_k=parallel_k, method=method, sigma=sigma, **kwargs,
        )

    def _resolve_fermi_energy(self):
        """Resolve the Fermi energy from the fermi_energy.json cache or info.json (no electron counting)."""
        fermi_json = self.data_path / FERMI_ENERGY_INFO_FILENAME
        if fermi_json.exists():
            with open(fermi_json, "r") as f:
                fermi_energy = json.load(f).get("fermi_energy_eV", None)
            if fermi_energy is not None:
                self._print(f"Use cached fermi energy from {FERMI_ENERGY_INFO_FILENAME}")
                self.fermi_energy = fermi_energy
                return
        if self.obj_H.fermi_energy is not None:
            self._print("Use the fermi energy from info.json")
            self.fermi_energy = self.obj_H.fermi_energy
            return
        raise ValueError("Can not resolve the fermi energy: neither fermi_energy.json nor info.json provides one.")

    def _calc_dos(self, eigvals, method, emin, emax, enum, sigma):
        """Evaluate the DOS on rank 0 only."""
        if self.rank == 0:
            super()._calc_dos(eigvals, method, emin, emax, enum, sigma)

    def _load_eigval_data_on_mesh(self):
        """Load cached eigenvalues on rank 0 and broadcast them to all ranks (partial spectra are accepted)."""
        if self.rank == 0:
            eigval_h5file = self.data_path / DEEPX_EIGVAL_FILENAME
            if eigval_h5file.exists():
                with h5py.File(eigval_h5file, "r") as h5file:
                    kpoint_density = np.array(h5file["kmesh"][:], dtype=int)
                    if np.allclose(kpoint_density, self.kpoint_density):
                        self._print(f"Use cached eigenvalues from {DEEPX_EIGVAL_FILENAME}")
                        self.eigvals = np.array(h5file["eigval_data"][:], dtype=float).T
        self.eigvals = self.comm.bcast(self.eigvals, root=0)
        if self.eigvals is not None:
            self.band_quantity = self.eigvals.shape[0]

    def _diag(self, n_jobs=None, parallel_k=None, **kwargs):
        """
        Diagonalize with the MPI-parallel PETSc/SLEPc solver.

        ``n_jobs`` and ``parallel_k`` are ignored: parallelism comes from MPI.
        """
        self._print(f"Calculating eigenvalues with k_mesh={self.kpoint_density} ...")
        self.partial_spectrum = True
        eigvals = self.obj_H.diag(self.ks, bands_only=True, **kwargs)
        self.band_quantity = eigvals.shape[0]
        return eigvals

    def dump_eigval_data(self):
        if self.rank == 0:
            super().dump_eigval_data()

    def dump_dos_data(self):
        if self.rank == 0:
            super().dump_dos_data()

    def plot_dos_data(self, *args, **kwargs):
        if self.rank == 0:
            super().plot_dos_data(*args, **kwargs)

    def find_fermi_energy(self, *args, **kwargs):
        raise NotImplementedError(
            "Sparse partial-spectrum diagonalization cannot provide the full set of eigenvalues "
            "required for electron-counting Fermi-level determination."
        )

    def dump_fermi_energy(self):
        raise NotImplementedError(
            "Sparse partial-spectrum diagonalization cannot provide the full set of eigenvalues "
            "required for electron-counting Fermi-level determination."
        )
