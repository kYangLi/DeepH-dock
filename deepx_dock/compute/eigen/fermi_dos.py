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


GAUSSIAN_NEGLECT_FACTOR = 9.0
DELTA_ENERGY = 0.1

def gaussian_smearing(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2.0*sigma**2)) / (np.sqrt(2.0*np.pi)*sigma)

libtetrabz_info = '''
+----------------------------------------------------+
| The tetrahedron method calls libtetrabz package    |
| https://github.com/mitsuaki1987/libtetrabz         |
| https://doi.org/10.1103/PhysRevB.89.094515         |
+----------------------------------------------------+
'''

class FermiEnergyAndDOSGenerator:
    def __init__(self, data_path: str | Path, obj_H: HamiltonianObj):
        self.data_path = Path(data_path)
        self.obj_H = obj_H
        self._parse_obj_H_info()
        self.eigvals = None
        self.fermi_energy = None
    
    def _parse_obj_H_info(self):
        self.reciprocal_lattice = self.obj_H.reciprocal_lattice
        self.band_quantity = self.obj_H.orbits_quantity * (1 + self.obj_H.spinful)
        self.occupation = self.obj_H.occupation
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
            raise ValueError(
                "Band occupation statistics show inconsistent patterns"
            )
        return significant_cate > 1

    def find_fermi_energy(self, dk=0.02, k_process_num=1, method="counting"):
        fermi_json = self.data_path / FERMI_ENERGY_INFO_FILENAME
        if fermi_json.exists():
            with open(fermi_json, "r") as f:
                self.fermi_energy = json.load(f).get("fermi_energy_eV", None)
                print(f"Use cached fermi energy from {FERMI_ENERGY_INFO_FILENAME}")
        elif self.occupation is None:
            print("Fermi energy can't be determined because occupation is None, use the original one in info.json")
            self.fermi_energy = self.obj_H.fermi_energy
        if self.fermi_energy is not None:
            return
        if self.band_quantity * (2.0 - self.spinful) < self.occupation:
            raise ValueError(f"occupation ({self.occupation}) is larger than the number of bands ({self.band_quantity * (2.0 - self.spinful)})")
        #
        self._calc_eigvals_on_mesh(dk, k_process_num)
        print(f"Determining fermi energy with method={method} ...")
        if "counting" == method:
            eigvals_flattened = self.eigvals.flatten()
            ifermi = int(self.nktot * self.occupation / (2 - self.spinful))
            val_N = np.partition(eigvals_flattened, ifermi-1)[ifermi-1]
            val_Np1 = np.partition(eigvals_flattened, ifermi)[ifermi]
            self.fermi_energy = (val_N + val_Np1) / 2.0
        elif "tetrahedron" == method:
            eigvals_reshaped = self.eigvals.T.reshape(
                *self.kpoint_density, self.band_quantity
            ) # [nk0, nk1, nk2, nband]
            n_elect = self.occupation / (2.0 - self.spinful)
            self.fermi_energy, _, _ = libtetrabz.fermieng(
                self.reciprocal_lattice, eigvals_reshaped, n_elect
            )
            print(libtetrabz_info)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calc_dos(self, dk=0.02, emin=None, emax=None, enum=None, k_process_num=1, method="gaussian", sigma=-1.0):
        self._calc_eigvals_on_mesh(dk, k_process_num)
        print(f"Calculating DOS with emin={emin} eV, emax={emax} eV, enum={enum}, method={method} ...")
        eigvals = self.eigvals - self.fermi_energy
        #
        if emin is None:
            emin = np.min(eigvals-GAUSSIAN_NEGLECT_FACTOR*sigma)
        if emax is None:
            print("  E_max is not specified, calculation maybe long ...")
            emax = np.max(eigvals+GAUSSIAN_NEGLECT_FACTOR*sigma)
        if enum is None:
            enum = 101
        self.egrid = np.linspace(emin, emax, enum)
        #
        if "gaussian" == method:
            tmp_num = np.sum(
                np.logical_and(eigvals > emin, eigvals < emax)
            ) / self.nktot**0.5
            sigma = sigma if sigma > 0.0 else (emax - emin) / tmp_num * 3.0
            print(f"  Using sigma={sigma} eV for gaussian smearing")
            eigvals_flat = eigvals[np.logical_and(
                eigvals > emin-GAUSSIAN_NEGLECT_FACTOR*sigma, 
                eigvals < emax+GAUSSIAN_NEGLECT_FACTOR*sigma
            )]
            self.dos = np.sum(gaussian_smearing(
                np.expand_dims(eigvals_flat,0), np.expand_dims(self.egrid,1), sigma
            ), axis=1) / self.nktot * (2.0 - self.spinful)
        elif "tetrahedron" == method:
            if self.nktot <= 27:
                print("  K points are too few for tetrahedron method, result maybe not accurate ...")
            eigvals = eigvals.T # [nktot, nband]
            band_min = np.min(eigvals, axis=0)
            band_max = np.max(eigvals, axis=0)
            eigvals = eigvals[:, np.logical_and(
                band_max > emin-DELTA_ENERGY,
                band_min < emax+DELTA_ENERGY
            )]
            eigvals_reshaped = eigvals.reshape(*self.kpoint_density, -1)
            weight_kbe = libtetrabz.dos(
                self.reciprocal_lattice, eigvals_reshaped, self.egrid
            ) # [nk0, nk1, nk2, nband, enum]
            self.dos = np.sum(weight_kbe, axis=(0,1,2,3)) * (2.0 - self.spinful)
            print(libtetrabz_info)
        else:
            raise ValueError(f"Unknown method: {method}")

    def plot_dos_data(self, plot_format="png", dpi=300):
        print("Ploting DOS ...")
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

    def _calc_eigvals_on_mesh(self, dk, k_process_num):
        if self.eigvals is not None:
            return
        b_lengths = np.linalg.norm(self.reciprocal_lattice, axis=1)
        self.kpoint_density = np.ceil(b_lengths / dk).astype(int)
        self.nktot = np.prod(self.kpoint_density)
        self.ks = np.stack([ki.reshape(-1) for ki in np.meshgrid(
            *[np.arange(nk) * 1.0 / nk for nk in self.kpoint_density],
            indexing='ij'
        )], axis=1) # [nktot, 3], k0 goes slowest and k2 goes fastest
        print(f"Use dk={dk} and k_mesh={self.kpoint_density}")
        #
        eigval_h5file = self.data_path / DEEPX_EIGVAL_FILENAME
        if eigval_h5file.exists():
            with h5py.File(eigval_h5file, 'r') as h5file:
                kpoint_density = np.array(h5file["kmesh"][:], dtype=int)
                if np.allclose(kpoint_density, self.kpoint_density):
                    print(f"Use cached eigenvalues from {DEEPX_EIGVAL_FILENAME}")
                    self.eigvals = np.array(h5file["eigval_data"][:], dtype=float).T
                    return
        print(f"Calculating eigenvalues with k_mesh={self.kpoint_density}, k_process_num={k_process_num} ...")
        self.eigvals = self.obj_H.diag(
            self.ks, k_process_num=k_process_num,
            sparse_calc=False, bands_only=True
        ) # [nband, nktot]

    def dump_fermi_energy(self):
        json_path = self.data_path / FERMI_ENERGY_INFO_FILENAME
        dump_json_file(json_path, {"fermi_energy_eV": self.fermi_energy})

    def dump_eigval_data(self):
        h5file_path = self.data_path / DEEPX_EIGVAL_FILENAME
        formatted_eigval_data = {
            "kmesh": self.kpoint_density,
            "kpoints": self.ks,
            "eigval_data": self.eigvals.T,
        }
        with h5py.File(h5file_path, 'w') as hf:
            for key, value in formatted_eigval_data.items():
                hf.create_dataset(key, data=value)

    def dump_dos_data(self):
        h5file_path = self.data_path / DEEPX_DOS_FILENAME
        formatted_dos_data = {
            "energy": self.egrid,
            "dos_data": self.dos,
            "fermi_energy_before_shift": self.fermi_energy,
        }
        with h5py.File(h5file_path, 'w') as hf:
            for key, value in formatted_dos_data.items():
                hf.create_dataset(key, data=value)

    def _setup_plot_style(self):
        # Set the Fonts
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'sans-serif',
        })
        # Set the spacing between the axis and labels
        plt.rcParams['xtick.major.pad']='6'
        plt.rcParams['ytick.major.pad']='6'
        # Set the ticks 'inside' the axis
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

