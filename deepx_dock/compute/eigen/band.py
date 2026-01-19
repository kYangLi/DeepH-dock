from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import h5py
from collections import Counter

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj


class BandDataGenerator:
    def __init__(self, obj_H:HamiltonianObj, band_conf):
        self.obj_H = obj_H
        self.band_conf = band_conf
        self._parse_obj_H_info()
        self._parse_band_conf()
        self._gen_all_k_list()

    def _parse_obj_H_info(self):
        self.rlv = self.obj_H.reciprocal_lattice
        self.spinful = self.obj_H.spinful
        self.band_quantity = self.obj_H.orbits_quantity * (1 + self.spinful)
        self.fermi_energy = self.obj_H.fermi_energy

    def _parse_band_conf(self):
        self.fermi_energy = \
            self.band_conf.get('fermi_energy_eV', self.fermi_energy)
        self._parse_k_list_spell(self.band_conf['k_list_spell'])

    def _parse_k_list_spell(self, spell: str):
        _spell_lines = [sp.strip() for sp in spell.splitlines() if sp.strip()]
        self.k_path_quantity = len(_spell_lines)
        self.k_path_density_list = [None for _ in range(self.k_path_quantity)]
        self.hsk_vector_list = [
            [None, None] for _ in range(self.k_path_quantity)
        ]
        self.hsk_symbol_list = [
            [None, None] for _ in range(self.k_path_quantity)
        ]
        for i, kpi in enumerate(_spell_lines):
            kpi = kpi.split()
            self.k_path_density_list[i] = int(kpi[0])
            self.hsk_vector_list[i][0] = np.array([
                float(kpi[1]), float(kpi[2]), float(kpi[3])
            ])
            self.hsk_vector_list[i][1] = np.array([
                float(kpi[4]), float(kpi[5]), float(kpi[6])
            ])
            self.hsk_symbol_list[i] = [kpi[7], kpi[8]]

    def _gen_all_k_list(self):
        # Get all of the fractional coordinates of the kpoints
        self.kpoints_quantity = sum(self.k_path_density_list)
        self.kpoints_frac_list = []
        for i_path in range(self.k_path_quantity):
            start_hsk, end_hsk = self.hsk_vector_list[i_path]
            num_k = self.k_path_density_list[i_path]
            _curr_path_ks = [
                start_hsk + (float(i) / (num_k - 1)) * (end_hsk - start_hsk) 
                for i in range(num_k)
            ]
            self.kpoints_frac_list.extend(_curr_path_ks)
        # Get the high symmetric kpoints distance list
        self.hsk_distance_list = np.zeros(self.k_path_quantity+1)
        for i_path in range(self.k_path_quantity):
            self.hsk_distance_list[i_path+1] = \
                self._cal_k_distance(
                    self.hsk_vector_list[i_path][0],
                    self.hsk_vector_list[i_path][1]
                ) + self.hsk_distance_list[i_path]
        # Get the k-space coordinates distance of the kpoints
        self.kpoints_distance_list = np.zeros(self.kpoints_quantity)
        for i_path, n_kp in enumerate(self.k_path_density_list):
            _dist = 0.0
            _prev_n_kp = sum(self.k_path_density_list[:i_path])
            _prev_dist = self.hsk_distance_list[i_path]
            self.kpoints_distance_list[_prev_n_kp] = _prev_dist
            for i_k in range(1,n_kp):
                _dist += self._cal_k_distance(
                    self.kpoints_frac_list[_prev_n_kp+i_k-1],
                    self.kpoints_frac_list[_prev_n_kp+i_k]
                )
                self.kpoints_distance_list[_prev_n_kp+i_k] = _prev_dist + _dist

    def _cal_k_distance(self, start_kpoint_frac, end_kpoint_frac):
        delta_frac = end_kpoint_frac - start_kpoint_frac
        delta_cart = delta_frac @ self.rlv
        return np.linalg.norm(delta_cart) / (2 * np.pi)

    def calc_band_data(self, k_process_num=1, sparse_calc=False):
        if sparse_calc:
            _num_band = self.band_conf.get("num_band", 50)
            _lowest_band_energy = self.band_conf.get("lowest_band_energy", -0.5)
            _maxiter = self.band_conf.get("maxiter", 300)
            self.band_quantity = _num_band
            print(f"Sparse calculation with num_band={_num_band}, lowest_band_energy={_lowest_band_energy}, maxiter={_maxiter} ...")
            kwargs = {
                'k': _num_band,
                'sigma': self.fermi_energy+_lowest_band_energy,
                'which': 'LA', 'maxiter': _maxiter, 'tol': 1e-5, 'mode': 'normal'
            }
        else:
            kwargs = {}
        self.band_data = self.obj_H.diag(
            self.kpoints_frac_list, k_process_num=k_process_num, 
            sparse_calc=sparse_calc, bands_only=True, **kwargs
        )
        if self.band_data.shape[0] != self.band_quantity:
            print(f"Warn: Only {self.band_data.shape[0]} bands are calculated, not {self.band_quantity} in input.")
            self.band_quantity = self.band_data.shape[0]
        self._shift_band_data_to_fermi_zero()
    
    def _shift_band_data_to_fermi_zero(self):
        self._calculate_occupation_stats()
        self._correct_fermi_energy()
        self.band_data -= self.fermi_energy
        self.fermi_energy_before_shift = self.fermi_energy
        self.fermi_energy = 0.0

    def _correct_fermi_energy(self):
        """
        E_fermi = (E_HOMO + E_LUMO) / 2
        """
        self.band_gap = 0.0
        if not self._is_metal:
            homo_index = np.sum(self.band_occupy_counts > 0.5) - 1
            if homo_index < 0 or homo_index >= self.band_quantity-1:
                print(
                    "Warn: The fermi energy is out of the energy range of calculated bands."
                )
                return
            lumo_index = homo_index + 1
            homo_energy = np.max(self.band_data[homo_index])
            lumo_energy = np.min(self.band_data[lumo_index])
            if not (lumo_energy >= self.fermi_energy >= homo_energy):
                print(
                    f"Warn: The original fermi energy ({self.fermi_energy}) is not in the band gap ({lumo_energy}, {homo_energy}), and it will be modified into the band gap."
                )
            self.fermi_energy = (homo_energy + lumo_energy) / 2
            self.band_gap = lumo_energy - homo_energy

    def _calculate_occupation_stats(self):
        self.k_occupy_counts = \
            np.sum(self.band_data < self.fermi_energy, axis=0)
        self.band_occupy_counts = \
            np.sum(self.band_data < self.fermi_energy, axis=1)

    @property
    def _is_metal(self) -> bool:
        stats =  Counter(self.k_occupy_counts)
        if len(stats) == 1:
            return False
        #
        THRESHOLD = 0.05
        min_count = max(1, int(THRESHOLD * self.kpoints_quantity / len(stats)))
        significant_cate = sum(count > min_count for count in stats.values())
        if significant_cate == 0:
            raise ValueError(
                "Band occupation statistics show inconsistent patterns"
            )
        return significant_cate > 1

    def dump_band_data(self, h5file_path: str):
        formatted_band_data = {
            "rlv" : self.rlv,
            "spinful":self.spinful,
            "band_quantity": self.band_quantity,
            "fermi_energy_before_shift_eV": self.fermi_energy_before_shift,
            "fermi_energy_eV" : self.fermi_energy,
            "k_path_quantity" : self.k_path_quantity,
            "k_path_density_list": self.k_path_density_list,
            "hsk_vector_list": self.hsk_vector_list,
            "hsk_symbol_list": self.hsk_symbol_list,
            "kpoints_quantity": self.kpoints_quantity,
            "kpoints_frac_list": self.kpoints_frac_list,
            "kpoints_distance_list": self.kpoints_distance_list,
            "hsk_distance_list": self.hsk_distance_list,
            "band_data": self.band_data,
            "band_gap": self.band_gap,
            "is_metal": self._is_metal,
        }
        with h5py.File(h5file_path, 'w') as hf:
            for key, value in formatted_band_data.items():
                hf.create_dataset(key, data=value)


class BandPlotter:
    def __init__(self, band_data_file_path: str | Path):
        self.band_data_file_path = Path(band_data_file_path)
        self._load_band_data(band_data_file_path)
    
    def _load_band_data(self, band_data_file_path: str | Path):
        with h5py.File(band_data_file_path, 'r') as hf:
            self.band_quantity = hf['band_quantity'][()]
            self.k_path_quantity = hf['k_path_quantity'][()]
            self.hsk_symbol_list = [
                [s.decode('utf-8') for s in v] 
                for v in hf['hsk_symbol_list'][()]
            ]
            self.kpoints_distance_list = hf['kpoints_distance_list'][()]
            self.hsk_distance_list = hf['hsk_distance_list'][()]
            self.band_data = hf['band_data'][()]
    
    def plot(self,
        Emin=-10.0, Emax=10.0, plot_format="png", dpi=300
    ):
        # Set the plot style
        self._setup_plot_style()
        # Create the figure and axis object
        fig = plt.figure()
        band_plot = fig.add_subplot(1, 1, 1)
        # Set the range of plot
        xmin, xmax = 0.0, self.kpoints_distance_list[-1]
        ymin, ymax = Emin, Emax
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # Set the label of x and y axis
        plt.xlabel('')
        plt.ylabel('Energy (eV)')
        # Set the Ticks of x and y axis
        plot_hsk_symbol_list = self._get_decelerated_hsk_symbol()
        plt.xticks(self.hsk_distance_list)
        band_plot.set_xticklabels(plot_hsk_symbol_list)
        # Plot the solid lines for High symmetric k-points
        for x in self.hsk_distance_list:
            plt.vlines(x, ymin, ymax, colors="black", linewidth=0.7)
        # Plot the fermi energy surface with a dashed line
        plt.hlines(
            0.0, xmin, xmax, colors="black", linestyles="dashed", linewidth=0.7
        )
        # Plot the Band Structure
        for band_index in range(self.band_quantity):
            x = self.kpoints_distance_list
            y = self.band_data[band_index]
            band_plot.plot(x, y, 'r-', linewidth=1.2)
        # Save the figure
        fig_save_path = self.band_data_file_path.parent / f"band.{plot_format}"
        plt.tight_layout()
        plt.savefig(fig_save_path, format=plot_format, dpi=dpi)

    def _get_decelerated_hsk_symbol(self):
        # Prepare the symbol of k-axis (x-tics)
        _hsk_symbol_list = ['' for _ in range(self.k_path_quantity+1)]
        _hsk_symbol_list[0] = self.hsk_symbol_list[0][0]
        _hsk_symbol_list[-1] = self.hsk_symbol_list[-1][1]
        for i_path in range(1, self.k_path_quantity):
            if self.hsk_symbol_list[i_path][0] == \
            self.hsk_symbol_list[i_path-1][1]:
                _hsk_symbol_list[i_path] = self.hsk_symbol_list[i_path][0]
            else:
                _hsk_symbol_list[i_path] = f"{self.hsk_symbol_list[i_path-1][1]}|{self.hsk_symbol_list[i_path][0]}"
        # Decelerate the symbols
        greek_symbol_list = [
            'Gamma','Delta','Theta','Lambda','Xi',
            'Pi','Sigma','Phi','Psi','Omega'
        ]
        for i, symbol in enumerate(_hsk_symbol_list):
            for greek_symbol in greek_symbol_list:
                latex_greek_symbol = "$\\" + greek_symbol + "$"
                symbol = re.sub(greek_symbol, "orz", symbol, flags=re.I)
                symbol = symbol.replace("orz", latex_greek_symbol)
            # The subscript
            symbol = re.sub(r'_\d+', lambda x:'$'+x[0]+'$', symbol)
            _hsk_symbol_list[i] = symbol
        # Return!
        return _hsk_symbol_list

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

