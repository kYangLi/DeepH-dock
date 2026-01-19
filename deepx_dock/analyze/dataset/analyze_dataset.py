from pathlib import Path
import json
import h5py
import re
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DFT_DIRNAME
from deepx_dock.CONSTANT import DATASET_SPLIT_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.analyze.dataset.dft_features import DFTDatasetFeaturesDetective
from deepx_dock.analyze.dataset.e3nn_irreps import Irreps

EDGE_QUANTITY_STATISTIC_FIGURE = "edge_quantity_statistics.png"


def _convert_orbital_string_to_list(s: str) -> List[int]:
    if (not s) or (s is None):
        return []
    orbital_map = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    if s[0] not in orbital_map:
        raise ValueError(f"The orbital string `{s}` in wrong format, it should be `s?p?d?...`.")
    result = []
    for match in re.findall(r'([spdfghijklmn])(\d*)', s.lower()):
        orbital, count_str = match
        count = int(count_str) if count_str else 1
        result.extend([orbital_map.index(orbital)]*count)
    return result


def _convert_list_to_orbital_string(ls):
    orbital_map = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    string = ""
    ls_counts = np.bincount(ls)
    for ll, n in enumerate(ls_counts):
        string += f'{orbital_map[ll]}{n}'
    if len(string) == 0:
        return None
    return string


def _common_orbital_types_to_irreps(
    common_orbital_types, spinful: bool, consider_parity: bool
):
    irreps_l_list = []
    for orb_l1 in common_orbital_types:
        for orb_l2 in common_orbital_types:
            use_odd_parity = (consider_parity) and (((orb_l1 + orb_l2) % 2) == 1)
            p = -1 if use_odd_parity else 1
            irreps = [(1, (orb_l, p)) for orb_l in range(abs(orb_l1 - orb_l2), orb_l1 + orb_l2 + 1)]
            if spinful:
                irreps_x1 = []
                for _, ir in irreps:
                    ir_x1 = [(1, (orb_l, p)) for orb_l in range(abs(ir[0] - 1), ir[0] + 2)]
                    irreps_x1.extend(ir_x1)
                irreps.extend(irreps_x1)
            irreps_l_list.extend(irreps)
    if spinful:
        irreps_l_list += irreps_l_list
    return Irreps(irreps_l_list)


class DatasetAnalyzer:
    def __init__(self, data_path, n_jobs=1, n_tier=0):
        self.data_path = Path(data_path)
        self.dft_data_path = self.data_path / DFT_DIRNAME
        self.n_jobs = n_jobs
        self.n_tier = n_tier

    # ------------------------------------------------
    # DFT Features Analysis
    # ------------------------------------------------
    @property
    def dft_features(self):
        return DFTDatasetFeaturesDetective(
            dft_path=self.dft_data_path, data_dir_depth=self.n_tier,
            parallel_num=self.n_jobs,
        ).features

    def analysis_dft_features(self,
        common_orbital_types=None, consider_parity=False
    ):
        # Getting the orbital info
        if common_orbital_types is not None and common_orbital_types:
            common_orbital_types = _convert_orbital_string_to_list(common_orbital_types)
            orbital_source = "user_specified"
        else:
            common_orbital_types = self.dft_features.common_orbital_types
            orbital_source = "auto_detected"
        
        spinful = self.dft_features.spinful
        irreps_comm_orb = _common_orbital_types_to_irreps(
            common_orbital_types, spinful, consider_parity
        ).regroup()
        
        # Getting the irreps' info
        irreps_in_suggest = self._gen_suggest_irreps_in(irreps_comm_orb, spinful)
        irreps_in_exp2 = self._gen_exp2_irreps_in(irreps_comm_orb)
        irreps_in_trivial = self._gen_trivial_irreps_in(irreps_comm_orb)
        
        # Getting elements' info
        elements = [
            PERIODIC_TABLE_INDEX_TO_SYMBOL[v] 
            for v in self.dft_features.elements_orbital_map.keys()
        ]
        
        # Output
        # - Basic info
        print("\n📊 BASIC DATASET INFO")
        print("-----------------------")
        print(f"  • Spinful:                {spinful}")
        print(f"  • Parity consideration:   {consider_parity}")
        print(f"  • Total data points:      {self.dft_features.all_dft_data_num:,}")
        
        # - Elements and orbital
        print("\n🧪 ELEMENT & ORBITAL INFO")
        print("---------------------------")
        print(f"  • Elements included:      {', '.join(elements)} ({len(elements)} elements)")
        print(f"  • Orbital source:         {orbital_source}")
        
        if common_orbital_types is not None:
            print(f"  • BS3B orbital types:     {common_orbital_types}")
        else:
            print(f"  • Common orbital types:   {_convert_list_to_orbital_string(common_orbital_types)}")
        
        # - Irreps
        print("\n🎯 IRREPS INFORMATION")
        print("-----------------------")
        print(f"  {'Irreps Type':<20} {'Irreps':<50} {'Dimension':<10}")
        print(f"  {'.'*20:<20} {'.'*50:<50} {'.'*10:<10}")
        print(f"  {'Common orbital':<20} {str(irreps_comm_orb):<50} {irreps_comm_orb.dim:<10}")
        print(f"  {'Suggested':<20} {str(irreps_in_suggest):<50} {irreps_in_suggest.dim:<10}")
        print(f"  {'Exp2':<20} {str(irreps_in_exp2):<50} {irreps_in_exp2.dim:<10}")
        print(f"  {'Trivial':<20} {str(irreps_in_trivial):<50} {irreps_in_trivial.dim:<10}")
        
        # Summary
        print("\n💡 RECOMMENDATIONS")
        print("--------------------")
        
        recommendations = []
        
        # - Based on quantity
        if self.dft_features.all_dft_data_num < 100:
            recommendations.append("Small dataset detected - consider data augmentation or transfer learning")
        elif self.dft_features.all_dft_data_num < 6000:
            recommendations.append("Moderate dataset size - regular training recommended")
        else:
            recommendations.append("Large dataset available - suitable for complex model training")
        
        # - Based on orb
        if irreps_comm_orb.dim > 120:
            recommendations.append("High-dimensional irreps - consider dimensionality reduction techniques")
        
        # - Based on spinful
        if spinful:
            recommendations.append("Spinful system - ensure spin-related symmetries are properly handled")
        
        # Print the summary
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Return!
        return {
            "spinful": spinful,
            "elements": elements,
            "common_orbital_types": common_orbital_types,
            "irreps_comm_orb": irreps_comm_orb,
            "irreps_in_suggest": irreps_in_suggest,
            "irreps_in_exp2": irreps_in_exp2,
            "irreps_in_trivial": irreps_in_trivial,
            "all_dft_data_num": self.dft_features.all_dft_data_num
        }

    @staticmethod
    def _gen_suggest_irreps_in(irreps_comm_orb, spinful):
        def suggest_mul(mul):
            _exp_suggest = 2**int(np.ceil(np.log2(mul)))
            _8_suggest = 8 * ((mul - 1) // 8 + 1)
            if mul <= 2:
                return 2
            elif mul <= 8:
                return _exp_suggest
            else:
                return _8_suggest
        return Irreps([
            (suggest_mul(mul),(ll,p))
            for mul,(ll,p) in irreps_comm_orb // (1+spinful)
        ]).regroup()
    
    @staticmethod
    def _gen_exp2_irreps_in(irreps_comm_orb):
        max_mul = max([v.mul for v in irreps_comm_orb])
        l0_mul = 2**int(np.ceil(np.log2(max_mul)))
        return Irreps([
            (max(int(l0_mul//(2**(i))),2),(ll,p))
            for i,(_,(ll,p)) in enumerate(irreps_comm_orb)
        ]).regroup()

    @staticmethod
    def _gen_trivial_irreps_in(irreps_comm_orb):
        max_mul = max([v.mul for v in irreps_comm_orb])
        l0_mul = 2**int(np.ceil(np.log2(max_mul)))
        return Irreps([
            (max(l0_mul,2),(ll,p))
            for i,(_,(ll,p)) in enumerate(irreps_comm_orb)
        ]).regroup()

    # ------------------------------------------------
    # Data Split JSON Generator
    # ------------------------------------------------
    def generate_data_split_json(self,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        max_edge_num=-1, rng_seed=137
    ):
        assert (train_ratio + val_ratio + test_ratio) <= 1.0
        worker = partial(
            self._check_data_validation,
            dft_data_path=self.dft_data_path,
            max_edge_num=max_edge_num
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(self.dft_features.all_dft_dirname,desc="Data Split")
        )
        available_data_dirs = [name for name in results if name is not None]
        # Generate the data split
        available_data_num = len(available_data_dirs)
        print(f"[info] Total available data dirs: {available_data_num}")
        _rng = np.random.default_rng(rng_seed)
        _rng.shuffle(available_data_dirs)
        n_train = int(available_data_num * train_ratio)
        n_val = int(available_data_num * val_ratio)
        data_split = {
            "train":  available_data_dirs[:n_train],
            "validate": available_data_dirs[n_train:n_train+n_val],
            "test": available_data_dirs[n_train+n_val:]
        }
        # Save the json to file
        with open(DATASET_SPLIT_FILENAME, "w") as jfrp:
            json.dump(data_split, jfrp)
        print(f"[info] Data split json saved to ./{DATASET_SPLIT_FILENAME}.")
        return data_split
    
    @staticmethod
    def _check_data_validation(
        dir_name: str, dft_data_path: str | Path, max_edge_num: int = -1
    ):
        dir_path = Path(dft_data_path) / dir_name
        h_path = dir_path / DEEPX_HAMILTONIAN_FILENAME
        if h_path.is_file():
            with h5py.File(h_path, "r") as fh5:
                edge_num = len(np.array(fh5["atom_pairs"][:]))
                if (max_edge_num < 0) or (edge_num <= max_edge_num):
                    return dir_name
        return None

    # ------------------------------------------------
    # Statistic edges quantity
    # ------------------------------------------------
    def statistic_edge_quantity(self, bins=None):
        # Read the edge_quantity
        cache_path = self.data_path / "edge_statistic.h5"
        if cache_path.is_file():
            with h5py.File(cache_path, 'r') as h5file:
                results = np.array(h5file["edges_quantity"][:], dtype=int)
        else:
            worker = partial(
                self._read_edge_info,
                dft_data_path=self.dft_data_path,
            )
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(worker)(dir_name)
                for dir_name in tqdm(
                    self.dft_features.all_dft_dirname, desc="Edge Analysis"
                )
            )
            results = np.array(results)
            with h5py.File(cache_path, "w") as h5file:
                h5file.create_dataset("edges_quantity", data=results)
        # Count!
        bins = 'auto' if bins is None else bins
        self.edge_counts, self.edge_bin = np.histogram(results, bins=bins)
    
    def plot_edge_quantity(self, dpi=300):
        bin_labels = [
            f"{int(self.edge_bin[i])}-{int(self.edge_bin[i+1])}" 
            for i in range(len(self.edge_bin)-1)
        ]
        save_figure_path = self.data_path / "edge_statistic.png"
        plt.bar(bin_labels, self.edge_counts, edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Edges Quantity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(save_figure_path, dpi=dpi)

    @staticmethod
    def _read_edge_info(dir_name: str, dft_data_path: str | Path):
        dir_path = Path(dft_data_path) / dir_name
        h_path = dir_path / DEEPX_HAMILTONIAN_FILENAME
        if h_path.is_file():
            with h5py.File(h_path, "r") as fh5:
                edge_num = len(np.array(fh5["atom_pairs"][:]))
                return edge_num

