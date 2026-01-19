from pathlib import Path
import h5py
from tqdm import tqdm
import json
from itertools import accumulate
from joblib import Parallel, delayed

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import mendeleev

from deepx_dock.misc import load_json_file, dump_json_file

from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_PREDICT_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX
from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.misc import get_data_dir_lister

MASK_THRESHOLD = 1E-10


# +------------------------------------------------------------------------+
# |                            Misc Func                                   |
# +------------------------------------------------------------------------+
def _validation_check(
    root_dir: Path, prev_dirname: Path
):
    target_file = root_dir / DEEPX_PREDICT_HAMILTONIAN_FILENAME
    if target_file.exists():
        yield prev_dirname

def _read_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
        boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
        shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
        dset = f['entries']
        entries_dtype = np.complex128 if dset.dtype.kind == 'c' else np.float64
        entries = np.array(dset[:], dtype=entries_dtype)
    return atom_pairs, boundaries, shapes, entries

def _read_poscar_elem(poscar_path):
    with open(poscar_path) as frp:
        lines = frp.readlines()
    element_list = lines[5].split()
    element_atom_num = list(map(int, lines[6].split()))
    atoms_elem = [
        elem for (elem, count) in zip(element_list, element_atom_num)
        for _ in range(count)
    ]
    return atoms_elem

def _read_info_json(info_path):
    with open(info_path) as jfrp:
        info = json.load(jfrp)
    elements_orbital_map = info["elements_orbital_map"]
    spinful = info["spinful"]
    return elements_orbital_map, spinful

def _convert_overlap_to_spinful(atom_pairs, boundaries, shapes, overlap):
    spinful_overlap = np.zeros(overlap.shape[0]*4, dtype=overlap.dtype)
    for i_ap, _ in enumerate(atom_pairs):
        overlap_chunk = overlap[boundaries[i_ap]:boundaries[i_ap+1]].reshape(shapes[i_ap])
        zero_chunk = np.zeros_like(overlap_chunk, dtype=overlap.dtype)
        spinful_overlap_chunk = np.block(
            [[overlap_chunk, zero_chunk], [zero_chunk, overlap_chunk]]
        )
        spinful_overlap[boundaries[i_ap]*4:boundaries[i_ap+1]*4] = spinful_overlap_chunk.reshape(-1)
    return spinful_overlap

def _read_in_all_necessary_data(
    sid: str, target_name: str, bm_dft_dir: str | Path, pred_dft_dir: str | Path
):
    bm_dft_dir = Path(bm_dft_dir)
    pred_dft_dir = Path(pred_dft_dir)
    
    if "H" == target_name:
        target_file_name = DEEPX_HAMILTONIAN_FILENAME
    elif "Rho" == target_name:
        target_file_name = DEEPX_DENSITY_MATRIX_FILENAME
    bm_path = bm_dft_dir / sid
    bm_H_path = bm_path / target_file_name
    poscar_path = bm_path / DEEPX_POSCAR_FILENAME
    info_path = bm_path / DEEPX_INFO_FILENAME
    pred_path = pred_dft_dir / sid
    pred_H_path = pred_path / DEEPX_PREDICT_HAMILTONIAN_FILENAME
    # Read in info json and elements
    elem_orbs_map, spinful = _read_info_json(info_path)
    atoms_elem = _read_poscar_elem(poscar_path)
    # Read in main data
    bm_atom_pairs, bm_boundaries, bm_shapes, bm_entries = _read_h5(bm_H_path)
    pred_atom_pairs, pred_boundaries, pred_shapes, pred_entries = _read_h5(pred_H_path)
    assert np.array_equal(bm_atom_pairs, pred_atom_pairs)
    assert np.array_equal(bm_boundaries, pred_boundaries)
    assert np.array_equal(bm_shapes, pred_shapes)
    # Read in overlap if exist
    overlap_path = bm_path / DEEPX_OVERLAP_FILENAME
    overlap = None
    if overlap_path.is_file():
        S_atom_pairs, S_boundaries, S_shapes, overlap = _read_h5(overlap_path)
        if spinful:
            overlap = _convert_overlap_to_spinful(
                S_atom_pairs, S_boundaries, S_shapes, overlap
            )
    return bm_atom_pairs, bm_boundaries, bm_shapes, bm_entries, pred_entries,\
        atoms_elem, elem_orbs_map, spinful, overlap

def _get_overlap_mask(
    atom_pairs, boundaries, shapes, overlap, atoms_elem, elem_orbs_map
):
    mask = np.zeros_like(overlap, dtype=bool)
    for i_ap, ap in enumerate(atom_pairs):
        # Self edge
        _self_edge = (ap[0]==0 and ap[1]==0 and ap[2]==0 and ap[3]==ap[4])
        if _self_edge:
            mask[boundaries[i_ap]:boundaries[i_ap+1]] = True
            continue
        # Not self edge
        elem_i, elem_j = atoms_elem[ap[3]], atoms_elem[ap[4]]
        overlap_chunk = overlap[boundaries[i_ap]:boundaries[i_ap+1]].reshape(shapes[i_ap])
        mask_chunk = np.zeros_like(overlap_chunk, dtype=bool)
        orb_i, orb_j = elem_orbs_map[elem_i], elem_orbs_map[elem_j]
        row_ = 0
        for li in orb_i:
            row_ += li * 2 + 1
            col_ = 0
            for lj in orb_j:
                col_ += lj * 2 + 1
                _chunk= overlap_chunk[row_-(li*2+1):row_,col_-(lj*2+1):col_]
                _mask = np.max(np.abs(_chunk)) > MASK_THRESHOLD
                mask_chunk[row_-(li*2+1):row_,col_-(lj*2+1):col_] = _mask
        mask[boundaries[i_ap]:boundaries[i_ap+1]] = mask_chunk.reshape(-1)
    return mask

def _get_error_entries(
    pred_entries, bm_entries, overlap, standardize_gauge, consider_overlap_mask,
    atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map,
    fix_shape=False, return_bm_entries=False
):
    if (not standardize_gauge) or (overlap is None):
        errors = np.abs(pred_entries - bm_entries)
    else:
        if consider_overlap_mask:
            _mask = _get_overlap_mask(
                atom_pairs, boundaries, shapes, overlap, atoms_elem,
                elem_orbs_map
            )
            if fix_shape:
                pred_entries[np.logical_not(_mask)] = 0.0
                bm_entries[np.logical_not(_mask)] = 0.0
            else:
                pred_entries = pred_entries[_mask]
                bm_entries = bm_entries[_mask]
                overlap = overlap[_mask]
        _mu = np.dot(pred_entries-bm_entries,overlap) / np.dot(overlap,overlap)
        errors = np.abs(pred_entries - bm_entries - _mu * overlap)
    if return_bm_entries:
        return errors, bm_entries
    return errors


# +------------------------------------------------------------------------+
# |                            Main Class                                  |
# +------------------------------------------------------------------------+
class BaseAnalyzer:
    TAG = "__null__"
    
    def __init__(self,
        pred_dft_dir, bm_dft_dir="", target_name="H",
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, parallel_num=1, tier_num=0,
    ):
        self.pred_dft_dir = Path(pred_dft_dir)
        self.bm_dft_dir = Path(bm_dft_dir) if bm_dft_dir else self.pred_dft_dir
        self.root_dir = self.pred_dft_dir.parent
        self.target_name = target_name
        self.standardize_gauge = standardize_gauge
        self.consider_overlap_mask = consider_overlap_mask
        self.data_split_json = data_split_json if data_split_json is None else Path(data_split_json)
        self.data_split_tags = data_split_tags
        self.cache_res = cache_res
        self.save_figure_path = self.root_dir / f'{self.TAG}.png'
        self.cached_result_path = self.root_dir / f'{self.TAG}.h5'
        self.parallel_num = parallel_num
        self.tier_num = tier_num

    def analyze_all(self):
        if self.cached_result_path.is_file():
            self._load_cached_result()
            return
        #
        self._analyze_all_from_rawdata()
        if self.cache_res:
            self._cache_the_result()
    
    def plot(self, *args):
        raise NotImplementedError("Not implemented!")

    def _cache_the_result(self):
        raise NotImplementedError("Not implemented!")
    
    def _load_cached_result(self):
        raise NotImplementedError("Not implemented!")
    
    def _analyze_all_from_rawdata(self):
        results = Parallel(n_jobs=self.parallel_num)(
            delayed(self._analysis_one_structure)(str(sid), self.target_name)
            for sid in tqdm(self._get_all_dft_dir(), desc="Error Analysis")
        )
        self._postprocess_results(results)
    
    def _get_all_dft_dir(self):
        if self.data_split_json is not None and self.data_split_json.is_file():
            data_split = load_json_file(self.data_split_json)
            all_dft_dir = []
            for tag in self.data_split_tags.split(','):
                all_dft_dir += data_split[tag]
        else:
            all_dft_dir = get_data_dir_lister(
                self.pred_dft_dir, self.tier_num, _validation_check
            )
        return all_dft_dir
    
    def _analysis_one_structure(self, sid, target_name):
        raise NotImplementedError("Not implemented!")
    
    def _postprocess_results(self, *args):
        raise NotImplementedError("Not implemented!")


class ErrorEachEntriesDistributionAnalyzer(BaseAnalyzer):
    TAG = "error_each_entries_distribution"
    
    def __init__(self, 
        pred_dft_dir, bm_dft_dir="", target_name="H", 
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, parallel_num=1, tier_num=0,
    ):
        super().__init__(
            pred_dft_dir, bm_dft_dir, target_name,
            standardize_gauge, consider_overlap_mask,
            data_split_json, data_split_tags, cache_res, parallel_num, tier_num
        )
        # Middle and output Data
        self.all_entries = []
        self.all_abs_errs = []
        self.all_rel_errs = []
        self.aver_abs_err = 0.0
        self.aver_rel_err = 0.0

    def _cache_the_result(self):
        with h5py.File(self.cached_result_path, "w") as h5file:
            h5file.create_dataset("all_entries", data=self.all_entries)
            h5file.create_dataset("all_abs_errs", data=self.all_abs_errs)
            h5file.create_dataset("all_rel_errs", data=self.all_rel_errs)
    
    def _load_cached_result(self):
        with h5py.File(self.cached_result_path, 'r') as h5file:
            dset = h5file["all_entries"]
            all_entries_dtype = np.complex128 if dset.dtype.kind == 'c' else np.float64
            self.all_entries = np.array(dset[:], dtype=all_entries_dtype)
            self.all_abs_errs= np.array(h5file["all_abs_errs"][:],dtype=np.float64)
            self.all_rel_errs= np.array(h5file["all_rel_errs"][:],dtype=np.float64)
        self.aver_abs_err = np.mean(self.all_abs_errs)
        self.aver_rel_err = np.mean(self.all_rel_errs)
        print(f"[info] Entries mean absolute error: {self.aver_abs_err:.3e} eV")
        print(f"[info] Entries mean relative error: {self.aver_rel_err:.3e}")

    def _postprocess_results(self, results):
        #
        self.all_entries = [r["entries"] for r in results]
        self.all_abs_errs = [r["abs_errs"] for r in results]
        self.all_rel_errs = [r["rel_errs"] for r in results]
        self.all_entries = np.concatenate(self.all_entries)
        self.all_abs_errs = np.concatenate(self.all_abs_errs)
        self.all_rel_errs = np.concatenate(self.all_rel_errs)
        #
        self.aver_abs_err = np.mean(self.all_abs_errs)
        self.aver_rel_err = np.mean(self.all_rel_errs)
        print(f"[info] Entries mean absolute error: {self.aver_abs_err:.3e} eV")
        print(f"[info] Entries mean relative error: {self.aver_rel_err:.3e}")
    
    def _analysis_one_structure(self, sid, target_name):
        atom_pairs, boundaries, shapes, bm_entries, pred_entries, atoms_elem,\
            elem_orbs_map, spinful, overlap = _read_in_all_necessary_data(
                sid, target_name, self.bm_dft_dir, self.pred_dft_dir
            )
        abs_errs, bm_entries = _get_error_entries(
            pred_entries, bm_entries, overlap,
            self.standardize_gauge, self.consider_overlap_mask,
            atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map,
            return_bm_entries=True,
        )
        rel_errs = abs_errs * np.abs(bm_entries) / (
            np.abs(bm_entries)**2 + abs_errs**2 + 1E-10
        )
        return {
            "entries": bm_entries, "abs_errs": abs_errs, "rel_errs": rel_errs,
        }

    def plot_scatter(self,
        plot_dpi=300, x_lim=None, y_abs_lim=None, y_rel_lim=None, unit='(eV)'
    ):
        if self.all_entries.dtype == np.complex128:
            self.all_entries_real = np.real(self.all_entries)
        else:
            self.all_entries_real = self.all_entries
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        # Abs plot
        ax1.plot(
            self.all_entries_real, self.all_abs_errs, '.', markersize=0.2, color='r'
        )
        ax1.axhline(
            self.aver_abs_err, color='b', linestyle='--', linewidth=1,
            xmin=ax1.get_xlim()[0], xmax=ax1.get_xlim()[1]
        )
        ax1.set_yscale('log')
        ax1.set_xlabel(f'Entries {unit}')
        ax1.set_ylabel(f'Absolute Error {unit}')
        if x_lim is not None:
            ax1.set_xlim(x_lim)
        if y_abs_lim is not None:
            ax1.set_ylim(y_abs_lim)
        # Rev plot
        ax2.plot(
            self.all_entries_real, self.all_rel_errs, '.', markersize=0.2, color='r'
        )
        ax2.axhline(
            self.aver_rel_err, color='b', linestyle='--', linewidth=1,
            xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1]
        )
        ax2.set_yscale('log')
        ax2.set_xlabel(f'Entries {unit}')
        ax2.set_ylabel('Relative Error')
        if x_lim is not None:
            ax2.set_xlim(x_lim)
        if y_rel_lim is not None:
            ax2.set_ylim(y_rel_lim)
        # Save figure
        plt.tight_layout()
        plt.savefig(self.save_figure_path, dpi=plot_dpi)

    def heatmap_statistic(self,
        entries_bucket_size=100, errs_bucket_size=100,
        entries_range=None, abs_errs_range=None, rel_errs_range=None
    ):
        if self.all_entries.dtype == np.complex128:
            self.all_entries_real = np.real(self.all_entries)
        else:
            self.all_entries_real = self.all_entries
        # Decided the analysis range
        if entries_range is None:
            entries_min = self.all_entries_real.min()
            entries_max = self.all_entries_real.max()
        else:
            entries_min, entries_max = entries_range
        if abs_errs_range is None:
            abs_errs_min = self.all_abs_errs.min() + 1E-10
            abs_errs_max = self.all_abs_errs.max()
        else:
            abs_errs_min, abs_errs_max = abs_errs_range
        if rel_errs_range is None:
            rel_errs_min = self.all_rel_errs.min() + 1E-10
            rel_errs_max = self.all_rel_errs.max()
        else:
            rel_errs_min, rel_errs_max = rel_errs_range
        # Build up the bin
        x_bins = np.linspace(
            entries_min, entries_max, entries_bucket_size
        )
        y1_bins_log = np.exp(np.linspace(
            np.log(abs_errs_min), np.log(abs_errs_max), errs_bucket_size
        ))
        y2_bins_log = np.exp(np.linspace(
            np.log(rel_errs_min), np.log(rel_errs_max), errs_bucket_size
        ))
        # Compute the counts
        counts1, x_edges, y1_edges = np.histogram2d(
            self.all_entries_real, self.all_abs_errs, bins=[x_bins, y1_bins_log]
        )
        counts1 = counts1.T ** 0.1
        counts1 = np.log(counts1 + 1E-10)
        counts1 = counts1 / counts1.max()
        counts2, _, y2_edges = np.histogram2d(
            self.all_entries_real, self.all_rel_errs, bins=[x_bins, y2_bins_log]
        )
        counts2 = counts2.T ** 0.1
        counts2 = np.log(counts2 + 1E-10)
        counts2 = counts2 / counts2.max()
        # Clear!
        del self.all_entries, self.all_entries_real, self.all_abs_errs, self.all_rel_errs
        # Return!
        self.entries_edges = x_edges
        self.abs_errs_edges = y1_edges
        self.rel_errs_edges = y2_edges
        self.abs_errs_counts = counts1
        self.rel_errs_counts = counts2

    def plot_heatmap(self, plot_dpi=300, unit='(eV)'):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        # Abs plot
        img1 = ax1.pcolormesh(
            self.entries_edges, self.abs_errs_edges, self.abs_errs_counts, 
            cmap='inferno', shading='auto', vmin=0.0
        )
        plt.colorbar(img1, pad=0.03, aspect=40).set_label('Density (a.u.)')
        ax1.set_yscale('log')
        ax1.axhline(
            self.aver_abs_err, color='white', linestyle='--', linewidth=1,
            xmin=ax1.get_xlim()[0], xmax=ax1.get_xlim()[1]
        )
        ax1.set_xlabel(f'Entries {unit}')
        ax1.set_ylabel(f'Absolute Error {unit}')
        # Rev plot
        img2 = ax2.pcolormesh(
            self.entries_edges, self.rel_errs_edges, self.rel_errs_counts, 
            cmap='inferno', shading='auto', vmin=0.0
        )
        plt.colorbar(img2, pad=0.03, aspect=40).set_label('Density (a.u.)')
        ax2.axhline(
            self.aver_rel_err, color='white', linestyle='--', linewidth=1,
            xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1]
        )
        ax2.set_yscale('log')
        ax2.set_xlabel(f'Entries {unit}')
        ax2.set_ylabel('Relative Error')
        # Save figure
        plt.tight_layout()
        plt.savefig(self.save_figure_path, dpi=plot_dpi)


class ErrorOrbitalResoluteDistributionAnalyzer(BaseAnalyzer):
    TAG = "error_orbital_resolute_distribution"
    
    def __init__(self, 
        pred_dft_dir, bm_dft_dir=None, target_name="H", 
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, pred_only=False, onsite_only=False, parallel_num=1,
        tier_num=0,
    ):
        super().__init__(
            pred_dft_dir, bm_dft_dir, target_name,
            standardize_gauge, consider_overlap_mask,
            data_split_json, data_split_tags, cache_res, parallel_num, tier_num
        )
        self.pred_only = pred_only
        self.onsite_only = onsite_only
        # Middle data
        self.elem_orbs_map = {}
        self.spinful = None
        self.atom_pairs_error_sum = {}
        self.atom_pairs_count = {}
        # Result
        self.elem_list = None
        self.result_matrix = None
    
    def _cache_the_result(self):
        with h5py.File(self.cached_result_path, "w") as h5file:
            h5file.create_dataset("spinful", data=self.spinful)
            h5file.create_dataset(
                "elem_orbs_map", data=json.dumps(self.elem_orbs_map)
            )
            h5file.create_dataset("result_matrix", data=self.result_matrix)
    
    def _load_cached_result(self):
        with h5py.File(self.cached_result_path, 'r') as h5file:
            self.spinful = h5file["spinful"][()]
            self.elem_orbs_map = \
                json.loads(str(h5file["elem_orbs_map"][()].decode('utf-8')))
            self.elem_list = self._get_elem_list_from_map()
            self.result_matrix = \
                np.array(h5file["result_matrix"][:], dtype=np.float64)
            self.average_error = np.mean(self.result_matrix)
            print(f"[info] Orbital average error: {self.average_error:.3e} eV")
    
    def _postprocess_results(self, results):
        self._parse_results(results)
        self._combine_parsed_results()

    def _analysis_one_structure(self, sid, target_name):
        # Get Errors
        atom_pairs, boundaries, shapes, bm_entries, pred_entries, atoms_elem,\
            elem_orbs_map, spinful, overlap = _read_in_all_necessary_data(
                sid, target_name, self.bm_dft_dir, self.pred_dft_dir
            )
        if self.pred_only:
            bm_entries = np.zeros_like(pred_entries)
        _errors = _get_error_entries(
            pred_entries, bm_entries, overlap,
            self.standardize_gauge, self.consider_overlap_mask,
            atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map,
            fix_shape=True
        )
        # Parse the error chunks with elem-orb pairs
        atom_pairs_count = {}
        atom_pairs_error_sum = {}
        for i_ap, ap in enumerate(atom_pairs):
            elem_i, elem_j = atoms_elem[ap[3]], atoms_elem[ap[4]]
            _self_loop = (ap[0]==0 and ap[1]==0 and ap[2]==0 and ap[3]==ap[4])
            chunk_key = (elem_i, elem_j)
            chunk_error = _errors[boundaries[i_ap]:boundaries[i_ap+1]].reshape(shapes[i_ap])
            count = 1
            #
            if self.onsite_only and (not _self_loop):
                chunk_error = np.zeros_like(chunk_error)
                count = 0
            #
            atom_pairs_count[chunk_key] = atom_pairs_count.get(chunk_key, 0) + count
            atom_pairs_error_sum[chunk_key] = atom_pairs_error_sum.get(chunk_key, 0.0) + chunk_error
        # Return!
        return {
            "elem_orbs_map": elem_orbs_map,
            "spinful": spinful,
            "atom_pairs_count": atom_pairs_count,
            "atom_pairs_error_sum": atom_pairs_error_sum,
        }
    
    def _parse_results(self, results):
        self.elem_orbs_map = {}
        self.spinful = None
        self.atom_pairs_error_sum = {}
        self.atom_pairs_count = {}
        for data in results:
            for elem, curr_orb in data["elem_orbs_map"].items():
                saved_orb = self.elem_orbs_map.get(elem, None)
                if saved_orb is None:
                    self.elem_orbs_map[elem] = curr_orb
                elif saved_orb != curr_orb:
                    raise ValueError(f"The orbital of element {elem} ({curr_orb}) dose not agree with the dataset used {saved_orb}.")
            # Set and check spinful
            if self.spinful is None:
                self.spinful = data["spinful"]
            elif data["spinful"] != self.spinful:
                raise ValueError(f"The spinful is not agree with the dataset's({self.spinful}).")
            # atom_pairs_count and errors
            for chunk_key in data["atom_pairs_error_sum"].keys():
                self.atom_pairs_error_sum[chunk_key] = \
                    self.atom_pairs_error_sum.get(chunk_key, 0) + \
                    data["atom_pairs_error_sum"][chunk_key]
                self.atom_pairs_count[chunk_key] = \
                    self.atom_pairs_count.get(chunk_key, 0) + \
                    data["atom_pairs_count"][chunk_key]

    def _combine_parsed_results(self):
        elem_list = self._get_elem_list_from_map()
        elem_orb_size_list = [sum(2 * orb + 1 for orb in self.elem_orbs_map[elem]) * (1 + self.spinful) for elem in elem_list]
        elem_orb_accum_size_list = [0] + list(accumulate(elem_orb_size_list))
        result_matrix_size = sum(elem_orb_size_list)
        result_matrix = np.zeros((result_matrix_size, result_matrix_size))
        for i, elem_i in enumerate(elem_list):
            i_start = elem_orb_accum_size_list[i]
            i_end = elem_orb_accum_size_list[i+1]
            for j, elem_j in enumerate(elem_list):
                chunk_key = (elem_i, elem_j)
                j_start = elem_orb_accum_size_list[j]
                j_end = elem_orb_accum_size_list[j+1]
                if chunk_key in self.atom_pairs_error_sum.keys():
                    val = self.atom_pairs_error_sum[chunk_key] / max(self.atom_pairs_count[chunk_key], 1)
                else:
                    val = 0.0
                result_matrix[i_start:i_end, j_start:j_end] = val
        # Result
        self.elem_list = elem_list
        self.result_matrix = result_matrix
        self.average_error = np.mean(result_matrix)
        print(f"[info] Orbital average error: {self.average_error:.3e} eV")
    
    def _get_elem_list_from_map(self):
        return sorted(
            list(self.elem_orbs_map.keys()),
            key=lambda x: PERIODIC_TABLE_SYMBOL_TO_INDEX.get(x, float('inf'))
        )

    def plot(self, 
        plot_vmax=None, plot_vmin=None, plot_with_log_scale=False, plot_dpi=300, unit='(meV)', scale=1000
    ):
        # Plot the main figure
        plot_norm = None
        if plot_with_log_scale:
            plot_norm = mcolors.LogNorm(
                vmin=self.result_matrix.min(), vmax=self.result_matrix.max()
            )
        plt.imshow(
            self.result_matrix*scale, cmap='Blues', norm=plot_norm,
            vmax=plot_vmax, vmin=plot_vmin
        )
        # Get the plot info
        orbital_boundaries, elem_ticks_coords, elem_ticks_names, elem_boundaries = \
            self._get_orbital_and_elem_plot_info()
        # Set the orbital type boundary lines
        for cut_line in orbital_boundaries:
            plt.axvline(x=cut_line-0.5, linestyle='--', color='black', linewidth=0.5)
            plt.axhline(y=cut_line-0.5, linestyle='--', color='black', linewidth=0.5)
        # Set the element type ticks and boundary lines
        ax = plt.gca()
        ax.tick_params(axis='x', length=0, labeltop=True, labelbottom=False)
        ax.tick_params(axis='y', length=0)
        ax.set_xticks(elem_ticks_coords)
        ax.set_xticklabels(elem_ticks_names, fontsize=12)
        ax.set_yticks(elem_ticks_coords)
        ax.set_yticklabels(elem_ticks_names, fontsize=12)
        for cut_line in elem_boundaries:
            plt.axvline(x=cut_line-0.5, linestyle='-', color='black', linewidth=1)
            plt.axhline(y=cut_line-0.5, linestyle='-', color='black', linewidth=1)
        # Color bar and other misc things
        cbar = plt.colorbar()
        cbar.set_label(f"MAE {unit}")
        cbar.ax.tick_params(axis='y', which='both', direction='in')
        # Save the figure
        plt.savefig(self.save_figure_path, dpi=plot_dpi)

    def _get_orbital_and_elem_plot_info(self):
        orbital_boundaries = [0, ]
        elem_ticks_coords = []
        elem_ticks_names = []
        elem_boundaries = [0, ]
        orb_num = 0
        for elem in self.elem_list:
            for spin in range(1+self.spinful):
                spin_label = "" if not self.spinful else "↑" if spin == 0 else "↓"
                local_orb_num = 0
                for ll in self.elem_orbs_map[elem]:
                    orb_num += (2 * ll + 1)
                    local_orb_num += (2 * ll + 1)
                    orbital_boundaries.append(orb_num)
                elem_ticks_coords.append(orb_num-1-((local_orb_num-1)/2))
                elem_ticks_names.append(elem+spin_label)
                elem_boundaries.append(orb_num)
        return orbital_boundaries, elem_ticks_coords, elem_ticks_names, elem_boundaries


class ErrorElementsPairDistributionAnalyzer(BaseAnalyzer):
    TAG = "error_elements_pair_distribution"
    
    def __init__(self, 
        pred_dft_dir, bm_dft_dir=None, target_name="H", 
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, pred_only=False, onsite_only=False, parallel_num=1,
        tier_num=0,
    ):
        super().__init__(
            pred_dft_dir, bm_dft_dir, target_name,
            standardize_gauge, consider_overlap_mask,
            data_split_json, data_split_tags, cache_res, parallel_num, tier_num
        )
        self.pred_only = pred_only
        self.onsite_only = onsite_only
        # Middle data
        self.elem_list = {}
        self.spinful = None
        self.atom_pairs_error_sum = {}
        self.atom_pairs_count = {}
        # Result
        self.elem_list = None
        self.result_matrix = None
    
    def _cache_the_result(self):
        with h5py.File(self.cached_result_path, "w") as h5file:
            h5file.create_dataset("spinful", data=self.spinful)
            h5file.create_dataset("elem_list", data=json.dumps(self.elem_list))
            h5file.create_dataset("result_matrix", data=self.result_matrix)
    
    def _load_cached_result(self):
        with h5py.File(self.cached_result_path, 'r') as h5file:
            self.spinful = h5file["spinful"][()]
            self.elem_list = json.loads(str(h5file["elem_list"][()].decode('utf-8')))
            self.result_matrix = \
                np.array(h5file["result_matrix"][:], dtype=np.float64)
            self.average_error = np.mean(self.result_matrix)
            print(f"[info] Elements-pair average error: {self.average_error:.3e} eV")
    
    def _postprocess_results(self, results):
        self._parse_results(results)
        self._combine_parsed_results()

    def _analysis_one_structure(self, sid, target_name):
        # Get Errors
        atom_pairs, boundaries, shapes, bm_entries, pred_entries, atoms_elem,\
            elem_orbs_map, spinful, overlap = _read_in_all_necessary_data(
                sid, target_name, self.bm_dft_dir, self.pred_dft_dir
            )
        if self.pred_only:
            bm_entries = np.zeros_like(pred_entries)
        _errors = _get_error_entries(
            pred_entries, bm_entries, overlap,
            self.standardize_gauge, self.consider_overlap_mask,
            atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map,
            fix_shape=True
        )
        # Parse the error chunks with elem-orb pairs
        atom_pairs_count = {}
        atom_pairs_error_sum = {}
        for i_ap, ap in enumerate(atom_pairs):
            elem_i, elem_j = atoms_elem[ap[3]], atoms_elem[ap[4]]
            _self_loop = (ap[0]==0 and ap[1]==0 and ap[2]==0 and ap[3]==ap[4])
            chunk_key = (elem_i, elem_j)
            chunk_error = np.mean(_errors[boundaries[i_ap]:boundaries[i_ap+1]])
            count = 1
            #
            if self.onsite_only and (not _self_loop):
                chunk_error, count = 0.0, 0
            #
            atom_pairs_count[chunk_key] = atom_pairs_count.get(chunk_key, 0) + count
            atom_pairs_error_sum[chunk_key] = atom_pairs_error_sum.get(chunk_key, 0.0) + chunk_error
        # Return!
        return {
            "elem_list": list(elem_orbs_map.keys()),
            "spinful": spinful,
            "atom_pairs_count": atom_pairs_count,
            "atom_pairs_error_sum": atom_pairs_error_sum,
        }
    
    def _parse_results(self, results):
        self.elem_list = []
        self.spinful = None
        self.atom_pairs_error_sum = {}
        self.atom_pairs_count = {}
        for data in results:
            for elem in data["elem_list"]:
                if elem not in self.elem_list:
                    self.elem_list.append(elem)
            # Set and check spinful
            if self.spinful is None:
                self.spinful = data["spinful"]
            elif data["spinful"] != self.spinful:
                raise ValueError(f"The spinful is not agree with the dataset's({self.spinful}).")
            # atom_pairs_count and errors
            for chunk_key in data["atom_pairs_error_sum"].keys():
                self.atom_pairs_error_sum[chunk_key] = \
                    self.atom_pairs_error_sum.get(chunk_key, 0) + \
                    data["atom_pairs_error_sum"][chunk_key]
                self.atom_pairs_count[chunk_key] = \
                    self.atom_pairs_count.get(chunk_key, 0) + \
                    data["atom_pairs_count"][chunk_key]

    def _combine_parsed_results(self):
        result_matrix_size = len(self.elem_list)
        result_matrix = np.zeros((result_matrix_size, result_matrix_size))
        self._sort_elem_list()
        for i, elem_i in enumerate(self.elem_list):
            for j, elem_j in enumerate(self.elem_list):
                chunk_key = (elem_i, elem_j)
                val = 0.0
                if chunk_key in self.atom_pairs_error_sum.keys():
                    val = self.atom_pairs_error_sum[chunk_key] / max(self.atom_pairs_count[chunk_key], 1)
                result_matrix[i, j] = val
        # Result
        self.result_matrix = result_matrix
        self.average_error = np.mean(result_matrix)
        print(f"[info] Elements-pair average error: {self.average_error:.3e} eV")

    def _sort_elem_list(self):
        self.elem_list = [
            PERIODIC_TABLE_INDEX_TO_SYMBOL[e_i] for e_i in sorted(
                PERIODIC_TABLE_SYMBOL_TO_INDEX[elem] for elem in self.elem_list
            )
        ]

    def plot(self, 
        plot_vmax=None, plot_vmin=None, plot_with_log_scale=False, plot_dpi=300, unit='(meV)', scale=1000
    ):
        # Get the ticks
        ticks_coord = list(range(len(self.elem_list)))
        ticks_lable = self.elem_list
        ticks_lable_size = min(330/len(self.elem_list), 12)
        # Plot the main figure
        plot_norm = None
        if plot_with_log_scale:
            plot_norm = mcolors.LogNorm(
                vmin=self.result_matrix.min(), vmax=self.result_matrix.max()
            )
        plt.imshow(
            self.result_matrix*scale, cmap='Blues', norm=plot_norm,
            vmax=plot_vmax, vmin=plot_vmin
        )
        # Set the element type ticks and boundary lines
        ax = plt.gca()
        ax.tick_params(axis='x', length=0, labeltop=True, labelbottom=False)
        ax.tick_params(axis='y', length=0)
        ax.set_xticks(ticks_coord)
        ax.set_xticklabels(ticks_lable, fontsize=ticks_lable_size)
        ax.set_yticks(ticks_coord)
        ax.set_yticklabels(ticks_lable, fontsize=ticks_lable_size)
        if len(self.elem_list) > 15:
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % 2 == 0:
                    label.set_y(1.0+(0.06-len(self.elem_list)/2000))
            for i, label in enumerate(ax.yaxis.get_ticklabels()):
                if i % 2 == 0:
                    label.set_x(0.0-(0.1-len(self.elem_list)/1000))
        for cut_line in ticks_coord:
            plt.axvline(x=cut_line-0.5, linestyle='--', color='black', linewidth=0.2)
            plt.axhline(y=cut_line-0.5, linestyle='--', color='black', linewidth=0.2)
        # Color bar and other misc things
        cbar = plt.colorbar()
        cbar.set_label(f"MAE {unit}")
        cbar.ax.tick_params(axis='y', which='both', direction='in')
        # Save the figure
        plt.savefig(self.save_figure_path, dpi=plot_dpi)


class ErrorElementsDistributionAnalyzer(BaseAnalyzer):
    TAG = "error_elements_distribution"
    
    def __init__(self, 
        pred_dft_dir, bm_dft_dir=None, target_name="H", 
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, parallel_num=1, tier_num=0,
    ):
        super().__init__(
            pred_dft_dir, bm_dft_dir, target_name,
            standardize_gauge, consider_overlap_mask,
            data_split_json, data_split_tags, cache_res, parallel_num, tier_num
        )
        self.cached_result_path = self.root_dir / f'{self.TAG}.json'
        # Result
        self.elem_error = None
        self.elem_count = None
        self.average_error = None
    
    def _cache_the_result(self):
        data = {
            "error": self.elem_error,
            "count": self.elem_count,
        }
        dump_json_file(self.cached_result_path, data)
    
    def _load_cached_result(self):
        data = load_json_file(self.cached_result_path)
        self.elem_error = data["error"]
        self.elem_count = data["count"]
        self.average_error = np.mean(list(self.elem_error.values()))
        print(f"[info] Elements average error: {self.average_error:.3e} eV")
        
    def _analysis_one_structure(self, sid, target_name):
        try:
            return self._analysis_one_structure_core(sid, target_name)
        except Exception as e:
            print(f"Error in analyzing structure {sid}: {e}")
            return None
    
    def _analysis_one_structure_core(self, sid, target_name):
        # Get Errors
        atom_pairs, boundaries, shapes, bm_entries, pred_entries, atoms_elem,\
            elem_orbs_map, spinful, overlap = _read_in_all_necessary_data(
                sid, target_name, self.bm_dft_dir, self.pred_dft_dir
            )
        _errors = _get_error_entries(
            pred_entries, bm_entries, overlap,
            self.standardize_gauge, self.consider_overlap_mask, 
            atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map
        )
        _error = np.mean(_errors)
        # Return!
        return {"elements": list(elem_orbs_map.keys()), "error": _error}
    
    def _postprocess_results(self, results):
        self.elem_error = {}
        self.elem_count = {}
        for data in results:
            if data is None:
                continue
            for elem in data["elements"]:
                _prev_elem_count = self.elem_count.get(elem, 0)
                _prev_elem_error = self.elem_error.get(elem, 0.0)
                self.elem_count[elem] = _prev_elem_count + 1
                self.elem_error[elem] = (
                    _prev_elem_error * _prev_elem_count + data["error"]
                ) / self.elem_count[elem]
        #
        self.average_error = np.mean(list(self.elem_error.values()))
        print(f"[info] Elements average error: {self.average_error:.3e} eV")

    def plot(self,
        plot_dpi=300, unit='(meV)', scale=1000, E_min=0.2, E_max=1.0
    ):
        CELL_LENGTH = 1
        CELL_GAP = 0.1
        CELL_EDGE_WIDTH = 0.6
        ELEMENT_RANGE = list(range(1,58))+list(range(72,87))
        XY_LENGTH = (18, 6)
        # Get elements
        elements = []
        for i in ELEMENT_RANGE:
            ele = mendeleev.element(i)
            elements.append([
                i, ele.symbol, ele.group_id, ele.period,
                self.elem_error.get(ele.symbol, 0.0) * scale
            ])
        # Plot
        fig = plt.figure(figsize=(8, 3.6))
        ax = fig.gca()
        my_cmap = plt.colormaps['YlOrRd']
        norm = mpl.colors.Normalize(E_min, E_max)
        my_cmap.set_under('None')
        mapable = cm.ScalarMappable(norm, my_cmap)
        cbar = plt.colorbar(mapable, ax=ax, fraction=0.05,pad=0.01,shrink=0.65)
        cbar.ax.tick_params(labelsize=12, direction='in')

        def get_x_y(i, j):
            x = (CELL_LENGTH + CELL_GAP) * (i - 1)
            y = XY_LENGTH[1] - ((CELL_LENGTH + CELL_GAP) * j)
            return x,y

        # Plot the cell for each element
        for e in elements:
            ele_number, ele_symbol, ele_group, ele_period, elem_error = e
            x, y = get_x_y(ele_group,ele_period)
            # For La and Ac outside series block
            if ele_period >= 8:
                y -= CELL_LENGTH * 0.5
            # For La and Ac inside one block
            if ele_number:
                fill_color = my_cmap(norm(elem_error))
                rect = patches.Rectangle(
                    xy=(x, y), width=CELL_LENGTH, height=CELL_LENGTH,
                    linewidth=CELL_EDGE_WIDTH, edgecolor='k',
                    facecolor=fill_color
                )
                plt.gca().add_patch(rect)

            # Add Element Number
            plt.text(
                x + 0.04, y + 0.8,
                ele_number,
                va='center', ha='left',
                fontdict={'size': 6, 'color': 'black'}
            )
            # Add Element Symbol
            plt.text(
                x + 0.5, y + 0.5,
                ele_symbol,
                va='center', ha='center',
                fontdict={'size': 9, 'color': 'black', 'weight': 'bold'}
            )
            # Add Error Value
            plt.text(
                x + 0.5, y + 0.12,
                "{:.2f}".format(elem_error) if elem_error else '',
                va='center', ha='center',
                fontdict={'size': 6, 'color': 'black'}
            )
        # Add title
        x,y=get_x_y(8,1)
        plt.text(
            x + 0.5, y + 0.5,
            f'MAE for each element {unit}',
            va='center', ha='center',
            fontdict={'size': 12, 'color': 'black', 'weight': 'bold'}
        )
        # Plot
        plt.axis('off')
        plt.ylim(0, XY_LENGTH[1])
        plt.xlim(0, XY_LENGTH[0])
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.save_figure_path, dpi=plot_dpi)


class ErrorStructureDistributionAnalyzer(BaseAnalyzer):
    TAG = "error_structure_distribution"
    
    def __init__(self, 
        pred_dft_dir, bm_dft_dir=None, target_name="H", 
        standardize_gauge=False, consider_overlap_mask=False,
        data_split_json=None, data_split_tags="train,validate,test",
        cache_res=False, parallel_num=1, tier_num=0,
    ):
        super().__init__(
            pred_dft_dir, bm_dft_dir, target_name,
            standardize_gauge, consider_overlap_mask,
            data_split_json, data_split_tags, cache_res, parallel_num, tier_num
        )
        self.cached_result_path = self.root_dir / f'{self.TAG}.json'
        # Result
        self.errors = None
        self.average_error = None

    def _cache_the_result(self):
        data = {"sids": self.sids, "errors": list(self.errors),}
        dump_json_file(self.cached_result_path, data)
    
    def _load_cached_result(self):
        data = load_json_file(self.cached_result_path)
        self.sids = data["sids"]
        self.errors = np.array(data["errors"])
        self.average_error = np.mean(self.errors)
        print(f"[info] Structures average error: {self.average_error:.3e} eV")

    def _analysis_one_structure(self, sid, target_name):
        try:
            return self._analysis_one_structure_core(sid, target_name)
        except Exception as e:
            print(f"Error in analyzing structure {sid}: {e}")
            return None

    def _analysis_one_structure_core(self, sid, target_name):
        # Get Errors
        atom_pairs, boundaries, shapes, bm_entries, pred_entries, atoms_elem,\
            elem_orbs_map, _spinful, overlap = _read_in_all_necessary_data(
                sid, target_name, self.bm_dft_dir, self.pred_dft_dir
            )
        _errors = _get_error_entries(
            pred_entries, bm_entries, overlap,
            self.standardize_gauge, self.consider_overlap_mask, 
            atom_pairs, boundaries, shapes, atoms_elem, elem_orbs_map
        )
        _error = np.mean(_errors)
        # Return!
        return sid, _error

    def _postprocess_results(self, results):
        self.sids = [val[0] for val in results]
        self.errors = np.array([val[1] for val in results])
        self.average_error = np.mean(self.errors)
        print(f"[info] Structures average error: {self.average_error:.3e} eV")

    def plot(self,
        plot_dpi=300, unit='(meV)', scale=1000, xlims=None, ylims=None
    ):
        # Get plot x and y
        from scipy.stats import gaussian_kde
        sorted_data = np.sort(self.errors)
        log_sorted_data = np.log(sorted_data)
        x = np.linspace(np.min(log_sorted_data), np.max(log_sorted_data), 4000)
        #
        x_accum = sorted_data * scale
        y_accum = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        x_density = np.exp(x) * scale
        y_density = gaussian_kde(log_sorted_data)(x)
        x_mean = np.mean(self.errors) * scale
        #
        plt.rcParams["font.size"] = 16
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #
        ax1.plot(
            x_accum, y_accum, linestyle='-', color='black',
            linewidth=3, label='Cumulative distribution'
        )
        ax2.plot(
            x_density, y_density, color='blue', linewidth=3,
            label='Probability density'
        )
        ax2.axvline(
            x_mean, color='gray', linestyle='--', label='Average MAE'
        )
        #
        ax1.set_xlabel(f'MAE {unit}')
        ax1.set_ylabel('Cumulative distribution')
        ax1.set_ylim(0.00, 1.00)
        ax1.grid(False)
        ax2.set_ylabel('Probability density', color='blue')
        ax2.set_ylim(ylims if ylims is not None else [0.00, None])
        ax2.grid(False)
        #
        ax1.set_xscale('log')
        ax1.set_xlim(
            xlims if xlims is not None else [min(x_accum), max(x_accum)]
        )
        ax1.tick_params(
            axis='both', direction='in', which='both', width=1.5,
            top=True, right=False, bottom=True, left=True
        )
        ax2.tick_params(
            axis='y', direction='in', which='both', width=1.5, colors='blue',
            top=False, right=True, bottom=False, left=False
        )
        ax1.spines[['top', 'bottom', 'left']].set_linewidth(1.5)
        ax1.spines['right'].set_visible(False)
        ax2.spines[['top', 'bottom', 'left']].set_visible(False)
        ax2.spines['right'].set_linewidth(1.5)
        ax2.spines['right'].set_color('blue')
        #
        plt.tight_layout()
        plt.savefig(self.save_figure_path, dpi=plot_dpi)

