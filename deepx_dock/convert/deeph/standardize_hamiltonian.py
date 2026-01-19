from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

from functools import partial
from joblib import Parallel, delayed

from deepx_dock.misc import load_json_file
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.misc import get_data_dir_lister

DEEPX_NECESSARY_FILES = set([DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME])
DEEPX_RAW_HAMILTONIAN_FILENAME = "hamiltonian_raw.h5"


def validation_check_H(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if DEEPX_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname


class DatasetHStandardize:
    def __init__(self, data_dir, h5_overwrite=True, n_jobs=1, n_tier=0):
        self.data_dir = Path(data_dir)
        self.h5_overwrite = h5_overwrite
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.data_dir.is_dir(), f"{data_dir} is not a directory"

    def standardize_all(self):
        worker = partial(
            self.standardize_one,
            all_data_dir=self.data_dir,
            overwrite=self.h5_overwrite,
        )
        data_dir_lister = get_data_dir_lister(
            self.data_dir, self.n_tier, validation_check_H
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name)
            for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def standardize_one(dir_name: str, all_data_dir, overwrite=False):
        try:
            #
            dft_dir_path = Path(all_data_dir) / dir_name
            spinful = DatasetHStandardize._get_spinful_info(dft_dir_path)
            #
            S_path = dft_dir_path / DEEPX_OVERLAP_FILENAME
            S_data = DatasetHStandardize._read_h5(S_path)
            S = S_data[3]
            if spinful:
                S = DatasetHStandardize._convert_overlap_to_spinful(*S_data)
            #
            H_path = dft_dir_path / DEEPX_HAMILTONIAN_FILENAME
            H_data = DatasetHStandardize._read_h5(H_path)
            H = H_data[3]
            #
            _mu = np.dot(H, S) / np.dot(S, S)
            H_std = H - _mu * S
            #
            if overwrite:
                H_path.unlink()
            else:
                H_raw_path = dft_dir_path / DEEPX_RAW_HAMILTONIAN_FILENAME
                H_path.rename(H_raw_path)
            H_std_data = {
                "atom_pairs": H_data[0],
                "chunk_boundaries": H_data[1],
                "chunk_shapes": H_data[2],
                "entries": H_std
            }
            DatasetHStandardize.dump_dft_dict_to_h5(H_path, H_std_data)
        except Exception as e:
            print(f"Error in translating {dir_name}: {e}")

    @staticmethod
    def _get_spinful_info(dft_dir_path):
        info_path = Path(dft_dir_path) / DEEPX_INFO_FILENAME
        return load_json_file(info_path)["spinful"]

    @staticmethod
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
    
    @staticmethod
    def _read_h5(h5_path):
        with h5py.File(h5_path, 'r') as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            dset = f['entries']
            entries_dtype = np.complex128 if dset.dtype.kind == 'c' else np.float64
            entries = np.array(dset[:], dtype=entries_dtype)
        return atom_pairs, boundaries, shapes, entries

    @staticmethod
    def dump_dft_dict_to_h5(file_path: str, data_dict: dict):
        with h5py.File(file_path, "w") as h5file:
            for key, value in data_dict.items():
                h5file.create_dataset(key, data=value)