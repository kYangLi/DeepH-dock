from pathlib import Path
from functools import partial
from typing import Dict

import h5py
import numpy as np

from deepx_dock.parallel import parallel_map
from deepx_dock.misc import dump_json_file, load_json_file, load_poscar_file, get_data_dir_lister
from deepx_dock.CONSTANT import DEEPX_E0S_FILENAME, DEEPX_FORCE_FILENAME, DEEPX_POSCAR_FILENAME


def _make_validation_check(target_filename: str):
    def _validation_check(root_dir: Path, prev_dirname: Path):
        all_files = {v.name for v in root_dir.iterdir()}
        if {DEEPX_POSCAR_FILENAME, target_filename}.issubset(all_files):
            yield prev_dirname
        else:
            print(f"Skip {prev_dirname}: missing {DEEPX_POSCAR_FILENAME} or {target_filename}.")

    return _validation_check


class DatasetForceStandardize:
    """Shift/unshift per-element reference energies (E0s) for force field data.

    Computes per-element reference energies via least-squares regression and
    applies the energy shift to force data files. Supports both forward (shift)
    and backward (unshift) operations.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing the dataset.
    backward : bool, default False
        If True, unshift (add back E0s). Requires ``e0s_file``.
    filename : str, default "force.h5"
        Target HDF5 filename within each data directory.
    e0s_file : str | Path | None, default None
        Path to e0s.json for loading E0s. Required for backward.
        If None in forward mode, E0s are computed from the dataset.
    e0s_output : str | Path | None, default None
        Path to save e0s.json in forward-compute mode.
        Defaults to ``./e0s.json`` in the current working directory.
    n_jobs : int, default -1
        Parallel processing number, -1 for all cores.
    n_tier : int, default 0
        Directory tier depth for data directory listing.
    """

    def __init__(
        self,
        data_dir,
        backward=False,
        filename=DEEPX_FORCE_FILENAME,
        e0s_file=None,
        e0s_output=None,
        n_jobs=-1,
        n_tier=0,
    ):
        self.data_dir = Path(data_dir)
        self.backward = backward
        self.filename = filename
        self.e0s_file = Path(e0s_file) if e0s_file else None
        self.e0s_output = Path(e0s_output) if e0s_output else Path(DEEPX_E0S_FILENAME)
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.data_dir.is_dir(), f"{data_dir} is not a directory"

        if backward:
            assert self.e0s_file is not None, "--e0s-file is required for --backward mode"

    def standardize_all(self):
        if self.e0s_file is not None:
            e0s_data = load_json_file(self.e0s_file)
            e0s = e0s_data["e0s"]
            already_standardized = e0s_data.get("standardized", False)
            if not self.backward and already_standardized:
                print(
                    f"[warning] The e0s file '{self.e0s_file}' indicates the dataset "
                    "has already been standardized. Applying again may cause double-shift."
                )
        else:
            e0s = self._compute_e0s()
            e0s_data = None

        validation_check = _make_validation_check(self.filename)
        data_dir_lister = get_data_dir_lister(self.data_dir, self.n_tier, validation_check)

        worker = partial(
            self._shift_one,
            all_data_dir=self.data_dir,
            filename=self.filename,
            e0s=e0s,
            backward=self.backward,
        )
        parallel_map(worker, data_dir_lister, n_jobs=self.n_jobs, desc="Data")

        new_standardized = not self.backward
        if e0s_data is not None:
            e0s_data["standardized"] = new_standardized
            try:
                dump_json_file(self.e0s_file, e0s_data)
            except Exception as e:
                print(f"[warning] Failed to update e0s.json at {self.e0s_file}: {e}")
        else:
            e0s_data = {"e0s": e0s, "standardized": new_standardized}
            dump_json_file(self.e0s_output, e0s_data)
            print(f"[info] E0s saved to {self.e0s_output}")

    def _compute_e0s(self) -> Dict[str, float]:
        validation_check = _make_validation_check(self.filename)
        data_dir_lister = get_data_dir_lister(self.data_dir, self.n_tier, validation_check)

        element_set = set()
        A_rows = []
        B_vals = []

        for dir_name in data_dir_lister:
            dft_dir = self.data_dir / dir_name
            poscar_data = load_poscar_file(dft_dir / DEEPX_POSCAR_FILENAME)
            elements_unique = poscar_data["elements_unique"]
            elements_counts = poscar_data["elements_counts"]
            element_set.update(elements_unique)

            with h5py.File(dft_dir / self.filename, "r") as f:
                energy = float(f["energy"][()])

            row = {e: 0 for e in element_set}
            for e, c in zip(elements_unique, elements_counts):
                row[e] = row.get(e, 0) + c

            A_rows.append(row)
            B_vals.append(energy)

        element_list = sorted(element_set)
        A = np.array(
            [[row.get(e, 0) for e in element_list] for row in A_rows],
            dtype=np.float64,
        )
        B = np.array(B_vals, dtype=np.float64)

        try:
            e0s_arr, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
            print(f"[info] E0s computed via least squares (rank={rank}).")
        except np.linalg.LinAlgError:
            print("[warning] Failed to compute E0s via least squares, using zeros.")
            e0s_arr = np.zeros(len(element_list))

        e0s = {e: float(v) for e, v in zip(element_list, e0s_arr)}
        print(f"[info] Per-element E0s: {e0s}")
        return e0s

    @staticmethod
    def _shift_one(
        dir_name,
        all_data_dir,
        filename,
        e0s,
        backward,
    ):
        try:
            dft_dir = Path(all_data_dir) / dir_name
            poscar_data = load_poscar_file(dft_dir / DEEPX_POSCAR_FILENAME)
            elements_unique = poscar_data["elements_unique"]
            elements_counts = poscar_data["elements_counts"]

            e0_sum = sum(e0s.get(e, 0.0) * c for e, c in zip(elements_unique, elements_counts))

            h5_path = dft_dir / filename

            with h5py.File(h5_path, "r+") as f:
                energy = f["energy"][()]
                if backward:
                    f["energy"][()] = energy + e0_sum
                else:
                    f["energy"][()] = energy - e0_sum
        except Exception as e:
            print(f"Error in {dir_name}: {e}")
