from itertools import accumulate
import numpy as np
import h5py
import json
import time
from pathlib import Path
from functools import partial
import os

import pytest

from deepx_dock.compute.overlap.overlap import (
    calc_overlap_in_memory,
    OPENMX_DEFAULT_ECUT, OPENMX_DEFAULT_KDENSE,
)
from deepx_dock.compute.overlap.position import calc_overlap_and_position_in_memory
from deepx_dock.misc import load_json_file, load_poscar_file, get_data_dir_lister


SPLINES_FOR_OVERLAP = {}
SPLINES_FOR_POSITION = {}

NUM_WORKERS = os.cpu_count() or 1


def _HPRO_ok():
    import inspect
    try:
        from HPRO.v2h.twocenter import calc_overlap as hpro_calc_overlap
        from HPRO.v2h.twocenter import calc_position as hpro_calc_position
    except Exception as e:
        return False
    params = inspect.signature(hpro_calc_overlap).parameters
    if ('splines' in params) and ('num_workers' in params):
        return True
    else:
        return False


pytestmark = pytest.mark.skipif(not _HPRO_ok(), reason="HPRO is not available or does not support position matrix calculations")


def print_time_interval(tag, start_time):
    end_time = time.perf_counter()
    time_spend = end_time - start_time
    print(f"[time] Time for {tag} {time_spend:.2f} s", flush=True)
    return end_time


def sort_atom_pairs(obs):
    at = obs["atom_pairs"]
    sort_idx = np.lexsort((at[:, 4], at[:, 3], at[:, 2], at[:, 1], at[:, 0]))
    atom_pairs_sorted = obs["atom_pairs"][sort_idx, :]
    chunk_shapes_sorted = obs["chunk_shapes"][sort_idx, :]
    chunk_boundaries_sorted = [0] + list(accumulate(chunk_shapes_sorted[:, 0] * chunk_shapes_sorted[:, 1]))
    entries_sorted = np.concatenate([
        obs["entries"][..., obs["chunk_boundaries"][idx]:obs["chunk_boundaries"][idx+1]] for idx in sort_idx
    ], axis=-1)
    return {
        "atom_pairs": atom_pairs_sorted,
        "chunk_shapes": chunk_shapes_sorted,
        "chunk_boundaries": chunk_boundaries_sorted,
        "entries": entries_sorted,
    }


@pytest.fixture
def dft_dir():
    parent_dir = Path(__file__).resolve().parent
    return parent_dir / "dft.bak"


@pytest.fixture
def basis_dir():
    parent_dir = Path(__file__).resolve().parent
    return parent_dir / "PAO"


@pytest.mark.parametrize("sid", ["0.original", "1.Ga_x", "2.As_x"])
def test_overlap_and_position(sid, dft_dir, basis_dir):
    sid = str(sid).strip("\n")
    print("[do]", sid, flush=True)
    ## structure
    poscar_path = dft_dir / sid / "POSCAR"
    poscar_dict = load_poscar_file(poscar_path)
    ## info, overlap and position matrix
    with h5py.File(dft_dir / sid / "overlap.h5", 'r') as f:
        overlap_openmx = {
            "atom_pairs": np.array(f["atom_pairs"][:]),
            "chunk_boundaries": np.array(f["chunk_boundaries"][:]),
            "chunk_shapes": np.array(f["chunk_shapes"][:]),
            "entries": np.array(f["entries"][:]),
        }
    with h5py.File(dft_dir / sid / "position_matrix.h5", 'r') as f:
        position_openmx = {
            "atom_pairs": np.array(f["atom_pairs"][:]),
            "chunk_boundaries": np.array(f["chunk_boundaries"][:]),
            "chunk_shapes": np.array(f["chunk_shapes"][:]),
            "entries": np.array(f["entries"][:]),
        }
    t1 = time.perf_counter()
    info_hpro, overlap_hpro = calc_overlap_in_memory(
        structure_dict=poscar_dict,
        basis_path=basis_dir,
        aocode="openmx",
        ecut=2.0*OPENMX_DEFAULT_ECUT,
        kdense=2.0*OPENMX_DEFAULT_KDENSE,
        spinful=False,
        splines_for_overlap=SPLINES_FOR_OVERLAP,
        num_workers=NUM_WORKERS,
    )
    t1 = print_time_interval("calc_overlap_in_memory", t1)
    info_hpro, overlap_hpro, position_hpro = calc_overlap_and_position_in_memory(
        structure_dict=poscar_dict,
        basis_path=basis_dir,
        aocode="openmx",
        ecut=2.0*OPENMX_DEFAULT_ECUT,
        kdense=2.0*OPENMX_DEFAULT_KDENSE,
        spinful=False,
        splines_for_overlap=SPLINES_FOR_OVERLAP,
        splines_for_position=SPLINES_FOR_POSITION,
        num_workers=NUM_WORKERS,
    )
    t1 = print_time_interval("calc_overlap_and_position_in_memory", t1)
    info_hpro["spinful"] = False
    info_hpro["fermi_energy_eV"] = None
    ## compare overlap and position_matrix
    overlap_openmx_sorted = sort_atom_pairs(overlap_openmx)
    overlap_hpro_sorted = sort_atom_pairs(overlap_hpro)
    position_openmx_sorted = sort_atom_pairs(position_openmx)
    position_hpro_sorted = sort_atom_pairs(position_hpro)
    if not np.allclose(overlap_hpro_sorted["atom_pairs"], overlap_openmx_sorted["atom_pairs"]):
        raise ValueError("atom_pairs are not the same!")
    overlap_diff = np.abs(overlap_hpro_sorted["entries"] - overlap_openmx_sorted["entries"])
    S_mae = np.mean(overlap_diff)
    S_mxe = np.max(overlap_diff)
    print("S mae", S_mae)
    print("S mxe", S_mxe)
    if not np.allclose(position_hpro_sorted["atom_pairs"], position_openmx_sorted["atom_pairs"]):
        raise ValueError("atom_pairs are not the same!")
    position_diff = np.abs(position_hpro_sorted["entries"] - position_openmx_sorted["entries"])
    r_mae = np.mean(position_diff)
    r_mxe = np.max(position_diff)
    print("r mae", r_mae)
    print("r mxe", r_mxe)
    assert np.allclose(overlap_hpro_sorted["entries"], overlap_openmx_sorted["entries"], atol=3e-4)
    assert np.allclose(position_hpro_sorted["entries"], position_openmx_sorted["entries"], atol=3e-4)


if __name__ == '__main__':
    n_tier = 0
    parent_dir = Path(__file__).resolve().parent
    dft_dir = parent_dir / "dft.bak"
    basis_dir = parent_dir / "PAO"
    outputs_dir = parent_dir

    worker = partial(
        test_overlap_and_position, dft_dir=dft_dir, basis_dir=basis_dir
    )
    results = [
        worker(sid) for sid in get_data_dir_lister(root_dir=dft_dir, depth=n_tier)
    ]
