import json
import shutil

import h5py
import numpy as np


def read_matrix_data(path):
    with h5py.File(path, "r") as h5_file:
        return {
            "atom_pairs": h5_file["atom_pairs"][()],
            "chunk_boundaries": h5_file["chunk_boundaries"][()],
            "chunk_shapes": h5_file["chunk_shapes"][()],
            "entries": h5_file["entries"][()],
        }


def _write_matrix_data(path, matrix_data):
    with h5py.File(path, "w") as h5_file:
        for key, value in matrix_data.items():
            h5_file.create_dataset(key, data=value)


def make_split_hkb(source_dir, target_dir):
    shutil.copytree(source_dir, target_dir)
    full_data = read_matrix_data(target_dir / "hamiltonian.h5")
    base_entries = []
    base_shapes = []
    base_boundaries = [0]
    hkb_entries = []

    for index, shape in enumerate(full_data["chunk_shapes"]):
        ni, nj = int(shape[0] // 2), int(shape[1] // 2)
        assert tuple(shape) == (2 * ni, 2 * nj)
        start, end = full_data["chunk_boundaries"][index : index + 2]
        full_block = full_data["entries"][start:end].reshape(shape)
        base_block = np.real(0.5 * (full_block[:ni, :nj] + full_block[ni:, nj:]))
        base_entries.append(base_block.reshape(-1))
        base_shapes.append(base_block.shape)
        base_boundaries.append(base_boundaries[-1] + base_block.size)
        spin_diagonal_base = np.zeros_like(full_block)
        spin_diagonal_base[:ni, :nj] = base_block
        spin_diagonal_base[ni:, nj:] = base_block
        hkb_entries.append((full_block - spin_diagonal_base).reshape(-1))

    base_data = {
        "atom_pairs": full_data["atom_pairs"],
        "chunk_boundaries": np.asarray(base_boundaries, dtype=np.int64),
        "chunk_shapes": np.asarray(base_shapes, dtype=np.int64),
        "entries": np.concatenate(base_entries),
    }
    hkb_data = {**full_data, "entries": np.concatenate(hkb_entries)}
    _write_matrix_data(target_dir / "hamiltonian.h5", base_data)
    _write_matrix_data(target_dir / "hkb.h5", hkb_data)

    info_path = target_dir / "info.json"
    with info_path.open() as file:
        info = json.load(file)
    info["split_hkb"] = True
    with info_path.open("w") as file:
        json.dump(info, file)
    return base_data, hkb_data, info


def expand_overlap_to_spinful(data_dir):
    """Rewrite a scalar overlap file as explicit spin-diagonal blocks."""
    overlap_data = read_matrix_data(data_dir / "overlap.h5")
    expanded_entries = []
    expanded_shapes = []
    expanded_boundaries = [0]
    for index, shape in enumerate(overlap_data["chunk_shapes"]):
        ni, nj = (int(value) for value in shape)
        start, end = overlap_data["chunk_boundaries"][index : index + 2]
        block = overlap_data["entries"][start:end].reshape(ni, nj)
        expanded = np.zeros((2 * ni, 2 * nj), dtype=block.dtype)
        expanded[:ni, :nj] = block
        expanded[ni:, nj:] = block
        expanded_entries.append(expanded.reshape(-1))
        expanded_shapes.append(expanded.shape)
        expanded_boundaries.append(expanded_boundaries[-1] + expanded.size)

    expanded_data = {
        "atom_pairs": overlap_data["atom_pairs"],
        "chunk_boundaries": np.asarray(expanded_boundaries, dtype=np.int64),
        "chunk_shapes": np.asarray(expanded_shapes, dtype=np.int64),
        "entries": np.concatenate(expanded_entries),
    }
    _write_matrix_data(data_dir / "overlap.h5", expanded_data)
    return expanded_data
