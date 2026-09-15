"""Regression coverage for position blocks with different orbital counts."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from deepx_dock.convert.deeph.translate_old_dataset_to_new import NewDatasetTranslator


@pytest.mark.parametrize("spinful", [False, True])
def test_position_upgrade_preserves_rectangular_blocks(tmp_path: Path, spinful: bool) -> None:
    factor = 1 + spinful
    shapes = [(factor, factor), (2 * factor, factor)]
    blocks = [np.arange(np.prod(shape)).reshape(shape) + (index + 1j) for index, shape in enumerate(shapes)]
    old_path = tmp_path / "positions.h5"
    new_path = tmp_path / "position_matrix.h5"
    with h5py.File(old_path, "w") as handle:
        handle[json.dumps([0, 0, 0, 1, 1, 1])] = blocks[0]
        handle[json.dumps([1, 0, 0, 2, 1, 3])] = blocks[1]

    pairs = [[1, 0, 0, 0, 1], [0, 0, 0, 1, 1]]
    NewDatasetTranslator._transfer_old_position_to_new(
        old_path,
        new_path,
        isspinful=spinful,
        elem_indices=[1, 2],
        orbs_save={"H": [0], "He": [0, 0]},
        coords_order_list=[1, 0],
        atom_pairs_order=pairs,
    )

    expected = np.zeros((3, blocks[1].size + blocks[0].size), dtype=complex)
    expected[2, : blocks[1].size] = blocks[1].ravel()
    expected[0, blocks[1].size :] = blocks[0].ravel()
    with h5py.File(new_path, "r") as handle:
        np.testing.assert_array_equal(handle["atom_pairs"][:], pairs)
        np.testing.assert_array_equal(handle["chunk_shapes"][:], shapes[::-1])
        np.testing.assert_array_equal(handle["chunk_boundaries"][:], [0, blocks[1].size, expected.shape[1]])
        np.testing.assert_array_equal(handle["entries"][:], expected)
