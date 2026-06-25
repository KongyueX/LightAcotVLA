import pathlib

import numpy as np

from openpi.action_cot import labels


def test_cap_skip_mask_keeps_lowest_entropy_skips():
    skip_mask = np.asarray([1, 0, 1, 1], dtype=np.int8)
    entropy = np.asarray([-2.0, -9.0, -5.0, -1.0], dtype=np.float32)

    capped = labels.cap_skip_mask(skip_mask, entropy, max_skip_segments=1)

    np.testing.assert_array_equal(capped, np.asarray([0, 0, 1, 0], dtype=np.int8))


def test_pad_segment_labels_adds_valid_mask():
    skip_mask = np.asarray([0, 1, 0], dtype=np.int8)

    padded, valid = labels.pad_segment_labels(skip_mask, max_segments=5)

    np.testing.assert_array_equal(padded, np.asarray([0, 1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(valid, np.asarray([1, 1, 1, 0, 0], dtype=np.float32))


def test_action_cot_label_loader_reads_npz(tmp_path: pathlib.Path):
    np.savez_compressed(
        tmp_path / "sample_000007.npz",
        skip_mask=np.asarray([1, 0, 1], dtype=np.int8),
        entropy=np.asarray([-3.0, -2.0, -4.0], dtype=np.float32),
    )
    loader = labels.ActionCotLabelLoader(tmp_path, max_segments=5, max_skip_segments=1)

    item = loader.load(7)

    np.testing.assert_array_equal(
        item["action_cot_skip_mask"],
        np.asarray([0, 0, 1, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        item["action_cot_skip_valid_mask"],
        np.asarray([1, 1, 1, 0, 0], dtype=np.float32),
    )


def test_action_cot_label_dataset_adds_labels(tmp_path: pathlib.Path):
    np.savez_compressed(
        tmp_path / "sample_000000.npz",
        skip_mask=np.asarray([0, 1, 0], dtype=np.int8),
        entropy=np.asarray([-1.0, -3.0, -2.0], dtype=np.float32),
    )
    dataset = labels.ActionCotLabelDataset(
        [{"actions": np.zeros((2, 3), dtype=np.float32)}],
        tmp_path,
        max_segments=5,
        max_skip_segments=1,
    )

    item = dataset[0]

    assert "actions" in item
    np.testing.assert_array_equal(
        item["action_cot_skip_mask"],
        np.asarray([0, 1, 0, 0, 0], dtype=np.float32),
    )
