from __future__ import annotations

import importlib
import json
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("calibrate_execution_horizon_predictor.py")
sys.path.insert(0, str(_SCRIPT.parent))
calibrator = importlib.import_module("calibrate_execution_horizon_predictor")


def test_bootstrap_train_indices_preserve_episode_multiplicity(tmp_path) -> None:
    arrays = {
        "task_id": np.zeros((6,), dtype=np.uint8),
        "episode_id": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint32),
    }
    manifest = {
        "bootstrap_episode_groups": True,
        "bootstrap_train_group_counts": {"0": 2, "1": 1},
        "train_group_ids": [0, 1],
        "validation_group_ids": [2],
        "calibration_group_ids": [2],
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    indices = calibrator._split_indices(arrays, manifest_path, "train")  # noqa: SLF001

    np.testing.assert_array_equal(indices, np.asarray([0, 1, 0, 1, 2, 3]))
