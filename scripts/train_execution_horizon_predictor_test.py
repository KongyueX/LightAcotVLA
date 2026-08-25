from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("train_execution_horizon_predictor.py")
sys.path.insert(0, str(_SCRIPT.parent))
trainer = importlib.import_module("train_execution_horizon_predictor")


def _arrays() -> dict[str, np.ndarray]:
    task_ids = np.repeat(np.arange(3, dtype=np.uint8), 8)
    episode_ids = np.tile(np.repeat(np.arange(4, dtype=np.uint32), 2), 3)
    return {"task_id": task_ids, "episode_id": episode_ids}


def test_transformer_split_is_episode_disjoint_and_independent_of_training_seed(tmp_path) -> None:
    common = {
        "dataset": (str(tmp_path),),
        "output_dir": str(tmp_path / "out"),
        "temporal_backbone": "transformer",
        "validation_fraction": 0.25,
        "calibration_fraction": 0.25,
        "split_seed": 42,
    }
    first = trainer._split_indices(_arrays(), trainer.Args(seed=7, **common))  # noqa: SLF001
    second = trainer._split_indices(_arrays(), trainer.Args(seed=11, **common))  # noqa: SLF001

    for first_partition, second_partition in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_partition, second_partition)

    arrays = _arrays()
    group_ids = arrays["task_id"].astype(np.uint64) * np.uint64(1_000_000_000)
    group_ids += arrays["episode_id"].astype(np.uint64)
    partition_groups = [set(group_ids[indices].tolist()) for indices in first]
    assert partition_groups[0].isdisjoint(partition_groups[1])
    assert partition_groups[0].isdisjoint(partition_groups[2])
    assert partition_groups[1].isdisjoint(partition_groups[2])
