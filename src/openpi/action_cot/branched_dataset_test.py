from __future__ import annotations

import pathlib

import numpy as np
import pytest

from openpi.action_cot import branched_dataset


def _record(shape: branched_dataset.BranchedDatasetShape, root_id: int = 3) -> dict[str, np.ndarray | int]:
    branches = shape.num_branches
    steps = np.full((branches,), shape.max_executed_steps, dtype=np.uint8)
    executed_valid = np.arange(shape.max_executed_steps)[None, :] < steps[:, None]
    return {
        "root_id": root_id,
        "task_id": 1,
        "episode_id": 2,
        "decision_step": 10,
        "policy_seed": 123,
        "branch_ids": np.arange(branches, dtype=np.uint8),
        "branch_steps": steps,
        "branch_valid": np.ones((branches,), dtype=np.bool_),
        "endpoint_done": np.zeros((branches,), dtype=np.bool_),
        "branch_strength": np.ones((branches,), dtype=np.float32),
        "physics_delta_l2": np.zeros((branches,), dtype=np.float32),
        "anchor_images": np.zeros(
            (shape.num_cameras, shape.image_height, shape.image_width, shape.image_channels),
            dtype=np.uint8,
        ),
        "current_images": np.zeros(
            (
                branches,
                shape.num_cameras,
                shape.image_height,
                shape.image_width,
                shape.image_channels,
            ),
            dtype=np.uint8,
        ),
        "anchor_state": np.zeros((shape.state_dim,), dtype=np.float32),
        "current_state": np.zeros((branches, shape.state_dim), dtype=np.float32),
        "cached_ear": np.zeros((shape.coarse_horizon, shape.action_dim), dtype=np.float32),
        "fresh_ear": np.zeros((branches, shape.coarse_horizon, shape.action_dim), dtype=np.float32),
        "cached_iar": np.zeros((shape.iar_tokens, shape.iar_dim), dtype=np.float32),
        "fresh_iar": np.zeros((branches, shape.iar_tokens, shape.iar_dim), dtype=np.float32),
        "cached_actions": np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        "fresh_actions": np.zeros((branches, shape.action_horizon, shape.action_dim), dtype=np.float32),
        "cached_actions_env": np.zeros((shape.action_horizon, shape.env_action_dim), dtype=np.float32),
        "fresh_actions_env": np.zeros(
            (branches, shape.action_horizon, shape.env_action_dim),
            dtype=np.float32,
        ),
        "executed_actions": np.zeros(
            (branches, shape.max_executed_steps, shape.env_action_dim),
            dtype=np.float32,
        ),
        "executed_valid": executed_valid,
    }


def test_round_trip_and_resume(tmp_path: pathlib.Path) -> None:
    shape = branched_dataset.BranchedDatasetShape(image_height=8, image_width=8, iar_dim=16)
    output = tmp_path / "branches"
    with branched_dataset.ShardedBranchedWriter(
        output,
        shape=shape,
        records_per_shard=1,
        metadata={"seed_protocol": "same_seed_per_root"},
    ) as writer:
        writer.append(_record(shape))
    with branched_dataset.ShardedBranchedWriter(
        output,
        shape=shape,
        records_per_shard=1,
        metadata={"seed_protocol": "same_seed_per_root"},
    ) as writer:
        writer.append(_record(shape, root_id=4))

    arrays = branched_dataset.load_branched_arrays((output,))
    assert arrays["root_id"].tolist() == [3, 4]
    assert arrays["fresh_ear"].shape == (2, shape.num_branches, shape.coarse_horizon, shape.action_dim)
    assert arrays["cached_iar"].dtype == np.float16


def test_rejects_noncanonical_branch_order(tmp_path: pathlib.Path) -> None:
    shape = branched_dataset.BranchedDatasetShape(image_height=8, image_width=8, iar_dim=16)
    record = _record(shape)
    record["branch_ids"] = np.arange(shape.num_branches, dtype=np.uint8)[::-1]
    with (
        pytest.raises(ValueError, match="canonical ordering"),
        branched_dataset.ShardedBranchedWriter(tmp_path, shape=shape) as writer,
    ):
        writer.append(record)
