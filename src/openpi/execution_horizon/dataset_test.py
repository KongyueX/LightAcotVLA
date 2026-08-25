from __future__ import annotations

import json

import h5py
import numpy as np

from openpi.execution_horizon import dataset


def _shape() -> dataset.DatasetShape:
    return dataset.DatasetShape(
        prefix_feature_dim=8,
        state_dim=4,
        action_dim=7,
        coarse_horizon=5,
        action_horizon=20,
        candidate_horizons=(3, 5, 7, 10, 15, 20),
        max_trials=3,
        prefix_token_count=4,
    )


def _record(shape: dataset.DatasetShape) -> dict[str, object]:
    candidates = shape.num_candidates
    trials = shape.max_trials
    success = np.ones((candidates, trials), dtype=np.bool_)
    success[-1, -1] = False
    valid = np.ones_like(success)
    timeout = ~success
    calls = np.broadcast_to(np.arange(1, candidates + 1)[:, None], (candidates, trials)).copy()
    steps = calls * 2
    elapsed = calls.astype(np.float32) / 10.0
    return {
        "prefix_feature": np.zeros((shape.prefix_feature_dim,), dtype=np.float32),
        "prefix_tokens": np.zeros((shape.prefix_token_count, shape.prefix_feature_dim), dtype=np.float32),
        "prefix_token_mask": np.asarray([True, True, True, False]),
        "state": np.zeros((shape.state_dim,), dtype=np.float32),
        "coarse_actions": np.zeros((shape.coarse_horizon, shape.action_dim), dtype=np.float32),
        "final_actions": np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        "previous_actions": np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        "previous_h": 10,
        "previous_valid": True,
        "budget_balance": 0.5,
        "episode_progress": 0.25,
        "final_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "action_cot_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "fused_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "event_mask": np.zeros((shape.action_horizon,), dtype=np.bool_),
        "risk_valid": np.ones((shape.action_horizon,), dtype=np.bool_),
        "hazard_event_count": np.zeros((shape.action_horizon,), dtype=np.uint16),
        "hazard_at_risk_count": np.full((shape.action_horizon,), 3, dtype=np.uint16),
        "raw_h": 10,
        "candidate_horizons": shape.candidate_horizons,
        "branch_success": success[:, 0],
        "branch_timeout": timeout[:, 0],
        "remaining_steps": steps[:, 0],
        "remaining_calls": calls[:, 0],
        "branch_valid": valid[:, 0],
        "success_count": np.sum(success, axis=-1),
        "timeout_count": np.sum(timeout, axis=-1),
        "trial_count": np.sum(valid, axis=-1),
        "remaining_steps_mean": np.mean(steps, axis=-1),
        "remaining_steps_variance": np.var(steps, axis=-1),
        "remaining_calls_mean": np.mean(calls, axis=-1),
        "remaining_calls_variance": np.var(calls, axis=-1),
        "elapsed_mean": np.mean(elapsed, axis=-1),
        "elapsed_variance": np.var(elapsed, axis=-1),
        "trial_success": success,
        "trial_timeout": timeout,
        "trial_remaining_steps": steps,
        "trial_remaining_calls": calls,
        "trial_elapsed": elapsed,
        "trial_valid": valid,
        "dangerous_long_count": np.asarray([0, 0, 0, 0, 0, 1]),
        "paired_trial_count": np.asarray([0, 0, 0, 0, 3, 3]),
        "physics_state": np.arange(9, dtype=np.float64),
        "task_id": 3,
        "episode_id": 7,
        "decision_step": 11,
        "root_seed": 19,
        "source_iteration": 0,
    }


def test_v2_count_aware_roundtrip(tmp_path):
    shape = _shape()
    with dataset.ShardedCounterfactualWriter(tmp_path, shape=shape, records_per_shard=1) as writer:
        writer.append(_record(shape))

    arrays = dataset.load_counterfactual_arrays((tmp_path,), include_physics=True)

    assert arrays["schema_version"].tolist() == [2]
    assert arrays["candidate_horizons"].tolist() == [list(shape.candidate_horizons)]
    assert arrays["trial_success"].shape == (1, shape.num_candidates, shape.max_trials)
    assert arrays["prefix_tokens"].shape == (1, shape.prefix_token_count, shape.prefix_feature_dim)
    assert arrays["dangerous_long_count"][0, -1] == 1
    np.testing.assert_array_equal(arrays["physics_state"][0], np.arange(9, dtype=np.float64))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert dataset.DatasetShape(**manifest["shape"]) == shape


def test_v1_loader_synthesizes_single_trial_counts(tmp_path):
    shape = dataset.DatasetShape(prefix_feature_dim=8, state_dim=4, action_dim=7, coarse_horizon=5)
    target = tmp_path / "shard-00000.h5"
    with h5py.File(target, "w") as handle:
        handle.attrs["schema_version"] = dataset.LEGACY_SCHEMA_VERSION
        for name, (dtype, dimensions) in dataset._LEGACY_FIXED_SPECS.items():  # noqa: SLF001
            value_shape = tuple(int(getattr(shape, dimension)) for dimension in dimensions)
            values = np.zeros((1, *value_shape), dtype=dtype)
            if name in {"branch_valid", "risk_valid"}:
                values[...] = True
            handle.create_dataset(name, data=values)
        variable_dtype = h5py.vlen_dtype(np.dtype(np.float64))
        physics = handle.create_dataset("physics_state", (1,), dtype=variable_dtype)
        physics[0] = np.arange(3, dtype=np.float64)

    arrays = dataset.load_counterfactual_arrays((target,))

    assert arrays["schema_version"].tolist() == [1]
    assert arrays["trial_count"].shape == (1, 10)
    assert np.all(arrays["trial_count"] == 1)
    assert np.all(np.isnan(arrays["elapsed_mean"]))
