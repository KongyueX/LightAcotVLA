# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("train_branched_refresh_router.py")
sys.path.insert(0, str(_SCRIPT.parent))
router = importlib.import_module("train_branched_refresh_router")


def test_action_risk_uses_cached_token_age_and_fresh_token_zero_6d() -> None:
    cached = np.zeros((2, 8, 7), dtype=np.float32)
    fresh = np.zeros((2, 10, 7), dtype=np.float32)
    cached[0, 4, :6] = 2.0
    cached[1, 4, :6] = np.arange(6, dtype=np.float32)
    cached[:, 4, 6] = 1000.0
    flat = {"cached_actions_env": cached, "fresh_actions_env": fresh}
    risk = router._action_risk(flat, age=4)
    np.testing.assert_allclose(risk, [4.0, np.mean(np.arange(6) ** 2)])


def test_auroc_and_selective_ranking_are_exact() -> None:
    risk = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    assert router._binary_auroc(risk >= 2.0, risk) == 1.0
    assert router._binary_auroc(risk >= 2.0, -risk) == 0.0
    result = router._selective_at_budget(risk, risk, budget=0.25, seed=7)
    assert result["refresh_count"] == 1
    assert result["selective_action_mse_6d"] == 0.75
    assert result["risk_removed"] == 0.75


def test_intended_prefix_is_identical_across_same_root_branches() -> None:
    branches = 2
    cached_actions = np.arange(8 * 7, dtype=np.float32).reshape(1, 8, 7)
    arrays = {
        "root_id": np.asarray([12]),
        "task_id": np.asarray([1]),
        "episode_id": np.asarray([2]),
        "branch_ids": np.asarray([[0, 1]]),
        "branch_steps": np.asarray([[4, 4]]),
        "anchor_images": np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
        "current_images": np.zeros((1, branches, 2, 8, 8, 3), dtype=np.uint8),
        "anchor_state": np.zeros((1, 3), dtype=np.float32),
        "current_state": np.zeros((1, branches, 3), dtype=np.float32),
        "cached_ear": np.zeros((1, 5, 7), dtype=np.float32),
        "cached_actions_env": cached_actions,
        "fresh_actions_env": np.zeros((1, branches, 8, 7), dtype=np.float32),
    }
    indices = router.progress_probe.BranchIndices(
        roots=np.asarray([0, 0]),
        branches=np.asarray([0, 1]),
    )
    flat = router._flat_arrays(arrays, indices, age=4, plan_horizon=8)
    np.testing.assert_array_equal(flat["intended_prefix"][0], cached_actions[0, :4])
    np.testing.assert_array_equal(flat["intended_prefix"][0], flat["intended_prefix"][1])
    np.testing.assert_array_equal(flat["cached_plan_tokens"][0, -8:, :7], cached_actions[0])
    assert np.all(flat["intended_valid"])


def test_root_nominal_excess_and_latency_gate_math() -> None:
    excess = router._root_nominal_excess(
        np.asarray([5, 5, 8, 8]),
        np.asarray([0, 1, 0, 1]),
        np.asarray([1.0, 3.0, 4.0, 2.0]),
    )
    np.testing.assert_allclose(excess, [0.0, 2.0, 0.0, -2.0])
    args = router.Args(dataset=("unused",), output_dir="unused")
    estimate = router._latency_estimate(args, 0.25)
    expected_average = (95.844 * 1.25 + 2.0) / 8
    assert np.isclose(
        estimate["conservative_fixed_calendar"]["average_latency_ms_per_action"],
        expected_average,
    )
    assert (
        estimate["conservative_fixed_calendar"]["estimated_speedup_vs_full_every_action"] > 3.0
    )
    renewal_average = (95.844 + 2.0) / (8 - 4 * 0.25)
    assert np.isclose(
        estimate["refresh_resets_plan_renewal"]["average_latency_ms_per_action"],
        renewal_average,
    )
