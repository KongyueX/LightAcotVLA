from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

from openpi.execution_horizon import dataset

_SCRIPT = pathlib.Path(__file__).with_name("collect_execution_horizon_counterfactuals.py")
sys.path.insert(0, str(_SCRIPT.parent))
collector = importlib.import_module("collect_execution_horizon_counterfactuals")


def _outcome(*, success: bool, repeat: int) -> dict[str, object]:
    return {
        "success": success,
        "timeout": not success,
        "remaining_steps": 20 + repeat,
        "remaining_calls": 4 + repeat,
        "elapsed_seconds": 1.0 + 0.1 * repeat,
    }


class _RecordingClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def infer(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return {}


def test_prefix_tokens_are_exported_only_for_collected_teacher_root() -> None:
    client = _RecordingClient()
    args = SimpleNamespace(
        action_cot_denoising_steps=10,
        teacher_samples=20,
        model_action_horizon=20,
        prefix_token_count=1024,
    )

    collector._policy_request(  # noqa: SLF001
        client,
        {},
        seed=7,
        args=args,
        teacher=True,
    )
    collector._policy_request(  # noqa: SLF001
        client,
        {},
        seed=8,
        args=args,
        teacher=False,
    )

    assert "export_execution_horizon_prefix_tokens" in client.requests[0]
    assert "export_execution_horizon_prefix_tokens" not in client.requests[1]


def test_fixed_h5_continuation_is_explicit_and_legacy_h9_is_preserved() -> None:
    assert (
        collector._fixed_continuation_horizon(  # noqa: SLF001
            SimpleNamespace(continuation_policy="fixed_h", fixed_continuation_horizon=5)
        )
        == 5
    )
    assert (
        collector._fixed_continuation_horizon(  # noqa: SLF001
            SimpleNamespace(continuation_policy="fixed_h9")
        )
        == 9
    )


def test_fixed_h_source_rollout_uses_reference_and_legacy_h9_is_preserved() -> None:
    assert (
        collector._nonstudent_source_horizon(  # noqa: SLF001
            SimpleNamespace(continuation_policy="fixed_h", reference_horizon=10)
        )
        == 10
    )
    assert (
        collector._nonstudent_source_horizon(  # noqa: SLF001
            SimpleNamespace(continuation_policy="fixed_h9", reference_horizon=10)
        )
        == 9
    )


def test_student_source_can_be_decoupled_from_fixed_branch_continuation() -> None:
    args = SimpleNamespace(
        source_policy="current_student",
        continuation_policy="fixed_h",
        fixed_continuation_horizon=5,
        reference_horizon=10,
    )
    assert collector._source_uses_student(args)  # noqa: SLF001
    assert collector._fixed_continuation_horizon(args) == 5  # noqa: SLF001


def test_source_policy_defaults_preserve_legacy_behavior() -> None:
    assert collector._source_uses_student(SimpleNamespace(continuation_policy="current_student"))  # noqa: SLF001
    assert not collector._source_uses_student(SimpleNamespace(continuation_policy="fixed_h"))  # noqa: SLF001
    assert (
        collector._nonstudent_source_horizon(  # noqa: SLF001
            SimpleNamespace(
                source_policy="fixed_reference",
                continuation_policy="fixed_h9",
                reference_horizon=10,
            )
        )
        == 10
    )


def test_root_record_aggregates_counts_elapsed_and_paired_regressions() -> None:
    shape = dataset.DatasetShape(
        prefix_feature_dim=8,
        state_dim=4,
        action_dim=7,
        coarse_horizon=5,
        action_horizon=25,
        candidate_horizons=(5, 10, 15, 20, 25),
        max_trials=3,
        prefix_token_count=6,
    )
    success_patterns = (
        (True, True, True),
        (True, True, True),
        (True, False, True),
        (False, False, True),
        (False, False, True),
    )
    branches = [
        [_outcome(success=success, repeat=repeat) for repeat, success in enumerate(pattern)]
        for pattern in success_patterns
    ]
    result = {
        "execution_horizon_prefix_feature": np.zeros((shape.prefix_feature_dim,), dtype=np.float32),
        "execution_horizon_state_normalized": np.zeros((shape.state_dim,), dtype=np.float32),
        "execution_horizon_prefix_tokens": np.zeros((4, shape.prefix_feature_dim), dtype=np.float32),
        "execution_horizon_prefix_mask": np.asarray([True, True, True, False]),
        "mc_actions_normalized": np.zeros((3, shape.action_horizon, shape.action_dim), dtype=np.float32),
        "mc_coarse_actions_normalized": np.zeros((3, shape.coarse_horizon, shape.action_dim), dtype=np.float32),
    }
    risk = {
        "event_index": -1,
        "final_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "action_cot_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "fused_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "event_mask": np.zeros((shape.action_horizon,), dtype=np.bool_),
    }
    snapshot = collector.SimulatorSnapshot(
        physics_state=np.arange(4, dtype=np.float64),
        scalar_attributes=[],
        random_states=[],
    )

    record = collector._root_record(  # noqa: SLF001
        result=result,
        risk=risk,
        branches=branches,
        snapshot=snapshot,
        task_id=1,
        episode_id=2,
        decision_step=3,
        root_seed=4,
        previous_actions_normalized=np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        previous_h=10,
        previous_valid=True,
        budget_balance=0.5,
        episode_progress=0.25,
        source_iteration=0,
        v2_min_horizon=3,
        shape=shape,
        reference_horizon=10,
    )

    np.testing.assert_array_equal(record["trial_count"], np.full((5,), 3))
    np.testing.assert_array_equal(record["success_count"], np.asarray([3, 3, 2, 1, 1]))
    np.testing.assert_array_equal(record["dangerous_long_count"], np.asarray([0, 0, 1, 2, 2]))
    np.testing.assert_array_equal(record["paired_trial_count"], np.asarray([0, 0, 3, 3, 3]))
    assert record["hazard_event_count"][14] == 1
    assert record["hazard_event_count"][19] == 1
    assert record["hazard_event_count"].sum() == 2
    np.testing.assert_array_equal(record["hazard_at_risk_count"][:15], 3)
    np.testing.assert_array_equal(record["hazard_at_risk_count"][15:20], 2)
    np.testing.assert_array_equal(record["hazard_at_risk_count"][20:], 1)
    np.testing.assert_allclose(record["elapsed_mean"], 1.1)
    np.testing.assert_allclose(record["elapsed_variance"], 0.01, rtol=1e-5)
    assert record["raw_h"] == 10
    assert record["prefix_tokens"].shape == (6, shape.prefix_feature_dim)
    np.testing.assert_array_equal(record["prefix_token_mask"], [True, True, True, False, False, False])


def test_root_record_excludes_nonmonotone_short_hazard_trials() -> None:
    shape = dataset.DatasetShape(
        prefix_feature_dim=8,
        state_dim=4,
        action_dim=7,
        coarse_horizon=5,
        action_horizon=25,
        candidate_horizons=(5, 10, 15, 20, 25),
        max_trials=3,
    )
    # repeat0: T,T,F,F -> event before H15; repeat1: all safe ->
    # right-censored at H20; repeat2: T,F,T,F -> contradictory and excluded.
    success_patterns = (
        (True, True, True),
        (True, True, False),
        (False, True, True),
        (False, True, False),
        (False, True, False),
    )
    branches = [
        [_outcome(success=success, repeat=repeat) for repeat, success in enumerate(pattern)]
        for pattern in success_patterns
    ]
    result = {
        "execution_horizon_prefix_feature": np.zeros((shape.prefix_feature_dim,), dtype=np.float32),
        "execution_horizon_state_normalized": np.zeros((shape.state_dim,), dtype=np.float32),
        "mc_actions_normalized": np.zeros((3, shape.action_horizon, shape.action_dim), dtype=np.float32),
        "mc_coarse_actions_normalized": np.zeros((3, shape.coarse_horizon, shape.action_dim), dtype=np.float32),
    }
    risk = {
        "event_index": -1,
        "final_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "action_cot_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "fused_risk": np.zeros((shape.action_horizon,), dtype=np.float32),
        "event_mask": np.zeros((shape.action_horizon,), dtype=np.bool_),
    }
    record = collector._root_record(  # noqa: SLF001
        result=result,
        risk=risk,
        branches=branches,
        snapshot=collector.SimulatorSnapshot(
            physics_state=np.arange(4, dtype=np.float64),
            scalar_attributes=[],
            random_states=[],
        ),
        task_id=1,
        episode_id=2,
        decision_step=3,
        root_seed=4,
        previous_actions_normalized=np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        previous_h=10,
        previous_valid=True,
        budget_balance=0.5,
        episode_progress=0.25,
        source_iteration=0,
        v2_min_horizon=3,
        shape=shape,
        reference_horizon=10,
    )

    assert record["hazard_event_count"][14] == 1
    assert record["hazard_event_count"].sum() == 1
    np.testing.assert_array_equal(record["hazard_at_risk_count"][:15], 2)
    np.testing.assert_array_equal(record["hazard_at_risk_count"][15:], 1)
