from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_root_seed_task_stride_is_opt_in_and_legacy_default_is_preserved() -> None:
    args = collector.build_parser().parse_args(["--output-dir", "/tmp/counterfactuals"])
    assert args.root_seed_task_stride == 1_000_000
    legacy_metadata = collector._seed_scheme_metadata(args)  # noqa: SLF001
    assert legacy_metadata["root_seed_scheme"] == "affine_task_episode_step_uint32_v1"
    assert legacy_metadata["root_seed_branch_continuation_offset"] == 100_000
    assert legacy_metadata["root_seed_strict_namespace_validation"] is False

    legacy_left = collector._root_seed(  # noqa: SLF001
        7,
        0,
        200,
        10,
        task_stride=1_000_000,
    )
    legacy_right = collector._root_seed(  # noqa: SLF001
        7,
        1,
        100,
        10,
        task_stride=1_000_000,
    )
    assert legacy_left == legacy_right

    widened_left = collector._root_seed(  # noqa: SLF001
        7,
        0,
        200,
        10,
        task_stride=250_000_000,
    )
    widened_right = collector._root_seed(  # noqa: SLF001
        7,
        1,
        100,
        10,
        task_stride=250_000_000,
    )
    assert widened_left != widened_right


def test_wide_root_seed_namespace_covers_high_episodes_branches_and_policy_seeds() -> None:
    collector._validate_seed_namespace(  # noqa: SLF001
        base_seed=7,
        task_ids=list(range(10)),
        episode_ids=list(range(100, 500)),
        maximum_episode_step=1_570,
        maximum_continuation_calls=1_570,
        task_stride=250_000_000,
        branch_repeats=10,
        branch_repeat_seed_stride=20_000_000,
        teacher_samples=20,
    )

    root_seed = collector._root_seed(  # noqa: SLF001
        7,
        9,
        499,
        1_570,
        task_stride=250_000_000,
    )
    branch_seed = collector._branch_seed(root_seed, 9, 20_000_000)  # noqa: SLF001
    _, schedule_offset, continuation_offset, strict = collector._seed_scheme(  # noqa: SLF001
        250_000_000,
        20_000_000,
    )
    assert strict
    root_base = collector._root_seed(7, 9, 499, 0, task_stride=250_000_000)  # noqa: SLF001
    assert root_seed + 20 < root_base + schedule_offset
    assert root_seed + schedule_offset < root_base + continuation_offset
    assert root_seed + continuation_offset + 1_570 < root_base + 20_000_000
    continuation_seed = collector._branch_continuation_seed(  # noqa: SLF001
        branch_seed,
        1_570,
        continuation_offset=continuation_offset,
    )
    assert continuation_seed <= np.iinfo(np.uint32).max


def test_seed_namespace_rejects_cross_episode_collision_and_uint32_overflow() -> None:
    with pytest.raises(ValueError, match="overlap across task/episode identities"):
        collector._validate_seed_namespace(  # noqa: SLF001
            base_seed=7,
            task_ids=[0],
            episode_ids=[100, 1_100],
            maximum_episode_step=1_570,
            maximum_continuation_calls=1_570,
            task_stride=250_000_000,
            branch_repeats=3,
            branch_repeat_seed_stride=20_000_000,
            teacher_samples=20,
        )

    with pytest.raises(ValueError, match="escapes task namespace"):
        collector._validate_seed_namespace(  # noqa: SLF001
            base_seed=7,
            task_ids=[0],
            episode_ids=[100],
            maximum_episode_step=1_570,
            maximum_continuation_calls=1_570,
            task_stride=30_000_000,
            branch_repeats=3,
            branch_repeat_seed_stride=20_000_000,
            teacher_samples=20,
        )

    with pytest.raises(ValueError, match="uint32 policy-seed range"):
        collector._validate_seed_namespace(  # noqa: SLF001
            base_seed=7,
            task_ids=[9],
            episode_ids=[100],
            maximum_episode_step=1_570,
            maximum_continuation_calls=1_570,
            task_stride=500_000_000,
            branch_repeats=1,
            branch_repeat_seed_stride=20_000_000,
            teacher_samples=20,
        )


def test_seed_scheme_metadata_records_formula_parameters() -> None:
    metadata = collector._seed_scheme_metadata(  # noqa: SLF001
        SimpleNamespace(root_seed_task_stride=250_000_000, branch_repeat_seed_stride=20_000_000)
    )
    assert metadata == {
        "root_seed_scheme": "affine_task_episode_repeat_schedule_continuation_lanes_uint32_v2",
        "root_seed_task_stride": 250_000_000,
        "root_seed_episode_stride": 10_000,
        "root_seed_branch_repeat_stride": 20_000_000,
        "root_seed_branch_schedule_offset": 5_000_000,
        "root_seed_branch_continuation_offset": 10_000_000,
        "root_seed_strict_namespace_validation": True,
        "root_seed_max_value": int(np.iinfo(np.uint32).max),
    }


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


def test_ordered_transformer_student_uses_selected_horizon_directly() -> None:
    args = SimpleNamespace(student_mode=collector.ORDERED_MODE, model_action_horizon=25)
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(15, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25], dtype=np.int32),
    }

    raw_horizon, selected_horizon = collector._student_horizon(  # noqa: SLF001
        result,
        args=args,
        budget_state=SimpleNamespace(),
    )

    assert (raw_horizon, selected_horizon) == (15, 15)


def test_ordered_transformer_student_rejects_horizon_beyond_model_chunk() -> None:
    args = SimpleNamespace(student_mode=collector.ORDERED_MODE, model_action_horizon=20)
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(25, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25], dtype=np.int32),
    }

    with pytest.raises(ValueError, match="model_action_horizon=20"):
        collector._student_horizon(  # noqa: SLF001
            result,
            args=args,
            budget_state=SimpleNamespace(),
        )


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
