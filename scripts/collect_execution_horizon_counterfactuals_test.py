from __future__ import annotations

import importlib
import json
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


def test_source_prefix_export_does_not_request_mc_teacher() -> None:
    client = _RecordingClient()
    args = collector.build_parser().parse_args(["--output-dir", "/tmp/counterfactuals", "--prefix-token-count", "2"])
    assert args.root_sampling == "call_offset"
    collector._policy_request(  # noqa: SLF001
        client, {}, seed=7, args=args, export_prefix_tokens=True
    )
    assert bool(client.requests[0]["export_execution_horizon_prefix_tokens"])
    assert "batched_mc_samples" not in client.requests[0]


@pytest.mark.parametrize("source_steps", [5, 30])
def test_trajectory_reservoir_finishes_source_and_preserves_cached_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source_steps: int
) -> None:
    args = collector.build_parser().parse_args(
        [
            "--output-dir", str(tmp_path),
            "--root-sampling", "trajectory_reservoir",
            "--source-policy", "current_student",
            "--continuation-policy", "current_student",
            "--student-mode", "ordered_transformer",
            "--max-roots-per-episode", "1",
            "--candidate-horizons", "5", "10",
            "--student-candidates", "5", "10",
            "--num-steps-wait", "0",
            "--model-action-dim", "7",
            "--model-state-dim", "3",
            "--model-coarse-horizon", "3",
            "--prefix-feature-dim", "4",
            "--prefix-token-count", "2",
            "--branch-repeats", "3",
            "--debug-failure-videos", "0",
        ]
    )
    events = []
    records = []

    class Env:
        pos = 0

        def __deepcopy__(self, _memo):
            raise AssertionError("The simulator object must not be deep-copied.")

        def reset(self):
            self.pos = 0

        def set_init_state(self, _initial_state):
            return {"source_position": self.pos}

        def step(self, action):
            assert action[0] == self.pos // 5 * 5 + 1
            self.pos += 1
            events.append(("step", self.pos))
            return {"source_position": self.pos}, 0, self.pos == source_steps, {}

    env = Env()

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def infer(self, request):
            position = int(request["source_position"])
            teacher = "batched_mc_samples" in request
            events.append(("teacher" if teacher else "source", position))
            if teacher:
                assert env.pos == source_steps
            value = 99 if teacher else position + 1
            result = {
                "actions": np.full((10, 7), value, dtype=np.float32),
                "execution_horizon_final_actions_normalized": np.full((10, 7), value + 0.25),
                "execution_horizon_coarse_actions_normalized": np.full((3, 7), value + 0.5),
                "execution_horizon_prefix_feature": np.full((4,), value + 0.75),
                "execution_horizon_state_normalized": np.full((3,), value + 0.125),
                "execution_horizon_ordered_selected_h": np.asarray(5),
                "execution_horizon_candidate_horizons": np.asarray([5, 10]),
            }
            if bool(request.get("export_execution_horizon_prefix_tokens", False)):
                result["execution_horizon_prefix_tokens"] = np.full((2, 4), value + 0.875)
                result["execution_horizon_prefix_mask"] = np.ones((2,), dtype=np.bool_)
            if teacher:
                result["mc_actions_normalized"] = np.full((20, 10, 7), value)
                result["mc_coarse_actions_normalized"] = np.full((20, 3, 7), value)
            return result

    class Writer:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def append(self, record):
            records.append(record)

    def run_branch(_env, snapshot, primary_actions, **kwargs):
        assert env.pos == source_steps
        assert ("step", source_steps) in events
        events.append(("branch", kwargs["forced_horizon"]))
        root_step = int(snapshot.physics_state[0])
        assert kwargs["root_step"] == root_step
        np.testing.assert_array_equal(primary_actions, root_step + 1)
        return True, False, 5, 1, 0.1, []

    suite = SimpleNamespace(
        n_tasks=1,
        get_task=lambda _task_id: "task",
        get_task_init_states=lambda _task_id: [np.zeros((1,))],
    )
    monkeypatch.setattr(collector.libero_eval.benchmark, "get_benchmark_dict", lambda: {"libero_10": lambda: suite})
    monkeypatch.setattr(collector.libero_eval, "_get_libero_env", lambda *_args: (env, "task"))
    monkeypatch.setattr(collector.libero_eval, "_env_horizon", lambda _env: source_steps)
    monkeypatch.setattr(collector.libero_eval, "_max_steps", lambda _suite: source_steps)
    monkeypatch.setattr(collector.libero_eval, "_env_success", lambda _env: env.pos == source_steps)
    monkeypatch.setattr(collector.libero_eval, "_safe_close_env", lambda _env: None)
    monkeypatch.setattr(collector.libero_eval, "_observation_to_policy_input", lambda obs, *_args: dict(obs))
    monkeypatch.setattr(collector.websocket_policy, "WebsocketClientPolicy", Client)
    monkeypatch.setattr(collector.horizon_dataset, "ShardedCounterfactualWriter", Writer)
    monkeypatch.setattr(
        collector, "_capture_snapshot",
        lambda _env: collector.SimulatorSnapshot(np.asarray([env.pos]), [(env, "pos", env.pos)], []),
    )
    monkeypatch.setattr(collector, "_run_branch", run_branch)
    monkeypatch.setattr(
        collector.v2,
        "risk_targets_from_normalized_mc",
        lambda *_args, **_kwargs: {
            "event_index": -1,
            "final_risk": np.zeros((10,)),
            "action_cot_risk": np.zeros((10,)),
            "fused_risk": np.zeros((10,)),
            "event_mask": np.zeros((10,), dtype=np.bool_),
        },
    )
    collector.main(args)

    assert len(records) == 1
    record = records[0]
    decision_count = source_steps // 5
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0, 0, 0x524F4F54]))
    expected_index = 0
    for index in range(decision_count):
        if int(rng.integers(index + 1)) == 0:
            expected_index = index
    root_step = expected_index * 5
    assert record["decision_step"] == root_step
    assert 0 <= root_step < source_steps
    assert len([event for event in events if event[0] == "source"]) == decision_count
    assert len([event for event in events if event[0] == "teacher"]) == 1
    assert len([event for event in events if event[0] == "branch"]) == 6
    assert events.index(("step", source_steps)) < next(i for i, event in enumerate(events) if event[0] == "branch")
    np.testing.assert_array_equal(record["final_actions"], root_step + 1.25)
    np.testing.assert_array_equal(record["coarse_actions"], root_step + 1.5)
    np.testing.assert_array_equal(record["state"], root_step + 1.125)
    np.testing.assert_array_equal(record["prefix_tokens"], root_step + 1.875)
    np.testing.assert_array_equal(record["physics_state"], [root_step])
    assert record["previous_valid"] == (expected_index > 0)
    np.testing.assert_array_equal(record["previous_actions"], root_step - 3.75 if expected_index else 0)
    assert record["episode_progress"] == root_step / source_steps
    outcome = json.loads((tmp_path / "repeated_branch_outcomes.jsonl").read_text())
    assert outcome["source_decision_index"] == expected_index
    assert outcome["source_decision_count"] == decision_count
    assert outcome["root_sampling"] == "trajectory_reservoir"
    assert "source_decision_count" not in record


@pytest.mark.parametrize("source_policy,max_roots", [("fixed_reference", 1), ("current_student", 0)])
def test_trajectory_reservoir_requires_one_root_and_student_source(source_policy: str, max_roots: int) -> None:
    args = collector.build_parser().parse_args(
        [
            "--output-dir", "/tmp/counterfactuals",
            "--root-sampling", "trajectory_reservoir",
            "--source-policy", source_policy,
            "--max-roots-per-episode", str(max_roots),
        ]
    )
    with pytest.raises(ValueError, match="trajectory_reservoir requires"):
        collector.main(args)


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
