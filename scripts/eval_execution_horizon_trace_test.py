"""Diagnostic tracing preserves the evaluator's actions and policy requests."""
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import eval_libero_execution_horizon as evaluator
import numpy as np
import pytest


class _Environment:
    def __init__(self) -> None:
        self.steps = 0
        self.actions = []
        self.closed = False
        self.control_freq = 20
        self.sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: np.asarray([self.steps, 3.0])))

    def observation(self):
        return {
            "robot0_eef_pos": np.asarray([self.steps, 0.0, 0.0]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.03, -0.03]),
            "object-state": np.asarray([2.0, self.steps]),
        }

    def reset(self):
        self.steps = 0

    def set_init_state(self, state):
        del state
        return self.observation()

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        return self.observation(), 0.0, self.steps >= 4, {}


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.requests = []
        self.fail = fail

    def infer(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.fail:
            raise RuntimeError("policy unavailable")
        return {
            "actions": np.arange(25 * 7, dtype=np.float32).reshape(25, 7) / 100,
            "execution_horizon_ordered_continuation_logits": np.zeros(4),
            "execution_horizon_ordered_horizon_probability": np.asarray([0.5, 0.25, 0.125, 0.0625, 0.0625]),
            "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
            "execution_horizon_ordered_selected_h": np.asarray(5),
        }


@pytest.fixture
def episode_setup(monkeypatch):
    environments = []

    def create_env(*args):
        del args
        env = _Environment()
        environments.append(env)
        return env, "task"

    monkeypatch.setattr(evaluator.libero_eval, "_get_libero_env", create_env)
    monkeypatch.setattr(evaluator.libero_eval, "_safe_close_env", lambda env: setattr(env, "closed", True))
    monkeypatch.setattr(evaluator.libero_eval, "_max_steps", lambda unused: 100)
    monkeypatch.setattr(evaluator.libero_eval, "_env_horizon", lambda unused: 100)
    monkeypatch.setattr(evaluator.libero_eval, "_env_success", lambda env: env.steps >= 4)
    monkeypatch.setattr(evaluator.libero_eval, "_observation_to_policy_input", lambda obs, *args: {
        "observation/state": np.concatenate([obs["robot0_eef_pos"], np.zeros(3), obs["robot0_gripper_qpos"]]),
    })
    suite = SimpleNamespace(
        get_task=lambda task: SimpleNamespace(name="task"),
        get_task_init_states=lambda task: np.zeros((1, 2)),
    )
    return suite, environments


def test_trace_records_each_successful_step_without_adding_or_changing_policy_calls(tmp_path, episode_setup):
    suite, environments = episode_setup
    args = evaluator.build_parser().parse_args([
        "--output-dir", str(tmp_path / "eval"), "--modes", "ordered_transformer",
        "--model-action-horizon", "25", "--num-steps-wait", "1",
    ])
    baseline_client = _Client()
    baseline_row, baseline_decisions = evaluator._run_episode(
        mode="ordered_transformer", task_id=0, episode=336, task_suite=suite, client=baseline_client, args=args,
    )
    args.trace_output_dir = str(tmp_path / "traces")
    traced_client = _Client()
    traced_row, traced_decisions = evaluator._run_episode(
        mode="ordered_transformer", task_id=0, episode=336, task_suite=suite, client=traced_client, args=args,
    )
    assert len(baseline_client.requests) == len(traced_client.requests) == 1
    assert baseline_client.requests[0].keys() == traced_client.requests[0].keys()
    for name, value in baseline_client.requests[0].items():
        np.testing.assert_array_equal(traced_client.requests[0][name], value)
    np.testing.assert_array_equal(environments[0].actions, environments[1].actions)
    assert baseline_row["success"] == traced_row["success"] == 1
    assert baseline_row["steps"] == traced_row["steps"] == 4
    assert baseline_decisions[0]["selected_horizon"] == traced_decisions[0]["selected_horizon"] == 5
    assert "diagnostic_only" not in baseline_row
    assert traced_row["diagnostic_only"] is True
    assert all(env.closed for env in environments)
    path = tmp_path / "traces/ordered_transformer/task00_ep000336.npz"
    with np.load(path, allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["decision_steps"], [1])
        np.testing.assert_array_equal(trace["execution_h"], [3])
        np.testing.assert_array_equal(trace["state_steps"], [1, 2, 3, 4])
        np.testing.assert_array_equal(trace["state_decision_index"], [-1, 0, 0, 0])
        np.testing.assert_array_equal(trace["decision_physics_state"], [[1, 3]])
        assert trace["decision_proprio"].shape == (1, 8)
        assert trace["request_seeds"][0] == args.seed + 336 * 10_000 + 1


def test_trace_closes_and_environment_releases_when_policy_raises(tmp_path, episode_setup):
    suite, environments = episode_setup
    args = evaluator.build_parser().parse_args([
        "--output-dir", str(tmp_path / "eval"), "--trace-output-dir", str(tmp_path / "traces"),
        "--modes", "ordered_transformer", "--model-action-horizon", "25", "--num-steps-wait", "1",
    ])
    client = _Client(fail=True)
    with pytest.raises(RuntimeError, match="policy unavailable"):
        evaluator._run_episode(
            mode="ordered_transformer", task_id=0, episode=336, task_suite=suite, client=client, args=args,
        )
    assert environments[0].closed
    assert len(client.requests) == 1
    metadata = json.loads((tmp_path / "traces/ordered_transformer/task00_ep000336.json").read_text())
    assert metadata["success"] is False
    assert metadata["error"] == "RuntimeError: policy unavailable"
    assert metadata["recorded_environment_steps"] == 1


def test_trace_options_only_enter_resume_signature_when_enabled(tmp_path):
    args = evaluator.build_parser().parse_args(["--output-dir", str(tmp_path)])
    original = evaluator._run_signature(args)
    assert not any(name.startswith("trace_") for name in original)
    args.trace_video_stride = 7
    assert evaluator._run_signature(args) == original
    args.trace_output_dir = str(tmp_path / "traces")
    enabled = evaluator._run_signature(args)
    assert enabled["trace_output_dir"] == str(tmp_path / "traces")
    assert enabled["trace_video_stride"] == 7
    evaluator._prepare_journal(tmp_path, args)
    args.resume = True
    args.trace_video_stride = 5
    with pytest.raises(ValueError, match="Resume configuration differs"):
        evaluator._prepare_journal(tmp_path, args)
