# ruff: noqa: SLF001
from __future__ import annotations

import importlib
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
collector = importlib.import_module("collect_execution_horizon_architecture_features")


def _response(step):
    return {
        "actions": np.full((25, 7), step / 100, dtype=np.float32),
        "execution_horizon_prefix_feature": np.full(2048, step, dtype=np.float32),
        "execution_horizon_state_normalized": np.full(32, step / 10, dtype=np.float32),
        "execution_horizon_temporal_feature": np.full(256, step / 20, dtype=np.float32),
        "execution_horizon_ordered_continuation_logits": np.zeros(4, dtype=np.float32),
        "execution_horizon_ordered_horizon_probability": np.asarray(
            [0.5, 0.25, 0.125, 0.0625, 0.0625], dtype=np.float32,
        ),
        "execution_horizon_ordered_selected_h": np.asarray(5),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
        "collector_wall_ms": np.asarray(99_000.0),
    }


def _case(target_index=1):
    step = 10 + 5 * target_index
    seed = 67007 + 2 * 250_000_000 + 300 * 10_000 + step
    result = _response(step)
    valid = target_index > 0
    previous = _response(step - 5) if valid else None
    previous_prefix = previous["execution_horizon_prefix_feature"] if valid else np.zeros(2048, dtype=np.float32)
    previous_state = previous["execution_horizon_state_normalized"] if valid else np.zeros(32, dtype=np.float32)
    record = {
        "task_id": np.asarray(2), "episode_id": np.asarray(300),
        "root_step": np.asarray(step), "root_seed": np.asarray(seed, dtype=np.uint32),
        "source_decision_index": np.asarray(target_index), "source_calls": np.asarray(20),
        "source_success": np.asarray(1, dtype=bool), "physics_state": np.asarray([step], dtype=np.float64),
        "primary_actions": result["actions"].copy(),
        "prefix_feature": result["execution_horizon_prefix_feature"].copy(),
        "state": result["execution_horizon_state_normalized"].copy(),
        "temporal_feature": result["execution_horizon_temporal_feature"].copy(),
        "continuation_logits": result["execution_horizon_ordered_continuation_logits"].copy(),
        "ordered_probability": result["execution_horizon_ordered_horizon_probability"].copy(),
        "selected_h": np.asarray(5),
        "previous_prefix_feature": previous_prefix.copy(), "previous_state": previous_state.copy(),
        "previous_h": np.asarray(5 if valid else 10), "elapsed_steps": np.asarray(5 if valid else 0),
        "history_valid": np.asarray(valid), "candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
        "trial_success": (np.arange(25).reshape(5, 5) % 3 == 0),
        "trial_rpc": np.arange(25, dtype=np.float64).reshape(5, 5) / 10 + 0.2,
        "trial_elapsed": np.arange(25, dtype=np.float64).reshape(5, 5) + 2.0,
        "trial_calls": np.full((5, 5), 3, dtype=np.int32),
        "trial_steps": np.full((5, 5), 15, dtype=np.int32),
        "trial_valid": np.ones((5, 5), dtype=bool),
        "branch_seeds": np.asarray([seed + repeat * 20_000_000 for repeat in range(5)], dtype=np.uint32),
        "root_rpc_seconds": np.asarray(0.125),
    }
    arguments = {
        "step": step, "seed": seed, "previous_h": 5 if valid else 10,
        "elapsed_steps": 5 if valid else 0, "history_valid": valid,
        "previous_prefix": previous_prefix.copy(), "previous_state": previous_state.copy(),
    }
    return record, result, arguments


@pytest.fixture
def selected_h(monkeypatch):
    monkeypatch.setattr(
        collector.feedback, "_selected_h", lambda result: int(result["execution_horizon_ordered_selected_h"]),
    )


def test_matching_replay_passes_without_changing_labeled_arrays(selected_h):
    record, result, arguments = _case()
    original = {name: value.copy() for name, value in record.items()}
    differences = collector.check_replay(record, result, record["physics_state"].copy(), **arguments)
    assert differences
    assert max(differences.values()) == 0
    for name, value in original.items():
        np.testing.assert_array_equal(record[name], value)


@pytest.mark.parametrize("field", ["physics", "actions"])
def test_diverged_replay_cannot_reuse_labels(selected_h, field):
    record, result, arguments = _case()
    physics = record["physics_state"].copy()
    changed = physics if field == "physics" else result["actions"]
    changed.flat[0] += 0.01
    with pytest.raises(ValueError, match=f"Replayed {field} differs.*do not reuse labels"):
        collector.check_replay(record, result, physics, **arguments)


@pytest.mark.parametrize(("field", "value"), [
    ("step", 16), ("seed", 67007), ("previous_h", 10),
    ("elapsed_steps", 10), ("history_valid", False),
])
def test_wrong_decision_or_history_cannot_reuse_labels(selected_h, field, value):
    record, result, arguments = _case()
    arguments[field] = value
    with pytest.raises(ValueError, match="Replayed root"):
        collector.check_replay(record, result, record["physics_state"], **arguments)


def test_changed_greedy_h_cannot_reuse_labels(selected_h):
    record, result, arguments = _case()
    result["execution_horizon_ordered_selected_h"] = np.asarray(10)
    with pytest.raises(ValueError, match="A selected a different H"):
        collector.check_replay(record, result, record["physics_state"], **arguments)


class _Env:
    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self):
        self.step_count = 0

    def set_init_state(self, state):
        return {"step": self.step_count}

    def step(self, action):
        self.step_count += 1
        return {"step": self.step_count}, 0.0, False, {}


class _Client:
    def __init__(self, *, bad_pool=False):
        self.requests = []
        self.bad_pool = bad_pool

    def infer(self, observation, **kwargs):
        self.requests.append((dict(observation), kwargs))
        step = observation["step"]
        result = _response(step)
        if observation.get("execution_horizon_export_architecture_cache", False):
            tokens = np.stack([
                np.full(2048, step - 1, dtype=np.float32),
                np.full(2048, step + 1, dtype=np.float32),
                np.full(2048, 1000, dtype=np.float32),
            ])
            if self.bad_pool:
                tokens[0, 0] += 1
            previous = np.zeros((25, 32), dtype=np.float32)
            if kwargs["previous_actions"] is not None:
                previous[:, :7] = kwargs["previous_actions"]
            result.update({
                "execution_horizon_coarse_actions_normalized": np.full((25, 32), 0.1, dtype=np.float32),
                "execution_horizon_final_actions_normalized": np.full((25, 32), 0.2, dtype=np.float32),
                "execution_horizon_previous_actions_normalized": previous,
                "execution_horizon_previous_h": np.asarray(kwargs["previous_h"]),
                "execution_horizon_previous_valid": np.asarray(kwargs["previous_actions"] is not None),
                "execution_horizon_budget_balance": np.asarray(kwargs["budget_balance"]),
                "execution_horizon_episode_progress": np.asarray(kwargs["episode_progress"]),
                "execution_horizon_prefix_tokens": tokens,
                "execution_horizon_prefix_mask": np.asarray([True, True, False]),
                "execution_horizon_expert_hidden": np.full((25, 1024), 0.3, dtype=np.float32),
            })
        return result


def _setup_replay(monkeypatch, tmp_path, *, target_index=1, bad_pool=False):
    record, _, _ = _case(target_index)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    path = source / "task02_ep000300.npz"
    np.savez_compressed(path, **record)
    args = collector.feedback.build_parser().parse_args([
        "--initial-state-bank", "/unused/bank", "--episodes", "300", "--output-dir", str(source),
    ])
    (source / "run_config.json").write_text(json.dumps(vars(args)))
    env = _Env()
    client = _Client(bad_pool=bad_pool)
    bank = SimpleNamespace(validate_presets=lambda *args: None, state=lambda *args: np.zeros(1))
    libero = SimpleNamespace(
        LIBERO_ENV_RESOLUTION=256, LIBERO_DUMMY_ACTION=np.zeros(7),
        _get_libero_env=lambda *args: (env, "task"), _max_steps=lambda name: 100,
        _env_horizon=lambda env: None, _env_success=lambda env: False,
        _is_terminated_episode_error=lambda exc: False,
        _observation_to_policy_input=lambda observation, description, size: dict(observation),
        _safe_close_env=lambda env: setattr(env, "closed", True),
    )
    base = SimpleNamespace(
        libero_eval=libero,
        horizon_initial_states=SimpleNamespace(InitialStateBank=lambda path: bank),
        _root_seed=lambda seed, task, episode, step, task_stride: seed + task * task_stride + episode * 10_000 + step,
        _capture_snapshot=lambda env: SimpleNamespace(physics_state=np.asarray([env.step_count], dtype=np.float64)),
        _policy_request=lambda client, observation, **kwargs: client.infer(observation, **kwargs),
    )
    monkeypatch.setattr(collector.feedback, "_collector", lambda: base)

    def forbidden(*args, **kwargs):
        pytest.fail("Feature replay must not sample roots or run counterfactual branches.")

    for name in ("collect_source_root", "collect_branches", "run_branch"):
        monkeypatch.setattr(collector.feedback, name, forbidden)
    suite = SimpleNamespace(get_task=lambda task: task, get_task_init_states=lambda task: np.zeros((1, 1)))
    return record, path, output, env, client, suite


@pytest.mark.parametrize("target_index", [0, 1])
def test_replay_exports_only_target_and_preserves_original_trials(selected_h, monkeypatch, tmp_path, target_index):
    record, path, output, env, client, suite = _setup_replay(monkeypatch, tmp_path, target_index=target_index)
    audit = collector.replay_root(path, client=client, suite=suite, output=output)
    assert len(client.requests) == target_index + 1
    assert [bool(request.get("execution_horizon_export_architecture_cache", False))
            for request, _ in client.requests] == [False] * target_index + [True]
    assert client.requests[0][1]["previous_actions"] is None
    assert client.requests[0][1]["previous_h"] == 10
    if target_index:
        np.testing.assert_array_equal(client.requests[-1][1]["previous_actions"], _response(10)["actions"])
    assert all(kwargs["profile"] and not kwargs["teacher"] for _, kwargs in client.requests)
    assert env.step_count == int(record["root_step"])
    assert env.closed
    assert audit["labels_reused"]
    assert audit["replayed_policy_calls"] == target_index + 1
    with np.load(output / path.name, allow_pickle=False) as saved:
        for name, value in record.items():
            np.testing.assert_array_equal(saved[name], value)
        assert saved["input_expert_hidden"].shape == (25, 1024)
        assert saved["input_prefix_tokens"].shape == (3, 2048)
        assert saved["root_rpc_seconds"].item() == 0.125
        assert bool(saved["input_previous_valid"]) is (target_index > 0)


def test_wrong_full_prefix_pool_prevents_cache_write(selected_h, monkeypatch, tmp_path):
    _, path, output, env, client, suite = _setup_replay(monkeypatch, tmp_path, bad_pool=True)
    with pytest.raises(ValueError, match="Full prefix cache does not reproduce"):
        collector.replay_root(path, client=client, suite=suite, output=output)
    assert env.closed
    assert not (output / path.name).exists()
