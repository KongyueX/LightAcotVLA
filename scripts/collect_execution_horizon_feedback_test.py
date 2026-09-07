# ruff: noqa: SLF001
from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
collector = importlib.import_module("collect_execution_horizon_feedback")


def _args(**overrides):
    args = collector.build_parser().parse_args([
        "--initial-state-bank", "/unused/bank", "--episodes", "300", "--output-dir", "/unused/output",
    ])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _response(value=0.0, horizon=5):
    probability = np.full(5, 0.025, dtype=np.float32)
    probability[collector.CANDIDATE_HORIZONS.index(horizon)] = 0.9
    return {
        "actions": np.full((25, 7), value + 1, dtype=np.float32),
        "execution_horizon_temporal_feature": np.full(256, value, dtype=np.float32),
        "execution_horizon_prefix_feature": np.full(2048, value, dtype=np.float32),
        "execution_horizon_state_normalized": np.full(32, value, dtype=np.float32),
        "execution_horizon_ordered_continuation_logits": np.zeros(4, dtype=np.float32),
        "execution_horizon_ordered_horizon_probability": probability,
        "execution_horizon_ordered_selected_h": np.asarray(horizon),
        "execution_horizon_candidate_horizons": np.asarray(collector.CANDIDATE_HORIZONS),
        "collector_wall_ms": 100.0,
    }


class _Env:
    def __init__(self, *, stop=15, succeeds=True):
        self.step_count = 0
        self.stop = stop
        self.succeeds = succeeds

    def step(self, action):
        self.step_count += 1
        return {"step": self.step_count}, 0.0, self.succeeds and self.step_count >= self.stop, {}


def _fake_base():
    def restore(env, snapshot):
        env.step_count = int(snapshot.physics_state[0])
        return {"step": env.step_count}

    return SimpleNamespace(
        libero_eval=SimpleNamespace(
            _observation_to_policy_input=lambda observation, description, size: dict(observation),
            _env_success=lambda env: env.succeeds and env.step_count >= env.stop,
            _is_terminated_episode_error=lambda exc: False,
        ),
        ordered=SimpleNamespace(selected_horizon=lambda result, model_action_horizon: int(
            result["execution_horizon_ordered_selected_h"]
        )),
        _capture_snapshot=lambda env: SimpleNamespace(physics_state=np.asarray([env.step_count], dtype=np.float64)),
        _restore_snapshot=restore,
        _root_seed=lambda seed, task, episode, step, task_stride: seed + task * task_stride + episode * 10_000 + step,
        _branch_seed=lambda seed, repeat, stride: seed + repeat * stride,
        _branch_schedule_seed=lambda seed, schedule_offset: seed + schedule_offset,
        _branch_continuation_seed=lambda seed, index, continuation_offset: seed + continuation_offset + index,
    )


def _root(horizon=5):
    return collector.FeedbackRoot(
        snapshot=SimpleNamespace(physics_state=np.asarray([10.0])), result=_response(10, horizon),
        primary_actions=np.full((25, 7), 11.0, dtype=np.float32),
        previous_prefix_feature=np.full(2048, 5.0, dtype=np.float32),
        previous_state=np.full(32, 5.0, dtype=np.float32), previous_h=5, elapsed_steps=5,
        history_valid=True, step=10, seed=3_067_017, decision_index=2,
    )


def test_request_uses_profile_and_explicit_final_nfe_without_teacher_or_prefix_export(monkeypatch):
    base = _fake_base()
    seen = {}

    def request(client, observation, **kwargs):
        seen.update(observation=observation, **kwargs)
        return _response()

    base._policy_request = request
    monkeypatch.setattr(collector, "_collector", lambda: base)
    args = _args()
    previous = np.ones((25, 7), dtype=np.float32)
    collector._request(None, {"step": 17}, "test", args=args, seed=99, step=17, step_limit=100,
                       previous_actions=previous, previous_h=7)
    assert seen["profile"] is True
    assert seen["teacher"] is False
    assert seen["run_student"] is True
    assert int(seen["observation"]["action_cot_final_denoising_steps"]) == 10
    assert int(seen["observation"]["action_cot_absolute_decision_step"]) == 17
    assert seen["previous_actions"] is previous
    assert seen["previous_h"] == 7
    assert seen["budget_balance"] == 0.5
    assert seen["episode_progress"] == 0.17
    assert args.prefix_token_count == 0


def test_reservoir_history_is_previous_real_call_and_is_outcome_blind(monkeypatch):
    monkeypatch.setattr(collector, "_collector", _fake_base)
    monkeypatch.setattr(collector.np.random, "default_rng", lambda seed: SimpleNamespace(integers=lambda size: 0))
    calls = []

    def request(client, observation, description, **kwargs):
        calls.append((kwargs["step"], kwargs["previous_h"], kwargs["previous_actions"]))
        return _response(kwargs["step"], 5)

    monkeypatch.setattr(collector, "_request", request)
    roots = []
    for succeeds in (True, False):
        root, success, count = collector.collect_source_root(
            _Env(succeeds=succeeds), {"step": 0}, "task", args=_args(), client=None,
            task_id=3, episode_id=300, step=0, step_limit=15,
        )
        assert count == 3
        assert success is succeeds
        assert root.step == 10
        assert root.history_valid
        assert root.elapsed_steps == 5
        assert root.previous_h == 5
        np.testing.assert_array_equal(root.previous_prefix_feature, np.full(2048, 5))
        np.testing.assert_array_equal(root.previous_state, np.full(32, 5))
        np.testing.assert_array_equal(root.primary_actions, np.full((25, 7), 11))
        roots.append(root)
    assert roots[0].seed == roots[1].seed
    np.testing.assert_array_equal(roots[0].previous_state, roots[1].previous_state)
    assert calls[0][:2] == (0, 10)
    assert calls[0][2] is None


def test_first_root_has_no_fabricated_history(monkeypatch):
    monkeypatch.setattr(collector, "_collector", _fake_base)
    monkeypatch.setattr(collector, "_request", lambda *a, **kw: _response(kw["step"], 5))
    root, success, count = collector.collect_source_root(
        _Env(stop=3), {"step": 0}, "task", args=_args(), client=None,
        task_id=0, episode_id=300, step=0, step_limit=30,
    )
    assert success
    assert count == 1
    assert root.step == 0
    assert not root.history_valid
    assert root.elapsed_steps == 0
    assert not np.any(root.previous_prefix_feature)
    assert not np.any(root.previous_state)


def test_each_branch_restores_root_and_updates_only_local_previous_actions(monkeypatch):
    base = _fake_base()
    monkeypatch.setattr(collector, "_collector", lambda: base)
    requests = []

    def request(client, observation, description, **kwargs):
        requests.append({**kwargs, "previous_actions": kwargs["previous_actions"].copy()})
        return _response(kwargs["step"], 5)

    monkeypatch.setattr(collector, "_request", request)
    root = _root()
    original_previous_state = root.previous_state.copy()
    env = _Env(stop=21)
    out1 = collector.run_branch(env, root, "task", args=_args(), client=None,
                                forced_horizon=5, repeat_seed=100, step_limit=100)
    out2 = collector.run_branch(env, root, "task", args=_args(), client=None,
                                forced_horizon=10, repeat_seed=100, step_limit=100)
    assert out1["steps"] == out2["steps"] == 11
    assert out1["calls"] == 3
    assert out2["calls"] == 2
    assert out1["rpc"] == pytest.approx(0.3)
    assert out2["rpc"] == pytest.approx(0.2)
    assert requests[0]["seed"] == requests[2]["seed"] == 10_000_100
    assert requests[0]["previous_h"] == 5
    assert requests[2]["previous_h"] == 10
    np.testing.assert_array_equal(requests[0]["previous_actions"], root.primary_actions)
    np.testing.assert_array_equal(requests[2]["previous_actions"], root.primary_actions)
    np.testing.assert_array_equal(root.previous_state, original_previous_state)
    np.testing.assert_array_equal(root.snapshot.physics_state, [10])


def test_branches_pair_seeds_and_keep_source_selected_h_and_history(monkeypatch):
    monkeypatch.setattr(collector, "_collector", _fake_base)
    calls = []

    def branch(env, root, description, **kwargs):
        calls.append((kwargs["repeat_seed"], kwargs["forced_horizon"]))
        return {"success": kwargs["forced_horizon"] >= 15, "elapsed": 2.0, "rpc": 0.4, "calls": 4, "steps": 20}

    monkeypatch.setattr(collector, "run_branch", branch)
    root = _root(horizon=20)
    record = collector.collect_branches(
        None, root, "task", args=_args(), client=None, task_id=3, episode_id=300,
        step_limit=100, source_success=False, source_calls=40,
    )
    assert len(calls) == 25
    for repeat in range(5):
        chunk = calls[repeat * 5 : repeat * 5 + 5]
        assert len({seed for seed, _ in chunk}) == 1
        assert sorted(h for _, h in chunk) == list(collector.CANDIDATE_HORIZONS)
    assert int(record["selected_h"]) == 20
    assert record["trial_success"].shape == (5, 5)
    assert int(record["elapsed_steps"]) == 5
    assert bool(record["history_valid"])
    assert not bool(record["source_success"])
    assert np.all(record["trial_valid"])
    assert record["temporal_feature"].shape == (256,)
    assert record["prefix_feature"].shape == (2048,)
    np.testing.assert_array_equal(record["previous_state"], root.previous_state)
    assert not np.shares_memory(record["previous_state"], root.previous_state)


def _diagnostic_record():
    success = np.zeros((5, 5), dtype=bool)
    success[1] = [True, False, True, False, True]
    success[2] = True
    return {
        "trial_success": success, "trial_rpc": np.ones((5, 5)), "trial_valid": np.ones((5, 5), dtype=bool),
        "selected_h": np.asarray(10), "task_id": np.asarray(3), "episode_id": np.asarray(300),
        "root_step": np.asarray(100), "history_valid": np.asarray(1, dtype=np.bool_), "source_success": np.asarray(0, dtype=np.bool_),
    }


def test_diagnostic_compares_against_actual_a_h_and_holds_out_seed():
    row = collector.root_diagnostics(_diagnostic_record())
    assert row["a_success_count"] == 3
    assert row["by_h"]["15"]["rescues_vs_a"] == 2
    assert row["by_h"]["15"]["regressions_vs_a"] == 0
    assert row["by_h"]["5"]["rescues_vs_a"] == 0
    assert row["by_h"]["5"]["regressions_vs_a"] == 3
    assert row["loo_diagnostic_success_count"] == 5
    assert row["loo_diagnostic_selected_h"] == [15] * 5
    assert row["any_paired_success_difference"]
    assert not row["all_h_failed_all_repeats"]
    record = _diagnostic_record()
    record["trial_success"][:] = False
    record["trial_success"][0] = [True, False, False, False, False]
    row = collector.root_diagnostics(record)
    assert row["empirical_best_success_count"] == 1
    assert row["loo_diagnostic_success_count"] == 0


def test_resume_summary_uses_only_closed_npz_and_rebuilds_rows(tmp_path):
    path = tmp_path / "task03_ep000300.npz"
    collector._save_record(path, _diagnostic_record())
    (tmp_path / "task03_ep000301.npz.tmp").write_bytes(b"incomplete")
    report = collector._refresh_summary(tmp_path, expected=2, metadata={})
    assert report["status"] == "collecting"
    assert report["num_roots"] == 1
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1
    collector._refresh_summary(tmp_path, expected=2, metadata={})
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1
    record = _diagnostic_record()
    record["episode_id"] = np.asarray(301)
    collector._save_record(tmp_path / "task03_ep000301.npz", record)
    report = collector._refresh_summary(tmp_path, expected=2, metadata={})
    assert report["status"] == "complete"
    assert report["num_roots"] == 2
    assert report["paired_vs_a"]["15"]["rescues_vs_a"] == 4


def test_incomplete_trial_matrix_is_not_a_closed_training_record():
    record = _diagnostic_record()
    record["trial_valid"][2, 3] = False
    with pytest.raises(ValueError, match="complete paired"):
        collector.root_diagnostics(record)
