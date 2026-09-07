"""Protocol checks for the bounded first-H intervention."""
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import probe_execution_horizon_first_divergence as probe
import pytest


def _case() -> dict:
    prefix = {name: {"max": 0.0} for name in (
        "generated_chunk_max_abs_difference", "decision_physics_max_abs_difference",
        "decision_raw_proprio_max_abs_difference", "executed_command_max_abs_difference",
        "eef_position_difference_m", "eef_rotation_difference_rad", "gripper_qpos_max_abs_difference",
    )}
    prefix.update(shared_decisions=2, matching_request_seeds=2)
    return {
        "task_id": 3, "episode_id": 338, "old_relation": "regression",
        "reproduction": {"original_pair_reproduced": True,
                         "trace_success": {probe.BASE_MODE: True, probe.HISTORY_MODE: False}},
        "alignment": {
            "first_shared_decision_with_h_difference": {"step": 35, "a_selected_h": 5, "history_selected_h": 10},
            "common_prefix_through_step": 35, "before_first_h_or_grid_difference": prefix,
        },
    }


def _response(value: float = 1.0) -> dict:
    return {
        "actions": np.full((25, 7), value, dtype=np.float32),
        "execution_horizon_prefix_feature": np.full(2048, value, dtype=np.float32),
        "execution_horizon_state_normalized": np.full(32, value, dtype=np.float32),
        "execution_horizon_temporal_feature": np.zeros(256, dtype=np.float32),
        "execution_horizon_ordered_continuation_logits": np.zeros(4, dtype=np.float32),
    }


def test_gate_requires_reproduction_and_common_prefix_without_selecting_by_outcome_direction():
    case = _case()
    assert probe.eligibility_reasons(case) == []
    case["old_relation"] = "rescue"
    assert probe.eligibility_reasons(case) == []
    case["reproduction"]["original_pair_reproduced"] = False
    assert probe.eligibility_reasons(case) == ["original_pair_not_reproduced"]
    case = _case()
    case["alignment"]["before_first_h_or_grid_difference"]["generated_chunk_max_abs_difference"]["max"] = 0.001
    assert probe.eligibility_reasons(case) == [
        "pre_divergence_generated_chunk_max_abs_difference_exceeds_tolerance_or_is_missing"
    ]
    case["alignment"]["first_shared_decision_with_h_difference"] = None
    assert probe.eligibility_reasons(case) == ["no_shared_decision_with_h_difference"]


def test_trace_root_uses_saved_history_chunk_and_the_actual_previous_plan(tmp_path):
    case = _case()
    for mode in (probe.BASE_MODE, probe.HISTORY_MODE):
        directory = tmp_path / mode
        directory.mkdir()
        path = directory / "task03_ep000338.npz"
        np.savez_compressed(
            path, decision_steps=[10, 35], selected_h=[25, 5 if mode == probe.BASE_MODE else 10],
            generated_action_chunks=np.stack([np.full((25, 7), 2), np.full((25, 7), 3)]),
            decision_physics_state=[[10, 1, 2], [35, 3, 4]], previous_h=[10, 25],
            episode_progress=[0.01, 0.035], request_seeds=[10010, 10035],
        )
        path.with_suffix(".json").write_text(json.dumps({
            "mode": mode, "task_id": 3, "episode_id": 338,
            "success": case["reproduction"]["trace_success"][mode], "error": None,
        }))
    root = probe._trace_root(case, tmp_path)
    assert root["episode_id"] == 338
    assert root["step"] == 35
    assert root["previous_h"] == 25
    np.testing.assert_array_equal(root["saved_actions"], np.full((25, 7), 3))
    np.testing.assert_array_equal(root["previous_actions"], np.full((25, 7), 2))


def test_forced_branches_share_continuation_seeds_and_use_root_feedback_after_actual_h(monkeypatch):
    class Env:
        def __init__(self):
            self.steps = 100
            self.actions = []

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            self.steps += 1
            return {"step": self.steps}, 0.0, self.steps == 118, {}

    def restore(env, snapshot):
        del snapshot
        env.steps = 100
        env.actions = []
        return {"step": 100}

    monkeypatch.setattr(probe.collector, "_restore_snapshot", restore)
    monkeypatch.setattr(probe.libero_eval, "_env_success", lambda env: env.steps == 118)
    monkeypatch.setattr(probe.libero_eval, "_observation_to_policy_input", lambda observation, *args: observation)
    requests = []

    def request(client, policy_input, **kwargs):
        del client, policy_input
        requests.append(copy.deepcopy(kwargs))
        return _response(9), {}

    monkeypatch.setattr(probe.evaluator, "_request", request)
    root = {"step": 100, "root_request_seed": 3000007, "saved_actions": np.ones((25, 7), dtype=np.float32)}
    cache = {"prefix": np.ones(2048), "state": np.ones(32), "step": 100}
    features = []
    selector = SimpleNamespace(decide=lambda inputs: features.append(copy.deepcopy(inputs)) or (10, {}))
    args = SimpleNamespace(resize_size=224, v2_initial_budget=6, v2_budget_capacity=12)
    results = []
    first_feedback = []
    for forced_h in (5, 10):
        env = Env()
        start = len(features)
        results.append(probe._run_forced_branch(
            env=env, snapshot=None, root=root, forced_h=forced_h, repeat=2,
            episode_step_limit=200, task_description="task", root_observation_cache=cache,
            client=None, selector=selector, args=args,
        ))
        first_feedback.append(features[start])
        np.testing.assert_array_equal(env.actions[:forced_h], np.ones((forced_h, 7)))
        np.testing.assert_array_equal(env.actions[forced_h:], np.full((18 - forced_h, 7), 9))
    assert results[0]["continuation_seeds"][0] == results[1]["continuation_seeds"][0]
    assert results[0]["continuation_seeds"][0] == probe._continuation_seed(root["root_request_seed"], 2, 0)
    assert [row["actual_first_h"] for row in results] == [5, 10]
    assert [row["calls"] for row in results] == [3, 2]
    assert all(row["success"] for row in results)
    for inputs, horizon in zip(first_feedback, (5, 10), strict=True):
        assert inputs["previous_h"] == horizon
        assert inputs["elapsed_steps"] == horizon
        assert inputs["history_valid"]
        np.testing.assert_array_equal(inputs["previous_prefix_feature"], np.ones(2048))


def test_root_regeneration_mismatch_skips_branches_and_exact_bank_episode_is_used(tmp_path, monkeypatch):
    root = {
        "task_id": 3, "episode_id": 338, "step": 35, "physics_state": np.asarray([35, 3, 4]),
        "saved_actions": np.ones((25, 7), dtype=np.float32), "previous_actions": np.zeros((25, 7)),
        "previous_h": 25, "episode_progress": 0.035, "root_request_seed": 10035,
        "history_h": 10, "a_h": 5,
    }
    monkeypatch.setattr(probe, "_trace_root", lambda *args: root)
    restored_steps = []
    initial_ids = []
    sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: root["physics_state"]))
    env = SimpleNamespace(
        sim=sim, reset=lambda: None, set_init_state=lambda state: None, closed=False,
    )
    monkeypatch.setattr(probe.libero_eval, "_get_libero_env", lambda *args: (env, "task"))
    monkeypatch.setattr(probe.libero_eval, "_safe_close_env", lambda value: setattr(value, "closed", True))
    monkeypatch.setattr(probe.libero_eval, "_env_horizon", lambda value: 100)
    monkeypatch.setattr(probe.libero_eval, "_max_steps", lambda name: 100)
    monkeypatch.setattr(probe.libero_eval, "_observation_to_policy_input", lambda *args: {})
    monkeypatch.setattr(probe.replay, "_saved_snapshot", lambda env, physics, step: restored_steps.append(step) or None)
    monkeypatch.setattr(probe.collector, "_restore_snapshot", lambda *args: {})
    monkeypatch.setattr(probe.evaluator, "_request", lambda *args, **kwargs: (_response(1.1), {"wall_ms": 123.0}))
    monkeypatch.setattr(probe, "_run_forced_branch", lambda **kwargs: pytest.fail("mismatched root cannot be probed"))
    bank = SimpleNamespace(
        validate_presets=lambda *args: None,
        state=lambda task, episode: initial_ids.append((task, episode)) or np.zeros(3),
    )
    suite = SimpleNamespace(get_task=lambda task: None, get_task_init_states=lambda task: [])
    args = SimpleNamespace(
        seed=7, num_steps_wait=0, task_suite_name="libero_10", resize_size=224,
        v2_initial_budget=6, v2_budget_capacity=12,
    )
    result = probe.probe_case(
        _case(), trace_dir=tmp_path, task_suite=suite, bank=bank, selector=None, client=None,
        args=args, repeats=3, output=tmp_path / "case.json",
    )
    assert initial_ids == [(3, 338)]
    assert restored_steps == [35]
    assert env.closed
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "regenerated_root_actions_differ_from_saved_actions"
    assert result["root_regeneration_policy_calls"] == 1
    assert result["branches"] == []


def test_baseline_protocol_and_analysis_directory_are_explicit(tmp_path):
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps({
        "model_action_horizon": 25, "action_cot_denoising_steps": 10, "final_denoising_steps": 10,
        "initial_state_bank": "/bank", "seed": 7,
    }))
    args = probe._baseline_args(config, tmp_path / "output")
    assert args.initial_state_bank == "/bank"
    assert args.final_denoising_steps == 10
    assert args.trace_output_dir is None
    assert args.modes == [probe.HISTORY_MODE]
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "task03_ep000338.json").write_text(json.dumps(_case()))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "complete"}))
    assert probe.load_analysis_cases(summary) == [_case()]
    assert probe.load_analysis_cases(episode_dir) == [_case()]
