from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from openpi.execution_horizon.trace import EpisodeTrace
from openpi.execution_horizon.trace import _agentview_frame


def _observation(step: int) -> dict:
    return {
        "robot0_eef_pos": np.asarray([step, 0, 0], dtype=np.float64),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.asarray([step / 10, -step / 10]),
        "object-state": np.asarray([step, 2, 3, 4], dtype=np.float64),
    }


def test_trace_preserves_same_call_heads_and_aligns_early_stop_actual_actions(tmp_path) -> None:
    physics = np.arange(8, dtype=np.float64)
    sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: physics))
    env = SimpleNamespace(env=SimpleNamespace(sim=sim, control_freq=20))
    trace = EpisodeTrace(tmp_path, mode="ordered_feedback_history", task_id=3, episode=338, seed=7)
    trace.record_initial(_observation(0))
    dummy = np.zeros(7, dtype=np.float32)
    trace.record_step(_observation(1), dummy, step=1, is_wait=True)
    actions = np.ones((25, 7), dtype=np.float32)
    result = {
        "actions": actions,
        "execution_horizon_ordered_continuation_logits": np.zeros(4),
        "execution_horizon_ordered_horizon_probability": np.asarray([0.5, 0.25, 0.125, 0.0625, 0.0625]),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
        "execution_horizon_ordered_selected_h": np.asarray(5),
    }
    trace.record_decision(
        env=env, observation=_observation(1), policy_input={"observation/state": np.arange(8)},
        result=result,
        selector_info={"ordered_horizon_probability": [0, 0, 0, 0, 1], "ordered_continuation_logits": [9] * 4},
        step=1, selected_h=25, execution_h=24, previous_h=10, episode_progress=0.01, request_seed=3380008,
    )
    physics[:] = 99
    actions[:] = 99
    trace.record_step(_observation(2), np.ones(7), step=2)
    trace.record_step(_observation(3), np.ones(7) * 2, step=3)
    trace.close(success=True, steps=3)
    path = tmp_path / "ordered_feedback_history/task03_ep000338"
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["decision_steps"], [1])
        np.testing.assert_array_equal(arrays["selected_h"], [25])
        np.testing.assert_array_equal(arrays["planned_execution_h"], [24])
        np.testing.assert_array_equal(arrays["execution_h"], [2])
        np.testing.assert_array_equal(arrays["state_steps"], [1, 2, 3])
        np.testing.assert_array_equal(arrays["state_decision_index"], [-1, 0, 0])
        np.testing.assert_array_equal(arrays["eef_pos"][:, 0], [1, 2, 3])
        np.testing.assert_array_equal(arrays["executed_actions"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(arrays["decision_physics_state"][0], np.arange(8))
        np.testing.assert_array_equal(arrays["generated_action_chunks"], np.ones((1, 25, 7)))
        np.testing.assert_array_equal(arrays["anchor_selected_h"], [5])
        np.testing.assert_array_equal(arrays["ordered_probabilities"], [[0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(arrays["initial_eef_pos"], [0, 0, 0])
        assert arrays["object_state"].shape == (3, 4)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["success"] is True
    assert metadata["final_step"] == 3
    assert metadata["schema_version"] == 1
    assert metadata["diagnostic_only"] is True
    assert metadata["policy_extra_calls"] == 0
    assert metadata["seed"] == 7


def test_existing_camera_frame_orientation_matches_policy_without_mutating_observation() -> None:
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    original = pixels.copy()
    frame = _agentview_frame({"agentview_image": pixels})
    np.testing.assert_array_equal(frame, pixels[::-1, ::-1])
    frame[:] = 0
    np.testing.assert_array_equal(pixels, original)


def test_empty_failed_trace_closes_without_pickle_or_video(tmp_path) -> None:
    trace = EpisodeTrace(tmp_path, mode="ordered_transformer", task_id=0, episode=336, seed=7)
    trace.record_initial(_observation(0))
    trace.close(success=False, steps=0, error="RuntimeError: policy disconnected")
    trace.close(success=True, steps=99)
    path = tmp_path / "ordered_transformer/task00_ep000336"
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert arrays["generated_action_chunks"].shape == (0, 25, 7)
        assert arrays["state_steps"].size == 0
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["success"] is False
    assert metadata["error"] == "RuntimeError: policy disconnected"
