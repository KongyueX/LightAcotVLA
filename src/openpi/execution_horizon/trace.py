"""Passive simulator traces for comparing execution-horizon decisions."""
# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Mapping
import json
import pathlib
from typing import Any

import numpy as np

from openpi.execution_horizon import privileged_progress

TRACE_VERSION = 1


def _observation_vector(observation: Mapping[str, Any], name: str) -> np.ndarray:
    return np.asarray(observation.get(name, []), dtype=np.float64).reshape(-1).copy()


def _observed_state(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "eef_pos": _observation_vector(observation, "robot0_eef_pos"),
        "eef_quat": _observation_vector(observation, "robot0_eef_quat"),
        "gripper_qpos": _observation_vector(observation, "robot0_gripper_qpos"),
        "object_state": _observation_vector(observation, "object-state"),
    }


def _agentview_frame(observation: Mapping[str, Any]) -> np.ndarray | None:
    if "agentview_image" not in observation:
        return None
    # Match the orientation of the images already supplied to the policy.
    return np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1]).copy()


def _physics_state(env: Any) -> tuple[np.ndarray, float | None]:
    frequency = None
    physics = None
    for candidate in privileged_progress._walk_env(env):
        control_frequency = getattr(candidate, "control_freq", None)
        if control_frequency is not None:
            frequency = float(control_frequency)
        simulator = getattr(candidate, "sim", None)
        if physics is None and simulator is not None and hasattr(simulator, "get_state"):
            physics = np.asarray(simulator.get_state().flatten(), dtype=np.float64).copy()
    if physics is None:
        raise AttributeError("A diagnostic trace requires the LIBERO simulator state.")
    return physics, frequency


def _distribution(value: Any | None, width: int) -> np.ndarray:
    if value is None:
        return np.full(width, np.nan, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(width).copy()


class EpisodeTrace:
    """Record observations and commands without calling the policy or stepping the simulator."""

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        *,
        mode: str,
        task_id: int,
        episode: int,
        seed: int,
        video_stride: int = 5,
    ) -> None:
        if video_stride <= 0:
            raise ValueError("trace video stride must be positive.")
        self.path = pathlib.Path(output_dir) / mode / f"task{task_id:02d}_ep{episode:06d}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.video_stride = video_stride
        self.metadata: dict[str, Any] = {
            "schema_version": TRACE_VERSION, "diagnostic_only": True,
            "mode": mode, "task_id": task_id, "episode_id": episode, "seed": seed,
            "video_stride": video_stride, "image_orientation": "agentview[::-1,::-1]",
            "state_alignment": "state_steps and eef/object observations are after executed_actions",
            "decision_alignment": "decision_steps and physics/proprio observations are before the selected chunk",
            "execution_h_definition": "recorded successful environment steps belonging to each decision",
            "policy_extra_calls": 0,
        }
        self.initial: dict[str, np.ndarray] = {}
        self.decisions: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []
        self.frame_steps: list[int] = []
        self.decision_frames: list[tuple[int, int, np.ndarray]] = []
        self.last_frame: np.ndarray | None = None
        self.last_frame_step = 0
        self.closed = False

    def record_initial(self, observation: Mapping[str, Any], *, step: int = 0) -> None:
        self.initial = {f"initial_{name}": value for name, value in _observed_state(observation).items()}
        self.initial["initial_step"] = np.asarray(step, dtype=np.int64)
        frame = _agentview_frame(observation)
        if frame is not None:
            self.frames.append(frame)
            self.frame_steps.append(step)
            self.last_frame = frame
            self.last_frame_step = step

    def record_decision(
        self,
        *,
        env: Any,
        observation: Mapping[str, Any],
        policy_input: Mapping[str, Any],
        result: Mapping[str, Any],
        selector_info: Mapping[str, Any],
        step: int,
        selected_h: int,
        execution_h: int,
        previous_h: int,
        episode_progress: float,
        request_seed: int,
    ) -> None:
        physics, frequency = _physics_state(env)
        if frequency is not None:
            self.metadata["control_frequency_hz"] = frequency
        anchor_logits = _distribution(result.get("execution_horizon_ordered_continuation_logits"), 4)
        anchor_probabilities = _distribution(result.get("execution_horizon_ordered_horizon_probability"), 5)
        anchor_selected_h = int(np.asarray(result.get("execution_horizon_ordered_selected_h", -1)).item())
        if "execution_horizon_candidate_horizons" in result:
            self.metadata["candidate_horizons"] = np.asarray(result["execution_horizon_candidate_horizons"]).tolist()
        ordered_logits = _distribution(selector_info.get("ordered_continuation_logits", anchor_logits), 4)
        ordered_probabilities = _distribution(selector_info.get("ordered_horizon_probability", anchor_probabilities), 5)
        observed = _observed_state(observation)
        self.decisions.append({
            "decision_steps": int(step), "selected_h": int(selected_h),
            "planned_execution_h": int(execution_h), "execution_h": 0,
            "generated_action_chunks": np.asarray(result["actions"], dtype=np.float32).copy(),
            "decision_physics_state": physics,
            "decision_proprio": np.asarray(policy_input["observation/state"], dtype=np.float64).reshape(-1).copy(),
            **{f"decision_{name}": value for name, value in observed.items()},
            "anchor_continuation_logits": anchor_logits, "anchor_probabilities": anchor_probabilities,
            "anchor_selected_h": anchor_selected_h,
            "ordered_continuation_logits": ordered_logits, "ordered_probabilities": ordered_probabilities,
            "previous_h": int(previous_h), "history_valid": bool(self.decisions),
            "episode_progress": float(episode_progress), "request_seeds": int(request_seed),
        })
        if len(self.decision_frames) < 6 and (len(self.decisions) == 1 or selected_h != anchor_selected_h):
            frame = _agentview_frame(observation)
            if frame is not None:
                self.decision_frames.append((len(self.decisions) - 1, step, frame))

    def record_step(
        self,
        observation: Mapping[str, Any],
        action: Any,
        *,
        step: int,
        is_wait: bool = False,
    ) -> None:
        decision_index = -1 if is_wait else len(self.decisions) - 1
        self.states.append({
            "state_steps": int(step), "state_decision_index": decision_index,
            "executed_actions": np.asarray(action, dtype=np.float32).reshape(-1).copy(),
            **_observed_state(observation),
        })
        if decision_index >= 0:
            self.decisions[decision_index]["execution_h"] += 1
        frame = _agentview_frame(observation)
        if frame is not None:
            self.last_frame = frame
            self.last_frame_step = step
            if step % self.video_stride == 0:
                self.frames.append(frame)
                self.frame_steps.append(step)

    def record_step_error(self, *, step: int, error: Exception) -> None:
        self.metadata.setdefault("step_errors", []).append({
            "attempted_step": int(step), "type": type(error).__name__, "message": str(error),
        })

    def close(self, *, success: bool, steps: int, error: str | None = None) -> None:
        if self.closed:
            return
        if self.last_frame is not None and (not self.frame_steps or self.frame_steps[-1] != self.last_frame_step):
            self.frames.append(self.last_frame)
            self.frame_steps.append(self.last_frame_step)
        arrays = dict(self.initial)
        for rows in (self.decisions, self.states):
            if rows:
                arrays.update({name: np.stack([row[name] for row in rows]) for name in rows[0]})
        if not self.decisions:
            arrays.update({
                "decision_steps": np.empty(0, dtype=np.int64),
                "selected_h": np.empty(0, dtype=np.int32), "execution_h": np.empty(0, dtype=np.int32),
                "planned_execution_h": np.empty(0, dtype=np.int32),
                "generated_action_chunks": np.empty((0, 25, 7), dtype=np.float32),
                "decision_physics_state": np.empty((0, 0), dtype=np.float64),
                "decision_proprio": np.empty((0, 0), dtype=np.float64),
            })
        if not self.states:
            arrays.update({
                "state_steps": np.empty(0, dtype=np.int64),
                "state_decision_index": np.empty(0, dtype=np.int32),
                "executed_actions": np.empty((0, 7), dtype=np.float32),
                "eef_pos": np.empty((0, 3), dtype=np.float64),
                "eef_quat": np.empty((0, 4), dtype=np.float64),
                "gripper_qpos": np.empty((0, 0), dtype=np.float64),
            })
        arrays["video_frame_steps"] = np.asarray(self.frame_steps, dtype=np.int64)
        np.savez_compressed(self.path.with_suffix(".npz"), **arrays)
        self.metadata.update({
            "success": bool(success), "final_step": int(steps), "error": error,
            "decisions": len(self.decisions), "recorded_environment_steps": len(self.states),
            "npz": str(self.path.with_suffix(".npz")),
        })
        if self.frames or self.decision_frames:
            try:
                import imageio.v2 as imageio

                if self.frames:
                    frequency = self.metadata.get("control_frequency_hz")
                    fps = frequency / self.video_stride if frequency is not None else 10.0
                    video_path = self.path.with_suffix(".mp4")
                    imageio.mimwrite(video_path, self.frames, fps=fps)
                    self.metadata.update(video=str(video_path), video_fps=fps)
                screenshots = []
                for decision_index, step, frame in self.decision_frames:
                    png = self.path.with_name(f"{self.path.name}_decision{decision_index:03d}_step{step:04d}.png")
                    imageio.imwrite(png, frame)
                    screenshots.append({"decision_index": decision_index, "step": step, "path": str(png)})
                self.metadata["decision_images"] = screenshots
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self.metadata["media_error"] = f"{type(exc).__name__}: {exc}"
        self.path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n")
        self.closed = True
