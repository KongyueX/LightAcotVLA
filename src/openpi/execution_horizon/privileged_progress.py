"""Dense privileged progress signals for short LIBERO candidate prefixes.

This module is intentionally simulator-side only. It turns the logical BDDL
goal into a stage-aware continuous score that can label action-chunk
candidates without looking at their later terminal outcomes.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

_PLACEMENT_PREDICATES = frozenset({"in", "on", "stack"})
_ARTICULATION_PREDICATES = frozenset({"close", "open", "turnoff", "turnon"})


@dataclasses.dataclass(frozen=True)
class LiberoGoalProgress:
    """Serializable dense progress evaluation for one simulator state."""

    score: float
    normalized_score: float
    satisfied_count: int
    total_goals: int
    active_kind: str
    active_progress: float
    components: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _walk_env(env: Any) -> list[Any]:
    queue = [env]
    result = []
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        result.append(candidate)
        for name in ("env", "_env", "unwrapped"):
            try:
                child = getattr(candidate, name, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                queue.append(child)
    return result


def _task_env(env: Any) -> Any:
    for candidate in _walk_env(env):
        if hasattr(candidate, "parsed_problem") and hasattr(candidate, "object_states_dict"):
            return candidate
    raise AttributeError("Could not find the LIBERO task environment in the wrapper chain.")


def _position(state: Any) -> np.ndarray:
    return np.asarray(state.get_geom_state()["pos"], dtype=np.float64)


def _eef_position(task_env: Any, observation: dict[str, Any] | None) -> np.ndarray:
    del observation
    # Read the site directly from the restored simulator. LIBERO's observation
    # wrapper can retain robot0_eef_pos from the state immediately before a
    # set_state/forward call, which would leak one candidate prefix into the
    # next candidate's progress score.
    robot = task_env.robots[0]
    return np.asarray(task_env.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64)


def _is_grasped(task_env: Any, object_name: str) -> bool:
    try:
        object_model = task_env.get_object(object_name)
        return bool(task_env._check_grasp(task_env.robots[0].gripper, object_model.contact_geoms))
    except Exception:
        return False


def _exp_closeness(distance: float, scale: float) -> float:
    return float(math.exp(-max(distance, 0.0) / scale))


def _placement_component(
    task_env: Any,
    state: list[str],
    *,
    satisfied: bool,
    eef_position: np.ndarray,
) -> dict[str, Any]:
    object_name, target_name = state[1], state[2]
    object_position = _position(task_env.object_states_dict[object_name])
    target_position = _position(task_env.object_states_dict[target_name])
    object_to_target = float(np.linalg.norm(object_position - target_position))
    eef_to_object = float(np.linalg.norm(eef_position - object_position))
    grasped = _is_grasped(task_env, object_name)

    if satisfied:
        continuous_progress = 1.0
    elif grasped:
        # Reaching/grasping is the first half of a placement. Once grasped,
        # reward movement of the object toward its logical target site.
        continuous_progress = 0.5 + 0.5 * _exp_closeness(object_to_target, 0.30)
    else:
        # Before grasp, either approaching the object or already nudging it
        # close to the target is useful. Keep this phase below 0.5 so a stable
        # grasp is always recognized as stage advancement.
        reach_progress = 0.5 * _exp_closeness(eef_to_object, 0.15)
        placement_progress = 0.5 * _exp_closeness(object_to_target, 0.30)
        continuous_progress = max(reach_progress, placement_progress)

    return {
        "predicate": state[0],
        "arguments": state[1:],
        "kind": "placement",
        "satisfied": satisfied,
        "continuous_progress": float(continuous_progress),
        "grasped": grasped,
        "eef_to_object_distance": eef_to_object,
        "object_to_target_distance": object_to_target,
    }


def _joint_values(object_state: Any) -> np.ndarray:
    try:
        values = object_state.get_joint_state()
    except Exception:
        return np.empty((0,), dtype=np.float64)
    flattened = []
    for value in values:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        flattened.extend(array.tolist())
    return np.asarray(flattened, dtype=np.float64)


def _articulation_fraction(task_env: Any, predicate: str, object_name: str) -> tuple[float, list[float]]:
    state = task_env.object_states_dict[object_name]
    joint_values = _joint_values(state)
    if joint_values.size == 0:
        return 0.0, []
    model = task_env.get_object(object_name)
    properties = getattr(model, "object_properties", {}).get("articulation", {})

    if predicate in {"close", "open"}:
        open_ranges = properties.get("default_open_ranges")
        close_ranges = properties.get("default_close_ranges")
        if open_ranges and close_ranges:
            open_boundary = float(max(open_ranges))
            close_boundary = float(min(close_ranges))
            denominator = close_boundary - open_boundary
            if abs(denominator) > 1e-8:
                close_fraction = np.clip((joint_values - open_boundary) / denominator, 0.0, 1.0)
                fraction = close_fraction if predicate == "close" else 1.0 - close_fraction
                return float(np.max(fraction)), joint_values.tolist()

    range_names = {
        "turnon": ("default_turnon_ranges", "default_turnoff_ranges"),
        "turnoff": ("default_turnoff_ranges", "default_turnon_ranges"),
    }
    desired_name, opposite_name = range_names.get(predicate, (None, None))
    desired = properties.get(desired_name) if desired_name else None
    opposite = properties.get(opposite_name) if opposite_name else None
    if desired and opposite:
        desired_center = float(np.mean(desired))
        opposite_center = float(np.mean(opposite))
        denominator = desired_center - opposite_center
        if abs(denominator) > 1e-8:
            fraction = np.clip((joint_values - opposite_center) / denominator, 0.0, 1.0)
            return float(np.max(fraction)), joint_values.tolist()
    return 0.0, joint_values.tolist()


def _articulation_component(
    task_env: Any,
    state: list[str],
    *,
    satisfied: bool,
    eef_position: np.ndarray,
) -> dict[str, Any]:
    object_name = state[1]
    object_position = _position(task_env.object_states_dict[object_name])
    eef_to_object = float(np.linalg.norm(eef_position - object_position))
    joint_progress, joint_values = _articulation_fraction(task_env, state[0], object_name)
    reach_progress = _exp_closeness(eef_to_object, 0.25)
    continuous_progress = 1.0 if satisfied else 0.85 * joint_progress + 0.15 * reach_progress
    return {
        "predicate": state[0],
        "arguments": state[1:],
        "kind": "articulation",
        "satisfied": satisfied,
        "continuous_progress": float(continuous_progress),
        "eef_to_object_distance": eef_to_object,
        "joint_progress": joint_progress,
        "joint_values": joint_values,
    }


def score_libero_goal_progress(
    env: Any,
    observation: dict[str, Any] | None = None,
) -> LiberoGoalProgress:
    """Score current task progress using BDDL predicates and MuJoCo state.

    Completed predicates contribute one point each. Among unfinished goals,
    only the currently actionable phase contributes dense progress: placement
    goals take precedence over articulation goals, and parallel placements use
    their maximum rather than their sum so working on either object is valid.
    """

    task_env = _task_env(env)
    goals = [list(state) for state in task_env.parsed_problem["goal_state"]]
    eef_position = _eef_position(task_env, observation)
    components: list[dict[str, Any]] = []
    for state in goals:
        predicate = str(state[0]).lower()
        normalized_state = [predicate, *state[1:]]
        try:
            satisfied = bool(task_env._eval_predicate(normalized_state))
        except Exception:
            satisfied = False
        if predicate in _PLACEMENT_PREDICATES and len(state) == 3:
            component = _placement_component(
                task_env,
                normalized_state,
                satisfied=satisfied,
                eef_position=eef_position,
            )
        elif predicate in _ARTICULATION_PREDICATES and len(state) == 2:
            component = _articulation_component(
                task_env,
                normalized_state,
                satisfied=satisfied,
                eef_position=eef_position,
            )
        else:
            component = {
                "predicate": predicate,
                "arguments": state[1:],
                "kind": "other",
                "satisfied": satisfied,
                "continuous_progress": float(satisfied),
            }
        components.append(component)

    satisfied_count = sum(bool(component["satisfied"]) for component in components)
    unfinished_placements = [
        component for component in components if component["kind"] == "placement" and not bool(component["satisfied"])
    ]
    unfinished_articulations = [
        component
        for component in components
        if component["kind"] == "articulation" and not bool(component["satisfied"])
    ]
    if unfinished_placements:
        active_kind = "placement"
        active_progress = max(float(component["continuous_progress"]) for component in unfinished_placements)
    elif unfinished_articulations:
        active_kind = "articulation"
        active_progress = max(float(component["continuous_progress"]) for component in unfinished_articulations)
    else:
        active_kind = "complete" if satisfied_count == len(goals) else "other"
        active_progress = 0.0

    score = float(satisfied_count + active_progress)
    total_goals = len(goals)
    return LiberoGoalProgress(
        score=score,
        normalized_score=score / max(total_goals, 1),
        satisfied_count=satisfied_count,
        total_goals=total_goals,
        active_kind=active_kind,
        active_progress=float(active_progress),
        components=components,
    )
