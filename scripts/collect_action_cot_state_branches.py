"""Collect same-root disturbed branches for persistent Action-CoT updates.

For every selected LIBERO root this collector:

1. captures the exact MuJoCo state;
2. obtains one full 10/10 Action-CoT cache;
3. restores that state for six fixed-age controlled execution branches;
4. obtains a fresh 10/10 target at every non-terminal endpoint.

The anchor and every endpoint request from a root use the same deterministic
policy seed.  This is the critical difference from adjacent-window exports:
teacher variation cannot masquerade as a response to the physical branch.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import math
import pathlib
import time
from typing import Any

import collect_execution_horizon_counterfactuals as horizon_collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy
from PIL import Image

from openpi.action_cot import branched_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--episode-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-roots-per-episode", type=int, default=5)
    parser.add_argument("--advance-horizon", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--records-per-shard", type=int, default=16)
    parser.add_argument("--snapshot-gate-repeats", type=int, default=3)
    parser.add_argument("--snapshot-physics-tolerance", type=float, default=1e-10)
    parser.add_argument("--snapshot-eef-tolerance", type=float, default=1e-10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    values = {
        "max_roots_per_episode": args.max_roots_per_episode,
        "advance_horizon": args.advance_horizon,
        "num_steps_wait": args.num_steps_wait,
        "resize_size": args.resize_size,
        "image_size": args.image_size,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "records_per_shard": args.records_per_shard,
        "snapshot_gating_repeats": args.snapshot_gate_repeats,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"{tuple(values)} must all be positive.")
    if args.snapshot_gate_repeats < 2:
        raise ValueError("snapshot_gate_repeats must be at least two.")
    if args.snapshot_physics_tolerance < 0 or args.snapshot_eef_tolerance < 0:
        raise ValueError("Snapshot determinism tolerances must be non-negative.")
    if args.advance_horizon > 10:
        raise ValueError("advance_horizon may not exceed the policy action horizon 10.")
    if not args.episode_ids or any(episode < 0 for episode in args.episode_ids):
        raise ValueError("episode_ids must contain non-negative values.")


def canonical_policy_seed(seed: int, task_id: int, episode_id: int, decision_step: int) -> int:
    """Return a stable seed tied to one physical root, not a dataset window."""

    values = (seed, task_id, episode_id, decision_step)
    if any(value < 0 for value in values):
        raise ValueError("Canonical seed inputs must be non-negative.")
    result = seed + task_id * 1_000_000 + episode_id * 10_000 + decision_step
    if result > np.iinfo(np.uint32).max:
        raise ValueError("Canonical policy seed exceeds uint32.")
    return result


@dataclasses.dataclass
class CanonicalSimulatorSnapshot:
    """MuJoCo plus robosuite controller state required for exact branches."""

    simulator: horizon_collector.SimulatorSnapshot
    simulator_ctrl: np.ndarray
    mutable_attributes: list[tuple[Any, str, Any]]


def _controller_objects(env: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    for candidate in horizon_collector._walk_env(env):
        robots = getattr(candidate, "robots", ())
        if robots is None:
            continue
        try:
            robot_values = list(robots)
        except TypeError:
            continue
        for robot in robot_values:
            controller = getattr(robot, "controller", None)
            for value in (
                robot,
                getattr(robot, "gripper", None),
                controller,
                getattr(controller, "interpolator_pos", None),
                getattr(controller, "interpolator_ori", None),
            ):
                if value is not None and id(value) not in seen:
                    seen.add(id(value))
                    result.append(value)
    return result


def _capture_canonical_snapshot(env: Any) -> CanonicalSimulatorSnapshot:
    simulator = horizon_collector._simulator(env)
    mutable_attributes: list[tuple[Any, str, Any]] = []
    names = (
        "current_action",
        "goal_pos",
        "goal_ori",
        "relative_ori",
        "ori_ref",
        "torques",
    )
    for owner in _controller_objects(env):
        mutable_attributes.extend(
            (owner, name, copy.deepcopy(getattr(owner, name)))
            for name in names
            if hasattr(owner, name)
        )
    return CanonicalSimulatorSnapshot(
        simulator=horizon_collector._capture_snapshot(env),
        simulator_ctrl=np.asarray(simulator.data.ctrl, dtype=np.float64).copy(),
        mutable_attributes=mutable_attributes,
    )


def _restore_mutable_attribute(owner: Any, name: str, value: Any) -> None:
    try:
        setattr(owner, name, copy.deepcopy(value))
        return
    except (AttributeError, TypeError):
        pass
    current = getattr(owner, name)
    current[...] = copy.deepcopy(value)


def _restore_canonical_snapshot(
    env: Any,
    snapshot: CanonicalSimulatorSnapshot,
) -> dict[str, Any]:
    simulator = horizon_collector._simulator(env)
    physics = snapshot.simulator.physics_state
    if hasattr(simulator, "set_state_from_flattened"):
        simulator.set_state_from_flattened(physics)
    else:
        simulator.set_state(physics)
    simulator.forward()
    for owner, name, value in snapshot.simulator.scalar_attributes:
        with contextlib.suppress(Exception):
            setattr(owner, name, copy.deepcopy(value))
    for owner, name, state in snapshot.simulator.random_states:
        generator = getattr(owner, name, None)
        with contextlib.suppress(Exception):
            if hasattr(generator, "set_state"):
                generator.set_state(copy.deepcopy(state))
            elif hasattr(generator, "bit_generator"):
                generator.bit_generator.state = copy.deepcopy(state)
    simulator.data.ctrl[...] = snapshot.simulator_ctrl
    for owner, name, value in snapshot.mutable_attributes:
        _restore_mutable_attribute(owner, name, value)
    for owner in _controller_objects(env):
        if hasattr(owner, "new_update"):
            owner.new_update = True

    for candidate in horizon_collector._walk_env(env):
        for method_name in ("_get_observations", "get_observations", "_get_observation"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    observation = method()
                    if isinstance(observation, dict) and "agentview_image" in observation:
                        return observation
    regenerate = getattr(env, "regenerate_obs_from_state", None)
    if callable(regenerate):
        observation = regenerate(physics)
        for owner, name, value in snapshot.simulator.scalar_attributes:
            with contextlib.suppress(Exception):
                setattr(owner, name, copy.deepcopy(value))
        return observation
    raise RuntimeError("Could not regenerate an observation after canonical snapshot restore.")


def _teacher_request(
    client: websocket_policy.WebsocketClientPolicy,
    policy_input: dict[str, Any],
    *,
    policy_seed: int,
    denoising_steps: int,
) -> dict[str, Any]:
    request = dict(policy_input)
    request["policy_seed"] = np.asarray(policy_seed, dtype=np.uint32)
    request["action_cot_denoising_steps"] = np.asarray(denoising_steps, dtype=np.int32)
    request["profile_policy_timing"] = np.zeros((), dtype=np.bool_)
    request["export_acot_cache"] = np.ones((), dtype=np.bool_)
    started = time.perf_counter()
    result = client.infer(request)
    result["collector_wall_ms"] = (time.perf_counter() - started) * 1000.0
    required = (
        "actions",
        "execution_horizon_state_normalized",
        "execution_horizon_coarse_actions_normalized",
        "execution_horizon_final_actions_normalized",
        "acot_iar_tokens",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"Policy response is missing branched-label fields: {missing}.")
    return result


def _resize_image(image: Any, size: int) -> np.ndarray:
    values = np.asarray(image)
    if values.dtype != np.uint8:
        values = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    resized = Image.fromarray(values).resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _images(policy_input: dict[str, Any], size: int) -> np.ndarray:
    return np.stack(
        (
            _resize_image(policy_input["observation/image"], size),
            _resize_image(policy_input["observation/wrist_image"], size),
        )
    )


def _pad_last_dim(values: Any, leading_shape: tuple[int, ...], target_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[:-1] != leading_shape:
        raise ValueError(f"Expected leading shape {leading_shape}, got {array.shape}.")
    if array.shape[-1] < target_dim:
        array = np.pad(array, [(0, 0)] * len(leading_shape) + [(0, target_dim - array.shape[-1])])
    return array[..., :target_dim]


def _state(result: dict[str, Any], shape: branched_dataset.BranchedDatasetShape) -> np.ndarray:
    values = np.asarray(result["execution_horizon_state_normalized"], dtype=np.float32).reshape(-1)
    if values.size < shape.state_dim:
        values = np.pad(values, (0, shape.state_dim - values.size))
    return values[: shape.state_dim]


def _teacher_tensors(
    result: dict[str, Any],
    shape: branched_dataset.BranchedDatasetShape,
) -> dict[str, np.ndarray]:
    ear = _pad_last_dim(
        result["execution_horizon_coarse_actions_normalized"],
        (shape.coarse_horizon,),
        shape.action_dim,
    )
    iar = np.asarray(result["acot_iar_tokens"], dtype=np.float32)
    if iar.shape != (shape.iar_tokens, shape.iar_dim):
        raise ValueError(f"Expected IAR {(shape.iar_tokens, shape.iar_dim)}, got {iar.shape}.")
    actions = _pad_last_dim(
        result["execution_horizon_final_actions_normalized"],
        (shape.action_horizon,),
        shape.action_dim,
    )
    actions_env = np.asarray(result["actions"], dtype=np.float32)
    if actions_env.ndim != 2 or actions_env.shape[0] < shape.action_horizon:
        raise ValueError(f"Expected at least {shape.action_horizon} raw actions, got {actions_env.shape}.")
    if actions_env.shape[1] < shape.env_action_dim:
        raise ValueError(f"Expected at least {shape.env_action_dim} raw action dims, got {actions_env.shape}.")
    return {
        "state": _state(result, shape),
        "ear": ear,
        "iar": iar,
        "actions": actions,
        "actions_env": actions_env[: shape.action_horizon, : shape.env_action_dim],
    }


def make_branch_actions(primary_actions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Create fixed-age branches in environment action space."""

    primary = np.asarray(primary_actions, dtype=np.float32)
    if primary.ndim != 2 or primary.shape[0] < 4 or primary.shape[1] < 7:
        raise ValueError(f"Expected primary actions [>=4, >=7], got {primary.shape}.")
    primary = primary[:, :7]

    nominal = primary[:4].copy()

    packet_fault = primary[:4].copy()
    packet_fault[1, :6] = 0.0
    packet_fault[1, 6] = packet_fault[0, 6]

    underact = primary[:4].copy()
    underact[:, :6] *= 0.5

    overact = primary[:4].copy()
    overact[:, :6] = np.clip(overact[:, :6] * 1.25, -1.0, 1.0)

    translation_pulse = primary[:4].copy()
    translation_pulse[1, 2] = np.clip(translation_pulse[1, 2] + 0.25, -1.0, 1.0)

    gripper_shift = primary[:4].copy()
    gripper_shift[1:, 6] = primary[:3, 6]

    branches = [
        nominal,
        packet_fault,
        underact,
        overact,
        translation_pulse,
        gripper_shift,
    ]
    strengths = np.asarray((1.0, 0.0, 0.5, 1.25, 0.25, 1.0), dtype=np.float32)
    return branches, strengths


def _step_actions(
    env: Any,
    observation: dict[str, Any],
    actions: np.ndarray,
) -> tuple[dict[str, Any], bool, list[np.ndarray]]:
    done = False
    executed: list[np.ndarray] = []
    for action in actions:
        try:
            observation, _, done, _ = env.step(np.asarray(action).tolist())
        except Exception as exc:
            if not libero_eval._is_terminated_episode_error(exc):
                raise
            done = bool(libero_eval._env_success(env))
            break
        executed.append(np.asarray(action, dtype=np.float32))
        if done or libero_eval._env_success(env):
            done = True
            break
    return observation, done, executed


def _maximum_absolute_difference(left: Any, right: Any) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if not left_array.size:
        return 0.0
    return float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))))


def _snapshot_determinism_gate(
    env: Any,
    snapshot: CanonicalSimulatorSnapshot,
    actions: np.ndarray,
    *,
    repeats: int,
    physics_tolerance: float,
    eef_tolerance: float,
) -> dict[str, float | int]:
    endpoints: list[dict[str, np.ndarray]] = []
    for _ in range(repeats):
        observation = _restore_canonical_snapshot(env, snapshot)
        observation, _, _ = _step_actions(env, observation, actions)
        endpoints.append(
            {
                "physics": horizon_collector._capture_snapshot(env).physics_state,
                "eef_pos": np.asarray(observation["robot0_eef_pos"]),
                "eef_quat": np.asarray(observation["robot0_eef_quat"]),
                "agentview": np.asarray(observation["agentview_image"]),
                "wrist": np.asarray(observation["robot0_eye_in_hand_image"]),
            }
        )
    _restore_canonical_snapshot(env, snapshot)
    reference = endpoints[0]
    physics_error = max(
        _maximum_absolute_difference(reference["physics"], endpoint["physics"])
        for endpoint in endpoints[1:]
    )
    eef_error = max(
        max(
            _maximum_absolute_difference(reference["eef_pos"], endpoint["eef_pos"]),
            _maximum_absolute_difference(reference["eef_quat"], endpoint["eef_quat"]),
        )
        for endpoint in endpoints[1:]
    )
    pixel_error = max(
        max(
            _maximum_absolute_difference(reference["agentview"], endpoint["agentview"]),
            _maximum_absolute_difference(reference["wrist"], endpoint["wrist"]),
        )
        for endpoint in endpoints[1:]
    )
    if physics_error > physics_tolerance or eef_error > eef_tolerance or pixel_error != 0.0:
        raise RuntimeError(
            "Canonical snapshot determinism gate failed: "
            f"physics={physics_error:.3e} (limit {physics_tolerance:.3e}), "
            f"eef={eef_error:.3e} (limit {eef_tolerance:.3e}), pixel={pixel_error:.0f}."
        )
    return {
        "repeats": repeats,
        "physics_max_abs": physics_error,
        "eef_max_abs": eef_error,
        "pixel_max_abs": pixel_error,
    }


def _empty_branch_arrays(
    shape: branched_dataset.BranchedDatasetShape,
    anchor: dict[str, np.ndarray],
    anchor_images: np.ndarray,
) -> dict[str, np.ndarray]:
    branches = shape.num_branches
    return {
        "branch_ids": np.arange(branches, dtype=np.uint8),
        "branch_steps": np.zeros((branches,), dtype=np.uint8),
        "branch_valid": np.zeros((branches,), dtype=np.bool_),
        "endpoint_done": np.zeros((branches,), dtype=np.bool_),
        "branch_strength": np.zeros((branches,), dtype=np.float32),
        "physics_delta_l2": np.zeros((branches,), dtype=np.float32),
        "current_images": np.repeat(anchor_images[None], branches, axis=0),
        "current_state": np.repeat(anchor["state"][None], branches, axis=0),
        "fresh_ear": np.repeat(anchor["ear"][None], branches, axis=0),
        "fresh_iar": np.repeat(anchor["iar"][None], branches, axis=0),
        "fresh_actions": np.repeat(anchor["actions"][None], branches, axis=0),
        "fresh_actions_env": np.repeat(anchor["actions_env"][None], branches, axis=0),
        "executed_actions": np.zeros(
            (branches, shape.max_executed_steps, shape.env_action_dim),
            dtype=np.float32,
        ),
        "executed_valid": np.zeros((branches, shape.max_executed_steps), dtype=np.bool_),
    }


def _collect_root(
    env: Any,
    observation: dict[str, Any],
    snapshot: CanonicalSimulatorSnapshot,
    *,
    client: websocket_policy.WebsocketClientPolicy,
    task_description: str,
    task_id: int,
    episode_id: int,
    decision_step: int,
    root_id: int,
    policy_seed: int,
    args: argparse.Namespace,
    shape: branched_dataset.BranchedDatasetShape,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float | int]]:
    policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
    anchor_result = _teacher_request(
        client,
        policy_input,
        policy_seed=policy_seed,
        denoising_steps=args.action_cot_denoising_steps,
    )
    anchor = _teacher_tensors(anchor_result, shape)
    anchor_images = _images(policy_input, shape.image_height)
    arrays = _empty_branch_arrays(shape, anchor, anchor_images)
    branch_actions, strengths = make_branch_actions(anchor["actions_env"])
    arrays["branch_strength"][:] = strengths
    gate = _snapshot_determinism_gate(
        env,
        snapshot,
        anchor["actions_env"][: shape.max_executed_steps],
        repeats=args.snapshot_gate_repeats,
        physics_tolerance=args.snapshot_physics_tolerance,
        eef_tolerance=args.snapshot_eef_tolerance,
    )

    root_physics = snapshot.simulator.physics_state
    for branch_id, actions in enumerate(branch_actions):
        branch_observation = _restore_canonical_snapshot(env, snapshot)
        branch_observation, done, executed = _step_actions(env, branch_observation, actions)
        arrays["branch_steps"][branch_id] = len(executed)
        arrays["endpoint_done"][branch_id] = done
        if executed:
            count = len(executed)
            arrays["executed_actions"][branch_id, :count] = np.stack(executed)
            arrays["executed_valid"][branch_id, :count] = True
        endpoint_physics = horizon_collector._capture_snapshot(env).physics_state
        arrays["physics_delta_l2"][branch_id] = float(
            np.linalg.norm(endpoint_physics - root_physics) / math.sqrt(max(root_physics.size, 1))
        )
        endpoint_input = libero_eval._observation_to_policy_input(
            branch_observation,
            task_description,
            args.resize_size,
        )
        arrays["current_images"][branch_id] = _images(endpoint_input, shape.image_height)
        if done:
            continue
        endpoint_result = _teacher_request(
            client,
            endpoint_input,
            policy_seed=policy_seed,
            denoising_steps=args.action_cot_denoising_steps,
        )
        target = _teacher_tensors(endpoint_result, shape)
        arrays["current_state"][branch_id] = target["state"]
        arrays["fresh_ear"][branch_id] = target["ear"]
        arrays["fresh_iar"][branch_id] = target["iar"]
        arrays["fresh_actions"][branch_id] = target["actions"]
        arrays["fresh_actions_env"][branch_id] = target["actions_env"]
        arrays["branch_valid"][branch_id] = True

    record: dict[str, Any] = {
        "root_id": root_id,
        "task_id": task_id,
        "episode_id": episode_id,
        "decision_step": decision_step,
        "policy_seed": policy_seed,
        "anchor_images": anchor_images,
        "anchor_state": anchor["state"],
        "cached_ear": anchor["ear"],
        "cached_iar": anchor["iar"],
        "cached_actions": anchor["actions"],
        "cached_actions_env": anchor["actions_env"],
        **arrays,
    }
    return record, anchor["actions_env"], gate


def _existing_keys(output_dir: pathlib.Path) -> set[tuple[int, int, int]]:
    if not list(output_dir.glob("shard-*.h5")):
        return set()
    arrays = branched_dataset.load_branched_arrays(
        (output_dir,),
        fields=("task_id", "episode_id", "decision_step"),
    )
    return set(
        zip(
            arrays["task_id"].astype(int),
            arrays["episode_id"].astype(int),
            arrays["decision_step"].astype(int),
            strict=True,
        )
    )


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    shape = branched_dataset.BranchedDatasetShape(
        image_height=args.image_size,
        image_width=args.image_size,
    )
    output_dir = pathlib.Path(args.output_dir)
    existing = _existing_keys(output_dir)
    metadata = {
        "task_suite": args.task_suite_name,
        "seed_protocol": "same canonical seed for anchor and every endpoint in a root",
        "branch_names": list(branched_dataset.BRANCH_NAMES),
        "episode_ids": list(dict.fromkeys(args.episode_ids)),
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "advance_horizon": args.advance_horizon,
        "image_size": args.image_size,
        "snapshot_gate_repeats": args.snapshot_gate_repeats,
        "snapshot_physics_tolerance": args.snapshot_physics_tolerance,
        "snapshot_eef_tolerance": args.snapshot_eef_tolerance,
        "definition": "exact MuJoCo same-root physical branches with fresh 10/10 EAR/IAR/action targets",
    }
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_end = suite.n_tasks if args.max_tasks is None else min(suite.n_tasks, args.task_start + args.max_tasks)
    max_steps = libero_eval._max_steps(args.task_suite_name)
    processed = 0
    skipped = 0
    failed = 0
    valid_by_branch = np.zeros((shape.num_branches,), dtype=np.int64)
    delta_by_branch: list[list[float]] = [[] for _ in range(shape.num_branches)]
    maximum_gate_errors = {
        "physics_max_abs": 0.0,
        "eef_max_abs": 0.0,
        "pixel_max_abs": 0.0,
    }
    started = time.monotonic()

    with branched_dataset.ShardedBranchedWriter(
        output_dir,
        shape=shape,
        records_per_shard=args.records_per_shard,
        metadata=metadata,
    ) as writer:
        for task_id in range(args.task_start, task_end):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            for episode_id in dict.fromkeys(args.episode_ids):
                env, task_description = libero_eval._get_libero_env(
                    task,
                    libero_eval.LIBERO_ENV_RESOLUTION,
                    args.seed,
                )
                try:
                    env.reset()
                    observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
                    step = 0
                    done = False
                    for _ in range(args.num_steps_wait):
                        observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
                        step += 1
                        if done:
                            break
                    environment_horizon = libero_eval._env_horizon(env)
                    episode_limit = max_steps + args.num_steps_wait
                    if environment_horizon is not None:
                        episode_limit = min(episode_limit, environment_horizon)

                    root_index = 0
                    while not done and step < episode_limit and root_index < args.max_roots_per_episode:
                        key = (task_id, episode_id, step)
                        snapshot = _capture_canonical_snapshot(env)
                        policy_seed = canonical_policy_seed(args.seed, task_id, episode_id, step)
                        root_id = task_id * 1_000_000 + episode_id * 1_000 + root_index
                        primary_actions: np.ndarray | None = None
                        try:
                            record, primary_actions, gate = _collect_root(
                                env,
                                observation,
                                snapshot,
                                client=client,
                                task_description=task_description,
                                task_id=task_id,
                                episode_id=episode_id,
                                decision_step=step,
                                root_id=root_id,
                                policy_seed=policy_seed,
                                args=args,
                                shape=shape,
                            )
                            for name, current_maximum in maximum_gate_errors.items():
                                maximum_gate_errors[name] = max(
                                    current_maximum,
                                    float(gate[name]),
                                )
                            if key in existing:
                                skipped += 1
                            else:
                                writer.append(record)
                                processed += 1
                                valid = np.asarray(record["branch_valid"], dtype=np.bool_)
                                valid_by_branch += valid.astype(np.int64)
                                target_delta = np.mean(
                                    np.square(
                                        np.asarray(record["fresh_ear"], dtype=np.float32)[..., :7]
                                        - np.asarray(record["cached_ear"], dtype=np.float32)[None, ..., :7]
                                    ),
                                    axis=(1, 2),
                                )
                                for branch_id in range(shape.num_branches):
                                    if valid[branch_id]:
                                        delta_by_branch[branch_id].append(float(target_delta[branch_id]))
                                print(
                                    json.dumps(
                                        {
                                            "processed_roots": processed,
                                            "task_id": task_id,
                                            "episode_id": episode_id,
                                            "decision_step": step,
                                            "valid_branches": int(np.sum(valid)),
                                            "target_ear_mse_vs_cache": target_delta.tolist(),
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                        except Exception as exc:
                            failed += 1
                            print(
                                json.dumps(
                                    {
                                        "task_id": task_id,
                                        "episode_id": episode_id,
                                        "decision_step": step,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            if not args.continue_on_error:
                                raise
                        finally:
                            observation = _restore_canonical_snapshot(env, snapshot)

                        if primary_actions is None:
                            break
                        observation, done, executed = _step_actions(
                            env,
                            observation,
                            primary_actions[: args.advance_horizon],
                        )
                        step += len(executed)
                        root_index += 1
                finally:
                    libero_eval._safe_close_env(env)

    summary = {
        "new_roots": processed,
        "existing_roots_skipped": skipped,
        "failed_roots": failed,
        "elapsed_seconds": time.monotonic() - started,
        "valid_branches": {
            name: int(valid_by_branch[index])
            for index, name in enumerate(branched_dataset.BRANCH_NAMES)
        },
        "mean_fresh_ear_mse_vs_cache_7d": {
            name: float(np.mean(delta_by_branch[index])) if delta_by_branch[index] else None
            for index, name in enumerate(branched_dataset.BRANCH_NAMES)
        },
        "snapshot_determinism_max": maximum_gate_errors,
        "note": "Open-loop same-root labels only; no closed-loop success claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
