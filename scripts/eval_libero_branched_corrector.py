"""Fast paired LIBERO audit for the fixed-age branched action corrector.

This deliberately leaves ``eval_libero_action_cot_pruning.py`` unchanged.  It
compares three execution schedules on identical LIBERO initial states:

* ``full_h4``: refresh the complete 10/10 ACoT policy every four actions.
* ``stale_h10``: refresh ACoT every ten actions and execute the whole chunk.
* ``corrector_h10``: refresh ACoT every ten actions, execute its first four
  actions, then replace actions 4:10 with the saved direct H6 corrector output.
* ``matched_diff_h10``: anchor the second H6 at stale and add only the clipped
  response between current and independently trained no-current checkpoints.

The corrector arm never makes a policy RPC at the four-action boundary.  Its
latest image is read directly from the environment and its latest state is
normalised locally with the exact checkpoint statistics.  Each completed
episode is appended to JSONL immediately so an interrupted audit can resume.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image

import eval_libero_action_cot_pruning as base_eval
from openpi.models import model as model_lib
from openpi.shared import normalize as _normalize
import train_branched_action_corrector as corrector_lib


MODES = ("full_h4", "stale_h10", "corrector_h10", "matched_diff_h10")
ACTION_DIM = 7
MODEL_ACTION_DIM = 32
ANCHOR_STEPS = 4
CORRECTED_STEPS = 6
SMALL_IMAGE_SIZE = 64
MATCHED_DIFF_ALPHA = 1.0
MATCHED_DIFF_TRUST_REGION_L2 = 0.025
CORRECTOR_MODES = ("corrector_h10", "matched_diff_h10")


@dataclasses.dataclass(frozen=True)
class CorrectorCache:
    anchor_images: np.ndarray
    anchor_state: np.ndarray
    cached_plan_tokens: np.ndarray
    cached_iar: np.ndarray
    intended_prefix: np.ndarray
    intended_valid: np.ndarray
    transported_ear: np.ndarray
    base_actions: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--trial-start", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument(
        "--matched-diff-min-control-step",
        type=int,
        default=0,
        help=(
            "For matched_diff_h10, retain stale H6 before this many post-wait "
            "control steps and enable the matched-difference correction afterwards."
        ),
    )
    parser.add_argument(
        "--matched-diff-max-response-l2",
        type=float,
        default=None,
        help=(
            "If set, retain stale H6 when the matched current-minus-no-current "
            "mean response L2 exceeds this deployable confidence threshold."
        ),
    )
    parser.add_argument(
        "--matched-diff-max-control-step",
        type=int,
        default=None,
        help=(
            "If set, retain stale H6 at and after this post-wait control step. "
            "Together with the minimum this can isolate a single recovery event."
        ),
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--corrector-summary", default=None)
    parser.add_argument("--corrector-params", default=None)
    parser.add_argument("--norm-stats-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.task_start < 0 or args.max_tasks <= 0:
        raise ValueError("task-start must be non-negative and max-tasks must be positive.")
    if args.trial_start < 0 or args.num_trials <= 0:
        raise ValueError("trial-start must be non-negative and num-trials must be positive.")
    if args.num_steps_wait < 0 or args.resize_size <= 0:
        raise ValueError("num-steps-wait must be non-negative and resize-size must be positive.")
    if args.action_cot_denoising_steps <= 0:
        raise ValueError("action-cot-denoising-steps must be positive.")
    if args.matched_diff_min_control_step < 0:
        raise ValueError("matched-diff-min-control-step must be non-negative.")
    if (
        args.matched_diff_max_response_l2 is not None
        and args.matched_diff_max_response_l2 <= 0
    ):
        raise ValueError("matched-diff-max-response-l2 must be positive when set.")
    if (
        args.matched_diff_max_control_step is not None
        and args.matched_diff_max_control_step
        <= args.matched_diff_min_control_step
    ):
        raise ValueError(
            "matched-diff-max-control-step must exceed matched-diff-min-control-step."
        )
    if any(mode in args.modes for mode in CORRECTOR_MODES):
        if args.corrector_summary is None:
            raise ValueError("Corrector modes require --corrector-summary.")
        if args.norm_stats_dir is None:
            raise ValueError("Corrector modes require --norm-stats-dir.")


def _small_images(element: dict[str, Any]) -> np.ndarray:
    images = []
    for key in ("observation/image", "observation/wrist_image"):
        values = np.asarray(element[key])
        if values.dtype != np.uint8:
            values = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        resized = Image.fromarray(values).resize(
            (SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
        images.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(images)


def _normalise_state(raw_state: Any, norm_stats: dict[str, Any]) -> np.ndarray:
    if "state" not in norm_stats:
        raise KeyError("The supplied norm stats do not contain a state entry.")
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    stats = norm_stats["state"]
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1)
    if state.size > mean.size or state.size > std.size:
        raise ValueError(
            f"Raw state has {state.size} dims but norm stats contain mean={mean.size}, std={std.size}."
        )
    normalised = (state - mean[: state.size]) / (std[: state.size] + 1e-6)
    if normalised.size > MODEL_ACTION_DIM:
        raise ValueError(f"Normalised state exceeds {MODEL_ACTION_DIM} dims: {normalised.shape}.")
    return np.pad(normalised, (0, MODEL_ACTION_DIM - normalised.size)).astype(np.float32)


class DirectCorrector:
    def __init__(
        self,
        summary_path: str,
        params_path: str | None,
        *,
        run_name: str = "current",
    ) -> None:
        if run_name not in {"current", "no_current"}:
            raise ValueError(f"Unsupported direct corrector run_name={run_name!r}.")
        summary_file = pathlib.Path(summary_path)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        args = dict(summary["args"])
        if args.get("mode") != "direct":
            raise ValueError("This runner intentionally supports only direct branched correctors.")
        if int(args.get("age", -1)) != ANCHOR_STEPS:
            raise ValueError(f"Corrector age must be {ANCHOR_STEPS}; got {args.get('age')}.")
        if int(args.get("rollout_horizon", -1)) != CORRECTED_STEPS:
            raise ValueError(
                f"Corrector rollout_horizon must be {CORRECTED_STEPS}; "
                f"got {args.get('rollout_horizon')}."
            )
        range_calibration = summary["range_calibration"]
        module = corrector_lib.BranchedActionCorrector(
            mode="direct",
            image_views=2,
            image_channels=3,
            state_dim=MODEL_ACTION_DIM,
            plan_dim=MODEL_ACTION_DIM,
            iar_dim=1024,
            env_action_dim=ACTION_DIM,
            max_executed_steps=ANCHOR_STEPS,
            ear_horizon=15,
            rollout_horizon=CORRECTED_STEPS,
            hidden_dim=int(args["hidden_dim"]),
            action_residual_scale=tuple(range_calibration["action_residual_scale"]),
            plan_residual_scale=tuple(range_calibration["plan_residual_scale"]),
            gripper_logit_scale=float(args["gripper_logit_scale"]),
            rngs=nnx.Rngs(int(args.get("seed", 0))),
            direct_head=str(args.get("direct_head", "global")),
            token_layers=int(args.get("token_layers", 4)),
            token_heads=int(args.get("token_heads", 4)),
        )
        selected_params = params_path or summary["train"][run_name]["params_path"]
        loaded = model_lib.convert_str_keys_to_int(
            model_lib.restore_params(selected_params, dtype=jnp.float32)
        )
        expected_name = f"branched_action_corrector_direct_{run_name}"
        if expected_name in loaded:
            loaded = loaded[expected_name]
        elif len(loaded) == 1:
            loaded = next(iter(loaded.values()))
        else:
            raise KeyError(
                f"Could not identify {expected_name!r} in corrector checkpoint keys {sorted(loaded)}."
            )
        graphdef, state = nnx.split(module)
        state.replace_by_pure_dict(loaded)
        self._graphdef = graphdef
        self._params = state
        self.summary_path = str(summary_file.resolve())
        self.params_path = str(pathlib.Path(selected_params).resolve())
        self.run_name = run_name

        @jax.jit
        def predict(current_params: nnx.State, batch: dict[str, jax.Array]):
            candidate = nnx.merge(self._graphdef, current_params)
            return candidate(
                batch["anchor_images"],
                batch["current_images"],
                batch["anchor_state"],
                batch["current_state"],
                batch["cached_plan_tokens"],
                batch["cached_iar"],
                batch["intended_prefix"],
                batch["intended_valid"],
                batch["transported_ear"],
                batch["base_actions"],
                batch["empty_features"],
                batch["empty_features"],
            )

        self._predict = predict

    def __call__(
        self,
        cache: CorrectorCache,
        current_images: np.ndarray,
        current_state: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        batch = {
            "anchor_images": jnp.asarray(cache.anchor_images[None], dtype=jnp.float32) / 255.0,
            "current_images": jnp.asarray(current_images[None], dtype=jnp.float32) / 255.0,
            "anchor_state": jnp.asarray(cache.anchor_state[None], dtype=jnp.float32),
            "current_state": jnp.asarray(current_state[None], dtype=jnp.float32),
            "cached_plan_tokens": jnp.asarray(cache.cached_plan_tokens[None], dtype=jnp.float32),
            "cached_iar": jnp.asarray(cache.cached_iar[None], dtype=jnp.float32),
            "intended_prefix": jnp.asarray(cache.intended_prefix[None], dtype=jnp.float32),
            "intended_valid": jnp.asarray(cache.intended_valid[None], dtype=jnp.bool_),
            "transported_ear": jnp.asarray(cache.transported_ear[None], dtype=jnp.float32),
            "base_actions": jnp.asarray(cache.base_actions[None], dtype=jnp.float32),
            "empty_features": jnp.zeros((1, 0), dtype=jnp.float32),
        }
        started = time.perf_counter()
        output = self._predict(self._params, batch)
        actions = np.asarray(jax.device_get(output.actions[0]), dtype=np.float32)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return actions, elapsed_ms


class MatchedDifferenceCorrector:
    """Conservative stale-anchored response from matched direct checkpoints."""

    def __init__(
        self,
        summary_path: str,
        current_params_path: str | None,
    ) -> None:
        self.current = DirectCorrector(
            summary_path,
            current_params_path,
            run_name="current",
        )
        self.no_current = DirectCorrector(
            summary_path,
            None,
            run_name="no_current",
        )
        self.summary_path = self.current.summary_path
        self.current_params_path = self.current.params_path
        self.no_current_params_path = self.no_current.params_path
        self.alpha = MATCHED_DIFF_ALPHA
        self.trust_region_l2 = MATCHED_DIFF_TRUST_REGION_L2

    def __call__(
        self,
        cache: CorrectorCache,
        current_images: np.ndarray,
        current_state: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        started = time.perf_counter()
        current_actions, current_ms = self.current(
            cache,
            current_images,
            current_state,
        )
        no_current_actions, no_current_ms = self.no_current(
            cache,
            cache.anchor_images,
            cache.anchor_state,
        )
        response = self.alpha * (
            current_actions[..., :6] - no_current_actions[..., :6]
        )
        response_l2 = np.linalg.norm(response, axis=-1)
        clip_scale = np.minimum(
            1.0,
            self.trust_region_l2 / np.maximum(response_l2, 1e-8),
        )
        clipped_response = response * clip_scale[:, None]
        actions = np.array(cache.base_actions, dtype=np.float32, copy=True)
        actions[..., :6] += clipped_response
        # Gripper is deliberately inherited from stale, irrespective of both
        # corrector outputs.
        actions[..., 6] = cache.base_actions[..., 6]
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return actions, {
            "elapsed_ms": float(elapsed_ms),
            "current_corrector_ms": float(current_ms),
            "no_current_corrector_ms": float(no_current_ms),
            "response_l2_mean": float(np.mean(response_l2)),
            "response_l2_max": float(np.max(response_l2)),
            "applied_response_l2_mean": float(
                np.mean(np.linalg.norm(clipped_response, axis=-1))
            ),
            "clip_fraction": float(np.mean(response_l2 > self.trust_region_l2)),
        }


def _full_request(
    client,
    element: dict[str, Any],
    *,
    seed: int,
    denoising_steps: int,
    export_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = dict(element)
    request["action_cot_denoising_steps"] = np.asarray(denoising_steps, dtype=np.int32)
    if export_cache:
        request["export_acot_cache"] = np.ones((), dtype=np.bool_)
    result, wall_ms, policy_ms, server_ms, stage_timing = base_eval._infer(
        client,
        request,
        seed=seed,
    )
    return result, {
        "wall_ms": wall_ms,
        "policy_ms": policy_ms,
        "server_ms": server_ms,
        "stage_timing": stage_timing,
    }


def _make_cache(
    element: dict[str, Any],
    result: dict[str, Any],
    norm_stats: dict[str, Any],
) -> CorrectorCache:
    required = (
        "actions",
        "execution_horizon_state_normalized",
        "execution_horizon_coarse_actions_normalized",
        "acot_iar_tokens",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"Policy response is missing corrector cache fields: {missing}.")

    actions = np.asarray(result["actions"], dtype=np.float32)
    cached_ear = np.asarray(
        result["execution_horizon_coarse_actions_normalized"],
        dtype=np.float32,
    )
    cached_iar = np.asarray(result["acot_iar_tokens"], dtype=np.float32)
    anchor_state = np.asarray(
        result["execution_horizon_state_normalized"],
        dtype=np.float32,
    ).reshape(-1)
    if actions.shape[0] < ANCHOR_STEPS + CORRECTED_STEPS or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected at least a [10, 7] action chunk, got {actions.shape}.")
    if cached_ear.shape != (15, MODEL_ACTION_DIM):
        raise ValueError(f"Expected cached EAR [15, 32], got {cached_ear.shape}.")
    if cached_iar.shape != (18, 1024):
        raise ValueError(f"Expected cached IAR [18, 1024], got {cached_iar.shape}.")
    if anchor_state.shape != (MODEL_ACTION_DIM,):
        raise ValueError(f"Expected normalised anchor state [32], got {anchor_state.shape}.")
    local_anchor_state = _normalise_state(element["observation/state"], norm_stats)
    normalisation_error = float(np.max(np.abs(local_anchor_state - anchor_state)))
    if normalisation_error > 1e-5:
        raise ValueError(
            "Local state normalisation does not match the policy server: "
            f"max_abs_error={normalisation_error:.8g}."
        )

    actions = actions[: ANCHOR_STEPS + CORRECTED_STEPS, :ACTION_DIM]
    padded_actions = np.zeros((actions.shape[0], MODEL_ACTION_DIM), dtype=np.float32)
    padded_actions[:, :ACTION_DIM] = actions
    transported_ear = corrector_lib._transport_ear(
        cached_ear[None, :, :ACTION_DIM],
        ANCHOR_STEPS / 2,
    )[0]
    return CorrectorCache(
        anchor_images=_small_images(element),
        anchor_state=anchor_state,
        cached_plan_tokens=np.concatenate([cached_ear, padded_actions], axis=0),
        cached_iar=cached_iar,
        intended_prefix=actions[:ANCHOR_STEPS],
        intended_valid=np.ones((ANCHOR_STEPS,), dtype=np.bool_),
        transported_ear=np.asarray(transported_ear, dtype=np.float32),
        base_actions=actions[ANCHOR_STEPS : ANCHOR_STEPS + CORRECTED_STEPS],
    )


def _action_metrics(actions: list[np.ndarray]) -> dict[str, float]:
    if not actions:
        return {
            "action_delta_l2": float("nan"),
            "action_jerk_l2": float("nan"),
            "gripper_flip_count": 0.0,
            "continuous_saturation_fraction": float("nan"),
        }
    values = np.asarray(actions, dtype=np.float64)
    deltas = np.diff(values[:, :6], axis=0)
    jerks = np.diff(deltas, axis=0)
    gripper = values[:, 6] >= 0
    return {
        "action_delta_l2": float(np.mean(np.linalg.norm(deltas, axis=-1))) if len(deltas) else 0.0,
        "action_jerk_l2": float(np.mean(np.linalg.norm(jerks, axis=-1))) if len(jerks) else 0.0,
        "gripper_flip_count": float(np.sum(gripper[1:] != gripper[:-1])),
        "continuous_saturation_fraction": float(np.mean(np.abs(values[:, :6]) >= 0.999)),
    }


def _run_episode(
    *,
    mode: str,
    task: Any,
    initial_state: Any,
    task_id: int,
    episode_idx: int,
    args: argparse.Namespace,
    client: Any,
    corrector: DirectCorrector | None,
    matched_diff_corrector: MatchedDifferenceCorrector | None,
    norm_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    env, task_description = base_eval._get_libero_env(
        task,
        base_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    started = time.monotonic()
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        environment_horizon = base_eval._env_horizon(env)
        step_limit = base_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        if environment_horizon is not None:
            step_limit = min(step_limit, environment_horizon)

        action_queue: collections.deque[np.ndarray] = collections.deque()
        cache: CorrectorCache | None = None
        actions_executed: list[np.ndarray] = []
        policy_wall_ms: list[float] = []
        policy_infer_ms: list[float] = []
        policy_server_ms: list[float] = []
        corrector_ms: list[float] = []
        matched_diff_current_ms: list[float] = []
        matched_diff_no_current_ms: list[float] = []
        matched_diff_response_l2: list[float] = []
        matched_diff_response_l2_max: list[float] = []
        matched_diff_applied_l2: list[float] = []
        matched_diff_clip_fraction: list[float] = []
        matched_diff_gate_applied_calls = 0
        matched_diff_gate_skipped_calls = 0
        matched_diff_gate_response_rejected_calls = 0
        correction_l2: list[float] = []
        correction_boundary_l2: list[float] = []
        correction_gripper_changes: list[float] = []
        total_return = 0.0
        step = 0
        done = False
        termination_reason = ""

        while step < step_limit:
            if step < args.num_steps_wait:
                try:
                    observation, reward, done, _ = env.step(base_eval.LIBERO_DUMMY_ACTION)
                except Exception as exc:
                    if not base_eval._is_terminated_episode_error(exc):
                        raise
                    done = base_eval._env_success(env)
                    termination_reason = "environment_terminated_during_wait"
                    break
                total_return += float(reward)
                step += 1
                if done:
                    break
                continue

            if not action_queue:
                if (
                    mode == "matched_diff_h10"
                    and cache is not None
                    and (
                        step - args.num_steps_wait
                        < args.matched_diff_min_control_step
                        or (
                            args.matched_diff_max_control_step is not None
                            and step - args.num_steps_wait
                            >= args.matched_diff_max_control_step
                        )
                    )
                ):
                    action_queue.extend(cache.base_actions)
                    cache = None
                    matched_diff_gate_skipped_calls += 1
                elif mode in CORRECTOR_MODES and cache is not None:
                    if norm_stats is None:
                        raise RuntimeError("Corrector runtime was not initialised.")
                    element = base_eval._observation_to_policy_input(
                        observation,
                        task_description,
                        args.resize_size,
                    )
                    current_images = _small_images(element)
                    current_state = _normalise_state(element["observation/state"], norm_stats)
                    if mode == "matched_diff_h10":
                        if matched_diff_corrector is None:
                            raise RuntimeError("Matched-difference runtime was not initialised.")
                        corrected_actions, matched_metrics = matched_diff_corrector(
                            cache,
                            current_images,
                            current_state,
                        )
                        elapsed_ms = matched_metrics["elapsed_ms"]
                        matched_diff_current_ms.append(
                            matched_metrics["current_corrector_ms"]
                        )
                        matched_diff_no_current_ms.append(
                            matched_metrics["no_current_corrector_ms"]
                        )
                        matched_diff_response_l2.append(
                            matched_metrics["response_l2_mean"]
                        )
                        matched_diff_response_l2_max.append(
                            matched_metrics["response_l2_max"]
                        )
                        matched_diff_applied_l2.append(
                            matched_metrics["applied_response_l2_mean"]
                        )
                        matched_diff_clip_fraction.append(
                            matched_metrics["clip_fraction"]
                        )
                        if (
                            args.matched_diff_max_response_l2 is not None
                            and matched_metrics["response_l2_mean"]
                            > args.matched_diff_max_response_l2
                        ):
                            corrected_actions = np.array(
                                cache.base_actions,
                                dtype=np.float32,
                                copy=True,
                            )
                            matched_diff_gate_response_rejected_calls += 1
                        else:
                            matched_diff_gate_applied_calls += 1
                    else:
                        if corrector is None:
                            raise RuntimeError("Direct corrector runtime was not initialised.")
                        corrected_actions, elapsed_ms = corrector(
                            cache,
                            current_images,
                            current_state,
                        )
                    if corrected_actions.shape != (CORRECTED_STEPS, ACTION_DIM):
                        raise ValueError(
                            f"Corrector returned {corrected_actions.shape}; expected {(CORRECTED_STEPS, ACTION_DIM)}."
                        )
                    corrector_ms.append(elapsed_ms)
                    correction_l2.append(
                        float(
                            np.mean(
                                np.linalg.norm(
                                    corrected_actions[:, :6] - cache.base_actions[:, :6],
                                    axis=-1,
                                )
                            )
                        )
                    )
                    correction_gripper_changes.append(
                        float(
                            np.sum(
                                (corrected_actions[:, 6] >= 0)
                                != (cache.base_actions[:, 6] >= 0)
                            )
                        )
                    )
                    if actions_executed:
                        correction_boundary_l2.append(
                            float(
                                np.linalg.norm(
                                    corrected_actions[0, :6] - actions_executed[-1][:6]
                                )
                            )
                        )
                    action_queue.extend(corrected_actions)
                    cache = None
                else:
                    element = base_eval._observation_to_policy_input(
                        observation,
                        task_description,
                        args.resize_size,
                    )
                    policy_seed = args.seed + task_id * 1_000_000 + episode_idx * 10_000 + step
                    matched_diff_midpoint_control_step = (
                        step - args.num_steps_wait + ANCHOR_STEPS
                    )
                    matched_diff_in_window = (
                        mode == "matched_diff_h10"
                        and matched_diff_midpoint_control_step
                        >= args.matched_diff_min_control_step
                        and (
                            args.matched_diff_max_control_step is None
                            or matched_diff_midpoint_control_step
                            < args.matched_diff_max_control_step
                        )
                    )
                    export_cache = (
                        mode == "corrector_h10" or matched_diff_in_window
                    )
                    result, timing = _full_request(
                        client,
                        element,
                        seed=policy_seed,
                        denoising_steps=args.action_cot_denoising_steps,
                        export_cache=export_cache,
                    )
                    policy_wall_ms.append(float(timing["wall_ms"]))
                    policy_infer_ms.append(float(timing["policy_ms"]))
                    policy_server_ms.append(float(timing["server_ms"]))
                    action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if mode == "full_h4":
                        if action_chunk.shape[0] < ANCHOR_STEPS:
                            raise ValueError(f"Policy returned too few actions: {action_chunk.shape}.")
                        action_queue.extend(action_chunk[:ANCHOR_STEPS, :ACTION_DIM])
                    elif mode == "stale_h10" or (
                        mode == "matched_diff_h10" and not export_cache
                    ):
                        required = ANCHOR_STEPS + CORRECTED_STEPS
                        if action_chunk.shape[0] < required:
                            raise ValueError(f"Policy returned too few actions: {action_chunk.shape}.")
                        action_queue.extend(action_chunk[:required, :ACTION_DIM])
                        if mode == "matched_diff_h10":
                            matched_diff_gate_skipped_calls += 1
                    else:
                        if norm_stats is None:
                            raise RuntimeError("Corrector mode requires norm stats.")
                        cache = _make_cache(element, result, norm_stats)
                        action_queue.extend(cache.intended_prefix)

            action = np.asarray(action_queue.popleft(), dtype=np.float32)
            actions_executed.append(action)
            try:
                observation, reward, done, _ = env.step(action.tolist())
            except Exception as exc:
                if not base_eval._is_terminated_episode_error(exc):
                    raise
                done = base_eval._env_success(env)
                termination_reason = "environment_terminated_before_action"
                break
            total_return += float(reward)
            step += 1
            if done:
                termination_reason = "success"
                break

        if not done:
            done = base_eval._env_success(env)
        if done and not termination_reason:
            termination_reason = "success"
        elif not done and not termination_reason:
            termination_reason = "step_limit"

        action_quality = _action_metrics(actions_executed)
        control_steps = max(0, step - args.num_steps_wait)
        return {
            "schema_version": 1,
            "mode": mode,
            "task_suite_name": args.task_suite_name,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "task_description": str(task_description),
            "seed": args.seed,
            "success": bool(done),
            "termination_reason": termination_reason,
            "total_return": total_return,
            "environment_steps": step,
            "control_steps": control_steps,
            "policy_calls": len(policy_wall_ms),
            "corrector_calls": len(corrector_ms),
            "policy_wall_ms_total": float(np.sum(policy_wall_ms)),
            "policy_wall_ms_mean": float(np.mean(policy_wall_ms)) if policy_wall_ms else float("nan"),
            "policy_infer_ms_total": float(np.sum(policy_infer_ms)),
            "policy_infer_ms_mean": float(np.mean(policy_infer_ms)) if policy_infer_ms else float("nan"),
            "policy_server_ms_total": float(np.sum(policy_server_ms)),
            "corrector_ms_total": float(np.sum(corrector_ms)),
            "corrector_ms_mean": float(np.mean(corrector_ms)) if corrector_ms else float("nan"),
            "matched_diff_current_corrector_ms_total": float(
                np.sum(matched_diff_current_ms)
            ),
            "matched_diff_current_corrector_ms_mean": (
                float(np.mean(matched_diff_current_ms))
                if matched_diff_current_ms
                else float("nan")
            ),
            "matched_diff_no_current_corrector_ms_total": float(
                np.sum(matched_diff_no_current_ms)
            ),
            "matched_diff_no_current_corrector_ms_mean": (
                float(np.mean(matched_diff_no_current_ms))
                if matched_diff_no_current_ms
                else float("nan")
            ),
            "matched_diff_dual_corrector_ms_total": (
                float(np.sum(corrector_ms))
                if mode == "matched_diff_h10"
                else float("nan")
            ),
            "matched_diff_dual_corrector_ms_mean": (
                float(np.mean(corrector_ms))
                if mode == "matched_diff_h10" and corrector_ms
                else float("nan")
            ),
            "matched_diff_response_l2_mean": (
                float(np.mean(matched_diff_response_l2))
                if matched_diff_response_l2
                else float("nan")
            ),
            "matched_diff_response_l2_max_mean": (
                float(np.mean(matched_diff_response_l2_max))
                if matched_diff_response_l2_max
                else float("nan")
            ),
            "matched_diff_applied_response_l2_mean": (
                float(np.mean(matched_diff_applied_l2))
                if matched_diff_applied_l2
                else float("nan")
            ),
            "matched_diff_clip_fraction_mean": (
                float(np.mean(matched_diff_clip_fraction))
                if matched_diff_clip_fraction
                else float("nan")
            ),
            "matched_diff_gate_applied_calls": matched_diff_gate_applied_calls,
            "matched_diff_gate_skipped_calls": matched_diff_gate_skipped_calls,
            "matched_diff_gate_response_rejected_calls": (
                matched_diff_gate_response_rejected_calls
            ),
            "matched_diff_gate_coverage": (
                matched_diff_gate_applied_calls
                / (
                    matched_diff_gate_applied_calls
                    + matched_diff_gate_skipped_calls
                    + matched_diff_gate_response_rejected_calls
                )
                if matched_diff_gate_applied_calls
                + matched_diff_gate_skipped_calls
                + matched_diff_gate_response_rejected_calls
                else float("nan")
            ),
            "correction_l2_mean": float(np.mean(correction_l2)) if correction_l2 else float("nan"),
            "correction_boundary_l2_mean": (
                float(np.mean(correction_boundary_l2)) if correction_boundary_l2 else float("nan")
            ),
            "correction_gripper_changes_total": float(np.sum(correction_gripper_changes)),
            "elapsed_seconds": time.monotonic() - started,
            **action_quality,
        }
    finally:
        base_eval._safe_close_env(env)


def _read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning("Ignoring malformed JSONL line %s in %s.", line_number, path)
    return rows


def _finite_mean(rows: list[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows if np.isfinite(float(row.get(name, float("nan"))))]
    return float(np.mean(values)) if values else float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        successes = sum(bool(row["success"]) for row in selected)
        by_mode[mode] = {
            "episodes": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected),
            "mean_control_steps": _finite_mean(selected, "control_steps"),
            "mean_policy_calls": _finite_mean(selected, "policy_calls"),
            "mean_corrector_calls": _finite_mean(selected, "corrector_calls"),
            "mean_policy_wall_ms_total": _finite_mean(selected, "policy_wall_ms_total"),
            "mean_policy_infer_ms_total": _finite_mean(selected, "policy_infer_ms_total"),
            "mean_corrector_ms_total": _finite_mean(selected, "corrector_ms_total"),
            "mean_matched_diff_current_corrector_ms_total": _finite_mean(
                selected,
                "matched_diff_current_corrector_ms_total",
            ),
            "mean_matched_diff_no_current_corrector_ms_total": _finite_mean(
                selected,
                "matched_diff_no_current_corrector_ms_total",
            ),
            "mean_matched_diff_dual_corrector_ms_total": _finite_mean(
                selected,
                "matched_diff_dual_corrector_ms_total",
            ),
            "mean_matched_diff_response_l2": _finite_mean(
                selected,
                "matched_diff_response_l2_mean",
            ),
            "mean_matched_diff_applied_response_l2": _finite_mean(
                selected,
                "matched_diff_applied_response_l2_mean",
            ),
            "mean_matched_diff_clip_fraction": _finite_mean(
                selected,
                "matched_diff_clip_fraction_mean",
            ),
            "mean_matched_diff_gate_applied_calls": _finite_mean(
                selected,
                "matched_diff_gate_applied_calls",
            ),
            "mean_matched_diff_gate_skipped_calls": _finite_mean(
                selected,
                "matched_diff_gate_skipped_calls",
            ),
            "mean_matched_diff_gate_response_rejected_calls": _finite_mean(
                selected,
                "matched_diff_gate_response_rejected_calls",
            ),
            "mean_matched_diff_gate_coverage": _finite_mean(
                selected,
                "matched_diff_gate_coverage",
            ),
            "mean_action_delta_l2": _finite_mean(selected, "action_delta_l2"),
            "mean_action_jerk_l2": _finite_mean(selected, "action_jerk_l2"),
            "mean_gripper_flip_count": _finite_mean(selected, "gripper_flip_count"),
            "mean_continuous_saturation_fraction": _finite_mean(
                selected,
                "continuous_saturation_fraction",
            ),
            "mean_correction_l2": _finite_mean(selected, "correction_l2_mean"),
            "mean_correction_boundary_l2": _finite_mean(
                selected,
                "correction_boundary_l2_mean",
            ),
        }

    lookup = {
        (str(row["mode"]), int(row["task_id"]), int(row["episode_idx"])): row
        for row in rows
    }
    paired = {}
    for candidate in CORRECTOR_MODES:
        for reference in ("full_h4", "stale_h10"):
            keys = sorted(
                (task_id, episode_idx)
                for mode, task_id, episode_idx in lookup
                if mode == reference
                and (candidate, task_id, episode_idx) in lookup
            )
            if not keys:
                continue
            rescues = 0
            regressions = 0
            agreements_success = 0
            agreements_failure = 0
            for task_id, episode_idx in keys:
                reference_success = bool(lookup[(reference, task_id, episode_idx)]["success"])
                candidate_success = bool(lookup[(candidate, task_id, episode_idx)]["success"])
                if candidate_success and not reference_success:
                    rescues += 1
                elif reference_success and not candidate_success:
                    regressions += 1
                elif reference_success:
                    agreements_success += 1
                else:
                    agreements_failure += 1
            paired[f"{candidate}_vs_{reference}"] = {
                "paired_episodes": len(keys),
                "rescues": rescues,
                "regressions": regressions,
                "net_rescues": rescues - regressions,
                "both_success": agreements_success,
                "both_failure": agreements_failure,
                "paired_success_rate_delta": (rescues - regressions) / len(keys),
            }
    return {"by_mode": by_mode, "paired": paired}


def _corrector_metadata(
    corrector: DirectCorrector | None,
    matched_diff_corrector: MatchedDifferenceCorrector | None,
) -> dict[str, Any] | None:
    if corrector is None and matched_diff_corrector is None:
        return None
    return {
        "direct": (
            {
                "summary_path": corrector.summary_path,
                "params_path": corrector.params_path,
            }
            if corrector is not None
            else None
        ),
        "matched_difference": (
            {
                "summary_path": matched_diff_corrector.summary_path,
                "current_params_path": matched_diff_corrector.current_params_path,
                "no_current_params_path": matched_diff_corrector.no_current_params_path,
                "alpha": matched_diff_corrector.alpha,
                "per_token_continuous_l2_trust_region": (
                    matched_diff_corrector.trust_region_l2
                ),
                "gripper": "stale",
            }
            if matched_diff_corrector is not None
            else None
        ),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_args(args)
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    rows = _read_rows(episodes_path)
    completed = {
        (str(row["mode"]), int(row["task_id"]), int(row["episode_idx"]))
        for row in rows
    }

    norm_stats = (
        _normalize.load(args.norm_stats_dir)
        if any(mode in args.modes for mode in CORRECTOR_MODES)
        else None
    )
    corrector = (
        DirectCorrector(args.corrector_summary, args.corrector_params)
        if "corrector_h10" in args.modes
        else None
    )
    matched_diff_corrector = (
        MatchedDifferenceCorrector(
            args.corrector_summary,
            args.corrector_params,
        )
        if "matched_diff_h10" in args.modes
        else None
    )
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = base_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_end = min(task_suite.n_tasks, args.task_start + args.max_tasks)

    with episodes_path.open("a", encoding="utf-8") as output_file:
        for mode in args.modes:
            for task_id in range(args.task_start, task_end):
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                trial_end = args.trial_start + args.num_trials
                if trial_end > len(initial_states):
                    raise ValueError(
                        f"Requested trials [{args.trial_start}, {trial_end}) but task {task_id} "
                        f"contains only {len(initial_states)} initial states."
                    )
                for episode_idx in range(args.trial_start, trial_end):
                    key = (mode, task_id, episode_idx)
                    if key in completed:
                        logging.info("Skipping completed mode=%s task=%s episode=%s", *key)
                        continue
                    logging.info("Running mode=%s task=%s episode=%s", *key)
                    row = _run_episode(
                        mode=mode,
                        task=task,
                        initial_state=initial_states[episode_idx],
                        task_id=task_id,
                        episode_idx=episode_idx,
                        args=args,
                        client=client,
                        corrector=corrector,
                        matched_diff_corrector=matched_diff_corrector,
                        norm_stats=norm_stats,
                    )
                    output_file.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
                    output_file.flush()
                    rows.append(row)
                    completed.add(key)
                    summary = {
                        "schema_version": 1,
                        "status": "running",
                        "protocol": {
                            "modes": args.modes,
                            "task_suite_name": args.task_suite_name,
                            "task_start": args.task_start,
                            "max_tasks": args.max_tasks,
                            "trial_start": args.trial_start,
                            "num_trials": args.num_trials,
                            "seed": args.seed,
                            "denoising_steps": args.action_cot_denoising_steps,
                            "matched_diff_min_control_step": (
                                args.matched_diff_min_control_step
                            ),
                            "matched_diff_max_response_l2": (
                                args.matched_diff_max_response_l2
                            ),
                            "matched_diff_max_control_step": (
                                args.matched_diff_max_control_step
                            ),
                            "corrector_schedule": (
                                "fresh ACoT first 4 actions; direct mode replaces actions 4:10; "
                                "matched-difference mode retains stale and adds the clipped "
                                "current-minus-independent-no-current response"
                            ),
                            "held_out_note": (
                                "The canonical corrector dataset contains collector episodes 0-9; "
                                "trial_start>=10 selects unseen LIBERO initial-state IDs."
                            ),
                        },
                        "corrector": _corrector_metadata(
                            corrector,
                            matched_diff_corrector,
                        ),
                        "completed_episode_rows": len(rows),
                        **_aggregate(rows),
                    }
                    (output_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2, allow_nan=True),
                        encoding="utf-8",
                    )

    expected = len(args.modes) * (task_end - args.task_start) * args.num_trials
    summary = {
        "schema_version": 1,
        "status": "complete" if len(completed) >= expected else "partial",
        "protocol": {
            "modes": args.modes,
            "task_suite_name": args.task_suite_name,
            "task_start": args.task_start,
            "max_tasks": args.max_tasks,
            "trial_start": args.trial_start,
            "num_trials": args.num_trials,
            "seed": args.seed,
            "denoising_steps": args.action_cot_denoising_steps,
            "matched_diff_min_control_step": args.matched_diff_min_control_step,
            "matched_diff_max_response_l2": args.matched_diff_max_response_l2,
            "matched_diff_max_control_step": args.matched_diff_max_control_step,
            "corrector_schedule": (
                "fresh ACoT first 4 actions; direct mode replaces actions 4:10; "
                "matched-difference mode retains stale and adds the clipped "
                "current-minus-independent-no-current response"
            ),
            "held_out_note": (
                "The canonical corrector dataset contains collector episodes 0-9; "
                "trial_start>=10 selects unseen LIBERO initial-state IDs."
            ),
        },
        "corrector": _corrector_metadata(
            corrector,
            matched_diff_corrector,
        ),
        "completed_episode_rows": len(rows),
        **_aggregate(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    logging.info("Wrote %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
