# ruff: noqa: SLF001

"""Train a continuous midpoint corrector on canonical branched Action-CoT data.

This is a deliberately aggressive offline go/no-go experiment.  Every sample
starts from the same cached ACoT plan after four intended actions, but its
latest observation comes from one of the canonical physical branches.  The
target is the fresh teacher's next action sub-chunk.

Two matched paths are supported:

* ``direct`` predicts an observation-conditioned residual around the cached
  final-action sub-chunk.  It is an A2C2-style capacity upper bound.
* ``plan`` first writes the observation-conditioned residual into the full
  explicit EAR and lets a plan-only decoder produce the action residual.

Neither path receives the synthetic branch id nor the actually injected fault
actions.  The only action history input is the controller-intended cached
prefix, which is available at deployment.  Roots are split by episode before
branches are flattened.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import pathlib
import time
from typing import Any, Literal, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import train_branched_effective_progress as progress_probe
import tyro

from openpi.action_cot import branched_dataset
from openpi.models import transported_action_cot

Mode = Literal["direct", "plan"]


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    mode: str = "direct"
    seed: int = 7
    split_seed: int = 7
    train_steps: int = 700
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    log_interval: int = 50
    age: int = 4
    rollout_horizon: int = 4
    coarse_time_stride: int = 2
    hidden_dim: int = 128
    action_huber_delta: float = 0.05
    plan_huber_delta: float = 0.1
    action_gripper_weight: float = 0.05
    plan_loss_weight: float = 0.5
    plan_gripper_weight: float = 0.05
    plan_velocity_weight: float = 0.1
    output_margin: float = 1.25
    minimum_action_residual_scale: float = 0.05
    minimum_plan_residual_scale: float = 0.05
    gripper_logit_scale: float = 8.0
    profile_warmup: int = 50
    profile_iterations: int = 500
    refresh_interval: int = 4
    full_acot_reference_ms: float = 95.844
    max_parameters: int = 1_000_000
    train_no_current: bool = True
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.mode not in {"direct", "plan"}:
        raise ValueError("mode must be either 'direct' or 'plan'.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seed and split_seed must be non-negative.")
    positive_integers = (
        args.train_steps,
        args.batch_size,
        args.eval_batch_size,
        args.log_interval,
        args.age,
        args.rollout_horizon,
        args.coarse_time_stride,
        args.hidden_dim,
        args.profile_iterations,
        args.refresh_interval,
        args.max_parameters,
    )
    if any(value <= 0 for value in positive_integers) or args.profile_warmup < 0:
        raise ValueError("Training, shape, profiling, and parameter sizes must be positive.")
    if args.refresh_interval <= 1:
        raise ValueError("refresh_interval must exceed one.")
    positive_scales = (
        args.learning_rate,
        args.gradient_clip_norm,
        args.action_huber_delta,
        args.plan_huber_delta,
        args.output_margin,
        args.minimum_action_residual_scale,
        args.minimum_plan_residual_scale,
        args.gripper_logit_scale,
        args.full_acot_reference_ms,
    )
    if any(value <= 0 for value in positive_scales) or args.weight_decay < 0:
        raise ValueError("Optimizer, loss, range, and latency scales must be positive.")
    if args.output_margin <= 1.0:
        raise ValueError("output_margin must exceed one.")
    nonnegative_weights = (
        args.action_gripper_weight,
        args.plan_loss_weight,
        args.plan_gripper_weight,
        args.plan_velocity_weight,
    )
    if any(value < 0 for value in nonnegative_weights):
        raise ValueError("Auxiliary loss weights must be non-negative.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if args.max_parameters > 1_000_000:
        raise ValueError("max_parameters may not exceed one million.")


def _linear_parameters(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


class CorrectorOutput(NamedTuple):
    actions: jax.Array
    action_gripper_logits: jax.Array
    transported_ear: jax.Array
    revised_ear: jax.Array
    plan_gripper_logits: jax.Array


class BranchedActionCorrector(nnx.Module):
    """Small shared encoder with either a direct or explicit-plan bottleneck."""

    def __init__(
        self,
        *,
        mode: Mode,
        image_views: int,
        image_channels: int,
        state_dim: int,
        plan_dim: int,
        iar_dim: int,
        env_action_dim: int,
        max_executed_steps: int,
        ear_horizon: int,
        rollout_horizon: int,
        hidden_dim: int,
        action_residual_scale: tuple[float, ...],
        plan_residual_scale: tuple[float, ...],
        gripper_logit_scale: float,
        rngs: nnx.Rngs,
        cnn_channels: tuple[int, ...] = (16, 32, 48),
    ) -> None:
        if mode not in {"direct", "plan"}:
            raise ValueError(f"Unsupported corrector mode: {mode!r}.")
        if env_action_dim != 7:
            raise ValueError("The canonical branched corrector currently requires seven environment action dims.")
        if len(action_residual_scale) != env_action_dim or len(plan_residual_scale) != env_action_dim:
            raise ValueError("Residual scales must match env_action_dim.")
        self.mode = mode
        self.image_views = image_views
        self.max_executed_steps = max_executed_steps
        self.ear_horizon = ear_horizon
        self.rollout_horizon = rollout_horizon
        self.hidden_dim = hidden_dim
        self.action_residual_scale = jnp.asarray(action_residual_scale, dtype=jnp.float32)
        self.plan_residual_scale = jnp.asarray(plan_residual_scale, dtype=jnp.float32)
        self.gripper_logit_scale = gripper_logit_scale

        image_convs = []
        in_channels = image_channels
        for out_channels, kernel_size in zip(cnn_channels, (5, 3, 3), strict=True):
            image_convs.append(progress_probe._Conv2D(in_channels, out_channels, kernel_size, rngs=rngs))
            in_channels = out_channels
        self.image_convs = image_convs
        image_summary = image_views * cnn_channels[-1]
        self.image_proj = nnx.Linear(4 * image_summary, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(3 * state_dim, hidden_dim, rngs=rngs)
        self.plan_token_proj = nnx.Linear(plan_dim, hidden_dim, rngs=rngs)
        self.plan_summary_proj = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.iar_proj = nnx.Linear(iar_dim, hidden_dim, rngs=rngs)
        self.prefix_proj = nnx.Linear(
            max_executed_steps * (env_action_dim + 1),
            hidden_dim,
            rngs=rngs,
        )
        self.age_proj = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.fusion = nnx.Linear(6 * hidden_dim, hidden_dim, rngs=rngs)
        self.hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        if mode == "direct":
            self.action_out = nnx.Linear(
                hidden_dim,
                rollout_horizon * env_action_dim,
                rngs=rngs,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        else:
            self.plan_out = nnx.Linear(
                hidden_dim,
                ear_horizon * env_action_dim,
                rngs=rngs,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
            self.plan_decoder = nnx.Linear(
                ear_horizon * env_action_dim,
                hidden_dim,
                rngs=rngs,
            )
            self.plan_decoder_hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.action_out = nnx.Linear(
                hidden_dim,
                rollout_horizon * env_action_dim,
                rngs=rngs,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )

    @staticmethod
    def _normalize_images(images: jax.Array) -> jax.Array:
        mean = jnp.mean(images, axis=(1, 2), keepdims=True)
        variance = jnp.mean(jnp.square(images - mean), axis=(1, 2), keepdims=True)
        return (images - mean) * jax.lax.rsqrt(variance + 1e-6)

    def _encode_images(self, images: jax.Array) -> jax.Array:
        batch_size, views, height, width, channels = images.shape
        if views != self.image_views:
            raise ValueError(f"Expected {self.image_views} image views, got {views}.")
        encoded = images.reshape((batch_size * views, height, width, channels))
        encoded = self._normalize_images(encoded)
        for convolution in self.image_convs:
            encoded = jax.nn.silu(convolution(encoded))
        encoded = jnp.mean(encoded, axis=(1, 2))
        return encoded.reshape((batch_size, -1))

    def _encode_context(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_plan_tokens: jax.Array,
        cached_iar: jax.Array,
        intended_prefix: jax.Array,
        intended_valid: jax.Array,
    ) -> jax.Array:
        anchor_visual = self._encode_images(anchor_images)
        current_visual = self._encode_images(current_images)
        visual_delta = current_visual - anchor_visual
        image_feature = jax.nn.silu(
            self.image_proj(
                jnp.concatenate(
                    [anchor_visual, current_visual, visual_delta, jnp.abs(visual_delta)],
                    axis=-1,
                )
            )
        )
        state_delta = current_state - anchor_state
        state_feature = jax.nn.silu(
            self.state_proj(jnp.concatenate([anchor_state, current_state, state_delta], axis=-1))
        )
        plan_tokens = jax.nn.silu(self.plan_token_proj(cached_plan_tokens))
        positions = jnp.linspace(-1.0, 1.0, cached_plan_tokens.shape[1], dtype=plan_tokens.dtype)
        plan_feature = jax.nn.silu(
            self.plan_summary_proj(
                jnp.concatenate(
                    [
                        jnp.mean(plan_tokens, axis=1),
                        jnp.mean(plan_tokens * positions[None, :, None], axis=1),
                    ],
                    axis=-1,
                )
            )
        )
        iar_feature = jnp.mean(jax.nn.silu(self.iar_proj(cached_iar)), axis=1)
        valid = intended_valid.astype(intended_prefix.dtype)
        prefix_feature = jax.nn.silu(
            self.prefix_proj(
                jnp.concatenate([intended_prefix * valid[..., None], valid[..., None]], axis=-1).reshape(
                    (cached_plan_tokens.shape[0], -1)
                )
            )
        )
        elapsed_fraction = jnp.sum(valid, axis=1, keepdims=True) / float(self.max_executed_steps)
        age_feature = jax.nn.silu(self.age_proj(elapsed_fraction))
        context = jax.nn.silu(
            self.fusion(
                jnp.concatenate(
                    [
                        image_feature,
                        state_feature,
                        plan_feature,
                        iar_feature,
                        prefix_feature,
                        age_feature,
                    ],
                    axis=-1,
                )
            )
        )
        return context + jax.nn.silu(self.hidden(context))

    def _actions_from_raw(
        self,
        base_actions: jax.Array,
        raw: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        residual = self.action_residual_scale[None, None, :6] * jnp.tanh(raw[..., :6])
        continuous = base_actions[..., :6] + residual
        base_sign = jnp.where(base_actions[..., 6] >= 0, 1.0, -1.0)
        base_logits = 4.0 * base_sign
        gripper_logits = base_logits + self.gripper_logit_scale * jnp.tanh(raw[..., 6])
        return jnp.concatenate([continuous, jnp.tanh(gripper_logits)[..., None]], axis=-1), gripper_logits

    def __call__(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_plan_tokens: jax.Array,
        cached_iar: jax.Array,
        intended_prefix: jax.Array,
        intended_valid: jax.Array,
        transported_ear: jax.Array,
        base_actions: jax.Array,
    ) -> CorrectorOutput:
        context = self._encode_context(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_plan_tokens,
            cached_iar,
            intended_prefix,
            intended_valid,
        )
        batch_size = context.shape[0]
        if self.mode == "direct":
            revised_ear = transported_ear
            plan_sign = jnp.where(revised_ear[..., 6] >= 0, 1.0, -1.0)
            plan_gripper_logits = 4.0 * plan_sign
            action_raw = self.action_out(context).reshape((batch_size, self.rollout_horizon, 7))
        else:
            plan_raw = self.plan_out(context).reshape((batch_size, self.ear_horizon, 7))
            continuous_plan = transported_ear[..., :6] + (
                self.plan_residual_scale[None, None, :6] * jnp.tanh(plan_raw[..., :6])
            )
            plan_sign = jnp.where(transported_ear[..., 6] >= 0, 1.0, -1.0)
            plan_gripper_logits = 4.0 * plan_sign + self.gripper_logit_scale * jnp.tanh(plan_raw[..., 6])
            revised_ear = jnp.concatenate(
                [continuous_plan, jnp.tanh(plan_gripper_logits)[..., None]],
                axis=-1,
            )
            plan_feature = jax.nn.silu(self.plan_decoder(revised_ear.reshape((batch_size, -1))))
            plan_feature = plan_feature + jax.nn.silu(self.plan_decoder_hidden(plan_feature))
            action_raw = self.action_out(plan_feature).reshape((batch_size, self.rollout_horizon, 7))
        actions, action_gripper_logits = self._actions_from_raw(base_actions, action_raw)
        return CorrectorOutput(
            actions=actions,
            action_gripper_logits=action_gripper_logits,
            transported_ear=transported_ear,
            revised_ear=revised_ear,
            plan_gripper_logits=plan_gripper_logits,
        )


def _estimate_parameter_count(model: BranchedActionCorrector) -> int:
    _, params = nnx.split(model)
    return int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))


def _transport_ear(cached_ear: np.ndarray, phase: float) -> np.ndarray:
    cached = np.asarray(cached_ear, dtype=np.float32)
    phases = np.full((cached.shape[0], cached.shape[1]), phase, dtype=np.float32)
    phases += np.arange(cached.shape[1], dtype=np.float32)[None]
    return np.asarray(transported_action_cot.interpolate_ear(jnp.asarray(cached), jnp.asarray(phases)))


def _flat_arrays(
    arrays: dict[str, np.ndarray],
    indices: progress_probe.BranchIndices,
    *,
    age: int,
    rollout_horizon: int,
    coarse_time_stride: int,
) -> dict[str, np.ndarray]:
    roots = indices.roots
    branches = indices.branches
    branch_steps = np.asarray(arrays["branch_steps"])[roots, branches]
    if np.any(branch_steps != age):
        raise ValueError(f"All selected branches must have age={age}; got {np.unique(branch_steps).tolist()}.")
    cached_actions_env = np.asarray(arrays["cached_actions_env"], dtype=np.float32)[roots]
    if age + rollout_horizon > cached_actions_env.shape[1]:
        raise ValueError("age + rollout_horizon exceeds the cached action horizon.")
    cached_ear = np.asarray(arrays["cached_ear"], dtype=np.float32)[roots]
    padded_actions = np.zeros(
        (cached_actions_env.shape[0], cached_actions_env.shape[1], cached_ear.shape[-1]),
        dtype=np.float32,
    )
    padded_actions[..., : cached_actions_env.shape[-1]] = cached_actions_env
    intended_prefix = cached_actions_env[:, :age]
    return {
        "root_id": np.asarray(arrays["root_id"])[roots],
        "task_id": np.asarray(arrays["task_id"])[roots],
        "episode_id": np.asarray(arrays["episode_id"])[roots],
        "branch_id": np.asarray(arrays["branch_ids"])[roots, branches],
        "anchor_images": np.asarray(arrays["anchor_images"])[roots],
        "current_images": np.asarray(arrays["current_images"])[roots, branches],
        "anchor_state": np.asarray(arrays["anchor_state"])[roots],
        "current_state": np.asarray(arrays["current_state"])[roots, branches],
        "cached_plan_tokens": np.concatenate([cached_ear, padded_actions], axis=1),
        "cached_iar": np.asarray(arrays["cached_iar"], dtype=np.float32)[roots],
        "intended_prefix": intended_prefix,
        "intended_valid": np.ones(intended_prefix.shape[:2], dtype=np.bool_),
        "transported_ear": _transport_ear(cached_ear[..., :7], age / coarse_time_stride),
        "base_actions": cached_actions_env[:, age : age + rollout_horizon, :7],
        "target_ear": np.asarray(arrays["fresh_ear"], dtype=np.float32)[roots, branches, :, :7],
        "target_actions": np.asarray(arrays["fresh_actions_env"], dtype=np.float32)[
            roots, branches, :rollout_horizon, :7
        ],
    }


def _calibrate_ranges(
    flat: dict[str, np.ndarray],
    *,
    output_margin: float,
    minimum_action: float,
    minimum_plan: float,
) -> dict[str, tuple[float, ...] | float]:
    action_residual = flat["target_actions"] - flat["base_actions"]
    plan_residual = flat["target_ear"] - flat["transported_ear"]
    action_scale = np.maximum(
        minimum_action,
        output_margin * np.quantile(np.abs(action_residual[..., :6]), 0.995, axis=(0, 1)),
    )
    plan_scale = np.maximum(
        minimum_plan,
        output_margin * np.quantile(np.abs(plan_residual[..., :6]), 0.995, axis=(0, 1)),
    )
    return {
        "action_residual_scale": tuple(float(value) for value in np.r_[action_scale, 1.0]),
        "plan_residual_scale": tuple(float(value) for value in np.r_[plan_scale, 1.0]),
        "action_out_of_range_fraction": float(
            np.mean(np.abs(action_residual[..., :6]) >= action_scale[None, None])
        ),
        "plan_out_of_range_fraction": float(
            np.mean(np.abs(plan_residual[..., :6]) >= plan_scale[None, None])
        ),
    }


def _batch(
    flat: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    current_equals_anchor: bool,
) -> dict[str, jax.Array]:
    anchor_images = flat["anchor_images"][indices].astype(np.float32) / 255.0
    anchor_state = flat["anchor_state"][indices].astype(np.float32)
    current_images = (
        anchor_images if current_equals_anchor else flat["current_images"][indices].astype(np.float32) / 255.0
    )
    current_state = anchor_state if current_equals_anchor else flat["current_state"][indices].astype(np.float32)
    return {
        "anchor_images": jnp.asarray(anchor_images),
        "current_images": jnp.asarray(current_images),
        "anchor_state": jnp.asarray(anchor_state),
        "current_state": jnp.asarray(current_state),
        "cached_plan_tokens": jnp.asarray(flat["cached_plan_tokens"][indices], dtype=jnp.float32),
        "cached_iar": jnp.asarray(flat["cached_iar"][indices], dtype=jnp.float32),
        "intended_prefix": jnp.asarray(flat["intended_prefix"][indices], dtype=jnp.float32),
        "intended_valid": jnp.asarray(flat["intended_valid"][indices], dtype=jnp.bool_),
        "transported_ear": jnp.asarray(flat["transported_ear"][indices], dtype=jnp.float32),
        "base_actions": jnp.asarray(flat["base_actions"][indices], dtype=jnp.float32),
        "target_ear": jnp.asarray(flat["target_ear"][indices], dtype=jnp.float32),
        "target_actions": jnp.asarray(flat["target_actions"][indices], dtype=jnp.float32),
    }


def _huber(values: jax.Array, delta: float) -> jax.Array:
    absolute = jnp.abs(values)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * jnp.square(quadratic) + delta * (absolute - quadratic)


def _binary_cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _loss(
    output: CorrectorOutput,
    batch: dict[str, jax.Array],
    *,
    mode: str,
    args: Args,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    action_continuous = jnp.mean(
        _huber(output.actions[..., :6] - batch["target_actions"][..., :6], args.action_huber_delta)
    )
    action_gripper_target = (batch["target_actions"][..., 6] >= 0).astype(jnp.float32)
    action_gripper = jnp.mean(
        _binary_cross_entropy(output.action_gripper_logits, action_gripper_target)
    )
    plan_continuous = jnp.asarray(0.0, dtype=jnp.float32)
    plan_gripper = jnp.asarray(0.0, dtype=jnp.float32)
    plan_velocity = jnp.asarray(0.0, dtype=jnp.float32)
    if mode == "plan":
        plan_continuous = jnp.mean(
            _huber(output.revised_ear[..., :6] - batch["target_ear"][..., :6], args.plan_huber_delta)
        )
        plan_gripper_target = (batch["target_ear"][..., 6] >= 0).astype(jnp.float32)
        plan_gripper = jnp.mean(
            _binary_cross_entropy(output.plan_gripper_logits, plan_gripper_target)
        )
        predicted_velocity = jnp.diff(output.revised_ear[..., :6], axis=1)
        target_velocity = jnp.diff(batch["target_ear"][..., :6], axis=1)
        plan_velocity = jnp.mean(
            _huber(predicted_velocity - target_velocity, args.plan_huber_delta)
        )
    total = (
        action_continuous
        + args.action_gripper_weight * action_gripper
        + args.plan_loss_weight * plan_continuous
        + args.plan_gripper_weight * plan_gripper
        + args.plan_velocity_weight * plan_velocity
    )
    return total, {
        "loss": total,
        "action_continuous_huber": action_continuous,
        "action_gripper_bce": action_gripper,
        "plan_continuous_huber": plan_continuous,
        "plan_gripper_bce": plan_gripper,
        "plan_velocity_huber": plan_velocity,
    }


def _predict_all(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], CorrectorOutput],
    params: nnx.State,
    flat: dict[str, np.ndarray],
    *,
    batch_size: int,
    current_equals_anchor: bool,
) -> dict[str, np.ndarray]:
    actions: list[np.ndarray] = []
    ears: list[np.ndarray] = []
    for start in range(0, len(flat["branch_id"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(flat["branch_id"])))
        output = predict_step(
            params,
            _batch(flat, indices, current_equals_anchor=current_equals_anchor),
        )
        actions.append(np.asarray(jax.device_get(output.actions), dtype=np.float32))
        ears.append(np.asarray(jax.device_get(output.revised_ear), dtype=np.float32))
    return {"actions": np.concatenate(actions), "ear": np.concatenate(ears)}


def _action_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    error = predicted - target
    return {
        "count": int(predicted.shape[0]),
        "mse_6d": float(np.mean(np.square(error[..., :6]))),
        "first_action_mse_6d": float(np.mean(np.square(error[:, 0, :6]))),
        "mse_7d": float(np.mean(np.square(error))),
        "gripper_sign_accuracy": float(np.mean((predicted[..., 6] >= 0) == (target[..., 6] >= 0))),
    }


def _ear_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return {
        "count": int(predicted.shape[0]),
        "mse_6d": float(np.mean(np.square(predicted[..., :6] - target[..., :6]))),
        "gripper_sign_accuracy": float(np.mean((predicted[..., 6] >= 0) == (target[..., 6] >= 0))),
    }


def _stratified_metrics(
    predicted_actions: np.ndarray,
    predicted_ear: np.ndarray,
    flat: dict[str, np.ndarray],
) -> dict[str, Any]:
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        return {
            "actions": _action_metrics(predicted_actions[mask], flat["target_actions"][mask]),
            "ear": _ear_metrics(predicted_ear[mask], flat["target_ear"][mask]),
        }

    return {
        "overall": evaluate(np.ones(branch_ids.shape, dtype=np.bool_)),
        "nominal_disturbed": {
            "nominal": evaluate(branch_ids == 0),
            "disturbed": evaluate(branch_ids != 0),
        },
        "by_branch": {
            name: evaluate(branch_ids == branch_id)
            for branch_id, name in enumerate(branched_dataset.BRANCH_NAMES)
            if np.any(branch_ids == branch_id)
        },
    }


def _save_params(params: nnx.State, target: pathlib.Path, *, name: str, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {name: params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=overwrite)


def _make_model(
    flat: dict[str, np.ndarray],
    ranges: dict[str, tuple[float, ...] | float],
    *,
    args: Args,
) -> BranchedActionCorrector:
    return BranchedActionCorrector(
        mode=args.mode,  # type: ignore[arg-type]
        image_views=int(flat["anchor_images"].shape[1]),
        image_channels=int(flat["anchor_images"].shape[-1]),
        state_dim=int(flat["anchor_state"].shape[-1]),
        plan_dim=int(flat["cached_plan_tokens"].shape[-1]),
        iar_dim=int(flat["cached_iar"].shape[-1]),
        env_action_dim=int(flat["target_actions"].shape[-1]),
        max_executed_steps=int(flat["intended_prefix"].shape[-2]),
        ear_horizon=int(flat["transported_ear"].shape[-2]),
        rollout_horizon=int(flat["target_actions"].shape[-2]),
        hidden_dim=args.hidden_dim,
        action_residual_scale=ranges["action_residual_scale"],  # type: ignore[arg-type]
        plan_residual_scale=ranges["plan_residual_scale"],  # type: ignore[arg-type]
        gripper_logit_scale=args.gripper_logit_scale,
        rngs=nnx.Rngs(args.seed),
    )


def _train_one(
    train_flat: dict[str, np.ndarray],
    validation_flat: dict[str, np.ndarray],
    ranges: dict[str, tuple[float, ...] | float],
    *,
    current_equals_anchor: bool,
    run_name: str,
    args: Args,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    model = _make_model(train_flat, ranges, args=args)
    parameter_count = _estimate_parameter_count(model)
    if parameter_count >= args.max_parameters:
        raise ValueError(f"Corrector has {parameter_count:,} parameters; limit={args.max_parameters:,}.")
    graphdef, params = nnx.split(model)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.train_steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: nnx.State,
        current_optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        current_model = nnx.merge(graphdef, current_params)

        def loss_fn(candidate: BranchedActionCorrector):
            output = candidate(
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
            )
            return _loss(output, batch, mode=args.mode, args=args)

        (_, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(current_model)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        return (
            optax.apply_updates(current_params, updates),
            next_optimizer_state,
            {**metrics, "gradient_norm": optax.global_norm(gradients)},
        )

    @jax.jit
    def predict_step(current_params: nnx.State, batch: dict[str, jax.Array]) -> CorrectorOutput:
        current_model = nnx.merge(graphdef, current_params)
        return current_model(
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
        )

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / f"metrics_{run_name}.jsonl"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Metrics already exist: {metrics_path}")
    best_score = float("inf")
    best_step = 0
    best_params: nnx.State | None = None
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                len(train_flat["branch_id"]),
                size=args.batch_size,
                replace=len(train_flat["branch_id"]) < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                _batch(train_flat, sampled, current_equals_anchor=current_equals_anchor),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation = _predict_all(
                    predict_step,
                    params,
                    validation_flat,
                    batch_size=args.eval_batch_size,
                    current_equals_anchor=current_equals_anchor,
                )
                validation_action = _action_metrics(
                    validation["actions"],
                    validation_flat["target_actions"],
                )
                score = float(
                    validation_action["mse_6d"]
                    + 0.01 * (1.0 - validation_action["gripper_sign_accuracy"])
                )
                record = {
                    "run": run_name,
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()},
                    **{f"validation/{name}": value for name, value in validation_action.items()},
                    "validation/selection_score": score,
                }
                if not all(isinstance(value, (str, int)) or np.isfinite(value) for value in record.values()):
                    raise FloatingPointError(f"Non-finite corrector metrics: {record}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if score < best_score:
                    best_score = score
                    best_step = step
                    best_params = params
    selected_params = best_params if best_params is not None else params
    params_path = output_dir / "final" / run_name / "params"
    _save_params(
        selected_params,
        params_path,
        name=f"branched_action_corrector_{args.mode}_{run_name}",
        overwrite=args.overwrite,
    )
    return predict_step, selected_params, {
        "parameter_count": parameter_count,
        "completed_steps": args.train_steps,
        "best_validation_step": best_step,
        "best_validation_score": best_score,
        "selection_criterion": "validation 4-step continuous-6D MSE + 0.01 * gripper sign error",
        "training_input": (
            "anchor and latest current observation"
            if not current_equals_anchor
            else "anchor substituted for current observation throughout training and evaluation"
        ),
        "independently_trained": True,
        "elapsed_seconds": time.monotonic() - started,
        "params_path": str(params_path.resolve()),
    }


def _profile(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], CorrectorOutput],
    params: nnx.State,
    flat: dict[str, np.ndarray],
    *,
    args: Args,
) -> dict[str, Any]:
    batch = _batch(flat, np.asarray([0]), current_equals_anchor=False)
    for _ in range(args.profile_warmup):
        jax.block_until_ready(predict_step(params, batch).actions)
    samples = np.empty((args.profile_iterations,), dtype=np.float64)
    for index in range(args.profile_iterations):
        started = time.perf_counter()
        jax.block_until_ready(predict_step(params, batch).actions)
        samples[index] = (time.perf_counter() - started) * 1_000.0
    mean_ms = float(np.mean(samples))
    amortized = (
        args.full_acot_reference_ms + (args.refresh_interval - 1) * mean_ms
    ) / args.refresh_interval
    return {
        "device": str(jax.devices()[0]),
        "batch_size": 1,
        "warmup": args.profile_warmup,
        "iterations": args.profile_iterations,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "refresh_interval": args.refresh_interval,
        "full_acot_reference_ms": args.full_acot_reference_ms,
        "theoretical_amortized_ms": amortized,
        "theoretical_speedup_vs_full_acot": args.full_acot_reference_ms / amortized,
        "note": "GPU sidecar-only latency; excludes image transport, environment, and closed-loop integration.",
    }


def _train_residual_baselines(train_flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    residual = train_flat["target_actions"] - train_flat["base_actions"]
    global_residual = np.mean(residual, axis=0)
    branch_residual = np.stack(
        [
            np.mean(residual[train_flat["branch_id"] == branch_id], axis=0)
            for branch_id in range(len(branched_dataset.BRANCH_NAMES))
        ]
    )
    return {"global": global_residual, "branch": branch_residual}


def _baseline_evaluations(
    flat: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
) -> dict[str, Any]:
    stale = flat["base_actions"]
    global_actions = stale + residuals["global"][None]
    branch_actions = stale + residuals["branch"][flat["branch_id"]]
    return {
        "stale_cached_chunk": _stratified_metrics(stale, flat["transported_ear"], flat),
        "train_global_residual": _stratified_metrics(global_actions, flat["transported_ear"], flat),
        "privileged_branch_id_train_residual": _stratified_metrics(
            branch_actions,
            flat["transported_ear"],
            flat,
        ),
    }


def _annotate_relative_metrics(evaluations: dict[str, Any]) -> None:
    stale = evaluations["stale_cached_chunk"]["overall"]["actions"]["mse_6d"]
    for value in evaluations.values():
        mse = value["overall"]["actions"]["mse_6d"]
        value["overall"]["actions"]["relative_risk_reduction_vs_stale_6d"] = (
            float((stale - mse) / stale) if stale > 0 else None
        )


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")
    arrays = branched_dataset.load_branched_arrays(
        args.dataset,
        fields=(
            "root_id",
            "task_id",
            "episode_id",
            "branch_ids",
            "branch_steps",
            "branch_valid",
            "anchor_images",
            "current_images",
            "anchor_state",
            "current_state",
            "cached_ear",
            "fresh_ear",
            "cached_iar",
            "cached_actions_env",
            "fresh_actions_env",
        ),
    )
    train_roots, validation_roots, test_roots = progress_probe._split_roots(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    partitions: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for name, roots in (
        ("train", train_roots),
        ("validation", validation_roots),
        ("test", test_roots),
    ):
        indices = progress_probe._flatten_valid_branches(arrays, roots)
        partitions[name] = (
            roots,
            _flat_arrays(
                arrays,
                indices,
                age=args.age,
                rollout_horizon=args.rollout_horizon,
                coarse_time_stride=args.coarse_time_stride,
            ),
        )
    train_flat = partitions["train"][1]
    validation_flat = partitions["validation"][1]
    test_flat = partitions["test"][1]
    ranges = _calibrate_ranges(
        train_flat,
        output_margin=args.output_margin,
        minimum_action=args.minimum_action_residual_scale,
        minimum_plan=args.minimum_plan_residual_scale,
    )
    residual_baselines = _train_residual_baselines(train_flat)

    current_predict, current_params, current_train = _train_one(
        train_flat,
        validation_flat,
        ranges,
        current_equals_anchor=False,
        run_name="current",
        args=args,
        output_dir=output_dir,
    )
    no_current_predict = None
    no_current_params = None
    no_current_train = None
    if args.train_no_current:
        no_current_predict, no_current_params, no_current_train = _train_one(
            train_flat,
            validation_flat,
            ranges,
            current_equals_anchor=True,
            run_name="no_current",
            args=args,
            output_dir=output_dir,
        )

    current_prediction = _predict_all(
        current_predict,
        current_params,
        test_flat,
        batch_size=args.eval_batch_size,
        current_equals_anchor=False,
    )
    evaluations = _baseline_evaluations(test_flat, residual_baselines)
    evaluations["learned_current"] = _stratified_metrics(
        current_prediction["actions"],
        current_prediction["ear"],
        test_flat,
    )
    if no_current_predict is not None and no_current_params is not None:
        no_current_prediction = _predict_all(
            no_current_predict,
            no_current_params,
            test_flat,
            batch_size=args.eval_batch_size,
            current_equals_anchor=True,
        )
        evaluations["learned_no_current_independent"] = _stratified_metrics(
            no_current_prediction["actions"],
            no_current_prediction["ear"],
            test_flat,
        )
    _annotate_relative_metrics(evaluations)

    profile = _profile(current_predict, current_params, test_flat, args=args)
    stale_mse = evaluations["stale_cached_chunk"]["overall"]["actions"]["mse_6d"]
    strong_baseline_mse = min(
        evaluations["train_global_residual"]["overall"]["actions"]["mse_6d"],
        evaluations["privileged_branch_id_train_residual"]["overall"]["actions"]["mse_6d"],
    )
    current_mse = evaluations["learned_current"]["overall"]["actions"]["mse_6d"]
    current_gripper = evaluations["learned_current"]["overall"]["actions"]["gripper_sign_accuracy"]
    no_current_mse = (
        evaluations["learned_no_current_independent"]["overall"]["actions"]["mse_6d"]
        if "learned_no_current_independent" in evaluations
        else None
    )
    current_gain_over_no_current = (
        float((no_current_mse - current_mse) / no_current_mse)
        if no_current_mse is not None and no_current_mse > 0
        else None
    )
    transported_ear_mse = evaluations["stale_cached_chunk"]["overall"]["ear"]["mse_6d"]
    learned_ear_mse = evaluations["learned_current"]["overall"]["ear"]["mse_6d"]
    plan_gap_closure = (
        float((transported_ear_mse - learned_ear_mse) / transported_ear_mse)
        if transported_ear_mse > 0
        else None
    )
    quality_gate = (
        current_mse <= 0.90 * strong_baseline_mse
        and current_gripper >= 0.96
        and (current_gain_over_no_current is None or current_gain_over_no_current >= 0.03)
        and (args.mode != "plan" or (plan_gap_closure is not None and plan_gap_closure >= 0.25))
    )
    speed_gate = profile["p95_ms"] < 2.0 and profile["theoretical_speedup_vs_full_acot"] >= 3.0
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": {
            "name": f"branched_action_cot_corrector_{args.mode}",
            "mode": args.mode,
            "current_observation_path": (
                "direct residual around cached actions"
                if args.mode == "direct"
                else "full EAR update followed by a plan-only action decoder"
            ),
            "deployment_safe_history": "controller-intended cached action prefix only",
            "forbidden_inputs": "no branch id and no actually injected synthetic fault actions",
        },
        "args": dataclasses.asdict(args),
        "split": {
            name: {
                "root_count": int(len(roots)),
                "valid_branch_count": int(len(flat["branch_id"])),
                "episodes_by_task": progress_probe._episode_summary(arrays, roots),
            }
            for name, (roots, flat) in partitions.items()
        },
        "range_calibration": {
            "source": "training partition only",
            **ranges,
        },
        "train": {
            "current": current_train,
            "no_current": no_current_train,
        },
        "profile": profile,
        "test": evaluations,
        "offline_gate": {
            "quality_gate_pass": quality_gate,
            "speed_gate_pass": speed_gate,
            "offline_gate_pass": quality_gate and speed_gate,
            "stale_action_mse_6d": stale_mse,
            "strong_train_only_baseline_mse_6d": strong_baseline_mse,
            "learned_current_action_mse_6d": current_mse,
            "required_relative_improvement_over_strong_baseline": 0.10,
            "learned_current_gripper_sign_accuracy": current_gripper,
            "minimum_gripper_sign_accuracy": 0.96,
            "learned_no_current_action_mse_6d": no_current_mse,
            "current_gain_over_no_current": current_gain_over_no_current,
            "minimum_current_gain_over_no_current": 0.03,
            "plan_gap_closure_6d": plan_gap_closure,
            "minimum_plan_gap_closure_6d": 0.25 if args.mode == "plan" else None,
            "minimum_theoretical_speedup": 3.0,
        },
        "note": (
            "Episode-held-out canonical branch action/EAR fidelity only. "
            "The amortized latency is a sidecar estimate, not a closed-loop LIBERO result."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
