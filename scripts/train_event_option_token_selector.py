# ruff: noqa: SLF001

"""Train a discrete Event-Option selector over a cached action chunk.

The selector is deliberately bottlenecked: observations may only choose one
of the ten cached action tokens.  They cannot produce an action residual or a
new action.  Training uses a differentiable soft mixture of cached continuous
actions, while the primary deployable evaluation uses hard argmax selection.

Roots are split by episode before branches are flattened.  A second,
independently trained no-current-observation model is included so that the
value of endpoint observations is compared under matched optimization.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import train_branched_effective_progress as phase_probe
import tyro

from openpi.action_cot import branched_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    seed: int = 7
    split_seed: int = 7
    train_steps: int = 500
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    log_interval: int = 50
    action_loss_weight: float = 1.0
    soft_target_ce_weight: float = 0.1
    soft_target_temperature: float = 0.02
    fixed_gripper_age: int = 4
    continuous_grid_size: int = 361
    max_parameters: int = 1_000_000
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seed and split_seed must be non-negative.")
    if (
        args.train_steps <= 0
        or args.batch_size <= 0
        or args.eval_batch_size <= 0
        or args.log_interval <= 0
        or args.continuous_grid_size < 2
    ):
        raise ValueError("Training, batching, logging, and grid sizes must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Optimizer scales must be positive and weight_decay non-negative.")
    if args.action_loss_weight <= 0 or args.soft_target_ce_weight < 0:
        raise ValueError("action_loss_weight must be positive and CE weight non-negative.")
    if args.soft_target_temperature <= 0:
        raise ValueError("soft_target_temperature must be positive.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if args.fixed_gripper_age < 0:
        raise ValueError("fixed_gripper_age must be non-negative.")
    if args.max_parameters <= 0 or args.max_parameters > 1_000_000:
        raise ValueError("max_parameters must lie in [1, 1_000_000].")


def _flat_arrays(
    arrays: dict[str, np.ndarray],
    indices: phase_probe.BranchIndices,
) -> dict[str, np.ndarray]:
    roots = indices.roots
    branches = indices.branches
    return {
        "root_id": np.asarray(arrays["root_id"])[roots],
        "task_id": np.asarray(arrays["task_id"])[roots],
        "episode_id": np.asarray(arrays["episode_id"])[roots],
        "branch_id": np.asarray(arrays["branch_ids"])[roots, branches],
        "branch_steps": np.asarray(arrays["branch_steps"])[roots, branches],
        "anchor_images": np.asarray(arrays["anchor_images"])[roots],
        "current_images": np.asarray(arrays["current_images"])[roots, branches],
        "anchor_state": np.asarray(arrays["anchor_state"])[roots],
        "current_state": np.asarray(arrays["current_state"])[roots, branches],
        "cached_actions": np.asarray(arrays["cached_actions"])[roots],
        "fresh_actions": np.asarray(arrays["fresh_actions"])[roots, branches],
        "executed_actions": np.asarray(arrays["executed_actions"])[roots, branches],
        "executed_valid": np.asarray(arrays["executed_valid"])[roots, branches],
    }


def _per_token_costs(cached_actions: np.ndarray, fresh_actions: np.ndarray) -> np.ndarray:
    cached = np.asarray(cached_actions, dtype=np.float32)
    fresh = np.asarray(fresh_actions, dtype=np.float32)
    if cached.ndim != 3 or fresh.ndim != 3 or cached.shape[0] != fresh.shape[0]:
        raise ValueError(f"Action arrays must be [N,H,D], got {cached.shape} and {fresh.shape}.")
    if cached.shape[-1] < 7 or fresh.shape[-1] < 7 or fresh.shape[1] == 0:
        raise ValueError("Actions need at least seven dimensions and a non-empty fresh horizon.")
    return np.mean(np.square(cached[..., :6] - fresh[:, None, 0, :6]), axis=-1)


def _discrete_oracle_token(cached_actions: np.ndarray, fresh_actions: np.ndarray) -> np.ndarray:
    return np.argmin(_per_token_costs(cached_actions, fresh_actions), axis=1).astype(np.int64)


def _calibrate_global_token(cached_actions: np.ndarray, fresh_actions: np.ndarray) -> int:
    mean_cost = np.mean(_per_token_costs(cached_actions, fresh_actions), axis=0)
    return int(np.argmin(mean_cost))


def _interpolate_action_token(cached_actions: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    cached = np.asarray(cached_actions, dtype=np.float32)
    phases = np.broadcast_to(np.asarray(phase, dtype=np.float32), (cached.shape[0],))
    phases = np.clip(phases, 0.0, float(cached.shape[1] - 1))
    lower = np.floor(phases).astype(np.int64)
    upper = np.minimum(lower + 1, cached.shape[1] - 1)
    alpha = phases - lower
    batch = np.arange(cached.shape[0])
    return cached[batch, lower] + alpha[:, None] * (cached[batch, upper] - cached[batch, lower])


def _calibrate_global_continuous(
    cached_actions: np.ndarray,
    fresh_actions: np.ndarray,
    *,
    grid_size: int,
) -> float:
    grid = np.linspace(0.0, cached_actions.shape[1] - 1, grid_size, dtype=np.float32)
    target = np.asarray(fresh_actions, dtype=np.float32)[:, 0, :6]
    errors = [
        float(np.mean(np.square(_interpolate_action_token(cached_actions, phase)[:, :6] - target)))
        for phase in grid
    ]
    return float(grid[int(np.argmin(errors))])


def _branch_tokens(
    branch_ids: np.ndarray,
    cached_actions: np.ndarray,
    fresh_actions: np.ndarray,
    *,
    global_token: int,
) -> dict[int, int]:
    outputs: dict[int, int] = {}
    for branch_id in range(len(branched_dataset.BRANCH_NAMES)):
        selected = np.asarray(branch_ids) == branch_id
        outputs[branch_id] = (
            _calibrate_global_token(cached_actions[selected], fresh_actions[selected])
            if np.any(selected)
            else global_token
        )
    return outputs


class EventOptionTokenSelector(nnx.Module):
    """The 221k visual/state/prefix/plan encoder with token logits."""

    def __init__(
        self,
        *,
        image_views: int,
        image_channels: int,
        state_dim: int,
        action_dim: int,
        env_action_dim: int,
        max_executed_steps: int,
        action_horizon: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 128,
        cnn_channels: tuple[int, ...] = (16, 32, 48),
    ) -> None:
        self.image_views = image_views
        self.max_executed_steps = max_executed_steps
        image_convs = []
        in_channels = image_channels
        for out_channels, kernel_size in zip(cnn_channels, (5, 3, 3), strict=True):
            image_convs.append(phase_probe._Conv2D(in_channels, out_channels, kernel_size, rngs=rngs))
            in_channels = out_channels
        self.image_convs = image_convs
        image_summary = image_views * cnn_channels[-1]
        self.image_proj = nnx.Linear(4 * image_summary, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(3 * state_dim, hidden_dim, rngs=rngs)
        self.plan_token_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.plan_summary_proj = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        prefix_dim = max_executed_steps * (env_action_dim + 1)
        self.prefix_proj = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.age_proj = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.fusion = nnx.Linear(5 * hidden_dim, hidden_dim, rngs=rngs)
        self.hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.token_out = nnx.Linear(
            hidden_dim,
            action_horizon,
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

    def __call__(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_actions: jax.Array,
        executed_actions: jax.Array,
        executed_valid: jax.Array,
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
        plan_tokens = jax.nn.silu(self.plan_token_proj(cached_actions))
        plan_positions = jnp.linspace(-1.0, 1.0, cached_actions.shape[1], dtype=plan_tokens.dtype)
        plan_feature = jax.nn.silu(
            self.plan_summary_proj(
                jnp.concatenate(
                    [
                        jnp.mean(plan_tokens, axis=1),
                        jnp.mean(plan_tokens * plan_positions[None, :, None], axis=1),
                    ],
                    axis=-1,
                )
            )
        )
        valid = executed_valid.astype(executed_actions.dtype)
        masked_actions = executed_actions * valid[..., None]
        prefix_feature = jax.nn.silu(
            self.prefix_proj(
                jnp.concatenate([masked_actions, valid[..., None]], axis=-1).reshape(
                    (cached_actions.shape[0], -1)
                )
            )
        )
        elapsed_fraction = jnp.sum(valid, axis=1, keepdims=True) / float(self.max_executed_steps)
        age_feature = jax.nn.silu(self.age_proj(elapsed_fraction))
        hidden = jax.nn.silu(
            self.fusion(
                jnp.concatenate(
                    [image_feature, state_feature, plan_feature, prefix_feature, age_feature],
                    axis=-1,
                )
            )
        )
        hidden = hidden + jax.nn.silu(self.hidden(hidden))
        return self.token_out(hidden)


def estimate_parameter_count(
    *,
    image_views: int,
    image_channels: int,
    state_dim: int,
    action_dim: int,
    env_action_dim: int,
    max_executed_steps: int,
    action_horizon: int,
    hidden_dim: int = 128,
    cnn_channels: tuple[int, ...] = (16, 32, 48),
) -> int:
    count = 0
    in_channels = image_channels
    for out_channels, kernel_size in zip(cnn_channels, (5, 3, 3), strict=True):
        count += kernel_size * kernel_size * in_channels * out_channels + out_channels
        in_channels = out_channels
    image_summary = image_views * cnn_channels[-1]
    count += phase_probe._linear_parameters(4 * image_summary, hidden_dim)
    count += phase_probe._linear_parameters(3 * state_dim, hidden_dim)
    count += phase_probe._linear_parameters(action_dim, hidden_dim)
    count += phase_probe._linear_parameters(2 * hidden_dim, hidden_dim)
    count += phase_probe._linear_parameters(max_executed_steps * (env_action_dim + 1), hidden_dim)
    count += phase_probe._linear_parameters(1, hidden_dim)
    count += phase_probe._linear_parameters(5 * hidden_dim, hidden_dim)
    count += phase_probe._linear_parameters(hidden_dim, hidden_dim)
    count += phase_probe._linear_parameters(hidden_dim, action_horizon)
    return count


def _batch(
    flat: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    current_equals_anchor: bool,
    token_permutation: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    anchor_images = flat["anchor_images"][indices].astype(np.float32) / 255.0
    anchor_state = flat["anchor_state"][indices].astype(np.float32)
    current_images = (
        anchor_images if current_equals_anchor else flat["current_images"][indices].astype(np.float32) / 255.0
    )
    current_state = anchor_state if current_equals_anchor else flat["current_state"][indices].astype(np.float32)
    cached_actions = flat["cached_actions"][indices].astype(np.float32)
    if token_permutation is not None:
        cached_actions = cached_actions[:, token_permutation]
    return {
        "anchor_images": jnp.asarray(anchor_images),
        "current_images": jnp.asarray(current_images),
        "anchor_state": jnp.asarray(anchor_state),
        "current_state": jnp.asarray(current_state),
        "cached_actions": jnp.asarray(cached_actions),
        "fresh_actions": jnp.asarray(flat["fresh_actions"][indices], dtype=jnp.float32),
        "executed_actions": jnp.asarray(flat["executed_actions"][indices], dtype=jnp.float32),
        "executed_valid": jnp.asarray(flat["executed_valid"][indices], dtype=jnp.bool_),
    }


def _loss(
    model: EventOptionTokenSelector,
    batch: dict[str, jax.Array],
    *,
    action_loss_weight: float,
    soft_target_ce_weight: float,
    soft_target_temperature: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = model(
        batch["anchor_images"],
        batch["current_images"],
        batch["anchor_state"],
        batch["current_state"],
        batch["cached_actions"],
        batch["executed_actions"],
        batch["executed_valid"],
    )
    probabilities = jax.nn.softmax(logits, axis=-1)
    mixed_action = jnp.einsum("nh,nhd->nd", probabilities, batch["cached_actions"][..., :6])
    target_action = batch["fresh_actions"][:, 0, :6]
    action_mse = jnp.mean(jnp.square(mixed_action - target_action))
    token_costs = jnp.mean(
        jnp.square(batch["cached_actions"][..., :6] - target_action[:, None]),
        axis=-1,
    )
    soft_target = jax.nn.softmax(-token_costs / soft_target_temperature, axis=-1)
    soft_target_ce = -jnp.mean(jnp.sum(soft_target * jax.nn.log_softmax(logits, axis=-1), axis=-1))
    total = action_loss_weight * action_mse + soft_target_ce_weight * soft_target_ce
    hard_tokens = jnp.argmax(logits, axis=-1)
    hard_action = batch["cached_actions"][
        jnp.arange(batch["cached_actions"].shape[0]),
        hard_tokens,
        :6,
    ]
    return total, {
        "loss": total,
        "soft_mixture_action_mse_6d": action_mse,
        "hard_argmax_action_mse_6d": jnp.mean(jnp.square(hard_action - target_action)),
        "soft_target_ce": soft_target_ce,
        "hard_oracle_token_accuracy": jnp.mean(
            hard_tokens == jnp.argmin(token_costs, axis=-1)
        ),
    }


def _predict_all(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], jax.Array],
    params: nnx.State,
    flat: dict[str, np.ndarray],
    *,
    batch_size: int,
    current_equals_anchor: bool,
    token_permutation: np.ndarray | None = None,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(flat["branch_id"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(flat["branch_id"])))
        output = predict_step(
            params,
            _batch(
                flat,
                indices,
                current_equals_anchor=current_equals_anchor,
                token_permutation=token_permutation,
            ),
        )
        outputs.append(np.asarray(jax.device_get(output), dtype=np.float32))
    return np.concatenate(outputs)


def _save_params(
    params: nnx.State,
    target: pathlib.Path,
    *,
    model_name: str,
    overwrite: bool,
) -> None:
    item = {"params": {model_name: params.to_pure_dict()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=overwrite)


def _action_from_hard_tokens(
    cached_actions: np.ndarray,
    tokens: np.ndarray,
    *,
    fixed_gripper_age: int,
) -> np.ndarray:
    cached = np.asarray(cached_actions, dtype=np.float32)
    selected = cached[np.arange(cached.shape[0]), np.asarray(tokens, dtype=np.int64), :7].copy()
    gripper_age = min(fixed_gripper_age, cached.shape[1] - 1)
    selected[:, 6] = cached[:, gripper_age, 6]
    return selected


def _action_from_probabilities(
    cached_actions: np.ndarray,
    probabilities: np.ndarray,
    *,
    fixed_gripper_age: int,
) -> np.ndarray:
    cached = np.asarray(cached_actions, dtype=np.float32)
    output = np.einsum("nh,nhd->nd", probabilities, cached[..., :7])
    gripper_age = min(fixed_gripper_age, cached.shape[1] - 1)
    output[:, 6] = cached[:, gripper_age, 6]
    return output


def _metrics_for_mask(
    actions: np.ndarray,
    predicted_tokens: np.ndarray | None,
    oracle_tokens: np.ndarray,
    fresh_actions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_actions = np.asarray(actions, dtype=np.float32)[mask]
    targets = np.asarray(fresh_actions, dtype=np.float32)[mask, 0, :7]
    result: dict[str, Any] = {
        "count": int(np.sum(mask)),
        "action_mse_6d": float(np.mean(np.square(selected_actions[:, :6] - targets[:, :6]))),
        "action_mse_7d_fixed_gripper": float(np.mean(np.square(selected_actions - targets))),
    }
    if predicted_tokens is not None:
        selected_tokens = np.asarray(predicted_tokens, dtype=np.int64)[mask]
        selected_oracle = np.asarray(oracle_tokens, dtype=np.int64)[mask]
        result.update(
            {
                "discrete_oracle_token_accuracy": float(np.mean(selected_tokens == selected_oracle)),
                "discrete_oracle_token_mae": float(np.mean(np.abs(selected_tokens - selected_oracle))),
                "predicted_token_mean": float(np.mean(selected_tokens)),
            }
        )
    return result


def _stratified_metrics(
    actions: np.ndarray,
    predicted_tokens: np.ndarray | None,
    oracle_tokens: np.ndarray,
    flat: dict[str, np.ndarray],
) -> dict[str, Any]:
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        return _metrics_for_mask(
            actions,
            predicted_tokens,
            oracle_tokens,
            flat["fresh_actions"],
            mask,
        )

    return {
        "overall": evaluate(np.ones((len(branch_ids),), dtype=np.bool_)),
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


def _train_one(
    train_flat: dict[str, np.ndarray],
    validation_flat: dict[str, np.ndarray],
    *,
    current_equals_anchor: bool,
    run_name: str,
    args: Args,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    image_shape = train_flat["anchor_images"].shape
    action_horizon = train_flat["cached_actions"].shape[1]
    model = EventOptionTokenSelector(
        image_views=image_shape[1],
        image_channels=image_shape[-1],
        state_dim=train_flat["anchor_state"].shape[-1],
        action_dim=train_flat["cached_actions"].shape[-1],
        env_action_dim=train_flat["executed_actions"].shape[-1],
        max_executed_steps=train_flat["executed_actions"].shape[-2],
        action_horizon=action_horizon,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(model)
    actual_parameters = int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))
    expected_parameters = estimate_parameter_count(
        image_views=image_shape[1],
        image_channels=image_shape[-1],
        state_dim=train_flat["anchor_state"].shape[-1],
        action_dim=train_flat["cached_actions"].shape[-1],
        env_action_dim=train_flat["executed_actions"].shape[-1],
        max_executed_steps=train_flat["executed_actions"].shape[-2],
        action_horizon=action_horizon,
    )
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Parameter accounting mismatch: actual={actual_parameters}, expected={expected_parameters}."
        )
    if actual_parameters >= args.max_parameters:
        raise ValueError(f"Selector has {actual_parameters:,} parameters; limit={args.max_parameters:,}.")
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

        def loss_fn(candidate: EventOptionTokenSelector):
            return _loss(
                candidate,
                batch,
                action_loss_weight=args.action_loss_weight,
                soft_target_ce_weight=args.soft_target_ce_weight,
                soft_target_temperature=args.soft_target_temperature,
            )

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
    def predict_step(current_params: nnx.State, batch: dict[str, jax.Array]) -> jax.Array:
        current_model = nnx.merge(graphdef, current_params)
        return current_model(
            batch["anchor_images"],
            batch["current_images"],
            batch["anchor_state"],
            batch["current_state"],
            batch["cached_actions"],
            batch["executed_actions"],
            batch["executed_valid"],
        )

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / f"metrics_{run_name}.jsonl"
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
                _batch(
                    train_flat,
                    sampled,
                    current_equals_anchor=current_equals_anchor,
                ),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_logits = _predict_all(
                    predict_step,
                    params,
                    validation_flat,
                    batch_size=args.eval_batch_size,
                    current_equals_anchor=current_equals_anchor,
                )
                validation_tokens = np.argmax(validation_logits, axis=-1)
                validation_action = _action_from_hard_tokens(
                    validation_flat["cached_actions"],
                    validation_tokens,
                    fixed_gripper_age=args.fixed_gripper_age,
                )
                validation_score = float(
                    np.mean(
                        np.square(
                            validation_action[:, :6] - validation_flat["fresh_actions"][:, 0, :6]
                        )
                    )
                )
                record = {
                    "run": run_name,
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()},
                    "validation/hard_argmax_action_mse_6d": validation_score,
                }
                if not all(isinstance(value, str) or np.isfinite(value) for value in record.values()):
                    raise FloatingPointError(f"Non-finite selector metric: {record}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if validation_score < best_score:
                    best_score = validation_score
                    best_step = step
                    best_params = params
    selected_params = best_params if best_params is not None else params
    params_path = output_dir / "final" / run_name / "params"
    _save_params(
        selected_params,
        params_path,
        model_name=f"event_option_token_selector_{run_name}",
        overwrite=args.overwrite,
    )
    return (
        predict_step,
        selected_params,
        {
            "training_input": (
                "anchor and current observations"
                if not current_equals_anchor
                else "anchor substituted for current observation during all training and evaluation"
            ),
            "independently_trained": True,
            "parameter_count": actual_parameters,
            "completed_steps": args.train_steps,
            "best_validation_step": best_step,
            "best_validation_hard_argmax_action_mse_6d": best_score,
            "selection_criterion": "validation hard-argmax cached-action MSE over dimensions 0:6",
            "elapsed_seconds": time.monotonic() - started,
            "params_path": str(params_path.resolve()),
        },
    )


def _evaluate_hard(
    tokens: np.ndarray,
    cached_actions: np.ndarray,
    oracle_tokens: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    fixed_gripper_age: int,
    metric_tokens: np.ndarray | None = None,
    gripper_source_actions: np.ndarray | None = None,
) -> dict[str, Any]:
    actions = _action_from_hard_tokens(
        cached_actions,
        tokens,
        fixed_gripper_age=fixed_gripper_age,
    )
    if gripper_source_actions is not None:
        actions[:, 6] = np.asarray(gripper_source_actions)[:, fixed_gripper_age, 6]
    return _stratified_metrics(
        actions,
        tokens if metric_tokens is None else metric_tokens,
        oracle_tokens,
        flat,
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
            "cached_actions",
            "fresh_actions",
            "executed_actions",
            "executed_valid",
        ),
    )
    train_roots, validation_roots, test_roots = phase_probe._split_roots(
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
        indices = phase_probe._flatten_valid_branches(arrays, roots)
        partitions[name] = (roots, _flat_arrays(arrays, indices))

    train_flat = partitions["train"][1]
    validation_flat = partitions["validation"][1]
    test_flat = partitions["test"][1]
    action_horizon = train_flat["cached_actions"].shape[1]
    if args.fixed_gripper_age >= action_horizon:
        raise ValueError(
            f"fixed_gripper_age={args.fixed_gripper_age} exceeds action horizon {action_horizon}."
        )
    global_token = _calibrate_global_token(
        train_flat["cached_actions"],
        train_flat["fresh_actions"],
    )
    global_continuous = _calibrate_global_continuous(
        train_flat["cached_actions"],
        train_flat["fresh_actions"],
        grid_size=args.continuous_grid_size,
    )
    branch_tokens = _branch_tokens(
        train_flat["branch_id"],
        train_flat["cached_actions"],
        train_flat["fresh_actions"],
        global_token=global_token,
    )
    current_predict, current_params, current_train = _train_one(
        train_flat,
        validation_flat,
        current_equals_anchor=False,
        run_name="current",
        args=args,
        output_dir=output_dir,
    )
    no_current_predict, no_current_params, no_current_train = _train_one(
        train_flat,
        validation_flat,
        current_equals_anchor=True,
        run_name="no_current",
        args=args,
        output_dir=output_dir,
    )

    current_logits = _predict_all(
        current_predict,
        current_params,
        test_flat,
        batch_size=args.eval_batch_size,
        current_equals_anchor=False,
    )
    no_current_logits = _predict_all(
        no_current_predict,
        no_current_params,
        test_flat,
        batch_size=args.eval_batch_size,
        current_equals_anchor=True,
    )
    # Reverse all tokens as one deterministic causal ablation.  Both the plan
    # presented to the selector and its candidate token bank are permuted.
    token_permutation = np.arange(action_horizon - 1, -1, -1, dtype=np.int64)
    shuffled_logits = _predict_all(
        current_predict,
        current_params,
        test_flat,
        batch_size=args.eval_batch_size,
        current_equals_anchor=False,
        token_permutation=token_permutation,
    )
    shuffled_actions = test_flat["cached_actions"][:, token_permutation]
    oracle_tokens = _discrete_oracle_token(
        test_flat["cached_actions"],
        test_flat["fresh_actions"],
    )
    branch_ids = np.asarray(test_flat["branch_id"], dtype=np.int64)
    fixed4_tokens = np.full_like(oracle_tokens, args.fixed_gripper_age)
    global_tokens = np.full_like(oracle_tokens, global_token)
    privileged_tokens = np.asarray([branch_tokens[int(branch_id)] for branch_id in branch_ids])
    current_tokens = np.argmax(current_logits, axis=-1)
    no_current_tokens = np.argmax(no_current_logits, axis=-1)
    shuffled_tokens = np.argmax(shuffled_logits, axis=-1)

    evaluations: dict[str, Any] = {
        "fixed_token4": _evaluate_hard(
            fixed4_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "train_global_calibrated_token": _evaluate_hard(
            global_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "privileged_branch_id_train_token": _evaluate_hard(
            privileged_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "dense_discrete_oracle": _evaluate_hard(
            oracle_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "learned_current_hard_argmax_deployable": _evaluate_hard(
            current_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "learned_no_current_independent_hard_argmax_deployable": _evaluate_hard(
            no_current_tokens,
            test_flat["cached_actions"],
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        "learned_current_shuffled_plan_and_candidates_hard_argmax": _evaluate_hard(
            shuffled_tokens,
            shuffled_actions,
            oracle_tokens,
            test_flat,
            fixed_gripper_age=args.fixed_gripper_age,
            metric_tokens=token_permutation[shuffled_tokens],
            gripper_source_actions=test_flat["cached_actions"],
        ),
    }
    continuous_action = _interpolate_action_token(
        test_flat["cached_actions"],
        global_continuous,
    )
    continuous_action[:, 6] = test_flat["cached_actions"][:, args.fixed_gripper_age, 6]
    evaluations["train_global_calibrated_continuous_diagnostic"] = _stratified_metrics(
        continuous_action[:, :7],
        None,
        oracle_tokens,
        test_flat,
    )
    current_probabilities = jax.nn.softmax(jnp.asarray(current_logits), axis=-1)
    no_current_probabilities = jax.nn.softmax(jnp.asarray(no_current_logits), axis=-1)
    shuffled_probabilities = jax.nn.softmax(jnp.asarray(shuffled_logits), axis=-1)
    evaluations["learned_current_soft_mixture_diagnostic"] = _stratified_metrics(
        _action_from_probabilities(
            test_flat["cached_actions"],
            np.asarray(current_probabilities),
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        None,
        oracle_tokens,
        test_flat,
    )
    evaluations["learned_no_current_independent_soft_mixture_diagnostic"] = _stratified_metrics(
        _action_from_probabilities(
            test_flat["cached_actions"],
            np.asarray(no_current_probabilities),
            fixed_gripper_age=args.fixed_gripper_age,
        ),
        None,
        oracle_tokens,
        test_flat,
    )
    shuffled_soft_actions = _action_from_probabilities(
        shuffled_actions,
        np.asarray(shuffled_probabilities),
        fixed_gripper_age=args.fixed_gripper_age,
    )
    shuffled_soft_actions[:, 6] = test_flat["cached_actions"][:, args.fixed_gripper_age, 6]
    evaluations["learned_current_shuffled_plan_and_candidates_soft_mixture_diagnostic"] = (
        _stratified_metrics(
            shuffled_soft_actions,
            None,
            oracle_tokens,
            test_flat,
        )
    )
    fixed4_mse = evaluations["fixed_token4"]["overall"]["action_mse_6d"]
    oracle_mse = evaluations["dense_discrete_oracle"]["overall"]["action_mse_6d"]
    oracle_gap = fixed4_mse - oracle_mse
    for evaluation in evaluations.values():
        current_mse = evaluation["overall"]["action_mse_6d"]
        evaluation["overall"]["relative_improvement_6d_vs_fixed_token4"] = (
            float((fixed4_mse - current_mse) / fixed4_mse) if fixed4_mse > 0 else None
        )
        evaluation["overall"]["gap_closure_6d_fixed4_to_discrete_oracle"] = (
            float((fixed4_mse - current_mse) / oracle_gap) if oracle_gap > 1e-12 else None
        )

    summary = {
        "method": {
            "name": "event_option_cached_action_token_selector",
            "causal_bottleneck": (
                "observation -> ten logits -> one cached action token; dimensions 0:6 only"
            ),
            "forbidden_path": "no observation-to-action residual and no action decoder",
            "training_relaxation": (
                "softmax mixture action MSE plus cost-derived soft-target cross entropy"
            ),
            "deployable_output": "hard argmax cached-action token",
            "gripper_rule": f"dimension 6 always comes from cached token index {args.fixed_gripper_age}",
        },
        "args": dataclasses.asdict(args),
        "split": {
            name: {
                "root_count": len(roots),
                "valid_branch_count": len(flat["branch_id"]),
                "episodes_by_task": phase_probe._episode_summary(arrays, roots),
            }
            for name, (roots, flat) in partitions.items()
        },
        "train_calibration": {
            "global_discrete_token": global_token,
            "global_continuous_token": global_continuous,
            "privileged_branch_id_token": {
                branched_dataset.BRANCH_NAMES[branch_id]: token
                for branch_id, token in branch_tokens.items()
            },
        },
        "train": {
            "current": current_train,
            "no_current_matched_independent": no_current_train,
        },
        "test": {
            "root_count": len(test_roots),
            "branch_count": len(test_flat["branch_id"]),
            "discrete_oracle_token_mean": float(np.mean(oracle_tokens)),
            "shuffled_token_permutation": token_permutation.tolist(),
            "evaluations": evaluations,
        },
        "interpretation_guard": (
            "This is an offline same-root selector probe. Soft mixtures and continuous calibration are "
            "diagnostics, while only hard argmax rows are deployable under the token bottleneck. It does "
            "not establish closed-loop success or end-to-end wall-clock speedup."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
