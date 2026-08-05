"""GPU-native JAX Phase-A oracle probe for a causal clean-plan compiler.

This is the JAX counterpart of ``train_causal_clean_plan_compiler.py``.  It
keeps the same episode-held-out split, semantic-intervention sampling, active-7
training objective, and held-out diagnostics while avoiding PyTorch CUDA.  The
model receives only a teacher EAR endpoint and the matched final-action noise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from typing import Any, Iterator, Sequence

from flax import linen as nn
from flax import traverse_util
from flax.core import unfreeze
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.action_cot import endpoint_dataset


@dataclasses.dataclass(frozen=True)
class ProbeArgs:
    dataset: tuple[str, ...]
    output_dir: str
    steps: int = 1_000
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 7
    log_interval: int = 25
    clean_loss_weight: float = 1.0
    intervention_loss_weight: float = 1.0
    response_loss_weight: float = 1.0
    plan_reconstruction_loss_weight: float = 0.25
    model_dim: int = 256
    feedforward_dim: int = 512
    plan_encoder_layers: int = 2
    attention_heads: int = 8
    latent_tokens: int = 4
    active_action_dim: int = 7
    dropout: float = 0.0
    device: str = "gpu"
    amp: str = "bfloat16"
    latency_warmup: int = 20
    latency_runs: int = 100
    overwrite: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--intervention-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--plan-reconstruction-loss-weight", type=float, default=0.25)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--plan-encoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--latent-tokens", type=int, default=4)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args() -> ProbeArgs:
    values = vars(_parser().parse_args())
    values["dataset"] = tuple(values["dataset"])
    return ProbeArgs(**values)


def _validate_args(args: ProbeArgs) -> None:
    for name in (
        "steps", "batch_size", "eval_batch_size", "warmup_steps", "log_interval",
        "model_dim", "feedforward_dim", "plan_encoder_layers", "attention_heads",
        "latent_tokens", "active_action_dim", "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.latency_warmup < 0:
        raise ValueError("--latency-warmup must be non-negative.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.learning_rate <= 0 or not 0 <= args.final_learning_rate <= args.learning_rate:
        raise ValueError("Learning rates are invalid.")
    if args.gradient_clip_norm <= 0 or args.weight_decay < 0:
        raise ValueError("Gradient clipping and weight decay values are invalid.")
    if min(
        args.clean_loss_weight, args.intervention_loss_weight,
        args.response_loss_weight, args.plan_reconstruction_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.model_dim % args.attention_heads:
        raise ValueError("--model-dim must be divisible by --attention-heads.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")


def _resolve_device(name: str) -> jax.Device:
    if name in {"gpu", "auto"}:
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if devices:
            return devices[0]
        if name == "gpu":
            raise RuntimeError(
                "No JAX GPU device is available. This script defaults to GPU and will not "
                "silently fall back to CPU; check jax.devices() and the CUDA JAX plugin."
            )
    return jax.devices("cpu")[0]


def _semantic_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    return np.asarray(arrays["intervention_valid"], dtype=np.bool_) & (
        np.asarray(arrays["intervention_ids"]) != null_id
    )


def _split_indices(
    arrays: dict[str, np.ndarray], *, validation_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly match the episode-held-out split used by the PyTorch probe."""
    eligible = np.any(_semantic_mask(arrays), axis=-1)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 2:
        raise ValueError("Endpoint dataset has fewer than two semantic-intervention records.")
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups[eligible_indices])
    rng = np.random.default_rng(seed)
    if unique_groups.size >= 2:
        rng.shuffle(unique_groups)
        validation_count = max(1, round(unique_groups.size * validation_fraction))
        validation_groups = unique_groups[:validation_count]
        validation_mask = eligible & np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(eligible & ~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        shuffled = eligible_indices.copy()
        rng.shuffle(shuffled)
        validation_count = min(max(1, round(shuffled.size * validation_fraction)), shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Train/validation split produced an empty partition.")
    return train_indices, validation_indices


def _choose_interventions(
    arrays: dict[str, np.ndarray], row_indices: np.ndarray, rng: np.random.Generator,
    *, deterministic: bool,
) -> np.ndarray:
    semantic = _semantic_mask(arrays)
    selected = np.empty(row_indices.shape[0], dtype=np.int64)
    for offset, row_index in enumerate(row_indices):
        candidates = np.flatnonzero(semantic[row_index])
        if not candidates.size:
            raise RuntimeError(f"Row {row_index} has no semantic intervention after filtering.")
        selected[offset] = int(candidates[0] if deterministic else rng.choice(candidates))
    return selected


def _make_batch(
    arrays: dict[str, np.ndarray], row_indices: np.ndarray, intervention_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "clean_plan": np.asarray(arrays["clean_coarse"][row_indices], dtype=np.float32),
        "clean_actions": np.asarray(arrays["clean_actions"][row_indices], dtype=np.float32),
        "action_noise": np.asarray(arrays["action_noise"][row_indices], dtype=np.float32),
        "intervention_plan": np.asarray(
            arrays["intervention_coarse"][row_indices, intervention_indices], dtype=np.float32,
        ),
        "intervention_actions": np.asarray(
            arrays["intervention_actions"][row_indices, intervention_indices], dtype=np.float32,
        ),
    }


class _TransformerBlock(nn.Module):
    dimension: int
    heads: int
    feedforward_dim: int
    dropout: float
    compute_dtype: Any

    @nn.compact
    def __call__(self, tokens: jax.Array, *, training: bool) -> jax.Array:
        normalized = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(tokens)
        attended = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.dimension,
            out_features=self.dimension,
            dropout_rate=self.dropout,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(normalized, deterministic=not training)
        tokens = tokens + nn.Dropout(rate=self.dropout)(attended, deterministic=not training)
        normalized = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(tokens)
        hidden = nn.Dense(
            self.feedforward_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
        )(normalized)
        hidden = nn.gelu(hidden, approximate=True)
        hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=not training)
        hidden = nn.Dense(
            self.dimension, dtype=self.compute_dtype, param_dtype=jnp.float32,
        )(hidden)
        return tokens + nn.Dropout(rate=self.dropout)(hidden, deterministic=not training)


class _CrossFeedForwardBlock(nn.Module):
    dimension: int
    heads: int
    feedforward_dim: int
    dropout: float
    compute_dtype: Any

    @nn.compact
    def __call__(
        self, queries: jax.Array, memory: jax.Array, *, training: bool,
    ) -> jax.Array:
        normalized_queries = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32, name="query_norm",
        )(queries)
        normalized_memory = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32, name="memory_norm",
        )(memory)
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.heads,
            qkv_features=self.dimension,
            out_features=self.dimension,
            dropout_rate=self.dropout,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="cross_attention",
        )(normalized_queries, normalized_memory, deterministic=not training)
        values = queries + nn.Dropout(
            rate=self.dropout, name="attention_dropout",
        )(attended, deterministic=not training)
        normalized = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32, name="feedforward_norm",
        )(values)
        hidden = nn.Dense(
            self.feedforward_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="feedforward_in",
        )(normalized)
        hidden = nn.gelu(hidden, approximate=True)
        hidden = nn.Dropout(rate=self.dropout, name="feedforward_dropout_1")(
            hidden, deterministic=not training,
        )
        hidden = nn.Dense(
            self.dimension, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="feedforward_out",
        )(hidden)
        hidden = nn.Dropout(rate=self.dropout, name="feedforward_dropout_2")(
            hidden, deterministic=not training,
        )
        return values + hidden


class CausalCleanPlanCompilerJax(nn.Module):
    action_dim: int
    plan_horizon: int
    action_horizon: int
    model_dim: int
    feedforward_dim: int
    plan_encoder_layers: int
    attention_heads: int
    latent_tokens: int
    active_action_dim: int
    dropout: float
    compute_dtype: Any

    @nn.compact
    def __call__(
        self, plan: jax.Array, action_noise: jax.Array, *, training: bool,
        return_reconstruction: bool = True,
    ) -> tuple[jax.Array, jax.Array | None]:
        plan_positions = self.param(
            "plan_positions", nn.initializers.truncated_normal(stddev=0.02),
            (1, self.plan_horizon, self.model_dim), jnp.float32,
        )
        plan_tokens = nn.Dense(
            self.model_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="plan_input",
        )(plan) + plan_positions.astype(self.compute_dtype)
        for layer in range(self.plan_encoder_layers):
            plan_tokens = _TransformerBlock(
                self.model_dim, self.attention_heads, self.feedforward_dim,
                self.dropout, self.compute_dtype, name=f"plan_encoder_{layer}",
            )(plan_tokens, training=training)
        plan_tokens = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32, name="plan_encoder_norm",
        )(plan_tokens)
        latent_queries = self.param(
            "latent_queries", nn.initializers.truncated_normal(stddev=0.02),
            (1, self.latent_tokens, self.model_dim), jnp.float32,
        )
        latent_queries = jnp.broadcast_to(
            latent_queries.astype(self.compute_dtype),
            (plan.shape[0], self.latent_tokens, self.model_dim),
        )
        latents = _CrossFeedForwardBlock(
            self.model_dim, self.attention_heads, self.feedforward_dim,
            self.dropout, self.compute_dtype, name="latent_compiler",
        )(latent_queries, plan_tokens, training=training)

        action_positions = self.param(
            "action_positions", nn.initializers.truncated_normal(stddev=0.02),
            (1, self.action_horizon, self.model_dim), jnp.float32,
        )
        action_queries = nn.Dense(
            self.model_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="noise_input",
        )(action_noise) + action_positions.astype(self.compute_dtype)
        normalized = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32, name="action_self_norm",
        )(action_queries)
        attended = nn.SelfAttention(
            num_heads=self.attention_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.dropout,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="action_self_attention",
        )(normalized, deterministic=not training)
        action_queries = action_queries + nn.Dropout(
            rate=self.dropout, name="action_self_dropout",
        )(attended, deterministic=not training)
        action_tokens = _CrossFeedForwardBlock(
            self.model_dim, self.attention_heads, self.feedforward_dim,
            self.dropout, self.compute_dtype, name="action_cross_decoder",
        )(action_queries, latents, training=training)
        endpoint = nn.Dense(
            self.action_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="action_output",
        )(action_tokens)
        active_mask = (jnp.arange(self.action_dim) < self.active_action_dim).astype(endpoint.dtype)
        endpoint = endpoint * active_mask

        if not return_reconstruction:
            return endpoint, None
        reconstruction_queries = self.param(
            "reconstruction_queries", nn.initializers.truncated_normal(stddev=0.02),
            (1, self.plan_horizon, self.model_dim), jnp.float32,
        )
        reconstruction_queries = jnp.broadcast_to(
            reconstruction_queries.astype(self.compute_dtype),
            (plan.shape[0], self.plan_horizon, self.model_dim),
        )
        normalized_queries = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="reconstruction_query_norm",
        )(reconstruction_queries)
        normalized_latents = nn.LayerNorm(
            dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="reconstruction_memory_norm",
        )(latents)
        decoded = nn.MultiHeadDotProductAttention(
            num_heads=self.attention_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.dropout,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="reconstruction_attention",
        )(normalized_queries, normalized_latents, deterministic=not training)
        reconstruction = nn.Dense(
            self.action_dim, dtype=self.compute_dtype, param_dtype=jnp.float32,
            name="reconstruction_output",
        )(reconstruction_queries + decoded)
        return endpoint, reconstruction * active_mask


def _losses(
    model: CausalCleanPlanCompilerJax,
    params: Any,
    batch: dict[str, jax.Array],
    args: ProbeArgs,
    dropout_key: jax.Array,
    *,
    training: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    variables = {"params": params}
    rngs = {"dropout": dropout_key}
    clean_prediction, clean_reconstruction = model.apply(
        variables, batch["clean_plan"], batch["action_noise"],
        training=training, return_reconstruction=True, rngs=rngs,
    )
    intervention_prediction, intervention_reconstruction = model.apply(
        variables, batch["intervention_plan"], batch["action_noise"],
        training=training, return_reconstruction=True,
        rngs={"dropout": jax.random.fold_in(dropout_key, 1)},
    )
    active = slice(0, args.active_action_dim)

    def mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(prediction.astype(jnp.float32) - target.astype(jnp.float32)))

    clean_mse = mse(clean_prediction[..., active], batch["clean_actions"][..., active])
    clean_mse_full = mse(clean_prediction, batch["clean_actions"])
    intervention_mse = mse(
        intervention_prediction[..., active], batch["intervention_actions"][..., active],
    )
    intervention_mse_full = mse(intervention_prediction, batch["intervention_actions"])
    predicted_response = intervention_prediction - clean_prediction
    teacher_response = batch["intervention_actions"] - batch["clean_actions"]
    response_mse = mse(predicted_response[..., active], teacher_response[..., active])
    response_mse_full = mse(predicted_response, teacher_response)
    clean_reconstruction_mse = mse(
        clean_reconstruction[..., active], batch["clean_plan"][..., active],
    )
    clean_reconstruction_mse_full = mse(clean_reconstruction, batch["clean_plan"])
    intervention_reconstruction_mse = mse(
        intervention_reconstruction[..., active], batch["intervention_plan"][..., active],
    )
    intervention_reconstruction_mse_full = mse(
        intervention_reconstruction, batch["intervention_plan"],
    )
    reconstruction_mse = 0.5 * (clean_reconstruction_mse + intervention_reconstruction_mse)
    reconstruction_mse_full = 0.5 * (
        clean_reconstruction_mse_full + intervention_reconstruction_mse_full
    )
    total = (
        args.clean_loss_weight * clean_mse
        + args.intervention_loss_weight * intervention_mse
        + args.response_loss_weight * response_mse
        + args.plan_reconstruction_loss_weight * reconstruction_mse
    )
    return total, {
        "total_loss": total,
        "clean_action_mse_active7": clean_mse,
        "clean_action_mse_full32": clean_mse_full,
        "intervention_action_mse_active7": intervention_mse,
        "intervention_action_mse_full32": intervention_mse_full,
        "response_mse_active7": response_mse,
        "response_mse_full32": response_mse_full,
        "plan_reconstruction_mse_active7": reconstruction_mse,
        "plan_reconstruction_mse_full32": reconstruction_mse_full,
        "clean_plan_reconstruction_mse_active7": clean_reconstruction_mse,
        "clean_plan_reconstruction_mse_full32": clean_reconstruction_mse_full,
        "intervention_plan_reconstruction_mse_active7": intervention_reconstruction_mse,
        "intervention_plan_reconstruction_mse_full32": intervention_reconstruction_mse_full,
    }


def _learning_rate(args: ProbeArgs, step: int) -> float:
    if step <= args.warmup_steps:
        return args.learning_rate * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * cosine


def _schedule(args: ProbeArgs):
    def schedule(count: jax.Array) -> jax.Array:
        step = count.astype(jnp.float32) + 1.0
        warmup = args.learning_rate * step / float(args.warmup_steps)
        progress = (step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay = args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * cosine
        return jnp.where(step <= args.warmup_steps, warmup, decay)
    return schedule


def _index_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, indices.size, batch_size):
        yield indices[start : start + batch_size]


def _all_semantic_pairs(
    arrays: dict[str, np.ndarray], validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    validation_offsets: list[int] = []
    interventions: list[int] = []
    semantic = _semantic_mask(arrays)
    for validation_offset, row_index in enumerate(validation_indices):
        for intervention_index in np.flatnonzero(semantic[row_index]):
            rows.append(int(row_index))
            validation_offsets.append(validation_offset)
            interventions.append(int(intervention_index))
    if not rows:
        raise RuntimeError("Held-out split contains no semantic intervention pairs.")
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(validation_offsets, dtype=np.int64),
        np.asarray(interventions, dtype=np.int64),
    )


def _same_task_shuffle_sources(
    arrays: dict[str, np.ndarray], validation_indices: np.ndarray, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tasks = np.asarray(arrays["task_id"])[validation_indices]
    episodes = np.asarray(arrays["episode_id"])[validation_indices]
    target_offsets: list[int] = []
    source_offsets: list[int] = []
    for target_offset in range(validation_indices.size):
        candidates = np.flatnonzero(
            (tasks == tasks[target_offset]) & (episodes != episodes[target_offset])
        )
        if not candidates.size:
            candidates = np.flatnonzero(tasks == tasks[target_offset])
            candidates = candidates[candidates != target_offset]
        if not candidates.size:
            continue
        target_offsets.append(target_offset)
        source_offsets.append(int(rng.choice(candidates)))
    return np.asarray(target_offsets, dtype=np.int64), np.asarray(source_offsets, dtype=np.int64)


def _batched_predict(
    predict_step: Any,
    params: Any,
    plans: np.ndarray,
    noise: np.ndarray,
    *,
    batch_size: int,
    device: jax.Device,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    positions = np.arange(plans.shape[0], dtype=np.int64)
    for selected in _index_batches(positions, batch_size):
        valid = selected.size
        selected_plans = np.asarray(plans[selected], dtype=np.float32)
        selected_noise = np.asarray(noise[selected], dtype=np.float32)
        if valid < batch_size:
            pad = batch_size - valid
            selected_plans = np.pad(selected_plans, ((0, pad), (0, 0), (0, 0)))
            selected_noise = np.pad(selected_noise, ((0, pad), (0, 0), (0, 0)))
        endpoint, reconstruction = predict_step(
            params,
            jax.device_put(selected_plans, device),
            jax.device_put(selected_noise, device),
        )
        endpoint, reconstruction = jax.device_get((endpoint, reconstruction))
        endpoints.append(np.asarray(endpoint[:valid], dtype=np.float32))
        reconstructions.append(np.asarray(reconstruction[:valid], dtype=np.float32))
    return np.concatenate(endpoints), np.concatenate(reconstructions)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(np.square(delta)))


def _cosine_mean(prediction: np.ndarray, target: np.ndarray) -> float:
    left = prediction.astype(np.float64).reshape(prediction.shape[0], -1)
    right = target.astype(np.float64).reshape(target.shape[0], -1)
    denominator = np.maximum(np.linalg.norm(left, axis=-1), 1e-8) * np.maximum(
        np.linalg.norm(right, axis=-1), 1e-8,
    )
    return float(np.mean(np.sum(left * right, axis=-1) / denominator))


def _paired_metrics(
    intervention_predictions: np.ndarray,
    intervention_targets: np.ndarray,
    clean_predictions: np.ndarray,
    clean_targets: np.ndarray,
    intervention_reconstructions: np.ndarray,
    intervention_plans: np.ndarray,
    *,
    active_action_dim: int,
) -> dict[str, float]:
    active = slice(0, active_action_dim)
    predicted_response = intervention_predictions - clean_predictions
    teacher_response = intervention_targets - clean_targets
    intervention_active_mse = _mse(
        intervention_predictions[..., active], intervention_targets[..., active],
    )
    intervention_full_mse = _mse(intervention_predictions, intervention_targets)
    response_active_mse = _mse(
        predicted_response[..., active], teacher_response[..., active],
    )
    response_full_mse = _mse(predicted_response, teacher_response)
    return {
        "intervention_action_mse_active7": intervention_active_mse,
        "intervention_action_rmse_active7": math.sqrt(intervention_active_mse),
        "intervention_action_mse_full32": intervention_full_mse,
        "intervention_action_rmse_full32": math.sqrt(intervention_full_mse),
        "response_mse_active7": response_active_mse,
        "response_rmse_active7": math.sqrt(response_active_mse),
        "response_mse_full32": response_full_mse,
        "response_rmse_full32": math.sqrt(response_full_mse),
        "response_cosine_active7": _cosine_mean(
            predicted_response[..., active], teacher_response[..., active],
        ),
        "response_cosine_full32": _cosine_mean(predicted_response, teacher_response),
        "intervention_plan_reconstruction_mse_active7": _mse(
            intervention_reconstructions[..., active], intervention_plans[..., active],
        ),
        "intervention_plan_reconstruction_mse_full32": _mse(
            intervention_reconstructions, intervention_plans,
        ),
    }


def _task_counts(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, int]:
    tasks, counts = np.unique(np.asarray(arrays["task_id"])[indices], return_counts=True)
    return {str(int(task)): int(count) for task, count in zip(tasks, counts, strict=True)}


def _episode_group_count(arrays: dict[str, np.ndarray], indices: np.ndarray) -> int:
    keys = np.stack(
        (np.asarray(arrays["task_id"])[indices], np.asarray(arrays["episode_id"])[indices]),
        axis=-1,
    )
    return int(np.unique(keys, axis=0).shape[0])


def _full_validation(
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    params: Any,
    predict_step: Any,
    *,
    batch_size: int,
    device: jax.Device,
    active_action_dim: int,
    seed: int,
) -> dict[str, Any]:
    active = slice(0, active_action_dim)
    clean_plans = np.asarray(arrays["clean_coarse"][validation_indices], dtype=np.float32)
    clean_noise = np.asarray(arrays["action_noise"][validation_indices], dtype=np.float32)
    clean_targets = np.asarray(arrays["clean_actions"][validation_indices], dtype=np.float32)
    clean_predictions, clean_reconstructions = _batched_predict(
        predict_step, params, clean_plans, clean_noise,
        batch_size=batch_size, device=device,
    )
    clean_active_mse = _mse(clean_predictions[..., active], clean_targets[..., active])
    clean_full_mse = _mse(clean_predictions, clean_targets)
    metrics: dict[str, Any] = {
        "clean_action_mse_active7": clean_active_mse,
        "clean_action_rmse_active7": math.sqrt(clean_active_mse),
        "clean_action_mse_full32": clean_full_mse,
        "clean_action_rmse_full32": math.sqrt(clean_full_mse),
        "clean_plan_reconstruction_mse_active7": _mse(
            clean_reconstructions[..., active], clean_plans[..., active],
        ),
        "clean_plan_reconstruction_mse_full32": _mse(clean_reconstructions, clean_plans),
    }

    rows, clean_offsets, intervention_slots = _all_semantic_pairs(arrays, validation_indices)
    intervention_plans = np.asarray(
        arrays["intervention_coarse"][rows, intervention_slots], dtype=np.float32,
    )
    intervention_targets = np.asarray(
        arrays["intervention_actions"][rows, intervention_slots], dtype=np.float32,
    )
    intervention_noise = np.asarray(arrays["action_noise"][rows], dtype=np.float32)
    intervention_predictions, intervention_reconstructions = _batched_predict(
        predict_step, params, intervention_plans, intervention_noise,
        batch_size=batch_size, device=device,
    )
    aggregate_pairs = _paired_metrics(
        intervention_predictions, intervention_targets,
        clean_predictions[clean_offsets], np.asarray(arrays["clean_actions"][rows], dtype=np.float32),
        intervention_reconstructions, intervention_plans,
        active_action_dim=active_action_dim,
    )
    metrics.update({"semantic_intervention_pairs": int(rows.size), **aggregate_pairs})

    pair_intervention_ids = np.asarray(arrays["intervention_ids"])[rows, intervention_slots]
    per_intervention: dict[str, Any] = {}
    for intervention_id in np.unique(pair_intervention_ids):
        selected = np.flatnonzero(pair_intervention_ids == intervention_id)
        name = endpoint_dataset.INTERVENTION_NAMES[int(intervention_id)]
        per_intervention[name] = {
            "semantic_intervention_pairs": int(selected.size),
            **_paired_metrics(
                intervention_predictions[selected], intervention_targets[selected],
                clean_predictions[clean_offsets[selected]],
                np.asarray(arrays["clean_actions"][rows[selected]], dtype=np.float32),
                intervention_reconstructions[selected], intervention_plans[selected],
                active_action_dim=active_action_dim,
            ),
        }
    metrics["per_intervention_metrics"] = per_intervention

    validation_tasks = np.asarray(arrays["task_id"])[validation_indices]
    pair_tasks = np.asarray(arrays["task_id"])[rows]
    per_task: dict[str, Any] = {}
    for task_id in np.unique(validation_tasks):
        clean_selected = np.flatnonzero(validation_tasks == task_id)
        pair_selected = np.flatnonzero(pair_tasks == task_id)
        task_clean_active = _mse(
            clean_predictions[clean_selected, ..., active], clean_targets[clean_selected, ..., active],
        )
        task_clean_full = _mse(clean_predictions[clean_selected], clean_targets[clean_selected])
        per_task[str(int(task_id))] = {
            "validation_records": int(clean_selected.size),
            "semantic_intervention_pairs": int(pair_selected.size),
            "clean_action_mse_active7": task_clean_active,
            "clean_action_rmse_active7": math.sqrt(task_clean_active),
            "clean_action_mse_full32": task_clean_full,
            "clean_action_rmse_full32": math.sqrt(task_clean_full),
            "clean_plan_reconstruction_mse_active7": _mse(
                clean_reconstructions[clean_selected, ..., active],
                clean_plans[clean_selected, ..., active],
            ),
            "clean_plan_reconstruction_mse_full32": _mse(
                clean_reconstructions[clean_selected], clean_plans[clean_selected],
            ),
            **_paired_metrics(
                intervention_predictions[pair_selected], intervention_targets[pair_selected],
                clean_predictions[clean_offsets[pair_selected]],
                np.asarray(arrays["clean_actions"][rows[pair_selected]], dtype=np.float32),
                intervention_reconstructions[pair_selected], intervention_plans[pair_selected],
                active_action_dim=active_action_dim,
            ),
        }
    metrics["per_task_metrics"] = per_task

    target_offsets, source_offsets = _same_task_shuffle_sources(
        arrays, validation_indices, seed=seed + 2,
    )
    if target_offsets.size:
        target_rows = validation_indices[target_offsets]
        source_rows = validation_indices[source_offsets]
        shuffled_predictions, _ = _batched_predict(
            predict_step, params,
            np.asarray(arrays["clean_coarse"][source_rows], dtype=np.float32),
            np.asarray(arrays["action_noise"][target_rows], dtype=np.float32),
            batch_size=batch_size, device=device,
        )
        shuffle_targets = np.asarray(arrays["clean_actions"][target_rows], dtype=np.float32)
        correct_predictions = clean_predictions[target_offsets]
        correct_active = _mse(correct_predictions[..., active], shuffle_targets[..., active])
        shuffled_active = _mse(shuffled_predictions[..., active], shuffle_targets[..., active])
        correct_full = _mse(correct_predictions, shuffle_targets)
        shuffled_full = _mse(shuffled_predictions, shuffle_targets)
        metrics.update({
            "same_task_shuffle_records": int(target_offsets.size),
            "shuffle_correct_action_mse_active7": correct_active,
            "same_task_shuffled_ear_action_mse_active7": shuffled_active,
            "same_task_shuffled_ear_action_mse_gap_active7": shuffled_active - correct_active,
            "shuffle_correct_action_mse_full32": correct_full,
            "same_task_shuffled_ear_action_mse_full32": shuffled_full,
            "same_task_shuffled_ear_action_mse_gap_full32": shuffled_full - correct_full,
        })
        shuffle_tasks = np.asarray(arrays["task_id"])[target_rows]
        shuffle_per_task: dict[str, Any] = {}
        for task_id in np.unique(shuffle_tasks):
            selected = np.flatnonzero(shuffle_tasks == task_id)
            task_correct_active = _mse(
                correct_predictions[selected, ..., active], shuffle_targets[selected, ..., active],
            )
            task_shuffled_active = _mse(
                shuffled_predictions[selected, ..., active], shuffle_targets[selected, ..., active],
            )
            task_correct_full = _mse(correct_predictions[selected], shuffle_targets[selected])
            task_shuffled_full = _mse(shuffled_predictions[selected], shuffle_targets[selected])
            shuffle_per_task[str(int(task_id))] = {
                "same_task_shuffle_records": int(selected.size),
                "shuffle_correct_action_mse_active7": task_correct_active,
                "same_task_shuffled_ear_action_mse_active7": task_shuffled_active,
                "same_task_shuffled_ear_action_mse_gap_active7": (
                    task_shuffled_active - task_correct_active
                ),
                "shuffle_correct_action_mse_full32": task_correct_full,
                "same_task_shuffled_ear_action_mse_full32": task_shuffled_full,
                "same_task_shuffled_ear_action_mse_gap_full32": (
                    task_shuffled_full - task_correct_full
                ),
            }
        metrics["same_task_shuffle_per_task"] = shuffle_per_task
    else:
        metrics.update({
            "same_task_shuffle_records": 0,
            "shuffle_correct_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_gap_active7": None,
            "shuffle_correct_action_mse_full32": None,
            "same_task_shuffled_ear_action_mse_full32": None,
            "same_task_shuffled_ear_action_mse_gap_full32": None,
            "same_task_shuffle_per_task": {},
        })
    return metrics


def _latency_metrics(
    predict_step: Any,
    params: Any,
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    device: jax.Device,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    row = validation_indices[:1]
    plan = jax.device_put(np.asarray(arrays["clean_coarse"][row], dtype=np.float32), device)
    noise = jax.device_put(np.asarray(arrays["action_noise"][row], dtype=np.float32), device)
    for _ in range(warmup):
        endpoint, _ = predict_step(params, plan, noise)
        jax.block_until_ready(endpoint)
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        endpoint, _ = predict_step(params, plan, noise)
        jax.block_until_ready(endpoint)
        durations.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(durations, dtype=np.float64)
    is_gpu = device.platform == "gpu"
    return {
        "latency_device": str(device),
        "latency_batch_size": 1,
        "latency_runs": runs,
        "latency_mean_ms": float(values.mean()),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "gpu_latency_mean_ms": float(values.mean()) if is_gpu else None,
        "gpu_latency_p95_ms": float(np.percentile(values, 95)) if is_gpu else None,
    }


def _save_checkpoint(
    path: pathlib.Path,
    params: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    steps: int,
    best_step: int,
) -> None:
    flat_params = traverse_util.flatten_dict(unfreeze(params), sep="/")
    payload: dict[str, np.ndarray] = {
        f"params/{name}": np.asarray(jax.device_get(value))
        for name, value in flat_params.items()
    }
    payload["metadata/completed_steps"] = np.asarray(steps, dtype=np.int64)
    payload["metadata/best_step"] = np.asarray(best_step, dtype=np.int64)
    payload["metadata/last_step"] = np.asarray(steps, dtype=np.int64)
    payload["metadata/train_indices"] = np.asarray(train_indices, dtype=np.int64)
    payload["metadata/validation_indices"] = np.asarray(validation_indices, dtype=np.int64)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main(args: ProbeArgs) -> None:
    _validate_args(args)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    compute_dtype = jnp.bfloat16 if args.amp == "bfloat16" else jnp.float32
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.npz"
    existing = [path for path in (metrics_path, summary_path, checkpoint_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Probe outputs already exist: {[str(path) for path in existing]}; "
            "choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(
        arrays, validation_fraction=args.validation_fraction, seed=args.seed,
    )
    action_dim = int(arrays["clean_actions"].shape[-1])
    plan_horizon = int(arrays["clean_coarse"].shape[1])
    action_horizon = int(arrays["clean_actions"].shape[1])
    if arrays["clean_coarse"].shape[-1] != action_dim:
        raise ValueError("EAR and final endpoints must share action_dim.")
    if arrays["action_noise"].shape[1:] != arrays["clean_actions"].shape[1:]:
        raise ValueError("action_noise and clean_actions shapes do not match.")
    if not 0 < args.active_action_dim <= action_dim:
        raise ValueError(f"active_action_dim must be in [1, {action_dim}].")

    model = CausalCleanPlanCompilerJax(
        action_dim=action_dim,
        plan_horizon=plan_horizon,
        action_horizon=action_horizon,
        model_dim=args.model_dim,
        feedforward_dim=args.feedforward_dim,
        plan_encoder_layers=args.plan_encoder_layers,
        attention_heads=args.attention_heads,
        latent_tokens=args.latent_tokens,
        active_action_dim=args.active_action_dim,
        dropout=args.dropout,
        compute_dtype=compute_dtype,
    )
    init_key, dropout_key = jax.random.split(jax.random.PRNGKey(args.seed))
    dummy_plan = jnp.zeros((1, plan_horizon, action_dim), dtype=jnp.float32)
    dummy_noise = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)
    with jax.default_device(device):
        variables = model.init(
            {"params": init_key, "dropout": dropout_key},
            dummy_plan, dummy_noise, training=True, return_reconstruction=True,
        )
    parameter_count = int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(variables["params"])))
    if not 1_000_000 <= parameter_count <= 3_000_000:
        print(f"WARNING: model has {parameter_count:,} parameters; Phase-A target is 1-3M.", flush=True)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(_schedule(args), weight_decay=args.weight_decay),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=jax.device_put(variables["params"], device),
        tx=optimizer,
    )

    @jax.jit
    def train_step(
        current_state: train_state.TrainState,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[train_state.TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _losses(model, params, batch, args, key, training=True)
        (_, values), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_state.params)
        values = {**values, "gradient_norm": optax.global_norm(grads)}
        return current_state.apply_gradients(grads=grads), values

    @jax.jit
    def validation_step(params: Any, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        _, values = _losses(
            model, params, batch, args, jax.random.PRNGKey(0), training=False,
        )
        return values

    @jax.jit
    def predict_step(
        params: Any, plan: jax.Array, noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        endpoint, reconstruction = model.apply(
            {"params": params}, plan, noise,
            training=False, return_reconstruction=True,
        )
        return endpoint, reconstruction

    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    training_key = jax.random.PRNGKey(args.seed + 10)
    started = time.monotonic()
    print(
        f"Initialized JAX causal clean-plan compiler: train={train_indices.size} "
        f"validation={validation_indices.size} params={parameter_count:,} "
        f"device={device} amp={args.amp}",
        flush=True,
    )

    last_record: dict[str, Any] = {}
    best_record: dict[str, Any] = {}
    best_params: Any | None = None
    best_score: tuple[float, float] | None = None
    best_step = 0
    metrics_mode = "w" if args.overwrite else "a"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            rows = rng.choice(
                train_indices, size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            intervention_indices = _choose_interventions(
                arrays, rows, rng, deterministic=False,
            )
            numpy_batch = _make_batch(arrays, rows, intervention_indices)
            batch = jax.device_put(numpy_batch, device)
            training_key, step_key = jax.random.split(training_key)
            state, train_metrics = train_step(state, batch, step_key)

            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            if not should_log:
                continue
            validation_rows = validation_rng.choice(
                validation_indices,
                size=min(args.batch_size, validation_indices.size),
                replace=False,
            )
            validation_interventions = _choose_interventions(
                arrays, validation_rows, validation_rng, deterministic=True,
            )
            validation_batch = jax.device_put(
                _make_batch(arrays, validation_rows, validation_interventions), device,
            )
            validation_metrics = validation_step(state.params, validation_batch)
            train_host, validation_host = jax.device_get((train_metrics, validation_metrics))
            candidate_score = (
                float(validation_host["response_mse_active7"]),
                float(validation_host["clean_action_mse_active7"]),
            )
            selected_as_best = best_score is None or candidate_score < best_score
            last_record = {
                "phase": "train",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": _learning_rate(args, step),
                "selected_as_best_checkpoint": selected_as_best,
                **{f"train/{name}": float(value) for name, value in train_host.items()},
                **{
                    f"validation_sample/{name}": float(value)
                    for name, value in validation_host.items()
                },
            }
            if selected_as_best:
                # No parameter donation is used, so this immutable pytree remains
                # the exact device-resident validation checkpoint.
                best_params = state.params
                best_score = candidate_score
                best_step = step
                best_record = dict(last_record)
            metrics_file.write(json.dumps(last_record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"step={step} train_total={last_record['train/total_loss']:.6f} "
                f"val_clean7={last_record['validation_sample/clean_action_mse_active7']:.6f} "
                f"val_response7={last_record['validation_sample/response_mse_active7']:.6f} "
                f"best={selected_as_best}",
                flush=True,
            )

        if best_params is None:
            raise RuntimeError("Training completed without selecting a validation checkpoint.")
        jax.block_until_ready(best_params)
        full_metrics = _full_validation(
            arrays, validation_indices, best_params, predict_step,
            batch_size=args.eval_batch_size, device=device,
            active_action_dim=args.active_action_dim, seed=args.seed,
        )
        full_metrics.update(_latency_metrics(
            predict_step, best_params, arrays, validation_indices,
            device=device, warmup=args.latency_warmup, runs=args.latency_runs,
        ))
        final_record = {
            "phase": "full_validation",
            "step": args.steps,
            "evaluated_checkpoint_step": best_step,
            "elapsed_seconds": time.monotonic() - started,
            **full_metrics,
        }
        metrics_file.write(json.dumps(final_record, sort_keys=True) + "\n")
        metrics_file.flush()

    _save_checkpoint(
        checkpoint_path, best_params, train_indices, validation_indices, args.steps, best_step,
    )
    summary = {
        "probe": "phase_a_causal_clean_plan_compiler_oracle_jax_gpu",
        "dataset": list(args.dataset),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "args": dataclasses.asdict(args),
        "input_contract": {
            "model_inputs": ["teacher_ear_endpoint", "shared_action_noise"],
            "clean_and_intervention_share_action_noise": True,
            "forbidden_inputs": ["observation", "image", "state", "IAR", "task_id", "episode_id"],
            "metadata_usage": "task_id/episode_id only for split and shuffled-EAR evaluation",
            "semantic_interventions_only": True,
        },
        "architecture": {
            "runtime": "JAX/Flax JIT; no PyTorch CUDA dependency",
            "plan_temporal_encoder": "two-layer pre-norm self-attention encoder",
            "latent_tokens": args.latent_tokens,
            "action_decoder": "action-noise self-attention then cross-attention to plan latents",
            "training_only_plan_reconstruction_head": True,
            "active_action_dim": args.active_action_dim,
            "inactive_output_dimensions_forced_zero": True,
            "parameter_count": parameter_count,
            "parameter_target_met": 1_000_000 <= parameter_count <= 3_000_000,
            "compute_dtype": args.amp,
            "parameter_dtype": "float32",
        },
        "dataset_records": int(len(arrays["dataset_index"])),
        "eligible_semantic_records": int(train_indices.size + validation_indices.size),
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "train_episode_groups": _episode_group_count(arrays, train_indices),
        "validation_episode_groups": _episode_group_count(arrays, validation_indices),
        "train_task_counts": _task_counts(arrays, train_indices),
        "validation_task_counts": _task_counts(arrays, validation_indices),
        "completed_steps": args.steps,
        "checkpoint_selection": {
            "primary_metric": "validation_sample/response_mse_active7",
            "tie_break_metric": "validation_sample/clean_action_mse_active7",
            "mode": "min",
            "best_step": best_step,
            "last_step": args.steps,
            "best_validation_sample_metrics": {
                name.removeprefix("validation_sample/"): value
                for name, value in best_record.items()
                if name.startswith("validation_sample/")
            },
            "last_validation_sample_metrics": {
                name.removeprefix("validation_sample/"): value
                for name, value in last_record.items()
                if name.startswith("validation_sample/")
            },
        },
        "best_step": best_step,
        "last_step": args.steps,
        "best_training_record": best_record,
        "last_training_record": last_record,
        "full_validation_metrics": full_metrics,
        "full_validation_checkpoint_step": best_step,
        "device": str(device),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(item) for item in jax.devices()],
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(
        "Full held-out: "
        f"clean_mse_active7={full_metrics['clean_action_mse_active7']:.6f} "
        f"response_mse_active7={full_metrics['response_mse_active7']:.6f} "
        f"response_cosine_active7={full_metrics['response_cosine_active7']:.4f} "
        f"shuffle_gap_active7={full_metrics['same_task_shuffled_ear_action_mse_gap_active7']} "
        f"gpu_latency_mean_ms={full_metrics['gpu_latency_mean_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
