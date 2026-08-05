"""GPU-native JAX Phase-A probe for a strict causal clean-plan compiler.

The probe intentionally has a narrow information bottleneck.  Its only model
inputs are the leading seven dimensions of a teacher EAR endpoint (15 x 7)
and the leading seven dimensions of the shared final-action noise (10 x 7).
Observation, image, state, IAR, task, and episode metadata are never model
inputs.  Task and episode IDs are used only for episode-held-out splitting and
evaluation grouping.

The compiler combines a learnable linear EAR-to-action transport with a small
residual MLP.  Clean and semantically intervened examples use identical action
noise, allowing direct supervision of the clean endpoint, intervened endpoint,
and causal response delta in one GPU-jitted training step.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.action_cot import endpoint_dataset


ArrayTree = Any


@dataclasses.dataclass(frozen=True)
class ProbeArgs:
    dataset: tuple[str, ...]
    output_dir: str
    steps: int = 1_000
    batch_size: int = 256
    eval_batch_size: int = 512
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
    model_width: int = 512
    residual_blocks: int = 2
    active_action_dim: int = 7
    latency_warmup: int = 25
    latency_runs: int = 200
    allow_cpu: bool = False
    overwrite: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
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
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument(
        "--active-action-dim",
        type=int,
        default=7,
        help="Leading endpoint dimensions admitted to the Phase-A bottleneck.",
    )
    parser.add_argument("--latency-warmup", type=int, default=25)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit an explicit CPU debug run. By default the probe fails if JAX has no GPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args() -> ProbeArgs:
    values = vars(_parser().parse_args())
    values["dataset"] = tuple(values["dataset"])
    return ProbeArgs(**values)


def _validate_args(args: ProbeArgs) -> None:
    positive = (
        "steps",
        "batch_size",
        "eval_batch_size",
        "warmup_steps",
        "log_interval",
        "model_width",
        "residual_blocks",
        "active_action_dim",
        "latency_runs",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.latency_warmup < 0:
        raise ValueError("--latency-warmup must be non-negative.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.learning_rate <= 0 or args.final_learning_rate < 0:
        raise ValueError("Learning rates are invalid.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate.")
    if args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Weight decay and gradient clipping values are invalid.")
    if min(
        args.clean_loss_weight,
        args.intervention_loss_weight,
        args.response_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")


def _select_device(*, allow_cpu: bool) -> jax.Device:
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if gpu_devices:
        return gpu_devices[0]
    if not allow_cpu:
        details = [f"{device.platform}:{device.device_kind}" for device in devices]
        raise RuntimeError(
            "This probe requires a JAX GPU backend and will not silently run on CPU. "
            f"Visible JAX devices: {details}. Pass --allow-cpu only for an explicit debug run."
        )
    return devices[0]


def _semantic_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    return np.asarray(arrays["intervention_valid"], dtype=np.bool_) & (
        np.asarray(arrays["intervention_ids"]) != null_id
    )


def _split_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the episode-held-out split in the endpoint distillation trainers."""

    eligible = np.any(_semantic_mask(arrays), axis=-1)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 2:
        raise ValueError("Endpoint dataset has fewer than two semantic-intervention records.")

    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = tasks * np.int64(1_000_000_000) + episodes
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
        validation_count = max(1, round(shuffled.size * validation_fraction))
        validation_count = min(validation_count, shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Train/validation split produced an empty partition.")
    return train_indices.astype(np.int32), validation_indices.astype(np.int32)


def _choose_interventions(
    semantic_mask: np.ndarray,
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> np.ndarray:
    selected = np.empty(row_indices.shape[0], dtype=np.int32)
    for offset, row_index in enumerate(row_indices):
        candidates = np.flatnonzero(semantic_mask[row_index])
        if not candidates.size:
            raise RuntimeError(f"Row {row_index} has no semantic intervention.")
        selected[offset] = int(candidates[0] if deterministic else rng.choice(candidates))
    return selected


def _dense_init(
    key: jax.Array,
    input_dim: int,
    output_dim: int,
    *,
    gain: float = 1.0,
) -> dict[str, jax.Array]:
    limit = gain * math.sqrt(6.0 / (input_dim + output_dim))
    return {
        "kernel": jax.random.uniform(
            key,
            (input_dim, output_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        "bias": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def _init_params(
    key: jax.Array,
    *,
    plan_dim: int,
    noise_dim: int,
    output_dim: int,
    width: int,
    residual_blocks: int,
) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 5 + 2 * residual_blocks))
    blocks: list[dict[str, Any]] = []
    for _ in range(residual_blocks):
        blocks.append(
            {
                "norm_scale": jnp.ones((width,), dtype=jnp.float32),
                "norm_bias": jnp.zeros((width,), dtype=jnp.float32),
                "input": _dense_init(next(keys), width, width),
                "output": _dense_init(next(keys), width, width, gain=0.1),
            }
        )
    return {
        "plan_input": _dense_init(next(keys), plan_dim, width),
        "noise_input": _dense_init(next(keys), noise_dim, width),
        "plan_transport": _dense_init(next(keys), plan_dim, output_dim, gain=0.1),
        "blocks": tuple(blocks),
        "final_norm": {
            "scale": jnp.ones((width,), dtype=jnp.float32),
            "bias": jnp.zeros((width,), dtype=jnp.float32),
        },
        "action_residual": _dense_init(next(keys), width, output_dim, gain=0.1),
    }


def _dense(values: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
    return values @ params["kernel"] + params["bias"]


def _layer_norm(
    values: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    *,
    epsilon: float = 1e-6,
) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def _model(
    params: Mapping[str, Any],
    plan_flat: jax.Array,
    action_noise_flat: jax.Array,
) -> jax.Array:
    plan_features = _dense(plan_flat, params["plan_input"])
    noise_features = _dense(action_noise_flat, params["noise_input"])
    hidden = jax.nn.gelu(plan_features + noise_features, approximate=True)
    for block in params["blocks"]:
        residual = _layer_norm(hidden, block["norm_scale"], block["norm_bias"])
        residual = jax.nn.gelu(_dense(residual, block["input"]), approximate=True)
        hidden = hidden + _dense(residual, block["output"])
    hidden = _layer_norm(
        hidden,
        params["final_norm"]["scale"],
        params["final_norm"]["bias"],
    )
    return _dense(plan_flat, params["plan_transport"]) + _dense(
        hidden, params["action_residual"]
    )


def _parameter_count(params: ArrayTree) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))


def _learning_rate_schedule(args: ProbeArgs) -> Callable[[jax.Array], jax.Array]:
    def schedule(count: jax.Array) -> jax.Array:
        step = count.astype(jnp.float32) + 1.0
        warmup_steps = float(args.warmup_steps)
        warmup = args.learning_rate * step / warmup_steps
        denominator = float(max(1, args.steps - args.warmup_steps))
        progress = jnp.clip((step - warmup_steps) / denominator, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decayed = args.final_learning_rate + (
            args.learning_rate - args.final_learning_rate
        ) * cosine
        return jnp.where(step <= warmup_steps, warmup, decayed)

    return schedule


def _loss(
    params: ArrayTree,
    dataset: Mapping[str, jax.Array],
    rows: jax.Array,
    intervention_slots: jax.Array,
    *,
    clean_weight: float,
    intervention_weight: float,
    response_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    clean_plan = dataset["clean_plan"][rows]
    action_noise = dataset["action_noise"][rows]
    clean_target = dataset["clean_action"][rows]
    intervention_plan = dataset["intervention_plan"][rows, intervention_slots]
    intervention_target = dataset["intervention_action"][rows, intervention_slots]

    # One fused call gives XLA a [clean; intervention] batch while preserving
    # exactly the same action-noise sample for each causal pair.
    paired_plans = jnp.concatenate((clean_plan, intervention_plan), axis=0)
    paired_noise = jnp.concatenate((action_noise, action_noise), axis=0)
    paired_prediction = _model(params, paired_plans, paired_noise)
    clean_prediction, intervention_prediction = jnp.split(paired_prediction, 2, axis=0)

    clean_mse = jnp.mean(jnp.square(clean_prediction - clean_target))
    intervention_mse = jnp.mean(
        jnp.square(intervention_prediction - intervention_target)
    )
    predicted_response = intervention_prediction - clean_prediction
    teacher_response = intervention_target - clean_target
    response_mse = jnp.mean(jnp.square(predicted_response - teacher_response))
    total = (
        clean_weight * clean_mse
        + intervention_weight * intervention_mse
        + response_weight * response_mse
    )
    predicted_flat = predicted_response.reshape((predicted_response.shape[0], -1))
    teacher_flat = teacher_response.reshape((teacher_response.shape[0], -1))
    response_cosine = jnp.mean(
        jnp.sum(predicted_flat * teacher_flat, axis=-1)
        / (
            jnp.linalg.norm(predicted_flat, axis=-1)
            * jnp.linalg.norm(teacher_flat, axis=-1)
            + 1e-8
        )
    )
    return total, {
        "total_loss": total,
        "clean_action_mse_active7": clean_mse,
        "intervention_action_mse_active7": intervention_mse,
        "response_mse_active7": response_mse,
        "response_cosine_active7": response_cosine,
    }


def _make_train_step(
    optimizer: optax.GradientTransformation,
    args: ProbeArgs,
) -> Callable[..., tuple[ArrayTree, optax.OptState, dict[str, jax.Array]]]:
    @jax.jit
    def train_step(
        params: ArrayTree,
        optimizer_state: optax.OptState,
        dataset: Mapping[str, jax.Array],
        rows: jax.Array,
        intervention_slots: jax.Array,
    ) -> tuple[ArrayTree, optax.OptState, dict[str, jax.Array]]:
        (_, metrics), gradients = jax.value_and_grad(_loss, has_aux=True)(
            params,
            dataset,
            rows,
            intervention_slots,
            clean_weight=args.clean_loss_weight,
            intervention_weight=args.intervention_loss_weight,
            response_weight=args.response_loss_weight,
        )
        updates, next_optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        next_params = optax.apply_updates(params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
        }

    return train_step


def _make_validation_step(args: ProbeArgs) -> Callable[..., dict[str, jax.Array]]:
    @jax.jit
    def validation_step(
        params: ArrayTree,
        dataset: Mapping[str, jax.Array],
        rows: jax.Array,
        intervention_slots: jax.Array,
    ) -> dict[str, jax.Array]:
        _, metrics = _loss(
            params,
            dataset,
            rows,
            intervention_slots,
            clean_weight=args.clean_loss_weight,
            intervention_weight=args.intervention_loss_weight,
            response_weight=args.response_loss_weight,
        )
        return metrics

    return validation_step


def _flatten_active(values: np.ndarray, active_action_dim: int) -> np.ndarray:
    active = np.asarray(values, dtype=np.float32)[..., :active_action_dim]
    return np.ascontiguousarray(active.reshape((active.shape[0], -1)))


def _build_device_dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    active_action_dim: int,
    device: jax.Device,
) -> dict[str, jax.Array]:
    clean_plan = _flatten_active(arrays["clean_coarse"], active_action_dim)
    clean_action = _flatten_active(arrays["clean_actions"], active_action_dim)
    action_noise = _flatten_active(arrays["action_noise"], active_action_dim)
    intervention_plan_values = np.asarray(
        arrays["intervention_coarse"], dtype=np.float32
    )[..., :active_action_dim]
    intervention_action_values = np.asarray(
        arrays["intervention_actions"], dtype=np.float32
    )[..., :active_action_dim]
    intervention_plan = np.ascontiguousarray(
        intervention_plan_values.reshape(
            (intervention_plan_values.shape[0], intervention_plan_values.shape[1], -1)
        )
    )
    intervention_action = np.ascontiguousarray(
        intervention_action_values.reshape(
            (
                intervention_action_values.shape[0],
                intervention_action_values.shape[1],
                -1,
            )
        )
    )
    host = {
        "clean_plan": clean_plan,
        "clean_action": clean_action,
        "action_noise": action_noise,
        "intervention_plan": intervention_plan,
        "intervention_action": intervention_action,
    }
    return {name: jax.device_put(jnp.asarray(value), device) for name, value in host.items()}


def _index_batches(size: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, size, batch_size):
        yield start, min(start + batch_size, size)


def _predict_batches(
    predict: Callable[[ArrayTree, jax.Array, jax.Array], jax.Array],
    params: ArrayTree,
    plans: np.ndarray,
    noise: np.ndarray,
    *,
    batch_size: int,
    device: jax.Device,
) -> np.ndarray:
    if plans.shape[0] != noise.shape[0]:
        raise ValueError("Plan and action-noise batch sizes differ.")
    outputs: list[np.ndarray] = []
    for start, stop in _index_batches(plans.shape[0], batch_size):
        count = stop - start
        plan_batch = np.zeros((batch_size, plans.shape[1]), dtype=np.float32)
        noise_batch = np.zeros((batch_size, noise.shape[1]), dtype=np.float32)
        plan_batch[:count] = plans[start:stop]
        noise_batch[:count] = noise[start:stop]
        prediction = predict(
            params,
            jax.device_put(plan_batch, device),
            jax.device_put(noise_batch, device),
        )
        outputs.append(np.asarray(prediction[:count]))
    return np.concatenate(outputs, axis=0)


def _all_semantic_pairs(
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    validation_offsets: list[int] = []
    intervention_slots: list[int] = []
    intervention_ids: list[int] = []
    semantic = _semantic_mask(arrays)
    ids = np.asarray(arrays["intervention_ids"])
    for validation_offset, row_index in enumerate(validation_indices):
        for slot in np.flatnonzero(semantic[row_index]):
            rows.append(int(row_index))
            validation_offsets.append(validation_offset)
            intervention_slots.append(int(slot))
            intervention_ids.append(int(ids[row_index, slot]))
    if not rows:
        raise RuntimeError("Held-out split contains no semantic intervention pairs.")
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(validation_offsets, dtype=np.int32),
        np.asarray(intervention_slots, dtype=np.int32),
        np.asarray(intervention_ids, dtype=np.int16),
    )


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(np.square(difference)))


def _mean_cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted_flat = prediction.astype(np.float64).reshape((prediction.shape[0], -1))
    target_flat = target.astype(np.float64).reshape((target.shape[0], -1))
    numerator = np.sum(predicted_flat * target_flat, axis=-1)
    denominator = np.linalg.norm(predicted_flat, axis=-1) * np.linalg.norm(
        target_flat, axis=-1
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-8)))


def _endpoint_metrics(
    intervention_prediction: np.ndarray,
    intervention_target: np.ndarray,
    predicted_response: np.ndarray,
    teacher_response: np.ndarray,
) -> dict[str, float]:
    intervention_mse = _mse(intervention_prediction, intervention_target)
    response_mse = _mse(predicted_response, teacher_response)
    return {
        "intervention_action_mse_active7": intervention_mse,
        "intervention_action_rmse_active7": math.sqrt(intervention_mse),
        "response_mse_active7": response_mse,
        "response_rmse_active7": math.sqrt(response_mse),
        "response_cosine_active7": _mean_cosine(predicted_response, teacher_response),
    }


def _same_task_shuffle_sources(
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    seed: int,
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
    return np.asarray(target_offsets, dtype=np.int32), np.asarray(source_offsets, dtype=np.int32)


def _safe_group_metrics(
    mask: np.ndarray,
    *,
    intervention_prediction: np.ndarray,
    intervention_target: np.ndarray,
    predicted_response: np.ndarray,
    teacher_response: np.ndarray,
) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(mask))
    if not count:
        return {
            "semantic_intervention_pairs": 0,
            "intervention_action_mse_active7": None,
            "intervention_action_rmse_active7": None,
            "response_mse_active7": None,
            "response_rmse_active7": None,
            "response_cosine_active7": None,
        }
    return {
        "semantic_intervention_pairs": count,
        **_endpoint_metrics(
            intervention_prediction[mask],
            intervention_target[mask],
            predicted_response[mask],
            teacher_response[mask],
        ),
    }


def _full_evaluation(
    params: ArrayTree,
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    active_action_dim: int,
    eval_batch_size: int,
    seed: int,
    device: jax.Device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predict = jax.jit(_model)
    clean_plan = _flatten_active(arrays["clean_coarse"][validation_indices], active_action_dim)
    clean_noise = _flatten_active(arrays["action_noise"][validation_indices], active_action_dim)
    clean_target = _flatten_active(arrays["clean_actions"][validation_indices], active_action_dim)
    clean_prediction = _predict_batches(
        predict,
        params,
        clean_plan,
        clean_noise,
        batch_size=eval_batch_size,
        device=device,
    )
    clean_mse = _mse(clean_prediction, clean_target)

    pair_rows, clean_offsets, pair_slots, pair_ids = _all_semantic_pairs(
        arrays, validation_indices
    )
    intervention_plan_values = np.asarray(
        arrays["intervention_coarse"][pair_rows, pair_slots], dtype=np.float32
    )[..., :active_action_dim]
    intervention_plan = np.ascontiguousarray(
        intervention_plan_values.reshape((intervention_plan_values.shape[0], -1))
    )
    pair_noise = _flatten_active(arrays["action_noise"][pair_rows], active_action_dim)
    intervention_target = _flatten_active(
        arrays["intervention_actions"][pair_rows, pair_slots], active_action_dim
    )
    intervention_prediction = _predict_batches(
        predict,
        params,
        intervention_plan,
        pair_noise,
        batch_size=eval_batch_size,
        device=device,
    )
    pair_clean_prediction = clean_prediction[clean_offsets]
    pair_clean_target = _flatten_active(arrays["clean_actions"][pair_rows], active_action_dim)
    predicted_response = intervention_prediction - pair_clean_prediction
    teacher_response = intervention_target - pair_clean_target
    global_pair_metrics = _endpoint_metrics(
        intervention_prediction,
        intervention_target,
        predicted_response,
        teacher_response,
    )

    target_offsets, source_offsets = _same_task_shuffle_sources(
        arrays, validation_indices, seed=seed + 2
    )
    shuffled_prediction: np.ndarray
    if target_offsets.size:
        shuffled_prediction = _predict_batches(
            predict,
            params,
            clean_plan[source_offsets],
            clean_noise[target_offsets],
            batch_size=eval_batch_size,
            device=device,
        )
        correct_shuffle_mse = _mse(
            clean_prediction[target_offsets], clean_target[target_offsets]
        )
        shuffled_mse = _mse(shuffled_prediction, clean_target[target_offsets])
        shuffle_metrics: dict[str, Any] = {
            "same_task_shuffle_records": int(target_offsets.size),
            "shuffle_correct_action_mse_active7": correct_shuffle_mse,
            "same_task_shuffled_ear_action_mse_active7": shuffled_mse,
            "same_task_shuffled_ear_action_mse_gap_active7": (
                shuffled_mse - correct_shuffle_mse
            ),
        }
    else:
        shuffled_prediction = np.empty((0, clean_prediction.shape[-1]), dtype=np.float32)
        shuffle_metrics = {
            "same_task_shuffle_records": 0,
            "shuffle_correct_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_gap_active7": None,
        }

    per_intervention: dict[str, Any] = {}
    for intervention_id, name in enumerate(endpoint_dataset.INTERVENTION_NAMES):
        if name == "null":
            continue
        per_intervention[name] = {
            "intervention_id": intervention_id,
            **_safe_group_metrics(
                pair_ids == intervention_id,
                intervention_prediction=intervention_prediction,
                intervention_target=intervention_target,
                predicted_response=predicted_response,
                teacher_response=teacher_response,
            ),
        }

    validation_tasks = np.asarray(arrays["task_id"])[validation_indices]
    pair_tasks = np.asarray(arrays["task_id"])[pair_rows]
    shuffled_target_tasks = validation_tasks[target_offsets]
    per_task: dict[str, Any] = {}
    for task_id in np.unique(validation_tasks):
        clean_mask = validation_tasks == task_id
        pair_mask = pair_tasks == task_id
        task_clean_mse = _mse(clean_prediction[clean_mask], clean_target[clean_mask])
        task_metrics: dict[str, Any] = {
            "validation_records": int(np.count_nonzero(clean_mask)),
            "clean_action_mse_active7": task_clean_mse,
            "clean_action_rmse_active7": math.sqrt(task_clean_mse),
            **_safe_group_metrics(
                pair_mask,
                intervention_prediction=intervention_prediction,
                intervention_target=intervention_target,
                predicted_response=predicted_response,
                teacher_response=teacher_response,
            ),
        }
        task_shuffle_mask = shuffled_target_tasks == task_id
        if np.any(task_shuffle_mask):
            task_correct_mse = _mse(
                clean_prediction[target_offsets[task_shuffle_mask]],
                clean_target[target_offsets[task_shuffle_mask]],
            )
            task_shuffled_mse = _mse(
                shuffled_prediction[task_shuffle_mask],
                clean_target[target_offsets[task_shuffle_mask]],
            )
            task_metrics.update(
                {
                    "same_task_shuffle_records": int(np.count_nonzero(task_shuffle_mask)),
                    "shuffle_correct_action_mse_active7": task_correct_mse,
                    "same_task_shuffled_ear_action_mse_active7": task_shuffled_mse,
                    "same_task_shuffled_ear_action_mse_gap_active7": (
                        task_shuffled_mse - task_correct_mse
                    ),
                }
            )
        else:
            task_metrics.update(
                {
                    "same_task_shuffle_records": 0,
                    "shuffle_correct_action_mse_active7": None,
                    "same_task_shuffled_ear_action_mse_active7": None,
                    "same_task_shuffled_ear_action_mse_gap_active7": None,
                }
            )
        per_task[str(int(task_id))] = task_metrics

    full_metrics = {
        "validation_records": int(validation_indices.size),
        "semantic_intervention_pairs": int(pair_rows.size),
        "clean_action_mse_active7": clean_mse,
        "clean_action_rmse_active7": math.sqrt(clean_mse),
        **global_pair_metrics,
        **shuffle_metrics,
        "per_intervention": per_intervention,
        "per_task": per_task,
    }
    evaluation_arrays = {
        "validation_indices": validation_indices,
        "validation_task_id": validation_tasks.astype(np.int16),
        "validation_episode_id": np.asarray(arrays["episode_id"])[validation_indices],
        "clean_prediction_active7_flat": clean_prediction,
        "clean_target_active7_flat": clean_target,
        "pair_row_index": pair_rows,
        "pair_validation_offset": clean_offsets,
        "pair_intervention_slot": pair_slots,
        "pair_intervention_id": pair_ids,
        "intervention_prediction_active7_flat": intervention_prediction,
        "intervention_target_active7_flat": intervention_target,
        "predicted_response_active7_flat": predicted_response,
        "teacher_response_active7_flat": teacher_response,
        "shuffle_target_validation_offset": target_offsets,
        "shuffle_source_validation_offset": source_offsets,
        "shuffle_prediction_active7_flat": shuffled_prediction,
    }
    return full_metrics, evaluation_arrays


def _latency_metrics(
    params: ArrayTree,
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    active_action_dim: int,
    warmup: int,
    runs: int,
    device: jax.Device,
) -> dict[str, Any]:
    plan = jax.device_put(
        _flatten_active(arrays["clean_coarse"][validation_indices[:1]], active_action_dim),
        device,
    )
    noise = jax.device_put(
        _flatten_active(arrays["action_noise"][validation_indices[:1]], active_action_dim),
        device,
    )
    infer = jax.jit(_model)
    compiled = infer.lower(params, plan, noise).compile()
    for _ in range(warmup):
        compiled(params, plan, noise).block_until_ready()
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        compiled(params, plan, noise).block_until_ready()
        durations.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(durations, dtype=np.float64)
    is_gpu = device.platform == "gpu"
    return {
        "latency_backend": "jax_jit",
        "latency_device_platform": device.platform,
        "latency_device_kind": device.device_kind,
        "latency_batch_size": 1,
        "latency_runs": runs,
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "gpu_latency_mean_ms": float(values.mean()) if is_gpu else None,
        "gpu_latency_p95_ms": float(np.percentile(values, 95)) if is_gpu else None,
    }


def _task_counts(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, int]:
    tasks, counts = np.unique(np.asarray(arrays["task_id"])[indices], return_counts=True)
    return {str(int(task)): int(count) for task, count in zip(tasks, counts, strict=True)}


def _episode_group_count(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> int:
    keys = np.stack(
        (
            np.asarray(arrays["task_id"])[indices],
            np.asarray(arrays["episode_id"])[indices],
        ),
        axis=-1,
    )
    return int(np.unique(keys, axis=0).shape[0])


def _tree_to_npz(params: ArrayTree) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, f"{prefix}/{name}" if prefix else str(name))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}/{index}")
        else:
            output[prefix] = np.asarray(value)

    visit(params, "params")
    return output


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: pathlib.Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def main(args: ProbeArgs) -> None:
    _validate_args(args)
    device = _select_device(allow_cpu=args.allow_cpu)
    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "model_params.npz"
    evaluation_path = output_dir / "heldout_predictions.npz"
    output_paths = (metrics_path, summary_path, checkpoint_path, evaluation_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Probe outputs already exist: {[str(path) for path in existing]}; "
            "choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(
        arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    action_dim = int(arrays["clean_actions"].shape[-1])
    plan_horizon = int(arrays["clean_coarse"].shape[1])
    action_horizon = int(arrays["clean_actions"].shape[1])
    if not 0 < args.active_action_dim <= action_dim:
        raise ValueError(f"active_action_dim must be in [1, {action_dim}].")
    if arrays["action_noise"].shape[1:] != arrays["clean_actions"].shape[1:]:
        raise ValueError("action_noise and clean_actions shapes do not match.")

    plan_dim = plan_horizon * args.active_action_dim
    noise_dim = action_horizon * args.active_action_dim
    output_dim = action_horizon * args.active_action_dim
    with jax.default_device(device):
        params = _init_params(
            jax.random.PRNGKey(args.seed),
            plan_dim=plan_dim,
            noise_dim=noise_dim,
            output_dim=output_dim,
            width=args.model_width,
            residual_blocks=args.residual_blocks,
        )
        params = jax.device_put(params, device)
        parameter_count = _parameter_count(params)
        learning_rate_schedule = _learning_rate_schedule(args)
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.gradient_clip_norm),
            optax.adamw(learning_rate_schedule, weight_decay=args.weight_decay),
        )
        optimizer_state = jax.device_put(optimizer.init(params), device)
        device_dataset = _build_device_dataset(
            arrays,
            active_action_dim=args.active_action_dim,
            device=device,
        )
        train_step = _make_train_step(optimizer, args)
        validation_step = _make_validation_step(args)

    if not 500_000 <= parameter_count <= 1_500_000:
        print(
            f"WARNING: model has {parameter_count:,} parameters; target is 0.5-1.5M.",
            flush=True,
        )
    print(
        "Initialized JAX causal clean-plan compiler: "
        f"train={train_indices.size} validation={validation_indices.size} "
        f"params={parameter_count:,} device={device.platform}:{device.device_kind}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    semantic_mask = _semantic_mask(arrays)
    started = time.monotonic()
    metrics_mode = "w" if args.overwrite else "a"
    last_record: dict[str, Any] = {}
    best_record: dict[str, Any] = {}
    best_params: ArrayTree | None = None
    best_score: tuple[float, float] | None = None
    best_step = 0
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            rows = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            ).astype(np.int32)
            intervention_slots = _choose_interventions(
                semantic_mask,
                rows,
                rng,
                deterministic=False,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                device_dataset,
                jax.device_put(rows, device),
                jax.device_put(intervention_slots, device),
            )

            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            if not should_log:
                continue
            validation_rows = validation_rng.choice(
                validation_indices,
                size=min(args.batch_size, validation_indices.size),
                replace=False,
            ).astype(np.int32)
            validation_slots = _choose_interventions(
                semantic_mask,
                validation_rows,
                validation_rng,
                deterministic=True,
            )
            validation_metrics = validation_step(
                params,
                device_dataset,
                jax.device_put(validation_rows, device),
                jax.device_put(validation_slots, device),
            )
            train_values = jax.device_get(train_metrics)
            validation_values = jax.device_get(validation_metrics)
            learning_rate = float(learning_rate_schedule(jnp.asarray(step - 1)))
            candidate_score = (
                float(validation_values["response_mse_active7"]),
                float(validation_values["clean_action_mse_active7"]),
            )
            selected_as_best = best_score is None or candidate_score < best_score
            last_record = {
                "phase": "train",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": learning_rate,
                "selected_as_best_checkpoint": selected_as_best,
                **{f"train/{name}": float(value) for name, value in train_values.items()},
                **{
                    f"validation_sample/{name}": float(value)
                    for name, value in validation_values.items()
                },
            }
            if selected_as_best:
                # JAX arrays are immutable and this training step does not donate
                # parameter buffers, so retaining this pytree preserves the exact
                # device-resident checkpoint without a CPU round trip.
                best_params = params
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
        full_metrics, evaluation_arrays = _full_evaluation(
            best_params,
            arrays,
            validation_indices,
            active_action_dim=args.active_action_dim,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=device,
        )
        latency_metrics = _latency_metrics(
            best_params,
            arrays,
            validation_indices,
            active_action_dim=args.active_action_dim,
            warmup=args.latency_warmup,
            runs=args.latency_runs,
            device=device,
        )
        full_metrics.update(latency_metrics)
        final_record = {
            "phase": "full_validation",
            "step": args.steps,
            "evaluated_checkpoint_step": best_step,
            "elapsed_seconds": time.monotonic() - started,
            **full_metrics,
        }
        metrics_file.write(json.dumps(final_record, sort_keys=True) + "\n")
        metrics_file.flush()

    checkpoint_values = {
        **_tree_to_npz(jax.device_get(best_params)),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "completed_steps": np.asarray(args.steps, dtype=np.int32),
        "best_step": np.asarray(best_step, dtype=np.int32),
        "last_step": np.asarray(args.steps, dtype=np.int32),
        "active_action_dim": np.asarray(args.active_action_dim, dtype=np.int32),
        "plan_horizon": np.asarray(plan_horizon, dtype=np.int32),
        "action_horizon": np.asarray(action_horizon, dtype=np.int32),
    }
    _atomic_npz(checkpoint_path, checkpoint_values)
    _atomic_npz(evaluation_path, evaluation_arrays)

    summary = {
        "probe": "phase_a_causal_clean_plan_compiler_jax_mlp_oracle",
        "dataset": list(args.dataset),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "heldout_predictions_path": str(evaluation_path.resolve()),
        "args": dataclasses.asdict(args),
        "input_contract": {
            "model_inputs": [
                f"teacher_ear_endpoint_{plan_horizon}x{args.active_action_dim}",
                f"shared_action_noise_{action_horizon}x{args.active_action_dim}",
            ],
            "clean_and_intervention_share_action_noise": True,
            "forbidden_inputs": [
                "observation",
                "image",
                "state",
                "IAR",
                "task_id",
                "episode_id",
            ],
            "metadata_usage": "task_id/episode_id only for split and evaluation grouping",
            "semantic_interventions_only": True,
        },
        "architecture": {
            "backend": "JAX/XLA JIT",
            "plan_representation": f"flattened {plan_horizon}x{args.active_action_dim}",
            "action_noise_representation": (
                f"flattened {action_horizon}x{args.active_action_dim}"
            ),
            "action_output_representation": (
                f"flattened {action_horizon}x{args.active_action_dim}"
            ),
            "compiler": "learnable linear EAR transport plus residual MLP",
            "model_width": args.model_width,
            "residual_blocks": args.residual_blocks,
            "parameter_count": parameter_count,
            "parameter_target_met": 500_000 <= parameter_count <= 1_500_000,
        },
        "training_objective": {
            "clean_action_mse_weight": args.clean_loss_weight,
            "intervention_action_mse_weight": args.intervention_loss_weight,
            "causal_response_delta_mse_weight": args.response_loss_weight,
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
        "device": {
            "platform": device.platform,
            "kind": device.device_kind,
            "id": int(device.id),
        },
        "jax_version": jax.__version__,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(summary_path, summary)
    print(
        "Full held-out: "
        f"clean_mse_active7={full_metrics['clean_action_mse_active7']:.6f} "
        f"intervention_mse_active7={full_metrics['intervention_action_mse_active7']:.6f} "
        f"response_mse_active7={full_metrics['response_mse_active7']:.6f} "
        f"response_cosine_active7={full_metrics['response_cosine_active7']:.4f} "
        f"shuffle_gap_active7={full_metrics['same_task_shuffled_ear_action_mse_gap_active7']} "
        f"gpu_latency_mean_ms={full_metrics['gpu_latency_mean_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
