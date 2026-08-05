"""GPU-native Phase-A ridge plus residual causal clean-plan compiler.

Only active-action EAR values (15 x 7) and shared action noise (10 x 7) are
model inputs.  A frozen ridge base is fitted exclusively from the training
episode split using clean samples and every valid semantic intervention pair.
A small JAX residual MLP is then optimized with clean, intervention, and causal
response losses.  Validation episodes are used only for checkpoint selection
and reporting; they never enter ridge statistics, ridge fitting, or gradients.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from typing import Any, Callable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.action_cot import endpoint_dataset

try:
    import train_causal_clean_plan_compiler_jax_mlp as base
except ModuleNotFoundError:
    from scripts import train_causal_clean_plan_compiler_jax_mlp as base


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
    ridge_lambda: float = 1e-3
    width: int = 256
    blocks: int = 2
    seed: int = 7
    log_interval: int = 25
    clean_loss_weight: float = 1.0
    intervention_loss_weight: float = 1.0
    response_loss_weight: float = 1.0
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
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--intervention-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--latency-warmup", type=int, default=25)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit an explicit CPU debug run; GPU is required by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args() -> ProbeArgs:
    values = vars(_parser().parse_args())
    values["dataset"] = tuple(values["dataset"])
    return ProbeArgs(**values)


def _validate_args(args: ProbeArgs) -> None:
    for name in (
        "steps",
        "batch_size",
        "eval_batch_size",
        "warmup_steps",
        "width",
        "blocks",
        "log_interval",
        "active_action_dim",
        "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.latency_warmup < 0:
        raise ValueError("--latency-warmup must be non-negative.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.ridge_lambda < 0:
        raise ValueError("--ridge-lambda must be non-negative.")
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


def _ridge_training_examples(
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    active_action_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    clean_plan = base._flatten_active(arrays["clean_coarse"][train_indices], active_action_dim)
    clean_noise = base._flatten_active(arrays["action_noise"][train_indices], active_action_dim)
    clean_action = base._flatten_active(arrays["clean_actions"][train_indices], active_action_dim)

    semantic = base._semantic_mask(arrays)[train_indices]
    train_offsets, intervention_slots = np.nonzero(semantic)
    pair_rows = train_indices[train_offsets]
    intervention_plan = base._flatten_active(
        arrays["intervention_coarse"][pair_rows, intervention_slots], active_action_dim
    )
    intervention_noise = base._flatten_active(
        arrays["action_noise"][pair_rows], active_action_dim
    )
    intervention_action = base._flatten_active(
        arrays["intervention_actions"][pair_rows, intervention_slots], active_action_dim
    )
    features = np.ascontiguousarray(
        np.concatenate(
            (
                np.concatenate((clean_plan, clean_noise), axis=-1),
                np.concatenate((intervention_plan, intervention_noise), axis=-1),
            ),
            axis=0,
        ),
        dtype=np.float32,
    )
    targets = np.ascontiguousarray(
        np.concatenate((clean_action, intervention_action), axis=0), dtype=np.float32
    )
    return features, targets, {
        "clean_examples": int(train_indices.size),
        "semantic_intervention_examples": int(pair_rows.size),
        "total_examples": int(features.shape[0]),
    }


def _fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float,
    device: jax.Device,
) -> dict[str, jax.Array]:
    @jax.jit
    def solve(x: jax.Array, y: jax.Array) -> dict[str, jax.Array]:
        feature_mean = jnp.mean(x, axis=0)
        feature_std = jnp.sqrt(jnp.mean(jnp.square(x - feature_mean), axis=0) + 1e-6)
        normalized = (x - feature_mean) / feature_std
        design = jnp.concatenate(
            (normalized, jnp.ones((normalized.shape[0], 1), dtype=normalized.dtype)),
            axis=-1,
        )
        sample_count = jnp.asarray(design.shape[0], dtype=design.dtype)
        regularizer = jnp.eye(design.shape[-1], dtype=design.dtype)
        regularizer = regularizer.at[-1, -1].set(0.0)
        gram = design.T @ design / sample_count + ridge_lambda * regularizer
        right_hand_side = design.T @ y / sample_count
        weights = jnp.linalg.solve(gram, right_hand_side)
        train_prediction = design @ weights
        train_mse = jnp.mean(jnp.square(train_prediction - y))
        return {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "weights": weights,
            "train_mse": train_mse,
        }

    result = solve(
        jax.device_put(features, device),
        jax.device_put(targets, device),
    )
    result["weights"].block_until_ready()
    return result


def _normalized_features(
    ridge: Mapping[str, jax.Array],
    plan_flat: jax.Array,
    action_noise_flat: jax.Array,
) -> jax.Array:
    features = jnp.concatenate((plan_flat, action_noise_flat), axis=-1)
    return (features - ridge["feature_mean"]) / ridge["feature_std"]


def _ridge_predict(
    unused_params: None,
    ridge: Mapping[str, jax.Array],
    plan_flat: jax.Array,
    action_noise_flat: jax.Array,
) -> jax.Array:
    del unused_params
    normalized = _normalized_features(ridge, plan_flat, action_noise_flat)
    design = jnp.concatenate(
        (normalized, jnp.ones((normalized.shape[0], 1), dtype=normalized.dtype)), axis=-1
    )
    return design @ ridge["weights"]


def _init_residual_params(
    key: jax.Array,
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    blocks: int,
) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 3 + 2 * blocks))
    residual_blocks: list[dict[str, Any]] = []
    for _ in range(blocks):
        residual_blocks.append(
            {
                "norm_scale": jnp.ones((width,), dtype=jnp.float32),
                "norm_bias": jnp.zeros((width,), dtype=jnp.float32),
                "input": base._dense_init(next(keys), width, width),
                "output": base._dense_init(next(keys), width, width, gain=0.05),
            }
        )
    return {
        "input": base._dense_init(next(keys), input_dim, width),
        "blocks": tuple(residual_blocks),
        "final_norm": {
            "scale": jnp.ones((width,), dtype=jnp.float32),
            "bias": jnp.zeros((width,), dtype=jnp.float32),
        },
        "output": base._dense_init(next(keys), width, output_dim, gain=0.01),
    }


def _residual_predict(
    params: Mapping[str, Any], normalized_features: jax.Array
) -> jax.Array:
    hidden = jax.nn.gelu(base._dense(normalized_features, params["input"]), approximate=True)
    for block in params["blocks"]:
        residual = base._layer_norm(hidden, block["norm_scale"], block["norm_bias"])
        residual = jax.nn.gelu(base._dense(residual, block["input"]), approximate=True)
        hidden = hidden + base._dense(residual, block["output"])
    hidden = base._layer_norm(
        hidden, params["final_norm"]["scale"], params["final_norm"]["bias"]
    )
    return base._dense(hidden, params["output"])


def _combined_predict(
    params: Mapping[str, Any],
    ridge: Mapping[str, jax.Array],
    plan_flat: jax.Array,
    action_noise_flat: jax.Array,
) -> jax.Array:
    normalized = _normalized_features(ridge, plan_flat, action_noise_flat)
    design = jnp.concatenate(
        (normalized, jnp.ones((normalized.shape[0], 1), dtype=normalized.dtype)), axis=-1
    )
    ridge_prediction = design @ ridge["weights"]
    return ridge_prediction + _residual_predict(params, normalized)


def _loss(
    params: ArrayTree,
    ridge: Mapping[str, jax.Array],
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
    paired_prediction = _combined_predict(
        params,
        ridge,
        jnp.concatenate((clean_plan, intervention_plan), axis=0),
        jnp.concatenate((action_noise, action_noise), axis=0),
    )
    clean_prediction, intervention_prediction = jnp.split(paired_prediction, 2, axis=0)
    clean_mse = jnp.mean(jnp.square(clean_prediction - clean_target))
    intervention_mse = jnp.mean(
        jnp.square(intervention_prediction - intervention_target)
    )
    predicted_response = intervention_prediction - clean_prediction
    teacher_response = intervention_target - clean_target
    response_mse = jnp.mean(jnp.square(predicted_response - teacher_response))
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
    total = (
        clean_weight * clean_mse
        + intervention_weight * intervention_mse
        + response_weight * response_mse
    )
    return total, {
        "total_loss": total,
        "clean_action_mse_active7": clean_mse,
        "intervention_action_mse_active7": intervention_mse,
        "response_mse_active7": response_mse,
        "response_cosine_active7": response_cosine,
    }


def _make_train_step(
    optimizer: optax.GradientTransformation, args: ProbeArgs
) -> Callable[..., tuple[ArrayTree, optax.OptState, dict[str, jax.Array]]]:
    @jax.jit
    def train_step(
        params: ArrayTree,
        optimizer_state: optax.OptState,
        ridge: Mapping[str, jax.Array],
        dataset: Mapping[str, jax.Array],
        rows: jax.Array,
        intervention_slots: jax.Array,
    ) -> tuple[ArrayTree, optax.OptState, dict[str, jax.Array]]:
        (_, metrics), gradients = jax.value_and_grad(_loss, has_aux=True)(
            params,
            ridge,
            dataset,
            rows,
            intervention_slots,
            clean_weight=args.clean_loss_weight,
            intervention_weight=args.intervention_loss_weight,
            response_weight=args.response_loss_weight,
        )
        updates, next_state = optimizer.update(gradients, optimizer_state, params)
        return optax.apply_updates(params, updates), next_state, {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
        }

    return train_step


def _make_validation_step(args: ProbeArgs) -> Callable[..., dict[str, jax.Array]]:
    @jax.jit
    def validation_step(
        params: ArrayTree,
        ridge: Mapping[str, jax.Array],
        dataset: Mapping[str, jax.Array],
        rows: jax.Array,
        intervention_slots: jax.Array,
    ) -> dict[str, jax.Array]:
        _, metrics = _loss(
            params,
            ridge,
            dataset,
            rows,
            intervention_slots,
            clean_weight=args.clean_loss_weight,
            intervention_weight=args.intervention_loss_weight,
            response_weight=args.response_loss_weight,
        )
        return metrics

    return validation_step


def _learning_rate_schedule(args: ProbeArgs) -> Callable[[jax.Array], jax.Array]:
    bridge = base.ProbeArgs(
        dataset=args.dataset,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        warmup_steps=args.warmup_steps,
    )
    return base._learning_rate_schedule(bridge)


def _predict_batches(
    predict: Callable[[ArrayTree, Mapping[str, jax.Array], jax.Array, jax.Array], jax.Array],
    params: ArrayTree,
    ridge: Mapping[str, jax.Array],
    plans: np.ndarray,
    noise: np.ndarray,
    *,
    batch_size: int,
    device: jax.Device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start, stop in _index_batches(plans.shape[0], batch_size):
        count = stop - start
        plan_batch = np.zeros((batch_size, plans.shape[1]), dtype=np.float32)
        noise_batch = np.zeros((batch_size, noise.shape[1]), dtype=np.float32)
        plan_batch[:count] = plans[start:stop]
        noise_batch[:count] = noise[start:stop]
        prediction = predict(
            params,
            ridge,
            jax.device_put(plan_batch, device),
            jax.device_put(noise_batch, device),
        )
        outputs.append(np.asarray(prediction[:count]))
    return np.concatenate(outputs, axis=0)


def _index_batches(size: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, size, batch_size):
        yield start, min(start + batch_size, size)


def _evaluate_predictor(
    predict_function: Callable[..., jax.Array],
    params: ArrayTree,
    ridge: Mapping[str, jax.Array],
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    active_action_dim: int,
    eval_batch_size: int,
    seed: int,
    device: jax.Device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predict = jax.jit(predict_function)
    clean_plan = base._flatten_active(
        arrays["clean_coarse"][validation_indices], active_action_dim
    )
    clean_noise = base._flatten_active(
        arrays["action_noise"][validation_indices], active_action_dim
    )
    clean_target = base._flatten_active(
        arrays["clean_actions"][validation_indices], active_action_dim
    )
    clean_prediction = _predict_batches(
        predict,
        params,
        ridge,
        clean_plan,
        clean_noise,
        batch_size=eval_batch_size,
        device=device,
    )
    clean_mse = base._mse(clean_prediction, clean_target)

    pair_rows, clean_offsets, pair_slots, pair_ids = base._all_semantic_pairs(
        arrays, validation_indices
    )
    intervention_plan = base._flatten_active(
        arrays["intervention_coarse"][pair_rows, pair_slots], active_action_dim
    )
    pair_noise = base._flatten_active(arrays["action_noise"][pair_rows], active_action_dim)
    intervention_target = base._flatten_active(
        arrays["intervention_actions"][pair_rows, pair_slots], active_action_dim
    )
    intervention_prediction = _predict_batches(
        predict,
        params,
        ridge,
        intervention_plan,
        pair_noise,
        batch_size=eval_batch_size,
        device=device,
    )
    pair_clean_prediction = clean_prediction[clean_offsets]
    pair_clean_target = base._flatten_active(
        arrays["clean_actions"][pair_rows], active_action_dim
    )
    predicted_response = intervention_prediction - pair_clean_prediction
    teacher_response = intervention_target - pair_clean_target
    pair_metrics = base._endpoint_metrics(
        intervention_prediction,
        intervention_target,
        predicted_response,
        teacher_response,
    )

    target_offsets, source_offsets = base._same_task_shuffle_sources(
        arrays, validation_indices, seed=seed + 2
    )
    if target_offsets.size:
        shuffled_prediction = _predict_batches(
            predict,
            params,
            ridge,
            clean_plan[source_offsets],
            clean_noise[target_offsets],
            batch_size=eval_batch_size,
            device=device,
        )
        correct_shuffle_mse = base._mse(
            clean_prediction[target_offsets], clean_target[target_offsets]
        )
        shuffled_mse = base._mse(shuffled_prediction, clean_target[target_offsets])
        shuffle_metrics: dict[str, Any] = {
            "same_task_shuffle_records": int(target_offsets.size),
            "shuffle_correct_action_mse_active7": correct_shuffle_mse,
            "same_task_shuffled_ear_action_mse_active7": shuffled_mse,
            "same_task_shuffled_ear_action_mse_gap_active7": shuffled_mse - correct_shuffle_mse,
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
            **base._safe_group_metrics(
                pair_ids == intervention_id,
                intervention_prediction=intervention_prediction,
                intervention_target=intervention_target,
                predicted_response=predicted_response,
                teacher_response=teacher_response,
            ),
        }

    validation_tasks = np.asarray(arrays["task_id"])[validation_indices]
    pair_tasks = np.asarray(arrays["task_id"])[pair_rows]
    shuffled_tasks = validation_tasks[target_offsets]
    per_task: dict[str, Any] = {}
    for task_id in np.unique(validation_tasks):
        clean_mask = validation_tasks == task_id
        pair_mask = pair_tasks == task_id
        task_clean_mse = base._mse(clean_prediction[clean_mask], clean_target[clean_mask])
        task_result: dict[str, Any] = {
            "validation_records": int(np.count_nonzero(clean_mask)),
            "clean_action_mse_active7": task_clean_mse,
            "clean_action_rmse_active7": math.sqrt(task_clean_mse),
            **base._safe_group_metrics(
                pair_mask,
                intervention_prediction=intervention_prediction,
                intervention_target=intervention_target,
                predicted_response=predicted_response,
                teacher_response=teacher_response,
            ),
        }
        shuffle_mask = shuffled_tasks == task_id
        if np.any(shuffle_mask):
            correct_mse = base._mse(
                clean_prediction[target_offsets[shuffle_mask]],
                clean_target[target_offsets[shuffle_mask]],
            )
            wrong_mse = base._mse(
                shuffled_prediction[shuffle_mask], clean_target[target_offsets[shuffle_mask]]
            )
            task_result.update(
                {
                    "same_task_shuffle_records": int(np.count_nonzero(shuffle_mask)),
                    "shuffle_correct_action_mse_active7": correct_mse,
                    "same_task_shuffled_ear_action_mse_active7": wrong_mse,
                    "same_task_shuffled_ear_action_mse_gap_active7": wrong_mse - correct_mse,
                }
            )
        else:
            task_result.update(
                {
                    "same_task_shuffle_records": 0,
                    "shuffle_correct_action_mse_active7": None,
                    "same_task_shuffled_ear_action_mse_active7": None,
                    "same_task_shuffled_ear_action_mse_gap_active7": None,
                }
            )
        per_task[str(int(task_id))] = task_result

    metrics = {
        "validation_records": int(validation_indices.size),
        "semantic_intervention_pairs": int(pair_rows.size),
        "clean_action_mse_active7": clean_mse,
        "clean_action_rmse_active7": math.sqrt(clean_mse),
        **pair_metrics,
        **shuffle_metrics,
        "per_intervention": per_intervention,
        "per_task": per_task,
    }
    predictions = {
        "validation_indices": validation_indices,
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
    return metrics, predictions


def _latency_metrics(
    predict_function: Callable[..., jax.Array],
    params: ArrayTree,
    ridge: Mapping[str, jax.Array],
    arrays: Mapping[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    active_action_dim: int,
    warmup: int,
    runs: int,
    device: jax.Device,
) -> dict[str, Any]:
    plan = jax.device_put(
        base._flatten_active(
            arrays["clean_coarse"][validation_indices[:1]], active_action_dim
        ),
        device,
    )
    noise = jax.device_put(
        base._flatten_active(arrays["action_noise"][validation_indices[:1]], active_action_dim),
        device,
    )
    compiled = jax.jit(predict_function).lower(params, ridge, plan, noise).compile()
    for _ in range(warmup):
        compiled(params, ridge, plan, noise).block_until_ready()
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        compiled(params, ridge, plan, noise).block_until_ready()
        durations.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(durations, dtype=np.float64)
    return {
        "latency_backend": "jax_jit",
        "latency_device_platform": device.platform,
        "latency_device_kind": device.device_kind,
        "latency_batch_size": 1,
        "latency_runs": runs,
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "gpu_latency_mean_ms": float(values.mean()) if device.platform == "gpu" else None,
        "gpu_latency_p95_ms": (
            float(np.percentile(values, 95)) if device.platform == "gpu" else None
        ),
    }


def _tree_to_npz(value: Any, prefix: str) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for name, child in node.items():
                visit(child, f"{path}/{name}")
        elif isinstance(node, (tuple, list)):
            for index, child in enumerate(node):
                visit(child, f"{path}/{index}")
        else:
            output[path] = np.asarray(node)

    visit(value, prefix)
    return output


def main(args: ProbeArgs) -> None:
    _validate_args(args)
    device = base._select_device(allow_cpu=args.allow_cpu)
    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "model_params.npz"
    prediction_path = output_dir / "heldout_predictions.npz"
    existing = [
        path
        for path in (metrics_path, summary_path, checkpoint_path, prediction_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Probe outputs already exist: {[str(path) for path in existing]}; "
            "choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = base._split_indices(
        arrays, validation_fraction=args.validation_fraction, seed=args.seed
    )
    action_dim = int(arrays["clean_actions"].shape[-1])
    plan_horizon = int(arrays["clean_coarse"].shape[1])
    action_horizon = int(arrays["clean_actions"].shape[1])
    if not 0 < args.active_action_dim <= action_dim:
        raise ValueError(f"active_action_dim must be in [1, {action_dim}].")
    feature_dim = (plan_horizon + action_horizon) * args.active_action_dim
    output_dim = action_horizon * args.active_action_dim

    ridge_features, ridge_targets, ridge_counts = _ridge_training_examples(
        arrays, train_indices, active_action_dim=args.active_action_dim
    )
    ridge_started = time.monotonic()
    ridge = _fit_ridge(
        ridge_features,
        ridge_targets,
        ridge_lambda=args.ridge_lambda,
        device=device,
    )
    ridge_fit_seconds = time.monotonic() - ridge_started
    ridge_train_mse = float(jax.device_get(ridge["train_mse"]))

    with jax.default_device(device):
        params = _init_residual_params(
            jax.random.PRNGKey(args.seed),
            input_dim=feature_dim,
            output_dim=output_dim,
            width=args.width,
            blocks=args.blocks,
        )
        params = jax.device_put(params, device)
        schedule = _learning_rate_schedule(args)
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.gradient_clip_norm),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        )
        optimizer_state = jax.device_put(optimizer.init(params), device)
        dataset = base._build_device_dataset(
            arrays, active_action_dim=args.active_action_dim, device=device
        )
        train_step = _make_train_step(optimizer, args)
        validation_step = _make_validation_step(args)

    residual_parameter_count = base._parameter_count(params)
    ridge_parameter_count = int(np.prod(ridge["weights"].shape))
    print(
        "Initialized JAX ridge+residual compiler: "
        f"train={train_indices.size} validation={validation_indices.size} "
        f"ridge_examples={ridge_counts['total_examples']} "
        f"ridge_train_mse={ridge_train_mse:.6f} residual_params={residual_parameter_count:,} "
        f"device={device.platform}:{device.device_kind}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    semantic_mask = base._semantic_mask(arrays)
    started = time.monotonic()
    best_params: ArrayTree | None = None
    best_score: tuple[float, float] | None = None
    best_step = 0
    best_record: dict[str, Any] = {}
    last_record: dict[str, Any] = {}
    with metrics_path.open("w" if args.overwrite else "a", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            rows = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            ).astype(np.int32)
            slots = base._choose_interventions(
                semantic_mask, rows, rng, deterministic=False
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                ridge,
                dataset,
                jax.device_put(rows, device),
                jax.device_put(slots, device),
            )
            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            if not should_log:
                continue
            validation_rows = validation_rng.choice(
                validation_indices,
                size=min(args.batch_size, validation_indices.size),
                replace=False,
            ).astype(np.int32)
            validation_slots = base._choose_interventions(
                semantic_mask, validation_rows, validation_rng, deterministic=True
            )
            validation_metrics = validation_step(
                params,
                ridge,
                dataset,
                jax.device_put(validation_rows, device),
                jax.device_put(validation_slots, device),
            )
            train_values = jax.device_get(train_metrics)
            validation_values = jax.device_get(validation_metrics)
            candidate_score = (
                float(validation_values["response_mse_active7"]),
                float(validation_values["clean_action_mse_active7"]),
            )
            selected_as_best = best_score is None or candidate_score < best_score
            last_record = {
                "phase": "train",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": float(schedule(jnp.asarray(step - 1))),
                "selected_as_best_checkpoint": selected_as_best,
                **{f"train/{name}": float(value) for name, value in train_values.items()},
                **{
                    f"validation_sample/{name}": float(value)
                    for name, value in validation_values.items()
                },
            }
            if selected_as_best:
                best_params = params
                best_score = candidate_score
                best_step = step
                best_record = dict(last_record)
            metrics_file.write(json.dumps(last_record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"step={step} train_total={last_record['train/total_loss']:.6f} "
                f"val_clean7={candidate_score[1]:.6f} "
                f"val_response7={candidate_score[0]:.6f} best={selected_as_best}",
                flush=True,
            )

        if best_params is None:
            raise RuntimeError("Training completed without selecting a validation checkpoint.")
        ridge_metrics, ridge_predictions = _evaluate_predictor(
            _ridge_predict,
            None,
            ridge,
            arrays,
            validation_indices,
            active_action_dim=args.active_action_dim,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=device,
        )
        combined_metrics, combined_predictions = _evaluate_predictor(
            _combined_predict,
            best_params,
            ridge,
            arrays,
            validation_indices,
            active_action_dim=args.active_action_dim,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=device,
        )
        ridge_metrics.update(
            _latency_metrics(
                _ridge_predict,
                None,
                ridge,
                arrays,
                validation_indices,
                active_action_dim=args.active_action_dim,
                warmup=args.latency_warmup,
                runs=args.latency_runs,
                device=device,
            )
        )
        combined_metrics.update(
            _latency_metrics(
                _combined_predict,
                best_params,
                ridge,
                arrays,
                validation_indices,
                active_action_dim=args.active_action_dim,
                warmup=args.latency_warmup,
                runs=args.latency_runs,
                device=device,
            )
        )
        full_metrics = {
            "ridge_only": ridge_metrics,
            "ridge_plus_residual": combined_metrics,
        }
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
        **_tree_to_npz(jax.device_get(best_params), "residual_params"),
        **_tree_to_npz(jax.device_get(ridge), "ridge"),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "best_step": np.asarray(best_step, dtype=np.int32),
        "last_step": np.asarray(args.steps, dtype=np.int32),
    }
    base._atomic_npz(checkpoint_path, checkpoint_values)
    prediction_values: dict[str, np.ndarray] = {}
    for name, value in ridge_predictions.items():
        prediction_values[f"ridge_only/{name}"] = value
    for name, value in combined_predictions.items():
        prediction_values[f"ridge_plus_residual/{name}"] = value
    base._atomic_npz(prediction_path, prediction_values)

    summary = {
        "probe": "phase_a_causal_clean_plan_compiler_jax_ridge_residual_oracle",
        "dataset": list(args.dataset),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "heldout_predictions_path": str(prediction_path.resolve()),
        "args": dataclasses.asdict(args),
        "input_contract": {
            "model_inputs": [
                f"teacher_ear_endpoint_{plan_horizon}x{args.active_action_dim}",
                f"shared_action_noise_{action_horizon}x{args.active_action_dim}",
            ],
            "forbidden_inputs": [
                "observation",
                "image",
                "state",
                "IAR",
                "task_id",
                "episode_id",
            ],
            "clean_and_intervention_share_action_noise": True,
        },
        "ridge_fit": {
            "split": "train_episode_split_only",
            "validation_data_used": False,
            "examples": ridge_counts,
            "lambda": args.ridge_lambda,
            "feature_dimension_without_bias": feature_dim,
            "design_dimension_with_bias": feature_dim + 1,
            "output_dimension": output_dim,
            "train_mse_active7": ridge_train_mse,
            "fit_seconds": ridge_fit_seconds,
        },
        "architecture": {
            "backend": "JAX/XLA JIT",
            "compiler": "frozen ridge base plus trainable residual MLP",
            "residual_width": args.width,
            "residual_blocks": args.blocks,
            "ridge_parameter_count": ridge_parameter_count,
            "residual_parameter_count": residual_parameter_count,
            "total_parameter_count": ridge_parameter_count + residual_parameter_count,
            "residual_output_initialization_gain": 0.01,
        },
        "dataset_records": int(len(arrays["dataset_index"])),
        "eligible_semantic_records": int(train_indices.size + validation_indices.size),
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "train_episode_groups": base._episode_group_count(arrays, train_indices),
        "validation_episode_groups": base._episode_group_count(arrays, validation_indices),
        "train_task_counts": base._task_counts(arrays, train_indices),
        "validation_task_counts": base._task_counts(arrays, validation_indices),
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
        "full_validation_checkpoint_step": best_step,
        "full_validation_metrics": full_metrics,
        "device": {
            "platform": device.platform,
            "kind": device.device_kind,
            "id": int(device.id),
        },
        "jax_version": jax.__version__,
        "elapsed_seconds": time.monotonic() - started,
    }
    base._atomic_json(summary_path, summary)
    print(
        "Full held-out ridge-only: "
        f"clean={ridge_metrics['clean_action_mse_active7']:.6f} "
        f"response={ridge_metrics['response_mse_active7']:.6f} "
        f"cosine={ridge_metrics['response_cosine_active7']:.4f} "
        f"gpu_ms={ridge_metrics['gpu_latency_mean_ms']}; "
        "ridge+residual: "
        f"clean={combined_metrics['clean_action_mse_active7']:.6f} "
        f"response={combined_metrics['response_mse_active7']:.6f} "
        f"cosine={combined_metrics['response_cosine_active7']:.4f} "
        f"gpu_ms={combined_metrics['gpu_latency_mean_ms']} best_step={best_step}",
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
