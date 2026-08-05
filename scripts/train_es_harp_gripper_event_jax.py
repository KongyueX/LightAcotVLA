"""Train the standalone ES-HARP sparse gripper-event student on GPU."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

from flax import traverse_util
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.models import es_harp_gripper_event as es_harp


LOGGER = logging.getLogger("train_es_harp_gripper_event_jax")


@dataclasses.dataclass(frozen=True)
class Args:
    pairs: str
    output_dir: str
    seed: int = 7
    validation_fraction: float = 0.2
    train_steps: int = 1_500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 32
    temporal_layers: int = 1
    kernel_size: int = 3
    sign_loss_weight: float = 1.0
    flip_loss_weight: float = 2.0
    focal_gamma: float = 2.0
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95)
    minimum_precision: float = 0.95
    minimum_calibration_predictions: int = 3
    minimum_calibration_true_positives: int = 2
    log_interval: int = 50
    latency_warmup: int = 10
    latency_runs: int = 100
    overwrite: bool = False


def _gpu() -> jax.Device:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("ES-HARP training is GPU-only; no JAX GPU device was found.")
    LOGGER.info("Using JAX device %s", devices[0])
    return devices[0]


def _validate_args(args: Args) -> None:
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must lie in (0, .5).")
    if not 0 < args.train_steps < 10_000:
        raise ValueError("--train-steps must lie in [1, 9999].")
    for name in (
        "batch_size",
        "hidden_dim",
        "temporal_layers",
        "kernel_size",
        "minimum_calibration_predictions",
        "minimum_calibration_true_positives",
        "log_interval",
        "latency_warmup",
        "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be odd.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Invalid optimizer settings.")
    if min(args.sign_loss_weight, args.flip_loss_weight, args.focal_gamma) < 0.0:
        raise ValueError("Loss weights and focal gamma must be non-negative.")
    if not 2 <= len(args.thresholds) <= 4:
        raise ValueError("--thresholds must contain between two and four candidates.")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("--thresholds must be unique.")
    if any(not 0.0 < value < 1.0 for value in args.thresholds):
        raise ValueError("Every event threshold must lie in (0, 1).")
    if not 0.0 < args.minimum_precision <= 1.0:
        raise ValueError("--minimum-precision must lie in (0, 1].")


def _output_paths(args: Args) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output_dir = pathlib.Path(args.output_dir).resolve()
    model_path = output_dir / "model_params.npz"
    metrics_path = output_dir / "metrics.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose another or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        model_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        (output_dir / "summary.json").unlink(missing_ok=True)
    return output_dir, model_path, metrics_path


def _load_pairs(
    path: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    resolved = pathlib.Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"ES-HARP pair file not found: {resolved}")
    required = {
        "task_id",
        "episode_id",
        "state",
        "ear",
        "iar",
        "action_noise",
        "action_nfe1",
        "action_nfe2",
    }
    with h5py.File(resolved, "r") as handle:
        schema = int(handle.attrs.get("schema_version", -1))
        if schema != 3:
            raise ValueError(f"ES-HARP requires HARP pair schema 3; got {schema}.")
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"ES-HARP pair file is missing fields: {missing}")
        for name in (
            "ear_normalization_scale",
            "ear_normalization_bias",
            "final_time_warp_alpha",
            "teacher_num_steps",
            "teacher_conditioned_times",
        ):
            if name not in handle.attrs:
                raise KeyError(f"ES-HARP pair file is missing attribute {name!r}.")
        arrays = {name: np.asarray(handle[name]) for name in required}
        metadata = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in handle.attrs.items()
        }
        ear_control_scale = np.asarray(
            handle.attrs["ear_normalization_scale"], dtype=np.float32
        )
        ear_control_bias = np.asarray(
            handle.attrs["ear_normalization_bias"], dtype=np.float32
        )
        alpha = float(handle.attrs["final_time_warp_alpha"])
        teacher_num_steps = int(handle.attrs["teacher_num_steps"])
        teacher_times = np.asarray(
            handle.attrs["teacher_conditioned_times"], dtype=np.float64
        )
    count = len(arrays["task_id"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("ES-HARP pair arrays have inconsistent record counts.")
    if arrays["action_nfe1"].shape != arrays["action_nfe2"].shape:
        raise ValueError("ES-HARP A1/A2 shapes differ.")
    if arrays["action_noise"].shape != arrays["action_nfe1"].shape:
        raise ValueError("ES-HARP action-noise and A1 shapes differ.")
    if arrays["action_nfe1"].shape[-1] <= es_harp.GRIPPER_INDEX:
        raise ValueError("ES-HARP pairs contain no seventh gripper dimension.")
    if arrays["ear"].shape[-1] <= es_harp.GRIPPER_INDEX:
        raise ValueError("ES-HARP EAR contains no seventh gripper dimension.")
    if ear_control_scale.shape != (es_harp.CONTROL_DIM,) or ear_control_bias.shape != (
        es_harp.CONTROL_DIM,
    ):
        raise ValueError("ES-HARP EAR control affine must contain six values.")
    if not np.all(np.isfinite(ear_control_scale)) or np.any(ear_control_scale <= 0.0):
        raise ValueError("ES-HARP EAR control scale must be finite and positive.")
    if not np.all(np.isfinite(ear_control_bias)):
        raise ValueError("ES-HARP EAR control bias must be finite.")
    if abs(alpha - es_harp.REQUIRED_DRAFT_FINAL_TIME_WARP_ALPHA) > 1e-7:
        raise ValueError(
            "ES-HARP requires matched final_time_warp_alpha=0.05 pairs; "
            f"got {alpha}."
        )
    expected_times = np.asarray([0.95, 0.475], dtype=np.float64)
    if teacher_num_steps != 2 or teacher_times.shape != (2,) or not np.allclose(
        teacher_times, expected_times, atol=1e-7, rtol=0.0
    ):
        raise ValueError(
            "ES-HARP requires the matched two-step teacher at conditioned times [.95, .475]."
        )
    return arrays, metadata, ear_control_scale, ear_control_bias


def _targets(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_nfe1 = np.asarray(arrays["action_nfe1"], dtype=np.float32)
    action_nfe2 = np.asarray(arrays["action_nfe2"], dtype=np.float32)
    a1_positive = action_nfe1[..., es_harp.GRIPPER_INDEX] >= 0.0
    a2_positive = action_nfe2[..., es_harp.GRIPPER_INDEX] >= 0.0
    flip = a1_positive != a2_positive
    return a1_positive, a2_positive.astype(np.float32), flip.astype(np.float32)


def _episode_split(
    arrays: dict[str, np.ndarray],
    sign_target: np.ndarray,
    flip_target: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = (
        np.asarray(arrays["task_id"], dtype=np.int64) * np.int64(1_000_000_000)
        + np.asarray(arrays["episode_id"], dtype=np.int64)
    )
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("ES-HARP requires at least two task/episode groups.")
    validation_count = min(
        max(1, round(unique_groups.size * validation_fraction)),
        unique_groups.size - 1,
    )
    for attempt in range(256):
        shuffled = np.array(unique_groups, copy=True)
        np.random.default_rng(seed + attempt).shuffle(shuffled)
        validation_groups = shuffled[:validation_count]
        validation = np.flatnonzero(np.isin(groups, validation_groups))
        train = np.flatnonzero(~np.isin(groups, validation_groups))
        if not train.size or not validation.size:
            continue
        train_flip = flip_target[train]
        validation_flip = flip_target[validation]
        train_sign = sign_target[train]
        validation_sign = sign_target[validation]
        if (
            np.any(train_flip > 0.5)
            and np.any(validation_flip > 0.5)
            and np.any(train_flip < 0.5)
            and np.any(validation_flip < 0.5)
            and np.any(train_sign > 0.5)
            and np.any(train_sign < 0.5)
            and np.any(validation_sign > 0.5)
            and np.any(validation_sign < 0.5)
        ):
            if np.intersect1d(groups[train], groups[validation]).size:
                raise AssertionError("Episode leakage detected in ES-HARP split.")
            return train, validation
    raise ValueError(
        "Could not form an episode-disjoint ES-HARP split containing both sign classes "
        "and at least one flip in each partition."
    )


def _prepare_features(
    arrays: dict[str, np.ndarray],
    ear_control_scale: np.ndarray,
    ear_control_bias: np.ndarray,
    device: jax.Device,
) -> np.ndarray:
    build = jax.jit(es_harp.build_gripper_event_features, device=device)
    features = build(
        jnp.asarray(arrays["action_nfe1"], dtype=jnp.float32),
        jnp.asarray(arrays["ear"], dtype=jnp.float32),
        jnp.asarray(arrays["iar"], dtype=jnp.float32),
        jnp.asarray(arrays["action_noise"], dtype=jnp.float32),
        jnp.asarray(arrays["state"], dtype=jnp.float32),
        jnp.asarray(ear_control_scale, dtype=jnp.float32),
        jnp.asarray(ear_control_bias, dtype=jnp.float32),
    )
    return np.asarray(jax.device_get(features), dtype=np.float32)


def _balanced_weights(target: np.ndarray) -> tuple[float, float]:
    positive_rate = float(np.mean(target, dtype=np.float64))
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("Class-balanced loss requires both target classes in training data.")
    return 0.5 / positive_rate, 0.5 / (1.0 - positive_rate)


def _loss_and_metrics(
    model: es_harp.GripperEventHead,
    params: Any,
    features: jax.Array,
    sign_target: jax.Array,
    flip_target: jax.Array,
    *,
    sign_positive_weight: float,
    sign_negative_weight: float,
    flip_positive_weight: float,
    flip_negative_weight: float,
    sign_loss_weight: float,
    flip_loss_weight: float,
    focal_gamma: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = model.apply({"params": params}, features)
    sign_logit = logits[..., 0]
    flip_logit = logits[..., 1]
    sign_bce = optax.sigmoid_binary_cross_entropy(sign_logit, sign_target)
    sign_weight = jnp.where(
        sign_target >= 0.5, sign_positive_weight, sign_negative_weight
    )
    balanced_sign_bce = jnp.sum(sign_weight * sign_bce) / jnp.maximum(
        jnp.sum(sign_weight), 1e-6
    )

    flip_bce = optax.sigmoid_binary_cross_entropy(flip_logit, flip_target)
    flip_probability = jax.nn.sigmoid(flip_logit)
    flip_pt = jnp.where(flip_target >= 0.5, flip_probability, 1.0 - flip_probability)
    flip_weight = jnp.where(
        flip_target >= 0.5, flip_positive_weight, flip_negative_weight
    )
    focal_factor = jnp.power(jnp.maximum(1.0 - flip_pt, 0.0), focal_gamma)
    balanced_focal_flip_bce = jnp.sum(flip_weight * focal_factor * flip_bce) / jnp.maximum(
        jnp.sum(flip_weight), 1e-6
    )
    loss = (
        sign_loss_weight * balanced_sign_bce
        + flip_loss_weight * balanced_focal_flip_bce
    )
    return loss, {
        "loss": loss,
        "balanced_sign_bce": balanced_sign_bce,
        "balanced_focal_flip_bce": balanced_focal_flip_bce,
        "sign_accuracy": jnp.mean((sign_logit >= 0.0) == (sign_target >= 0.5)),
        "flip_accuracy_at_0_5": jnp.mean((flip_logit >= 0.0) == (flip_target >= 0.5)),
        "mean_flip_probability": jnp.mean(flip_probability),
    }


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(target, dtype=np.bool_).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(np.count_nonzero(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both positive and negative labels.")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        stop = start + 1
        while stop < labels.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(target, dtype=np.bool_).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        raise ValueError("Average precision requires a positive label.")
    order = np.argsort(-values, kind="mergesort")
    ranked_labels = labels[order]
    cumulative_true = np.cumsum(ranked_labels)
    precision = cumulative_true / np.arange(1, labels.size + 1)
    return float(np.sum(precision[ranked_labels]) / positives)


def _threshold_metrics(
    flip_target: np.ndarray,
    a1_positive: np.ndarray,
    predicted_positive: np.ndarray,
    flip_probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = np.asarray(flip_target, dtype=np.bool_)
    consistent = np.asarray(predicted_positive, dtype=np.bool_) != np.asarray(
        a1_positive, dtype=np.bool_
    )
    selected = (np.asarray(flip_probability) >= threshold) & consistent
    true_positive = int(np.count_nonzero(selected & target))
    false_positive = int(np.count_nonzero(selected & ~target))
    false_negative = int(np.count_nonzero(~selected & target))
    predicted_count = true_positive + false_positive
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "gate_rate": float(np.mean(selected)),
    }


def _select_threshold(
    metrics: list[dict[str, float | int]], args: Args
) -> tuple[dict[str, float | int], bool]:
    eligible = [
        item
        for item in metrics
        if float(item["precision"]) >= args.minimum_precision
        and int(item["predicted_count"]) >= args.minimum_calibration_predictions
        and int(item["true_positive"]) >= args.minimum_calibration_true_positives
    ]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                float(item["recall"]),
                float(item["f1"]),
                float(item["precision"]),
                float(item["threshold"]),
            ),
        ), True
    nonempty = [item for item in metrics if int(item["predicted_count"]) > 0]
    if not nonempty:
        raise ValueError("No candidate ES-HARP threshold emitted any validation event.")
    return max(
        nonempty,
        key=lambda item: (
            float(item["precision"]),
            float(item["recall"]),
            float(item["f1"]),
            float(item["threshold"]),
        ),
    ), False


def _save_sidecar(
    path: pathlib.Path,
    *,
    config: es_harp.GripperEventHeadConfig,
    params: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    ear_control_scale: np.ndarray,
    ear_control_bias: np.ndarray,
    event_threshold: float,
    positive_gripper_value: float,
    negative_gripper_value: float,
    metadata: dict[str, Any],
) -> None:
    count = es_harp.parameter_count(params)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(es_harp.SCHEMA_VERSION, dtype=np.int32),
        "input_dim": np.asarray(config.input_dim, dtype=np.int32),
        "hidden_dim": np.asarray(config.hidden_dim, dtype=np.int32),
        "temporal_layers": np.asarray(config.temporal_layers, dtype=np.int32),
        "kernel_size": np.asarray(config.kernel_size, dtype=np.int32),
        "parameter_count": np.asarray(count, dtype=np.int32),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "ear_control_scale": np.asarray(ear_control_scale, dtype=np.float32),
        "ear_control_bias": np.asarray(ear_control_bias, dtype=np.float32),
        "event_threshold": np.asarray(event_threshold, dtype=np.float32),
        "positive_gripper_value": np.asarray(positive_gripper_value, dtype=np.float32),
        "negative_gripper_value": np.asarray(negative_gripper_value, dtype=np.float32),
        "draft_final_time_warp_alpha": np.asarray(
            es_harp.REQUIRED_DRAFT_FINAL_TIME_WARP_ALPHA, dtype=np.float32
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for key, value in traverse_util.flatten_dict(params).items():
        payload[f"{es_harp.PARAM_PREFIX}{'/'.join(map(str, key))}"] = np.asarray(
            value, dtype=np.float32
        )
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
    temporary.replace(path)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = _gpu()
    output_dir, model_path, metrics_path = _output_paths(args)
    arrays, pair_metadata, ear_control_scale, ear_control_bias = _load_pairs(args.pairs)
    a1_positive, sign_target, flip_target = _targets(arrays)
    train_indices, validation_indices = _episode_split(
        arrays,
        sign_target,
        flip_target,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    features = _prepare_features(arrays, ear_control_scale, ear_control_bias, device)
    feature_mean = np.mean(features[train_indices], axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.std(features[train_indices], axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-5)
    normalized_features = (features - feature_mean) / feature_std

    sign_positive_weight, sign_negative_weight = _balanced_weights(sign_target[train_indices])
    flip_positive_weight, flip_negative_weight = _balanced_weights(flip_target[train_indices])
    config = es_harp.GripperEventHeadConfig(
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        kernel_size=args.kernel_size,
    )
    model = es_harp.GripperEventHead(config)
    init_features = jax.device_put(
        jnp.asarray(normalized_features[:1], dtype=jnp.float32), device
    )
    params = model.init(jax.random.key(args.seed), init_features)["params"]
    parameter_count = es_harp.parameter_count(params)
    if parameter_count >= 50_000:
        raise ValueError(
            f"ES-HARP configuration has {parameter_count} parameters; limit is 49999."
        )
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    optimizer_state = optimizer.init(params)
    loss_kwargs = {
        "sign_positive_weight": sign_positive_weight,
        "sign_negative_weight": sign_negative_weight,
        "flip_positive_weight": flip_positive_weight,
        "flip_negative_weight": flip_negative_weight,
        "sign_loss_weight": args.sign_loss_weight,
        "flip_loss_weight": args.flip_loss_weight,
        "focal_gamma": args.focal_gamma,
    }

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_features: jax.Array,
        batch_sign: jax.Array,
        batch_flip: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        def objective(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _loss_and_metrics(
                model,
                candidate,
                batch_features,
                batch_sign,
                batch_flip,
                **loss_kwargs,
            )

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current_params)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
        }

    validation_features_device = jax.device_put(
        jnp.asarray(normalized_features[validation_indices], dtype=jnp.float32), device
    )
    validation_sign_device = jax.device_put(
        jnp.asarray(sign_target[validation_indices], dtype=jnp.float32), device
    )
    validation_flip_device = jax.device_put(
        jnp.asarray(flip_target[validation_indices], dtype=jnp.float32), device
    )

    @jax.jit
    def validation_step(current_params: Any) -> tuple[dict[str, jax.Array], jax.Array]:
        _, metrics = _loss_and_metrics(
            model,
            current_params,
            validation_features_device,
            validation_sign_device,
            validation_flip_device,
            **loss_kwargs,
        )
        logits = model.apply({"params": current_params}, validation_features_device)
        return metrics, logits

    LOGGER.info(
        "Initialized ES-HARP: train=%s validation=%s params=%s train_flip_rate=%.5f",
        train_indices.size,
        validation_indices.size,
        parameter_count,
        float(np.mean(flip_target[train_indices])),
    )
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    last_validation_metrics: dict[str, float] = {}
    metrics_mode = "w" if args.overwrite else "a"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            selected = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                jax.device_put(jnp.asarray(normalized_features[selected]), device),
                jax.device_put(jnp.asarray(sign_target[selected]), device),
                jax.device_put(jnp.asarray(flip_target[selected]), device),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_metrics, _ = validation_step(params)
                train_values = {
                    f"train/{name}": float(value)
                    for name, value in jax.device_get(train_metrics).items()
                }
                last_validation_metrics = {
                    f"validation/{name}": float(value)
                    for name, value in jax.device_get(validation_metrics).items()
                }
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "elapsed_seconds": time.monotonic() - started,
                            **train_values,
                            **last_validation_metrics,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_file.flush()
                LOGGER.info(
                    "step=%s train_loss=%.6f validation_loss=%.6f",
                    step,
                    train_values["train/loss"],
                    last_validation_metrics["validation/loss"],
                )

    _, validation_logits_device = validation_step(params)
    validation_logits = np.asarray(
        jax.device_get(validation_logits_device), dtype=np.float32
    )
    sign_probability = np.asarray(jax.nn.sigmoid(validation_logits[..., 0]))
    flip_probability = np.asarray(jax.nn.sigmoid(validation_logits[..., 1]))
    predicted_positive = sign_probability >= 0.5
    threshold_metrics = [
        _threshold_metrics(
            flip_target[validation_indices],
            a1_positive[validation_indices],
            predicted_positive,
            flip_probability,
            threshold,
        )
        for threshold in args.thresholds
    ]
    selected_metrics, met_precision_target = _select_threshold(threshold_metrics, args)
    selected_threshold = float(selected_metrics["threshold"])

    training_a2_gripper = np.asarray(
        arrays["action_nfe2"][train_indices, :, es_harp.GRIPPER_INDEX], dtype=np.float32
    )
    positive_values = training_a2_gripper[training_a2_gripper >= 0.0]
    negative_values = training_a2_gripper[training_a2_gripper < 0.0]
    if not positive_values.size or not negative_values.size:
        raise ValueError("Training data lacks one A2 gripper-sign prototype.")
    positive_gripper_value = float(np.median(positive_values))
    negative_gripper_value = float(np.median(negative_values))

    metadata = {
        "method": "ES-HARP-gripper-event",
        "pair_path": str(pathlib.Path(args.pairs).resolve()),
        "pair_contract": pair_metadata.get("contract", "unknown"),
        "seed": args.seed,
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "episode_disjoint_split": True,
        "flip_definition": "(A1_gripper>=0)!=(A2_gripper>=0)",
        "threshold_selection": (
            "precision>=minimum then maximum recall; fallback maximum precision"
        ),
        "threshold_candidates": threshold_metrics,
        "threshold_met_precision_target": met_precision_target,
        "draft_final_time_warp_alpha": es_harp.REQUIRED_DRAFT_FINAL_TIME_WARP_ALPHA,
        "copy_contract": "only dimension 6 may change; all other A1 dimensions copied",
    }
    _save_sidecar(
        model_path,
        config=config,
        params=params,
        feature_mean=feature_mean,
        feature_std=feature_std,
        ear_control_scale=ear_control_scale,
        ear_control_bias=ear_control_bias,
        event_threshold=selected_threshold,
        positive_gripper_value=positive_gripper_value,
        negative_gripper_value=negative_gripper_value,
        metadata=metadata,
    )

    sidecar = es_harp.load_gripper_event_sidecar(model_path)
    apply_sidecar = jax.jit(sidecar.predict_and_correct, device=device)
    latency_index = int(validation_indices[0])
    latency_inputs = tuple(
        jax.device_put(jnp.asarray(value[latency_index : latency_index + 1], dtype=jnp.float32), device)
        for value in (
            arrays["action_nfe1"],
            arrays["ear"],
            arrays["iar"],
            arrays["action_noise"],
            arrays["state"],
        )
    )
    for _ in range(args.latency_warmup):
        jax.block_until_ready(apply_sidecar(*latency_inputs)["actions"])
    latency_ms = []
    for _ in range(args.latency_runs):
        latency_started = time.perf_counter()
        jax.block_until_ready(apply_sidecar(*latency_inputs)["actions"])
        latency_ms.append((time.perf_counter() - latency_started) * 1000.0)

    validation_sign_bool = sign_target[validation_indices] >= 0.5
    summary = {
        **metadata,
        "model_params_path": str(model_path),
        "schema_version": es_harp.SCHEMA_VERSION,
        "parameter_count": parameter_count,
        "feature_dim": es_harp.FEATURE_DIM,
        "train_steps": args.train_steps,
        "train_flip_rate": float(np.mean(flip_target[train_indices])),
        "validation_flip_rate": float(np.mean(flip_target[validation_indices])),
        "validation_flip_roc_auc": _binary_auc(
            flip_target[validation_indices], flip_probability
        ),
        "validation_flip_average_precision": _average_precision(
            flip_target[validation_indices], flip_probability
        ),
        "validation_sign_roc_auc": _binary_auc(validation_sign_bool, sign_probability),
        "validation_sign_accuracy": float(np.mean(predicted_positive == validation_sign_bool)),
        "selected_threshold": selected_threshold,
        "selected_precision": float(selected_metrics["precision"]),
        "selected_recall": float(selected_metrics["recall"]),
        "selected_f1": float(selected_metrics["f1"]),
        "selected_true_positive": int(selected_metrics["true_positive"]),
        "selected_false_positive": int(selected_metrics["false_positive"]),
        "selected_gate_rate": float(selected_metrics["gate_rate"]),
        "positive_gripper_value": positive_gripper_value,
        "negative_gripper_value": negative_gripper_value,
        "sidecar_latency_p95_ms_batch1": float(np.percentile(latency_ms, 95)),
        "latency_runs": args.latency_runs,
        "last_validation_metrics": last_validation_metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "ES-HARP complete: threshold=%.3f precision=%.4f recall=%.4f F1=%.4f params=%s",
        selected_threshold,
        float(selected_metrics["precision"]),
        float(selected_metrics["recall"]),
        float(selected_metrics["f1"]),
        parameter_count,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
