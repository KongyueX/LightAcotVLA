"""Train and calibrate the sub-100k JAX/Flax HARP temporal residual sidecar."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
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

from openpi.models import harp_temporal_residual as harp


LOGGER = logging.getLogger("train_harp_residual_jax")


@dataclasses.dataclass(frozen=True)
class Args:
    pairs: str
    output_dir: str
    seed: int = 7
    validation_fraction: float = 0.1
    train_steps: int = 1_500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 32
    temporal_layers: int = 2
    kernel_size: int = 3
    residual_mse_weight: float = 1.0
    temporal_delta_weight: float = 0.05
    margin_nll_weight: float = 0.25
    margin_train_fraction: float = 0.33
    conformal_alpha: float = 0.1
    margin_threshold: float = 0.0
    log_interval: int = 50
    latency_warmup: int = 10
    latency_runs: int = 100
    overwrite: bool = False


def _gpu() -> jax.Device:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("HARP training is GPU-only; no JAX GPU device was found.")
    LOGGER.info("Using JAX device %s", devices[0])
    return devices[0]


def _validate_args(args: Args) -> None:
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, .5).")
    if not 1 < args.train_steps < 10_000:
        raise ValueError("--train-steps must be in [2, 9999] for two-stage HARP training.")
    for name in (
        "batch_size",
        "hidden_dim",
        "temporal_layers",
        "kernel_size",
        "log_interval",
        "latency_warmup",
        "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be odd for same-time alignment.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings.")
    if min(args.residual_mse_weight, args.temporal_delta_weight, args.margin_nll_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if not 0.0 < args.margin_train_fraction < 1.0:
        raise ValueError("--margin-train-fraction must lie in (0, 1).")
    if not 0.0 < args.conformal_alpha < 1.0:
        raise ValueError("--conformal-alpha must lie in (0, 1).")
    if not np.isfinite(args.margin_threshold):
        raise ValueError("--margin-threshold must be finite.")


def _output_paths(args: Args) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output_dir = pathlib.Path(args.output_dir).resolve()
    model_path = output_dir / "model_params.npz"
    metrics_path = output_dir / "metrics.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
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
        raise FileNotFoundError(f"HARP pair file not found: {resolved}")
    required = {
        "dataset_index",
        "task_id",
        "episode_id",
        "frame_id",
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
            raise ValueError(f"Unsupported HARP pair schema {schema}; expected 3.")
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"HARP pair file is missing fields: {missing}")
        arrays = {name: np.asarray(handle[name]) for name in required}
        metadata = {
            name: (
                value.item() if isinstance(value, np.generic) else value
            )
            for name, value in handle.attrs.items()
        }
        for name in (
            "ear_normalization_scale",
            "ear_normalization_bias",
            "final_time_warp_alpha",
        ):
            if name not in handle.attrs:
                raise KeyError(f"HARP-v3 pair file is missing attribute {name!r}.")
        ear_normalization_scale = np.asarray(
            handle.attrs["ear_normalization_scale"], dtype=np.float32
        )
        ear_normalization_bias = np.asarray(
            handle.attrs["ear_normalization_bias"], dtype=np.float32
        )
    count = len(arrays["dataset_index"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("HARP pair arrays have inconsistent record counts.")
    if arrays["action_nfe1"].shape != arrays["action_nfe2"].shape:
        raise ValueError("NFE1/NFE2 pair shapes differ.")
    if arrays["action_nfe1"].shape[-1] < harp.CONTROL_DIM:
        raise ValueError("HARP pairs contain fewer than six continuous action dimensions.")
    if ear_normalization_scale.shape != (harp.CONTROL_DIM,):
        raise ValueError(
            "HARP-v2 ear_normalization_scale must have shape "
            f"({harp.CONTROL_DIM},), got {ear_normalization_scale.shape}."
        )
    if ear_normalization_bias.shape != (harp.CONTROL_DIM,):
        raise ValueError(
            "HARP-v2 ear_normalization_bias must have shape "
            f"({harp.CONTROL_DIM},), got {ear_normalization_bias.shape}."
        )
    if not np.all(np.isfinite(ear_normalization_scale)) or np.any(
        ear_normalization_scale <= 0.0
    ):
        raise ValueError("HARP-v2 ear_normalization_scale must be finite and positive.")
    if not np.all(np.isfinite(ear_normalization_bias)):
        raise ValueError("HARP-v2 ear_normalization_bias must be finite.")
    draft_final_time_warp_alpha = float(metadata["final_time_warp_alpha"])
    if not 0.0 <= draft_final_time_warp_alpha < 1.0:
        raise ValueError(
            "HARP-v3 final_time_warp_alpha must lie in [0, 1); "
            f"got {draft_final_time_warp_alpha}."
        )
    return arrays, metadata, ear_normalization_scale, ear_normalization_bias


def _episode_split(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    conformal_alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups)
    minimum_validation_groups = max(1, math.ceil(1.0 / conformal_alpha) - 1)
    if unique_groups.size <= minimum_validation_groups:
        raise ValueError(
            "HARP grouped conformal calibration needs more episode groups than its "
            f"minimum calibration count ({minimum_validation_groups}); got {unique_groups.size}."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    validation_count = min(
        max(minimum_validation_groups, round(unique_groups.size * validation_fraction)),
        unique_groups.size - 1,
    )
    validation_groups = unique_groups[:validation_count]
    validation = np.flatnonzero(np.isin(groups, validation_groups))
    train = np.flatnonzero(~np.isin(groups, validation_groups))
    if not train.size or not validation.size:
        raise ValueError("Episode-disjoint HARP split produced an empty partition.")
    if np.intersect1d(groups[train], groups[validation]).size:
        raise AssertionError("Episode leakage detected in HARP split.")
    return train, validation


def _prepare_examples(
    arrays: dict[str, np.ndarray],
    device: jax.Device,
    ear_normalization_scale: np.ndarray,
    ear_normalization_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_ear = jnp.asarray(arrays["ear"], dtype=jnp.float32)
    mapped_ear = jax.jit(harp.map_ear_to_action_normalization, device=device)(
        raw_ear,
        jnp.asarray(ear_normalization_scale, dtype=jnp.float32),
        jnp.asarray(ear_normalization_bias, dtype=jnp.float32),
    )
    build_features = jax.jit(harp.build_harp_features, device=device)
    features = build_features(
        jnp.asarray(arrays["action_nfe1"], dtype=jnp.float32),
        mapped_ear,
        jnp.asarray(arrays["iar"], dtype=jnp.float32),
        jnp.asarray(arrays["action_noise"], dtype=jnp.float32),
        jnp.asarray(arrays["state"], dtype=jnp.float32),
    )
    action_nfe1 = jax.device_put(
        jnp.asarray(arrays["action_nfe1"][..., : harp.CONTROL_DIM], dtype=jnp.float32),
        device,
    )
    action_nfe2 = jax.device_put(
        jnp.asarray(arrays["action_nfe2"][..., : harp.CONTROL_DIM], dtype=jnp.float32),
        device,
    )
    residual = action_nfe2 - action_nfe1
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in jax.device_get((features, residual))
    )


def _loss_and_metrics(
    model: harp.HARPTemporalResidualHead,
    params: Any,
    features: jax.Array,
    residual_target: jax.Array,
    *,
    target_scale: jax.Array,
    residual_gain: jax.Array,
    margin_center: jax.Array,
    margin_scale: jax.Array,
    residual_mse_weight: float,
    temporal_delta_weight: float,
    margin_nll_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    output = model.apply({"params": params}, features)
    predicted_residual = output[..., : harp.CONTROL_DIM]
    residual_error = jnp.mean(jnp.square(predicted_residual - residual_target), axis=-1)
    residual_mse = jnp.mean(residual_error)
    predicted_delta = predicted_residual[:, 1:] - predicted_residual[:, :-1]
    target_delta = residual_target[:, 1:] - residual_target[:, :-1]
    temporal_delta_mse = jnp.mean(jnp.square(predicted_delta - target_delta))

    raw_target = residual_target * target_scale
    candidate_residual = predicted_residual * target_scale * residual_gain
    margin_target = jax.lax.stop_gradient(
        jnp.sum(
            jnp.square(raw_target) - jnp.square(raw_target - candidate_residual),
            axis=-1,
        )
    )
    normalized_margin_target = (margin_target - margin_center) / margin_scale
    predicted_margin = output[..., harp.CONTROL_DIM]
    margin_log_variance = jnp.clip(output[..., harp.CONTROL_DIM + 1], -8.0, 8.0)
    margin_error = jnp.square(predicted_margin - normalized_margin_target)
    margin_nll = jnp.mean(
        0.5 * (jnp.exp(-margin_log_variance) * margin_error + margin_log_variance)
    )
    loss = (
        residual_mse_weight * residual_mse
        + temporal_delta_weight * temporal_delta_mse
        + margin_nll_weight * margin_nll
    )
    return loss, {
        "loss": loss,
        "residual_mse_normalized": residual_mse,
        "temporal_delta_mse": temporal_delta_mse,
        "margin_nll": margin_nll,
        "margin_mse_normalized": jnp.mean(margin_error),
        "mean_margin_target": jnp.mean(margin_target),
        "mean_margin_log_variance": jnp.mean(margin_log_variance),
    }


def _predict_outputs(
    model: harp.HARPTemporalResidualHead,
    params: Any,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    device: jax.Device,
    batch_size: int,
) -> np.ndarray:
    @jax.jit
    def apply(current_params: Any, batch: jax.Array) -> jax.Array:
        return model.apply({"params": current_params}, batch)

    chunks = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        batch = jax.device_put(jnp.asarray(features[selected], dtype=jnp.float32), device)
        chunks.append(np.asarray(jax.device_get(apply(params, batch)), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _fit_residual_gain(raw_residual: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(
        raw_residual.astype(np.float64) * target.astype(np.float64), axis=(0, 1)
    )
    denominator = np.sum(np.square(raw_residual.astype(np.float64)), axis=(0, 1))
    return np.clip(numerator / (denominator + 1e-8), 0.0, 1.0).astype(np.float32)


def _margin_target(
    target_residual: np.ndarray,
    predicted_residual: np.ndarray,
    residual_gain: np.ndarray,
) -> np.ndarray:
    candidate = predicted_residual * residual_gain
    return np.sum(
        np.square(target_residual) - np.square(target_residual - candidate), axis=-1
    ).astype(np.float32)


def _grouped_conformal_quantile(
    predicted_margin: np.ndarray,
    predicted_std: np.ndarray,
    target_margin: np.ndarray,
    task_id: np.ndarray,
    episode_id: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, int, int, np.ndarray]:
    if predicted_margin.shape != target_margin.shape or predicted_std.shape != target_margin.shape:
        raise ValueError("Conformal margin arrays must have identical [N,T] shapes.")
    if not np.all(np.isfinite(predicted_margin)) or not np.all(np.isfinite(target_margin)):
        raise ValueError("Conformal margins contain non-finite values.")
    if not np.all(np.isfinite(predicted_std)) or np.any(predicted_std <= 0.0):
        raise ValueError("Conformal margin standard deviations must be finite and positive.")
    groups = (
        np.asarray(task_id, dtype=np.int64) * np.int64(1_000_000_000)
        + np.asarray(episode_id, dtype=np.int64)
    )
    scores = (predicted_margin - target_margin) / np.maximum(predicted_std, 1e-8)
    unique_groups = np.unique(groups)
    group_scores = np.asarray(
        [np.max(scores[groups == group]) for group in unique_groups], dtype=np.float64
    )
    rank = math.ceil((unique_groups.size + 1) * (1.0 - alpha))
    if rank > unique_groups.size:
        raise ValueError(
            "Too few validation episode groups for the requested finite-sample conformal alpha: "
            f"groups={unique_groups.size}, alpha={alpha}."
        )
    quantile = float(np.partition(group_scores, rank - 1)[rank - 1])
    if not np.isfinite(quantile):
        raise ValueError("Grouped conformal quantile is non-finite.")
    return quantile, int(unique_groups.size), rank, group_scores


def _save_sidecar(
    target: pathlib.Path,
    *,
    model_config: harp.HARPHeadConfig,
    params: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_scale: np.ndarray,
    ear_normalization_scale: np.ndarray,
    ear_normalization_bias: np.ndarray,
    residual_gain: np.ndarray,
    margin_center: float,
    margin_scale: float,
    conformal_quantile: float,
    conformal_alpha: float,
    conformal_group_count: int,
    conformal_rank: int,
    margin_threshold: float,
    draft_final_time_warp_alpha: float,
    metadata: dict[str, Any],
) -> None:
    flat_params = traverse_util.flatten_dict(params)
    count = harp.parameter_count(params)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(harp.SCHEMA_VERSION, dtype=np.int32),
        "input_dim": np.asarray(model_config.input_dim, dtype=np.int32),
        "hidden_dim": np.asarray(model_config.hidden_dim, dtype=np.int32),
        "temporal_layers": np.asarray(model_config.temporal_layers, dtype=np.int32),
        "kernel_size": np.asarray(model_config.kernel_size, dtype=np.int32),
        "control_dim": np.asarray(model_config.control_dim, dtype=np.int32),
        "parameter_count": np.asarray(count, dtype=np.int32),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "target_scale": np.asarray(target_scale, dtype=np.float32),
        "ear_normalization_scale": np.asarray(ear_normalization_scale, dtype=np.float32),
        "ear_normalization_bias": np.asarray(ear_normalization_bias, dtype=np.float32),
        "residual_gain": np.asarray(residual_gain, dtype=np.float32),
        "margin_center": np.asarray(margin_center, dtype=np.float32),
        "margin_scale": np.asarray(margin_scale, dtype=np.float32),
        "conformal_quantile": np.asarray(conformal_quantile, dtype=np.float32),
        "conformal_alpha": np.asarray(conformal_alpha, dtype=np.float32),
        "conformal_group_count": np.asarray(conformal_group_count, dtype=np.int32),
        "conformal_rank": np.asarray(conformal_rank, dtype=np.int32),
        "margin_threshold": np.asarray(margin_threshold, dtype=np.float32),
        "draft_final_time_warp_alpha": np.asarray(
            draft_final_time_warp_alpha, dtype=np.float32
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for path, value in flat_params.items():
        payload[f"{harp.PARAM_PREFIX}{'/'.join(map(str, path))}"] = np.asarray(value, dtype=np.float32)
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
    temporary.replace(target)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = _gpu()
    output_dir, model_path, metrics_path = _output_paths(args)
    (
        arrays,
        pair_metadata,
        ear_normalization_scale,
        ear_normalization_bias,
    ) = _load_pairs(args.pairs)
    train_indices, validation_indices = _episode_split(
        arrays,
        validation_fraction=args.validation_fraction,
        conformal_alpha=args.conformal_alpha,
        seed=args.seed,
    )
    features, residual = _prepare_examples(
        arrays,
        device,
        ear_normalization_scale,
        ear_normalization_bias,
    )

    feature_mean = np.mean(features[train_indices], axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.std(features[train_indices], axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-5)
    target_scale = np.std(residual[train_indices], axis=(0, 1), dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(target_scale, 1e-4)
    normalized_features = (features - feature_mean) / feature_std
    normalized_residual = residual / target_scale

    config = harp.HARPHeadConfig(
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        kernel_size=args.kernel_size,
    )
    model = harp.HARPTemporalResidualHead(config)
    init_features = jax.device_put(
        jnp.asarray(normalized_features[:1], dtype=jnp.float32), device
    )
    params = model.init(jax.random.key(args.seed), init_features)["params"]
    parameter_count = harp.parameter_count(params)
    if parameter_count >= 100_000:
        raise ValueError(
            f"HARP configuration has {parameter_count} parameters; the pilot limit is 99999."
        )
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    optimizer_state = optimizer.init(params)
    target_scale_device = jax.device_put(jnp.asarray(target_scale), device)

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_features: jax.Array,
        batch_residual: jax.Array,
        residual_gain: jax.Array,
        margin_center: jax.Array,
        margin_scale: jax.Array,
        margin_nll_weight: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        def objective(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _loss_and_metrics(
                model,
                candidate,
                batch_features,
                batch_residual,
                target_scale=target_scale_device,
                residual_gain=residual_gain,
                margin_center=margin_center,
                margin_scale=margin_scale,
                residual_mse_weight=args.residual_mse_weight,
                temporal_delta_weight=args.temporal_delta_weight,
                margin_nll_weight=margin_nll_weight,
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
    validation_residual_device = jax.device_put(
        jnp.asarray(normalized_residual[validation_indices], dtype=jnp.float32), device
    )

    @jax.jit
    def validation_step(
        current_params: Any,
        residual_gain: jax.Array,
        margin_center: jax.Array,
        margin_scale: jax.Array,
        margin_nll_weight: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        _, metrics = _loss_and_metrics(
            model,
            current_params,
            validation_features_device,
            validation_residual_device,
            target_scale=target_scale_device,
            residual_gain=residual_gain,
            margin_center=margin_center,
            margin_scale=margin_scale,
            residual_mse_weight=args.residual_mse_weight,
            temporal_delta_weight=args.temporal_delta_weight,
            margin_nll_weight=margin_nll_weight,
        )
        output = model.apply({"params": current_params}, validation_features_device)
        return metrics, output

    margin_steps = max(1, round(args.train_steps * args.margin_train_fraction))
    residual_steps = args.train_steps - margin_steps
    if residual_steps <= 0:
        raise ValueError("--margin-train-fraction leaves no residual-only training steps.")
    LOGGER.info(
        "Initialized HARP-v3: train=%s calibration=%s params=%s residual_steps=%s margin_steps=%s",
        train_indices.size,
        validation_indices.size,
        parameter_count,
        residual_steps,
        margin_steps,
    )
    rng = np.random.default_rng(args.seed)
    metrics_mode = "w" if args.overwrite else "a"
    started = time.monotonic()
    last_validation_metrics: dict[str, float] = {}
    unit_gain_device = jax.device_put(jnp.ones((harp.CONTROL_DIM,), dtype=jnp.float32), device)
    zero_device = jax.device_put(jnp.asarray(0.0, dtype=jnp.float32), device)
    one_device = jax.device_put(jnp.asarray(1.0, dtype=jnp.float32), device)

    def log_metrics(
        metrics_file: Any,
        *,
        step: int,
        phase: str,
        train_metrics: dict[str, jax.Array],
        validation_metrics: dict[str, jax.Array],
    ) -> None:
        nonlocal last_validation_metrics
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
                    "phase": phase,
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
            "step=%s phase=%s train_loss=%.6f validation_loss=%.6f",
            step,
            phase,
            train_values["train/loss"],
            last_validation_metrics["validation/loss"],
        )

    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, residual_steps + 1):
            selected = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                jax.device_put(jnp.asarray(normalized_features[selected]), device),
                jax.device_put(jnp.asarray(normalized_residual[selected]), device),
                unit_gain_device,
                zero_device,
                one_device,
                zero_device,
            )
            if step == 1 or step % args.log_interval == 0 or step == residual_steps:
                validation_metrics, _ = validation_step(
                    params, unit_gain_device, zero_device, one_device, zero_device
                )
                log_metrics(
                    metrics_file,
                    step=step,
                    phase="residual",
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                )

        stage1_train_output = _predict_outputs(
            model,
            params,
            normalized_features,
            train_indices,
            device=device,
            batch_size=args.batch_size,
        )
        stage1_train_residual = (
            stage1_train_output[..., : harp.CONTROL_DIM] * target_scale
        )
        residual_gain = _fit_residual_gain(
            stage1_train_residual, residual[train_indices]
        )
        training_margin = _margin_target(
            residual[train_indices], stage1_train_residual, residual_gain
        )
        margin_center = float(np.mean(training_margin, dtype=np.float64))
        margin_scale = max(float(np.std(training_margin, dtype=np.float64)), 1e-6)
        gain_device = jax.device_put(jnp.asarray(residual_gain), device)
        margin_center_device = jax.device_put(jnp.asarray(margin_center, dtype=jnp.float32), device)
        margin_scale_device = jax.device_put(jnp.asarray(margin_scale, dtype=jnp.float32), device)
        margin_weight_device = jax.device_put(
            jnp.asarray(args.margin_nll_weight, dtype=jnp.float32), device
        )

        for margin_step in range(1, margin_steps + 1):
            step = residual_steps + margin_step
            selected = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                jax.device_put(jnp.asarray(normalized_features[selected]), device),
                jax.device_put(jnp.asarray(normalized_residual[selected]), device),
                gain_device,
                margin_center_device,
                margin_scale_device,
                margin_weight_device,
            )
            if margin_step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_metrics, _ = validation_step(
                    params,
                    gain_device,
                    margin_center_device,
                    margin_scale_device,
                    margin_weight_device,
                )
                log_metrics(
                    metrics_file,
                    step=step,
                    phase="margin",
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                )

    validation_output = _predict_outputs(
        model,
        params,
        normalized_features,
        validation_indices,
        device=device,
        batch_size=args.batch_size,
    )
    raw_validation_residual = validation_output[..., : harp.CONTROL_DIM] * target_scale
    validation_target = residual[validation_indices]
    validation_margin_target = _margin_target(
        validation_target, raw_validation_residual, residual_gain
    )
    validation_margin_mean = (
        validation_output[..., harp.CONTROL_DIM] * margin_scale + margin_center
    )
    validation_margin_log_variance = np.clip(
        validation_output[..., harp.CONTROL_DIM + 1], -8.0, 8.0
    )
    validation_margin_std = (
        np.exp(0.5 * validation_margin_log_variance.astype(np.float64)) * margin_scale
    ).astype(np.float32)
    conformal_quantile, conformal_group_count, conformal_rank, group_scores = (
        _grouped_conformal_quantile(
            validation_margin_mean,
            validation_margin_std,
            validation_margin_target,
            arrays["task_id"][validation_indices],
            arrays["episode_id"][validation_indices],
            alpha=args.conformal_alpha,
        )
    )
    validation_margin_lcb = (
        validation_margin_mean - conformal_quantile * validation_margin_std
    )
    validation_gate = validation_margin_lcb > args.margin_threshold
    candidate_validation_residual = raw_validation_residual * residual_gain
    applied_validation_residual = np.where(
        validation_gate[..., None], candidate_validation_residual, 0.0
    )

    baseline_mse = float(np.mean(np.square(validation_target)))
    raw_head_mse = float(np.mean(np.square(raw_validation_residual - validation_target)))
    gain_head_mse = float(
        np.mean(np.square(candidate_validation_residual - validation_target))
    )
    gated_mse = float(
        np.mean(np.square(applied_validation_residual - validation_target))
    )
    groups = (
        np.asarray(arrays["task_id"][validation_indices], dtype=np.int64)
        * np.int64(1_000_000_000)
        + np.asarray(arrays["episode_id"][validation_indices], dtype=np.int64)
    )
    margin_covered = validation_margin_target >= validation_margin_lcb
    episode_coverage = float(
        np.mean(
            [
                np.all(margin_covered[groups == group])
                for group in np.unique(groups)
            ]
        )
    )
    accepted_count = int(np.count_nonzero(validation_gate))
    accepted_harmful_rate = (
        float(np.mean(validation_margin_target[validation_gate] <= 0.0))
        if accepted_count
        else 0.0
    )
    draft_final_time_warp_alpha = float(pair_metadata["final_time_warp_alpha"])

    metadata = {
        "method": "HARP-v3-margin",
        "pair_path": str(pathlib.Path(args.pairs).resolve()),
        "pair_contract": pair_metadata.get("contract", "unknown"),
        "draft_final_time_warp_alpha": draft_final_time_warp_alpha,
        "draft_sampler": pair_metadata.get("draft_sampler", "unknown"),
        "teacher_sampler": pair_metadata.get("teacher_sampler", "unknown"),
        "seed": args.seed,
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "episode_disjoint_calibration": True,
        "gain_fit_partition": "train_only_after_residual_stage",
        "margin_formula": "||r_star||^2 - ||r_star - gain * mu||^2",
        "group_score": "max_episode((margin_mean-margin_target)/(margin_std+eps))",
        "gripper_unchanged": True,
        "ear_normalization_scale": ear_normalization_scale.tolist(),
        "ear_normalization_bias": ear_normalization_bias.tolist(),
        "residual_gain": residual_gain.tolist(),
    }
    _save_sidecar(
        model_path,
        model_config=config,
        params=params,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_scale=target_scale,
        ear_normalization_scale=ear_normalization_scale,
        ear_normalization_bias=ear_normalization_bias,
        residual_gain=residual_gain,
        margin_center=margin_center,
        margin_scale=margin_scale,
        conformal_quantile=conformal_quantile,
        conformal_alpha=args.conformal_alpha,
        conformal_group_count=conformal_group_count,
        conformal_rank=conformal_rank,
        margin_threshold=args.margin_threshold,
        draft_final_time_warp_alpha=draft_final_time_warp_alpha,
        metadata=metadata,
    )

    sidecar = harp.load_harp_residual_sidecar(model_path)
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
    latency_p95_ms = float(np.percentile(latency_ms, 95))

    summary = {
        **metadata,
        "model_params_path": str(model_path),
        "schema_version": harp.SCHEMA_VERSION,
        "parameter_count": parameter_count,
        "train_steps": args.train_steps,
        "residual_train_steps": residual_steps,
        "margin_train_steps": margin_steps,
        "feature_dim": harp.FEATURE_DIM,
        "margin_center": margin_center,
        "margin_scale": margin_scale,
        "conformal_alpha": args.conformal_alpha,
        "conformal_quantile": conformal_quantile,
        "conformal_group_count": conformal_group_count,
        "conformal_rank": conformal_rank,
        "conformal_group_score_min": float(np.min(group_scores)),
        "conformal_group_score_max": float(np.max(group_scores)),
        "margin_threshold": args.margin_threshold,
        "calibration_timestep_coverage": float(np.mean(margin_covered)),
        "calibration_episode_simultaneous_coverage": episode_coverage,
        "calibration_gate_rate": float(np.mean(validation_gate)),
        "calibration_accepted_count": accepted_count,
        "calibration_accepted_harmful_margin_rate": accepted_harmful_rate,
        "validation_nfe1_to_nfe2_mse_first6": baseline_mse,
        "validation_raw_head_mse_first6": raw_head_mse,
        "validation_gain_head_mse_first6": gain_head_mse,
        "validation_margin_gated_mse_first6": gated_mse,
        "validation_relative_mse_reduction": (
            (baseline_mse - gated_mse) / baseline_mse if baseline_mse > 0 else 0.0
        ),
        "sidecar_latency_p95_ms_batch1": latency_p95_ms,
        "latency_runs": args.latency_runs,
        "last_validation_metrics": last_validation_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "serving_contract": (
            "strict opt-in after direct final IR NFE1; apply gain*mu only when the "
            "episode-grouped conformal lower confidence bound exceeds the margin "
            "threshold; otherwise select A1 exactly; first six dimensions only"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "HARP-v3 complete: calibration MSE %.6f -> %.6f, gate=%.2f%%, params=%s, p95=%.4fms",
        baseline_mse,
        gated_mse,
        100.0 * float(np.mean(validation_gate)),
        parameter_count,
        latency_p95_ms,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
