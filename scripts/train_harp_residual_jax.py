"""Train and calibrate the sub-10k JAX/Flax HARP temporal residual sidecar."""

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
    gaussian_nll_weight: float = 1.0
    residual_mse_weight: float = 0.25
    reliability_weight: float = 0.1
    temporal_delta_weight: float = 0.05
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
    if not 0 < args.train_steps < 10_000:
        raise ValueError("--train-steps must be in [1, 9999] for the minimal HARP pilot.")
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
    if min(
        args.gaussian_nll_weight,
        args.residual_mse_weight,
        args.reliability_weight,
        args.temporal_delta_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")


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


def _load_pairs(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        if schema != 1:
            raise ValueError(f"Unsupported HARP pair schema {schema}; expected 1.")
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
    count = len(arrays["dataset_index"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("HARP pair arrays have inconsistent record counts.")
    if arrays["action_nfe1"].shape != arrays["action_nfe2"].shape:
        raise ValueError("NFE1/NFE2 pair shapes differ.")
    if arrays["action_nfe1"].shape[-1] < harp.CONTROL_DIM:
        raise ValueError("HARP pairs contain fewer than six continuous action dimensions.")
    return arrays, metadata


def _episode_split(
    arrays: dict[str, np.ndarray], *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(
            "HARP calibration requires at least two task/episode groups; record-level fallback is forbidden."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    validation_count = min(
        max(1, round(unique_groups.size * validation_fraction)),
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
    arrays: dict[str, np.ndarray], device: jax.Device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    build_features = jax.jit(harp.build_harp_features, device=device)
    features = build_features(
        jnp.asarray(arrays["action_nfe1"], dtype=jnp.float32),
        jnp.asarray(arrays["ear"], dtype=jnp.float32),
        jnp.asarray(arrays["iar"], dtype=jnp.float32),
        jnp.asarray(arrays["action_noise"], dtype=jnp.float32),
        jnp.asarray(arrays["state"], dtype=jnp.float32),
    )
    aligned_ear = jax.jit(harp.align_ear_to_action_time, static_argnums=(1,), device=device)(
        jnp.asarray(arrays["ear"], dtype=jnp.float32),
        arrays["action_nfe1"].shape[1],
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
    a1_error = jnp.mean(jnp.square(action_nfe1 - action_nfe2), axis=-1)
    ear_error = jnp.mean(
        jnp.square(aligned_ear[..., : harp.CONTROL_DIM] - action_nfe2), axis=-1
    )
    reliability = (ear_error < a1_error).astype(jnp.float32)
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in jax.device_get((features, residual, reliability))
    )


def _loss_and_metrics(
    model: harp.HARPTemporalResidualHead,
    params: Any,
    features: jax.Array,
    residual_target: jax.Array,
    reliability_target: jax.Array,
    *,
    gaussian_nll_weight: float,
    residual_mse_weight: float,
    reliability_weight: float,
    temporal_delta_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    output = model.apply({"params": params}, features)
    predicted_residual = output[..., : harp.CONTROL_DIM]
    log_variance = jnp.clip(output[..., harp.CONTROL_DIM], -8.0, 8.0)
    reliability_logit = output[..., harp.CONTROL_DIM + 1]
    residual_error = jnp.mean(jnp.square(predicted_residual - residual_target), axis=-1)
    gaussian_nll = jnp.mean(0.5 * (jnp.exp(-log_variance) * residual_error + log_variance))
    residual_mse = jnp.mean(residual_error)
    reliability_bce = jnp.mean(
        optax.sigmoid_binary_cross_entropy(reliability_logit, reliability_target)
    )
    predicted_delta = predicted_residual[:, 1:] - predicted_residual[:, :-1]
    target_delta = residual_target[:, 1:] - residual_target[:, :-1]
    temporal_delta_mse = jnp.mean(jnp.square(predicted_delta - target_delta))
    loss = (
        gaussian_nll_weight * gaussian_nll
        + residual_mse_weight * residual_mse
        + reliability_weight * reliability_bce
        + temporal_delta_weight * temporal_delta_mse
    )
    reliability_prediction = reliability_logit >= 0.0
    reliability_accuracy = jnp.mean(
        reliability_prediction == (reliability_target >= 0.5)
    )
    return loss, {
        "loss": loss,
        "gaussian_nll": gaussian_nll,
        "residual_mse_normalized": residual_mse,
        "reliability_bce": reliability_bce,
        "reliability_accuracy": reliability_accuracy,
        "temporal_delta_mse": temporal_delta_mse,
        "mean_log_variance": jnp.mean(log_variance),
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _calibrated_residual(
    raw_residual: np.ndarray,
    log_variance: np.ndarray,
    reliability: np.ndarray,
    *,
    residual_abs_limit: np.ndarray,
    residual_norm_limit: float,
    confidence_temperature: float,
    confidence_bias: float,
    trust_scale: float,
) -> np.ndarray:
    confidence = _sigmoid(
        (confidence_bias - log_variance) / max(confidence_temperature, 1e-3)
    )
    reliability_gate = np.clip(2.0 * reliability, 0.0, 1.0)
    projected = np.clip(
        raw_residual,
        -residual_abs_limit * trust_scale,
        residual_abs_limit * trust_scale,
    )
    norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    limit = max(residual_norm_limit * trust_scale, 1e-6)
    projected *= np.minimum(1.0, limit / (norm + 1e-6))
    return projected * (confidence * reliability_gate)[..., None]


def _save_sidecar(
    target: pathlib.Path,
    *,
    model_config: harp.HARPHeadConfig,
    params: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_scale: np.ndarray,
    residual_abs_limit: np.ndarray,
    residual_norm_limit: float,
    confidence_temperature: float,
    confidence_bias: float,
    trust_scale: float,
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
        "residual_abs_limit": np.asarray(residual_abs_limit, dtype=np.float32),
        "residual_norm_limit": np.asarray(residual_norm_limit, dtype=np.float32),
        "confidence_temperature": np.asarray(confidence_temperature, dtype=np.float32),
        "confidence_bias": np.asarray(confidence_bias, dtype=np.float32),
        "trust_scale": np.asarray(trust_scale, dtype=np.float32),
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
    arrays, pair_metadata = _load_pairs(args.pairs)
    train_indices, validation_indices = _episode_split(
        arrays, validation_fraction=args.validation_fraction, seed=args.seed
    )
    features, residual, reliability = _prepare_examples(arrays, device)

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

    loss_kwargs = {
        "gaussian_nll_weight": args.gaussian_nll_weight,
        "residual_mse_weight": args.residual_mse_weight,
        "reliability_weight": args.reliability_weight,
        "temporal_delta_weight": args.temporal_delta_weight,
    }

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_features: jax.Array,
        batch_residual: jax.Array,
        batch_reliability: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        def objective(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _loss_and_metrics(
                model,
                candidate,
                batch_features,
                batch_residual,
                batch_reliability,
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

    @jax.jit
    def validation_step(current_params: Any) -> tuple[dict[str, jax.Array], jax.Array]:
        validation_features = jnp.asarray(normalized_features[validation_indices], dtype=jnp.float32)
        validation_residual = jnp.asarray(normalized_residual[validation_indices], dtype=jnp.float32)
        validation_reliability = jnp.asarray(reliability[validation_indices], dtype=jnp.float32)
        _, metrics = _loss_and_metrics(
            model,
            current_params,
            validation_features,
            validation_residual,
            validation_reliability,
            **loss_kwargs,
        )
        output = model.apply({"params": current_params}, validation_features)
        return metrics, output

    LOGGER.info(
        "Initialized HARP: train=%s validation=%s episode-disjoint params=%s",
        train_indices.size,
        validation_indices.size,
        parameter_count,
    )
    rng = np.random.default_rng(args.seed)
    metrics_mode = "w" if args.overwrite else "a"
    started = time.monotonic()
    last_validation_metrics: dict[str, float] = {}
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            selected = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            batch_features = jax.device_put(
                jnp.asarray(normalized_features[selected], dtype=jnp.float32), device
            )
            batch_residual = jax.device_put(
                jnp.asarray(normalized_residual[selected], dtype=jnp.float32), device
            )
            batch_reliability = jax.device_put(
                jnp.asarray(reliability[selected], dtype=jnp.float32), device
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                batch_features,
                batch_residual,
                batch_reliability,
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
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **train_values,
                    **last_validation_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "step=%s train_loss=%.6f validation_loss=%.6f",
                    step,
                    train_values["train/loss"],
                    last_validation_metrics["validation/loss"],
                )

    _, validation_output_device = validation_step(params)
    validation_output = np.asarray(jax.device_get(validation_output_device), dtype=np.float32)
    raw_validation_residual = (
        validation_output[..., : harp.CONTROL_DIM] * target_scale
    )
    validation_log_variance = validation_output[..., harp.CONTROL_DIM]
    validation_reliability = _sigmoid(
        validation_output[..., harp.CONTROL_DIM + 1].astype(np.float64)
    )
    training_residual = residual[train_indices]
    residual_abs_limit = np.maximum(
        np.quantile(np.abs(training_residual), 0.95, axis=(0, 1)).astype(np.float32),
        1e-4,
    )
    residual_norm_limit = max(
        float(np.quantile(np.linalg.norm(training_residual, axis=-1), 0.95)),
        1e-4,
    )
    validation_target = residual[validation_indices]
    best: tuple[float, float, float, float] | None = None
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
        for bias in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            for trust_scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
                applied = _calibrated_residual(
                    raw_validation_residual,
                    validation_log_variance,
                    validation_reliability,
                    residual_abs_limit=residual_abs_limit,
                    residual_norm_limit=residual_norm_limit,
                    confidence_temperature=temperature,
                    confidence_bias=bias,
                    trust_scale=trust_scale,
                )
                mse = float(np.mean(np.square(applied - validation_target)))
                candidate = (mse, temperature, bias, trust_scale)
                if best is None or candidate < best:
                    best = candidate
    assert best is not None
    calibrated_mse, confidence_temperature, confidence_bias, trust_scale = best
    baseline_mse = float(np.mean(np.square(validation_target)))
    raw_head_mse = float(
        np.mean(np.square(raw_validation_residual - validation_target))
    )
    validation_reliability_target = reliability[validation_indices] >= 0.5
    reliability_accuracy = float(
        np.mean((validation_reliability >= 0.5) == validation_reliability_target)
    )

    metadata = {
        "method": "HARP",
        "pair_path": str(pathlib.Path(args.pairs).resolve()),
        "pair_contract": pair_metadata.get("contract", "unknown"),
        "seed": args.seed,
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "episode_disjoint_calibration": True,
        "gripper_unchanged": True,
    }
    _save_sidecar(
        model_path,
        model_config=config,
        params=params,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_scale=target_scale,
        residual_abs_limit=residual_abs_limit,
        residual_norm_limit=residual_norm_limit,
        confidence_temperature=confidence_temperature,
        confidence_bias=confidence_bias,
        trust_scale=trust_scale,
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
        "parameter_count": parameter_count,
        "train_steps": args.train_steps,
        "feature_dim": harp.FEATURE_DIM,
        "validation_nfe1_to_nfe2_mse_first6": baseline_mse,
        "validation_raw_head_mse_first6": raw_head_mse,
        "validation_calibrated_mse_first6": calibrated_mse,
        "validation_relative_mse_reduction": (
            (baseline_mse - calibrated_mse) / baseline_mse if baseline_mse > 0 else 0.0
        ),
        "validation_ear_reliability_accuracy": reliability_accuracy,
        "confidence_temperature": confidence_temperature,
        "confidence_bias": confidence_bias,
        "trust_scale": trust_scale,
        "residual_norm_limit": residual_norm_limit,
        "sidecar_latency_p95_ms_batch1": latency_p95_ms,
        "latency_runs": args.latency_runs,
        "last_validation_metrics": last_validation_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "serving_contract": (
            "strict opt-in after direct final IR NFE1; confidence shrink + trust projection "
            "on device; first six dimensions only"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "HARP complete: validation MSE %.6f -> %.6f, params=%s, p95=%.4fms",
        baseline_mse,
        calibrated_mse,
        parameter_count,
        latency_p95_ms,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
