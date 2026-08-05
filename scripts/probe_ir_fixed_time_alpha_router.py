"""Probe fixed final-time warps and fit an episode-held-out linear router.

This is an intentionally standalone GPU probe.  It loads the frozen ACoT-VLA
base checkpoint plus a final-IR endpoint sidecar, reconstructs the canonical
training observations for a teacher-label dataset, and evaluates a grid of
direct one-step final-action calls.  Each observation runs the VLM prefix and
IAR exactly once; the fixed teacher EAR, final-flow noise, prefix cache, and IAR
are then shared by every alpha candidate.

The script never changes the normal policy/evaluation path.  It writes:

* per-record current-observation features, alpha actions, and action risks;
* a multi-output ridge model that predicts risk for every alpha;
* a scalar ridge approximation in the existing adaptive time-warp gate format;
* gate-only and final-IR-plus-gate Orbax sidecars; and
* a leakage-safe nested/outer episode-held-out report.

The offline teacher-action risk is a routing probe, not a task-success result.
The exported router must still be checked with closed-loop LIBERO evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import logging
import pathlib
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import endpoint_dataset
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader

try:
    import train_acot_endpoint_distillation as endpoint_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_acot_endpoint_distillation as endpoint_trainer


LOGGER = logging.getLogger("probe_ir_fixed_time_alpha_router")

_ADAPTIVE_CENTER = 0.05
_ADAPTIVE_RADIUS = 0.05


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    seed: int = 7
    outer_holdout_fraction: float = 0.1
    inner_validation_fraction: float = 0.1
    expected_outer_holdout_count: int | None = 206
    batch_size: int = 8
    alphas: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.1)
    ridge_lambdas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1_000.0)
    huber_delta: float = 0.1
    gripper_weight: float = 4.0
    gripper_sign_threshold: float = 0.0
    feature_std_floor: float = 1e-5
    gate_alpha_margin: float = 1e-3
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    ridge_lambda: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(features, dtype=np.float32) - self.feature_mean
        ) / self.feature_std
        return normalized @ self.weights + self.target_mean


@dataclasses.dataclass(frozen=True)
class RidgeSpectrum:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    normalized_features: jax.Array
    eigenvalues: jax.Array
    eigenvectors: jax.Array
    device: jax.Device


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.outer_holdout_fraction < 0.5:
        raise ValueError("--outer-holdout-fraction must be in (0, .5).")
    if not 0.0 < args.inner_validation_fraction < 0.5:
        raise ValueError("--inner-validation-fraction must be in (0, .5).")
    if args.expected_outer_holdout_count is not None and args.expected_outer_holdout_count <= 0:
        raise ValueError("--expected-outer-holdout-count must be positive or None.")
    if not args.alphas:
        raise ValueError("--alphas cannot be empty.")
    alpha_values = np.asarray(args.alphas, dtype=np.float64)
    if np.any(~np.isfinite(alpha_values)) or np.any(np.diff(alpha_values) <= 0):
        raise ValueError("--alphas must be finite, unique, and strictly increasing.")
    if alpha_values[0] < 0.0 or alpha_values[-1] > 0.1:
        raise ValueError(
            "The existing adaptive gate represents alpha in [0, .1]; keep the probe grid in that range."
        )
    if not np.any(np.isclose(alpha_values, 0.0, atol=1e-12)):
        raise ValueError("--alphas must contain the unwarped alpha=0 baseline.")
    if not args.ridge_lambdas or any(value <= 0 for value in args.ridge_lambdas):
        raise ValueError("--ridge-lambdas must contain positive values.")
    if args.huber_delta <= 0 or args.gripper_weight <= 0:
        raise ValueError("--huber-delta and --gripper-weight must be positive.")
    if args.feature_std_floor <= 0:
        raise ValueError("--feature-std-floor must be positive.")
    if not 0.0 < args.gate_alpha_margin < _ADAPTIVE_RADIUS:
        raise ValueError("--gate-alpha-margin must be in (0, .05).")


def _select_gpu() -> jax.Device:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError(
            "This probe is GPU-only because it evaluates the ACoT backbone; no JAX GPU device was found."
        )
    device = devices[0]
    LOGGER.info("Using JAX device %s", device)
    return device


def _prepare_output_dir(args: Args) -> pathlib.Path:
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _episode_groups(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    task = np.asarray(arrays["task_id"], dtype=np.int64)[indices]
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)[indices]
    return task * np.int64(1_000_000_000) + episode


def _nested_episode_split(
    arrays: dict[str, np.ndarray],
    outer_train_indices: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the endpoint trainer's exact grouped split inside outer train."""

    subset = {name: values[outer_train_indices] for name, values in arrays.items()}
    inner_train_relative, inner_validation_relative = endpoint_trainer._split_indices(  # noqa: SLF001
        subset,
        validation_fraction=validation_fraction,
        seed=seed,
        require_semantic_intervention=True,
    )
    inner_train = outer_train_indices[inner_train_relative]
    inner_validation = outer_train_indices[inner_validation_relative]
    if np.intersect1d(
        _episode_groups(arrays, inner_train),
        _episode_groups(arrays, inner_validation),
    ).size:
        raise AssertionError("Inner train/validation episode leakage detected.")
    return inner_train, inner_validation


def _load_model_and_observations(
    args: Args,
) -> tuple[
    Any,
    data_loader.Dataset,
    dict[str, np.ndarray],
    dict[str, Any],
    pathlib.Path,
]:
    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_config = config_lib.get_config(args.config_name)
    model_config = train_config.model

    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    base_params_path = checkpoint_dir / "params"
    if not base_params_path.exists():
        raise FileNotFoundError(f"Base checkpoint params not found: {base_params_path}")
    base_params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(base_params_path, dtype=jnp.bfloat16)
    )

    sidecar_path = pathlib.Path(download.maybe_download(args.endpoint_student_params))
    sidecar_params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(sidecar_path, dtype=jnp.bfloat16)
    )
    merged_params = policy_config.merge_acot_endpoint_student_params(
        base_params,
        sidecar_params,
    )
    model = model_config.load(merged_params)
    model.eval()

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = endpoint_trainer._load_norm_stats(  # noqa: SLF001
        train_config,
        data_config,
        checkpoint_dir,
    )
    data_config = endpoint_trainer._with_norm_stats(data_config, norm_stats)  # noqa: SLF001
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    observation_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    return model, observation_dataset, arrays, sidecar_params, sidecar_path


def _pooled_prefix(prefix_state: Mapping[str, Any]) -> jax.Array:
    prefix_out = prefix_state["prefix_out"]
    mask = prefix_state["prefix_mask"].astype(prefix_out.dtype)
    return jnp.asarray(
        jnp.sum(prefix_out * mask[..., None], axis=1)
        / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0),
        dtype=jnp.float32,
    )


def _extract_grid(
    model: Any,
    observation_dataset: data_loader.Dataset,
    arrays: dict[str, np.ndarray],
    *,
    alphas: Sequence[float],
    batch_size: int,
    seed: int,
    device: jax.Device,
) -> dict[str, np.ndarray]:
    prefix_fn = nnx_utils.module_jit(model.sample_actions_profile_prefix)
    implicit_fn = nnx_utils.module_jit(model.sample_actions_profile_implicit)
    # Alpha remains dynamic in this offline sweep, so all five values share one
    # compiled graph.  Deployment can still specialize the chosen fixed alpha.
    expert_fn = nnx_utils.module_jit(model.sample_actions_profile_direct_one_step_expert)

    count = len(arrays["dataset_index"])
    alpha_values = np.asarray(alphas, dtype=np.float32)
    features: list[np.ndarray] = []
    prefix_features: list[np.ndarray] = []
    normalized_states: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        selected = np.arange(start, stop, dtype=np.int64)
        real_count = selected.size
        if real_count < batch_size:
            selected = np.pad(selected, (0, batch_size - real_count), mode="edge")
        batch = endpoint_trainer._make_batch(  # noqa: SLF001
            observation_dataset,
            arrays,
            selected,
            rng,
            deterministic_intervention=True,
        )
        observation = jax.device_put(batch["observation"], device)
        action_noise = jax.device_put(
            jnp.asarray(batch["action_noise"], dtype=jnp.float32),
            device,
        )
        teacher_coarse = jax.device_put(
            jnp.asarray(batch["teacher_coarse"], dtype=jnp.float32),
            device,
        )

        batch_key = jax.random.fold_in(jax.random.key(seed), start)
        prefix_state = prefix_fn(batch_key, observation)
        # Endpoint labels store the exact final-flow noise used by the teacher.
        # Replacing only this generated field preserves the shared prefix/cache.
        prefix_state = {**prefix_state, "expert_action_noise": action_noise}
        implicit_reason = implicit_fn(prefix_state)["implicit_action_reason"]
        if implicit_reason is None:
            raise ValueError("The selected config does not expose an implicit action reasoner.")

        pooled = _pooled_prefix(prefix_state)
        state = jnp.asarray(
            prefix_state["observation"].state[..., : model.action_dim],
            dtype=jnp.float32,
        )
        alpha_outputs = []
        for alpha in alpha_values:
            output = expert_fn(
                prefix_state,
                teacher_coarse,
                implicit_reason,
                jnp.asarray(alpha, dtype=jnp.float32),
            )
            alpha_outputs.append(output["actions"])
        actions = jnp.stack(alpha_outputs, axis=1)
        pooled, state, actions = jax.device_get((pooled, state, actions))

        pooled_np = np.asarray(pooled[:real_count], dtype=np.float32)
        state_np = np.asarray(state[:real_count], dtype=np.float32)
        prefix_features.append(pooled_np)
        normalized_states.append(state_np)
        features.append(np.concatenate([pooled_np, state_np], axis=-1))
        predicted_actions.append(
            np.asarray(actions[:real_count, :, :, :7], dtype=np.float32)
        )
        LOGGER.info("Extracted alpha grid for %s/%s records", stop, count)

    result = {
        "prefix_feature": np.concatenate(prefix_features, axis=0),
        "normalized_state": np.concatenate(normalized_states, axis=0),
        "router_feature": np.concatenate(features, axis=0),
        "predicted_actions_active7": np.concatenate(predicted_actions, axis=0),
    }
    non_finite = {
        name: int(np.sum(~np.isfinite(values)))
        for name, values in result.items()
    }
    if any(non_finite.values()):
        raise FloatingPointError(f"Alpha extraction produced non-finite values: {non_finite}")
    return result


def _huber(error: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(error)
    quadratic = np.minimum(absolute, delta)
    return 0.5 * np.square(quadratic) + delta * (absolute - quadratic)


def _action_risks(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta: float,
    gripper_weight: float,
    gripper_sign_threshold: float,
) -> dict[str, np.ndarray]:
    """Return per-alpha first-7D risks without using future observations."""

    target_active = np.asarray(target, dtype=np.float32)[..., :7]
    error = np.asarray(predicted, dtype=np.float32) - target_active[:, None, :, :]
    squared = np.square(error)
    huber = _huber(error, huber_delta)
    per_dim_mse = np.mean(squared, axis=2)
    per_dim_huber = np.mean(huber, axis=2)

    predicted_gripper = predicted[..., 6] >= gripper_sign_threshold
    target_gripper = target_active[:, None, :, 6] >= gripper_sign_threshold
    gripper_sign_error = np.mean(predicted_gripper != target_gripper, axis=2).astype(np.float32)

    continuous_mse = np.mean(per_dim_mse[..., :6], axis=-1)
    continuous_huber = np.mean(per_dim_huber[..., :6], axis=-1)
    gripper_mse = per_dim_mse[..., 6]
    gripper_huber = per_dim_huber[..., 6]
    weighted_mse = (
        np.sum(per_dim_mse[..., :6], axis=-1) + gripper_weight * gripper_mse
    ) / (6.0 + gripper_weight)
    weighted_huber = (
        np.sum(per_dim_huber[..., :6], axis=-1) + gripper_weight * gripper_huber
    ) / (6.0 + gripper_weight)
    # This is the router label: robust continuous error plus the deployment-
    # relevant gripper event/sign failure requested by the probe.
    weighted_error = (
        np.sum(per_dim_huber[..., :6], axis=-1)
        + gripper_weight * gripper_sign_error
    ) / (6.0 + gripper_weight)
    return {
        "per_dim_mse_active7": per_dim_mse.astype(np.float32),
        "per_dim_huber_active7": per_dim_huber.astype(np.float32),
        "continuous_mse_6d": continuous_mse.astype(np.float32),
        "continuous_huber_6d": continuous_huber.astype(np.float32),
        "gripper_mse": gripper_mse.astype(np.float32),
        "gripper_huber": gripper_huber.astype(np.float32),
        "gripper_sign_error": gripper_sign_error,
        "weighted_mse": weighted_mse.astype(np.float32),
        "weighted_huber": weighted_huber.astype(np.float32),
        "weighted_error": weighted_error.astype(np.float32),
    }


def _build_ridge_spectrum(
    features: np.ndarray,
    *,
    std_floor: float,
    device: jax.Device,
) -> RidgeSpectrum:
    values = np.asarray(features, dtype=np.float32)
    feature_mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, std_floor)
    normalized = (values - feature_mean) / feature_std
    normalized_device = jax.device_put(jnp.asarray(normalized, dtype=jnp.float32), device)
    gram = normalized_device @ normalized_device.T
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    jax.block_until_ready(eigenvectors)
    return RidgeSpectrum(
        feature_mean=feature_mean,
        feature_std=feature_std,
        normalized_features=normalized_device,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        device=device,
    )


def _fit_from_spectrum(
    spectrum: RidgeSpectrum,
    targets: np.ndarray,
    ridge_lambda: float,
) -> RidgeModel:
    target_values = np.asarray(targets, dtype=np.float32)
    if target_values.ndim == 1:
        target_values = target_values[:, None]
    target_mean = np.mean(target_values, axis=0, dtype=np.float64).astype(np.float32)
    centered = jax.device_put(
        jnp.asarray(target_values - target_mean, dtype=jnp.float32),
        spectrum.device,
    )
    projected = spectrum.eigenvectors.T @ centered
    inverse = 1.0 / (spectrum.eigenvalues + float(ridge_lambda))
    dual = spectrum.eigenvectors @ (projected * inverse[:, None])
    weights = spectrum.normalized_features.T @ dual
    weights_np = np.asarray(jax.device_get(weights), dtype=np.float32)
    return RidgeModel(
        feature_mean=spectrum.feature_mean,
        feature_std=spectrum.feature_std,
        target_mean=target_mean,
        weights=weights_np,
        ridge_lambda=float(ridge_lambda),
    )


def _baseline_index(alphas: np.ndarray) -> int:
    matches = np.flatnonzero(np.isclose(alphas, 0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError("Expected exactly one alpha=0 baseline.")
    return int(matches[0])


def _selection_metrics(
    selected: np.ndarray,
    actual_risk: np.ndarray,
    oracle: np.ndarray,
    *,
    alphas: np.ndarray,
    baseline_index: int,
) -> dict[str, Any]:
    rows = np.arange(selected.size)
    realized = actual_risk[rows, selected]
    oracle_risk = actual_risk[rows, oracle]
    baseline_risk = actual_risk[:, baseline_index]
    regret = realized - oracle_risk
    relative_regret = regret / np.maximum(baseline_risk, 1e-8)
    return {
        "count": int(selected.size),
        "top1_accuracy": float(np.mean(selected == oracle)),
        "selected_nonzero_coverage": float(np.mean(selected != baseline_index)),
        "realized_improvement_coverage_vs_alpha0": float(np.mean(realized < baseline_risk)),
        "oracle_improvement_coverage_vs_alpha0": float(np.mean(oracle_risk < baseline_risk)),
        "mean_realized_risk": float(np.mean(realized)),
        "mean_alpha0_risk": float(np.mean(baseline_risk)),
        "mean_oracle_risk": float(np.mean(oracle_risk)),
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.quantile(regret, 0.9)),
        "mean_relative_regret_vs_alpha0": float(np.mean(relative_regret)),
        "selected_alpha_distribution": {
            str(float(alpha)): int(np.sum(selected == index))
            for index, alpha in enumerate(alphas)
        },
    }


def _risk_router_metrics(
    predicted_risk: np.ndarray,
    actual_risk: np.ndarray,
    oracle: np.ndarray,
    *,
    alphas: np.ndarray,
    baseline_index: int,
) -> dict[str, Any]:
    selected = np.argmin(predicted_risk, axis=-1)
    metrics = _selection_metrics(
        selected,
        actual_risk,
        oracle,
        alphas=alphas,
        baseline_index=baseline_index,
    )
    top2 = np.argsort(predicted_risk, axis=-1)[:, : min(2, predicted_risk.shape[1])]
    metrics["top2_oracle_coverage"] = float(np.mean(np.any(top2 == oracle[:, None], axis=1)))
    metrics["risk_prediction_mse"] = float(np.mean(np.square(predicted_risk - actual_risk)))
    return metrics


def _nearest_alpha_indices(alpha: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(alpha[:, None] - alphas[None, :]), axis=-1)


def _oracle_gate_logits(
    oracle: np.ndarray,
    alphas: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:
    oracle_alpha = alphas[oracle]
    clipped = np.clip(oracle_alpha, margin, 0.1 - margin)
    normalized = (clipped - _ADAPTIVE_CENTER) / _ADAPTIVE_RADIUS
    return np.arctanh(np.clip(normalized, -0.999999, 0.999999)).astype(np.float32)


def _gate_selection(
    model: RidgeModel,
    features: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logit = model.predict(features).reshape(-1)
    continuous_alpha = _ADAPTIVE_CENTER + _ADAPTIVE_RADIUS * np.tanh(logit)
    selected = _nearest_alpha_indices(continuous_alpha, alphas)
    return selected, continuous_alpha.astype(np.float32), logit.astype(np.float32)


def _choose_models(
    features: np.ndarray,
    risks: np.ndarray,
    oracle: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    *,
    alphas: np.ndarray,
    ridge_lambdas: Sequence[float],
    feature_std_floor: float,
    gate_alpha_margin: float,
    device: jax.Device,
) -> tuple[float, float, dict[str, Any]]:
    baseline_index = _baseline_index(alphas)
    spectrum = _build_ridge_spectrum(
        features[inner_train],
        std_floor=feature_std_floor,
        device=device,
    )
    gate_targets = _oracle_gate_logits(
        oracle[inner_train],
        alphas,
        margin=gate_alpha_margin,
    )
    risk_candidates: dict[str, Any] = {}
    gate_candidates: dict[str, Any] = {}
    risk_rankings: list[tuple[float, float, float]] = []
    gate_rankings: list[tuple[float, float, float]] = []
    for ridge_lambda in ridge_lambdas:
        risk_model = _fit_from_spectrum(spectrum, risks[inner_train], ridge_lambda)
        risk_metrics = _risk_router_metrics(
            risk_model.predict(features[inner_validation]),
            risks[inner_validation],
            oracle[inner_validation],
            alphas=alphas,
            baseline_index=baseline_index,
        )
        risk_candidates[str(float(ridge_lambda))] = risk_metrics
        risk_rankings.append(
            (
                float(risk_metrics["mean_regret"]),
                -float(risk_metrics["top1_accuracy"]),
                float(ridge_lambda),
            )
        )

        gate_model = _fit_from_spectrum(spectrum, gate_targets, ridge_lambda)
        gate_selected, _, _ = _gate_selection(
            gate_model,
            features[inner_validation],
            alphas,
        )
        gate_metrics = _selection_metrics(
            gate_selected,
            risks[inner_validation],
            oracle[inner_validation],
            alphas=alphas,
            baseline_index=baseline_index,
        )
        gate_candidates[str(float(ridge_lambda))] = gate_metrics
        gate_rankings.append(
            (
                float(gate_metrics["mean_regret"]),
                -float(gate_metrics["top1_accuracy"]),
                float(ridge_lambda),
            )
        )

    selected_risk_lambda = min(risk_rankings)[2]
    selected_gate_lambda = min(gate_rankings)[2]
    return selected_risk_lambda, selected_gate_lambda, {
        "risk_router": risk_candidates,
        "adaptive_gate_approximation": gate_candidates,
        "selected_risk_lambda": selected_risk_lambda,
        "selected_gate_lambda": selected_gate_lambda,
        "selection_rule": "minimum mean realized regret, then maximum top1, then smaller lambda",
    }


def _raw_affine(model: RidgeModel) -> tuple[np.ndarray, np.ndarray]:
    kernel = model.weights / model.feature_std[:, None]
    bias = model.target_mean - (model.feature_mean / model.feature_std) @ model.weights
    return kernel.astype(np.float32), np.asarray(bias, dtype=np.float32)


def _save_router(path: pathlib.Path, model: RidgeModel, alphas: np.ndarray) -> None:
    kernel, bias = _raw_affine(model)
    np.savez_compressed(
        path,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        target_mean=model.target_mean,
        weights=model.weights,
        raw_kernel=kernel,
        raw_bias=bias,
        ridge_lambda=np.asarray(model.ridge_lambda, dtype=np.float32),
        alphas=alphas,
        feature_order=np.asarray(["pooled_prefix_2048", "normalized_state_32"]),
    )


def _save_orbax_sidecar(
    target: pathlib.Path,
    params: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, {"params": params}, force=overwrite)


def _save_gate_sidecars(
    output_dir: pathlib.Path,
    gate_model: RidgeModel,
    final_ir_params: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    kernel, bias = _raw_affine(gate_model)
    if kernel.shape[1:] != (1,) or bias.shape != (1,):
        raise ValueError(f"Adaptive gate must be scalar; got kernel={kernel.shape}, bias={bias.shape}.")
    gate_flat = {
        ("adaptive_final_time_warp_gate", "kernel"): kernel,
        ("adaptive_final_time_warp_gate", "bias"): bias,
    }
    gate_params = traverse_util.unflatten_dict(gate_flat)
    combined_flat = dict(traverse_util.flatten_dict(final_ir_params))
    overlap = set(combined_flat).intersection(gate_flat)
    if overlap:
        raise ValueError(f"Input final-IR sidecar already contains adaptive gate leaves: {sorted(overlap)}")
    combined_flat.update(gate_flat)
    combined_params = traverse_util.unflatten_dict(combined_flat)

    gate_path = output_dir / "adaptive_gate_only" / "params"
    combined_path = output_dir / "combined_final_ir_adaptive_gate" / "params"
    _save_orbax_sidecar(gate_path, gate_params, overwrite=overwrite)
    _save_orbax_sidecar(combined_path, combined_params, overwrite=overwrite)
    return gate_path, combined_path


def _json_dump(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    _validate_args(args)
    output_dir = _prepare_output_dir(args)
    device = _select_gpu()
    alphas = np.asarray(args.alphas, dtype=np.float32)
    baseline_index = _baseline_index(alphas)

    model, observation_dataset, arrays, final_ir_params, final_ir_path = (
        _load_model_and_observations(args)
    )
    outer_train, outer_holdout = endpoint_trainer._split_indices(  # noqa: SLF001
        arrays,
        validation_fraction=args.outer_holdout_fraction,
        seed=args.seed,
        require_semantic_intervention=True,
    )
    if (
        args.expected_outer_holdout_count is not None
        and outer_holdout.size != args.expected_outer_holdout_count
    ):
        raise ValueError(
            "Outer holdout count does not match the expected final_teacher_2k split: "
            f"expected {args.expected_outer_holdout_count}, got {outer_holdout.size}."
        )
    if np.intersect1d(
        _episode_groups(arrays, outer_train),
        _episode_groups(arrays, outer_holdout),
    ).size:
        raise AssertionError("Outer train/holdout episode leakage detected.")
    inner_train, inner_validation = _nested_episode_split(
        arrays,
        outer_train,
        validation_fraction=args.inner_validation_fraction,
        seed=args.seed + 1,
    )
    LOGGER.info(
        "Episode-held-out split: inner_train=%s inner_validation=%s outer_train=%s outer_holdout=%s",
        inner_train.size,
        inner_validation.size,
        outer_train.size,
        outer_holdout.size,
    )

    extracted = _extract_grid(
        model,
        observation_dataset,
        arrays,
        alphas=alphas,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    target_active7 = np.asarray(arrays["clean_actions"], dtype=np.float32)[..., :7]
    risk_values = _action_risks(
        extracted["predicted_actions_active7"],
        target_active7,
        huber_delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
        gripper_sign_threshold=args.gripper_sign_threshold,
    )
    router_risk = risk_values["weighted_error"]
    oracle_index = np.argmin(router_risk, axis=-1).astype(np.int32)
    row = np.arange(oracle_index.size)
    oracle_risk = router_risk[row, oracle_index]
    baseline_risk = router_risk[:, baseline_index]
    oracle_gain = baseline_risk - oracle_risk
    oracle_relative_gain = oracle_gain / np.maximum(baseline_risk, 1e-8)

    inner_split = np.full(len(oracle_index), -1, dtype=np.int8)
    inner_split[inner_train] = 0
    inner_split[inner_validation] = 1
    outer_split = np.full(len(oracle_index), -1, dtype=np.int8)
    outer_split[outer_train] = 0
    outer_split[outer_holdout] = 1
    examples_path = output_dir / "alpha_grid_examples.npz"
    np.savez_compressed(
        examples_path,
        row_index=np.arange(len(oracle_index), dtype=np.int32),
        dataset_index=np.asarray(arrays["dataset_index"]),
        task_id=np.asarray(arrays["task_id"]),
        episode_id=np.asarray(arrays["episode_id"]),
        frame_id=np.asarray(arrays["frame_id"]),
        outer_split=outer_split,
        inner_split=inner_split,
        alphas=alphas,
        prefix_feature=extracted["prefix_feature"].astype(np.float16),
        normalized_state=extracted["normalized_state"].astype(np.float16),
        predicted_actions_active7=extracted["predicted_actions_active7"].astype(np.float16),
        teacher_actions_active7=target_active7.astype(np.float16),
        **risk_values,
        oracle_alpha_index=oracle_index,
        oracle_alpha=alphas[oracle_index],
        oracle_gain=oracle_gain.astype(np.float32),
        oracle_relative_gain=oracle_relative_gain.astype(np.float32),
    )

    selected_risk_lambda, selected_gate_lambda, inner_report = _choose_models(
        extracted["router_feature"],
        router_risk,
        oracle_index,
        inner_train,
        inner_validation,
        alphas=alphas,
        ridge_lambdas=args.ridge_lambdas,
        feature_std_floor=args.feature_std_floor,
        gate_alpha_margin=args.gate_alpha_margin,
        device=device,
    )

    outer_spectrum = _build_ridge_spectrum(
        extracted["router_feature"][outer_train],
        std_floor=args.feature_std_floor,
        device=device,
    )
    risk_model = _fit_from_spectrum(
        outer_spectrum,
        router_risk[outer_train],
        selected_risk_lambda,
    )
    gate_target = _oracle_gate_logits(
        oracle_index[outer_train],
        alphas,
        margin=args.gate_alpha_margin,
    )
    gate_model = _fit_from_spectrum(
        outer_spectrum,
        gate_target,
        selected_gate_lambda,
    )

    holdout_risk_prediction = risk_model.predict(
        extracted["router_feature"][outer_holdout]
    )
    holdout_risk_metrics = _risk_router_metrics(
        holdout_risk_prediction,
        router_risk[outer_holdout],
        oracle_index[outer_holdout],
        alphas=alphas,
        baseline_index=baseline_index,
    )
    holdout_gate_selected, holdout_gate_alpha, holdout_gate_logit = _gate_selection(
        gate_model,
        extracted["router_feature"][outer_holdout],
        alphas,
    )
    holdout_gate_metrics = _selection_metrics(
        holdout_gate_selected,
        router_risk[outer_holdout],
        oracle_index[outer_holdout],
        alphas=alphas,
        baseline_index=baseline_index,
    )

    risk_router_path = output_dir / "risk_router.npz"
    adaptive_gate_path = output_dir / "adaptive_gate_affine.npz"
    _save_router(risk_router_path, risk_model, alphas)
    _save_router(adaptive_gate_path, gate_model, alphas)
    gate_sidecar_path, combined_sidecar_path = _save_gate_sidecars(
        output_dir,
        gate_model,
        final_ir_params,
        overwrite=args.overwrite,
    )
    predictions_path = output_dir / "outer_holdout_predictions.npz"
    np.savez_compressed(
        predictions_path,
        row_index=outer_holdout.astype(np.int32),
        dataset_index=np.asarray(arrays["dataset_index"])[outer_holdout],
        actual_risk=router_risk[outer_holdout],
        oracle_alpha_index=oracle_index[outer_holdout],
        risk_router_predicted_risk=holdout_risk_prediction.astype(np.float32),
        risk_router_selected_alpha_index=np.argmin(holdout_risk_prediction, axis=-1).astype(np.int32),
        adaptive_gate_logit=holdout_gate_logit,
        adaptive_gate_alpha=holdout_gate_alpha,
        adaptive_gate_nearest_alpha_index=holdout_gate_selected.astype(np.int32),
    )

    holdout_oracle = oracle_index[outer_holdout]
    holdout_oracle_distribution = {
        str(float(alpha)): {
            "count": int(np.sum(holdout_oracle == index)),
            "fraction": float(np.mean(holdout_oracle == index)),
        }
        for index, alpha in enumerate(alphas)
    }
    summary = {
        "name": "IR fixed-time alpha grid with linear risk router",
        "status": "offline teacher-action routing probe; closed-loop validation required",
        "config": {
            **dataclasses.asdict(args),
            "dataset": list(args.dataset),
            "alphas": [float(value) for value in alphas],
            "ridge_lambdas": [float(value) for value in args.ridge_lambdas],
            "router_feature": "pooled current-observation prefix[2048] + normalized state[32]",
            "shared_compute": "one prefix and one IAR call per record; fixed teacher EAR/action noise for all alphas",
            "router_risk": (
                "mean 6D Huber(delta) plus gripper sign error weighted by gripper_weight, "
                "divided by 6+gripper_weight"
            ),
            "adaptive_gate_semantics": "alpha=.05+.05*tanh(pooled_prefix_state @ kernel + bias)",
        },
        "sources": {
            "checkpoint_dir": str(pathlib.Path(args.checkpoint_dir)),
            "final_ir_sidecar": str(final_ir_path),
        },
        "split": {
            "outer_protocol": "exact endpoint trainer episode-disjoint split, semantic-intervention eligible",
            "outer_train_count": int(outer_train.size),
            "outer_holdout_count": int(outer_holdout.size),
            "outer_train_episode_groups": int(np.unique(_episode_groups(arrays, outer_train)).size),
            "outer_holdout_episode_groups": int(np.unique(_episode_groups(arrays, outer_holdout)).size),
            "inner_protocol": "same grouped split applied only within outer train, seed+1",
            "inner_train_count": int(inner_train.size),
            "inner_validation_count": int(inner_validation.size),
            "inner_train_episode_groups": int(np.unique(_episode_groups(arrays, inner_train)).size),
            "inner_validation_episode_groups": int(np.unique(_episode_groups(arrays, inner_validation)).size),
            "episode_leakage": False,
        },
        "inner_model_selection": inner_report,
        "outer_holdout": {
            "oracle_alpha_distribution": holdout_oracle_distribution,
            "oracle_gain_vs_alpha0": {
                "mean": float(np.mean(oracle_gain[outer_holdout])),
                "median": float(np.median(oracle_gain[outer_holdout])),
                "p90": float(np.quantile(oracle_gain[outer_holdout], 0.9)),
                "mean_relative": float(np.mean(oracle_relative_gain[outer_holdout])),
            },
            "linear_risk_router": holdout_risk_metrics,
            "existing_scalar_gate_approximation": holdout_gate_metrics,
        },
        "artifacts": {
            "examples": str(examples_path),
            "risk_router": str(risk_router_path),
            "adaptive_gate_affine": str(adaptive_gate_path),
            "adaptive_gate_only_sidecar": str(gate_sidecar_path),
            "combined_final_ir_adaptive_gate_sidecar": str(combined_sidecar_path),
            "outer_holdout_predictions": str(predictions_path),
        },
        "deployment_note": (
            "risk_router.npz is the faithful five-risk argmin router. The existing one-output adaptive gate "
            "cannot represent five independent risks, so its sidecar is a separately scored linear-logit "
            "approximation to oracle alpha. Prefer the risk router if its outer regret is materially lower."
        ),
    }
    _json_dump(output_dir / "summary.json", summary)
    LOGGER.info(
        "Done: holdout=%s risk-router top1=%.4f regret=%.6g gate top1=%.4f regret=%.6g",
        outer_holdout.size,
        holdout_risk_metrics["top1_accuracy"],
        holdout_risk_metrics["mean_regret"],
        holdout_gate_metrics["top1_accuracy"],
        holdout_gate_metrics["mean_regret"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
