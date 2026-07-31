"""Probe a deployment-safe confidence gate for a saved direct corrector.

The saved corrector is evaluated twice with the same parameters:

* ``current`` receives the real branch-end observation.
* ``anchor_substitution`` receives the anchor observation in both slots.

Their output difference measures how strongly the latest observation changes
the proposed correction.  A small ridge model combines that response with
action-continuity and observation-delta features to predict whether applying
the current correction reduces H-step 6-DoF MSE to the fresh teacher.

Fresh actions and synthetic branch identities are labels/evaluation strata
only.  They are never included in the gate features.  Roots use the same
task-stratified, episode-disjoint split as the corrector trainer.  A routing
threshold is selected on validation data and frozen before test evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import branched_dataset
from openpi.models import model as model_lib
import train_branched_action_corrector as corrector_lib
import train_branched_effective_progress as progress_probe


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    corrector_summary: str
    output_dir: str
    corrector_params: str | None = None
    seed: int = 7
    split_seed: int = 7
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    eval_batch_size: int = 256
    response_alpha: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    trust_region_l2: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.40, 0.0)
    ridge_l2: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )
    validation_regression_rate_limit: float = 0.10
    validation_regression_confidence_z: float = 1.645
    minimum_validation_selected: int = 32
    focus_task: int = 8
    overwrite: bool = False


_BUDGETS = (0.05, 0.10, 0.25, 0.50, 1.0)
_EPSILON = 1e-8


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seed and split_seed must be non-negative.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if args.eval_batch_size <= 0 or args.minimum_validation_selected <= 0:
        raise ValueError("Batch size and minimum validation selection must be positive.")
    if not args.response_alpha or any(value <= 0 for value in args.response_alpha):
        raise ValueError("response_alpha must contain positive values.")
    if not args.trust_region_l2 or any(value < 0 for value in args.trust_region_l2):
        raise ValueError("trust_region_l2 must contain non-negative values; zero means no clipping.")
    if not args.ridge_l2 or any(value <= 0 for value in args.ridge_l2):
        raise ValueError("ridge_l2 must contain positive regularisation values.")
    if not 0 <= args.validation_regression_rate_limit < 1:
        raise ValueError("validation_regression_rate_limit must lie in [0, 1).")
    if args.validation_regression_confidence_z < 0:
        raise ValueError("validation_regression_confidence_z must be non-negative.")


def _corrector_args(summary: dict[str, Any]) -> corrector_lib.Args:
    saved = dict(summary["args"])
    field_names = {field.name for field in dataclasses.fields(corrector_lib.Args)}
    defaults = corrector_lib.Args(
        dataset=tuple(saved.get("dataset", ())),
        output_dir=str(saved.get("output_dir", ".")),
    )
    updates = {name: value for name, value in saved.items() if name in field_names}
    if "dataset" in updates:
        updates["dataset"] = tuple(updates["dataset"])
    return dataclasses.replace(defaults, **updates)


def _load_corrector(
    summary: dict[str, Any],
    train_flat: dict[str, np.ndarray],
    *,
    params_override: str | None,
    run_name: str,
) -> tuple[Any, nnx.State, str, corrector_lib.Args]:
    model_args = _corrector_args(summary)
    if model_args.mode != "direct":
        raise ValueError(
            f"Confidence probe requires a direct corrector checkpoint; got mode={model_args.mode!r}."
        )
    module = corrector_lib._make_model(  # noqa: SLF001
        train_flat,
        summary["range_calibration"],
        args=model_args,
        dual_bases=None,
    )
    selected_params = params_override or summary["train"][run_name]["params_path"]
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
            f"Could not identify {expected_name!r} in checkpoint keys {sorted(loaded)}."
        )
    graphdef, params = nnx.split(module)
    params.replace_by_pure_dict(loaded)

    @jax.jit
    def predict_step(current_params: nnx.State, batch: dict[str, jax.Array]):
        candidate = nnx.merge(graphdef, current_params)
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
            batch["anchor_feedback_features"],
            batch["current_feedback_features"],
        )

    return predict_step, params, str(pathlib.Path(selected_params).resolve()), model_args


def _predict(
    predict_step: Any,
    params: nnx.State,
    flat: dict[str, np.ndarray],
    *,
    batch_size: int,
    current_equals_anchor: bool,
) -> np.ndarray:
    outputs = corrector_lib._predict_all(  # noqa: SLF001
        predict_step,
        params,
        flat,
        batch_size=batch_size,
        current_equals_anchor=current_equals_anchor,
    )
    return np.asarray(outputs["actions"], dtype=np.float32)


def _sample_mse_6d(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.asarray(predicted, dtype=np.float64)[..., :6] - np.asarray(
        target,
        dtype=np.float64,
    )[..., :6]
    return np.mean(np.square(error), axis=(1, 2))


def _trust_region_response_candidate(
    base_actions: np.ndarray,
    response: np.ndarray,
    *,
    alpha: float,
    trust_region_l2: float,
) -> np.ndarray:
    """Apply a continuous response around stale while retaining its gripper."""

    base = np.asarray(base_actions, dtype=np.float64)
    delta = alpha * np.asarray(response, dtype=np.float64)[..., :6]
    if trust_region_l2 > 0:
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        delta = delta * np.minimum(1.0, trust_region_l2 / np.maximum(norm, _EPSILON))
    candidate = np.array(base, copy=True)
    candidate[..., :6] += delta
    return candidate.astype(np.float32)


def _paired_delta_mse_6d(
    predicted: np.ndarray,
    target: np.ndarray,
    root_ids: np.ndarray,
    branch_ids: np.ndarray,
) -> float:
    errors: list[np.ndarray] = []
    roots = np.asarray(root_ids)
    branches = np.asarray(branch_ids)
    for root_id in np.unique(roots):
        root_indices = np.flatnonzero(roots == root_id)
        nominal = root_indices[branches[root_indices] == 0]
        if nominal.size != 1:
            raise ValueError(f"Root {int(root_id)} must contain exactly one nominal branch.")
        nominal_index = int(nominal[0])
        disturbed = root_indices[branches[root_indices] != 0]
        predicted_delta = (
            predicted[disturbed, ..., :6] - predicted[nominal_index, ..., :6]
        )
        target_delta = target[disturbed, ..., :6] - target[nominal_index, ..., :6]
        errors.append(predicted_delta - target_delta)
    if not errors:
        raise ValueError("Paired evaluation requires at least one disturbed branch.")
    return float(np.mean(np.square(np.concatenate(errors, axis=0))))


def _candidate_action_metrics(
    predicted: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    zero_pair_mse_6d: float,
) -> dict[str, Any]:
    target = np.asarray(flat["target_actions"], dtype=np.float32)
    sample_mse = _sample_mse_6d(predicted, target)
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)
    nominal = branch_ids == 0
    disturbed = ~nominal
    paired_mse = _paired_delta_mse_6d(
        predicted,
        target,
        np.asarray(flat["root_id"]),
        branch_ids,
    )
    return {
        "overall_action_mse_6d": float(np.mean(sample_mse)),
        "nominal_action_mse_6d": float(np.mean(sample_mse[nominal])),
        "disturbed_action_mse_6d": float(np.mean(sample_mse[disturbed])),
        "paired_action_delta_mse_6d": paired_mse,
        "paired_action_gap_closure_6d": (
            float((zero_pair_mse_6d - paired_mse) / zero_pair_mse_6d)
            if zero_pair_mse_6d > 0
            else None
        ),
        "gripper_sign_accuracy": float(
            np.mean((predicted[..., 6] >= 0) == (target[..., 6] >= 0))
        ),
    }


def _scan_response_family(
    response: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    alphas: tuple[float, ...],
    trust_regions: tuple[float, ...],
) -> dict[str, Any]:
    base = np.asarray(flat["base_actions"], dtype=np.float32)
    target = np.asarray(flat["target_actions"], dtype=np.float32)
    root_ids = np.asarray(flat["root_id"])
    branch_ids = np.asarray(flat["branch_id"])
    zero_pair = _paired_delta_mse_6d(base, target, root_ids, branch_ids)
    baseline = _candidate_action_metrics(base, flat, zero_pair_mse_6d=zero_pair)
    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        for trust_region in trust_regions:
            actions = _trust_region_response_candidate(
                base,
                response,
                alpha=alpha,
                trust_region_l2=trust_region,
            )
            metrics = _candidate_action_metrics(
                actions,
                flat,
                zero_pair_mse_6d=zero_pair,
            )
            selection_score = (
                0.50
                * metrics["overall_action_mse_6d"]
                / max(baseline["overall_action_mse_6d"], _EPSILON)
                + 0.25
                * metrics["nominal_action_mse_6d"]
                / max(baseline["nominal_action_mse_6d"], _EPSILON)
                + 0.25
                * metrics["paired_action_delta_mse_6d"]
                / max(zero_pair, _EPSILON)
            )
            candidates.append(
                {
                    "alpha": float(alpha),
                    "trust_region_l2": (
                        None if trust_region == 0 else float(trust_region)
                    ),
                    "selection_score": float(selection_score),
                    "conservative_validation_gate": bool(
                        metrics["overall_action_mse_6d"]
                        < baseline["overall_action_mse_6d"]
                        and metrics["nominal_action_mse_6d"]
                        <= baseline["nominal_action_mse_6d"]
                        and metrics["paired_action_delta_mse_6d"] < zero_pair
                    ),
                    **metrics,
                }
            )
    feasible = [candidate for candidate in candidates if candidate["conservative_validation_gate"]]
    pool = feasible if feasible else candidates
    selected = min(
        pool,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["overall_action_mse_6d"],
            candidate["alpha"],
        ),
    )
    return {
        "stale_baseline": baseline,
        "selection_criterion": (
            "minimum 0.50 overall/stale + 0.25 nominal/stale + 0.25 paired/zero-pair "
            "among candidates improving overall, not worsening nominal, and improving paired; "
            "fall back to the same score if none passes"
        ),
        "conservative_candidate_count": len(feasible),
        "selected": selected,
        "candidates": candidates,
    }


def _evaluate_selected_response(
    response: np.ndarray,
    flat: dict[str, np.ndarray],
    selected: dict[str, Any],
) -> dict[str, Any]:
    trust_region = selected["trust_region_l2"]
    candidate = _trust_region_response_candidate(
        flat["base_actions"],
        response,
        alpha=float(selected["alpha"]),
        trust_region_l2=0.0 if trust_region is None else float(trust_region),
    )
    zero_pair = _paired_delta_mse_6d(
        flat["base_actions"],
        flat["target_actions"],
        flat["root_id"],
        flat["branch_id"],
    )
    return {
        "selected_alpha": float(selected["alpha"]),
        "selected_trust_region_l2": trust_region,
        "selection_source": "validation only",
        "metrics": _candidate_action_metrics(
            candidate,
            flat,
            zero_pair_mse_6d=zero_pair,
        ),
    }


def _vector_rms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    axes = tuple(range(1, array.ndim))
    return np.sqrt(np.mean(np.square(array), axis=axes))


def _token_l2(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=np.float64), axis=-1)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = np.asarray(left, dtype=np.float64).reshape((len(left), -1))
    right_flat = np.asarray(right, dtype=np.float64).reshape((len(right), -1))
    numerator = np.sum(left_flat * right_flat, axis=-1)
    denominator = np.linalg.norm(left_flat, axis=-1) * np.linalg.norm(right_flat, axis=-1)
    return numerator / np.maximum(denominator, _EPSILON)


def _trajectory_kinematics(
    intended_prefix: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    prefix = np.asarray(intended_prefix, dtype=np.float64)[..., :6]
    candidate = np.asarray(actions, dtype=np.float64)[..., :6]
    history = np.concatenate([prefix[:, -2:], candidate], axis=1)
    velocity = _token_l2(np.diff(history, axis=1))
    jerk = _token_l2(np.diff(history, n=2, axis=1))
    return {
        "delta_mean": np.mean(velocity, axis=1),
        "delta_max": np.max(velocity, axis=1),
        "jerk_mean": np.mean(jerk, axis=1),
        "jerk_max": np.max(jerk, axis=1),
    }


def _feature_matrix(
    flat: dict[str, np.ndarray],
    current_actions: np.ndarray,
    anchor_substitution_actions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base = np.asarray(flat["base_actions"], dtype=np.float64)
    current = np.asarray(current_actions, dtype=np.float64)
    prior = np.asarray(anchor_substitution_actions, dtype=np.float64)
    prefix = np.asarray(flat["intended_prefix"], dtype=np.float64)

    response = current[..., :6] - prior[..., :6]
    correction = current[..., :6] - base[..., :6]
    prior_correction = prior[..., :6] - base[..., :6]
    response_l2 = _token_l2(response)
    correction_l2 = _token_l2(correction)
    prior_correction_l2 = _token_l2(prior_correction)

    state_delta = (
        np.asarray(flat["current_state"], dtype=np.float64)
        - np.asarray(flat["anchor_state"], dtype=np.float64)
    )
    image_delta = (
        np.asarray(flat["current_images"], dtype=np.float64)
        - np.asarray(flat["anchor_images"], dtype=np.float64)
    ) / 255.0
    image_view_rms = np.sqrt(np.mean(np.square(image_delta), axis=(2, 3, 4)))

    previous_action = prefix[:, -1, :6]
    stale_boundary = np.linalg.norm(base[:, 0, :6] - previous_action, axis=-1)
    current_boundary = np.linalg.norm(current[:, 0, :6] - previous_action, axis=-1)
    prior_boundary = np.linalg.norm(prior[:, 0, :6] - previous_action, axis=-1)
    stale_kinematics = _trajectory_kinematics(prefix, base)
    current_kinematics = _trajectory_kinematics(prefix, current)
    prior_kinematics = _trajectory_kinematics(prefix, prior)

    feature_values: list[np.ndarray] = []
    feature_names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(base),):
            raise ValueError(f"Feature {name!r} has shape {array.shape}; expected {(len(base),)}.")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"Feature {name!r} contains non-finite values.")
        feature_names.append(name)
        feature_values.append(array)

    add("observation_response_rms_6d", _vector_rms(response))
    add("observation_response_token_l2_mean", np.mean(response_l2, axis=1))
    add("observation_response_token_l2_max", np.max(response_l2, axis=1))
    add("observation_response_first_l2", response_l2[:, 0])
    add("observation_response_last_l2", response_l2[:, -1])
    add("full_correction_rms_6d", _vector_rms(correction))
    add("full_correction_token_l2_mean", np.mean(correction_l2, axis=1))
    add("full_correction_token_l2_max", np.max(correction_l2, axis=1))
    add("full_correction_first_l2", correction_l2[:, 0])
    add("full_correction_last_l2", correction_l2[:, -1])
    add("prior_correction_rms_6d", _vector_rms(prior_correction))
    add("prior_correction_token_l2_mean", np.mean(prior_correction_l2, axis=1))
    add(
        "response_to_correction_ratio",
        _vector_rms(response) / np.maximum(_vector_rms(correction), _EPSILON),
    )
    add(
        "response_to_prior_correction_ratio",
        _vector_rms(response) / np.maximum(_vector_rms(prior_correction), _EPSILON),
    )
    add("response_correction_cosine", _cosine(response, correction))
    add("response_prior_correction_cosine", _cosine(response, prior_correction))
    add("state_delta_rms", _vector_rms(state_delta))
    add("state_delta_l2", np.linalg.norm(state_delta, axis=-1))
    add("state_delta_mean_abs", np.mean(np.abs(state_delta), axis=-1))
    add("state_delta_max_abs", np.max(np.abs(state_delta), axis=-1))
    add("image_delta_rms", _vector_rms(image_delta))
    add("image_delta_mean_abs", np.mean(np.abs(image_delta), axis=(1, 2, 3, 4)))
    for view in range(image_view_rms.shape[1]):
        add(f"image_delta_view{view}_rms", image_view_rms[:, view])
    add("stale_boundary_jump_l2", stale_boundary)
    add("current_boundary_jump_l2", current_boundary)
    add("prior_boundary_jump_l2", prior_boundary)
    add("current_minus_stale_boundary_jump", current_boundary - stale_boundary)
    add("current_minus_prior_boundary_jump", current_boundary - prior_boundary)
    add("stale_action_delta_l2_mean", stale_kinematics["delta_mean"])
    add("current_action_delta_l2_mean", current_kinematics["delta_mean"])
    add("prior_action_delta_l2_mean", prior_kinematics["delta_mean"])
    add("current_minus_stale_delta_l2_mean", current_kinematics["delta_mean"] - stale_kinematics["delta_mean"])
    add("stale_action_delta_l2_max", stale_kinematics["delta_max"])
    add("current_action_delta_l2_max", current_kinematics["delta_max"])
    add("current_minus_stale_delta_l2_max", current_kinematics["delta_max"] - stale_kinematics["delta_max"])
    add("stale_action_jerk_l2_mean", stale_kinematics["jerk_mean"])
    add("current_action_jerk_l2_mean", current_kinematics["jerk_mean"])
    add("prior_action_jerk_l2_mean", prior_kinematics["jerk_mean"])
    add("current_minus_stale_jerk_l2_mean", current_kinematics["jerk_mean"] - stale_kinematics["jerk_mean"])
    add("stale_action_jerk_l2_max", stale_kinematics["jerk_max"])
    add("current_action_jerk_l2_max", current_kinematics["jerk_max"])
    add("current_minus_stale_jerk_l2_max", current_kinematics["jerk_max"] - stale_kinematics["jerk_max"])
    add("base_action_rms_6d", _vector_rms(base[..., :6]))
    add("current_action_rms_6d", _vector_rms(current[..., :6]))
    add("current_continuous_saturation_fraction", np.mean(np.abs(current[..., :6]) >= 0.999, axis=(1, 2)))
    add(
        "current_vs_stale_gripper_disagreement",
        np.mean((current[..., 6] >= 0) != (base[..., 6] >= 0), axis=1),
    )
    add(
        "current_vs_prior_gripper_disagreement",
        np.mean((current[..., 6] >= 0) != (prior[..., 6] >= 0), axis=1),
    )
    add(
        "current_boundary_gripper_flip",
        (current[:, 0, 6] >= 0) != (prefix[:, -1, 6] >= 0),
    )
    add(
        "stale_boundary_gripper_flip",
        (base[:, 0, 6] >= 0) != (prefix[:, -1, 6] >= 0),
    )
    return np.stack(feature_values, axis=1), feature_names


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    mann_whitney = np.sum(ranks[labels]) - positives * (positives + 1) / 2
    return float(mann_whitney / (positives * negatives))


def _fit_ridge(
    train_features: np.ndarray,
    train_benefit: np.ndarray,
    validation_features: np.ndarray,
    validation_benefit: np.ndarray,
    *,
    ridge_values: tuple[float, ...],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = np.mean(train_features, axis=0)
    feature_scale = np.std(train_features, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    train_standardized = (train_features - feature_mean) / feature_scale
    validation_standardized = (validation_features - feature_mean) / feature_scale
    target_mean = float(np.mean(train_benefit))
    centered_target = train_benefit - target_mean
    gram = train_standardized.T @ train_standardized
    cross = train_standardized.T @ centered_target
    identity = np.eye(train_standardized.shape[1], dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    weights_by_l2: list[np.ndarray] = []
    for ridge in ridge_values:
        weights = np.linalg.solve(gram + ridge * identity, cross)
        validation_score = target_mean + validation_standardized @ weights
        auroc = _binary_auroc(validation_benefit > 0, validation_score)
        mse = float(np.mean(np.square(validation_score - validation_benefit)))
        candidates.append(
            {
                "ridge_l2": float(ridge),
                "validation_auroc": auroc,
                "validation_benefit_mse": mse,
            }
        )
        weights_by_l2.append(weights)

    def selection_key(index: int) -> tuple[float, float, float]:
        auroc = candidates[index]["validation_auroc"]
        return (
            float(auroc) if auroc is not None else -1.0,
            -float(candidates[index]["validation_benefit_mse"]),
            -float(candidates[index]["ridge_l2"]),
        )

    selected_index = max(range(len(candidates)), key=selection_key)
    weights = weights_by_l2[selected_index]
    return (
        {
            "target": "stale H-step 6-DoF MSE minus current-corrector H-step 6-DoF MSE",
            "selected_ridge_l2": candidates[selected_index]["ridge_l2"],
            "selection_criterion": "highest validation benefit-sign AUROC, then lowest benefit MSE",
            "candidate_metrics": candidates,
            "target_mean": target_mean,
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "standardized_weights": weights.tolist(),
        },
        weights,
        feature_mean,
        feature_scale,
    )


def _predict_ridge(
    features: np.ndarray,
    weights: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    return target_mean + ((features - feature_mean) / feature_scale) @ weights


def _wilson_upper(regressions: int, selected: int, z: float) -> float | None:
    if selected <= 0:
        return None
    rate = regressions / selected
    denominator = 1.0 + z * z / selected
    center = rate + z * z / (2.0 * selected)
    radius = z * np.sqrt(rate * (1.0 - rate) / selected + z * z / (4.0 * selected * selected))
    return float((center + radius) / denominator)


def _selection_metrics(
    selected: np.ndarray,
    stale_mse: np.ndarray,
    current_mse: np.ndarray,
    benefit: np.ndarray,
    branch_ids: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.bool_)
    count = int(selected.size)
    selected_count = int(np.sum(selected))
    routed = np.where(selected, current_mse, stale_mse)
    nominal = np.asarray(branch_ids) == 0
    disturbed = ~nominal
    positive_oracle_gain = float(np.sum(np.maximum(benefit, 0.0)))
    selected_gain = float(np.sum(benefit[selected])) if selected_count else 0.0
    regression_count = int(np.sum(benefit[selected] < 0)) if selected_count else 0
    stale_mean = float(np.mean(stale_mse))
    routed_mean = float(np.mean(routed))
    return {
        "count": count,
        "selected_count": selected_count,
        "coverage": float(selected_count / count),
        "stale_action_mse_6d": stale_mean,
        "routed_action_mse_6d": routed_mean,
        "absolute_action_mse_gain": float(stale_mean - routed_mean),
        "relative_risk_reduction_vs_stale": (
            float((stale_mean - routed_mean) / stale_mean) if stale_mean > 0 else None
        ),
        "selected_stale_action_mse_6d": (
            float(np.mean(stale_mse[selected])) if selected_count else None
        ),
        "selected_current_action_mse_6d": (
            float(np.mean(current_mse[selected])) if selected_count else None
        ),
        "selected_mean_benefit": (
            float(np.mean(benefit[selected])) if selected_count else None
        ),
        "selected_regression_count": regression_count,
        "selected_regression_rate": (
            float(regression_count / selected_count) if selected_count else None
        ),
        "oracle_positive_gain_captured_fraction": (
            float(selected_gain / positive_oracle_gain) if positive_oracle_gain > 0 else None
        ),
        "nominal_count": int(np.sum(nominal)),
        "nominal_retention": float(1.0 - np.mean(selected[nominal])) if np.any(nominal) else None,
        "disturbed_count": int(np.sum(disturbed)),
        "disturbed_selection_rate": float(np.mean(selected[disturbed])) if np.any(disturbed) else None,
    }


def _ranked_selection(scores: np.ndarray, coverage: float, *, seed: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    count = values.size if coverage >= 1.0 else max(1, int(np.ceil(values.size * coverage)))
    tie_break = np.random.default_rng(seed).random(values.size)
    order = np.lexsort((tie_break, -values))
    selected = np.zeros(values.size, dtype=np.bool_)
    selected[order[:count]] = True
    return selected


def _budgeted_metrics(
    scores: np.ndarray,
    stale_mse: np.ndarray,
    current_mse: np.ndarray,
    benefit: np.ndarray,
    branch_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        f"{round(100 * budget)}%": _selection_metrics(
            _ranked_selection(scores, budget, seed=seed + index),
            stale_mse,
            current_mse,
            benefit,
            branch_ids,
        )
        for index, budget in enumerate(_BUDGETS)
    }


def _select_conservative_threshold(
    scores: np.ndarray,
    benefit: np.ndarray,
    branch_ids: np.ndarray,
    stale_mse: np.ndarray,
    current_mse: np.ndarray,
    *,
    minimum_selected: int,
    regression_rate_limit: float,
    confidence_z: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(np.asarray(scores, dtype=np.float64))[::-1]:
        selected = scores >= threshold
        selected_count = int(np.sum(selected))
        if selected_count < minimum_selected:
            continue
        regression_count = int(np.sum(benefit[selected] < 0))
        regression_upper = _wilson_upper(regression_count, selected_count, confidence_z)
        selected_gain = float(np.sum(benefit[selected]))
        if (
            regression_upper is None
            or regression_upper > regression_rate_limit
            or selected_gain <= 0
        ):
            continue
        metrics = _selection_metrics(
            selected,
            stale_mse,
            current_mse,
            benefit,
            branch_ids,
        )
        candidate = {
            "status": "feasible",
            "threshold": float(threshold),
            "validation_regression_rate_upper_bound": regression_upper,
            "validation_regression_confidence_z": confidence_z,
            **metrics,
        }
        if best is None or (
            candidate["selected_count"],
            candidate["absolute_action_mse_gain"],
            candidate["threshold"],
        ) > (
            best["selected_count"],
            best["absolute_action_mse_gain"],
            best["threshold"],
        ):
            best = candidate
    if best is not None:
        return best
    return {
        "status": "no_nonempty_threshold_satisfies_validation_constraint",
        "threshold": None,
        "minimum_validation_selected": minimum_selected,
        "validation_regression_rate_limit": regression_rate_limit,
        "validation_regression_confidence_z": confidence_z,
    }


def _partition_report(
    scores: np.ndarray,
    stale_mse: np.ndarray,
    current_mse: np.ndarray,
    benefit: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    seed: int,
    threshold: float | None,
) -> dict[str, Any]:
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)
    report = {
        "count": int(len(scores)),
        "improvement_fraction": float(np.mean(benefit > 0)),
        "regression_fraction": float(np.mean(benefit < 0)),
        "benefit_mean": float(np.mean(benefit)),
        "benefit_median": float(np.median(benefit)),
        "stale_action_mse_6d": float(np.mean(stale_mse)),
        "current_action_mse_6d": float(np.mean(current_mse)),
        "benefit_sign_auroc": _binary_auroc(benefit > 0, scores),
        "budgeted": _budgeted_metrics(
            scores,
            stale_mse,
            current_mse,
            benefit,
            branch_ids,
            seed=seed,
        ),
    }
    if threshold is None:
        report["validation_selected_threshold"] = {
            "status": "disabled_because_validation_found_no_feasible_threshold",
            "selected_count": 0,
            "coverage": 0.0,
        }
    else:
        selected = scores >= threshold
        threshold_metrics = _selection_metrics(
            selected,
            stale_mse,
            current_mse,
            benefit,
            branch_ids,
        )
        threshold_metrics["threshold"] = float(threshold)
        threshold_metrics["selected_regression_rate_upper_bound"] = _wilson_upper(
            int(threshold_metrics["selected_regression_count"]),
            int(threshold_metrics["selected_count"]),
            1.645,
        )
        report["validation_selected_threshold"] = threshold_metrics
    return report


def _masked_partition_report(
    mask: np.ndarray,
    scores: np.ndarray,
    stale_mse: np.ndarray,
    current_mse: np.ndarray,
    benefit: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    seed: int,
    threshold: float | None,
) -> dict[str, Any] | None:
    selected = np.asarray(mask, dtype=np.bool_)
    if not np.any(selected):
        return None
    sliced_flat = {"branch_id": np.asarray(flat["branch_id"])[selected]}
    return _partition_report(
        scores[selected],
        stale_mse[selected],
        current_mse[selected],
        benefit[selected],
        sliced_flat,
        seed=seed,
        threshold=threshold,
    )


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")

    corrector_summary_path = pathlib.Path(args.corrector_summary)
    corrector_summary = json.loads(corrector_summary_path.read_text(encoding="utf-8"))
    saved_args = _corrector_args(corrector_summary)
    if saved_args.mode != "direct":
        raise ValueError("Only direct corrector checkpoints are supported.")

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
    train_roots, validation_roots, test_roots = progress_probe._split_roots(  # noqa: SLF001
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
        indices = progress_probe._flatten_valid_branches(arrays, roots)  # noqa: SLF001
        partitions[name] = (
            roots,
            corrector_lib._flat_arrays(  # noqa: SLF001
                arrays,
                indices,
                age=saved_args.age,
                rollout_horizon=saved_args.rollout_horizon,
                coarse_time_stride=saved_args.coarse_time_stride,
            ),
        )

    predict_step, params, params_path, loaded_args = _load_corrector(
        corrector_summary,
        partitions["train"][1],
        params_override=args.corrector_params,
        run_name="current",
    )
    no_current_train = corrector_summary.get("train", {}).get("no_current")
    if not no_current_train or not no_current_train.get("params_path"):
        raise ValueError(
            "Matched-difference response requires summary train.no_current.params_path."
        )
    (
        no_current_predict_step,
        no_current_params,
        no_current_params_path,
        _,
    ) = _load_corrector(
        corrector_summary,
        partitions["train"][1],
        params_override=str(no_current_train["params_path"]),
        run_name="no_current",
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    features: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    stale_mse: dict[str, np.ndarray] = {}
    current_mse: dict[str, np.ndarray] = {}
    benefits: dict[str, np.ndarray] = {}
    for name, (_, flat) in partitions.items():
        current_actions = _predict(
            predict_step,
            params,
            flat,
            batch_size=args.eval_batch_size,
            current_equals_anchor=False,
        )
        anchor_substitution_actions = _predict(
            predict_step,
            params,
            flat,
            batch_size=args.eval_batch_size,
            current_equals_anchor=True,
        )
        independent_no_current_actions = _predict(
            no_current_predict_step,
            no_current_params,
            flat,
            batch_size=args.eval_batch_size,
            current_equals_anchor=True,
        )
        partition_features, names = _feature_matrix(
            flat,
            current_actions,
            anchor_substitution_actions,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature ordering changed across partitions.")
        features[name] = partition_features
        predictions[name] = {
            "current": current_actions,
            "anchor_substitution": anchor_substitution_actions,
            "independent_no_current": independent_no_current_actions,
        }
        stale_mse[name] = _sample_mse_6d(flat["base_actions"], flat["target_actions"])
        current_mse[name] = _sample_mse_6d(current_actions, flat["target_actions"])
        benefits[name] = stale_mse[name] - current_mse[name]
    assert feature_names is not None

    ridge_summary, weights, feature_mean, feature_scale = _fit_ridge(
        features["train"],
        benefits["train"],
        features["validation"],
        benefits["validation"],
        ridge_values=args.ridge_l2,
    )
    scores = {
        name: _predict_ridge(
            values,
            weights,
            feature_mean,
            feature_scale,
            float(ridge_summary["target_mean"]),
        )
        for name, values in features.items()
    }
    validation_flat = partitions["validation"][1]
    conservative = _select_conservative_threshold(
        scores["validation"],
        benefits["validation"],
        np.asarray(validation_flat["branch_id"]),
        stale_mse["validation"],
        current_mse["validation"],
        minimum_selected=args.minimum_validation_selected,
        regression_rate_limit=args.validation_regression_rate_limit,
        confidence_z=args.validation_regression_confidence_z,
    )
    selected_threshold = (
        float(conservative["threshold"])
        if conservative.get("threshold") is not None
        else None
    )

    validation_same_model_response = (
        predictions["validation"]["current"]
        - predictions["validation"]["anchor_substitution"]
    )
    validation_matched_difference = (
        predictions["validation"]["current"]
        - predictions["validation"]["independent_no_current"]
    )
    same_model_scan = _scan_response_family(
        validation_same_model_response,
        partitions["validation"][1],
        alphas=args.response_alpha,
        trust_regions=args.trust_region_l2,
    )
    matched_difference_scan = _scan_response_family(
        validation_matched_difference,
        partitions["validation"][1],
        alphas=args.response_alpha,
        trust_regions=args.trust_region_l2,
    )
    test_same_model_response = (
        predictions["test"]["current"]
        - predictions["test"]["anchor_substitution"]
    )
    test_matched_difference = (
        predictions["test"]["current"]
        - predictions["test"]["independent_no_current"]
    )
    response_test = {
        "same_model_counterfactual_response": _evaluate_selected_response(
            test_same_model_response,
            partitions["test"][1],
            same_model_scan["selected"],
        ),
        "matched_difference": _evaluate_selected_response(
            test_matched_difference,
            partitions["test"][1],
            matched_difference_scan["selected"],
        ),
    }

    test_flat = partitions["test"][1]
    task_ids = np.asarray(test_flat["task_id"], dtype=np.int64)
    branch_ids = np.asarray(test_flat["branch_id"], dtype=np.int64)
    test_report = {
        "overall": _partition_report(
            scores["test"],
            stale_mse["test"],
            current_mse["test"],
            benefits["test"],
            test_flat,
            seed=args.seed,
            threshold=selected_threshold,
        ),
        "focus_task": {
            "task_id": args.focus_task,
            "metrics": _masked_partition_report(
                task_ids == args.focus_task,
                scores["test"],
                stale_mse["test"],
                current_mse["test"],
                benefits["test"],
                test_flat,
                seed=args.seed + 1_000,
                threshold=selected_threshold,
            ),
        },
        "nominal": _masked_partition_report(
            branch_ids == 0,
            scores["test"],
            stale_mse["test"],
            current_mse["test"],
            benefits["test"],
            test_flat,
            seed=args.seed + 2_000,
            threshold=selected_threshold,
        ),
        "disturbed": _masked_partition_report(
            branch_ids != 0,
            scores["test"],
            stale_mse["test"],
            current_mse["test"],
            benefits["test"],
            test_flat,
            seed=args.seed + 3_000,
            threshold=selected_threshold,
        ),
        "by_task": {
            str(int(task_id)): _masked_partition_report(
                task_ids == task_id,
                scores["test"],
                stale_mse["test"],
                current_mse["test"],
                benefits["test"],
                test_flat,
                seed=args.seed + 4_000 + int(task_id),
                threshold=selected_threshold,
            )
            for task_id in np.unique(task_ids)
        },
    }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": {
            "name": "direct_corrector_deployment_safe_confidence_probe",
            "candidate": "saved direct current corrector",
            "gate": "ridge prediction of per-sample H-step action-MSE benefit",
            "observation_response": (
                "same checkpoint current-observation output minus the same checkpoint "
                "with current observation replaced by anchor"
            ),
            "label": (
                "stale cached H-step 6-DoF action MSE minus current-corrector H-step "
                "6-DoF action MSE to the fresh same-seed teacher"
            ),
            "deployment_boundary": (
                "features use cached plan/prefix, anchor/current image and state, and the two "
                "same-checkpoint outputs only; fresh actions and branch id are labels/strata only"
            ),
            "threshold_protocol": (
                "fit on train, choose ridge regularisation and conservative routing threshold "
                "on validation, evaluate the frozen model and threshold once on test"
            ),
        },
        "args": dataclasses.asdict(args),
        "corrector": {
            "summary_path": str(corrector_summary_path.resolve()),
            "params_path": params_path,
            "no_current_params_path": no_current_params_path,
            "saved_mode": loaded_args.mode,
            "saved_direct_head": loaded_args.direct_head,
            "saved_age": loaded_args.age,
            "saved_rollout_horizon": loaded_args.rollout_horizon,
        },
        "split": {
            name: {
                "root_count": int(len(roots)),
                "valid_branch_count": int(len(flat["branch_id"])),
                "episodes_by_task": progress_probe._episode_summary(arrays, roots),  # noqa: SLF001
            }
            for name, (roots, flat) in partitions.items()
        },
        "features": {
            "count": len(feature_names),
            "names": feature_names,
            "forbidden": ["task_id", "episode_id", "root_id", "branch_id", "fresh_actions"],
        },
        "ridge": ridge_summary,
        "validation": {
            "benefit_sign_auroc": _binary_auroc(
                benefits["validation"] > 0,
                scores["validation"],
            ),
            "conservative_threshold_search": conservative,
            "response_candidate_scan": {
                "same_model_counterfactual_response": same_model_scan,
                "matched_difference": matched_difference_scan,
            },
        },
        "test": {
            **test_report,
            "response_candidates": response_test,
        },
        "decision_guard": {
            "offline_only": True,
            "meaning": (
                "A positive result shows that deployable signals can rank teacher-action benefit. "
                "It does not establish LIBERO task-success improvement; the frozen gate must next "
                "be tested in paired Task8 closed-loop trials."
            ),
            "recommended_closed_loop_trials_per_arm": 20,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
