"""Fit and audit a confidence router on top of a frozen Action-CoT editor.

This is a deliberately narrow go/no-go experiment. The frozen editor produces
both an unmodified phase-transport base and a raw revised proposal. A separate
linear router predicts whether the complete proposal has lower fresh-target
risk than the base. The router never multiplies editor residuals and cannot
change editor parameters.

The existing task-stratified, episode-disjoint train/validation/test split is
recreated from the editor run. Router weights are fit on train episodes,
validation selects one hard-routing threshold, and test is evaluated once.
These synthetic local time-warp windows can establish offline selective-update
signal; they cannot validate full-ACoT refresh or disturbed closed-loop states.
"""

from __future__ import annotations

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
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import model as model_lib
from openpi.models import transported_action_cot


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    editor_params: str
    editor_summary: str
    output_dir: str
    seed: int = 7
    split_seed: int = 7
    train_steps: int = 400
    batch_size: int = 512
    eval_batch_size: int = 512
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    improvement_margin: float = 0.01
    action_risk_weight: float = 1.0
    transport_risk_weight: float = 1.0
    huber_delta: float = 0.1
    gripper_weight: float = 4.0
    minimum_accept_coverage: float = 0.10
    maximum_accept_coverage: float = 0.90
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    log_interval: int = 50
    profile_warmup: int = 20
    profile_iterations: int = 200
    refresh_interval: int = 4
    full_acot_reference_ms: float = 95.844
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class PairIndices:
    windows: np.ndarray
    elapsed_age: np.ndarray
    physical_progress: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.windows.ndim != 1
            or self.elapsed_age.shape != self.windows.shape
            or self.physical_progress.shape != self.windows.shape
        ):
            raise ValueError("Pair arrays must be matching rank-one arrays.")

    def __len__(self) -> int:
        return int(self.windows.size)

    def take(self, selection: slice) -> PairIndices:
        return PairIndices(
            windows=self.windows[selection],
            elapsed_age=self.elapsed_age[selection],
            physical_progress=self.physical_progress[selection],
        )


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.train_steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Training steps and batch sizes must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if not 0.0 <= args.improvement_margin < 1.0:
        raise ValueError("improvement_margin must be in [0, 1).")
    if args.action_risk_weight < 0 or args.transport_risk_weight < 0:
        raise ValueError("Risk weights must be non-negative.")
    if args.action_risk_weight + args.transport_risk_weight <= 0:
        raise ValueError("At least one risk weight must be positive.")
    if args.huber_delta <= 0 or args.gripper_weight <= 0:
        raise ValueError("Huber delta and gripper weight must be positive.")
    if not 0.0 <= args.minimum_accept_coverage < args.maximum_accept_coverage <= 1.0:
        raise ValueError("Accept coverage bounds must satisfy 0 <= minimum < maximum <= 1.")
    if args.profile_warmup < 0 or args.profile_iterations <= 0:
        raise ValueError("Profiling warmup must be non-negative and iterations positive.")
    if args.refresh_interval <= 1 or args.full_acot_reference_ms <= 0:
        raise ValueError("refresh_interval must exceed one and full_acot_reference_ms must be positive.")


def _split_windows(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recreate the editor's task-stratified, episode-disjoint split."""

    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for task_id in np.unique(tasks):
        task_indices = np.flatnonzero(tasks == task_id)
        task_episodes = np.unique(episodes[task_indices])
        if task_episodes.size < 3:
            raise ValueError(f"Task {task_id} needs at least three episodes.")
        rng.shuffle(task_episodes)
        test_count = max(1, round(task_episodes.size * test_fraction))
        validation_count = max(1, round(task_episodes.size * validation_fraction))
        if test_count + validation_count >= task_episodes.size:
            validation_count = 1
            test_count = 1
        test_episodes = task_episodes[:test_count]
        validation_episodes = task_episodes[test_count : test_count + validation_count]
        test.append(task_indices[np.isin(episodes[task_indices], test_episodes)])
        validation.append(task_indices[np.isin(episodes[task_indices], validation_episodes)])
        train.append(
            task_indices[
                ~np.isin(
                    episodes[task_indices],
                    np.concatenate([test_episodes, validation_episodes]),
                )
            ]
        )
    outputs = tuple(np.sort(np.concatenate(parts)) for parts in (train, validation, test))
    if any(not values.size for values in outputs):
        raise ValueError("Episode-level split produced an empty partition.")
    return outputs  # type: ignore[return-value]


def _all_time_warp_pairs(window_indices: np.ndarray) -> PairIndices:
    triples: list[tuple[int, int, int]] = []
    for window in np.asarray(window_indices, dtype=np.int64):
        for elapsed_age in range(1, 4):
            lower = max(0, elapsed_age - 1)
            upper = min(3, elapsed_age + 1)
            triples.extend((int(window), elapsed_age, progress) for progress in range(lower, upper + 1))
    values = np.asarray(triples, dtype=np.int64)
    return PairIndices(
        windows=values[:, 0],
        elapsed_age=values[:, 1],
        physical_progress=values[:, 2],
    )


def _batch(arrays: dict[str, np.ndarray], pairs: PairIndices) -> dict[str, jax.Array]:
    windows = pairs.windows
    progress = pairs.physical_progress
    return {
        "anchor_images": jnp.asarray(arrays["images"][windows, 0].astype(np.float32) / 255.0),
        "current_images": jnp.asarray(arrays["images"][windows, progress].astype(np.float32) / 255.0),
        "anchor_state": jnp.asarray(arrays["states"][windows, 0], dtype=jnp.float32),
        "current_state": jnp.asarray(arrays["states"][windows, progress], dtype=jnp.float32),
        "cached_ear": jnp.asarray(arrays["fresh_ear"][windows, 0], dtype=jnp.float32),
        "cached_iar": jnp.asarray(arrays["fresh_iar"][windows, 0], dtype=jnp.float32),
        "cache_age": jnp.asarray(pairs.elapsed_age, dtype=jnp.int32),
    }


def _config_from_summary(path: pathlib.Path) -> transported_action_cot.TransportedActionCoTConfig:
    summary = json.loads(path.read_text(encoding="utf-8"))
    values = dict(summary["config"])
    for name in ("cnn_channels", "cnn_kernel_sizes", "geometry_scale"):
        values[name] = tuple(values[name])
    return transported_action_cot.TransportedActionCoTConfig(**values)


def _restore_editor(
    config: transported_action_cot.TransportedActionCoTConfig,
    params_path: str,
    *,
    seed: int,
) -> transported_action_cot.TransportedActionCoTExecutor:
    editor = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(seed))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(params_path, dtype=jnp.float32))
    if "transported_action_cot" in loaded:
        loaded = loaded["transported_action_cot"]
    graphdef, state = nnx.split(editor)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _predict_editor(
    editor: transported_action_cot.TransportedActionCoTExecutor,
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    graphdef, params = nnx.split(editor)

    @jax.jit
    def infer(current_params: nnx.State, batch: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
        current_editor = nnx.merge(graphdef, current_params)
        output = current_editor.forward_with_details(**batch)
        return (
            output.action,
            output.base_action,
            output.revised_ear,
            output.transported_ear,
            output.update_context,
        )

    pieces: list[list[np.ndarray]] = [[] for _ in range(5)]
    for start in range(0, len(pairs), batch_size):
        selected = pairs.take(slice(start, start + batch_size))
        outputs = infer(params, _batch(arrays, selected))
        for destination, value in zip(pieces, outputs, strict=True):
            destination.append(np.asarray(value))
    names = ("action", "base_action", "ear", "base_ear", "context")
    predictions = {name: np.concatenate(values, axis=0) for name, values in zip(names, pieces, strict=True)}
    non_finite = {name: int(np.sum(~np.isfinite(value))) for name, value in predictions.items()}
    if any(non_finite.values()):
        raise FloatingPointError(f"Editor produced non-finite predictions: {non_finite}.")
    return predictions


def _numpy_huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    return 0.5 * np.square(quadratic) + delta * (absolute - quadratic)


def _per_example_weighted_huber_7d(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    delta: float,
    gripper_weight: float,
) -> np.ndarray:
    error = np.asarray(predicted, dtype=np.float32)[..., :7] - np.asarray(target, dtype=np.float32)[..., :7]
    values = _numpy_huber(error, delta)
    weights = np.ones((7,), dtype=np.float32)
    weights[6] = gripper_weight
    weighted = values * weights
    reduction_axes = tuple(range(1, weighted.ndim))
    denominator = np.prod(weighted.shape[1:-1], dtype=np.int64) * np.sum(weights)
    return np.sum(weighted, axis=reduction_axes) / denominator


def _router_examples(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
    *,
    args: Args,
) -> dict[str, np.ndarray]:
    target_action = arrays["teacher_actions"][pairs.windows, pairs.physical_progress].astype(np.float32)
    target_ear = arrays["fresh_ear"][pairs.windows, pairs.physical_progress].astype(np.float32)
    base_action_risk = _per_example_weighted_huber_7d(
        predictions["base_action"],
        target_action,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    update_action_risk = _per_example_weighted_huber_7d(
        predictions["action"],
        target_action,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    base_transport_risk = _per_example_weighted_huber_7d(
        predictions["base_ear"],
        target_ear,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    update_transport_risk = _per_example_weighted_huber_7d(
        predictions["ear"],
        target_ear,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    base_risk = args.action_risk_weight * base_action_risk + args.transport_risk_weight * base_transport_risk
    update_risk = (
        args.action_risk_weight * update_action_risk + args.transport_risk_weight * update_transport_risk
    )
    relative_gain = (base_risk - update_risk) / np.maximum(base_risk + update_risk, 1e-8)
    return {
        **predictions,
        "target_action": target_action,
        "target_ear": target_ear,
        "base_risk": base_risk,
        "update_risk": update_risk,
        "relative_gain": relative_gain,
        "decisive": np.abs(relative_gain) > args.improvement_margin,
        "label": relative_gain > args.improvement_margin,
    }


def _fit_router(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    args: Args,
) -> tuple[transported_action_cot.ActionCoTUpdateConfidence, dict[str, Any]]:
    train_mask = np.asarray(train["decisive"], dtype=np.bool_)
    validation_mask = np.asarray(validation["decisive"], dtype=np.bool_)
    train_labels = np.asarray(train["label"][train_mask], dtype=np.float32)
    validation_labels = np.asarray(validation["label"][validation_mask], dtype=np.float32)
    if train_labels.size == 0 or validation_labels.size == 0:
        raise ValueError("Router requires decisive train and validation examples.")
    if np.unique(train_labels).size < 2 or np.unique(validation_labels).size < 2:
        raise ValueError("Router requires both accept and reject labels in train and validation.")
    train_features = jnp.asarray(train["context"][train_mask], dtype=jnp.float32)
    validation_features = jnp.asarray(validation["context"][validation_mask], dtype=jnp.float32)
    train_targets = jnp.asarray(train_labels, dtype=jnp.float32)
    validation_targets = jnp.asarray(validation_labels, dtype=jnp.float32)
    positive_fraction = float(np.mean(train_labels))
    positive_weight = 0.5 / positive_fraction
    negative_weight = 0.5 / (1.0 - positive_fraction)

    router = transported_action_cot.ActionCoTUpdateConfidence(
        int(train_features.shape[-1]),
        rngs=nnx.Rngs(args.seed + 10_000),
    )
    graphdef, params = nnx.split(router)
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: nnx.State,
        current_optimizer_state: optax.OptState,
        features: jax.Array,
        targets: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, jax.Array]:
        def loss_fn(candidate_params: nnx.State) -> jax.Array:
            candidate = nnx.merge(graphdef, candidate_params)
            logits = candidate(features)
            weights = jnp.where(targets > 0.5, positive_weight, negative_weight)
            return jnp.mean(weights * optax.sigmoid_binary_cross_entropy(logits, targets))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = optimizer.update(gradients, current_optimizer_state, current_params)
        return optax.apply_updates(current_params, updates), next_optimizer_state, loss

    @jax.jit
    def validation_loss(current_params: nnx.State) -> jax.Array:
        candidate = nnx.merge(graphdef, current_params)
        logits = candidate(validation_features)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, validation_targets))

    rng = np.random.default_rng(args.seed + 20_000)
    best_validation_loss = float("inf")
    best_step = 0
    best_params = params
    records: list[dict[str, Any]] = []
    for step in range(1, args.train_steps + 1):
        selected = rng.choice(
            train_features.shape[0],
            size=min(args.batch_size, train_features.shape[0]),
            replace=train_features.shape[0] < args.batch_size,
        )
        params, optimizer_state, loss = train_step(
            params,
            optimizer_state,
            train_features[selected],
            train_targets[selected],
        )
        if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
            current_validation_loss = float(validation_loss(params))
            record = {
                "step": step,
                "train_loss": float(loss),
                "validation_bce": current_validation_loss,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_step = step
                best_params = jax.tree.map(lambda value: jnp.array(value), params)
    return nnx.merge(graphdef, best_params), {
        "requested_steps": args.train_steps,
        "best_step": best_step,
        "best_validation_bce": best_validation_loss,
        "train_decisive_count": int(train_labels.size),
        "train_accept_count": int(np.sum(train_labels)),
        "train_reject_count": int(train_labels.size - np.sum(train_labels)),
        "validation_decisive_count": int(validation_labels.size),
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "records": records,
    }


def _router_probabilities(
    router: transported_action_cot.ActionCoTUpdateConfidence,
    context: np.ndarray,
) -> np.ndarray:
    graphdef, params = nnx.split(router)

    @jax.jit
    def predict(current_params: nnx.State, features: jax.Array) -> jax.Array:
        current_router = nnx.merge(graphdef, current_params)
        return jax.nn.sigmoid(current_router(features))

    return np.asarray(predict(params, jnp.asarray(context, dtype=jnp.float32)))


def _select_threshold(
    probabilities: np.ndarray,
    base_risk: np.ndarray,
    update_risk: np.ndarray,
    *,
    minimum_coverage: float,
    maximum_coverage: float,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([-np.inf, np.inf]),
                np.quantile(probabilities, np.linspace(0.0, 1.0, 201)),
            ]
        )
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        accept = probabilities >= threshold
        coverage = float(np.mean(accept))
        if not minimum_coverage <= coverage <= maximum_coverage:
            continue
        risk = float(np.mean(np.where(accept, update_risk, base_risk)))
        candidate = (risk, -coverage, float(threshold))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No validation threshold satisfies the requested coverage interval.")
    return {
        "threshold": best[2],
        "validation_risk": best[0],
        "validation_coverage": -best[1],
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels))
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        raise ValueError("ROC AUC requires both classes.")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if np.any(selected):
            value += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected])) - float(np.mean(labels[selected]))
            )
    return value


def _mse_7d(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(predicted[..., :7] - target[..., :7])))


def _evaluate_router(
    examples: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
    *,
    args: Args,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    accept = probabilities >= threshold
    base_risk = examples["base_risk"]
    update_risk = examples["update_risk"]
    routed_risk = np.where(accept, update_risk, base_risk)
    oracle_accept = update_risk < base_risk
    oracle_risk = np.minimum(update_risk, base_risk)
    coverage = float(np.mean(accept))
    routed_action = np.where(accept[:, None], examples["action"], examples["base_action"])
    routed_ear = np.where(accept[:, None, None], examples["ear"], examples["base_ear"])
    oracle_action = np.where(oracle_accept[:, None], examples["action"], examples["base_action"])
    oracle_ear = np.where(oracle_accept[:, None, None], examples["ear"], examples["base_ear"])
    target_action = examples["target_action"]
    target_ear = examples["target_ear"]
    decisive = np.asarray(examples["decisive"], dtype=np.bool_)
    labels = np.asarray(examples["label"][decisive], dtype=np.bool_)
    decisive_probabilities = probabilities[decisive]
    regression = update_risk > base_risk
    improvement = update_risk < base_risk
    best_raw_risk = min(float(np.mean(base_risk)), float(np.mean(update_risk)))
    routed_mean_risk = float(np.mean(routed_risk))
    oracle_mean_risk = float(np.mean(oracle_risk))
    oracle_gap = best_raw_risk - oracle_mean_risk
    return {
        "count": int(probabilities.size),
        "decisive_count": int(np.sum(decisive)),
        "accept_coverage": coverage,
        "threshold": threshold,
        "auroc": _roc_auc(labels, decisive_probabilities),
        "brier": float(np.mean(np.square(decisive_probabilities - labels))),
        "ece_10bin": _expected_calibration_error(labels, decisive_probabilities),
        "base_mean_risk": float(np.mean(base_risk)),
        "update_mean_risk": float(np.mean(update_risk)),
        "learned_route_mean_risk": routed_mean_risk,
        "random_same_coverage_mean_risk": float(
            coverage * np.mean(update_risk) + (1.0 - coverage) * np.mean(base_risk)
        ),
        "oracle_route_mean_risk": oracle_mean_risk,
        "fraction_of_oracle_gap_closed": (
            (best_raw_risk - routed_mean_risk) / oracle_gap if oracle_gap > 0 else 0.0
        ),
        "accepted_regression_rate": float(np.mean(regression[accept])) if np.any(accept) else 0.0,
        "rejected_improvement_rate": float(np.mean(improvement[~accept])) if np.any(~accept) else 0.0,
        "base_action_mse_7d": _mse_7d(examples["base_action"], target_action),
        "update_action_mse_7d": _mse_7d(examples["action"], target_action),
        "learned_route_action_mse_7d": _mse_7d(routed_action, target_action),
        "oracle_route_action_mse_7d": _mse_7d(oracle_action, target_action),
        "base_transport_mse_7d": _mse_7d(examples["base_ear"], target_ear),
        "update_transport_mse_7d": _mse_7d(examples["ear"], target_ear),
        "learned_route_transport_mse_7d": _mse_7d(routed_ear, target_ear),
        "oracle_route_transport_mse_7d": _mse_7d(oracle_ear, target_ear),
    }


def _profile(
    editor: transported_action_cot.TransportedActionCoTExecutor,
    router: transported_action_cot.ActionCoTUpdateConfidence,
    batch: dict[str, jax.Array],
    *,
    args: Args,
) -> dict[str, Any]:
    editor_graphdef, editor_params = nnx.split(editor)
    router_graphdef, router_params = nnx.split(router)
    profile_batch = {name: value[:1] for name, value in batch.items()}

    @jax.jit
    def editor_only(current_editor_params: nnx.State, inputs: dict[str, jax.Array]) -> jax.Array:
        current_editor = nnx.merge(editor_graphdef, current_editor_params)
        return current_editor.forward_with_details(**inputs).action

    @jax.jit
    def editor_and_router(
        current_editor_params: nnx.State,
        current_router_params: nnx.State,
        inputs: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        current_editor = nnx.merge(editor_graphdef, current_editor_params)
        current_router = nnx.merge(router_graphdef, current_router_params)
        output = current_editor.forward_with_details(**inputs)
        return output.action, current_router(output.update_context)

    for _ in range(args.profile_warmup):
        jax.block_until_ready(editor_only(editor_params, profile_batch))
        jax.block_until_ready(editor_and_router(editor_params, router_params, profile_batch))
    editor_ms = np.empty((args.profile_iterations,), dtype=np.float64)
    combined_ms = np.empty_like(editor_ms)
    for index in range(args.profile_iterations):
        started = time.perf_counter()
        jax.block_until_ready(editor_only(editor_params, profile_batch))
        editor_ms[index] = (time.perf_counter() - started) * 1_000.0
        started = time.perf_counter()
        jax.block_until_ready(editor_and_router(editor_params, router_params, profile_batch))
        combined_ms[index] = (time.perf_counter() - started) * 1_000.0
    combined_mean = float(np.mean(combined_ms))
    amortized = (
        args.full_acot_reference_ms + (args.refresh_interval - 1) * combined_mean
    ) / args.refresh_interval
    return {
        "device": str(jax.devices()[0]),
        "iterations": args.profile_iterations,
        "editor_mean_ms": float(np.mean(editor_ms)),
        "editor_p95_ms": float(np.percentile(editor_ms, 95)),
        "editor_plus_router_mean_ms": combined_mean,
        "editor_plus_router_p95_ms": float(np.percentile(combined_ms, 95)),
        "router_incremental_mean_ms": combined_mean - float(np.mean(editor_ms)),
        "fixed_refresh_interval": args.refresh_interval,
        "theoretical_amortized_ms": amortized,
        "theoretical_speedup_vs_full_acot": args.full_acot_reference_ms / amortized,
        "note": "Reject keeps the transported base and does not trigger an extra full-ACoT call in this pilot.",
    }


def _save_router(router: transported_action_cot.ActionCoTUpdateConfidence, target: pathlib.Path) -> None:
    _, params = nnx.split(router)
    item = {"params": {"action_cot_update_confidence": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=True)


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")
    config = _config_from_summary(pathlib.Path(args.editor_summary))
    if config.correction_mode not in {"event", "plan"}:
        raise ValueError("Confidence routing pilot requires a structured event or plan editor.")
    arrays = multirate_dataset.load_multirate_arrays(args.dataset)
    train_windows, validation_windows, test_windows = _split_windows(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    editor = _restore_editor(config, args.editor_params, seed=args.seed)
    partitions: dict[str, dict[str, np.ndarray]] = {}
    pair_counts: dict[str, int] = {}
    for name, windows in (
        ("train", train_windows),
        ("validation", validation_windows),
        ("test", test_windows),
    ):
        pairs = _all_time_warp_pairs(windows)
        predictions = _predict_editor(editor, arrays, pairs, batch_size=args.eval_batch_size)
        partitions[name] = _router_examples(predictions, arrays, pairs, args=args)
        pair_counts[name] = len(pairs)
    router, train_summary = _fit_router(partitions["train"], partitions["validation"], args=args)
    probabilities = {
        name: _router_probabilities(router, examples["context"]) for name, examples in partitions.items()
    }
    threshold_selection = _select_threshold(
        probabilities["validation"],
        partitions["validation"]["base_risk"],
        partitions["validation"]["update_risk"],
        minimum_coverage=args.minimum_accept_coverage,
        maximum_coverage=args.maximum_accept_coverage,
    )
    threshold = threshold_selection["threshold"]
    evaluations = {
        name: _evaluate_router(examples, probabilities[name], threshold, args=args)
        for name, examples in partitions.items()
    }
    profile_pairs = _all_time_warp_pairs(test_windows[:1]).take(slice(0, 1))
    profile = _profile(editor, router, _batch(arrays, profile_pairs), args=args)
    router_path = output_dir / "router_params"
    _save_router(router, router_path)
    test_metrics = evaluations["test"]
    router_gate_pass = (
        args.minimum_accept_coverage
        <= test_metrics["accept_coverage"]
        <= args.maximum_accept_coverage
        and test_metrics["auroc"] > 0.55
        and test_metrics["learned_route_mean_risk"]
        < min(test_metrics["base_mean_risk"], test_metrics["update_mean_risk"])
        and test_metrics["learned_route_mean_risk"] < test_metrics["random_same_coverage_mean_risk"]
        and test_metrics["learned_route_action_mse_7d"]
        < min(test_metrics["base_action_mse_7d"], test_metrics["update_action_mse_7d"])
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": "frozen_action_cot_editor_plus_plan_update_confidence",
        "editor": {
            "params": str(pathlib.Path(args.editor_params).resolve()),
            "summary": str(pathlib.Path(args.editor_summary).resolve()),
            "correction_mode": config.correction_mode,
        },
        "router_params": str(router_path.resolve()),
        "router_parameter_count": transported_action_cot.estimate_update_confidence_parameter_count(
            config.hidden_dim
        ),
        "split": "same task-stratified, episode-disjoint split as editor",
        "pair_counts": pair_counts,
        "args": dataclasses.asdict(args),
        "label": {
            "definition": (
                "(base combined risk - update combined risk) / "
                "max(base combined risk + update combined risk, 1e-8)"
            ),
            "accept_if_gain_above": args.improvement_margin,
            "reject_if_gain_below": -args.improvement_margin,
            "ties_ignored_for_bce": True,
            "fresh_teacher_used_only_for_router_supervision": True,
        },
        "train": train_summary,
        "threshold_selection": threshold_selection,
        "evaluation": evaluations,
        "profile": profile,
        "offline_gate": {
            "pass": router_gate_pass,
            "requirements": {
                "test_coverage": [
                    args.minimum_accept_coverage,
                    args.maximum_accept_coverage,
                ],
                "test_auroc_above": 0.55,
                "routed_risk_below_both_raw_choices": True,
                "routed_risk_below_random_same_coverage": True,
                "routed_action_mse_below_both_raw_choices": True,
            },
        },
        "limitations": [
            "The editor is frozen and reject means keep the transported base, not rerun full ACoT.",
            "The router is trained on synthetic local time-warp expert windows, not disturbed closed-loop branches.",
            "The editor and router share train episodes in this aggressive pilot; a paper result needs out-of-fold router labels.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
