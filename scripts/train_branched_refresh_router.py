"""Probe whether an age-four observation can route a mid-horizon ACoT refresh.

The deployed policy is deliberately narrow: after four actions from an
eight-step plan it scores the risk of continuing with cached action token 4.
It never predicts or edits an action.  The supervision target is the 6-DoF
MSE between cached environment action 4 and fresh-teacher environment action
0 at the branched endpoint.

To avoid leaking the synthetic branch identity, the prefix input is always the
controller-intended prefix ``cached_actions_env[:4]``.  In particular, the
model never receives the actually executed scale/drop/gripper fault actions.
Roots are split by episode before branches are flattened.
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
import train_branched_effective_progress as progress_probe
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
    age: int = 4
    plan_horizon: int = 8
    failure_quantile: float = 0.75
    score_mse_weight: float = 0.25
    full_inference_ms: float = 95.844
    router_latency_guard_ms: float = 2.0
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
    ):
        raise ValueError("Training, batching, evaluation, and logging sizes must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Optimizer scales must be positive and weight_decay non-negative.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if args.age <= 0 or args.plan_horizon <= args.age:
        raise ValueError("Require 0 < age < plan_horizon.")
    if not 0 < args.failure_quantile < 1:
        raise ValueError("failure_quantile must lie in (0, 1).")
    if args.score_mse_weight < 0:
        raise ValueError("score_mse_weight must be non-negative.")
    if args.full_inference_ms <= 0 or args.router_latency_guard_ms < 0:
        raise ValueError("Latency values must be non-negative, with positive full inference latency.")
    if args.max_parameters <= 0 or args.max_parameters > 1_000_000:
        raise ValueError("max_parameters must lie in [1, 1_000_000].")


def _flat_arrays(
    arrays: dict[str, np.ndarray],
    indices: progress_probe.BranchIndices,
    *,
    age: int,
    plan_horizon: int,
) -> dict[str, np.ndarray]:
    roots = indices.roots
    branches = indices.branches
    branch_steps = np.asarray(arrays["branch_steps"])[roots, branches]
    if np.any(branch_steps != age):
        values, counts = np.unique(branch_steps, return_counts=True)
        histogram = dict(zip(values.astype(int).tolist(), counts.astype(int).tolist(), strict=True))
        raise ValueError(f"Refresh-router protocol requires every endpoint at age={age}; got {histogram}.")
    cached_actions_env = np.asarray(arrays["cached_actions_env"])[roots]
    if cached_actions_env.shape[1] <= age:
        raise ValueError(
            f"cached_actions_env horizon {cached_actions_env.shape[1]} has no token at age={age}."
        )
    if cached_actions_env.shape[1] < plan_horizon:
        raise ValueError(
            f"cached_actions_env horizon {cached_actions_env.shape[1]} is below plan_horizon={plan_horizon}."
        )
    cached_ear = np.asarray(arrays["cached_ear"])[roots]
    if cached_actions_env.shape[-1] > cached_ear.shape[-1]:
        raise ValueError("Environment action width exceeds the EAR token width.")
    padded_action_tokens = np.zeros(
        (cached_actions_env.shape[0], plan_horizon, cached_ear.shape[-1]),
        dtype=np.float32,
    )
    padded_action_tokens[..., : cached_actions_env.shape[-1]] = cached_actions_env[:, :plan_horizon]
    cached_plan_tokens = np.concatenate(
        [cached_ear.astype(np.float32), padded_action_tokens],
        axis=1,
    )
    intended_prefix = cached_actions_env[:, :age]
    intended_valid = np.ones(intended_prefix.shape[:2], dtype=np.bool_)
    return {
        "root_id": np.asarray(arrays["root_id"])[roots],
        "task_id": np.asarray(arrays["task_id"])[roots],
        "episode_id": np.asarray(arrays["episode_id"])[roots],
        "branch_id": np.asarray(arrays["branch_ids"])[roots, branches],
        "anchor_images": np.asarray(arrays["anchor_images"])[roots],
        "current_images": np.asarray(arrays["current_images"])[roots, branches],
        "anchor_state": np.asarray(arrays["anchor_state"])[roots],
        "current_state": np.asarray(arrays["current_state"])[roots, branches],
        "cached_plan_tokens": cached_plan_tokens,
        "cached_actions_env": cached_actions_env,
        "fresh_actions_env": np.asarray(arrays["fresh_actions_env"])[roots, branches],
        # Deployment-safe controller intent.  Never replace this with the
        # synthetic branch's actually executed actions in the main result.
        "intended_prefix": intended_prefix,
        "intended_valid": intended_valid,
    }


def _action_risk(flat: dict[str, np.ndarray], *, age: int) -> np.ndarray:
    cached = np.asarray(flat["cached_actions_env"], dtype=np.float32)
    fresh = np.asarray(flat["fresh_actions_env"], dtype=np.float32)
    if cached.ndim != 3 or fresh.ndim != 3 or cached.shape[0] != fresh.shape[0]:
        raise ValueError(f"Expected matching [N,H,D] action arrays, got {cached.shape}, {fresh.shape}.")
    if cached.shape[-1] < 6 or fresh.shape[-1] < 6:
        raise ValueError("Action arrays must have at least six continuous dimensions.")
    return np.mean(np.square(cached[:, age, :6] - fresh[:, 0, :6]), axis=-1).astype(np.float32)


def _root_nominal_excess(
    root_ids: np.ndarray,
    branch_ids: np.ndarray,
    risk: np.ndarray,
) -> np.ndarray:
    roots = np.asarray(root_ids)
    branches = np.asarray(branch_ids)
    values = np.asarray(risk, dtype=np.float32)
    nominal_by_root: dict[int, float] = {}
    for root in np.unique(roots):
        selected = (roots == root) & (branches == 0)
        if np.sum(selected) != 1:
            raise ValueError(f"Root {int(root)} must have exactly one valid nominal branch.")
        nominal_by_root[int(root)] = float(values[selected][0])
    return np.asarray(
        [value - nominal_by_root[int(root)] for root, value in zip(roots, values, strict=True)],
        dtype=np.float32,
    )


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
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be matching rank-one arrays.")
    positives = int(np.sum(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    mann_whitney = np.sum(ranks[labels]) - positives * (positives + 1) / 2
    return float(mann_whitney / (positives * negatives))


def _spearman_correlation(target: np.ndarray, score: np.ndarray) -> float | None:
    target_rank = _average_ranks(np.asarray(target))
    score_rank = _average_ranks(np.asarray(score))
    if np.std(target_rank) <= 1e-12 or np.std(score_rank) <= 1e-12:
        return None
    return float(np.corrcoef(target_rank, score_rank)[0, 1])


def _empirical_percentile_targets(train_risk: np.ndarray, risk: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(train_risk, dtype=np.float64))
    values = np.asarray(risk, dtype=np.float64)
    if reference.ndim != 1 or not reference.size or values.ndim != 1:
        raise ValueError("Risk arrays must be non-empty and rank one.")
    lower = np.searchsorted(reference, values, side="left")
    upper = np.searchsorted(reference, values, side="right")
    midrank = 0.5 * (lower + upper)
    return np.asarray(midrank / reference.size, dtype=np.float32)


def _positive_quantile_threshold(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float32)
    positive = array[array > 0]
    if not positive.size:
        raise ValueError("Positive-excess routing requires at least one positive training example.")
    return max(float(np.quantile(array, quantile)), float(np.min(positive)))


def _ranking_order(scores: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    tie_break = np.random.default_rng(seed).random(values.size)
    return np.lexsort((tie_break, -values))


def _selective_at_budget(
    risk: np.ndarray,
    scores: np.ndarray,
    *,
    budget: float,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(risk, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("risk must be a non-empty rank-one array.")
    if np.asarray(scores).shape != values.shape:
        raise ValueError("scores must match risk.")
    count = min(values.size, max(1, int(np.ceil(values.size * budget))))
    refreshed = _ranking_order(scores, seed=seed)[:count]
    residual = values.copy()
    residual[refreshed] = 0.0
    return {
        "count": int(values.size),
        "refresh_count": int(count),
        "realized_budget": float(count / values.size),
        "selective_action_mse_6d": float(np.mean(residual)),
        "risk_removed": float(np.mean(values) - np.mean(residual)),
    }


_BUDGETS = (0.10, 0.25, 0.50)


def _evaluate_scores(
    risk: np.ndarray,
    scores: np.ndarray,
    *,
    failure_threshold: float,
    seed: int,
    oracle_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(risk, dtype=np.float32)
    predictions = np.asarray(scores, dtype=np.float32)
    selective: dict[str, Any] = {}
    for offset, budget in enumerate(_BUDGETS):
        key = f"{round(100 * budget)}%"
        result = _selective_at_budget(values, predictions, budget=budget, seed=seed + offset)
        if oracle_scores is not None:
            oracle = _selective_at_budget(
                values,
                oracle_scores,
                budget=budget,
                seed=seed + offset,
            )
            no_refresh = float(np.mean(values))
            denominator = no_refresh - float(oracle["selective_action_mse_6d"])
            result["oracle_selective_action_mse_6d"] = oracle["selective_action_mse_6d"]
            result["oracle_closable_risk_gap_fraction"] = (
                float((no_refresh - float(result["selective_action_mse_6d"])) / denominator)
                if denominator > 1e-12
                else None
            )
        selective[key] = result
    return {
        "count": int(values.size),
        "mean_total_risk": float(np.mean(values)),
        "failure_rate": float(np.mean(values >= failure_threshold)),
        "auroc": _binary_auroc(values >= failure_threshold, predictions),
        "spearman_risk_score": _spearman_correlation(values, predictions),
        "selective": selective,
    }


def _stratified_evaluation(
    risk: np.ndarray,
    scores: np.ndarray,
    flat: dict[str, np.ndarray],
    *,
    failure_threshold: float,
    seed: int,
) -> dict[str, Any]:
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)

    def evaluate(mask: np.ndarray, offset: int) -> dict[str, Any]:
        return _evaluate_scores(
            risk[mask],
            scores[mask],
            failure_threshold=failure_threshold,
            seed=seed + offset,
            oracle_scores=risk[mask],
        )

    return {
        "overall": evaluate(np.ones(risk.shape, dtype=np.bool_), 0),
        "nominal_disturbed": {
            "nominal": evaluate(branch_ids == 0, 100),
            "disturbed": evaluate(branch_ids != 0, 200),
        },
        "by_branch": {
            name: evaluate(branch_ids == branch_id, 300 + branch_id)
            for branch_id, name in enumerate(branched_dataset.BRANCH_NAMES)
            if np.any(branch_ids == branch_id)
        },
    }


def _batch(
    flat: dict[str, np.ndarray],
    percentile_targets: np.ndarray,
    failure_labels: np.ndarray,
    indices: np.ndarray,
    *,
    current_equals_anchor: bool,
) -> dict[str, jax.Array]:
    anchor_images = flat["anchor_images"][indices].astype(np.float32) / 255.0
    anchor_state = flat["anchor_state"][indices].astype(np.float32)
    current_images = (
        anchor_images if current_equals_anchor else flat["current_images"][indices].astype(np.float32) / 255.0
    )
    current_state = anchor_state if current_equals_anchor else flat["current_state"][indices].astype(np.float32)
    return {
        "anchor_images": jnp.asarray(anchor_images),
        "current_images": jnp.asarray(current_images),
        "anchor_state": jnp.asarray(anchor_state),
        "current_state": jnp.asarray(current_state),
        "cached_plan_tokens": jnp.asarray(flat["cached_plan_tokens"][indices], dtype=jnp.float32),
        "intended_prefix": jnp.asarray(flat["intended_prefix"][indices], dtype=jnp.float32),
        "intended_valid": jnp.asarray(flat["intended_valid"][indices], dtype=jnp.bool_),
        "percentile_target": jnp.asarray(percentile_targets[indices], dtype=jnp.float32),
        "failure_label": jnp.asarray(failure_labels[indices], dtype=jnp.float32),
    }


def _loss(
    model: progress_probe.EffectiveProgressProbe,
    batch: dict[str, jax.Array],
    *,
    score_mse_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    score = model(
        batch["anchor_images"],
        batch["current_images"],
        batch["anchor_state"],
        batch["current_state"],
        batch["cached_plan_tokens"],
        batch["intended_prefix"],
        batch["intended_valid"],
    )
    score = jnp.clip(score, 1e-6, 1.0 - 1e-6)
    binary_cross_entropy = -jnp.mean(
        batch["failure_label"] * jnp.log(score)
        + (1.0 - batch["failure_label"]) * jnp.log1p(-score)
    )
    score_mse = jnp.mean(jnp.square(score - batch["percentile_target"]))
    total = binary_cross_entropy + score_mse_weight * score_mse
    return total, {
        "loss": total,
        "binary_cross_entropy": binary_cross_entropy,
        "percentile_score_mse": score_mse,
        "score_mean": jnp.mean(score),
    }


def _predict_all(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], jax.Array],
    params: nnx.State,
    flat: dict[str, np.ndarray],
    percentile_targets: np.ndarray,
    failure_labels: np.ndarray,
    *,
    batch_size: int,
    current_equals_anchor: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(percentile_targets), batch_size):
        indices = np.arange(start, min(start + batch_size, len(percentile_targets)))
        prediction = predict_step(
            params,
            _batch(
                flat,
                percentile_targets,
                failure_labels,
                indices,
                current_equals_anchor=current_equals_anchor,
            ),
        )
        outputs.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
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


def _train_one(
    train_flat: dict[str, np.ndarray],
    train_percentile: np.ndarray,
    train_failure: np.ndarray,
    validation_flat: dict[str, np.ndarray],
    validation_percentile: np.ndarray,
    validation_failure: np.ndarray,
    validation_risk: np.ndarray,
    *,
    current_equals_anchor: bool,
    model_name: str,
    args: Args,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    image_shape = train_flat["anchor_images"].shape
    model = progress_probe.EffectiveProgressProbe(
        image_views=image_shape[1],
        image_channels=image_shape[-1],
        state_dim=train_flat["anchor_state"].shape[-1],
        action_dim=train_flat["cached_plan_tokens"].shape[-1],
        env_action_dim=train_flat["intended_prefix"].shape[-1],
        max_executed_steps=train_flat["intended_prefix"].shape[-2],
        max_phase=1.0,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(model)
    parameter_count = int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))
    if parameter_count >= args.max_parameters:
        raise ValueError(f"{model_name} has {parameter_count:,} parameters; limit={args.max_parameters:,}.")
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

        def loss_fn(candidate: progress_probe.EffectiveProgressProbe):
            return _loss(candidate, batch, score_mse_weight=args.score_mse_weight)

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
            batch["cached_plan_tokens"],
            batch["intended_prefix"],
            batch["intended_valid"],
        )

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / f"metrics_{model_name}.jsonl"
    best_score = float("inf")
    best_step = 0
    best_params: nnx.State | None = None
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                len(train_percentile),
                size=args.batch_size,
                replace=len(train_percentile) < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                _batch(
                    train_flat,
                    train_percentile,
                    train_failure,
                    sampled,
                    current_equals_anchor=current_equals_anchor,
                ),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_scores = _predict_all(
                    predict_step,
                    params,
                    validation_flat,
                    validation_percentile,
                    validation_failure,
                    batch_size=args.eval_batch_size,
                    current_equals_anchor=current_equals_anchor,
                )
                validation_selective = _selective_at_budget(
                    validation_risk,
                    validation_scores,
                    budget=0.25,
                    seed=args.seed,
                )
                selection_score = float(validation_selective["selective_action_mse_6d"])
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()},
                    "validation/auroc": _binary_auroc(validation_failure, validation_scores),
                    "validation/selective_action_mse_6d_at_25%": selection_score,
                }
                finite_values = [value for value in record.values() if value is not None]
                if not all(np.isfinite(value) for value in finite_values):
                    raise FloatingPointError(f"Non-finite refresh-router metric: {record}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps({"model": model_name, **record}, sort_keys=True), flush=True)
                if selection_score < best_score:
                    best_score = selection_score
                    best_step = step
                    best_params = params
    selected_params = best_params if best_params is not None else params
    params_path = output_dir / model_name / "params"
    _save_params(
        selected_params,
        params_path,
        model_name=model_name,
        overwrite=args.overwrite,
    )
    return (
        predict_step,
        selected_params,
        {
            "parameter_count": parameter_count,
            "completed_steps": args.train_steps,
            "best_validation_step": best_step,
            "best_validation_selective_action_mse_6d_at_25%": best_score,
            "selection_criterion": "validation selective action MSE at 25% refresh budget",
            "elapsed_seconds": time.monotonic() - started,
            "params_path": str(params_path.resolve()),
        },
    )


def _calibration_scores(
    train_risk: np.ndarray,
    train_branch_ids: np.ndarray,
    test_branch_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    global_mean = float(np.mean(train_risk))
    branch_means = {}
    for branch_id in range(len(branched_dataset.BRANCH_NAMES)):
        selected = train_branch_ids == branch_id
        branch_means[branch_id] = float(np.mean(train_risk[selected])) if np.any(selected) else global_mean
    return {
        "random": np.random.default_rng(seed).random(len(test_branch_ids)).astype(np.float32),
        "train_global_constant": np.full((len(test_branch_ids),), global_mean, dtype=np.float32),
        "branch_id_train_mean_privileged": np.asarray(
            [branch_means[int(branch_id)] for branch_id in test_branch_ids],
            dtype=np.float32,
        ),
        "oracle_total_risk": np.empty((0,), dtype=np.float32),
    }


def _latency_estimate(args: Args, refresh_budget: float) -> dict[str, Any]:
    fixed_calendar_average = (
        args.full_inference_ms * (1.0 + refresh_budget) + args.router_latency_guard_ms
    ) / args.plan_horizon
    fixed_calendar_max_router_for_3x = (
        args.plan_horizon * args.full_inference_ms / 3.0
        - args.full_inference_ms * (1.0 + refresh_budget)
    )
    renewal_steps = args.plan_horizon - args.age * refresh_budget
    renewal_average = (args.full_inference_ms + args.router_latency_guard_ms) / renewal_steps
    renewal_max_router_for_3x = (
        renewal_steps * args.full_inference_ms / 3.0 - args.full_inference_ms
    )
    return {
        "refresh_budget": refresh_budget,
        "assumed_router_latency_guard_ms": args.router_latency_guard_ms,
        "conservative_fixed_calendar": {
            "definition": "(1 + refresh_budget) * full / 8, plus one router call per 8 actions",
            "average_latency_ms_per_action": fixed_calendar_average,
            "estimated_speedup_vs_full_every_action": args.full_inference_ms / fixed_calendar_average,
            "max_router_latency_ms_for_3x": fixed_calendar_max_router_for_3x,
        },
        "refresh_resets_plan_renewal": {
            "definition": "(full + router) / (8 - 4 * refresh_budget)",
            "average_cycle_actions": renewal_steps,
            "average_latency_ms_per_action": renewal_average,
            "estimated_speedup_vs_full_every_action": args.full_inference_ms / renewal_average,
            "max_router_latency_ms_for_3x": renewal_max_router_for_3x,
        },
    }


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
            "cached_ear",
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
            _flat_arrays(
                arrays,
                indices,
                age=args.age,
                plan_horizon=args.plan_horizon,
            ),
        )

    risks = {name: _action_risk(flat, age=args.age) for name, (_, flat) in partitions.items()}
    excess = {
        name: _root_nominal_excess(flat["root_id"], flat["branch_id"], risks[name])
        for name, (_, flat) in partitions.items()
    }
    positive_excess = {name: np.maximum(values, 0.0) for name, values in excess.items()}
    failure_threshold = float(np.quantile(risks["train"], args.failure_quantile))
    excess_failure_threshold = _positive_quantile_threshold(
        positive_excess["train"],
        args.failure_quantile,
    )
    percentiles = {
        name: _empirical_percentile_targets(risks["train"], risk) for name, risk in risks.items()
    }
    failures = {name: risk >= failure_threshold for name, risk in risks.items()}

    trained: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name, current_equals_anchor in (
        ("current_observation", False),
        ("matched_no_current", True),
    ):
        predict_step, params, train_summary = _train_one(
            partitions["train"][1],
            percentiles["train"],
            failures["train"],
            partitions["validation"][1],
            percentiles["validation"],
            failures["validation"],
            risks["validation"],
            current_equals_anchor=current_equals_anchor,
            model_name=model_name,
            args=args,
            output_dir=output_dir,
        )
        predictions[model_name] = _predict_all(
            predict_step,
            params,
            partitions["test"][1],
            percentiles["test"],
            failures["test"],
            batch_size=args.eval_batch_size,
            current_equals_anchor=current_equals_anchor,
        )
        trained[model_name] = train_summary

    test_flat = partitions["test"][1]
    baseline_scores = _calibration_scores(
        risks["train"],
        np.asarray(partitions["train"][1]["branch_id"], dtype=np.int64),
        np.asarray(test_flat["branch_id"], dtype=np.int64),
        seed=args.seed,
    )
    baseline_scores["oracle_total_risk"] = risks["test"]
    all_scores = {**baseline_scores, **predictions}
    evaluations = {
        name: _stratified_evaluation(
            risks["test"],
            scores,
            test_flat,
            failure_threshold=failure_threshold,
            seed=args.seed,
        )
        for name, scores in all_scores.items()
    }
    excess_evaluations = {
        name: _stratified_evaluation(
            positive_excess["test"],
            scores,
            test_flat,
            failure_threshold=excess_failure_threshold,
            seed=args.seed + 10_000,
        )
        for name, scores in all_scores.items()
    }
    current_auroc = evaluations["current_observation"]["overall"]["auroc"]
    current_excess_auroc = excess_evaluations["current_observation"]["overall"]["auroc"]
    no_current_excess_auroc = excess_evaluations["matched_no_current"]["overall"]["auroc"]
    current_gap = evaluations["current_observation"]["overall"]["selective"]["25%"][
        "oracle_closable_risk_gap_fraction"
    ]
    latency_25 = _latency_estimate(args, 0.25)
    gates = {
        "current_test_auroc_at_least_0.70": current_auroc is not None and current_auroc >= 0.70,
        "current_minus_no_current_excess_auroc_at_least_0.05": (
            current_excess_auroc is not None
            and no_current_excess_auroc is not None
            and current_excess_auroc - no_current_excess_auroc >= 0.05
        ),
        "25%_budget_closes_at_least_half_oracle_gap": current_gap is not None and current_gap >= 0.50,
        "conservative_fixed_calendar_speedup_at_least_3x": (
            latency_25["conservative_fixed_calendar"]["estimated_speedup_vs_full_every_action"] >= 3.0
        ),
    }
    gates["all_hard_gates_pass"] = all(gates.values())

    summary = {
        "method": {
            "name": "branched_age4_refresh_router_probe",
            "deployment": "1:8 plan; score once at age 4; optionally refresh the full teacher",
            "output": "one refresh-risk score only; never an action or action residual",
            "total_risk_target": "mean squared error over cached action token 4 vs fresh action token 0, dims 0:6",
            "prefix_protocol": (
                "controller-intended cached_actions_env[:4] for every branch; actual synthetic fault actions "
                "are excluded to prevent branch leakage"
            ),
            "plan_input": "cached EAR followed by the padded complete cached 1:8 environment-action plan",
            "failure_definition": (
                f"total risk at or above train quantile {args.failure_quantile} "
                f"(threshold={failure_threshold:.9g})"
            ),
            "positive_excess_definition": (
                "max(total risk - same-root nominal-branch risk, 0); "
                f"train quantile threshold={excess_failure_threshold:.9g}"
            ),
            "current_and_no_current": "independently trained matched architectures and optimization budgets",
        },
        "args": dataclasses.asdict(args),
        "split": {
            name: {
                "root_count": len(roots),
                "valid_branch_count": len(flat["branch_id"]),
                "episodes_by_task": progress_probe._episode_summary(arrays, roots),  # noqa: SLF001
            }
            for name, (roots, flat) in partitions.items()
        },
        "risk_diagnostics": {
            name: {
                "total_risk_mean": float(np.mean(risks[name])),
                "total_risk_std": float(np.std(risks[name])),
                "root_nominal_counterfactual_excess_mean": float(np.mean(excess[name])),
                "root_nominal_counterfactual_excess_positive_fraction": float(np.mean(excess[name] > 0)),
                "positive_excess_mean": float(np.mean(positive_excess[name])),
            }
            for name in partitions
        },
        "train": trained,
        "test": {
            "root_count": len(test_roots),
            "branch_count": len(risks["test"]),
            "total_risk_evaluations": evaluations,
            "positive_excess_evaluations": excess_evaluations,
        },
        "latency_estimates": {
            f"{round(100 * budget)}%": _latency_estimate(args, budget) for budget in _BUDGETS
        },
        "hard_gates": gates,
        "interpretation_guard": (
            "This is an offline same-root routing probe. Refreshed samples are assigned zero action error, "
            "so selective MSE is an oracle-refresh accounting metric, not closed-loop task success. Router "
            "latency uses the explicit guard value and must be measured before a deployment speed claim."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
