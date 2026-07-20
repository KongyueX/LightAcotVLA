"""Train a frozen-feature Q-guided execution-horizon selector.

This is the supervised warm start for the RL pilot.  The ACoT-VLA base policy
and the existing V2-P predictor are loaded read-only.  A small client-side
actor/critic is trained from H1-H10 counterfactual labels, with repeated
counterfactual outcomes receiving stronger probability supervision.
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
import tyro

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import rl_selector
from openpi.models import model as model_lib
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictor
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    repeated_outcomes: tuple[str, ...]
    predictor_params: str
    output_dir: str
    candidates: tuple[int, ...] = rl_selector.DEFAULT_CANDIDATES
    seed: int = 7
    predictor_hidden_dim: int = 256
    predictor_temporal_layers: int = 3
    selector_hidden_dim: int = 64
    batch_size: int = 128
    critic_steps: int = 1500
    distillation_steps: int = 750
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    repeated_min_trials: int = 5
    empirical_success_margin: float = 0.4
    repeated_sample_multiplier: float = 4.0
    focus_task_ids: tuple[int, ...] = (8, 9)
    focus_task_multiplier: float = 2.0
    q_cost_weight: float = 0.25
    value_weight: float = 0.25
    base_q_anchor_weight: float = 0.10
    actor_weight: float = 1.0
    actor_reference_anchor_weight: float = 0.05
    selector_minimum_success_probability: float = 0.5
    selector_reference_slack: float = 0.05
    selector_q_tie_margin: float = 0.03
    log_interval: int = 100


_INPUT_FIELDS = (
    "prefix_feature",
    "state",
    "coarse_actions",
    "final_actions",
    "previous_actions",
    "previous_h",
    "budget_balance",
    "episode_progress",
    "previous_valid",
)


def _predictor_config(arrays: dict[str, np.ndarray], args: Args) -> ExecutionHorizonPredictorConfig:
    return ExecutionHorizonPredictorConfig(
        prefix_feature_dim=int(arrays["prefix_feature"].shape[-1]),
        state_dim=int(arrays["state"].shape[-1]),
        action_dim=int(arrays["final_actions"].shape[-1]),
        coarse_horizon=int(arrays["coarse_actions"].shape[-2]),
        action_horizon=int(arrays["final_actions"].shape[-2]),
        hidden_dim=args.predictor_hidden_dim,
        temporal_layers=args.predictor_temporal_layers,
    )


def _restore_predictor(
    config: ExecutionHorizonPredictorConfig,
    predictor_params: str,
    *,
    seed: int,
) -> ExecutionHorizonPredictor:
    module = ExecutionHorizonPredictor(config, rngs=nnx.Rngs(seed))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(predictor_params, dtype=jnp.float32))
    if "execution_horizon_predictor" in loaded:
        loaded = loaded["execution_horizon_predictor"]
    graphdef, state = nnx.split(module)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _predict_all(
    module: ExecutionHorizonPredictor,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    @jax.jit
    def infer(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return module(**batch)

    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(arrays["task_id"]), batch_size):
        selected = slice(start, start + batch_size)
        batch = {name: jnp.asarray(arrays[name][selected]) for name in _INPUT_FIELDS}
        prediction = jax.device_get(infer(batch))
        for name, value in prediction.items():
            pieces.setdefault(name, []).append(np.asarray(value))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def _load_repeated_outcomes(paths: tuple[str, ...]) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    records: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for item in paths:
        path = pathlib.Path(item)
        with path.open() as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    int(record["task_id"]),
                    int(record["episode_id"]),
                    int(record["decision_step"]),
                    int(record["root_seed"]),
                )
                if key in records:
                    raise ValueError(f"Duplicate repeated-outcome root {key} in {path}:{line_number}.")
                records[key] = record
    return records


def _training_targets(
    arrays: dict[str, np.ndarray],
    repeated_records: dict[tuple[int, int, int, int], dict[str, Any]],
    args: Args,
) -> dict[str, np.ndarray]:
    indices = rl_selector.candidate_indices(args.candidates)
    success = np.asarray(arrays["branch_success"][:, indices], dtype=np.float32)
    cost = np.asarray(arrays["remaining_calls"][:, indices], dtype=np.float32) / 100.0
    label_weight = np.asarray(arrays["branch_valid"][:, indices], dtype=np.float32)
    repeated_mask = np.zeros(len(success), dtype=np.bool_)
    matched_repeated = 0
    for row_index in range(len(success)):
        key = (
            int(arrays["task_id"][row_index]),
            int(arrays["episode_id"][row_index]),
            int(arrays["decision_step"][row_index]),
            int(arrays["root_seed"][row_index]),
        )
        record = repeated_records.get(key)
        if record is None:
            continue
        counts = []
        for candidate_index, horizon in enumerate(args.candidates):
            outcomes = record["outcomes_by_h"].get(str(horizon), [])
            if not outcomes:
                continue
            success[row_index, candidate_index] = np.mean([bool(outcome["success"]) for outcome in outcomes])
            cost[row_index, candidate_index] = (
                np.mean([float(outcome["remaining_calls"]) for outcome in outcomes]) / 100.0
            )
            label_weight[row_index, candidate_index] = len(outcomes)
            counts.append(len(outcomes))
        if len(counts) == len(args.candidates) and min(counts) >= args.repeated_min_trials:
            repeated_mask[row_index] = True
        matched_repeated += 1

    reference_index = args.candidates.index(10)
    empirical_target = np.full(len(success), reference_index, dtype=np.int32)
    for row_index in np.flatnonzero(repeated_mask):
        advantage = success[row_index] - success[row_index, reference_index]
        improved = advantage >= args.empirical_success_margin - 1e-7
        if not np.any(improved):
            continue
        best_success = np.max(success[row_index, improved])
        near_best = improved & (success[row_index] >= best_success - 1e-7)
        empirical_target[row_index] = int(np.argmin(np.where(near_best, cost[row_index], np.inf)))

    if repeated_records and matched_repeated == 0:
        raise ValueError(
            "No repeated-outcome record matched the HDF5 roots. Include the corresponding repeated HDF5 datasets."
        )
    return {
        "success": success,
        "cost": cost,
        "label_weight": label_weight,
        "repeated_mask": repeated_mask,
        "empirical_target": empirical_target,
        "matched_repeated": np.asarray(matched_repeated),
    }


def _initialize_params(
    rng: np.random.Generator,
    *,
    feature_dim: int,
    hidden_dim: int,
    num_actions: int,
    reference_index: int,
) -> dict[str, jax.Array]:
    trunk_scale = np.sqrt(2.0 / (feature_dim + hidden_dim))
    head_scale = np.sqrt(2.0 / (hidden_dim + num_actions))
    actor_b = np.zeros(num_actions, dtype=np.float32)
    actor_b[reference_index] = 1.5
    return {
        "trunk_w": jnp.asarray(rng.normal(0.0, trunk_scale, (feature_dim, hidden_dim)), dtype=jnp.float32),
        "trunk_b": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "actor_w": jnp.asarray(rng.normal(0.0, head_scale, (hidden_dim, num_actions)), dtype=jnp.float32),
        "actor_b": jnp.asarray(actor_b),
        "q_success_w": jnp.asarray(rng.normal(0.0, head_scale, (hidden_dim, num_actions)), dtype=jnp.float32),
        "q_success_b": jnp.zeros((num_actions,), dtype=jnp.float32),
        "q_cost_w": jnp.asarray(rng.normal(0.0, head_scale, (hidden_dim, num_actions)), dtype=jnp.float32),
        "q_cost_b": jnp.zeros((num_actions,), dtype=jnp.float32),
        "value_w": jnp.asarray(rng.normal(0.0, head_scale, (hidden_dim, 1)), dtype=jnp.float32),
        "value_b": jnp.zeros((1,), dtype=jnp.float32),
    }


def _forward(params: dict[str, jax.Array], feature: jax.Array) -> dict[str, jax.Array]:
    hidden = jnp.tanh(feature @ params["trunk_w"] + params["trunk_b"])
    return {
        "actor_logits": hidden @ params["actor_w"] + params["actor_b"],
        "q_success_logits": hidden @ params["q_success_w"] + params["q_success_b"],
        "q_cost": jax.nn.softplus(hidden @ params["q_cost_w"] + params["q_cost_b"]),
        "value_logits": (hidden @ params["value_w"] + params["value_b"])[..., 0],
    }


def _bce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _huber(values: jax.Array, delta: float = 1.0) -> jax.Array:
    absolute = jnp.abs(values)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * quadratic**2 + delta * (absolute - quadratic)


def _q_distillation_targets(
    q_success: jax.Array,
    q_cost: jax.Array,
    empirical_target: jax.Array,
    repeated_mask: jax.Array,
    *,
    reference_index: int,
    minimum_success_probability: float,
    reference_slack: float,
    q_tie_margin: float,
) -> jax.Array:
    reference = q_success[:, reference_index]
    threshold = jnp.maximum(minimum_success_probability, reference - reference_slack)
    eligible = q_success >= threshold[:, None]
    eligible = eligible.at[:, reference_index].set(True)
    best_success = jnp.max(jnp.where(eligible, q_success, -jnp.inf), axis=-1)
    near_best = eligible & (q_success >= best_success[:, None] - q_tie_margin)
    critic_target = jnp.argmin(jnp.where(near_best, q_cost, jnp.inf), axis=-1)
    return jnp.where(repeated_mask, empirical_target, critic_target)


def _make_train_step(args: Args, *, reference_index: int):
    def loss_fn(
        params: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        actor_scale: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = _forward(params, batch["feature"])
        label_weight = batch["label_weight"]
        q_bce = jnp.sum(_bce(outputs["q_success_logits"], batch["success"]) * label_weight)
        q_bce /= jnp.maximum(jnp.sum(label_weight), 1.0)
        cost_huber = jnp.sum(_huber(outputs["q_cost"] - batch["cost"]) * label_weight)
        cost_huber /= jnp.maximum(jnp.sum(label_weight), 1.0)
        q_probability = jax.nn.sigmoid(outputs["q_success_logits"])
        base_anchor = jnp.mean((q_probability - batch["base_q_probability"]) ** 2)

        target_index = _q_distillation_targets(
            jax.lax.stop_gradient(q_probability),
            jax.lax.stop_gradient(outputs["q_cost"]),
            batch["empirical_target"],
            batch["repeated_mask"],
            reference_index=reference_index,
            minimum_success_probability=args.selector_minimum_success_probability,
            reference_slack=args.selector_reference_slack,
            q_tie_margin=args.selector_q_tie_margin,
        )
        selected_success = jnp.take_along_axis(batch["success"], target_index[:, None], axis=-1)[:, 0]
        value_bce = jnp.mean(_bce(outputs["value_logits"], selected_success))

        actor_log_probability = jax.nn.log_softmax(outputs["actor_logits"], axis=-1)
        actor_ce_per_example = -jnp.take_along_axis(actor_log_probability, target_index[:, None], axis=-1)[:, 0]
        actor_example_weight = jnp.where(batch["repeated_mask"], args.repeated_sample_multiplier, 1.0)
        actor_ce = jnp.sum(actor_ce_per_example * actor_example_weight) / jnp.sum(actor_example_weight)
        reference_anchor = -jnp.mean(actor_log_probability[:, reference_index])
        loss = (
            q_bce
            + args.q_cost_weight * cost_huber
            + args.value_weight * value_bce
            + args.base_q_anchor_weight * base_anchor
            + actor_scale * args.actor_weight * (actor_ce + args.actor_reference_anchor_weight * reference_anchor)
        )
        metrics = {
            "loss": loss,
            "q_success_bce": q_bce,
            "q_cost_huber": cost_huber,
            "value_bce": value_bce,
            "base_q_anchor_mse": base_anchor,
            "actor_ce": actor_ce,
            "reference_anchor_ce": reference_anchor,
            "target_reference_rate": jnp.mean(target_index == reference_index),
            "predicted_success_mean": jnp.mean(q_probability),
        }
        return loss, metrics

    @jax.jit
    def train_step(
        params: dict[str, jax.Array],
        optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
        actor_scale: jax.Array,
    ) -> tuple[dict[str, jax.Array], optax.OptState, dict[str, jax.Array]]:
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, actor_scale)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return params, optimizer_state, metrics

    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    return optimizer, train_step


def _full_metrics(
    params: dict[str, jax.Array],
    data: dict[str, np.ndarray],
    *,
    args: Args,
    reference_index: int,
) -> dict[str, Any]:
    outputs = jax.device_get(_forward(params, jnp.asarray(data["feature"])))
    q_probability = np.asarray(jax.nn.sigmoid(outputs["q_success_logits"]))
    q_cost = np.asarray(outputs["q_cost"])
    target = np.asarray(
        _q_distillation_targets(
            jnp.asarray(q_probability),
            jnp.asarray(q_cost),
            jnp.asarray(data["empirical_target"]),
            jnp.asarray(data["repeated_mask"]),
            reference_index=reference_index,
            minimum_success_probability=args.selector_minimum_success_probability,
            reference_slack=args.selector_reference_slack,
            q_tie_margin=args.selector_q_tie_margin,
        )
    )
    actor_choice = np.argmax(np.asarray(outputs["actor_logits"]), axis=-1)
    repeated = np.asarray(data["repeated_mask"], dtype=np.bool_)
    result: dict[str, Any] = {
        "q_target_h_distribution": {
            str(args.candidates[index]): int(np.sum(target == index)) for index in range(len(args.candidates))
        },
        "actor_h_distribution": {
            str(args.candidates[index]): int(np.sum(actor_choice == index)) for index in range(len(args.candidates))
        },
        "actor_q_target_agreement": float(np.mean(actor_choice == target)),
        "q_success_brier": float(np.average((q_probability - data["success"]) ** 2, weights=data["label_weight"])),
    }
    if np.any(repeated):
        repeated_target = np.asarray(data["empirical_target"])[repeated]
        repeated_actor = actor_choice[repeated]
        repeated_q = target[repeated]
        result.update(
            {
                "num_repeated_roots": int(np.sum(repeated)),
                "repeated_actor_empirical_target_accuracy": float(np.mean(repeated_actor == repeated_target)),
                "repeated_q_empirical_target_accuracy": float(np.mean(repeated_q == repeated_target)),
                "repeated_empirical_target_h_distribution": {
                    str(args.candidates[index]): int(np.sum(repeated_target == index))
                    for index in range(len(args.candidates))
                },
            }
        )
    return result


def main(args: Args) -> None:
    if args.batch_size <= 0 or args.critic_steps < 0 or args.distillation_steps <= 0:
        raise ValueError("batch_size and distillation_steps must be positive; critic_steps must be non-negative.")
    if tuple(sorted(set(args.candidates))) != args.candidates or 10 not in args.candidates:
        raise ValueError("candidates must be unique, sorted and include H=10.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Training is already complete: {summary_path}")

    started = time.perf_counter()
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    predictor = _restore_predictor(
        _predictor_config(arrays, args),
        args.predictor_params,
        seed=args.seed,
    )
    predictions = _predict_all(predictor, arrays, batch_size=args.batch_size)
    feature = rl_selector.build_selector_feature(predictions)
    feature_mean = np.mean(feature, axis=0).astype(np.float32)
    feature_std = np.std(feature, axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-3)
    normalized_feature = (feature - feature_mean) / feature_std

    repeated_records = _load_repeated_outcomes(args.repeated_outcomes)
    targets = _training_targets(arrays, repeated_records, args)
    candidate_indices = rl_selector.candidate_indices(args.candidates)
    base_q_probability = rl_selector.sigmoid(predictions["success_logits"][:, candidate_indices]).astype(np.float32)
    data = {
        "feature": normalized_feature.astype(np.float32),
        "success": targets["success"],
        "cost": targets["cost"],
        "label_weight": targets["label_weight"],
        "repeated_mask": targets["repeated_mask"],
        "empirical_target": targets["empirical_target"],
        "base_q_probability": base_q_probability,
    }

    rng = np.random.default_rng(args.seed)
    reference_index = args.candidates.index(10)
    params = _initialize_params(
        rng,
        feature_dim=feature.shape[-1],
        hidden_dim=args.selector_hidden_dim,
        num_actions=len(args.candidates),
        reference_index=reference_index,
    )
    optimizer, train_step = _make_train_step(args, reference_index=reference_index)
    optimizer_state = optimizer.init(params)
    sample_weight = np.ones(len(feature), dtype=np.float64)
    sample_weight *= np.where(
        np.isin(arrays["task_id"], np.asarray(args.focus_task_ids)),
        args.focus_task_multiplier,
        1.0,
    )
    sample_weight *= np.where(targets["repeated_mask"], args.repeated_sample_multiplier, 1.0)
    sample_weight /= sample_weight.sum()
    total_steps = args.critic_steps + args.distillation_steps
    metrics_path = output_dir / "metrics.jsonl"
    last_metrics: dict[str, float] = {}
    with metrics_path.open("w") as metrics_file:
        for step in range(total_steps):
            selected = rng.choice(len(feature), size=args.batch_size, replace=True, p=sample_weight)
            batch = {name: jnp.asarray(value[selected]) for name, value in data.items()}
            actor_scale = jnp.asarray(float(step >= args.critic_steps), dtype=jnp.float32)
            params, optimizer_state, metrics = train_step(params, optimizer_state, batch, actor_scale)
            if step % args.log_interval == 0 or step + 1 == total_steps:
                last_metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
                record = {
                    "step": step + 1,
                    "phase": "critic" if step < args.critic_steps else "distill",
                    **last_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)

    selector = rl_selector.FrozenFeatureSelector(
        candidates=args.candidates,
        feature_mean=feature_mean,
        feature_std=feature_std,
        params={name: np.asarray(value) for name, value in jax.device_get(params).items()},
        metadata={
            "algorithm": "q_guided_sft",
            "base_policy_frozen": True,
            "v2p_encoder_frozen": True,
            "predictor_params": str(pathlib.Path(args.predictor_params).resolve()),
            "selector_minimum_success_probability": args.selector_minimum_success_probability,
            "selector_reference_slack": args.selector_reference_slack,
            "selector_q_tie_margin": args.selector_q_tie_margin,
        },
    )
    selector_path = selector.save(output_dir / "selector_sft.npz")
    replay_path = output_dir / "selector_replay.npz"
    with replay_path.open("wb") as file:
        np.savez_compressed(
            file,
            feature=data["feature"],
            success=data["success"],
            cost=data["cost"],
            label_weight=data["label_weight"],
        )
    audit = _full_metrics(params, data, args=args, reference_index=reference_index)
    summary = {
        "status": "complete",
        "base_policy_loaded": False,
        "base_policy_frozen": True,
        "v2p_predictor_loaded": True,
        "v2p_encoder_frozen": True,
        "dataset_inputs": list(args.dataset),
        "repeated_outcome_inputs": list(args.repeated_outcomes),
        "num_records": len(feature),
        "num_repeated_json_roots": len(repeated_records),
        "num_matched_repeated_roots": int(targets["matched_repeated"]),
        "selector_params": str(selector_path.resolve()),
        "selector_replay": str(replay_path.resolve()),
        "feature_dim": int(feature.shape[-1]),
        "last_train_metrics": last_metrics,
        "offline_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "config": dataclasses.asdict(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
