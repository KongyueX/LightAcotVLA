"""Update a small ordered selector from one batch of sampled complete episodes."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.execution_horizon.ordered_smdp import OrderedSMDPSelector
from openpi.execution_horizon.ordered_smdp import compute_smdp_gae


@dataclasses.dataclass(frozen=True)
class Args:
    rollout_dir: str
    input_checkpoint: str
    output_dir: str
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    clip_ratio: float = 0.1
    anchor_kl_weight: float = 0.1
    success_value_weight: float = 0.5
    cost_value_weight: float = 0.5
    target_kl: float = 0.02
    gamma: float = 1.0
    gae_lambda: float = 0.995
    rpc_budget_seconds: float = 3.0
    dual_learning_rate: float = 0.01
    max_cost_multiplier: float = 0.05
    seed: int = 7


def _validate_args(args: Args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    for name in ("learning_rate", "critic_learning_rate", "target_kl", "rpc_budget_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("anchor_kl_weight", "success_value_weight", "cost_value_weight", "dual_learning_rate", "max_cost_multiplier"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not 0 < args.clip_ratio < 1 or not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("Invalid clipping, discount, or GAE coefficient.")


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    previous_limit = csv.field_size_limit()
    try:
        # A serialized 29x256 token cache exceeds the CSV default of 128 KiB.
        csv.field_size_limit(max(previous_limit, 1024 * 1024))
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    finally:
        csv.field_size_limit(previous_limit)


def _episode_targets(
    decisions: list[dict[str, str]], outcome: dict[str, str], args: Args
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray([int(row["environment_step"]) for row in decisions], dtype=np.int64)
    durations = np.diff(np.concatenate([steps, [int(outcome["steps"])]]))
    if np.any(durations <= 0):
        raise ValueError("Each decision must cover a positive observed duration before the terminal step.")
    info = [json.loads(row["selector_json"]) for row in decisions]
    success_values = np.asarray([row["smdp_success_value"] for row in info], dtype=np.float32)
    cost_values = np.asarray([row["smdp_cost_value"] for row in info], dtype=np.float32)
    costs = np.asarray([float(row["wall_ms"]) / 1000.0 for row in decisions], dtype=np.float32)
    if (
        not np.all(np.isfinite(success_values))
        or not np.all((success_values >= 0) & (success_values <= 1))
        or not np.all(np.isfinite(cost_values))
        or np.any(cost_values < 0)
        or not np.all(np.isfinite(costs))
        or np.any(costs < 0)
    ):
        raise ValueError("Recorded values and RPC costs must be finite and in their supported ranges.")
    rewards = np.zeros(len(decisions), dtype=np.float32)
    rewards[-1] = float(outcome["success"]) * args.gamma ** (int(durations[-1]) - 1)
    success_advantage, success_return = compute_smdp_gae(
        rewards, success_values, durations, gamma=args.gamma, gae_lambda=args.gae_lambda
    )
    cost_advantage, cost_return = compute_smdp_gae(
        costs, cost_values, durations, gamma=args.gamma, gae_lambda=args.gae_lambda
    )
    return durations, success_advantage, cost_advantage, np.clip(success_return, 0.0, 1.0), np.maximum(cost_return, 0.0)


def _load_rollouts(
    args: Args, selector: OrderedSMDPSelector
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = pathlib.Path(args.rollout_dir).resolve()
    config = json.loads((directory / "run_config.json").read_text())
    if config.get("ordered_smdp_sample") is not True:
        raise ValueError("On-policy training requires a rollout collected with ordered_smdp_sample=true.")
    recorded_checkpoint = config.get("ordered_smdp_params")
    if not recorded_checkpoint or pathlib.Path(recorded_checkpoint).resolve() != pathlib.Path(args.input_checkpoint).resolve():
        raise ValueError("Rollout ordered_smdp_params does not match the input checkpoint.")
    if json.loads((directory / "summary.json").read_text()).get("status") != "complete":
        raise ValueError("On-policy training requires a completed rollout batch.")
    outcomes: dict[tuple[int, int], dict[str, str]] = {}
    for row in _read_csv(directory / "rollout_rows.csv"):
        if row["mode"] != "ordered_smdp":
            continue
        key = (int(row["task_id"]), int(row["episode"]))
        if key in outcomes or int(row["success"]) not in {0, 1}:
            raise ValueError("Rollout outcomes must contain unique completed binary-success episodes.")
        outcomes[key] = row
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {key: [] for key in outcomes}
    for row in _read_csv(directory / "decisions.csv"):
        if row["mode"] != "ordered_smdp":
            continue
        key = (int(row["task_id"]), int(row["episode"]))
        if key not in outcomes:
            raise ValueError("A decision belongs to an incomplete episode.")
        grouped[key].append(row)
    entries: list[dict[str, Any]] = []
    episode_rpc: list[float] = []
    for key, outcome in outcomes.items():
        decisions = sorted(grouped[key], key=lambda row: int(row["environment_step"]))
        if len(decisions) != int(outcome["policy_calls"]):
            raise ValueError("Decision rows do not match the completed episode's policy_calls.")
        rpc = float(outcome["policy_rpc_wall_total_ms"]) / 1000.0 if decisions else 0.0
        if not math.isfinite(rpc) or rpc < 0:
            raise ValueError("Episode RPC cost must be finite and non-negative.")
        episode_rpc.append(rpc)
        if not decisions:
            continue
        durations, success_advantage, cost_advantage, success_returns, cost_returns = _episode_targets(
            decisions, outcome, args
        )
        for index, row in enumerate(decisions):
            info = json.loads(row["selector_json"])
            action = int(info["smdp_action_index"])
            if not 0 <= action < len(selector.candidates) or int(row["selected_horizon"]) != selector.candidates[action]:
                raise ValueError("Recorded sampled action does not match selected_horizon.")
            entries.append({
                "feature": info["smdp_feature"],
                "anchor_logits": info["smdp_anchor_logits"],
                "old_probabilities": info["smdp_probabilities"],
                "old_log_prob": info["smdp_old_log_prob"],
                "action": action,
                "duration": durations[index],
                "success_advantage": success_advantage[index],
                "cost_advantage": cost_advantage[index],
                "success_return": success_returns[index],
                "cost_return": cost_returns[index],
            })
    if not entries:
        raise ValueError("The rollout contains no complete sampled ordered_smdp transitions.")
    data = {name: np.asarray([row[name] for row in entries], dtype=np.float32) for name in entries[0]}
    data["action"] = data["action"].astype(np.int32)
    data["duration"] = data["duration"].astype(np.int32)
    count = len(entries)
    expected = {
        "feature": (count, selector.feature_mean.size),
        "anchor_logits": (count, len(selector.candidates) - 1),
        "old_probabilities": (count, len(selector.candidates)),
    }
    if any(data[name].shape != shape for name, shape in expected.items()) or any(
        not np.all(np.isfinite(values)) for values in data.values()
    ):
        raise ValueError("Recorded selector features, probabilities, and targets must have finite expected shapes.")
    old_probabilities = data["old_probabilities"]
    if np.any(old_probabilities < 0) or not np.allclose(old_probabilities.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("Recorded old policy probabilities must be categorical distributions.")
    selected_probability = old_probabilities[np.arange(count), data["action"]]
    if np.any(selected_probability <= 0) or not np.allclose(
        np.log(selected_probability), data["old_log_prob"], atol=1e-5
    ):
        raise ValueError("Recorded sampled log probability does not match the old policy distribution.")
    data["feature"] = ((data["feature"] - selector.feature_mean) / selector.feature_std).astype(np.float32)
    previous_multiplier = float(selector.metadata.get("cost_multiplier", 0.02))
    if not math.isfinite(previous_multiplier) or not 0 <= previous_multiplier <= args.max_cost_multiplier:
        raise ValueError("Checkpoint cost_multiplier must lie in the configured dual range.")
    mean_rpc = float(np.mean(episode_rpc))
    multiplier = float(np.clip(
        previous_multiplier + args.dual_learning_rate * (mean_rpc - args.rpc_budget_seconds),
        0.0, args.max_cost_multiplier,
    ))
    combined = data["success_advantage"] - multiplier * data["cost_advantage"]
    data["advantage"] = ((combined - combined.mean()) / max(float(combined.std()), 1e-6)).astype(np.float32)
    statistics = {
        "num_episodes": len(outcomes), "num_transitions": count,
        "rollout_success_count": sum(int(row["success"]) for row in outcomes.values()),
        "rollout_success_rate": float(np.mean([int(row["success"]) for row in outcomes.values()])),
        "mean_episode_rpc_seconds": mean_rpc,
        "cost_multiplier_before": previous_multiplier, "cost_multiplier_after": multiplier,
        "combined_advantage_mean": float(combined.mean()), "combined_advantage_std": float(combined.std()),
        "observed_duration_min": int(data["duration"].min()), "observed_duration_max": int(data["duration"].max()),
    }
    return data, statistics


def _ordered_log_probabilities(logits: jax.Array) -> jax.Array:
    log_continue = -jax.nn.softplus(-logits)
    prefix = jnp.concatenate([jnp.zeros_like(logits[..., :1]), jnp.cumsum(log_continue, axis=-1)], axis=-1)
    log_mass = jnp.concatenate([prefix[..., :-1] - jax.nn.softplus(logits), prefix[..., -1:]], axis=-1)
    return jax.nn.log_softmax(log_mass, axis=-1)


def _forward(params: dict[str, jax.Array], features: jax.Array, anchor_logits: jax.Array) -> dict[str, jax.Array]:
    logits = anchor_logits + features @ params["actor_w"] + params["actor_b"]
    critic = jnp.tanh(features @ params["critic_w"] + params["critic_b"])
    values = critic @ params["value_w"] + params["value_b"]
    return {
        "log_probabilities": _ordered_log_probabilities(logits),
        "success_value": jax.nn.sigmoid(values[..., 0]),
        "cost_value": jax.nn.softplus(values[..., 1]),
    }


def _loss(params: dict[str, jax.Array], batch: dict[str, jax.Array], args: Args):
    outputs = _forward(params, batch["feature"], batch["anchor_logits"])
    log_probabilities = outputs["log_probabilities"]
    selected_log_prob = jnp.take_along_axis(log_probabilities, batch["action"][:, None], axis=-1)[:, 0]
    ratio = jnp.exp(selected_log_prob - batch["old_log_prob"])
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * batch["advantage"],
        jnp.clip(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * batch["advantage"],
    ))
    anchor_log_prob = _ordered_log_probabilities(batch["anchor_logits"])
    anchor_kl = jnp.mean(jnp.sum(jnp.exp(anchor_log_prob) * (anchor_log_prob - log_probabilities), axis=-1))
    old_probabilities = batch["old_probabilities"]
    old_policy_kl = jnp.mean(jnp.sum(
        old_probabilities * (jnp.log(jnp.maximum(old_probabilities, 1e-30)) - log_probabilities), axis=-1
    ))
    success_value_mse = jnp.mean(jnp.square(outputs["success_value"] - batch["success_return"]))
    cost_value_huber = jnp.mean(optax.huber_loss(outputs["cost_value"], batch["cost_return"]))
    loss = (
        policy_loss + args.anchor_kl_weight * anchor_kl
        + args.success_value_weight * success_value_mse + args.cost_value_weight * cost_value_huber
    )
    return loss, {
        "loss": loss, "policy_loss": policy_loss,
        "success_value_mse": success_value_mse, "cost_value_huber": cost_value_huber,
        "anchor_kl": anchor_kl, "old_policy_kl": old_policy_kl,
        "ratio_mean": jnp.mean(ratio), "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > args.clip_ratio),
        "max_old_probability_error": jnp.max(jnp.abs(jnp.exp(log_probabilities) - old_probabilities)),
    }


def main(args: Args) -> None:
    _validate_args(args)
    started = time.monotonic()
    checkpoint = pathlib.Path(args.input_checkpoint).resolve()
    selector = OrderedSMDPSelector.load(checkpoint)
    if np.any(selector.feature_mean != 0) or np.any(selector.feature_std != 1):
        raise ValueError("This experiment requires fixed feature_mean=0 and feature_std=1.")
    data, statistics = _load_rollouts(args, selector)
    output = pathlib.Path(args.output_dir).resolve()
    if any((output / name).exists() for name in ("selector.npz", "metrics.jsonl", "summary.json")):
        raise FileExistsError("Training output already contains selector artifacts; use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    params = {name: jnp.asarray(value) for name, value in selector.params.items()}
    labels = {name: "actor" if name.startswith("actor_") else "critic" for name in params}
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({
            "actor": optax.adam(args.learning_rate), "critic": optax.adam(args.critic_learning_rate),
        }, labels),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state, batch):
        (_, _), gradients = jax.value_and_grad(_loss, has_aux=True)(current_params, batch, args)
        updates, next_state = optimizer.update(gradients, current_state, current_params)
        return optax.apply_updates(current_params, updates), next_state, optax.global_norm(gradients)

    full_batch = {name: jnp.asarray(value) for name, value in data.items()}

    @jax.jit
    def evaluate(current_params):
        return _loss(current_params, full_batch, args)[1]

    def read_metrics(current_params):
        return {name: float(value) for name, value in jax.device_get(evaluate(current_params)).items()}

    initial_metrics = read_metrics(params)
    if not all(math.isfinite(value) for value in initial_metrics.values()) or initial_metrics["max_old_probability_error"] > 1e-5:
        raise ValueError("Input checkpoint does not reproduce the recorded on-policy distribution.")
    accepted_params = params
    accepted_optimizer_state = optimizer_state
    accepted_metrics = initial_metrics
    accepted_epochs = 0
    rejected_epoch = None
    stop_reason = "epochs_complete"
    rng = np.random.default_rng(args.seed)
    attempted_steps = 0
    with (output / "metrics.jsonl").open("x") as metrics_file:
        metrics_file.write(json.dumps({"epoch": 0, "accepted": True, "finite": True, **initial_metrics}) + "\n")
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(statistics["num_transitions"])
            gradient_finite = True
            max_gradient_norm = 0.0
            for begin in range(0, len(order), args.batch_size):
                batch = {name: jnp.asarray(value[order[begin : begin + args.batch_size]]) for name, value in data.items()}
                params, optimizer_state, gradient_norm = train_step(params, optimizer_state, batch)
                attempted_steps += 1
                gradient_norm = float(gradient_norm)
                gradient_finite &= math.isfinite(gradient_norm)
                if not gradient_finite:
                    break
                max_gradient_norm = max(max_gradient_norm, gradient_norm)
            metrics = read_metrics(params)
            finite = gradient_finite and all(math.isfinite(value) for value in metrics.values())
            accepted = finite and metrics["old_policy_kl"] <= args.target_kl
            record = {
                "epoch": epoch, "accepted": accepted, "finite": finite,
                "gradient_norm_max": max_gradient_norm, **metrics,
            }
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if not accepted:
                rejected_epoch = epoch
                stop_reason = "target_kl_rollback" if finite else "nonfinite_rollback"
                params = accepted_params
                optimizer_state = accepted_optimizer_state
                break
            accepted_params = params
            accepted_optimizer_state = optimizer_state
            accepted_metrics = metrics
            accepted_epochs = epoch
    selector.params = {name: np.asarray(value) for name, value in jax.device_get(accepted_params).items()}
    selector.metadata.update({
        "cost_multiplier": statistics["cost_multiplier_after"],
        "input_checkpoint": str(checkpoint), "training_rollout_dir": str(pathlib.Path(args.rollout_dir).resolve()),
        "accepted_epochs": accepted_epochs, "old_policy_kl": accepted_metrics["old_policy_kl"],
    })
    selector.save(output / "selector.npz")
    summary = {
        "status": "complete", "algorithm": "ordered_smdp_ppo", "finite": True,
        "base_policy_frozen": True, "encoder_frozen": True, "feature_normalization_frozen": True,
        "input_checkpoint": str(checkpoint), "rollout_dir": str(pathlib.Path(args.rollout_dir).resolve()),
        "selector_params": str(output / "selector.npz"), "config": dataclasses.asdict(args),
        "requested_epochs": args.epochs, "accepted_epochs": accepted_epochs, "rejected_epoch": rejected_epoch,
        "update_accepted": accepted_epochs > 0, "rolled_back": rejected_epoch is not None, "stop_reason": stop_reason,
        "optimizer_steps_attempted": attempted_steps, "old_policy_kl": accepted_metrics["old_policy_kl"],
        "initial_metrics": initial_metrics, "final_metrics": accepted_metrics,
        "elapsed_seconds": time.monotonic() - started, **statistics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
