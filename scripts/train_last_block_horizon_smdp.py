"""Warm up a Monte Carlo critic, then update the final ordered predictor block."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import train_ordered_horizon_smdp as base_trainer
import tyro

from openpi.execution_horizon import last_block_smdp as last_block
from openpi.execution_horizon.ordered_smdp import compute_smdp_gae


@dataclasses.dataclass(frozen=True)
class Args(base_trainer.Args):
    learning_rate: float = 2e-5
    batch_size: int = 64
    critic_warmup_epochs: int = 10


def _partition(params: nnx.State, roots: tuple[str, ...]) -> nnx.State:
    flat = {path: value for path, value in params.flat_state().items() if path[0] in roots}
    return nnx.State(traverse_util.unflatten_dict(flat))


def _replace_roots(params: nnx.State, replacement: nnx.State) -> nnx.State:
    replacement_flat = replacement.flat_state()
    roots = {path[0] for path in replacement_flat}
    flat = {path: value for path, value in params.flat_state().items() if path[0] not in roots}
    flat.update(replacement_flat)
    return nnx.State(traverse_util.unflatten_dict(flat))


def _actor_changed(before: nnx.State, after: nnx.State) -> bool:
    first = jax.tree.leaves(_partition(before, last_block.ACTOR_ROOTS))
    second = jax.tree.leaves(_partition(after, last_block.ACTOR_ROOTS))
    return any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second, strict=True))


def _mc_returns(durations: np.ndarray, rpc_costs: np.ndarray, success: int, gamma: float):
    remaining_steps = np.cumsum(durations[::-1])[::-1]
    success_returns = success * np.power(gamma, remaining_steps - 1)
    cost_returns = np.empty_like(rpc_costs, dtype=np.float64)
    following_cost = 0.0
    for index in range(len(durations) - 1, -1, -1):
        following_cost = rpc_costs[index] + gamma ** int(durations[index]) * following_cost
        cost_returns[index] = following_cost
    return success_returns.astype(np.float32), cost_returns.astype(np.float32)


def _load_rollouts(args: Args, selector: last_block.LastBlockHorizonSelector):
    directory = pathlib.Path(args.rollout_dir).resolve()
    config = json.loads((directory / "run_config.json").read_text())
    if config.get("last_block_smdp_sample") is not True:
        raise ValueError("Training requires last_block_smdp_sample=true in the source rollout.")
    source = config.get("last_block_smdp_params")
    if not source or pathlib.Path(source).resolve() != pathlib.Path(args.input_checkpoint).resolve():
        raise ValueError("Rollout last_block_smdp_params differs from the input checkpoint.")
    if json.loads((directory / "summary.json").read_text()).get("status") != "complete":
        raise ValueError("Training requires a completed sampled rollout batch.")
    mode = "ordered_smdp_last_block"
    outcomes = {}
    for row in base_trainer._read_csv(directory / "rollout_rows.csv"):  # noqa: SLF001
        if row["mode"] != mode:
            continue
        key = (int(row["task_id"]), int(row["episode"]))
        if key in outcomes or int(row["success"]) not in {0, 1}:
            raise ValueError("Completed rollout episodes must be unique with binary success.")
        outcomes[key] = row
    grouped = {key: [] for key in outcomes}
    for row in base_trainer._read_csv(directory / "decisions.csv"):  # noqa: SLF001
        if row["mode"] != mode:
            continue
        key = (int(row["task_id"]), int(row["episode"]))
        if key not in grouped:
            raise ValueError("A cached decision has no completed episode outcome.")
        grouped[key].append(row)
    entries = []
    episode_costs = []
    for episode_index, (key, outcome) in enumerate(outcomes.items()):
        decisions = sorted(grouped[key], key=lambda row: int(row["environment_step"]))
        if not decisions or len(decisions) != int(outcome["policy_calls"]):
            raise ValueError("Each training episode must contain its complete sampled decision sequence.")
        steps = np.asarray([int(row["environment_step"]) for row in decisions], dtype=np.int64)
        durations = np.diff(np.r_[steps, int(outcome["steps"])])
        costs = np.asarray([float(row["wall_ms"]) / 1000 for row in decisions])
        if np.any(durations <= 0) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("Observed durations and RPC costs must be positive-duration, finite transitions.")
        episode_cost = float(outcome["policy_rpc_wall_total_ms"]) / 1000
        if not math.isfinite(episode_cost) or episode_cost < 0:
            raise ValueError("Completed episode RPC cost must be finite and non-negative.")
        episode_costs.append(episode_cost)
        success_mc, cost_mc = _mc_returns(durations, costs, int(outcome["success"]), args.gamma)
        for index, row in enumerate(decisions):
            info = json.loads(row["selector_json"])
            action = int(info["smdp_action_index"])
            if not 0 <= action < len(selector.candidates) or int(row["selected_horizon"]) != selector.candidates[action]:
                raise ValueError("Recorded sampled action and selected horizon differ.")
            entries.append({
                "tokens": info["smdp_last_block_input"], "context": info["smdp_last_block_context"],
                "anchor_logits": info["smdp_anchor_logits"], "old_probabilities": info["smdp_probabilities"],
                "old_log_prob": info["smdp_old_log_prob"], "action": action,
                "duration": durations[index], "rpc_cost": costs[index], "success": int(outcome["success"]),
                "success_mc": success_mc[index], "cost_mc": cost_mc[index],
                "episode_index": episode_index, "task_id": key[0], "episode_id": key[1],
            })
    if not entries:
        raise ValueError("No complete sampled last-block rollout transitions were found.")
    data = {name: np.asarray([row[name] for row in entries], dtype=np.float32) for name in entries[0]}
    for name in ("action", "duration", "episode_index", "task_id", "episode_id"):
        data[name] = data[name].astype(np.int32)
    n, h = len(entries), selector.config.hidden_dim
    expected = {
        "tokens": (n, selector.config.action_horizon + selector.config.visual_num_queries, h),
        "context": (n, h), "anchor_logits": (n, len(selector.candidates) - 1),
        "old_probabilities": (n, len(selector.candidates)),
    }
    if any(data[name].shape != shape for name, shape in expected.items()) or any(
        not np.all(np.isfinite(value)) for value in data.values()
    ):
        raise ValueError("Cached tokens, context, probabilities and targets must have finite expected shapes.")
    probabilities = data["old_probabilities"]
    selected = probabilities[np.arange(n), data["action"]]
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("Recorded policy probabilities must form categorical distributions.")
    if np.any(selected <= 0) or not np.allclose(np.log(selected), data["old_log_prob"], atol=1e-5):
        raise ValueError("Recorded sampled log probabilities do not match the behavior policy.")
    old_multiplier = float(selector.metadata.get("cost_multiplier", 0.02))
    if not math.isfinite(old_multiplier) or not 0 <= old_multiplier <= args.max_cost_multiplier:
        raise ValueError("Input cost_multiplier is outside the configured range.")
    mean_rpc = float(np.mean(episode_costs))
    multiplier = float(np.clip(
        old_multiplier + args.dual_learning_rate * (mean_rpc - args.rpc_budget_seconds), 0, args.max_cost_multiplier
    ))
    statistics = {
        "num_episodes": len(outcomes), "num_transitions": n,
        "rollout_success_count": sum(int(row["success"]) for row in outcomes.values()),
        "rollout_success_rate": float(np.mean([int(row["success"]) for row in outcomes.values()])),
        "mean_episode_rpc_seconds": mean_rpc, "cost_multiplier_before": old_multiplier,
        "cost_multiplier_after": multiplier, "observed_duration_min": int(data["duration"].min()),
        "observed_duration_max": int(data["duration"].max()),
    }
    return data, statistics


def _critic_split(data: dict[str, np.ndarray], seed: int):
    rng = np.random.default_rng(seed)
    heldout_groups = []
    for task in np.unique(data["task_id"]):
        groups = np.unique(data["episode_index"][data["task_id"] == task])
        if len(groups) < 2:
            raise ValueError("Critic warmup needs at least two complete episodes per task for an isolated holdout.")
        rng.shuffle(groups)
        heldout_groups.extend(groups[:max(1, round(0.2 * len(groups)))].tolist())
    heldout = np.isin(data["episode_index"], heldout_groups)
    return np.flatnonzero(~heldout), np.flatnonzero(heldout)


def _critic_loss(outputs, batch, *, cost_weight: float):
    bce = jnp.mean(optax.sigmoid_binary_cross_entropy(outputs["success_logits"], batch["success_mc"]))
    brier = jnp.mean(jnp.square(outputs["success_value"] - batch["success_mc"]))
    cost_huber = jnp.mean(optax.huber_loss(outputs["cost_value"], batch["cost_mc"]))
    return bce + cost_weight * cost_huber, {
        "mc_success_bce": bce, "mc_success_brier": brier, "mc_cost_huber": cost_huber,
        "mc_objective": bce + cost_weight * cost_huber,
    }


def _constant_metrics(data, indices, probability, cost):
    success = data["success_mc"][indices]
    delta = np.abs(cost - data["cost_mc"][indices])
    bce = float(np.mean(-success * np.log(probability) - (1 - success) * np.log1p(-probability)))
    huber = float(np.mean(np.where(delta <= 1.0, 0.5 * delta**2, delta - 0.5)))
    return {
        "mc_success_bce": bce, "mc_success_brier": float(np.mean((probability - success)**2)),
        "mc_cost_huber": huber, "mc_objective": bce + 0.1 * huber,
    }


def _batched_outputs(selector, params, data, batch_size):
    graphdef = selector.graphdef
    apply = jax.jit(lambda p, t, c: last_block.apply_last_block(graphdef, p, t, c))
    chunks = []
    for start in range(0, len(data["tokens"]), batch_size):
        chunks.append(jax.device_get(apply(
            params, jnp.asarray(data["tokens"][start:start + batch_size]),
            jnp.asarray(data["context"][start:start + batch_size]),
        )))
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


def _warm_critic(selector, params, data, frozen_summary, args, metrics_file):
    train_indices, heldout_indices = _critic_split(data, args.seed)
    graphdef = selector.graphdef
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.critic_learning_rate))
    state = optimizer.init(_partition(params, last_block.CRITIC_ROOTS))
    warm_data = {
        "summary": frozen_summary, "success_mc": data["success_mc"], "cost_mc": data["cost_mc"],
    }

    @jax.jit
    def step(current_params, optimizer_state, batch):
        critic_params = _partition(current_params, last_block.CRITIC_ROOTS)

        def loss_fn(critic):
            outputs = last_block.apply_critic(graphdef, _replace_roots(current_params, critic), batch["summary"])
            return _critic_loss(outputs, batch, cost_weight=0.1)[0]

        loss, gradients = jax.value_and_grad(loss_fn)(critic_params)
        updates, next_state = optimizer.update(gradients, optimizer_state, critic_params)
        return _replace_roots(current_params, optax.apply_updates(critic_params, updates)), next_state, loss

    @jax.jit
    def evaluate(current_params, batch):
        outputs = last_block.apply_critic(graphdef, current_params, batch["summary"])
        return _critic_loss(outputs, batch, cost_weight=0.1)[1]

    def read_metrics(current_params, indices):
        batch = {name: jnp.asarray(value[indices]) for name, value in warm_data.items()}
        return {name: float(value) for name, value in jax.device_get(evaluate(current_params, batch)).items()}

    initial_train = read_metrics(params, train_indices)
    initial_heldout = read_metrics(params, heldout_indices)
    if not all(math.isfinite(value) for metrics in (initial_train, initial_heldout) for value in metrics.values()):
        raise ValueError("Initial MC critic metrics must be finite.")
    best_params, best_epoch, best_objective = params, 0, initial_heldout["mc_objective"]
    best_train, best_heldout = initial_train, initial_heldout
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.critic_warmup_epochs + 1):
        if epoch:
            order = rng.permutation(train_indices)
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                batch = {name: jnp.asarray(value[indices]) for name, value in warm_data.items()}
                params, state, loss = step(params, state, batch)
                if not math.isfinite(float(loss)):
                    break
        train_metrics = read_metrics(params, train_indices)
        heldout_metrics = read_metrics(params, heldout_indices)
        finite = all(math.isfinite(value) for m in (train_metrics, heldout_metrics) for value in m.values())
        record = {"phase": "critic_warmup", "epoch": epoch, "finite": finite,
                  "train": train_metrics, "heldout": heldout_metrics}
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        metrics_file.flush()
        print(json.dumps(record, sort_keys=True), flush=True)
        if not finite:
            break
        if heldout_metrics["mc_objective"] < best_objective:
            best_params, best_epoch, best_objective = params, epoch, heldout_metrics["mc_objective"]
            best_train, best_heldout = train_metrics, heldout_metrics
    probability = float(np.clip(data["success_mc"][train_indices].mean(), 1e-6, 1 - 1e-6))
    cost = float(data["cost_mc"][train_indices].mean())
    return best_params, {
        "selected_epoch": best_epoch, "requested_epochs": args.critic_warmup_epochs,
        "num_train_episodes": int(np.unique(data["episode_index"][train_indices]).size),
        "num_heldout_episodes": int(np.unique(data["episode_index"][heldout_indices]).size),
        "train_episode_indices": np.unique(data["episode_index"][train_indices]).tolist(),
        "heldout_episode_indices": np.unique(data["episode_index"][heldout_indices]).tolist(),
        "initial_train": initial_train, "initial_heldout": initial_heldout,
        "selected_train": best_train, "selected_heldout": best_heldout,
        "constant_success_probability": probability, "constant_cost_seconds": cost,
        "constant_train": _constant_metrics(data, train_indices, probability, cost),
        "constant_heldout": _constant_metrics(data, heldout_indices, probability, cost),
        "selection_objective": "mc_success_bce + 0.1 * mc_cost_huber",
        "holdout_scope": "Task-stratified episodes within this sampled training batch only.",
    }


def _advantages(data, success_values, cost_values, args, multiplier):
    success_advantage = np.zeros(len(data["action"]), dtype=np.float32)
    cost_advantage = np.zeros_like(success_advantage)
    for episode in np.unique(data["episode_index"]):
        indices = np.flatnonzero(data["episode_index"] == episode)
        durations = data["duration"][indices]
        rewards = np.zeros(len(indices), dtype=np.float32)
        rewards[-1] = data["success"][indices[-1]] * args.gamma ** (int(durations[-1]) - 1)
        success_advantage[indices] = compute_smdp_gae(
            rewards, success_values[indices], durations, args.gamma, args.gae_lambda
        )[0]
        cost_advantage[indices] = compute_smdp_gae(
            data["rpc_cost"][indices], cost_values[indices], durations, args.gamma, args.gae_lambda
        )[0]
    combined = success_advantage - multiplier * cost_advantage
    normalized = (combined - combined.mean()) / max(float(combined.std()), 1e-6)
    return normalized.astype(np.float32), {
        "combined_advantage_mean": float(combined.mean()), "combined_advantage_std": float(combined.std()),
    }


def _ppo_loss(graphdef, params, batch, args):
    outputs = last_block.apply_last_block(graphdef, params, batch["tokens"], batch["context"])
    log_probabilities = outputs["log_probabilities"]
    selected_log_prob = jnp.take_along_axis(log_probabilities, batch["action"][:, None], axis=-1)[:, 0]
    ratio = jnp.exp(selected_log_prob - batch["old_log_prob"])
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * batch["advantage"], jnp.clip(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * batch["advantage"],
    ))
    anchor_log = base_trainer._ordered_log_probabilities(batch["anchor_logits"])  # noqa: SLF001
    anchor_kl = jnp.mean(jnp.sum(jnp.exp(anchor_log) * (anchor_log - log_probabilities), axis=-1))
    old_probability = batch["old_probabilities"]
    old_kl = jnp.mean(jnp.sum(
        old_probability * (jnp.log(jnp.maximum(old_probability, 1e-30)) - log_probabilities), axis=-1
    ))
    _, critic_metrics = _critic_loss(outputs, batch, cost_weight=0.1)
    loss = (policy_loss + args.anchor_kl_weight * anchor_kl
            + args.success_value_weight * critic_metrics["mc_success_bce"]
            + args.cost_value_weight * critic_metrics["mc_cost_huber"])
    return loss, {
        "loss": loss, "policy_loss": policy_loss, "anchor_kl": anchor_kl, "old_policy_kl": old_kl,
        "ratio_mean": jnp.mean(ratio), "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > args.clip_ratio),
        **critic_metrics,
    }


def _make_ppo_step(graphdef, args, actor_optimizer, critic_optimizer):
    @jax.jit
    def train_step(current_params, state, batch):
        (_, _), gradients = jax.value_and_grad(
            lambda p: _ppo_loss(graphdef, p, batch, args), has_aux=True
        )(current_params)
        norm = optax.global_norm(gradients)
        gradients = jax.tree.map(lambda value: value * jnp.minimum(1.0, 1.0 / jnp.maximum(norm, 1e-12)), gradients)
        actor = _partition(current_params, last_block.ACTOR_ROOTS)
        critic = _partition(current_params, last_block.CRITIC_ROOTS)
        actor_updates, actor_state = actor_optimizer.update(_partition(gradients, last_block.ACTOR_ROOTS), state[0], actor)
        critic_updates, critic_state = critic_optimizer.update(_partition(gradients, last_block.CRITIC_ROOTS), state[1], critic)
        updated = _replace_roots(current_params, optax.apply_updates(actor, actor_updates))
        updated = _replace_roots(updated, optax.apply_updates(critic, critic_updates))
        return updated, (actor_state, critic_state), norm

    return train_step


def main(args: Args) -> None:
    base_trainer._validate_args(args)  # noqa: SLF001
    if args.critic_warmup_epochs < 0:
        raise ValueError("critic_warmup_epochs must be non-negative.")
    started = time.monotonic()
    selector = last_block.LastBlockHorizonSelector.load(args.input_checkpoint)
    initial_params = selector.params
    data, statistics = _load_rollouts(args, selector)
    output = pathlib.Path(args.output_dir).resolve()
    if any((output / name).exists() for name in ("checkpoint", "metrics.jsonl", "summary.json")):
        raise FileExistsError("Training output contains artifacts; use a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    before = _batched_outputs(selector, initial_params, data, args.batch_size)
    if not np.allclose(before["probabilities"], data["old_probabilities"], rtol=1e-5, atol=1e-6):
        raise ValueError("Input last-block actor does not reproduce the sampled behavior policy.")
    with (output / "metrics.jsonl").open("x") as metrics_file:
        params, warmup = _warm_critic(selector, initial_params, data, before["summary"], args, metrics_file)
        if _actor_changed(initial_params, params):
            raise ValueError("Critic-only warmup changed frozen actor parameters.")
        warmed = _batched_outputs(selector, params, data, args.batch_size)
        if not np.array_equal(before["probabilities"], warmed["probabilities"]):
            raise ValueError("Critic warmup changed the behavior-policy distribution.")
        data["advantage"], advantage_statistics = _advantages(
            data, warmed["success_value"], warmed["cost_value"], args, statistics["cost_multiplier_after"]
        )
        graphdef = selector.graphdef
        actor_optimizer = optax.adam(args.learning_rate)
        critic_optimizer = optax.adam(args.critic_learning_rate)
        optimizer_state = (
            actor_optimizer.init(_partition(params, last_block.ACTOR_ROOTS)),
            critic_optimizer.init(_partition(params, last_block.CRITIC_ROOTS)),
        )

        train_step = _make_ppo_step(graphdef, args, actor_optimizer, critic_optimizer)
        evaluate = jax.jit(lambda p, batch: _ppo_loss(graphdef, p, batch, args)[1])

        def read_metrics(current_params):
            totals = {}
            n = len(data["action"])
            for start in range(0, n, args.batch_size):
                batch = {name: jnp.asarray(value[start:start + args.batch_size]) for name, value in data.items()}
                metrics = jax.device_get(evaluate(current_params, batch))
                weight = min(args.batch_size, n - start) / n
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * weight
            return totals

        accepted_params, accepted_state = params, optimizer_state
        initial_metrics = read_metrics(params)
        if not all(math.isfinite(value) for value in initial_metrics.values()):
            raise ValueError("Initial PPO metrics must be finite after critic warmup.")
        accepted_metrics = initial_metrics
        accepted_epochs, attempted_steps, rejected_epoch = 0, 0, None
        stop_reason = "epochs_complete"
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(data["action"]))
            gradient_finite = True
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                batch = {name: jnp.asarray(value[indices]) for name, value in data.items()}
                params, optimizer_state, norm = train_step(params, optimizer_state, batch)
                attempted_steps += 1
                gradient_finite &= math.isfinite(float(norm))
                if not gradient_finite:
                    break
            metrics = read_metrics(params)
            finite = gradient_finite and all(math.isfinite(value) for value in metrics.values())
            accepted = finite and metrics["old_policy_kl"] <= args.target_kl
            record = {"phase": "ppo", "epoch": epoch, "accepted": accepted, "finite": finite, **metrics}
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if not accepted:
                params, optimizer_state = accepted_params, accepted_state
                rejected_epoch = epoch
                stop_reason = "target_kl_rollback" if finite else "nonfinite_rollback"
                break
            accepted_params, accepted_state, accepted_metrics = params, optimizer_state, metrics
            accepted_epochs = epoch
    actor_changed = _actor_changed(initial_params, accepted_params)
    selector.params = accepted_params
    selector.metadata.update({
        "cost_multiplier": statistics["cost_multiplier_after"], "actor_changed": actor_changed,
        "input_checkpoint": str(pathlib.Path(args.input_checkpoint).resolve()),
        "training_rollout_dir": str(pathlib.Path(args.rollout_dir).resolve()),
        "accepted_epochs": accepted_epochs, "critic_warmup_selected_epoch": warmup["selected_epoch"],
    })
    selector.save(output / "checkpoint")
    summary = {
        "status": "complete", "algorithm": "last_block_smdp_ppo_with_mc_critic_warmup", "finite": True,
        "base_policy_frozen": True, "earlier_predictor_layers_frozen": True,
        "actor_trainable_roots": list(last_block.ACTOR_ROOTS), "critic_trainable_roots": list(last_block.CRITIC_ROOTS),
        "actor_changed": actor_changed, "accepted_epochs": accepted_epochs, "requested_epochs": args.epochs,
        "rejected_epoch": rejected_epoch, "rolled_back": rejected_epoch is not None, "stop_reason": stop_reason,
        "optimizer_steps_attempted": attempted_steps, "old_policy_kl": accepted_metrics["old_policy_kl"],
        "selector_params": str(output / "checkpoint"), "config": dataclasses.asdict(args),
        "critic_warmup": warmup, "initial_metrics": initial_metrics, "final_metrics": accepted_metrics,
        "mc_target_semantics": "Finite-episode terminal success and measured cumulative RPC, discounted by actual duration.",
        "critic_gradient_into_actor": False, "elapsed_seconds": time.monotonic() - started,
        **statistics, **advantage_statistics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
