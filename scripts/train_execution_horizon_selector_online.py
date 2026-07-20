"""Online selector update from closed-loop LIBERO rollout journals.

Two deliberately small updates are supported:

* ``q_distill``: fit the chosen-H success critic to online episode outcomes,
  retain the SFT policy with a KL anchor, and redistill the actor from Q.
* ``ppo``: update only the actor and state-value heads with chunk-duration
  (SMDP) GAE, PPO clipping, and KL regularization to the SFT actor.

The ACoT-VLA policy, V2-P encoder, and all action-generation parameters remain
frozen in both cases.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.execution_horizon import rl_selector


@dataclasses.dataclass(frozen=True)
class Args:
    selector_params: str
    rollout_dirs: tuple[str, ...]
    output_dir: str
    algorithm: Literal["q_distill", "ppo"]
    offline_replay: str | None = None
    offline_replay_weight: float = 0.5
    rollout_mode: str = "sft_selector"
    seed: int = 7
    steps: int = 750
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.005
    anchor_kl_weight: float = 0.10
    parameter_anchor_weight: float = 1e-4
    online_q_weight: float = 1.0
    q_actor_distillation_weight: float = 0.5
    efficiency_reward_weight: float = 0.10
    efficiency_call_reference: float = 80.0
    selector_minimum_success_probability: float = 0.5
    selector_reference_slack: float = 0.05
    selector_q_tie_margin: float = 0.03
    log_interval: int = 100


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _load_rollouts(
    args: Args,
    selector: rl_selector.FrozenFeatureSelector,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    transitions: list[dict[str, object]] = []
    episode_outcomes: list[float] = []
    for item in args.rollout_dirs:
        directory = pathlib.Path(item)
        rollout_rows = _read_csv(directory / "rollout_rows.csv")
        decision_rows = _read_csv(directory / "decisions.csv")
        outcomes = {
            (row["mode"], int(row["task_id"]), int(row["episode"])): row
            for row in rollout_rows
            if row["mode"] == args.rollout_mode
        }
        grouped: dict[tuple[str, int, int], list[dict[str, str]]] = {}
        for row in decision_rows:
            key = (row["mode"], int(row["task_id"]), int(row["episode"]))
            if key in outcomes:
                grouped.setdefault(key, []).append(row)
        for key, decisions in grouped.items():
            decisions.sort(key=lambda row: int(row["environment_step"]))
            outcome = outcomes[key]
            success = float(outcome["success"])
            calls = float(outcome["policy_calls"])
            terminal_reward = success * (
                1.0 - args.efficiency_reward_weight * min(calls / max(args.efficiency_call_reference, 1.0), 1.0)
            )
            old_values: list[float] = []
            parsed: list[dict[str, object]] = []
            for decision in decisions:
                info = json.loads(decision["selector_json"])
                if "selector_feature" not in info:
                    raise ValueError(
                        f"{directory}/decisions.csv does not contain selector features. "
                        "Collect with --record-selector-features."
                    )
                action_index = int(info["selector_action_index"])
                if action_index < 0 or action_index >= len(selector.candidates):
                    raise ValueError(f"Invalid selector action index {action_index}.")
                entry = {
                    "feature": np.asarray(info["selector_feature"], dtype=np.float32),
                    "action": action_index,
                    "old_log_prob": float(info["selector_old_log_prob"]),
                    "old_value": float(info["selector_value"]),
                    "eligible": np.asarray(info["selector_eligible"], dtype=np.bool_),
                    "anchor_logits": np.asarray(info["selector_actor_logits"], dtype=np.float32),
                    "duration": int(decision["execution_horizon"]),
                    "success": success,
                }
                parsed.append(entry)
                old_values.append(float(entry["old_value"]))

            advantages = np.zeros(len(parsed), dtype=np.float32)
            next_advantage = 0.0
            for index in range(len(parsed) - 1, -1, -1):
                reward = terminal_reward if index == len(parsed) - 1 else 0.0
                next_value = old_values[index + 1] if index + 1 < len(parsed) else 0.0
                discount = args.gamma ** int(parsed[index]["duration"])
                delta = reward + discount * next_value - old_values[index]
                next_advantage = delta + discount * args.gae_lambda * next_advantage
                advantages[index] = next_advantage
            returns = advantages + np.asarray(old_values, dtype=np.float32)
            remaining_calls = np.arange(len(parsed), 0, -1, dtype=np.float32) / 100.0
            for index, entry in enumerate(parsed):
                transitions.append(
                    {
                        **entry,
                        "advantage": advantages[index],
                        "return": np.clip(returns[index], 0.0, 1.0),
                        "remaining_calls": remaining_calls[index],
                    }
                )
            episode_outcomes.append(success)

    if not transitions:
        raise ValueError(f"No {args.rollout_mode!r} decisions were found in {args.rollout_dirs}.")
    feature = np.stack([np.asarray(item["feature"]) for item in transitions])
    if feature.shape[1] != selector.feature_mean.shape[0]:
        raise ValueError(
            f"Rollout feature dim {feature.shape[1]} does not match selector dim {selector.feature_mean.shape[0]}."
        )
    normalized = (feature - selector.feature_mean) / selector.feature_std
    advantages = np.asarray([float(item["advantage"]) for item in transitions], dtype=np.float32)
    advantage_std = max(float(np.std(advantages)), 1e-6)
    normalized_advantages = (advantages - np.mean(advantages)) / advantage_std
    data = {
        "feature": normalized.astype(np.float32),
        "action": np.asarray([int(item["action"]) for item in transitions], dtype=np.int32),
        "old_log_prob": np.asarray([float(item["old_log_prob"]) for item in transitions], dtype=np.float32),
        "old_value": np.asarray([float(item["old_value"]) for item in transitions], dtype=np.float32),
        "eligible": np.stack([np.asarray(item["eligible"]) for item in transitions]),
        "anchor_logits": np.stack([np.asarray(item["anchor_logits"]) for item in transitions]),
        "success": np.asarray([float(item["success"]) for item in transitions], dtype=np.float32),
        "advantage": normalized_advantages.astype(np.float32),
        "return": np.asarray([float(item["return"]) for item in transitions], dtype=np.float32),
        "remaining_calls": np.asarray([float(item["remaining_calls"]) for item in transitions], dtype=np.float32),
    }
    stats = {
        "num_episodes": float(len(episode_outcomes)),
        "num_transitions": float(len(transitions)),
        "rollout_success_rate": float(np.mean(episode_outcomes)),
        "raw_advantage_mean": float(np.mean(advantages)),
        "raw_advantage_std": float(np.std(advantages)),
    }
    return data, stats


def _load_offline_replay(
    path: str | None,
    *,
    feature_dim: int,
    num_actions: int,
) -> dict[str, np.ndarray]:
    if path is None:
        return {
            "feature": np.zeros((1, feature_dim), dtype=np.float32),
            "success": np.zeros((1, num_actions), dtype=np.float32),
            "cost": np.zeros((1, num_actions), dtype=np.float32),
            "label_weight": np.zeros((1, num_actions), dtype=np.float32),
        }
    with np.load(path, allow_pickle=False) as archive:
        result = {
            name: np.asarray(archive[name], dtype=np.float32) for name in ("feature", "success", "cost", "label_weight")
        }
    if result["feature"].shape[1] != feature_dim:
        raise ValueError(
            f"Offline replay feature dim {result['feature'].shape[1]} does not match selector dim {feature_dim}."
        )
    if result["success"].shape[1] != num_actions:
        raise ValueError("Offline replay action dimension does not match selector candidates.")
    return result


def _forward(params: dict[str, jax.Array], feature: jax.Array) -> dict[str, jax.Array]:
    hidden = jnp.tanh(feature @ params["trunk_w"] + params["trunk_b"])
    return {
        "actor_logits": hidden @ params["actor_w"] + params["actor_b"],
        "q_success_logits": hidden @ params["q_success_w"] + params["q_success_b"],
        "q_cost": jax.nn.softplus(hidden @ params["q_cost_w"] + params["q_cost_b"]),
        "value_logits": (hidden @ params["value_w"] + params["value_b"])[..., 0],
    }


def _masked_log_probabilities(logits: jax.Array, eligible: jax.Array) -> jax.Array:
    masked_logits = jnp.where(eligible, logits, -1e9)
    return jax.nn.log_softmax(masked_logits, axis=-1)


def _bce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _q_targets(
    q_success_probability: jax.Array,
    q_cost: jax.Array,
    *,
    reference_index: int,
    args: Args,
) -> jax.Array:
    reference = q_success_probability[:, reference_index]
    threshold = jnp.maximum(args.selector_minimum_success_probability, reference - args.selector_reference_slack)
    eligible = q_success_probability >= threshold[:, None]
    eligible = eligible.at[:, reference_index].set(True)
    best_success = jnp.max(jnp.where(eligible, q_success_probability, -jnp.inf), axis=-1)
    near_best = eligible & (q_success_probability >= best_success[:, None] - args.selector_q_tie_margin)
    return jnp.argmin(jnp.where(near_best, q_cost, jnp.inf), axis=-1)


def _make_train_step(
    args: Args,
    *,
    initial_params: dict[str, jax.Array],
    reference_index: int,
):
    if args.algorithm == "q_distill":
        trainable = {"actor_w", "actor_b", "q_success_w", "q_success_b", "value_w", "value_b"}
    else:
        trainable = {"actor_w", "actor_b", "value_w", "value_b"}

    def loss_fn(
        params: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        offline_batch: dict[str, jax.Array],
    ):
        outputs = _forward(params, batch["feature"])
        log_probabilities = _masked_log_probabilities(outputs["actor_logits"], batch["eligible"])
        selected_log_probability = jnp.take_along_axis(log_probabilities, batch["action"][:, None], axis=-1)[:, 0]
        anchor_log_probabilities = _masked_log_probabilities(batch["anchor_logits"], batch["eligible"])
        anchor_probabilities = jnp.exp(anchor_log_probabilities)
        anchor_kl = jnp.mean(
            jnp.sum(
                anchor_probabilities * (anchor_log_probabilities - log_probabilities),
                axis=-1,
            )
        )
        value = jax.nn.sigmoid(outputs["value_logits"])
        value_loss = jnp.mean((value - batch["return"]) ** 2)
        probabilities = jnp.exp(log_probabilities)
        entropy = -jnp.mean(jnp.sum(probabilities * log_probabilities, axis=-1))

        if args.algorithm == "ppo":
            ratio = jnp.exp(selected_log_probability - batch["old_log_prob"])
            unclipped = ratio * batch["advantage"]
            clipped = jnp.clip(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * batch["advantage"]
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            q_online_bce = jnp.asarray(0.0)
            distillation_ce = jnp.asarray(0.0)
            loss = (
                policy_loss
                + args.value_weight * value_loss
                + args.anchor_kl_weight * anchor_kl
                - args.entropy_weight * entropy
            )
            clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > args.ppo_clip)
        else:
            selected_q_logits = jnp.take_along_axis(outputs["q_success_logits"], batch["action"][:, None], axis=-1)[
                :, 0
            ]
            q_online_bce = jnp.mean(_bce(selected_q_logits, batch["success"]))
            q_target = _q_targets(
                jax.lax.stop_gradient(jax.nn.sigmoid(outputs["q_success_logits"])),
                jax.lax.stop_gradient(outputs["q_cost"]),
                reference_index=reference_index,
                args=args,
            )
            full_actor_log_probabilities = jax.nn.log_softmax(outputs["actor_logits"], axis=-1)
            distillation_ce = -jnp.mean(jnp.take_along_axis(full_actor_log_probabilities, q_target[:, None], axis=-1))
            offline_outputs = _forward(params, offline_batch["feature"])
            offline_weight = offline_batch["label_weight"]
            offline_q_bce = jnp.sum(
                _bce(offline_outputs["q_success_logits"], offline_batch["success"]) * offline_weight
            ) / jnp.maximum(jnp.sum(offline_weight), 1.0)
            offline_cost_huber = jnp.sum(
                optax.huber_loss(offline_outputs["q_cost"], offline_batch["cost"]) * offline_weight
            ) / jnp.maximum(jnp.sum(offline_weight), 1.0)
            policy_loss = distillation_ce
            loss = (
                args.online_q_weight * q_online_bce
                + args.q_actor_distillation_weight * distillation_ce
                + args.offline_replay_weight * (offline_q_bce + 0.25 * offline_cost_huber)
                + args.value_weight * value_loss
                + args.anchor_kl_weight * anchor_kl
                - args.entropy_weight * entropy
            )
            clip_fraction = jnp.asarray(0.0)
        if args.algorithm == "ppo":
            offline_q_bce = jnp.asarray(0.0)
            offline_cost_huber = jnp.asarray(0.0)

        parameter_anchor = sum(jnp.sum((params[name] - initial_params[name]) ** 2) for name in trainable)
        loss += args.parameter_anchor_weight * parameter_anchor
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "q_online_bce": q_online_bce,
            "offline_q_bce": offline_q_bce,
            "offline_cost_huber": offline_cost_huber,
            "distillation_ce": distillation_ce,
            "value_mse": value_loss,
            "anchor_kl": anchor_kl,
            "entropy": entropy,
            "clip_fraction": clip_fraction,
            "parameter_anchor": parameter_anchor,
            "selected_log_probability": jnp.mean(selected_log_probability),
        }

    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)

    @jax.jit
    def train_step(
        params: dict[str, jax.Array],
        optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
        offline_batch: dict[str, jax.Array],
    ):
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, offline_batch)
        gradients = {
            name: gradient if name in trainable else jnp.zeros_like(gradient) for name, gradient in gradients.items()
        }
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        updated_params = optax.apply_updates(params, updates)
        # AdamW applies decoupled weight decay even to zero gradients, so copy
        # non-trainable leaves back explicitly to keep the frozen boundary real.
        params = {name: updated_params[name] if name in trainable else params[name] for name in params}
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return params, optimizer_state, metrics

    return optimizer, train_step


def _audit(
    params: dict[str, jax.Array],
    data: dict[str, np.ndarray],
    *,
    selector: rl_selector.FrozenFeatureSelector,
) -> dict[str, float | dict[str, int]]:
    outputs = jax.device_get(_forward(params, jnp.asarray(data["feature"])))
    log_probabilities = np.asarray(
        _masked_log_probabilities(
            jnp.asarray(outputs["actor_logits"]),
            jnp.asarray(data["eligible"]),
        )
    )
    choices = np.argmax(log_probabilities, axis=-1)
    selected_q = np.take_along_axis(
        np.asarray(jax.nn.sigmoid(outputs["q_success_logits"])),
        data["action"][:, None],
        axis=-1,
    )[:, 0]
    return {
        "deterministic_actor_h_distribution": {
            str(selector.candidates[index]): int(np.sum(choices == index)) for index in range(len(selector.candidates))
        },
        "selected_q_online_brier": float(np.mean((selected_q - data["success"]) ** 2)),
        "value_mse": float(np.mean((np.asarray(jax.nn.sigmoid(outputs["value_logits"])) - data["return"]) ** 2)),
    }


def main(args: Args) -> None:
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Online selector update is already complete: {summary_path}")
    started = time.perf_counter()
    selector = rl_selector.FrozenFeatureSelector.load(args.selector_params)
    data, rollout_stats = _load_rollouts(args, selector)
    offline_data = _load_offline_replay(
        args.offline_replay,
        feature_dim=selector.feature_mean.shape[0],
        num_actions=len(selector.candidates),
    )
    params = {name: jnp.asarray(value) for name, value in selector.params.items()}
    initial_params = {name: jnp.array(value) for name, value in params.items()}
    reference_index = selector.reference_index
    optimizer, train_step = _make_train_step(
        args,
        initial_params=initial_params,
        reference_index=reference_index,
    )
    optimizer_state = optimizer.init(params)
    rng = np.random.default_rng(args.seed)
    last_metrics: dict[str, float] = {}
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
        for step in range(args.steps):
            selected = rng.choice(len(data["feature"]), size=args.batch_size, replace=True)
            batch = {name: jnp.asarray(value[selected]) for name, value in data.items()}
            offline_selected = rng.choice(
                len(offline_data["feature"]),
                size=args.batch_size,
                replace=True,
            )
            offline_batch = {name: jnp.asarray(value[offline_selected]) for name, value in offline_data.items()}
            params, optimizer_state, metrics = train_step(
                params,
                optimizer_state,
                batch,
                offline_batch,
            )
            if step % args.log_interval == 0 or step + 1 == args.steps:
                last_metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
                record = {"step": step + 1, "algorithm": args.algorithm, **last_metrics}
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)

    updated = rl_selector.copy_selector(
        selector,
        params={name: np.asarray(value) for name, value in jax.device_get(params).items()},
        metadata_updates={
            "algorithm": args.algorithm,
            "online_update": True,
            "online_rollout_mode": args.rollout_mode,
            "online_rollout_dirs": list(args.rollout_dirs),
            "base_policy_frozen": True,
            "v2p_encoder_frozen": True,
        },
    )
    filename = "selector_q_online.npz" if args.algorithm == "q_distill" else "selector_ppo.npz"
    selector_path = updated.save(output_dir / filename)
    summary = {
        "status": "complete",
        "algorithm": args.algorithm,
        "base_policy_loaded": False,
        "base_policy_frozen": True,
        "v2p_encoder_frozen": True,
        "selector_params": str(selector_path.resolve()),
        "rollout_stats": rollout_stats,
        "last_train_metrics": last_metrics,
        "online_audit": _audit(params, data, selector=selector),
        "elapsed_seconds": time.perf_counter() - started,
        "config": dataclasses.asdict(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
