"""Fit a frozen-A feedback residual using paired Monte Carlo branch outcomes."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import json
import pathlib
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.execution_horizon.feedback import FeedbackSelector
from openpi.execution_horizon.ordered_smdp import ordered_log_probabilities
from openpi.models.execution_horizon_predictor import ordered_continuation_distribution

RPC_WEIGHT = 0.02
ANCHOR_KL_WEIGHT = 0.05


@dataclasses.dataclass(frozen=True)
class Args:
    train_dir: str
    validation_dir: str
    a_predictor_dir: str
    output_dir: str
    variant: Literal["current", "history"] = "history"
    seed: int = 7
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_updates: int = 2000
    log_every: int = 25
    patience: int = 8
    selection_metric: Literal["expected_loss", "greedy"] = "expected_loss"


def paired_advantages(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compare each H with the source A choice using only matching valid repeats."""
    success = np.asarray(record["trial_success"], dtype=np.float32)
    rpc = np.asarray(record["trial_rpc"], dtype=np.float32)
    valid = np.asarray(record["trial_valid"], dtype=np.bool_)
    if success.ndim != 2 or success.shape[0] != 5 or success.shape[1] < 1:
        raise ValueError("trial_success must have shape (5, repeats) with at least one repeat.")
    if rpc.shape != success.shape or valid.shape != success.shape:
        raise ValueError("trial_rpc/trial_valid must match trial_success.")
    if (
        not np.all(np.isfinite(success[valid]))
        or not np.all(np.isin(success[valid], [0.0, 1.0]))
        or not np.all(np.isfinite(rpc[valid]))
        or np.any(rpc[valid] < 0)
    ):
        raise ValueError("Valid trials require binary success and finite non-negative RPC seconds.")
    for name in ("trial_elapsed", "trial_calls"):
        values = np.asarray(record[name], dtype=np.float32)
        if values.shape != success.shape or not np.all(np.isfinite(values[valid])) or np.any(values[valid] < 0):
            raise ValueError(f"{name} must match trial_success and contain valid non-negative observations.")
    selected_h = int(np.asarray(record["selected_h"]).item())
    candidates = (5, 10, 15, 20, 25)
    if selected_h not in candidates:
        raise ValueError("selected_h must be the source A choice in H5/10/15/20/25.")
    anchor_index = candidates.index(selected_h)
    paired_valid = valid & valid[anchor_index][None, :]
    paired_count = np.sum(paired_valid, axis=-1)
    if np.any(paired_count == 0):
        raise ValueError("Each candidate requires at least one valid repeat paired with source A.")
    safe_success = np.where(valid, success, 0.0)
    safe_rpc = np.where(valid, rpc, 0.0)
    success_delta = np.sum(
        np.where(paired_valid, safe_success - safe_success[anchor_index], 0.0), axis=-1
    ) / paired_count
    rpc_delta = np.sum(np.where(paired_valid, safe_rpc - safe_rpc[anchor_index], 0.0), axis=-1) / paired_count
    return {
        "advantage": (success_delta - RPC_WEIGHT * rpc_delta).astype(np.float32),
        "success_delta": success_delta.astype(np.float32),
        "rpc_delta_seconds": rpc_delta.astype(np.float32),
        "paired_count": paired_count.astype(np.int32),
    }


def load_roots(
    directory: str | pathlib.Path, selector: FeedbackSelector
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = pathlib.Path(directory).resolve()
    paths = sorted(directory.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No root NPZ files found in {directory}.")
    rows = []
    identities = []
    source_successes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            record = {name: archive[name] for name in archive.files}
        feature = selector.build_features(record)
        labels = paired_advantages(record)
        anchor_h = selector.candidates[int(np.argmax(ordered_log_probabilities(record["continuation_logits"])))]
        if anchor_h != int(np.asarray(record["selected_h"]).item()):
            raise ValueError(f"selected_h differs from the stored greedy A distribution in {path}.")
        task_id = int(np.asarray(record["task_id"]).item())
        episode_id = int(np.asarray(record["episode_id"]).item())
        root_step = int(np.asarray(record["root_step"]).item())
        source_success = float(np.asarray(record["source_success"]).item())
        if source_success not in (0.0, 1.0):
            raise ValueError(f"source_success must be binary in {path}.")
        identities.append((task_id, episode_id, root_step))
        source_successes.append(source_success)
        rows.append({
            "feature": feature,
            "anchor_logits": np.asarray(record["continuation_logits"], dtype=np.float32),
            "history_valid": np.float32(np.asarray(record["history_valid"]).item()),
            **labels,
        })
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate task/episode/root_step roots found in {directory}.")
    data = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    return data, {
        "directory": str(directory), "roots": len(paths), "paths": [str(path) for path in paths],
        "identities": [list(identity) for identity in identities],
        "episode_groups": [list(group) for group in sorted({identity[:2] for identity in identities})],
        "history_valid_roots": int(np.sum(data["history_valid"])),
        "source_successes": int(sum(source_successes)),
        "paired_repeats_min": int(np.min(data["paired_count"])),
        "paired_repeats_max": int(np.max(data["paired_count"])),
    }


def forward(
    params: dict[str, jax.Array], features: jax.Array, anchor_logits: jax.Array, history_valid: jax.Array
) -> dict[str, jax.Array]:
    hidden = jnp.tanh(features @ params["hidden_w"] + params["hidden_b"])
    residual = hidden @ params["output_w"] + params["output_b"]
    logits = anchor_logits + history_valid[..., None] * residual
    log_probabilities, probabilities = ordered_continuation_distribution(logits)
    return {"continuation_logits": logits, "log_probabilities": log_probabilities, "probabilities": probabilities}


def loss(params: dict[str, jax.Array], batch: dict[str, jax.Array]) -> tuple[jax.Array, dict[str, jax.Array]]:
    prediction = forward(params, batch["feature"], batch["anchor_logits"], batch["history_valid"])
    anchor_log_probabilities, anchor_probabilities = ordered_continuation_distribution(batch["anchor_logits"])
    expected_advantage = jnp.mean(jnp.sum(prediction["probabilities"] * batch["advantage"], axis=-1))
    anchor_kl = jnp.mean(jnp.sum(
        anchor_probabilities * (anchor_log_probabilities - prediction["log_probabilities"]), axis=-1
    ))
    objective = -expected_advantage + ANCHOR_KL_WEIGHT * anchor_kl
    return objective, {
        "loss": objective,
        "expected_paired_advantage": expected_advantage,
        "expected_success_delta": jnp.mean(jnp.sum(prediction["probabilities"] * batch["success_delta"], axis=-1)),
        "expected_rpc_delta_seconds": jnp.mean(jnp.sum(
            prediction["probabilities"] * batch["rpc_delta_seconds"], axis=-1
        )),
        "anchor_kl": anchor_kl,
    }


def _metrics_to_python(metrics: dict[str, jax.Array]) -> dict[str, float]:
    result = {name: float(value) for name, value in jax.device_get(metrics).items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Feedback optimization produced non-finite metrics.")
    return result


def greedy_validation_metrics(
    selector: FeedbackSelector,
    params: dict[str, Any],
    raw_features: np.ndarray,
    validation: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Score deployment NumPy argmax choices against root-equal paired labels."""
    deployed = dataclasses.replace(selector, params={name: np.asarray(value) for name, value in params.items()})
    selected_indices = []
    anchor_indices = []
    for feature, logits, valid in zip(
        raw_features, validation["anchor_logits"], validation["history_valid"], strict=True
    ):
        prediction = deployed.forward(feature, logits, history_valid=bool(valid))
        selected_indices.append(int(np.argmax(prediction["probabilities"])))
        anchor_indices.append(int(np.argmax(ordered_log_probabilities(logits))))
    selected_indices = np.asarray(selected_indices, dtype=np.int32)
    rows = np.arange(len(selected_indices))
    paired_counts = np.asarray(validation["paired_count"])[rows, selected_indices].astype(np.int64)
    success_deltas = np.asarray(validation["success_delta"], dtype=np.float64)[rows, selected_indices]
    # These labels are ratios of integer paired binary outcomes. Restore their
    # numerators so equal successes do not become unequal through float32 sums.
    net_successes = np.rint(success_deltas * paired_counts).astype(np.int64)
    success_fraction = sum(
        (Fraction(int(net), int(count)) for net, count in zip(net_successes, paired_counts, strict=True)),
        Fraction(0),
    ) / len(rows)
    rpc_delta = float(np.mean(
        np.asarray(validation["rpc_delta_seconds"], dtype=np.float64)[rows, selected_indices]
    ))
    selected_h = [selector.candidates[index] for index in selected_indices]
    return {
        "roots": len(rows),
        "greedy_success_delta": float(success_fraction),
        "greedy_success_delta_fraction": {
            "numerator": success_fraction.numerator, "denominator": success_fraction.denominator,
        },
        "greedy_rpc_delta_seconds": rpc_delta,
        "greedy_paired_advantage": float(success_fraction) - RPC_WEIGHT * rpc_delta,
        "paired_net_success_count": int(net_successes.sum()),
        "selected_paired_trial_count": int(paired_counts.sum()),
        "changed_h_roots_vs_A": int(np.sum(selected_indices != np.asarray(anchor_indices))),
        "greedy_selected_h": selected_h,
        "greedy_selected_h_counts": {str(h): selected_h.count(h) for h in selector.candidates},
        "semantics": "Deployment NumPy argmax; mean paired success difference per root, then mean RPC difference in seconds.",
    }


def checkpoint_improves(
    selection_metric: Literal["expected_loss", "greedy"],
    candidate_validation: dict[str, float],
    best_validation: dict[str, float],
    *,
    candidate_greedy: dict[str, Any] | None = None,
    best_greedy: dict[str, Any] | None = None,
) -> bool:
    if selection_metric == "expected_loss":
        return candidate_validation["loss"] < best_validation["loss"]
    if selection_metric != "greedy" or candidate_greedy is None or best_greedy is None:
        raise ValueError("Greedy checkpoint selection requires candidate and incumbent greedy validation metrics.")

    def score(metrics: dict[str, Any]) -> tuple[Fraction, float]:
        fraction = metrics["greedy_success_delta_fraction"]
        return Fraction(fraction["numerator"], fraction["denominator"]), -metrics["greedy_rpc_delta_seconds"]

    return score(candidate_greedy) > score(best_greedy)


def train(args: Args) -> dict[str, Any]:
    if any(value <= 0 for value in (args.batch_size, args.max_updates, args.log_every, args.patience)):
        raise ValueError("Batch size, update budget, logging period and patience must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if args.selection_metric not in {"expected_loss", "greedy"}:
        raise ValueError("selection_metric must be expected_loss or greedy.")
    output = pathlib.Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Feedback output directory must be empty.")
    selector = FeedbackSelector.initialize_from_predictor(args.a_predictor_dir, variant=args.variant, seed=args.seed)
    training, training_summary = load_roots(args.train_dir, selector)
    validation, validation_summary = load_roots(args.validation_dir, selector)
    training_groups = {tuple(group) for group in training_summary["episode_groups"]}
    validation_groups = {tuple(group) for group in validation_summary["episode_groups"]}
    if training_groups & validation_groups:
        raise ValueError("Training and validation task/episode groups must be disjoint.")
    validation_raw_features = validation["feature"].copy() if args.selection_metric == "greedy" else None
    selector.fit_normalization(training["feature"])
    for data in (training, validation):
        data["feature"] = ((data["feature"] - selector.feature_mean) / selector.feature_std).astype(np.float32)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        params = {name: jnp.asarray(value) for name, value in selector.params.items()}
        optimizer = optax.adam(args.learning_rate)
        optimizer_state = optimizer.init(params)
        validation_batch = {name: jnp.asarray(value) for name, value in validation.items()}
        training_batch = {name: jnp.asarray(value) for name, value in training.items()}

        @jax.jit
        def train_step(current_params, current_optimizer_state, batch):
            (objective, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(current_params, batch)
            updates, next_optimizer_state = optimizer.update(gradients, current_optimizer_state, current_params)
            return optax.apply_updates(current_params, updates), next_optimizer_state, objective, metrics

        evaluate = jax.jit(lambda current_params, batch: loss(current_params, batch)[1])
        initial_validation = _metrics_to_python(evaluate(params, validation_batch))
        initial_training = _metrics_to_python(evaluate(params, training_batch))
        initial_greedy = (
            greedy_validation_metrics(selector, params, validation_raw_features, validation)
            if validation_raw_features is not None else None
        )
        best_params = jax.tree.map(lambda value: np.asarray(value).copy(), params)
        best_validation = initial_validation
        best_greedy = initial_greedy
        last_greedy = initial_greedy
        best_step = 0
        stale = 0
        final_step = 0
        rng = np.random.default_rng(args.seed)
        started = time.monotonic()
        output.mkdir(parents=True, exist_ok=True)
        with (output / "training_log.jsonl").open("w") as log_handle:
            initial_row = {"step": 0, "train": initial_training, "validation": initial_validation, "best_step": 0}
            if initial_greedy is not None:
                initial_row["validation_greedy"] = initial_greedy
            log_handle.write(json.dumps(initial_row, sort_keys=True) + "\n")
            log_handle.flush()
            print(json.dumps(initial_row, sort_keys=True), flush=True)
            for step in range(1, args.max_updates + 1):
                indices = rng.choice(
                    len(training["feature"]), size=min(args.batch_size, len(training["feature"])), replace=False
                )
                batch = {name: value[indices] for name, value in training_batch.items()}
                params, optimizer_state, objective, _ = train_step(params, optimizer_state, batch)
                if not np.isfinite(float(objective)):
                    raise ValueError(f"Non-finite feedback objective at update {step}.")
                final_step = step
                if step % args.log_every and step != args.max_updates:
                    continue
                train_metrics = _metrics_to_python(evaluate(params, training_batch))
                validation_metrics = _metrics_to_python(evaluate(params, validation_batch))
                last_greedy = (
                    greedy_validation_metrics(selector, params, validation_raw_features, validation)
                    if validation_raw_features is not None else None
                )
                if checkpoint_improves(
                    args.selection_metric, validation_metrics, best_validation,
                    candidate_greedy=last_greedy, best_greedy=best_greedy,
                ):
                    best_step = step
                    best_validation = validation_metrics
                    best_greedy = last_greedy
                    best_params = jax.tree.map(lambda value: np.asarray(value).copy(), params)
                    stale = 0
                else:
                    stale += 1
                row = {"step": step, "train": train_metrics, "validation": validation_metrics, "best_step": best_step}
                if last_greedy is not None:
                    row["validation_greedy"] = last_greedy
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                if stale >= args.patience:
                    break
    selector.params = best_params
    selector.metadata.update({
        "training_objective": "paired_monte_carlo_advantage_with_anchor_forward_kl",
        "rpc_weight_per_second": RPC_WEIGHT, "anchor_kl_weight": ANCHOR_KL_WEIGHT,
        "normalization_fit": "train_only", "best_step": best_step,
        "selection_metric": args.selection_metric,
        "train_dir": str(pathlib.Path(args.train_dir).resolve()),
        "validation_dir": str(pathlib.Path(args.validation_dir).resolve()),
    })
    checkpoint_path = output / "checkpoint.npz"
    selector.save(checkpoint_path)
    summary = {
        "status": "complete", "variant": args.variant, "checkpoint": str(checkpoint_path),
        "args": dataclasses.asdict(args), "best_step": best_step, "updates": final_step,
        "step0_included": True, "initial_training": initial_training, "initial_validation": initial_validation,
        "best_validation": best_validation, "training": training_summary, "validation": validation_summary,
        "training_seconds": time.monotonic() - started,
        "rpc_weight_per_second": RPC_WEIGHT, "anchor_kl_weight": ANCHOR_KL_WEIGHT,
        "success_signal": "paired binary terminal success difference",
        "cost_signal": "paired remaining RPC difference in seconds",
        "a_frozen": True, "critic": False,
        "residual_changed": bool(np.any(best_params["output_w"] != 0) or np.any(best_params["output_b"] != 0)),
        "selection_metric": args.selection_metric,
    }
    if best_greedy is not None:
        summary.update({
            "checkpoint_selection": "Maximize root-equal greedy paired success delta; equal success chooses lower RPC delta seconds; step0 included.",
            "initial_greedy_validation": initial_greedy,
            "best_greedy_validation": best_greedy,
            "last_greedy_validation": last_greedy,
            "selected_h_changes_from_step0": sum(
                before != after for before, after in zip(
                    initial_greedy["greedy_selected_h"], best_greedy["greedy_selected_h"], strict=True,
                )
            ),
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(train(tyro.cli(Args)), indent=2, sort_keys=True))
