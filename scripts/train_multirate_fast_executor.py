"""Train and evaluate the fixed 1:4 plan-conditioned fast executor.

The base ACoT checkpoint is not loaded.  Two matched small policies are
trained from exported windows:

* ``plan`` consumes the anchor EAR/IAR cache;
* ``direct`` receives zero EAR/IAR with exactly the same architecture.

The held-out report also evaluates Hold-4, B6, fresh-plan oracle, and a
same-task shuffled-plan necessity intervention.
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
from openpi.action_cot import multirate_dataset
from openpi.models import multirate_fast_executor
import optax
import orbax.checkpoint as ocp
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    variants: tuple[str, ...] = ("plan", "direct")
    seed: int = 7
    train_steps: int = 1_500
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    log_interval: int = 100
    early_stopping_patience_logs: int = 5
    early_stopping_min_delta: float = 1e-5
    event_weight: float = 2.0
    age_zero_weight: float = 0.5
    residual_scale: float = 2.0
    direct_output_margin: float = 1.10
    minimum_gripper_transition_count: int = 20
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if args.train_steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("train_steps and batch sizes must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.early_stopping_patience_logs < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping settings must be non-negative.")
    if args.minimum_gripper_transition_count <= 0:
        raise ValueError("minimum_gripper_transition_count must be positive.")
    if (
        args.event_weight < 0
        or args.age_zero_weight <= 0
        or args.residual_scale <= 0
        or args.direct_output_margin <= 1.0
    ):
        raise ValueError(
            "Loss weights and residual_scale must be positive/non-negative, and direct_output_margin must exceed one."
        )
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    if not args.variants or any(variant not in {"plan", "direct"} for variant in args.variants):
        raise ValueError("variants must contain plan and/or direct.")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must not contain duplicates.")


def _split_windows(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            raise ValueError(f"Task {task_id} needs at least three episodes for train/val/test.")
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


def _flat_indices(window_indices: np.ndarray, window_size: int) -> np.ndarray:
    ages = np.arange(window_size, dtype=np.int64)
    return (window_indices[:, None] * window_size + ages[None, :]).reshape(-1)


def _phase_aligned_bases(
    arrays: dict[str, np.ndarray],
    window_indices: np.ndarray,
    *,
    coarse_time_stride: int,
) -> np.ndarray:
    cached_ear = arrays["fresh_ear"][window_indices, 0].astype(np.float32)
    ages = np.arange(arrays["images"].shape[1], dtype=np.float32)
    phases = ages / coarse_time_stride
    lower = np.minimum(np.floor(phases).astype(np.int64), cached_ear.shape[1] - 1)
    upper = np.minimum(lower + 1, cached_ear.shape[1] - 1)
    interpolation = phases - lower
    return cached_ear[:, lower] + interpolation[None, :, None] * (cached_ear[:, upper] - cached_ear[:, lower])


def _batch(
    arrays: dict[str, np.ndarray],
    flat_indices: np.ndarray,
    *,
    variant: str,
    event_weight: float,
    age_zero_weight: float,
    plan_window_override: np.ndarray | None = None,
    fresh_plan: bool = False,
) -> dict[str, jax.Array]:
    window_size = arrays["images"].shape[1]
    windows = flat_indices // window_size
    ages = flat_indices % window_size
    plan_windows = windows if plan_window_override is None else plan_window_override[windows]
    if fresh_plan:
        cached_ear = arrays["fresh_ear"][windows, ages]
        cached_iar = arrays["fresh_iar"][windows, ages]
        cache_age = np.zeros_like(ages)
    else:
        cached_ear = arrays["fresh_ear"][plan_windows, 0]
        cached_iar = arrays["fresh_iar"][plan_windows, 0]
        cache_age = ages
    if variant == "direct":
        cached_ear = np.zeros_like(cached_ear)
        cached_iar = np.zeros_like(cached_iar)

    sample_weight = 1.0 + event_weight * arrays["event_mask"][windows, ages].astype(np.float32)
    sample_weight *= np.where(ages == 0, age_zero_weight, 1.0)
    valid_mask = np.zeros((flat_indices.size, arrays["teacher_actions"].shape[-1]), dtype=np.float32)
    valid_mask[:, :7] = sample_weight[:, None]
    return {
        "current_images": jnp.asarray(arrays["images"][windows, ages].astype(np.float32) / 255.0),
        "state": jnp.asarray(arrays["states"][windows, ages], dtype=jnp.float32),
        "cached_ear": jnp.asarray(cached_ear, dtype=jnp.float32),
        "cached_iar": jnp.asarray(cached_iar, dtype=jnp.float32),
        "cache_age": jnp.asarray(cache_age, dtype=jnp.int32),
        "target_action": jnp.asarray(
            arrays["teacher_actions"][windows, ages],
            dtype=jnp.float32,
        ),
        "valid_mask": jnp.asarray(valid_mask),
    }


def _save_params(params: nnx.State, target: pathlib.Path, *, overwrite: bool) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"multirate_fast_executor": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=overwrite)


def _train_variant(
    arrays: dict[str, np.ndarray],
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    *,
    variant: str,
    args: Args,
    config: multirate_fast_executor.MultiRateFastExecutorConfig,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    model = multirate_fast_executor.MultiRateFastExecutor(
        config,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(model)
    loss_config = multirate_fast_executor.FastExecutorLossConfig(action_dim=config.action_dim)
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

        def loss_fn(candidate: multirate_fast_executor.MultiRateFastExecutor):
            predicted = candidate(
                batch["current_images"],
                batch["state"],
                batch["cached_ear"],
                batch["cached_iar"],
                batch["cache_age"],
            )
            return multirate_fast_executor.multirate_fast_executor_loss(
                predicted,
                batch["target_action"],
                config=loss_config,
                valid_mask=batch["valid_mask"],
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(current_model)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        return (
            next_params,
            next_optimizer_state,
            {
                **metrics,
                "loss": loss,
                "gradient_norm": optax.global_norm(gradients),
            },
        )

    @jax.jit
    def validation_step(
        current_params: nnx.State,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        current_model = nnx.merge(graphdef, current_params)
        predicted = current_model(
            batch["current_images"],
            batch["state"],
            batch["cached_ear"],
            batch["cached_iar"],
            batch["cache_age"],
        )
        _, metrics = multirate_fast_executor.multirate_fast_executor_loss(
            predicted,
            batch["target_action"],
            config=loss_config,
            valid_mask=batch["valid_mask"],
        )
        first_seven_error = predicted[:, :7] - batch["target_action"][:, :7]
        return {
            **metrics,
            "action_mse_7d": jnp.mean(jnp.square(first_seven_error)),
        }

    train_flat = _flat_indices(train_windows, arrays["images"].shape[1])
    validation_flat = _flat_indices(validation_windows, arrays["images"].shape[1])
    # Both variants see exactly the same training index sequence.  Keeping the
    # data order matched makes the direct-policy comparison attributable to the
    # cached plan rather than batch-sampling noise.
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    fixed_validation_sample = validation_rng.choice(
        validation_flat,
        size=min(args.eval_batch_size, validation_flat.size),
        replace=False,
    )
    fixed_validation_batch = _batch(
        arrays,
        fixed_validation_sample,
        variant=variant,
        event_weight=args.event_weight,
        age_zero_weight=args.age_zero_weight,
    )
    metrics_path = output_dir / variant / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Metrics already exist: {metrics_path}")
    best_loss = float("inf")
    best_step = 0
    best_params: nnx.State | None = None
    stale_logs = 0
    completed_steps = 0
    started = time.monotonic()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_flat,
                size=args.batch_size,
                replace=train_flat.size < args.batch_size,
            )
            train_batch = _batch(
                arrays,
                sampled,
                variant=variant,
                event_weight=args.event_weight,
                age_zero_weight=args.age_zero_weight,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                train_batch,
            )
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_metrics = validation_step(
                    params,
                    fixed_validation_batch,
                )
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()},
                    **{
                        f"validation/{name}": float(value) for name, value in jax.device_get(validation_metrics).items()
                    },
                }
                non_finite = {
                    name: value for name, value in record.items() if isinstance(value, float) and not np.isfinite(value)
                }
                if non_finite:
                    raise FloatingPointError(f"Non-finite training metrics for {variant}: {non_finite}")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(
                    json.dumps({"variant": variant, **record}, sort_keys=True),
                    flush=True,
                )
                validation_loss = record["validation/loss"]
                if validation_loss < best_loss - args.early_stopping_min_delta:
                    best_loss = validation_loss
                    best_step = step
                    best_params = params
                    stale_logs = 0
                else:
                    stale_logs += 1
                if args.early_stopping_patience_logs > 0 and stale_logs >= args.early_stopping_patience_logs:
                    break

    selected_params = best_params if best_params is not None else params
    params_path = output_dir / variant / "final" / "params"
    _save_params(selected_params, params_path, overwrite=args.overwrite)
    summary = {
        "variant": variant,
        "completed_steps": completed_steps,
        "requested_steps": args.train_steps,
        "best_validation_step": best_step,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.monotonic() - started,
        "params_path": str(params_path.resolve()),
        "parameter_count": multirate_fast_executor.estimate_parameter_count(config),
    }
    (output_dir / variant / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return graphdef, selected_params, summary


def _predict(
    graphdef: Any,
    params: nnx.State,
    arrays: dict[str, np.ndarray],
    flat_indices: np.ndarray,
    *,
    variant: str,
    args: Args,
    plan_window_override: np.ndarray | None = None,
    fresh_plan: bool = False,
) -> np.ndarray:
    @jax.jit
    def predict_batch(current_params: nnx.State, batch: dict[str, jax.Array]) -> jax.Array:
        model = nnx.merge(graphdef, current_params)
        return model(
            batch["current_images"],
            batch["state"],
            batch["cached_ear"],
            batch["cached_iar"],
            batch["cache_age"],
        )

    pieces = []
    for start in range(0, flat_indices.size, args.eval_batch_size):
        selected = flat_indices[start : start + args.eval_batch_size]
        batch = _batch(
            arrays,
            selected,
            variant=variant,
            event_weight=0.0,
            age_zero_weight=1.0,
            plan_window_override=plan_window_override,
            fresh_plan=fresh_plan,
        )
        pieces.append(np.asarray(predict_batch(params, batch)))
    predicted = np.concatenate(pieces, axis=0)
    if not np.all(np.isfinite(predicted)):
        raise FloatingPointError(f"{variant} produced non-finite held-out predictions.")
    return predicted


def _cosine(predicted: np.ndarray, target: np.ndarray, dims: slice) -> float:
    left = predicted[:, dims]
    right = target[:, dims]
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    valid = denominator > 1e-8
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0


def _action_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    ages: np.ndarray,
    event_mask: np.ndarray,
    gripper_transition_mask: np.ndarray,
) -> dict[str, Any]:
    error = predicted[:, :7] - target[:, :7]

    def mse(mask: np.ndarray) -> float:
        return float(np.mean(np.square(error[mask]))) if np.any(mask) else float("nan")

    stale = ages > 0
    teacher_gripper = target[:, 6] >= 0
    predicted_gripper = predicted[:, 6] >= 0
    event = event_mask.astype(np.bool_)
    gripper_transition = gripper_transition_mask.astype(np.bool_)
    return {
        "mse_7d": mse(np.ones_like(ages, dtype=np.bool_)),
        "mse_age0": mse(ages == 0),
        "mse_age1_to_3": mse(stale),
        "mse_age3": mse(ages == 3),
        "translation_cosine_age1_to_3": _cosine(predicted[stale], target[stale], slice(0, 3)),
        "rotation_cosine_age1_to_3": _cosine(predicted[stale], target[stale], slice(3, 6)),
        "gripper_sign_accuracy": float(np.mean(predicted_gripper == teacher_gripper)),
        "event_gripper_sign_accuracy": (
            float(np.mean(predicted_gripper[event] == teacher_gripper[event])) if np.any(event) else None
        ),
        "gripper_transition_sign_accuracy": (
            float(np.mean(predicted_gripper[gripper_transition] == teacher_gripper[gripper_transition]))
            if np.any(gripper_transition)
            else None
        ),
        "count": int(ages.size),
        "stale_count": int(np.sum(stale)),
        "event_count": int(np.sum(event)),
        "gripper_transition_count": int(np.sum(gripper_transition)),
    }


def _shuffled_plan_map(
    arrays: dict[str, np.ndarray],
    target_windows: np.ndarray,
    donor_windows: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if not donor_windows.size:
        raise ValueError("Shuffled-plan evaluation requires at least one donor window.")
    mapping = np.arange(arrays["task_id"].shape[0], dtype=np.int64)
    tasks = np.asarray(arrays["task_id"])
    episodes = np.asarray(arrays["episode_id"])
    rng = np.random.default_rng(seed)
    for window in target_windows:
        candidates = donor_windows[
            (tasks[donor_windows] == tasks[window])
            & (episodes[donor_windows] != episodes[window])
            & (donor_windows != window)
        ]
        if not candidates.size:
            raise ValueError(
                "No different-episode shuffled-plan donor for "
                f"window={window}, task={tasks[window]}, episode={episodes[window]}."
            )
        mapping[window] = int(rng.choice(candidates))
    return mapping


def _evaluate(
    arrays: dict[str, np.ndarray],
    test_windows: np.ndarray,
    plan_donor_windows: np.ndarray,
    trained: dict[str, tuple[Any, nnx.State]],
    *,
    args: Args,
) -> dict[str, Any]:
    window_size = arrays["images"].shape[1]
    flat = _flat_indices(test_windows, window_size)
    windows = flat // window_size
    ages = flat % window_size
    target = arrays["teacher_actions"][windows, ages].astype(np.float32)
    event = arrays["event_mask"][windows, ages]
    test_gripper = arrays["teacher_actions"][test_windows, :, 6].astype(np.float32)
    gripper_transition_by_window = np.zeros_like(test_gripper, dtype=np.bool_)
    gripper_transition_by_window[:, 1:] = np.abs(np.diff(test_gripper, axis=1)) > 0.5
    gripper_transition = gripper_transition_by_window.reshape(-1)
    metrics: dict[str, Any] = {
        "b6": _action_metrics(
            arrays["b6_actions"][windows, ages].astype(np.float32),
            target,
            ages,
            event,
            gripper_transition,
        ),
        "hold4": _action_metrics(
            arrays["hold_actions"][windows, ages].astype(np.float32),
            target,
            ages,
            event,
            gripper_transition,
        ),
    }
    predictions: dict[str, np.ndarray] = {}
    for variant, (graphdef, params) in trained.items():
        predictions[variant] = _predict(
            graphdef,
            params,
            arrays,
            flat,
            variant=variant,
            args=args,
        )
        metrics[variant] = _action_metrics(
            predictions[variant],
            target,
            ages,
            event,
            gripper_transition,
        )

    if "plan" in trained:
        graphdef, params = trained["plan"]
        fresh = _predict(
            graphdef,
            params,
            arrays,
            flat,
            variant="plan",
            args=args,
            fresh_plan=True,
        )
        shuffled = _predict(
            graphdef,
            params,
            arrays,
            flat,
            variant="plan",
            args=args,
            plan_window_override=_shuffled_plan_map(
                arrays,
                test_windows,
                plan_donor_windows,
                seed=args.seed,
            ),
        )
        metrics["fresh_plan_oracle"] = _action_metrics(
            fresh,
            target,
            ages,
            event,
            gripper_transition,
        )
        metrics["shuffled_plan"] = _action_metrics(
            shuffled,
            target,
            ages,
            event,
            gripper_transition,
        )

    gates: dict[str, Any] = {"evaluated": False}
    if "plan" in metrics and "direct" in metrics:
        plan = metrics["plan"]
        b6 = metrics["b6"]
        hold = metrics["hold4"]
        direct = metrics["direct"]
        shuffled = metrics["shuffled_plan"]

        def ratio(numerator: float, denominator: float) -> float:
            return float(numerator / max(denominator, 1e-12))

        values = {
            "plan_age0_over_b6": ratio(plan["mse_age0"], b6["mse_age0"]),
            "plan_stale_over_b6": ratio(plan["mse_age1_to_3"], b6["mse_age1_to_3"]),
            "plan_age3_over_b6": ratio(plan["mse_age3"], b6["mse_age3"]),
            "plan_stale_over_hold4": ratio(plan["mse_age1_to_3"], hold["mse_age1_to_3"]),
            "plan_stale_over_direct": ratio(plan["mse_age1_to_3"], direct["mse_age1_to_3"]),
            "shuffled_over_plan": ratio(
                shuffled["mse_age1_to_3"],
                plan["mse_age1_to_3"],
            ),
        }
        checks = {
            "fresh_capacity": values["plan_age0_over_b6"] <= 1.10,
            "stale_vs_b6": values["plan_stale_over_b6"] <= 1.25,
            "age3_vs_b6": values["plan_age3_over_b6"] <= 1.50,
            "beats_hold4": values["plan_stale_over_hold4"] <= 0.75,
            "beats_direct": values["plan_stale_over_direct"] <= 0.90,
            "plan_necessity": values["shuffled_over_plan"] >= 1.15,
            "gripper_transition": (
                plan["gripper_transition_count"] >= args.minimum_gripper_transition_count
                and plan["gripper_transition_sign_accuracy"] is not None
                and plan["gripper_transition_sign_accuracy"] >= 0.90
            ),
            "translation_cosine": plan["translation_cosine_age1_to_3"] >= 0.90,
            "rotation_cosine": plan["rotation_cosine_age1_to_3"] >= 0.90,
        }
        gates = {
            "evaluated": True,
            "values": values,
            "checks": checks,
            "offline_gate_pass": all(checks.values()),
            "immediate_stop": (
                values["plan_age0_over_b6"] > 1.25
                or values["plan_stale_over_b6"] > 1.50
                or values["plan_stale_over_hold4"] >= 1.0
                or values["plan_stale_over_direct"] >= 1.0
                or (
                    plan["gripper_transition_count"] >= args.minimum_gripper_transition_count
                    and plan["gripper_transition_sign_accuracy"] is not None
                    and plan["gripper_transition_sign_accuracy"] < 0.85
                )
            ),
        }
    return {
        "num_test_windows": int(test_windows.size),
        "num_test_frames": int(flat.size),
        "metrics": metrics,
        "gates": gates,
        "note": "Episode-held-out open-loop action fidelity; not a LIBERO success-rate result.",
    }


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = multirate_dataset.load_multirate_arrays(args.dataset)
    if arrays["images"].shape[1] != 4:
        raise ValueError(f"Expected four-frame windows, got {arrays['images'].shape}.")
    train_windows, validation_windows, test_windows = _split_windows(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    coarse_time_stride = 2
    train_targets = arrays["teacher_actions"][train_windows].astype(np.float32)
    train_phase_bases = _phase_aligned_bases(
        arrays,
        train_windows,
        coarse_time_stride=coarse_time_stride,
    )
    train_target_max_abs_7d = float(np.max(np.abs(train_targets[..., :7])))
    train_plan_residual_max_abs_7d = float(np.max(np.abs(train_targets[..., :7] - train_phase_bases[..., :7])))
    effective_residual_scale = max(
        args.residual_scale,
        args.direct_output_margin * max(train_target_max_abs_7d, train_plan_residual_max_abs_7d),
    )
    all_target_abs_7d = np.abs(arrays["teacher_actions"][..., :7].astype(np.float32))
    all_phase_bases = _phase_aligned_bases(
        arrays,
        np.arange(arrays["anchor_index"].shape[0]),
        coarse_time_stride=coarse_time_stride,
    )
    all_plan_residual_abs_7d = np.abs(arrays["teacher_actions"][..., :7].astype(np.float32) - all_phase_bases[..., :7])
    direct_out_of_range_fraction = float(np.mean(all_target_abs_7d >= 0.98 * effective_residual_scale))
    plan_out_of_range_fraction = float(np.mean(all_plan_residual_abs_7d >= 0.98 * effective_residual_scale))
    if direct_out_of_range_fraction > 0 or plan_out_of_range_fraction > 0:
        raise ValueError(
            "The matched models cannot represent every 7D target within the "
            f"tanh bound {effective_residual_scale:.6f}: "
            f"direct={direct_out_of_range_fraction:.6%}, "
            f"plan_residual={plan_out_of_range_fraction:.6%}. "
            "Increase --residual-scale without tuning it to held-out metrics."
        )
    config = multirate_fast_executor.MultiRateFastExecutorConfig(
        image_size=int(arrays["images"].shape[3]),
        state_dim=int(arrays["states"].shape[-1]),
        action_dim=int(arrays["teacher_actions"].shape[-1]),
        ear_horizon=int(arrays["fresh_ear"].shape[-2]),
        iar_dim=int(arrays["fresh_iar"].shape[-1]),
        coarse_time_stride=coarse_time_stride,
        residual_scale=effective_residual_scale,
    )
    trained: dict[str, tuple[Any, nnx.State]] = {}
    train_summaries = {}
    for variant in args.variants:
        graphdef, params, summary = _train_variant(
            arrays,
            train_windows,
            validation_windows,
            variant=variant,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        trained[variant] = (graphdef, params)
        train_summaries[variant] = summary

    evaluation = _evaluate(
        arrays,
        test_windows,
        validation_windows,
        trained,
        args=args,
    )
    summary = {
        "status": "complete",
        "dataset": list(args.dataset),
        "num_windows": int(arrays["anchor_index"].shape[0]),
        "num_train_windows": int(train_windows.size),
        "num_validation_windows": int(validation_windows.size),
        "num_test_windows": int(test_windows.size),
        "train_target_max_abs_7d": train_target_max_abs_7d,
        "train_plan_residual_max_abs_7d": train_plan_residual_max_abs_7d,
        "direct_out_of_range_fraction": direct_out_of_range_fraction,
        "plan_out_of_range_fraction": plan_out_of_range_fraction,
        "effective_residual_scale": effective_residual_scale,
        "config": dataclasses.asdict(config),
        "train": train_summaries,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
