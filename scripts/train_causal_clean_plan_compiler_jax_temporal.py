"""Low-parameter GPU JAX temporal compiler for the Phase-A EAR probe.

The model has a strict 15x7 EAR and 10x7 action-noise input contract.  EAR is
linearly resampled onto the ten action positions, processed by time-shared
per-step MLPs plus local depthwise temporal mixing, and decoded to 10x7 action
endpoints.  No sequence flattening or attention is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from typing import Any

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    from scripts import train_causal_clean_plan_compiler_jax as protocol
except ModuleNotFoundError:
    import train_causal_clean_plan_compiler_jax as protocol

from openpi.action_cot import endpoint_dataset


@dataclasses.dataclass(frozen=True)
class TemporalArgs:
    dataset: tuple[str, ...]
    output_dir: str
    steps: int = 1_000
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 5e-4
    final_learning_rate: float = 5e-5
    warmup_steps: int = 50
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 7
    log_interval: int = 25
    clean_loss_weight: float = 1.0
    intervention_loss_weight: float = 1.0
    response_loss_weight: float = 1.0
    plan_reconstruction_loss_weight: float = 0.25
    model_dim: int = 128
    feedforward_dim: int = 256
    temporal_blocks: int = 3
    active_action_dim: int = 7
    dropout: float = 0.0
    device: str = "gpu"
    amp: str = "bfloat16"
    latency_warmup: int = 20
    latency_runs: int = 100
    overwrite: bool = False


def _parse_args() -> TemporalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--final-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--intervention-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--plan-reconstruction-loss-weight", type=float, default=0.25)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--temporal-blocks", type=int, default=3)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    values = vars(parser.parse_args())
    values["dataset"] = tuple(values["dataset"])
    return TemporalArgs(**values)


def _validate_args(args: TemporalArgs) -> None:
    for name in (
        "steps", "batch_size", "eval_batch_size", "warmup_steps", "log_interval",
        "model_dim", "feedforward_dim", "temporal_blocks", "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.active_action_dim != 7:
        raise ValueError("The temporal Phase-A compiler has a strict 7-D input/output contract.")
    if args.latency_warmup < 0 or not 0 < args.validation_fraction < 0.5:
        raise ValueError("Latency warmup or validation fraction is invalid.")
    if args.learning_rate <= 0 or not 0 <= args.final_learning_rate <= args.learning_rate:
        raise ValueError("Learning rates are invalid.")
    if args.gradient_clip_norm <= 0 or args.weight_decay < 0:
        raise ValueError("Gradient clipping and weight decay values are invalid.")
    if min(
        args.clean_loss_weight, args.intervention_loss_weight,
        args.response_loss_weight, args.plan_reconstruction_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")


def _resample_time(values: jax.Array, target_horizon: int) -> jax.Array:
    """Deterministic endpoint-aligned linear interpolation along time."""
    source_horizon = values.shape[1]
    positions = jnp.linspace(0.0, float(source_horizon - 1), target_horizon)
    left = jnp.floor(positions).astype(jnp.int32)
    right = jnp.minimum(left + 1, source_horizon - 1)
    fraction = (positions - left.astype(positions.dtype))[None, :, None]
    return values[:, left, :] * (1.0 - fraction) + values[:, right, :] * fraction


class _TemporalMixerBlock(nn.Module):
    dimension: int
    feedforward_dim: int
    dropout: float
    compute_dtype: Any

    @nn.compact
    def __call__(self, values: jax.Array, *, training: bool) -> jax.Array:
        normalized = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(values)
        temporal = nn.Conv(
            features=self.dimension,
            kernel_size=(3,),
            padding="SAME",
            feature_group_count=self.dimension,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="depthwise_temporal",
        )(normalized)
        temporal = nn.Dense(
            self.dimension,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="temporal_channel_mix",
        )(temporal)
        values = values + nn.Dropout(rate=self.dropout)(temporal, deterministic=not training)
        normalized = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(values)
        hidden = nn.Dense(
            self.feedforward_dim,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="per_step_mlp_in",
        )(normalized)
        hidden = nn.silu(hidden)
        hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=not training)
        hidden = nn.Dense(
            self.dimension,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="per_step_mlp_out",
        )(hidden)
        return values + nn.Dropout(rate=self.dropout)(hidden, deterministic=not training)


class TemporalCleanPlanCompiler(nn.Module):
    plan_horizon: int
    action_horizon: int
    model_dim: int
    feedforward_dim: int
    temporal_blocks: int
    dropout: float
    compute_dtype: Any

    @nn.compact
    def __call__(
        self,
        plan_7d: jax.Array,
        action_noise_7d: jax.Array,
        *,
        training: bool,
        return_reconstruction: bool = True,
    ) -> tuple[jax.Array, jax.Array | None]:
        aligned_plan = _resample_time(plan_7d, self.action_horizon)
        position = self.param(
            "action_position_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.action_horizon, self.model_dim),
            jnp.float32,
        )
        plan_features = nn.Dense(
            self.model_dim,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="shared_plan_projection",
        )(aligned_plan) + position.astype(self.compute_dtype)
        for block_index in range(max(0, self.temporal_blocks - 1)):
            plan_features = _TemporalMixerBlock(
                self.model_dim,
                self.feedforward_dim,
                self.dropout,
                self.compute_dtype,
                name=f"plan_temporal_block_{block_index}",
            )(plan_features, training=training)

        reconstruction = None
        if return_reconstruction:
            reconstruction_features = _resample_time(plan_features, self.plan_horizon)
            reconstruction = nn.Dense(
                7,
                dtype=self.compute_dtype,
                param_dtype=jnp.float32,
                name="shared_plan_reconstruction",
            )(reconstruction_features)

        noise_features = nn.Dense(
            self.model_dim,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="shared_noise_projection",
        )(action_noise_7d)
        action_features = nn.silu(plan_features + noise_features)
        action_features = _TemporalMixerBlock(
            self.model_dim,
            self.feedforward_dim,
            self.dropout,
            self.compute_dtype,
            name="action_temporal_block",
        )(action_features, training=training)
        endpoint = nn.Dense(
            7,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="shared_action_output",
        )(action_features)
        return endpoint, reconstruction


def _pad_to_full(values: jax.Array, full_dim: int) -> jax.Array:
    return jnp.pad(values, ((0, 0), (0, 0), (0, full_dim - values.shape[-1])))


def _losses(
    model: TemporalCleanPlanCompiler,
    params: Any,
    batch: dict[str, jax.Array],
    args: TemporalArgs,
    dropout_key: jax.Array,
    *,
    training: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    variables = {"params": params}
    clean_prediction_7d, clean_reconstruction_7d = model.apply(
        variables,
        batch["clean_plan"][..., :7],
        batch["action_noise"][..., :7],
        training=training,
        return_reconstruction=True,
        rngs={"dropout": dropout_key},
    )
    intervention_prediction_7d, intervention_reconstruction_7d = model.apply(
        variables,
        batch["intervention_plan"][..., :7],
        batch["action_noise"][..., :7],
        training=training,
        return_reconstruction=True,
        rngs={"dropout": jax.random.fold_in(dropout_key, 1)},
    )
    full_dim = batch["clean_actions"].shape[-1]
    clean_prediction = _pad_to_full(clean_prediction_7d, full_dim)
    intervention_prediction = _pad_to_full(intervention_prediction_7d, full_dim)
    clean_reconstruction = _pad_to_full(clean_reconstruction_7d, full_dim)
    intervention_reconstruction = _pad_to_full(intervention_reconstruction_7d, full_dim)

    def mse(left: jax.Array, right: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(left.astype(jnp.float32) - right.astype(jnp.float32)))

    clean_mse = mse(clean_prediction_7d, batch["clean_actions"][..., :7])
    clean_mse_full = mse(clean_prediction, batch["clean_actions"])
    intervention_mse = mse(
        intervention_prediction_7d, batch["intervention_actions"][..., :7],
    )
    intervention_mse_full = mse(intervention_prediction, batch["intervention_actions"])
    predicted_response_7d = intervention_prediction_7d - clean_prediction_7d
    teacher_response_7d = (
        batch["intervention_actions"][..., :7] - batch["clean_actions"][..., :7]
    )
    response_mse = mse(predicted_response_7d, teacher_response_7d)
    response_mse_full = mse(
        intervention_prediction - clean_prediction,
        batch["intervention_actions"] - batch["clean_actions"],
    )
    clean_reconstruction_mse = mse(clean_reconstruction_7d, batch["clean_plan"][..., :7])
    clean_reconstruction_mse_full = mse(clean_reconstruction, batch["clean_plan"])
    intervention_reconstruction_mse = mse(
        intervention_reconstruction_7d, batch["intervention_plan"][..., :7],
    )
    intervention_reconstruction_mse_full = mse(
        intervention_reconstruction, batch["intervention_plan"],
    )
    reconstruction_mse = 0.5 * (clean_reconstruction_mse + intervention_reconstruction_mse)
    reconstruction_mse_full = 0.5 * (
        clean_reconstruction_mse_full + intervention_reconstruction_mse_full
    )
    total = (
        args.clean_loss_weight * clean_mse
        + args.intervention_loss_weight * intervention_mse
        + args.response_loss_weight * response_mse
        + args.plan_reconstruction_loss_weight * reconstruction_mse
    )
    return total, {
        "total_loss": total,
        "clean_action_mse_active7": clean_mse,
        "clean_action_mse_full32": clean_mse_full,
        "intervention_action_mse_active7": intervention_mse,
        "intervention_action_mse_full32": intervention_mse_full,
        "response_mse_active7": response_mse,
        "response_mse_full32": response_mse_full,
        "plan_reconstruction_mse_active7": reconstruction_mse,
        "plan_reconstruction_mse_full32": reconstruction_mse_full,
        "clean_plan_reconstruction_mse_active7": clean_reconstruction_mse,
        "clean_plan_reconstruction_mse_full32": clean_reconstruction_mse_full,
        "intervention_plan_reconstruction_mse_active7": intervention_reconstruction_mse,
        "intervention_plan_reconstruction_mse_full32": intervention_reconstruction_mse_full,
    }


def _learning_rate(args: TemporalArgs, step: int) -> float:
    if step <= args.warmup_steps:
        return args.learning_rate * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * cosine


def _schedule(args: TemporalArgs):
    def schedule(count: jax.Array) -> jax.Array:
        step = count.astype(jnp.float32) + 1.0
        warmup = args.learning_rate * step / float(args.warmup_steps)
        progress = jnp.clip(
            (step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps)),
            0.0,
            1.0,
        )
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay = args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * cosine
        return jnp.where(step <= args.warmup_steps, warmup, decay)
    return schedule


def main(args: TemporalArgs) -> None:
    _validate_args(args)
    np.random.seed(args.seed)
    device = protocol._resolve_device(args.device)
    compute_dtype = jnp.bfloat16 if args.amp == "bfloat16" else jnp.float32
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.npz"
    existing = [path for path in (metrics_path, summary_path, checkpoint_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Probe outputs already exist: {[str(path) for path in existing]}; "
            "choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = protocol._split_indices(
        arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    plan_horizon = int(arrays["clean_coarse"].shape[1])
    action_horizon = int(arrays["clean_actions"].shape[1])
    action_dim = int(arrays["clean_actions"].shape[-1])
    if plan_horizon != 15 or action_horizon != 10 or action_dim < 7:
        raise ValueError(
            f"Expected EAR 15x>=7 and action 10x>=7; got "
            f"{arrays['clean_coarse'].shape[1:]} and {arrays['clean_actions'].shape[1:]}."
        )

    model = TemporalCleanPlanCompiler(
        plan_horizon=15,
        action_horizon=10,
        model_dim=args.model_dim,
        feedforward_dim=args.feedforward_dim,
        temporal_blocks=args.temporal_blocks,
        dropout=args.dropout,
        compute_dtype=compute_dtype,
    )
    init_key, dropout_key = jax.random.split(jax.random.PRNGKey(args.seed))
    with jax.default_device(device):
        variables = model.init(
            {"params": init_key, "dropout": dropout_key},
            jnp.zeros((1, 15, 7), dtype=jnp.float32),
            jnp.zeros((1, 10, 7), dtype=jnp.float32),
            training=True,
            return_reconstruction=True,
        )
    parameter_count = int(
        sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(variables["params"]))
    )
    if parameter_count >= 300_000:
        raise ValueError(
            f"Temporal compiler has {parameter_count:,} parameters; required target is <300k."
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(_schedule(args), weight_decay=args.weight_decay),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=jax.device_put(variables["params"], device),
        tx=optimizer,
    )

    @jax.jit
    def train_step(
        current_state: train_state.TrainState,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[train_state.TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _losses(model, params, batch, args, key, training=True)
        (_, values), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_state.params)
        values = {**values, "gradient_norm": optax.global_norm(grads)}
        return current_state.apply_gradients(grads=grads), values

    @jax.jit
    def validation_step(params: Any, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        _, values = _losses(
            model, params, batch, args, jax.random.PRNGKey(0), training=False,
        )
        return values

    @jax.jit
    def predict_step(
        params: Any,
        full_plan: jax.Array,
        full_noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        endpoint_7d, reconstruction_7d = model.apply(
            {"params": params},
            full_plan[..., :7],
            full_noise[..., :7],
            training=False,
            return_reconstruction=True,
        )
        return _pad_to_full(endpoint_7d, full_plan.shape[-1]), _pad_to_full(
            reconstruction_7d, full_plan.shape[-1],
        )

    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    training_key = jax.random.PRNGKey(args.seed + 10)
    started = time.monotonic()
    print(
        f"Initialized temporal JAX compiler: train={train_indices.size} "
        f"validation={validation_indices.size} params={parameter_count:,} "
        f"device={device} amp={args.amp}",
        flush=True,
    )

    last_record: dict[str, Any] = {}
    best_record: dict[str, Any] = {}
    best_params: Any | None = None
    best_score: tuple[float, float] | None = None
    best_step = 0
    metrics_mode = "w" if args.overwrite else "a"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            rows = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            intervention_slots = protocol._choose_interventions(
                arrays, rows, rng, deterministic=False,
            )
            batch = jax.device_put(protocol._make_batch(arrays, rows, intervention_slots), device)
            training_key, step_key = jax.random.split(training_key)
            state, train_metrics = train_step(state, batch, step_key)

            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            if not should_log:
                continue
            validation_rows = validation_rng.choice(
                validation_indices,
                size=min(args.batch_size, validation_indices.size),
                replace=False,
            )
            validation_slots = protocol._choose_interventions(
                arrays, validation_rows, validation_rng, deterministic=True,
            )
            validation_batch = jax.device_put(
                protocol._make_batch(arrays, validation_rows, validation_slots), device,
            )
            validation_metrics = validation_step(state.params, validation_batch)
            train_host, validation_host = jax.device_get((train_metrics, validation_metrics))
            candidate_score = (
                float(validation_host["response_mse_active7"]),
                float(validation_host["clean_action_mse_active7"]),
            )
            selected_as_best = best_score is None or candidate_score < best_score
            last_record = {
                "phase": "train",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": _learning_rate(args, step),
                "selected_as_best_checkpoint": selected_as_best,
                **{f"train/{name}": float(value) for name, value in train_host.items()},
                **{
                    f"validation_sample/{name}": float(value)
                    for name, value in validation_host.items()
                },
            }
            if selected_as_best:
                best_params = state.params
                best_score = candidate_score
                best_step = step
                best_record = dict(last_record)
            metrics_file.write(json.dumps(last_record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"step={step} train_total={last_record['train/total_loss']:.6f} "
                f"val_clean7={last_record['validation_sample/clean_action_mse_active7']:.6f} "
                f"val_response7={last_record['validation_sample/response_mse_active7']:.6f} "
                f"best={selected_as_best}",
                flush=True,
            )

        if best_params is None:
            raise RuntimeError("Training completed without selecting a validation checkpoint.")
        jax.block_until_ready(best_params)
        full_metrics = protocol._full_validation(
            arrays,
            validation_indices,
            best_params,
            predict_step,
            batch_size=args.eval_batch_size,
            device=device,
            active_action_dim=7,
            seed=args.seed,
        )
        full_metrics.update(protocol._latency_metrics(
            predict_step,
            best_params,
            arrays,
            validation_indices,
            device=device,
            warmup=args.latency_warmup,
            runs=args.latency_runs,
        ))
        final_record = {
            "phase": "full_validation",
            "step": args.steps,
            "evaluated_checkpoint_step": best_step,
            "elapsed_seconds": time.monotonic() - started,
            **full_metrics,
        }
        metrics_file.write(json.dumps(final_record, sort_keys=True) + "\n")
        metrics_file.flush()

    protocol._save_checkpoint(
        checkpoint_path,
        best_params,
        train_indices,
        validation_indices,
        args.steps,
        best_step,
    )
    summary = {
        "probe": "phase_a_causal_clean_plan_compiler_jax_temporal_oracle",
        "dataset": list(args.dataset),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "args": dataclasses.asdict(args),
        "input_contract": {
            "model_inputs": ["teacher_ear_endpoint_15x7", "shared_action_noise_10x7"],
            "clean_and_intervention_share_action_noise": True,
            "forbidden_inputs": [
                "observation", "image", "state", "IAR", "task_id", "episode_id",
                "inactive_action_dimensions_7_to_31",
            ],
            "metadata_usage": "task_id/episode_id only for split and shuffled-EAR evaluation",
            "semantic_interventions_only": True,
        },
        "architecture": {
            "runtime": "JAX/Flax JIT; no PyTorch CUDA dependency",
            "ear_alignment": "fixed endpoint-aligned linear interpolation from 15 to 10 steps",
            "temporal_model": "shared per-step MLP plus depthwise kernel-3 residual mixing",
            "flattened_sequence_mlp": False,
            "attention": False,
            "strict_model_input_shapes": {"ear": [15, 7], "action_noise": [10, 7]},
            "strict_model_output_shape": [10, 7],
            "parameter_count": parameter_count,
            "parameter_target_met": parameter_count < 300_000,
            "compute_dtype": args.amp,
            "parameter_dtype": "float32",
        },
        "training_objective": {
            "clean_action_mse_weight": args.clean_loss_weight,
            "intervention_action_mse_weight": args.intervention_loss_weight,
            "causal_response_delta_mse_weight": args.response_loss_weight,
            "plan_reconstruction_mse_weight": args.plan_reconstruction_loss_weight,
            "optimized_dimensions": 7,
        },
        "dataset_records": int(len(arrays["dataset_index"])),
        "eligible_semantic_records": int(train_indices.size + validation_indices.size),
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "train_episode_groups": protocol._episode_group_count(arrays, train_indices),
        "validation_episode_groups": protocol._episode_group_count(arrays, validation_indices),
        "train_task_counts": protocol._task_counts(arrays, train_indices),
        "validation_task_counts": protocol._task_counts(arrays, validation_indices),
        "completed_steps": args.steps,
        "checkpoint_selection": {
            "primary_metric": "validation_sample/response_mse_active7",
            "tie_break_metric": "validation_sample/clean_action_mse_active7",
            "mode": "min",
            "best_step": best_step,
            "last_step": args.steps,
            "best_validation_sample_metrics": {
                name.removeprefix("validation_sample/"): value
                for name, value in best_record.items()
                if name.startswith("validation_sample/")
            },
            "last_validation_sample_metrics": {
                name.removeprefix("validation_sample/"): value
                for name, value in last_record.items()
                if name.startswith("validation_sample/")
            },
        },
        "best_step": best_step,
        "last_step": args.steps,
        "best_training_record": best_record,
        "last_training_record": last_record,
        "full_validation_metrics": full_metrics,
        "full_validation_checkpoint_step": best_step,
        "device": str(device),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(
        "Full held-out: "
        f"clean_mse_active7={full_metrics['clean_action_mse_active7']:.6f} "
        f"response_mse_active7={full_metrics['response_mse_active7']:.6f} "
        f"response_cosine_active7={full_metrics['response_cosine_active7']:.4f} "
        f"shuffle_gap_active7={full_metrics['same_task_shuffled_ear_action_mse_gap_active7']} "
        f"gpu_latency_mean_ms={full_metrics['gpu_latency_mean_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
