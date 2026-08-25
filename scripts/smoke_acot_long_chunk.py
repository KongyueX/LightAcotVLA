"""GPU smoke test for opt-in ACoT-VLA long-action-chunk training.

The smoke uses the real training config and data pipeline. It checks the
configured batch shapes, restores both the student and frozen prefix anchor,
evaluates the paired long-flow/prefix-retention loss, and runs a one-step
sampling forward.
It does not update parameters or write a training checkpoint.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
from typing import Any

from flax import nnx
import jax
import numpy as np
import tyro

from openpi.models import model as model_lib
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders

try:
    import train as train_lib
except ImportError:  # pragma: no cover - supports python -m scripts.smoke_acot_long_chunk
    from scripts import train as train_lib


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "acot_libero_long_chunk_h15"
    checkpoint_params: str | None = None
    batch_size: int = 1
    inference_steps: int = 1
    output_json: str | None = None


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


def _validate_args(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.inference_steps <= 0:
        raise ValueError("inference_steps must be positive.")


def main(args: Args) -> None:
    _validate_args(args)
    base_config = config_lib.get_config(args.config_name)
    if base_config.prefix_retention_loss_weight <= 0:
        raise ValueError(f"Config {args.config_name!r} does not enable prefix retention.")
    if base_config.model.model_type not in {
        model_lib.ModelType.ACOT_VLA_PI0,
        model_lib.ModelType.ACOT_VLA_PI05,
    }:
        raise ValueError("Long-chunk smoke requires an ACoT-VLA config.")

    config = dataclasses.replace(
        base_config,
        exp_name="long_chunk_smoke",
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
    )
    if args.checkpoint_params is not None:
        anchor_loader = weight_loaders.CheckpointWeightLoader(args.checkpoint_params)
        config = dataclasses.replace(
            config,
            weight_loader=anchor_loader,
            prefix_retention_teacher_weight_loader=anchor_loader,
        )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )
    loader = data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=1,
    )
    batch = next(iter(loader))
    observation, actions, coarse_actions = batch[:3]
    expected_actions = (args.batch_size, config.model.action_horizon, config.model.action_dim)
    expected_coarse = (
        args.batch_size,
        config.model.coarse_action_horizon,
        config.model.action_dim,
    )
    if actions.shape != expected_actions:
        raise ValueError(f"Final-action batch shape mismatch: expected {expected_actions}, got {actions.shape}.")
    if coarse_actions.shape != expected_coarse:
        raise ValueError(f"Coarse-action batch shape mismatch: expected {expected_coarse}, got {coarse_actions.shape}.")

    init_rng = jax.random.key(config.seed)
    train_state, train_state_sharding = train_lib.init_train_state(
        config,
        init_rng,
        mesh,
        resume=False,
    )
    jax.block_until_ready(train_state)
    params_shape = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
        train_state.params,
    )
    reference_params, reference_sharding = train_lib.init_prefix_retention_reference_params(
        config,
        jax.random.fold_in(init_rng, 17_015),
        mesh,
        params_shape,
        train_state_sharding.params,
    )
    jax.block_until_ready(reference_params)

    validation_step = jax.jit(
        functools.partial(train_lib.acot_validation_step, config),
        in_shardings=(
            replicated_sharding,
            train_state_sharding,
            data_sharding,
            reference_sharding,
        ),
        out_shardings=replicated_sharding,
    )
    validation_metrics = validation_step(
        jax.random.key(config.seed + 1),
        train_state,
        batch,
        reference_params,
    )
    jax.block_until_ready(validation_metrics)
    metric_values = {name: _json_value(value) for name, value in validation_metrics.items()}
    if not all(np.isfinite(value) for value in metric_values.values()):
        raise FloatingPointError(f"Long-chunk loss smoke returned non-finite metrics: {metric_values}")

    evaluation_params = train_state.ema_params if train_state.ema_params is not None else train_state.params

    def sample(params: nnx.State, rng: jax.Array, obs: model_lib.Observation):
        model = nnx.merge(train_state.model_def, params)
        model.eval()
        return model.sample_actions(
            rng,
            obs,
            num_steps=args.inference_steps,
            action_cot_denoising_steps=args.inference_steps,
            final_denoising_steps=args.inference_steps,
        )

    sample_forward = jax.jit(
        sample,
        in_shardings=(train_state_sharding.params, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    sampled = sample_forward(evaluation_params, jax.random.key(config.seed + 2), observation)
    jax.block_until_ready(sampled)
    sampled_actions = np.asarray(sampled["actions"] if isinstance(sampled, dict) else sampled)
    if sampled_actions.shape != expected_actions:
        raise ValueError(f"Sampling forward shape mismatch: expected {expected_actions}, got {sampled_actions.shape}.")
    if not np.all(np.isfinite(sampled_actions)):
        raise FloatingPointError("Sampling forward returned non-finite actions.")

    joint_action_shifts = tuple(loader.data_config().joint_action_shifts)
    requested_action_window = max(
        (config.model.coarse_action_horizon - 1) * int(joint_action_shifts[0]) + 1,
        (config.model.action_horizon - 1) * int(joint_action_shifts[1]) + 1,
    )
    summary = {
        "status": "passed",
        "config_name": args.config_name,
        "action_horizon": config.model.action_horizon,
        "coarse_action_horizon": config.model.coarse_action_horizon,
        "joint_action_shifts": list(joint_action_shifts),
        "requested_source_action_window": requested_action_window,
        "batch_shapes": {
            "actions": list(actions.shape),
            "coarse_actions": list(coarse_actions.shape),
        },
        "checkpoint_restore": "passed",
        "loss_forward": metric_values,
        "sampling_forward_shape": list(sampled_actions.shape),
        "sampling_forward_steps": args.inference_steps,
        "trainable_parameters": training_utils.count_parameters(train_state.params.filter(config.trainable_filter)),
        "anchored_parameters": training_utils.count_parameters(reference_params),
        "optimizer_updates": 0,
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output_json is not None:
        output_path = pathlib.Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized)


if __name__ == "__main__":
    main(tyro.cli(Args))
