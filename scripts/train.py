import dataclasses
import functools
import json
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def init_prefix_retention_reference_params(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    params_shape: nnx.State,
    params_sharding: nnx.State,
) -> tuple[nnx.State, nnx.State]:
    """Restore the frozen anchor subset used by long-chunk prefix retention."""

    loader = config.prefix_retention_teacher_weight_loader
    if config.prefix_retention_loss_weight <= 0 or loader is None:
        raise ValueError("Prefix-retention reference initialization requires an enabled anchor loader.")
    partial_params = _load_weights_and_validate(loader, params_shape.to_pure_dict())
    reference_sharding = params_sharding.filter(config.trainable_filter)

    def init(rng: at.KeyArrayLike, loaded_params: at.Params) -> nnx.State:
        model = config.model.create(rng)
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(loaded_params)
        model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return params.filter(config.trainable_filter)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    reference_params = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=reference_sharding,
    )(init_rng, partial_params)
    return reference_params, reference_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
    prefix_retention_reference_params: nnx.State | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    reference_model = None
    if config.prefix_retention_loss_weight > 0:
        if prefix_retention_reference_params is None:
            raise ValueError("Enabled prefix retention is missing frozen reference parameters.")
        reference_model = nnx.merge(state.model_def, state.params)
        nnx.update(reference_model, prefix_retention_reference_params)
        reference_model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        action_cot_skip_mask=None,
        action_cot_skip_valid_mask=None,
        action_cot_step_label=None,
    ):
        if config.prefix_retention_loss_weight > 0:
            if reference_model is None:
                raise ValueError("Prefix-retention reference model was not constructed.")
            return model.compute_long_chunk_loss(
                reference_model,
                rng,
                observation,
                actions,
                coarse_actions,
                prefix_horizon=config.prefix_retention_horizon,
                prefix_retention_weight=config.prefix_retention_loss_weight,
                train=True,
            )
        loss = model.compute_loss(
            rng,
            observation,
            actions,
            coarse_actions,
            action_cot_skip_mask=action_cot_skip_mask,
            action_cot_skip_valid_mask=action_cot_skip_valid_mask,
            action_cot_step_label=action_cot_step_label,
            train=True,
        )
        return loss, {}

    train_rng = jax.random.fold_in(rng, state.step)
    if len(batch) == 6:
        (
            observation,
            actions,
            coarse_actions,
            action_cot_skip_mask,
            action_cot_skip_valid_mask,
            action_cot_step_label,
        ) = batch
    elif len(batch) == 5:
        observation, actions, coarse_actions, action_cot_skip_mask, action_cot_skip_valid_mask = batch
        action_cot_step_label = None
    else:
        observation, actions, coarse_actions = batch
        action_cot_skip_mask = None
        action_cot_skip_valid_mask = None
        action_cot_step_label = None

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(
        loss_fn,
        argnums=diff_state,
        has_aux=True,
    )(
        model,
        train_rng,
        observation,
        actions,
        coarse_actions,
        action_cot_skip_mask,
        action_cot_skip_valid_mask,
        action_cot_step_label,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **loss_metrics,
    }
    return new_state, info


@at.typecheck
def acot_validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
    prefix_retention_reference_params: nnx.State | None = None,
) -> dict[str, at.Array]:
    """Evaluate the exact inference-parameter source used by checkpoint saves."""

    evaluation_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, evaluation_params)
    model.eval()
    reference_model = None
    if config.prefix_retention_loss_weight > 0:
        if prefix_retention_reference_params is None:
            raise ValueError("Enabled prefix retention is missing frozen reference parameters.")
        reference_model = nnx.merge(state.model_def, evaluation_params)
        nnx.update(reference_model, prefix_retention_reference_params)
        reference_model.eval()

    observation, actions, coarse_actions = batch[:3]
    if config.prefix_retention_loss_weight > 0:
        if reference_model is None:
            raise ValueError("Prefix-retention reference model was not constructed.")
        loss, metrics = model.compute_long_chunk_loss(
            reference_model,
            rng,
            observation,
            actions,
            coarse_actions,
            prefix_horizon=config.prefix_retention_horizon,
            prefix_retention_weight=config.prefix_retention_loss_weight,
            train=False,
        )
    else:
        loss = model.compute_loss(rng, observation, actions, coarse_actions, train=False)
        metrics = {}
    return {
        "validation_loss": loss,
        **{f"validation_{name}": value for name, value in metrics.items()},
    }


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    validation_enabled = config.validation_fraction > 0
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        episode_split="train" if validation_enabled else None,
        validation_fraction=config.validation_fraction,
    )
    validation_loader = None
    if validation_enabled:
        validation_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.validation_batches,
            episode_split="validation",
            validation_fraction=config.validation_fraction,
        )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    is_acot = config.model.model_type in {
        _model.ModelType.ACOT_VLA_PI05,
        _model.ModelType.ACOT_VLA_PI0,
    }
    prefix_retention_reference_params = None
    prefix_retention_reference_sharding = None
    if config.prefix_retention_loss_weight > 0:
        if not is_acot:
            raise ValueError("Prefix-retention training requires an ACoT-VLA model.")
        params_shape = jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
            train_state.params,
        )
        prefix_retention_reference_params, prefix_retention_reference_sharding = init_prefix_retention_reference_params(
            config,
            jax.random.fold_in(init_rng, 17_015),
            mesh,
            params_shape,
            train_state_sharding.params,
        )
        logging.info(
            "Loaded frozen prefix-retention anchor with %d parameters",
            training_utils.count_parameters(prefix_retention_reference_params),
        )

    pvalidation_step = None
    if is_acot and prefix_retention_reference_params is not None:
        assert prefix_retention_reference_sharding is not None
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(
                replicated_sharding,
                train_state_sharding,
                data_sharding,
                prefix_retention_reference_sharding,
            ),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        if validation_enabled:
            pvalidation_step = jax.jit(
                functools.partial(acot_validation_step, config),
                in_shardings=(
                    replicated_sharding,
                    train_state_sharding,
                    data_sharding,
                    prefix_retention_reference_sharding,
                ),
                out_shardings=replicated_sharding,
            )
    elif is_acot:
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        if validation_enabled:
            pvalidation_step = jax.jit(
                functools.partial(acot_validation_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=replicated_sharding,
            )
    else:
        if validation_enabled:
            raise ValueError("Episode-disjoint validation is currently implemented only for ACoT-VLA training.")
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    best_validation_loss = float("inf")
    best_validation_step: int | None = None
    validation_checks_without_improvement = 0
    validation_best_path = config.checkpoint_dir / "validation_best.json"
    if resuming and validation_enabled and validation_best_path.exists():
        saved_best = json.loads(validation_best_path.read_text())
        best_validation_loss = float(saved_best["best_validation_loss"])
        best_validation_step = int(saved_best["best_step"])
        validation_checks_without_improvement = int(saved_best.get("checks_without_improvement", 0))

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if prefix_retention_reference_params is None:
                train_state, info = ptrain_step(train_rng, train_state, batch)
            else:
                train_state, info = ptrain_step(
                    train_rng,
                    train_state,
                    batch,
                    prefix_retention_reference_params,
                )
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        saved_this_step = False
        checkpoint_step = int(train_state.step)
        should_stop_early = False
        if validation_enabled and checkpoint_step % config.validation_interval == 0:
            if validation_loader is None or pvalidation_step is None:
                raise ValueError("Validation was enabled but its loader or compiled step is missing.")
            validation_infos = []
            for validation_index, validation_batch in enumerate(validation_loader):
                validation_rng = jax.random.fold_in(
                    jax.random.key(config.seed + 1_000_003),
                    validation_index,
                )
                with sharding.set_mesh(mesh):
                    if prefix_retention_reference_params is None:
                        validation_info = pvalidation_step(
                            validation_rng,
                            train_state,
                            validation_batch,
                        )
                    else:
                        validation_info = pvalidation_step(
                            validation_rng,
                            train_state,
                            validation_batch,
                            prefix_retention_reference_params,
                        )
                validation_infos.append(validation_info)
            reduced_validation = jax.device_get(
                jax.tree.map(
                    jnp.mean,
                    common_utils.stack_forest(validation_infos),
                )
            )
            validation_loss = float(reduced_validation["validation_loss"])
            if not np.isfinite(validation_loss):
                raise FloatingPointError(f"Validation loss is non-finite at step {checkpoint_step}: {validation_loss}")
            validation_text = ", ".join(f"{name}={value:.4f}" for name, value in reduced_validation.items())
            pbar.write(f"Validation step {checkpoint_step}: {validation_text}")
            wandb.log(reduced_validation, step=checkpoint_step)

            improved = validation_loss < best_validation_loss - config.validation_min_delta
            if improved:
                best_validation_loss = validation_loss
                best_validation_step = checkpoint_step
                validation_checks_without_improvement = 0
                _checkpoints.save_state(
                    checkpoint_manager,
                    train_state,
                    data_loader,
                    checkpoint_step,
                )
                saved_this_step = True
            else:
                validation_checks_without_improvement += 1

            validation_best_path.write_text(
                json.dumps(
                    {
                        "best_step": best_validation_step,
                        "best_validation_loss": best_validation_loss,
                        "checks_without_improvement": validation_checks_without_improvement,
                        "last_validation_step": checkpoint_step,
                        "last_validation_loss": validation_loss,
                        "selection_params": "ema" if train_state.ema_params is not None else "online",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            should_stop_early = (
                config.early_stopping_patience is not None
                and validation_checks_without_improvement >= config.early_stopping_patience
            )

        if validation_enabled:
            scheduled_save = checkpoint_step % config.save_interval == 0
            final_save = checkpoint_step == config.num_train_steps
            if (scheduled_save or final_save) and not saved_this_step:
                _checkpoints.save_state(
                    checkpoint_manager,
                    train_state,
                    data_loader,
                    checkpoint_step,
                )
        elif (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # Preserve the checkpoint numbering and cadence of every legacy config.
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        if should_stop_early:
            logging.info(
                "Early stopping at step %d after %d validation checks without improvement; best step=%s loss=%.6f",
                checkpoint_step,
                validation_checks_without_improvement,
                best_validation_step,
                best_validation_loss,
            )
            break

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
