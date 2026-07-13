import dataclasses
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    execution_horizon_predictor_params: pathlib.Path | str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model_config = train_config.model
    base_params = _model.convert_str_keys_to_int(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )
    if execution_horizon_predictor_params is not None:
        if not hasattr(model_config, "execution_horizon_predictor"):
            raise ValueError("Execution-horizon sidecars are only supported by ACOTConfig.")
        model_config = dataclasses.replace(model_config, execution_horizon_predictor=True)
        expected_model = nnx.eval_shape(model_config.create, jax.random.key(0))
        expected_params = nnx.state(expected_model).to_pure_dict()
        flat_merged = traverse_util.flatten_dict(base_params)
        flat_expected = traverse_util.flatten_dict(expected_params)
        missing = set(flat_expected).difference(flat_merged)
        invalid_missing = sorted(
            "/".join(map(str, key))
            for key in missing
            if not key or key[0] != "execution_horizon_predictor"
        )
        if invalid_missing:
            raise ValueError(f"Base checkpoint is missing non-predictor parameters: {invalid_missing[:5]}")
        for key in missing:
            flat_merged[key] = flat_expected[key]
        sidecar_path = download.maybe_download(str(execution_horizon_predictor_params))
        sidecar_params = _model.convert_str_keys_to_int(
            _model.restore_params(sidecar_path, dtype=jnp.float32)
        )
        if "execution_horizon_predictor" not in sidecar_params:
            sidecar_params = {"execution_horizon_predictor": sidecar_params}
        flat_sidecar = traverse_util.flatten_dict(sidecar_params)
        unexpected = sorted("/".join(map(str, key)) for key in flat_sidecar if key not in flat_merged)
        if unexpected:
            raise ValueError(f"Unexpected execution-horizon sidecar parameters: {unexpected[:5]}")
        for key, value in flat_sidecar.items():
            expected = flat_merged[key]
            if expected.shape != value.shape:
                path = "/".join(map(str, key))
                raise ValueError(f"Sidecar shape mismatch at {path}: expected {expected.shape}, got {value.shape}")
            flat_merged[key] = value.astype(expected.dtype)
        base_params = traverse_util.unflatten_dict(flat_merged)
        logging.info("Loaded execution-horizon predictor sidecar from %s", sidecar_path)
    model = model_config.load(base_params)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantile_norm=data_config.use_quantile_norm,
        action_dim=model_config.action_dim,
    )
