import dataclasses
import logging
import pathlib
import re
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

_ACOT_ENDPOINT_STUDENT_PATH = re.compile(
    r"^(?:"
    r"PaliGemma/llm/.*_(?:1|2)(?:/.*)?|"
    r"coarse_action_in_proj/.*|"
    r"coarse_time_mlp_in/.*|"
    r"coarse_time_mlp_out/.*|"
    r"coarse_action_time_mlp_in/.*|"
    r"coarse_action_time_mlp_out/.*|"
    r"coarse_action_out_proj/.*|"
    r"action_in_proj/.*|"
    r"time_mlp_in/.*|"
    r"time_mlp_out/.*|"
    r"action_time_mlp_in/.*|"
    r"action_time_mlp_out/.*|"
    r"action_out_proj/.*"
    r")$"
)


def _path_string(path: tuple[Any, ...]) -> str:
    return "/".join(map(str, path))


def merge_acot_endpoint_student_params(
    base_params: dict[str, Any],
    student_params: dict[str, Any],
) -> dict[str, Any]:
    """Replace only the EAR/final expert parameters allowed by the fast student.

    Endpoint checkpoints are deliberately delta sidecars: they must match an
    existing parameter in the base checkpoint and may not introduce a new
    module. Keeping this validation here prevents an accidentally broad
    fine-tuning checkpoint from silently replacing the frozen VLM, IAR, or
    fusion modules at serving time.
    """

    flat_base = traverse_util.flatten_dict(base_params)
    flat_student = traverse_util.flatten_dict(student_params)
    if not flat_student:
        raise ValueError("ACoT endpoint student sidecar contains no parameters.")

    disallowed = sorted(
        _path_string(path)
        for path in flat_student
        if _ACOT_ENDPOINT_STUDENT_PATH.fullmatch(_path_string(path)) is None
    )
    if disallowed:
        raise ValueError(f"Disallowed ACoT endpoint student parameters: {disallowed[:5]}")

    unexpected = sorted(_path_string(path) for path in flat_student if path not in flat_base)
    if unexpected:
        raise ValueError(f"Unexpected ACoT endpoint student parameters: {unexpected[:5]}")

    for path, value in flat_student.items():
        expected = flat_base[path]
        if expected.shape != value.shape:
            path_text = _path_string(path)
            raise ValueError(
                f"ACoT endpoint sidecar shape mismatch at {path_text}: "
                f"expected {expected.shape}, got {value.shape}"
            )
        flat_base[path] = value.astype(expected.dtype)
    return traverse_util.unflatten_dict(flat_base)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    execution_horizon_predictor_params: pathlib.Path | str | None = None,
    acot_endpoint_student_params: pathlib.Path | str | None = None,
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
        acot_endpoint_student_params: Optional delta sidecar containing the
            one-step EAR/final endpoint student. When present, inference
            defaults to one EAR step and one final-action step.
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

    if acot_endpoint_student_params is not None:
        if not hasattr(model_config, "adopt_explicit_action_reasoner"):
            raise ValueError("ACoT endpoint student sidecars are only supported by ACOTConfig.")
        sidecar_path = download.maybe_download(str(acot_endpoint_student_params))
        sidecar_params = _model.convert_str_keys_to_int(
            _model.restore_params(sidecar_path, dtype=jnp.bfloat16)
        )
        flat_sidecar_paths = [
            _path_string(path) for path in traverse_util.flatten_dict(sidecar_params)
        ]
        base_params = merge_acot_endpoint_student_params(base_params, sidecar_params)
        has_coarse_student = any(
            path.startswith("coarse_") or re.match(r"^PaliGemma/llm/.*_1(?:/|$)", path)
            for path in flat_sidecar_paths
        )
        has_final_student = any(
            path.startswith(("action_", "time_mlp_"))
            or re.match(r"^PaliGemma/llm/.*_2(?:/|$)", path)
            for path in flat_sidecar_paths
        )
        if has_coarse_student and has_final_student:
            # Explicit caller values remain authoritative. This makes the
            # sidecar safe for 2/3/... step ablations while selecting the fast
            # path by default for a complete EAR+final student.
            sample_kwargs = {
                "num_steps": 1,
                "action_cot_denoising_steps": 1,
                **(sample_kwargs or {}),
            }
        else:
            logging.warning(
                "Endpoint sidecar contains only the %s branch; denoising defaults were not changed.",
                "coarse" if has_coarse_student else "final",
            )
        logging.info("Loaded one-step ACoT endpoint student sidecar from %s", sidecar_path)
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
