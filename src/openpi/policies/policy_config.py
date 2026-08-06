import dataclasses
import logging
import pathlib
import re
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

from openpi.models import contextual_plan_compiler as _contextual_plan_compiler
from openpi.models import es_harp_gripper_event as _es_harp_gripper_event
from openpi.models import harp_temporal_residual as _harp_temporal_residual
import openpi.models.model as _model
from openpi.policies import compact_alpha_router as _compact_alpha_router
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
    r"action_out_proj/.*|"
    r"adaptive_final_time_warp_gate/.*|"
    r"pact_flow_scheduler/.*"
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
    acot_contextual_compiler_params: pathlib.Path | str | None = None,
    acot_compact_alpha_router_params: pathlib.Path | str | None = None,
    acot_harp_residual_params: pathlib.Path | str | None = None,
    acot_harp_gripper_event_params: pathlib.Path | str | None = None,
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
        acot_contextual_compiler_params: Optional independent contextual
            compiler sidecar. A directory is resolved to ``model_params.npz``.
            When present, the sequential Action-CoT path bypasses the final
            300M action expert and uses this compiler instead.
        acot_compact_alpha_router_params: Optional compact ridge-router NPZ.
            It remains inactive until a request explicitly enables it.
        acot_harp_residual_params: Optional tiny temporal residual NPZ. Loading
            it is inert; each request must additionally set
            ``action_cot_harp_residual=True``.
        acot_harp_gripper_event_params: Optional independent ES-HARP gripper
            event NPZ. Loading it is inert; each request must additionally set
            ``action_cot_harp_gripper_event=True``.
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
        flat_sidecar = traverse_util.flatten_dict(sidecar_params)
        flat_sidecar_paths = [_path_string(path) for path in flat_sidecar]
        has_adaptive_final_time_warp = any(
            path.startswith("adaptive_final_time_warp_gate/")
            for path in flat_sidecar_paths
        )
        has_pact_flow_scheduler = any(
            path.startswith("pact_flow_scheduler/") for path in flat_sidecar_paths
        )
        if has_adaptive_final_time_warp and has_pact_flow_scheduler:
            raise ValueError(
                "Endpoint sidecars cannot contain both adaptive_final_time_warp_gate "
                "and pact_flow_scheduler parameters."
            )
        extension_root = None
        if has_adaptive_final_time_warp or has_pact_flow_scheduler:
            if has_adaptive_final_time_warp:
                if not hasattr(model_config, "adaptive_final_time_warp"):
                    raise ValueError("Adaptive final time-warp sidecars require ACOTConfig.")
                model_config = dataclasses.replace(
                    model_config,
                    adaptive_final_time_warp=True,
                )
                extension_root = "adaptive_final_time_warp_gate"
                extension_label = "Adaptive final time-warp"
            else:
                if not hasattr(model_config, "pact_flow_scheduler"):
                    raise ValueError("PACT flow-scheduler sidecars require ACOTConfig.")
                model_config = dataclasses.replace(
                    model_config,
                    pact_flow_scheduler=True,
                )
                extension_root = "pact_flow_scheduler"
                extension_label = "PACT flow-scheduler"
            expected_model = nnx.eval_shape(model_config.create, jax.random.key(0))
            expected_params = nnx.state(expected_model).to_pure_dict()
            flat_base = traverse_util.flatten_dict(base_params)
            flat_expected = traverse_util.flatten_dict(expected_params)
            expected_extension_paths = {
                path
                for path in flat_expected
                if path and path[0] == extension_root
            }
            sidecar_extension_paths = {
                path
                for path in flat_sidecar
                if path and path[0] == extension_root
            }
            missing_extension_paths = expected_extension_paths.difference(
                sidecar_extension_paths
            )
            unexpected_extension_paths = sidecar_extension_paths.difference(
                expected_extension_paths
            )
            if missing_extension_paths:
                raise ValueError(
                    f"{extension_label} sidecar is incomplete: "
                    f"{sorted(_path_string(path) for path in missing_extension_paths)[:5]}"
                )
            if unexpected_extension_paths:
                raise ValueError(
                    f"{extension_label} sidecar contains unexpected parameters: "
                    f"{sorted(_path_string(path) for path in unexpected_extension_paths)[:5]}"
                )
            missing = set(flat_expected).difference(flat_base)
            invalid_missing = sorted(
                _path_string(path)
                for path in missing
                if not path or path[0] != extension_root
            )
            if invalid_missing:
                raise ValueError(
                    "Base checkpoint is missing non-extension parameters: "
                    f"{invalid_missing[:5]}"
                )
            for path in missing:
                flat_base[path] = flat_expected[path]
            base_params = traverse_util.unflatten_dict(flat_base)
        base_params = merge_acot_endpoint_student_params(base_params, sidecar_params)
        has_coarse_student = any(
            path.startswith("coarse_") or re.match(r"^PaliGemma/llm/.*_1(?:/|$)", path)
            for path in flat_sidecar_paths
        )
        has_final_student = any(
            path.startswith(
                (
                    "action_",
                    "time_mlp_",
                    "adaptive_final_time_warp_gate/",
                    "pact_flow_scheduler/",
                )
            )
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

    contextual_compiler = None
    if acot_contextual_compiler_params is not None:
        if not hasattr(model_config, "adopt_explicit_action_reasoner"):
            raise ValueError("Contextual compiler sidecars are only supported by ACOTConfig.")
        compiler_path = pathlib.Path(
            download.maybe_download(str(acot_contextual_compiler_params))
        )
        if compiler_path.is_dir():
            compiler_path = compiler_path / "model_params.npz"
        contextual_compiler = _contextual_plan_compiler.load_contextual_plan_compiler(
            compiler_path
        )
        logging.info("Loaded contextual Action-CoT compiler sidecar from %s", compiler_path)

    compact_alpha_router = None
    if acot_compact_alpha_router_params is not None:
        if acot_endpoint_student_params is None:
            raise ValueError(
                "The compact alpha router was fit for the one-step endpoint student; "
                "load acot_endpoint_student_params in the same server."
            )
        router_path = pathlib.Path(
            download.maybe_download(str(acot_compact_alpha_router_params))
        )
        compact_alpha_router = _compact_alpha_router.load_compact_alpha_router(router_path)
        logging.info("Loaded compact ACoT alpha router from %s", router_path)

    harp_residual = None
    if acot_harp_residual_params is not None:
        if acot_endpoint_student_params is None:
            raise ValueError(
                "HARP was trained on deployed one-step IR endpoints; load "
                "acot_endpoint_student_params in the same server."
            )
        harp_path = pathlib.Path(
            download.maybe_download(str(acot_harp_residual_params))
        )
        harp_residual = _harp_temporal_residual.load_harp_residual_sidecar(harp_path)
        logging.info(
            "Loaded inert HARP residual sidecar from %s (%s parameters)",
            harp_path,
            harp_residual.parameter_count,
        )

    harp_gripper_event = None
    if acot_harp_gripper_event_params is not None:
        if acot_endpoint_student_params is None:
            raise ValueError(
                "ES-HARP was trained on deployed one-step IR endpoints; load "
                "acot_endpoint_student_params in the same server."
            )
        gripper_path = pathlib.Path(
            download.maybe_download(str(acot_harp_gripper_event_params))
        )
        harp_gripper_event = _es_harp_gripper_event.load_gripper_event_sidecar(
            gripper_path
        )
        logging.info(
            "Loaded inert ES-HARP gripper-event sidecar from %s (%s parameters)",
            gripper_path,
            harp_gripper_event.parameter_count,
        )

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
        acot_contextual_compiler=contextual_compiler,
        acot_compact_alpha_router=compact_alpha_router,
        acot_harp_residual=harp_residual,
        acot_harp_gripper_event=harp_gripper_event,
    )
