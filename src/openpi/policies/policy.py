from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import contextual_plan_compiler as _contextual_plan_compiler
from openpi.models import es_harp_gripper_event as _es_harp_gripper_event
from openpi.models import harp_temporal_residual as _harp_temporal_residual
from openpi.models import model as _model
from openpi.policies import compact_alpha_router as _compact_alpha_router
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _as_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


def _block_until_ready(value: Any) -> None:
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


@jax.jit
def _fuse_contextual_semantic_actions(
    compiled_actions: jax.Array,
    expert_actions: jax.Array,
    translation_tau: jax.Array,
    rotation_tau: jax.Array,
    gripper_tau: jax.Array,
    gate_width: jax.Array,
    high_source_expert: jax.Array,
) -> dict[str, jax.Array]:
    """Fuse compiler/expert chunks with semantic, temporally coherent gates.

    The two branches have already consumed the same observation, EAR, IAR, and
    final-action noise.  Consequently this verifier adds no model evaluation:
    it only compares their normalized action chunks.  Translation and rotation
    each receive one gate per timestep so a vector is never mixed dimension by
    dimension.  Discrete gripper events use a hard branch choice rather than an
    invalid interpolation across open/close signs.
    """

    disagreement = jnp.abs(expert_actions[..., :7] - compiled_actions[..., :7])

    def _dilate_one_step(values: jax.Array) -> jax.Array:
        previous = jnp.concatenate((values[:, :1], values[:, :-1]), axis=1)
        following = jnp.concatenate((values[:, 1:], values[:, -1:]), axis=1)
        return jnp.maximum(jnp.maximum(previous, values), following)

    def _continuous_gate(start: int, end: int, tau: jax.Array) -> jax.Array:
        # Group RMS consumes every per-dimension disagreement while preserving
        # the geometry of the translation/rotation vector during blending.
        score = jnp.sqrt(
            jnp.mean(jnp.square(disagreement[..., start:end]), axis=-1, keepdims=True)
            + jnp.asarray(1e-8, dtype=disagreement.dtype)
        )
        score = _dilate_one_step(score)
        width = jnp.maximum(
            jnp.asarray(gate_width, dtype=score.dtype),
            jnp.asarray(1e-4, dtype=score.dtype),
        )
        return jax.nn.sigmoid((score - jnp.asarray(tau, dtype=score.dtype)) / width)

    translation_weight = _continuous_gate(0, 3, translation_tau)
    rotation_weight = _continuous_gate(3, 6, rotation_tau)

    expert_gripper = expert_actions[..., 6:7]
    compiled_gripper = compiled_actions[..., 6:7]
    sign_disagreement = jnp.signbit(expert_gripper) != jnp.signbit(compiled_gripper)
    expert_transition = jnp.concatenate(
        (
            jnp.zeros_like(sign_disagreement[:, :1]),
            jnp.signbit(expert_gripper[:, 1:]) != jnp.signbit(expert_gripper[:, :-1]),
        ),
        axis=1,
    )
    compiler_transition = jnp.concatenate(
        (
            jnp.zeros_like(sign_disagreement[:, :1]),
            jnp.signbit(compiled_gripper[:, 1:]) != jnp.signbit(compiled_gripper[:, :-1]),
        ),
        axis=1,
    )
    low_gripper_margin = jnp.minimum(
        jnp.abs(expert_gripper), jnp.abs(compiled_gripper)
    ) < jnp.asarray(gripper_tau, dtype=expert_gripper.dtype)
    gripper_event = sign_disagreement | expert_transition | compiler_transition | low_gripper_margin
    gripper_event = _dilate_one_step(gripper_event)

    # Contact/open-close boundaries are semantic events for the continuous
    # controls too.  Use the configured high-disagreement source throughout a
    # one-step neighborhood instead of creating a mixed contact transition.
    event_weight = gripper_event.astype(translation_weight.dtype)
    translation_weight = jnp.maximum(translation_weight, event_weight)
    rotation_weight = jnp.maximum(rotation_weight, event_weight)

    high_actions = jnp.where(high_source_expert, expert_actions, compiled_actions)
    low_actions = jnp.where(high_source_expert, compiled_actions, expert_actions)
    translation = low_actions[..., :3] + translation_weight * (
        high_actions[..., :3] - low_actions[..., :3]
    )
    rotation = low_actions[..., 3:6] + rotation_weight * (
        high_actions[..., 3:6] - low_actions[..., 3:6]
    )
    gripper = jnp.where(gripper_event, high_actions[..., 6:7], low_actions[..., 6:7])
    actions = jnp.concatenate(
        (translation, rotation, gripper, expert_actions[..., 7:]), axis=-1
    )
    expert_translation_weight = jnp.where(
        high_source_expert, translation_weight, 1.0 - translation_weight
    )
    expert_rotation_weight = jnp.where(
        high_source_expert, rotation_weight, 1.0 - rotation_weight
    )
    expert_gripper_gate = jnp.where(
        high_source_expert, gripper_event, ~gripper_event
    ).astype(translation_weight.dtype)
    expert_gate_rate = (
        3.0 * jnp.mean(expert_translation_weight, axis=(1, 2))
        + 3.0 * jnp.mean(expert_rotation_weight, axis=(1, 2))
        + jnp.mean(expert_gripper_gate, axis=(1, 2))
    ) / 7.0
    return {
        "actions": actions,
        "contextual_fusion_translation_disagreement_mean": jnp.mean(
            disagreement[..., :3], axis=(1, 2)
        ),
        "contextual_fusion_rotation_disagreement_mean": jnp.mean(
            disagreement[..., 3:6], axis=(1, 2)
        ),
        "contextual_fusion_gripper_conflict_rate": jnp.mean(
            sign_disagreement.astype(disagreement.dtype), axis=(1, 2)
        ),
        "contextual_fusion_expert_gate_rate": expert_gate_rate,
        "contextual_fusion_high_disagreement_source_expert": jnp.full(
            (actions.shape[0],), high_source_expert, dtype=jnp.bool_
        ),
    }


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantile_norm: bool = False,
        action_dim: int | None = None,
        acot_contextual_compiler: _contextual_plan_compiler.ContextualPlanCompiler | None = None,
        acot_compact_alpha_router: _compact_alpha_router.CompactAlphaRouter | None = None,
        acot_harp_residual: _harp_temporal_residual.HARPResidualSidecar | None = None,
        acot_harp_gripper_event: _es_harp_gripper_event.GripperEventSidecar | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._norm_stats = norm_stats
        self._use_quantile_norm = use_quantile_norm
        self._action_dim = action_dim or model.action_dim
        self._acot_contextual_compiler = acot_contextual_compiler
        self._acot_compact_alpha_router = acot_compact_alpha_router
        self._acot_harp_residual = acot_harp_residual
        self._apply_harp_residual = (
            jax.jit(acot_harp_residual.predict_and_correct)
            if acot_harp_residual is not None
            else None
        )
        self._acot_harp_gripper_event = acot_harp_gripper_event
        self._apply_harp_gripper_event = (
            jax.jit(acot_harp_gripper_event.predict_and_correct)
            if acot_harp_gripper_event is not None
            else None
        )
        self._sample_actions_profile_prefix = None
        self._sample_actions_profile_implicit = None
        self._sample_actions_profile_coarse = None
        self._sample_actions_profile_expert = None
        self._sample_actions_profile_direct_one_step_expert = None
        self._sample_actions_profile_direct_endpoint_conditioned_one_step_expert = None
        self._sample_actions_profile_second_half_expert = None
        self._sample_actions_profile_midpoint_expert = None
        self._sample_actions_profile_adaptive_one_step_expert = None
        self._sample_actions_profile_ofp_expert = None
        self._sample_actions_joint_coupled = None
        self._sample_actions_batched_mc = None
        self._predict_execution_horizon = None
        if hasattr(model, "sample_actions_joint_coupled"):
            self._sample_actions_joint_coupled = nnx_utils.module_jit(model.sample_actions_joint_coupled)
        if hasattr(model, "sample_actions_batched_mc"):
            self._sample_actions_batched_mc = nnx_utils.module_jit(model.sample_actions_batched_mc)
        if getattr(model, "execution_horizon_predictor_enabled", False):
            self._predict_execution_horizon = nnx_utils.module_jit(model.predict_execution_horizon)
        if all(
            hasattr(model, name)
            for name in (
                "sample_actions_profile_prefix",
                "sample_actions_profile_implicit",
                "sample_actions_profile_coarse",
                "sample_actions_profile_expert",
            )
        ):
            self._sample_actions_profile_prefix = nnx_utils.module_jit(model.sample_actions_profile_prefix)
            self._sample_actions_profile_implicit = nnx_utils.module_jit(model.sample_actions_profile_implicit)
            self._sample_actions_profile_coarse = nnx_utils.module_jit(model.sample_actions_profile_coarse)
            self._sample_actions_profile_expert = nnx_utils.module_jit(model.sample_actions_profile_expert)
        if hasattr(model, "sample_actions_profile_direct_one_step_expert"):
            self._sample_actions_profile_direct_one_step_expert = nnx_utils.module_jit(
                model.sample_actions_profile_direct_one_step_expert,
                # module_jit prepends module state: alpha is argument four.
                static_argnums=(4,),
            )
        if hasattr(model, "sample_actions_profile_direct_endpoint_conditioned_one_step_expert"):
            self._sample_actions_profile_direct_endpoint_conditioned_one_step_expert = (
                nnx_utils.module_jit(
                    model.sample_actions_profile_direct_endpoint_conditioned_one_step_expert,
                    # module_jit prepends module state: strength is argument four.
                    static_argnums=(4,),
                )
            )
        if hasattr(model, "sample_actions_profile_second_half_expert"):
            self._sample_actions_profile_second_half_expert = nnx_utils.module_jit(
                model.sample_actions_profile_second_half_expert,
                # module_jit prepends module state: alpha is argument five.
                static_argnums=(5,),
            )
        if hasattr(model, "sample_actions_profile_midpoint_expert"):
            self._sample_actions_profile_midpoint_expert = nnx_utils.module_jit(
                model.sample_actions_profile_midpoint_expert,
                # module_jit prepends module state: alpha is argument four.
                static_argnums=(4,),
            )
        # Compact routing deliberately keeps the original two static endpoint
        # graphs. The first routed request warms both so a later alpha switch
        # cannot inject a second compilation into a measured episode.
        self._acot_compact_alpha_static_graphs_warmed = False
        if getattr(model, "adaptive_final_time_warp", False):
            self._sample_actions_profile_adaptive_one_step_expert = nnx_utils.module_jit(
                model.sample_actions_profile_adaptive_one_step_expert
            )
        if hasattr(model, "sample_actions_profile_ofp_expert"):
            self._sample_actions_profile_ofp_expert = nnx_utils.module_jit(
                model.sample_actions_profile_ofp_expert,
                # Internal jitted function prepends module state, making the
                # mode the eighth positional argument.  Keeping it static
                # lets time_blend compile a graph with only one posemb call.
                static_argnums=(8,),
            )
        if self._acot_contextual_compiler is not None and not self._can_profile_sample_actions():
            raise ValueError(
                "A contextual Action-CoT compiler requires the model's sequential profile entrypoints."
            )
        if self._acot_compact_alpha_router is not None and (
            not self._can_profile_sample_actions()
            or self._sample_actions_profile_direct_one_step_expert is None
        ):
            raise ValueError(
                "A compact alpha router requires the model's sequential direct one-step entrypoint."
            )
        if self._acot_harp_residual is not None and (
            not self._can_profile_sample_actions()
            or self._sample_actions_profile_direct_one_step_expert is None
        ):
            raise ValueError(
                "A HARP sidecar requires the sequential direct one-step Action-CoT entrypoint."
            )
        if self._acot_harp_gripper_event is not None and (
            not self._can_profile_sample_actions()
            or self._sample_actions_profile_direct_one_step_expert is None
        ):
            raise ValueError(
                "An ES-HARP gripper sidecar requires the sequential direct one-step Action-CoT entrypoint."
            )

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        policy_seed = inputs.pop("policy_seed", None)
        coarse_actions_override = inputs.pop("coarse_actions_override", None)
        action_cot_skip_segment = inputs.pop("action_cot_skip_segment", None)
        profile_policy_timing = _as_bool(inputs.pop("profile_policy_timing", False))
        export_acot_cache = _as_bool(inputs.pop("export_acot_cache", False))
        compact_alpha_router_enabled = _as_bool(
            inputs.pop("action_cot_compact_alpha_router", False)
        )
        contextual_fusion_mode = str(
            np.asarray(inputs.pop("action_cot_contextual_fusion_mode", "compiler")).item()
        )
        if contextual_fusion_mode not in {
            "compiler",
            "expert",
            "control_compiler",
            "gripper_compiler",
            "blend50",
            "semantic_gate",
            "phase_compiler_expert",
            "phase_expert_compiler",
        }:
            raise ValueError("Unsupported action_cot_contextual_fusion_mode.")
        if self._acot_contextual_compiler is None and contextual_fusion_mode != "compiler":
            raise ValueError(
                "Contextual fusion requires a contextual plan compiler at server startup."
            )
        contextual_fusion_translation_tau = float(
            np.asarray(inputs.pop("action_cot_contextual_fusion_translation_tau", 0.20)).item()
        )
        contextual_fusion_rotation_tau = float(
            np.asarray(inputs.pop("action_cot_contextual_fusion_rotation_tau", 0.20)).item()
        )
        contextual_fusion_gripper_tau = float(
            np.asarray(inputs.pop("action_cot_contextual_fusion_gripper_tau", 0.15)).item()
        )
        contextual_fusion_gate_width = float(
            np.asarray(inputs.pop("action_cot_contextual_fusion_gate_width", 0.05)).item()
        )
        contextual_fusion_high_disagreement_source = str(
            np.asarray(
                inputs.pop(
                    "action_cot_contextual_fusion_high_disagreement_source", "expert"
                )
            ).item()
        )
        if contextual_fusion_high_disagreement_source not in {"expert", "compiler"}:
            raise ValueError(
                "action_cot_contextual_fusion_high_disagreement_source must be "
                "'expert' or 'compiler'."
            )
        for name, value in (
            ("translation_tau", contextual_fusion_translation_tau),
            ("rotation_tau", contextual_fusion_rotation_tau),
            ("gripper_tau", contextual_fusion_gripper_tau),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"action_cot_contextual_fusion_{name} must be finite and non-negative.")
        if not np.isfinite(contextual_fusion_gate_width) or contextual_fusion_gate_width <= 0.0:
            raise ValueError(
                "action_cot_contextual_fusion_gate_width must be finite and positive."
            )
        contextual_fusion_switch_step_raw = np.asarray(
            inputs.pop("action_cot_contextual_fusion_switch_step", 400)
        ).item()
        contextual_fusion_switch_step = int(contextual_fusion_switch_step_raw)
        if (
            contextual_fusion_switch_step < 0
            or float(contextual_fusion_switch_step)
            != float(contextual_fusion_switch_step_raw)
        ):
            raise ValueError(
                "action_cot_contextual_fusion_switch_step must be a non-negative integer."
            )
        harp_residual_enabled = _as_bool(
            inputs.pop("action_cot_harp_residual", False)
        )
        harp_gripper_event_enabled = _as_bool(
            inputs.pop("action_cot_harp_gripper_event", False)
        )
        final_hybrid_mode = str(
            np.asarray(inputs.pop("action_cot_final_hybrid_mode", "none")).item()
        )
        if final_hybrid_mode not in {"none", "control_nfe2", "gripper_nfe2"}:
            raise ValueError(
                "action_cot_final_hybrid_mode must be one of "
                "'none', 'control_nfe2', or 'gripper_nfe2'."
            )
        selective_gripper_refinement_enabled = _as_bool(
            inputs.pop("action_cot_selective_gripper_refinement", False)
        )
        selective_gripper_tau = float(
            np.asarray(inputs.pop("action_cot_selective_gripper_tau", 0.15)).item()
        )
        selective_refinement_mode = str(
            np.asarray(inputs.pop("action_cot_selective_refinement_mode", "gripper")).item()
        )
        if selective_refinement_mode not in {"gripper", "full"}:
            raise ValueError(
                "action_cot_selective_refinement_mode must be 'gripper' or 'full'."
            )
        if not np.isfinite(selective_gripper_tau) or not 0.0 <= selective_gripper_tau <= 1.0:
            raise ValueError("action_cot_selective_gripper_tau must be finite and in [0, 1].")
        absolute_decision_step_raw = inputs.pop(
            "action_cot_absolute_decision_step",
            None,
        )
        absolute_decision_step = None
        if absolute_decision_step_raw is not None:
            raw_step = np.asarray(absolute_decision_step_raw).item()
            absolute_decision_step = int(raw_step)
            if absolute_decision_step < 0 or float(absolute_decision_step) != float(raw_step):
                raise ValueError("action_cot_absolute_decision_step must be a non-negative integer.")
        if compact_alpha_router_enabled:
            if self._acot_compact_alpha_router is None:
                raise ValueError(
                    "action_cot_compact_alpha_router=True requires a compact router NPZ at serve startup."
                )
            if absolute_decision_step is None:
                raise ValueError(
                    "action_cot_compact_alpha_router=True requires action_cot_absolute_decision_step."
                )
        if (
            contextual_fusion_mode
            in {"phase_compiler_expert", "phase_expert_compiler"}
            and absolute_decision_step is None
        ):
            raise ValueError(
                "Phase contextual fusion requires action_cot_absolute_decision_step."
            )
        action_cot_denoising_steps = inputs.pop("action_cot_denoising_steps", None)
        action_cot_dynamic_denoising_steps = inputs.pop("action_cot_dynamic_denoising_steps", None)
        final_denoising_steps_raw = inputs.pop("action_cot_final_denoising_steps", None)
        final_denoising_steps = None
        if final_denoising_steps_raw is not None:
            raw_value = np.asarray(final_denoising_steps_raw).item()
            final_denoising_steps = int(raw_value)
            if final_denoising_steps <= 0 or float(final_denoising_steps) != float(raw_value):
                raise ValueError("action_cot_final_denoising_steps must be a positive integer.")
        final_time_warp_alpha = float(
            np.asarray(inputs.pop("action_cot_final_time_warp_alpha", 0.0)).item()
        )
        final_endpoint_condition_strength = float(
            np.asarray(inputs.pop("action_cot_final_endpoint_condition_strength", 0.0)).item()
        )
        final_midpoint_enabled = _as_bool(inputs.pop("action_cot_final_midpoint", False))
        requested_final_time_warp_alpha = final_time_warp_alpha
        adaptive_final_time_warp = _as_bool(
            inputs.pop("action_cot_adaptive_final_time_warp", False)
        )
        ofp_interval_flow = _as_bool(inputs.pop("action_cot_ofp_interval_flow", False))
        ofp_warm_start_actions = inputs.pop("action_cot_ofp_warm_start_actions", None)
        ofp_warm_start_valid = inputs.pop(
            "action_cot_ofp_warm_start_valid",
            ofp_warm_start_actions is not None,
        )
        ofp_warm_start_time = float(
            np.asarray(inputs.pop("action_cot_ofp_warm_start_time", 1.0)).item()
        )
        ofp_interval_condition_strength = float(
            np.asarray(inputs.pop("action_cot_ofp_interval_condition_strength", 1.0)).item()
        )
        ofp_interval_condition_mode = str(
            np.asarray(inputs.pop("action_cot_ofp_interval_condition_mode", "half_concat")).item()
        )
        joint_coupled_sampler = _as_bool(
            inputs.pop("joint_coupled_sampler", self._sample_kwargs.get("joint_coupled_sampler", False))
        )
        batched_mc_samples = int(np.asarray(inputs.pop("batched_mc_samples", 0)).item())
        run_execution_horizon_predictor = _as_bool(inputs.pop("run_execution_horizon_predictor", False))
        previous_actions = inputs.pop("execution_horizon_previous_actions", None)
        previous_h = inputs.pop("execution_horizon_previous_h", 1)
        budget_balance = inputs.pop("execution_horizon_budget_balance", 0.0)
        episode_progress = inputs.pop("execution_horizon_episode_progress", 0.0)
        previous_valid = inputs.pop("execution_horizon_previous_valid", previous_actions is not None)
        transformed_coarse_actions_override = None
        if coarse_actions_override is not None:
            override_inputs = jax.tree.map(lambda x: x, obs)
            override_inputs.pop("coarse_actions_override", None)
            override_inputs.pop("policy_seed", None)
            override_inputs.pop("action_cot_skip_segment", None)
            override_inputs.pop("profile_policy_timing", None)
            override_inputs.pop("export_acot_cache", None)
            override_inputs.pop("action_cot_compact_alpha_router", None)
            override_inputs.pop("action_cot_contextual_fusion_mode", None)
            override_inputs.pop("action_cot_contextual_fusion_translation_tau", None)
            override_inputs.pop("action_cot_contextual_fusion_rotation_tau", None)
            override_inputs.pop("action_cot_contextual_fusion_gripper_tau", None)
            override_inputs.pop("action_cot_contextual_fusion_gate_width", None)
            override_inputs.pop(
                "action_cot_contextual_fusion_high_disagreement_source", None
            )
            override_inputs.pop("action_cot_contextual_fusion_switch_step", None)
            override_inputs.pop("action_cot_harp_residual", None)
            override_inputs.pop("action_cot_harp_gripper_event", None)
            override_inputs.pop("action_cot_final_hybrid_mode", None)
            override_inputs.pop("action_cot_selective_gripper_refinement", None)
            override_inputs.pop("action_cot_selective_gripper_tau", None)
            override_inputs.pop("action_cot_selective_refinement_mode", None)
            override_inputs.pop("action_cot_absolute_decision_step", None)
            override_inputs.pop("action_cot_denoising_steps", None)
            override_inputs.pop("action_cot_dynamic_denoising_steps", None)
            override_inputs.pop("action_cot_final_denoising_steps", None)
            override_inputs.pop("action_cot_final_time_warp_alpha", None)
            override_inputs.pop("action_cot_final_endpoint_condition_strength", None)
            override_inputs.pop("action_cot_final_midpoint", None)
            override_inputs.pop("action_cot_adaptive_final_time_warp", None)
            override_inputs.pop("action_cot_ofp_interval_flow", None)
            override_inputs.pop("action_cot_ofp_warm_start_actions", None)
            override_inputs.pop("action_cot_ofp_warm_start_valid", None)
            override_inputs.pop("action_cot_ofp_warm_start_time", None)
            override_inputs.pop("action_cot_ofp_interval_condition_strength", None)
            override_inputs.pop("action_cot_ofp_interval_condition_mode", None)
            override_inputs.pop("joint_coupled_sampler", None)
            override_inputs.pop("batched_mc_samples", None)
            override_inputs.pop("run_execution_horizon_predictor", None)
            override_inputs.pop("execution_horizon_previous_actions", None)
            override_inputs.pop("execution_horizon_previous_h", None)
            override_inputs.pop("execution_horizon_budget_balance", None)
            override_inputs.pop("execution_horizon_episode_progress", None)
            override_inputs.pop("execution_horizon_previous_valid", None)
            # Avoid data transforms regenerating coarse_actions from expert actions.
            override_inputs.pop("actions", None)
            override_inputs["coarse_actions"] = coarse_actions_override
            override_inputs = self._input_transform(override_inputs)
            if "coarse_actions" not in override_inputs:
                raise KeyError("Input transforms did not preserve coarse_actions_override as coarse_actions.")
            transformed_coarse_actions_override = override_inputs["coarse_actions"]

        inputs = self._input_transform(inputs)
        compact_alpha_router_score = None
        compact_alpha_router_selected_alpha = None
        compact_alpha_router_ms = 0.0
        if compact_alpha_router_enabled:
            configured_final_time_warp_alpha = float(
                np.asarray(self._sample_kwargs.get("final_time_warp_alpha", 0.0)).item()
            )
            if (
                requested_final_time_warp_alpha != 0.0
                or configured_final_time_warp_alpha != 0.0
            ):
                raise ValueError(
                    "Compact alpha routing owns final_time_warp_alpha and cannot use a fixed alpha."
                )
            assert self._acot_compact_alpha_router is not None
            assert absolute_decision_step is not None
            router_started = time.monotonic()
            compact_alpha_router_score, compact_alpha_router_selected_alpha = (
                self._acot_compact_alpha_router.route(
                    np.asarray(inputs["state"]),
                    absolute_decision_step,
                )
            )
            compact_alpha_router_ms = (time.monotonic() - router_started) * 1000
            # Preserve the original float64 CPU threshold and the original
            # static endpoint graph selected by its Python float alpha.
            final_time_warp_alpha = compact_alpha_router_selected_alpha
        if transformed_coarse_actions_override is not None:
            inputs["coarse_actions_override"] = transformed_coarse_actions_override

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        if policy_seed is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            sample_rngs = jax.random.split(sample_rng, batched_mc_samples) if batched_mc_samples else None
        else:
            policy_seed_int = int(np.asarray(policy_seed).item())
            sample_rng = jax.random.key(policy_seed_int)
            sample_rngs = (
                jax.vmap(jax.random.key)(
                    jnp.arange(
                        policy_seed_int,
                        policy_seed_int + batched_mc_samples,
                        dtype=jnp.uint32,
                    )
                )
                if batched_mc_samples
                else None
            )
        outputs = {
            "state": inputs["state"],
            # This key is intentionally not present in norm_stats, so output
            # transforms preserve the exact normalized predictor input.
            "execution_horizon_state_normalized": inputs["state"],
        }
        # joint_coupled_sampler selects a separate jitted entrypoint and must
        # not be forwarded to the legacy sample_actions signature.
        sample_kwargs = {
            key: value for key, value in self._sample_kwargs.items() if key != "joint_coupled_sampler"
        }
        if "coarse_actions_override" in inputs:
            sample_kwargs = {
                **sample_kwargs,
                "explicit_action_reason_override": inputs.pop("coarse_actions_override"),
            }
        if action_cot_skip_segment is not None:
            sample_kwargs = {
                **sample_kwargs,
                "explicit_action_skip_segment": np.asarray(action_cot_skip_segment, dtype=np.int32).reshape(()),
            }
        if action_cot_denoising_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "action_cot_denoising_steps": np.asarray(action_cot_denoising_steps, dtype=np.int32).reshape(()),
            }
        if action_cot_dynamic_denoising_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "dynamic_denoising_steps": bool(np.asarray(action_cot_dynamic_denoising_steps).item()),
            }
        if final_denoising_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "final_denoising_steps": np.asarray(final_denoising_steps, dtype=np.int32).reshape(()),
            }
        if not 0.0 <= final_time_warp_alpha < 1.0:
            raise ValueError("action_cot_final_time_warp_alpha must be in [0, 1).")
        if not 0.0 <= final_endpoint_condition_strength <= 1.0:
            raise ValueError("action_cot_final_endpoint_condition_strength must be in [0, 1].")
        if final_time_warp_alpha > 0.0 and final_endpoint_condition_strength > 0.0:
            raise ValueError(
                "Final time warp and endpoint conditioning are mutually exclusive."
            )
        if compact_alpha_router_enabled:
            sample_kwargs = {
                **sample_kwargs,
                "final_time_warp_alpha": final_time_warp_alpha,
                "force_direct_one_step_expert": True,
                "warm_compact_alpha_static_graphs": True,
            }
        elif final_time_warp_alpha > 0.0:
            sample_kwargs = {
                **sample_kwargs,
                # Keep this hashable so the direct endpoint JIT can specialize
                # and constant-fold the effective time.
                "final_time_warp_alpha": final_time_warp_alpha,
            }
        if final_endpoint_condition_strength > 0.0:
            sample_kwargs = {
                **sample_kwargs,
                "final_endpoint_condition_strength": final_endpoint_condition_strength,
            }
        if final_midpoint_enabled:
            sample_kwargs = {
                **sample_kwargs,
                "final_midpoint": True,
            }
        if self._acot_contextual_compiler is not None:
            sample_kwargs = {
                **sample_kwargs,
                "contextual_fusion_mode": contextual_fusion_mode,
                "contextual_fusion_translation_tau": contextual_fusion_translation_tau,
                "contextual_fusion_rotation_tau": contextual_fusion_rotation_tau,
                "contextual_fusion_gripper_tau": contextual_fusion_gripper_tau,
                "contextual_fusion_gate_width": contextual_fusion_gate_width,
                "contextual_fusion_high_disagreement_source": (
                    contextual_fusion_high_disagreement_source
                ),
                "contextual_fusion_switch_step": contextual_fusion_switch_step,
                "contextual_fusion_absolute_decision_step": absolute_decision_step,
            }
        if adaptive_final_time_warp:
            sample_kwargs = {
                **sample_kwargs,
                "adaptive_final_time_warp": True,
            }
        if harp_residual_enabled:
            sample_kwargs = {
                **sample_kwargs,
                "force_direct_one_step_expert": True,
                "apply_harp_residual": True,
            }
        if harp_gripper_event_enabled:
            sample_kwargs = {
                **sample_kwargs,
                "force_direct_one_step_expert": True,
                "apply_harp_gripper_event": True,
            }
        if final_hybrid_mode != "none":
            sample_kwargs = {
                **sample_kwargs,
                "final_hybrid_mode": final_hybrid_mode,
            }
        if selective_gripper_refinement_enabled:
            sample_kwargs = {
                **sample_kwargs,
                "force_direct_one_step_expert": True,
                "selective_gripper_refinement": True,
                "selective_gripper_tau": selective_gripper_tau,
                "selective_refinement_mode": selective_refinement_mode,
            }
        if ofp_interval_flow:
            if not 0.0 < ofp_warm_start_time <= 1.0:
                raise ValueError("action_cot_ofp_warm_start_time must be in (0, 1].")
            if not 0.0 <= ofp_interval_condition_strength <= 1.0:
                raise ValueError(
                    "action_cot_ofp_interval_condition_strength must be in [0, 1]."
                )
            if ofp_interval_condition_mode not in {"half_concat", "time_blend"}:
                raise ValueError(
                    "action_cot_ofp_interval_condition_mode must be "
                    "'half_concat' or 'time_blend'."
                )
            sample_kwargs = {
                **sample_kwargs,
                "ofp_interval_flow": True,
                "ofp_warm_start_actions": jnp.asarray(
                    self._normalize_previous_actions(ofp_warm_start_actions)
                )[None, ...],
                "ofp_warm_start_valid": jnp.asarray(ofp_warm_start_valid, dtype=jnp.bool_).reshape((1,)),
                "ofp_warm_start_time": jnp.asarray(ofp_warm_start_time, dtype=jnp.float32).reshape((1,)),
                "ofp_interval_condition_strength": jnp.asarray(
                    ofp_interval_condition_strength, dtype=jnp.float32
                ).reshape((1,)),
                "ofp_interval_condition_mode": ofp_interval_condition_mode,
            }
        if final_endpoint_condition_strength > 0.0:
            if self._sample_actions_profile_direct_endpoint_conditioned_one_step_expert is None:
                raise ValueError(
                    "Endpoint conditioning requires the static one-step model entrypoint."
                )
            if (
                final_denoising_steps is not None
                or adaptive_final_time_warp
                or compact_alpha_router_enabled
                or harp_residual_enabled
                or harp_gripper_event_enabled
                or final_hybrid_mode != "none"
                or selective_gripper_refinement_enabled
                or ofp_interval_flow
                or joint_coupled_sampler
                or batched_mc_samples
                or run_execution_horizon_predictor
                or self._acot_contextual_compiler is not None
            ):
                raise ValueError(
                    "Endpoint conditioning is a standalone one-step final mode."
                )
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("Endpoint conditioning requires a fixed one-step EAR.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            if coarse_steps != 1:
                raise ValueError(
                    "Endpoint conditioning requires endpoint-student EAR NFE1; "
                    f"got EAR NFE={coarse_steps}."
                )
        if final_midpoint_enabled:
            if self._sample_actions_profile_midpoint_expert is None:
                raise ValueError("The loaded policy does not implement midpoint inference.")
            if (
                final_denoising_steps is not None
                or final_endpoint_condition_strength > 0.0
                or adaptive_final_time_warp
                or compact_alpha_router_enabled
                or harp_residual_enabled
                or harp_gripper_event_enabled
                or final_hybrid_mode != "none"
                or selective_gripper_refinement_enabled
                or ofp_interval_flow
                or joint_coupled_sampler
                or batched_mc_samples
                or run_execution_horizon_predictor
                or self._acot_contextual_compiler is not None
            ):
                raise ValueError("Midpoint inference is a standalone two-NFE final mode.")
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("Midpoint inference requires a fixed one-step EAR.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            if coarse_steps != 1:
                raise ValueError(
                    "Midpoint inference requires endpoint-student EAR NFE1; "
                    f"got EAR NFE={coarse_steps}."
                )
        observation = _model.Observation.from_dict(inputs)
        detailed_timing = {}
        if compact_alpha_router_enabled:
            detailed_timing["compact_alpha_router_ms"] = compact_alpha_router_ms
            if final_denoising_steps is not None or "final_denoising_steps" in self._sample_kwargs:
                raise ValueError(
                    "Compact alpha routing forces direct final NFE1 and cannot use final_denoising_steps."
                )
            if (
                adaptive_final_time_warp
                or ofp_interval_flow
                or _as_bool(self._sample_kwargs.get("adaptive_final_time_warp", False))
                or _as_bool(self._sample_kwargs.get("ofp_interval_flow", False))
            ):
                raise ValueError(
                    "Compact alpha routing cannot be combined with adaptive time warp or OFP inference."
                )
            if joint_coupled_sampler or batched_mc_samples:
                raise ValueError(
                    "Compact alpha routing cannot be combined with coupled or batched-MC sampling."
                )
            if self._acot_contextual_compiler is not None:
                raise ValueError(
                    "Compact alpha routing cannot be combined with a contextual Action-CoT compiler."
                )
        if final_hybrid_mode != "none":
            if self._sample_actions_profile_direct_one_step_expert is None:
                raise ValueError(
                    "Final hybrid diagnostics require the static direct one-step final expert."
                )
            if final_denoising_steps is not None or "final_denoising_steps" in self._sample_kwargs:
                raise ValueError(
                    "Final hybrid diagnostics own both NFE1 and NFE2 paths; "
                    "do not set final_denoising_steps."
                )
            if (
                compact_alpha_router_enabled
                or harp_residual_enabled
                or harp_gripper_event_enabled
                or selective_gripper_refinement_enabled
                or adaptive_final_time_warp
                or ofp_interval_flow
                or _as_bool(self._sample_kwargs.get("apply_harp_residual", False))
                or _as_bool(self._sample_kwargs.get("apply_harp_gripper_event", False))
                or _as_bool(self._sample_kwargs.get("adaptive_final_time_warp", False))
                or _as_bool(self._sample_kwargs.get("ofp_interval_flow", False))
            ):
                raise ValueError(
                    "Final hybrid diagnostics cannot be combined with HARP, compact routing, "
                    "selective gripper refinement, adaptive time warp, or OFP inference."
                )
            if joint_coupled_sampler or batched_mc_samples or run_execution_horizon_predictor:
                raise ValueError(
                    "Final hybrid diagnostics cannot be combined with coupled sampling, "
                    "batched MC, or execution-horizon prediction."
                )
            if self._acot_contextual_compiler is not None:
                raise ValueError(
                    "Final hybrid diagnostics cannot be combined with a contextual Action-CoT compiler."
                )
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("Final hybrid diagnostics require a fixed one-step EAR.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            if coarse_steps != 1:
                raise ValueError(
                    "Final hybrid diagnostics require endpoint-student EAR NFE1; "
                    f"got EAR NFE={coarse_steps}."
                )
        if selective_gripper_refinement_enabled:
            if (
                self._sample_actions_profile_direct_one_step_expert is None
                or self._sample_actions_profile_second_half_expert is None
            ):
                raise ValueError(
                    "Selective gripper refinement requires direct A1 and second-half final experts."
                )
            if self._action_dim <= 6:
                raise ValueError("Selective gripper refinement requires a gripper action dimension.")
            if final_denoising_steps is not None or "final_denoising_steps" in self._sample_kwargs:
                raise ValueError(
                    "Selective gripper refinement owns direct A1 and conditional NFE2; "
                    "do not set final_denoising_steps."
                )
            if (
                final_hybrid_mode != "none"
                or compact_alpha_router_enabled
                or harp_residual_enabled
                or harp_gripper_event_enabled
                or adaptive_final_time_warp
                or ofp_interval_flow
                or _as_bool(self._sample_kwargs.get("apply_harp_residual", False))
                or _as_bool(self._sample_kwargs.get("apply_harp_gripper_event", False))
                or _as_bool(self._sample_kwargs.get("adaptive_final_time_warp", False))
                or _as_bool(self._sample_kwargs.get("ofp_interval_flow", False))
            ):
                raise ValueError(
                    "Selective gripper refinement cannot be combined with HARP, a gripper-event "
                    "student, hybrid diagnostics, compact routing, adaptive time warp, or OFP."
                )
            if joint_coupled_sampler or batched_mc_samples or run_execution_horizon_predictor:
                raise ValueError(
                    "Selective gripper refinement cannot be combined with coupled sampling, "
                    "batched MC, or execution-horizon prediction."
                )
            if self._acot_contextual_compiler is not None:
                raise ValueError(
                    "Selective gripper refinement cannot be combined with a contextual compiler."
                )
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("Selective gripper refinement requires a fixed one-step EAR.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            if coarse_steps != 1:
                raise ValueError(
                    "Selective gripper refinement requires endpoint-student EAR NFE1; "
                    f"got EAR NFE={coarse_steps}."
                )
        if harp_residual_enabled:
            if self._apply_harp_residual is None:
                raise ValueError(
                    "action_cot_harp_residual=True requires a HARP NPZ at serve startup."
                )
            if (
                compact_alpha_router_enabled
                or harp_gripper_event_enabled
                or adaptive_final_time_warp
                or ofp_interval_flow
                or _as_bool(self._sample_kwargs.get("adaptive_final_time_warp", False))
                or _as_bool(self._sample_kwargs.get("ofp_interval_flow", False))
                or _as_bool(self._sample_kwargs.get("apply_harp_gripper_event", False))
            ):
                raise ValueError(
                    "HARP cannot be combined with ES-HARP, compact routing, adaptive time warp, or OFP inference."
                )
            configured_final_time_warp_alpha = float(
                np.asarray(sample_kwargs.get("final_time_warp_alpha", 0.0)).item()
            )
            expected_final_time_warp_alpha = self._acot_harp_residual.draft_final_time_warp_alpha
            if (
                abs(final_time_warp_alpha - expected_final_time_warp_alpha) > 1e-7
                or abs(configured_final_time_warp_alpha - expected_final_time_warp_alpha) > 1e-7
            ):
                raise ValueError(
                    "HARP request/config final_time_warp_alpha must match its sidecar: "
                    f"request={final_time_warp_alpha}, config={configured_final_time_warp_alpha}, "
                    f"sidecar={expected_final_time_warp_alpha}."
                )
            if joint_coupled_sampler or batched_mc_samples:
                raise ValueError("HARP cannot be combined with coupled or batched-MC sampling.")
            if self._acot_contextual_compiler is not None:
                raise ValueError("HARP cannot be combined with a contextual Action-CoT compiler.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            final_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "final_denoising_steps",
                        sample_kwargs.get("num_steps", 10),
                    )
                ).item()
            )
            if coarse_steps != 1 or final_steps != 1:
                raise ValueError(
                    "HARP was calibrated for endpoint-student EAR NFE1 and direct final NFE1; "
                    f"got EAR NFE={coarse_steps}, final NFE={final_steps}."
                )
        if harp_gripper_event_enabled:
            if self._apply_harp_gripper_event is None:
                raise ValueError(
                    "action_cot_harp_gripper_event=True requires an ES-HARP NPZ at serve startup."
                )
            if self._action_dim <= _es_harp_gripper_event.GRIPPER_INDEX:
                raise ValueError("ES-HARP requires a seventh gripper action dimension.")
            if final_denoising_steps is not None or "final_denoising_steps" in self._sample_kwargs:
                raise ValueError(
                    "ES-HARP forces direct final NFE1; do not set final_denoising_steps."
                )
            if (
                final_hybrid_mode != "none"
                or selective_gripper_refinement_enabled
                or compact_alpha_router_enabled
                or harp_residual_enabled
                or adaptive_final_time_warp
                or ofp_interval_flow
                or _as_bool(self._sample_kwargs.get("apply_harp_residual", False))
                or _as_bool(self._sample_kwargs.get("adaptive_final_time_warp", False))
                or _as_bool(self._sample_kwargs.get("ofp_interval_flow", False))
            ):
                raise ValueError(
                    "ES-HARP cannot be combined with continuous HARP, selective refinement, "
                    "hybrid diagnostics, compact routing, adaptive time warp, or OFP."
                )
            configured_alpha = float(
                np.asarray(sample_kwargs.get("final_time_warp_alpha", 0.0)).item()
            )
            expected_alpha = self._acot_harp_gripper_event.draft_final_time_warp_alpha
            if (
                abs(final_time_warp_alpha - expected_alpha) > 1e-7
                or abs(configured_alpha - expected_alpha) > 1e-7
            ):
                raise ValueError(
                    "ES-HARP request/config final_time_warp_alpha must match its sidecar: "
                    f"request={final_time_warp_alpha}, config={configured_alpha}, "
                    f"sidecar={expected_alpha}."
                )
            if joint_coupled_sampler or batched_mc_samples:
                raise ValueError("ES-HARP cannot be combined with coupled or batched-MC sampling.")
            if self._acot_contextual_compiler is not None:
                raise ValueError("ES-HARP cannot be combined with a contextual compiler.")
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("ES-HARP requires a fixed one-step EAR.")
            coarse_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "action_cot_denoising_steps", sample_kwargs.get("num_steps", 10)
                    )
                ).item()
            )
            final_steps = int(
                np.asarray(
                    sample_kwargs.get(
                        "final_denoising_steps", sample_kwargs.get("num_steps", 10)
                    )
                ).item()
            )
            if coarse_steps != 1 or final_steps != 1:
                raise ValueError(
                    "ES-HARP was calibrated for EAR NFE1 and direct final NFE1; "
                    f"got EAR NFE={coarse_steps}, final NFE={final_steps}."
                )
        if joint_coupled_sampler and batched_mc_samples:
            raise ValueError("joint_coupled_sampler cannot be combined with batched_mc_samples.")
        if self._acot_contextual_compiler is not None and batched_mc_samples:
            raise ValueError("A contextual Action-CoT compiler cannot be combined with batched MC sampling.")
        if self._acot_contextual_compiler is not None and joint_coupled_sampler:
            raise ValueError("A contextual Action-CoT compiler cannot be combined with joint coupled sampling.")
        if ofp_interval_flow and self._acot_contextual_compiler is not None:
            raise ValueError("OFP interval inference cannot be combined with a contextual Action-CoT compiler.")
        if (
            final_time_warp_alpha > 0.0
            and self._acot_contextual_compiler is not None
            and contextual_fusion_mode == "compiler"
        ):
            raise ValueError(
                "Compiler-only inference does not consume final time warp; use a fusion mode."
            )
        if adaptive_final_time_warp and self._acot_contextual_compiler is not None:
            raise ValueError("Adaptive final time warp cannot be combined with a contextual Action-CoT compiler.")
        if final_denoising_steps is not None and self._acot_contextual_compiler is not None:
            raise ValueError("Independent final denoising steps cannot be combined with a contextual compiler.")
        if final_time_warp_alpha > 0.0 and ofp_interval_flow:
            raise ValueError("Final time warp and OFP interval inference are mutually exclusive.")
        if adaptive_final_time_warp and (final_time_warp_alpha > 0.0 or ofp_interval_flow):
            raise ValueError(
                "Adaptive final time warp is mutually exclusive with fixed time warp and OFP inference."
            )
        if adaptive_final_time_warp and (joint_coupled_sampler or batched_mc_samples):
            raise ValueError(
                "Adaptive final time warp cannot be combined with coupled or batched-MC sampling."
            )
        if final_denoising_steps is not None and ofp_interval_flow:
            raise ValueError("Independent final denoising steps cannot be combined with OFP inference.")
        if final_denoising_steps is not None and adaptive_final_time_warp:
            raise ValueError("Independent final denoising steps cannot be combined with adaptive final time warp.")
        if final_denoising_steps is not None and (joint_coupled_sampler or batched_mc_samples):
            raise ValueError(
                "Independent final denoising steps cannot be combined with coupled or batched-MC sampling."
            )
        if adaptive_final_time_warp and self._sample_actions_profile_adaptive_one_step_expert is None:
            raise ValueError("The loaded policy does not contain an adaptive final time-warp gate.")
        if ofp_interval_flow and (joint_coupled_sampler or batched_mc_samples):
            raise ValueError("OFP interval inference cannot be combined with coupled or batched-MC sampling.")
        if ofp_interval_flow and self._sample_actions_profile_ofp_expert is None:
            raise ValueError("The loaded policy does not implement OFP interval inference.")
        if joint_coupled_sampler and export_acot_cache:
            raise ValueError("joint_coupled_sampler does not yet support export_acot_cache.")
        if batched_mc_samples:
            if self._sample_actions_batched_mc is None:
                raise ValueError("The loaded policy does not implement a batched MC teacher.")
            if batched_mc_samples not in (10, 20, 32):
                raise ValueError("batched_mc_samples must be one of 10, 20, or 32.")
            teacher_start = time.monotonic()
            result = self._sample_actions_batched_mc(
                sample_rngs,
                observation,
                num_steps=sample_kwargs.get("num_steps", 10),
                action_cot_denoising_steps=sample_kwargs.get("action_cot_denoising_steps", 10),
            )
            _block_until_ready(result)
            detailed_timing["batched_mc_teacher_ms"] = (time.monotonic() - teacher_start) * 1000
        elif joint_coupled_sampler:
            if self._sample_actions_joint_coupled is None:
                raise ValueError("The loaded policy does not implement joint coupled Action-CoT sampling.")
            if "explicit_action_reason_override" in sample_kwargs:
                raise ValueError("joint_coupled_sampler cannot use coarse_actions_override.")
            if "explicit_action_skip_segment" in sample_kwargs:
                raise ValueError("joint_coupled_sampler cannot use action_cot_skip_segment.")
            if _as_bool(sample_kwargs.get("dynamic_denoising_steps", False)):
                raise ValueError("joint_coupled_sampler does not support dynamic denoising steps.")
            coupled_steps = int(np.asarray(sample_kwargs.get("num_steps", 10)).item())
            coarse_steps = int(
                np.asarray(sample_kwargs.get("action_cot_denoising_steps", coupled_steps)).item()
            )
            if coupled_steps <= 0 or coarse_steps != coupled_steps:
                raise ValueError(
                    "joint_coupled_sampler requires equal positive coarse and final denoising steps."
                )
            stage_start = time.monotonic()
            result = self._sample_actions_joint_coupled(
                sample_rng,
                observation,
                num_steps=coupled_steps,
            )
            if profile_policy_timing:
                _block_until_ready(result)
                detailed_timing["coupled_action_cot_ms"] = (time.monotonic() - stage_start) * 1000
        elif (
            self._acot_contextual_compiler is not None
            or ofp_interval_flow
            or final_denoising_steps is not None
            or final_time_warp_alpha > 0.0
            or final_endpoint_condition_strength > 0.0
            or final_midpoint_enabled
            or adaptive_final_time_warp
            or compact_alpha_router_enabled
            or harp_residual_enabled
            or harp_gripper_event_enabled
            or final_hybrid_mode != "none"
            or selective_gripper_refinement_enabled
            or profile_policy_timing
            or export_acot_cache
        ) and self._can_profile_sample_actions():
            result, detailed_timing = self._profile_sample_actions(
                sample_rng,
                observation,
                sample_kwargs,
                export_acot_cache=export_acot_cache,
            )
        else:
            result = self._sample_actions(sample_rng, observation, **sample_kwargs)

        if isinstance(result, dict):
            if "actions" in result:
                result["execution_horizon_final_actions_normalized"] = result["actions"]
            if "coarse_actions" in result:
                result["execution_horizon_coarse_actions_normalized"] = result["coarse_actions"]

        if run_execution_horizon_predictor:
            if self._predict_execution_horizon is None:
                raise ValueError("run_execution_horizon_predictor=True requires a V2-P predictor sidecar checkpoint.")
            if not isinstance(result, dict):
                raise TypeError("Execution-horizon prediction requires structured action outputs.")
            if "execution_horizon_prefix_feature" not in result:
                raise KeyError("Policy result did not expose the shared prefix feature.")
            normalized_previous_actions = self._normalize_previous_actions(previous_actions)
            predictor_start = time.monotonic()
            predictor_outputs = self._predict_execution_horizon(
                prefix_feature=result["execution_horizon_prefix_feature"],
                proprioception=inputs["state"],
                coarse_actions=result["coarse_actions"],
                final_actions=result["actions"],
                previous_actions=jnp.asarray(normalized_previous_actions)[None, ...],
                previous_h=jnp.asarray(previous_h, dtype=jnp.float32).reshape((1,)),
                budget_balance=jnp.asarray(budget_balance, dtype=jnp.float32).reshape((1,)),
                episode_progress=jnp.asarray(episode_progress, dtype=jnp.float32).reshape((1,)),
                previous_valid=jnp.asarray(previous_valid, dtype=jnp.bool_).reshape((1,)),
            )
            _block_until_ready(predictor_outputs)
            detailed_timing["execution_horizon_predictor_ms"] = (time.monotonic() - predictor_start) * 1000
            result.update({f"execution_horizon_{key}": value for key, value in predictor_outputs.items()})

        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        if compact_alpha_router_enabled:
            assert compact_alpha_router_score is not None
            assert compact_alpha_router_selected_alpha is not None
            assert absolute_decision_step is not None
            # Inject diagnostics only after the model outputs are already on
            # the host. In particular, do not create three tiny JAX arrays and
            # then synchronize them independently during device_get.
            outputs.update(
                {
                    "compact_alpha_router_score": np.asarray(
                        compact_alpha_router_score,
                        dtype=np.float32,
                    ),
                    "compact_alpha_router_selected_alpha": np.asarray(
                        compact_alpha_router_selected_alpha,
                        dtype=np.float32,
                    ),
                    "compact_alpha_router_absolute_decision_step": np.asarray(
                        absolute_decision_step,
                        dtype=np.int32,
                    ),
                }
            )
        mc_actions = None
        if "mc_actions_normalized" in outputs:
            # The regular output pipeline only knows how to transform one
            # action chunk stored under the ``actions`` key.  Apply that same
            # pipeline independently to every MC candidate so collectors can
            # execute them in the environment while retaining the normalized
            # tensor for uncertainty/risk targets.
            candidate_state = np.asarray(outputs["state"])
            transformed_candidates = []
            for candidate in np.asarray(outputs["mc_actions_normalized"]):
                candidate_outputs = self._output_transform(
                    {
                        "state": np.array(candidate_state, copy=True),
                        "actions": np.array(candidate, copy=True),
                    }
                )
                transformed_candidates.append(np.asarray(candidate_outputs["actions"]))
            mc_actions = np.stack(transformed_candidates, axis=0)
        model_time = time.monotonic() - start_time + compact_alpha_router_ms / 1000
        if detailed_timing:
            stage_total_ms = sum(
                detailed_timing.get(key, 0.0)
                for key in (
                    "vlm_ms",
                    "implicit_action_reasoner_ms",
                    "coarse_action_expert_ms",
                    "action_expert_ms",
                    "contextual_compiler_ms",
                    "contextual_fusion_ms",
                    "coupled_action_cot_ms",
                    "batched_mc_teacher_ms",
                    "execution_horizon_predictor_ms",
                    "compact_alpha_router_ms",
                    "harp_residual_ms",
                    "harp_gripper_event_ms",
                )
            )
            detailed_timing["profile_overhead_ms"] = max(0.0, model_time * 1000 - stage_total_ms)

        outputs = self._output_transform(outputs)
        if mc_actions is not None:
            # Environment-space K x T x D action chunks.  This is additive and
            # does not change the existing candidate-0 ``actions`` response.
            outputs["mc_actions"] = mc_actions
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            **detailed_timing,
        }
        return self.post_process(obs, outputs)

    def _normalize_previous_actions(self, previous_actions: Any) -> np.ndarray:
        if previous_actions is None:
            return np.zeros((10, self._action_dim), dtype=np.float32)
        # websocket/msgpack inputs may be backed by a read-only buffer.
        actions = np.array(previous_actions, dtype=np.float32, copy=True)
        if actions.ndim != 2:
            raise ValueError(f"execution_horizon_previous_actions must be rank 2, got {actions.shape}.")
        if self._norm_stats is not None and "actions" in self._norm_stats:
            stats = self._norm_stats["actions"]
            dim = min(actions.shape[-1], np.asarray(stats.mean).shape[-1])
            if self._use_quantile_norm:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError("Quantile normalization requested but action q01/q99 are unavailable.")
                q01 = np.asarray(stats.q01)[..., :dim]
                q99 = np.asarray(stats.q99)[..., :dim]
                actions[..., :dim] = (actions[..., :dim] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean = np.asarray(stats.mean)[..., :dim]
                std = np.asarray(stats.std)[..., :dim]
                actions[..., :dim] = (actions[..., :dim] - mean) / (std + 1e-6)
        actions = actions[:10]
        if actions.shape[0] < 10:
            actions = np.pad(actions, ((0, 10 - actions.shape[0]), (0, 0)))
        if actions.shape[-1] < self._action_dim:
            actions = np.pad(actions, ((0, 0), (0, self._action_dim - actions.shape[-1])))
        return actions[:, : self._action_dim]

    def _can_profile_sample_actions(self) -> bool:
        return (
            self._sample_actions_profile_prefix is not None
            and self._sample_actions_profile_implicit is not None
            and self._sample_actions_profile_coarse is not None
            and self._sample_actions_profile_expert is not None
        )

    def _profile_sample_actions(
        self,
        sample_rng: at.KeyArrayLike,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        *,
        export_acot_cache: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        assert self._sample_actions_profile_prefix is not None
        assert self._sample_actions_profile_implicit is not None
        assert self._sample_actions_profile_coarse is not None
        assert self._sample_actions_profile_expert is not None

        timing: dict[str, float] = {}

        stage_start = time.monotonic()
        prefix_state = self._sample_actions_profile_prefix(sample_rng, observation)
        _block_until_ready(prefix_state)
        timing["vlm_ms"] = (time.monotonic() - stage_start) * 1000

        stage_start = time.monotonic()
        implicit_outputs = self._sample_actions_profile_implicit(prefix_state)
        _block_until_ready(implicit_outputs)
        timing["implicit_action_reasoner_ms"] = (time.monotonic() - stage_start) * 1000

        coarse_kwargs = {
            key: sample_kwargs[key]
            for key in (
                "action_cot_denoising_steps",
                "dynamic_denoising_steps",
                "explicit_action_reason_override",
                "explicit_action_skip_segment",
            )
            if key in sample_kwargs
        }
        coarse_kwargs["num_steps"] = sample_kwargs.get("num_steps", 10)
        stage_start = time.monotonic()
        coarse_outputs = self._sample_actions_profile_coarse(prefix_state, **coarse_kwargs)
        _block_until_ready(coarse_outputs)
        timing["coarse_action_expert_ms"] = (time.monotonic() - stage_start) * 1000

        stage_start = time.monotonic()
        prefix_feature = None
        explicit_final_steps = sample_kwargs.get("final_denoising_steps")
        final_time_warp_alpha = float(sample_kwargs.get("final_time_warp_alpha", 0.0))
        final_endpoint_condition_strength = float(
            sample_kwargs.get("final_endpoint_condition_strength", 0.0)
        )
        final_midpoint = _as_bool(sample_kwargs.get("final_midpoint", False))
        requested_contextual_fusion_mode = str(
            sample_kwargs.get("contextual_fusion_mode", "compiler")
        )
        contextual_fusion_mode = requested_contextual_fusion_mode
        contextual_fusion_translation_tau = float(
            sample_kwargs.get("contextual_fusion_translation_tau", 0.20)
        )
        contextual_fusion_rotation_tau = float(
            sample_kwargs.get("contextual_fusion_rotation_tau", 0.20)
        )
        contextual_fusion_gripper_tau = float(
            sample_kwargs.get("contextual_fusion_gripper_tau", 0.15)
        )
        contextual_fusion_gate_width = float(
            sample_kwargs.get("contextual_fusion_gate_width", 0.05)
        )
        contextual_fusion_high_disagreement_source = str(
            sample_kwargs.get(
                "contextual_fusion_high_disagreement_source", "expert"
            )
        )
        contextual_fusion_phase_selected_expert = None
        if requested_contextual_fusion_mode in {
            "phase_compiler_expert",
            "phase_expert_compiler",
        }:
            phase_step = sample_kwargs.get("contextual_fusion_absolute_decision_step")
            if phase_step is None:
                raise ValueError("Phase contextual fusion requires an absolute decision step.")
            phase_step = int(phase_step)
            phase_switch_step = int(
                sample_kwargs.get("contextual_fusion_switch_step", 400)
            )
            before_switch = phase_step < phase_switch_step
            if requested_contextual_fusion_mode == "phase_compiler_expert":
                contextual_fusion_mode = "compiler" if before_switch else "expert"
            else:
                contextual_fusion_mode = "expert" if before_switch else "compiler"
            contextual_fusion_phase_selected_expert = contextual_fusion_mode == "expert"
        final_hybrid_mode = str(sample_kwargs.get("final_hybrid_mode", "none"))
        selective_gripper_refinement = _as_bool(
            sample_kwargs.get("selective_gripper_refinement", False)
        )
        selective_gripper_tau = float(sample_kwargs.get("selective_gripper_tau", 0.15))
        selective_refinement_mode = str(
            sample_kwargs.get("selective_refinement_mode", "gripper")
        )
        force_direct_one_step = _as_bool(
            sample_kwargs.get("force_direct_one_step_expert", False)
        )
        use_direct_final_expert = (
            explicit_final_steps is None
            and (final_time_warp_alpha > 0.0 or force_direct_one_step)
        ) or (
            explicit_final_steps is not None
            and int(np.asarray(explicit_final_steps).item()) == 1
        )
        if self._acot_contextual_compiler is None:
            if _as_bool(sample_kwargs.get("ofp_interval_flow", False)):
                if self._sample_actions_profile_ofp_expert is None:
                    raise ValueError("The loaded policy does not implement OFP interval inference.")
                expert_outputs = self._sample_actions_profile_ofp_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    sample_kwargs["ofp_warm_start_actions"],
                    sample_kwargs["ofp_warm_start_valid"],
                    sample_kwargs["ofp_warm_start_time"],
                    sample_kwargs["ofp_interval_condition_strength"],
                    sample_kwargs["ofp_interval_condition_mode"],
                )
            elif _as_bool(sample_kwargs.get("adaptive_final_time_warp", False)):
                if self._sample_actions_profile_adaptive_one_step_expert is None:
                    raise ValueError("The loaded policy does not contain an adaptive final time-warp gate.")
                expert_outputs = self._sample_actions_profile_adaptive_one_step_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                )
            elif final_midpoint:
                if self._sample_actions_profile_midpoint_expert is None:
                    raise ValueError("The loaded policy does not implement midpoint inference.")
                expert_outputs = self._sample_actions_profile_midpoint_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    final_time_warp_alpha,
                )
            elif selective_gripper_refinement:
                if (
                    self._sample_actions_profile_direct_one_step_expert is None
                    or self._sample_actions_profile_second_half_expert is None
                ):
                    raise ValueError(
                        "Selective gripper refinement requires direct A1 and second-half experts."
                    )
                direct_outputs = self._sample_actions_profile_direct_one_step_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    final_time_warp_alpha,
                )
                _block_until_ready(direct_outputs)
                action_nfe1 = direct_outputs["actions"]

                verifier_started = time.monotonic()
                # Index 6 is the LIBERO gripper; dimensions 7+ are padding and
                # must not make the uncertainty detector trigger permanently.
                gripper_nfe1 = np.asarray(action_nfe1)[..., 6:7]
                trigger_min_abs = float(np.min(np.abs(gripper_nfe1)))
                trigger_sign_transition = bool(
                    np.any(
                        np.signbit(gripper_nfe1[:, 1:, :])
                        != np.signbit(gripper_nfe1[:, :-1, :])
                    )
                )
                trigger = trigger_min_abs < selective_gripper_tau or trigger_sign_transition
                timing["selective_gripper_verifier_ms"] = (
                    time.monotonic() - verifier_started
                ) * 1000
                timing["selective_gripper_refinement_ms"] = 0.0

                expert_outputs = dict(direct_outputs)
                if trigger:
                    refinement_started = time.monotonic()
                    second_half_outputs = self._sample_actions_profile_second_half_expert(
                        prefix_state,
                        coarse_outputs["explicit_action_reason"],
                        implicit_outputs["implicit_action_reason"],
                        action_nfe1,
                        final_time_warp_alpha,
                    )
                    _block_until_ready(second_half_outputs)
                    timing["selective_gripper_refinement_ms"] = (
                        time.monotonic() - refinement_started
                    ) * 1000
                    if selective_refinement_mode == "full":
                        expert_outputs["actions"] = second_half_outputs["actions"]
                    else:
                        expert_outputs["actions"] = jnp.concatenate(
                            [
                                action_nfe1[..., :6],
                                second_half_outputs["actions"][..., 6:7],
                                action_nfe1[..., 7:],
                            ],
                            axis=-1,
                        )
                batch_size = action_nfe1.shape[0]
                expert_outputs.update(
                    {
                        # Keep host-side branch diagnostics on the host. Four
                        # tiny device arrays would otherwise cause four extra
                        # synchronizations during the generic tree transfer.
                        "selective_gripper_triggered": np.full(
                            (batch_size,), trigger, dtype=np.bool_
                        ),
                        "selective_gripper_trigger_min_abs": np.full(
                            (batch_size,), trigger_min_abs, dtype=np.float32
                        ),
                        "selective_gripper_trigger_sign_transition": np.full(
                            (batch_size,), trigger_sign_transition, dtype=np.bool_
                        ),
                        "selective_gripper_tau": np.full(
                            (batch_size,), selective_gripper_tau, dtype=np.float32
                        ),
                        "selective_refinement_full": np.full(
                            (batch_size,), selective_refinement_mode == "full", dtype=np.bool_
                        ),
                    }
                )
            elif final_endpoint_condition_strength > 0.0:
                if self._sample_actions_profile_direct_endpoint_conditioned_one_step_expert is None:
                    raise ValueError(
                        "The loaded policy does not implement static endpoint-conditioned inference."
                    )
                expert_outputs = (
                    self._sample_actions_profile_direct_endpoint_conditioned_one_step_expert(
                        prefix_state,
                        coarse_outputs["explicit_action_reason"],
                        implicit_outputs["implicit_action_reason"],
                        final_endpoint_condition_strength,
                    )
                )
            elif final_hybrid_mode != "none":
                if self._sample_actions_profile_direct_one_step_expert is None:
                    raise ValueError(
                        "Final hybrid diagnostics require the direct one-step final expert."
                    )
                direct_outputs = self._sample_actions_profile_direct_one_step_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    final_time_warp_alpha,
                )
                nfe2_outputs = self._sample_actions_profile_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    num_steps=2,
                    final_time_warp_alpha=final_time_warp_alpha,
                )
                action_nfe1 = direct_outputs["actions"]
                action_nfe2 = nfe2_outputs["actions"]
                if final_hybrid_mode == "control_nfe2":
                    hybrid_actions = jnp.concatenate(
                        [action_nfe2[..., :6], action_nfe1[..., 6:]],
                        axis=-1,
                    )
                elif final_hybrid_mode == "gripper_nfe2":
                    hybrid_actions = jnp.concatenate(
                        [action_nfe1[..., :6], action_nfe2[..., 6:]],
                        axis=-1,
                    )
                else:
                    raise AssertionError(f"Unexpected final hybrid mode: {final_hybrid_mode}")
                expert_outputs = {
                    "actions": hybrid_actions,
                    "final_hybrid_action_nfe1_normalized": action_nfe1,
                    "final_hybrid_action_nfe2_normalized": action_nfe2,
                }
            elif use_direct_final_expert:
                if self._sample_actions_profile_direct_one_step_expert is None:
                    raise ValueError("The loaded policy does not implement direct one-step final inference.")
                warm_both_static_graphs = _as_bool(
                    sample_kwargs.get("warm_compact_alpha_static_graphs", False)
                )
                if warm_both_static_graphs and not self._acot_compact_alpha_static_graphs_warmed:
                    warmed_outputs = {}
                    for alpha in (0.0, 0.05):
                        candidate = self._sample_actions_profile_direct_one_step_expert(
                            prefix_state,
                            coarse_outputs["explicit_action_reason"],
                            implicit_outputs["implicit_action_reason"],
                            alpha,
                        )
                        _block_until_ready(candidate)
                        warmed_outputs[alpha] = candidate
                    self._acot_compact_alpha_static_graphs_warmed = True
                    expert_outputs = warmed_outputs[final_time_warp_alpha]
                else:
                    expert_outputs = self._sample_actions_profile_direct_one_step_expert(
                        prefix_state,
                        coarse_outputs["explicit_action_reason"],
                        implicit_outputs["implicit_action_reason"],
                        final_time_warp_alpha,
                    )
            else:
                expert_outputs = self._sample_actions_profile_expert(
                    prefix_state,
                    coarse_outputs["explicit_action_reason"],
                    implicit_outputs["implicit_action_reason"],
                    num_steps=(
                        explicit_final_steps
                        if explicit_final_steps is not None
                        else sample_kwargs.get("num_steps", 10)
                    ),
                    final_time_warp_alpha=final_time_warp_alpha,
                )
            _block_until_ready(expert_outputs)
            timing["action_expert_ms"] = (time.monotonic() - stage_start) * 1000
            result = dict(expert_outputs)
            if _as_bool(sample_kwargs.get("apply_harp_residual", False)):
                if self._apply_harp_residual is None:
                    raise ValueError("HARP was requested but no residual sidecar is loaded.")
                explicit_action_reason = coarse_outputs.get("explicit_action_reason")
                implicit_action_reason = implicit_outputs.get("implicit_action_reason")
                if explicit_action_reason is None or implicit_action_reason is None:
                    raise ValueError("HARP requires both EAR and IAR from the current Action-CoT pass.")
                action_nfe1 = expert_outputs["actions"]
                harp_started = time.monotonic()
                harp_outputs = self._apply_harp_residual(
                    action_nfe1,
                    explicit_action_reason,
                    implicit_action_reason,
                    prefix_state["expert_action_noise"],
                    prefix_state["observation"].state,
                )
                _block_until_ready(harp_outputs)
                timing["harp_residual_ms"] = (time.monotonic() - harp_started) * 1000
                result.update(
                    {
                        "actions": harp_outputs["actions"],
                        "harp_action_nfe1_normalized": action_nfe1,
                        "harp_raw_residual": harp_outputs["raw_residual"],
                        "harp_candidate_residual": harp_outputs["candidate_residual"],
                        "harp_applied_residual": harp_outputs["applied_residual"],
                        "harp_margin_mean": harp_outputs["margin_mean"],
                        "harp_margin_log_variance": harp_outputs[
                            "margin_log_variance"
                        ],
                        "harp_margin_std": harp_outputs["margin_std"],
                        "harp_margin_lcb": harp_outputs["margin_lcb"],
                        "harp_margin_gate": harp_outputs["margin_gate"],
                    }
                )
            if _as_bool(sample_kwargs.get("apply_harp_gripper_event", False)):
                if self._apply_harp_gripper_event is None:
                    raise ValueError(
                        "ES-HARP was requested but no gripper-event sidecar is loaded."
                    )
                explicit_action_reason = coarse_outputs.get("explicit_action_reason")
                implicit_action_reason = implicit_outputs.get("implicit_action_reason")
                if explicit_action_reason is None or implicit_action_reason is None:
                    raise ValueError(
                        "ES-HARP requires both EAR and IAR from the current Action-CoT pass."
                    )
                action_nfe1 = expert_outputs["actions"]
                event_started = time.monotonic()
                event_outputs = self._apply_harp_gripper_event(
                    action_nfe1,
                    explicit_action_reason,
                    implicit_action_reason,
                    prefix_state["expert_action_noise"],
                    prefix_state["observation"].state,
                )
                _block_until_ready(event_outputs)
                timing["harp_gripper_event_ms"] = (
                    time.monotonic() - event_started
                ) * 1000
                result.update(
                    {
                        "actions": event_outputs["actions"],
                        "harp_gripper_action_nfe1_normalized": action_nfe1,
                        "harp_gripper_sign_probability": event_outputs[
                            "sign_probability"
                        ],
                        "harp_gripper_flip_probability": event_outputs[
                            "flip_probability"
                        ],
                        "harp_gripper_predicted_positive": event_outputs[
                            "predicted_positive"
                        ],
                        "harp_gripper_flip_consistent": event_outputs[
                            "flip_consistent"
                        ],
                        "harp_gripper_event_gate": event_outputs["event_gate"],
                        "harp_gripper_original": event_outputs["original_gripper"],
                        "harp_gripper_corrected": event_outputs[
                            "corrected_gripper"
                        ],
                    }
                )
        else:
            explicit_action_reason = coarse_outputs.get("explicit_action_reason")
            implicit_action_reason = implicit_outputs.get("implicit_action_reason")
            expert_action_noise = prefix_state.get("expert_action_noise")
            if explicit_action_reason is None:
                raise ValueError("Contextual compiler inference requires an explicit Action-CoT trajectory.")
            if implicit_action_reason is None:
                raise ValueError("Contextual compiler inference requires IAR tokens.")
            if expert_action_noise is None:
                raise ValueError("Contextual compiler inference requires final-action flow noise.")
            if contextual_fusion_mode != "compiler" and (
                self._sample_actions_profile_direct_one_step_expert is None
            ):
                raise ValueError(
                    "Contextual fusion requires the direct one-step final expert."
                )
            prefix_mask = prefix_state["prefix_mask"].astype(prefix_state["prefix_out"].dtype)
            prefix_feature = jnp.asarray(
                jnp.sum(prefix_state["prefix_out"] * prefix_mask[..., None], axis=1)
                / jnp.maximum(jnp.sum(prefix_mask, axis=1, keepdims=True), 1.0),
                dtype=jnp.float32,
            )
            compiled_actions = None
            if contextual_fusion_mode != "expert":
                compiler_started = time.monotonic()
                compiled_actions = self._acot_contextual_compiler.predict_batch(
                    explicit_action_reason,
                    expert_action_noise,
                    prefix_feature,
                    implicit_action_reason,
                    observation.state,
                )
                _block_until_ready(compiled_actions)
                timing["contextual_compiler_ms"] = (
                    time.monotonic() - compiler_started
                ) * 1000
            expert_actions = None
            if contextual_fusion_mode != "compiler":
                expert_started = time.monotonic()
                assert self._sample_actions_profile_direct_one_step_expert is not None
                expert_outputs = self._sample_actions_profile_direct_one_step_expert(
                    prefix_state,
                    explicit_action_reason,
                    implicit_action_reason,
                    final_time_warp_alpha,
                )
                _block_until_ready(expert_outputs)
                timing["action_expert_ms"] = (time.monotonic() - expert_started) * 1000
                expert_actions = expert_outputs["actions"]

            if contextual_fusion_mode == "compiler":
                assert compiled_actions is not None
                actions = compiled_actions
            elif contextual_fusion_mode == "expert":
                assert expert_actions is not None
                actions = expert_actions
            elif contextual_fusion_mode == "control_compiler":
                assert compiled_actions is not None and expert_actions is not None
                actions = jnp.concatenate(
                    (compiled_actions[..., :6], expert_actions[..., 6:]), axis=-1
                )
            elif contextual_fusion_mode == "gripper_compiler":
                assert compiled_actions is not None and expert_actions is not None
                actions = jnp.concatenate(
                    (
                        expert_actions[..., :6],
                        compiled_actions[..., 6:7],
                        expert_actions[..., 7:],
                    ),
                    axis=-1,
                )
            elif contextual_fusion_mode == "blend50":
                assert compiled_actions is not None and expert_actions is not None
                actions = expert_actions.at[..., :7].set(
                    0.5 * (
                        expert_actions[..., :7] + compiled_actions[..., :7]
                    )
                )
            elif contextual_fusion_mode == "semantic_gate":
                assert compiled_actions is not None and expert_actions is not None
                fusion_started = time.monotonic()
                fusion_outputs = _fuse_contextual_semantic_actions(
                    compiled_actions,
                    expert_actions,
                    jnp.asarray(contextual_fusion_translation_tau, dtype=jnp.float32),
                    jnp.asarray(contextual_fusion_rotation_tau, dtype=jnp.float32),
                    jnp.asarray(contextual_fusion_gripper_tau, dtype=jnp.float32),
                    jnp.asarray(contextual_fusion_gate_width, dtype=jnp.float32),
                    jnp.asarray(
                        contextual_fusion_high_disagreement_source == "expert",
                        dtype=jnp.bool_,
                    ),
                )
                _block_until_ready(fusion_outputs)
                timing["contextual_fusion_ms"] = (
                    time.monotonic() - fusion_started
                ) * 1000
                actions = fusion_outputs.pop("actions")
            else:
                raise AssertionError(contextual_fusion_mode)
            result = {"actions": actions}
            if contextual_fusion_mode == "semantic_gate":
                result.update(fusion_outputs)
            if contextual_fusion_phase_selected_expert is not None:
                batch_size = actions.shape[0]
                result.update(
                    {
                        "contextual_fusion_phase_selected_expert": jnp.full(
                            (batch_size,),
                            contextual_fusion_phase_selected_expert,
                            dtype=jnp.bool_,
                        ),
                        "contextual_fusion_absolute_decision_step": jnp.full(
                            (batch_size,), phase_step, dtype=jnp.int32
                        ),
                        "contextual_fusion_switch_step": jnp.full(
                            (batch_size,), phase_switch_step, dtype=jnp.int32
                        ),
                    }
                )
        if coarse_outputs.get("explicit_action_reason") is not None:
            result["coarse_actions"] = coarse_outputs["explicit_action_reason"]
            result["action_cot_denoising_steps"] = coarse_outputs["action_cot_denoising_steps"]
        if "execution_horizon_prefix_feature" in prefix_state:
            result["execution_horizon_prefix_feature"] = prefix_state["execution_horizon_prefix_feature"]
        if export_acot_cache:
            implicit_action_reason = implicit_outputs.get("implicit_action_reason")
            if implicit_action_reason is None:
                raise ValueError("export_acot_cache=True requires the implicit action reasoner.")
            # bfloat16 is not portable through msgpack/websocket encoders.  The
            # training exporter stores this opt-in cache as float16 after it
            # crosses the policy boundary, while ordinary inference remains
            # unchanged and does not return the extra tensor.
            result["acot_iar_tokens"] = jnp.asarray(implicit_action_reason, dtype=jnp.float32)
            # This pooled prefix is derived from the same current-observation
            # VLM pass that deployment already executes.  It is exposed only
            # for opt-in teacher export and contains no future/outcome data.
            if prefix_feature is None:
                prefix_mask = prefix_state["prefix_mask"].astype(prefix_state["prefix_out"].dtype)
                prefix_feature = jnp.asarray(
                    jnp.sum(prefix_state["prefix_out"] * prefix_mask[..., None], axis=1)
                    / jnp.maximum(jnp.sum(prefix_mask, axis=1, keepdims=True), 1.0),
                    dtype=jnp.float32,
                )
            result["acot_prefix_feature"] = prefix_feature
        return result, timing

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        print(
            f"Policy infering for task: {task_name}, with inference time: {outputs['policy_timing']['infer_ms']:.3f} ms"
        )
        if task_name not in task_name_requiring_waist:
            # cut off waist actions for tasks that don't require it
            outputs["actions"] = outputs["actions"][:, :16]

        else:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None, "State is required for post-processing waist actions"
            # freeze four waist actions to the current state, utilizing only the last action for policy output
            outputs["actions"][:, 16:20] = raw_state[16:20]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
