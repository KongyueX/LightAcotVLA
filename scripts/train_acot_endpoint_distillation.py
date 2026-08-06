"""Train one-step EAR/final endpoint students from compact HDF5 labels.

This trainer keeps the VLM, IAR, and reasoning-fusion modules frozen. Only the
selected coarse/final action-expert LLM branch plus its local
input/time/output projections is optimized. It writes a small delta sidecar,
not a full ACoT-VLA checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import json
import logging
import pathlib
import re
import time
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import endpoint_dataset
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import optimizer as optimizer_lib
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders

try:
    import train as train_lib
except ImportError:  # pragma: no cover - supports python -m scripts.train_acot_endpoint_distillation
    from scripts import train as train_lib


LOGGER = logging.getLogger("train_acot_endpoint_distillation")

_COARSE_PATH = re.compile(
    r"^(?:"
    r"PaliGemma/llm/.*_1(?:/.*)?|"
    r"coarse_action_in_proj/.*|"
    r"coarse_time_mlp_in/.*|"
    r"coarse_time_mlp_out/.*|"
    r"coarse_action_time_mlp_in/.*|"
    r"coarse_action_time_mlp_out/.*|"
    r"coarse_action_out_proj/.*"
    r")$"
)
_FINAL_PATH = re.compile(
    r"^(?:"
    r"PaliGemma/llm/.*_2(?:/.*)?|"
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
_FINAL_ADAPTER_PATH = re.compile(
    r"^(?:"
    r"time_mlp_in/.*|"
    r"time_mlp_out/.*|"
    r"action_time_mlp_in/.*|"
    r"action_time_mlp_out/.*|"
    r"action_out_proj/.*"
    r")$"
)
_ADAPTIVE_FINAL_TIME_WARP_PATH = re.compile(
    r"^adaptive_final_time_warp_gate/.*$"
)
_PACT_FLOW_SCHEDULER_PATH = re.compile(r"^pact_flow_scheduler/.*$")


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    resume_sidecar_params: str | None = None
    causal_audit_summary: str | None = None
    stage: str = "final"
    variant: str = "ir"
    seed: int = 7
    train_steps: int = 1_000
    batch_size: int = 8
    learning_rate: float = 3e-6
    decay_learning_rate: float = 3e-7
    warmup_steps: int = 50
    weight_decay: float = 1e-10
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    log_interval: int = 25
    checkpoint_interval: int = 250
    coarse_loss_weight: float = 1.0
    final_loss_weight: float = 1.0
    ir_loss_weight: float = 0.5
    multi_time_flow_loss_weight: float = 0.0
    multi_time_response_loss_weight: float = 0.0
    multi_time_timestep: float = 0.5
    joint_coupled_training: bool = False
    use_student_coarse: bool = False
    # Opt-in final-action One-Step Flow Policy self-consistency.  This is
    # OFP-SC, not full OFP: the current ACoT input path has no safe null
    # condition / condition-dropout contract for OFP self-guidance.
    ofp_sc: bool = False
    ofp_flow_anchor_loss_weight: float = 1.0
    ofp_self_consistency_loss_weight: float = 1.0
    ofp_endpoint_anchor_loss_weight: float = 0.0
    ofp_ema_decay: float = 0.999
    ofp_min_interval: float = 0.05
    ofp_contraction_power: float = 1.0
    ofp_interval_condition_strength: float = 1.0
    ofp_interval_condition_mode: str = "half_concat"
    ofp_adapter_only: bool = False
    adaptive_final_time_warp: bool = False
    # Plan-anchored cross-level action-chunk time scheduling. This path jointly
    # updates the final action expert and scheduler; the frozen VLM/EAR/IAR
    # contract is unchanged. The remaining probability mass is assigned to the
    # deployment endpoint objective.
    pact_flow_scheduler: bool = False
    pact_heterogeneous_flow_probability: float = 0.25
    pact_scalar_flow_probability: float = 0.25
    pact_plan_anchor_loss_weight: float = 1.0
    pact_uncertainty_loss_weight: float = 0.01
    pact_schedule_smoothness_loss_weight: float = 0.01
    fsdp_devices: int = 1
    overwrite: bool = False
    allow_failed_audit: bool = False


def _path_text(path: tuple[Any, ...]) -> str:
    return "/".join(map(str, path))


def _matches_stage(
    path: tuple[Any, ...],
    stage: str,
    *,
    ofp_adapter_only: bool = False,
    adaptive_final_time_warp: bool = False,
) -> bool:
    text = _path_text(path)
    if adaptive_final_time_warp:
        return stage == "final" and _ADAPTIVE_FINAL_TIME_WARP_PATH.fullmatch(text) is not None
    if ofp_adapter_only:
        return stage == "final" and _FINAL_ADAPTER_PATH.fullmatch(text) is not None
    return (
        (stage in {"coarse", "dual"} and _COARSE_PATH.fullmatch(text) is not None)
        or (stage in {"final", "dual"} and _FINAL_PATH.fullmatch(text) is not None)
    )


def _train_filter(
    stage: str,
    *,
    ofp_adapter_only: bool = False,
    adaptive_final_time_warp: bool = False,
) -> nnx.filterlib.Filter:
    if adaptive_final_time_warp:
        if stage != "final":
            raise ValueError("Adaptive final time warp training requires stage='final'.")
        return nnx_utils.PathRegex(r"adaptive_final_time_warp_gate/.*")
    if ofp_adapter_only:
        if stage != "final":
            raise ValueError("OFP adapter-only training requires stage='final'.")
        # Deliberately exclude PaliGemma/llm/*_2 and action_in_proj.  This
        # scope only calibrates the existing final time-conditioning stack and
        # endpoint projection; it introduces no random checkpoint modules.
        return nnx.Any(
            nnx_utils.PathRegex(r"time_mlp_in/.*"),
            nnx_utils.PathRegex(r"time_mlp_out/.*"),
            nnx_utils.PathRegex(r"action_time_mlp_in/.*"),
            nnx_utils.PathRegex(r"action_time_mlp_out/.*"),
            nnx_utils.PathRegex(r"action_out_proj/.*"),
        )
    filters: list[nnx.filterlib.Filter] = []
    if stage in {"coarse", "dual"}:
        filters.extend(
            [
                nnx_utils.PathRegex(r"PaliGemma/llm/.*_1(?:/.*)?"),
                nnx_utils.PathRegex(r"coarse_action_in_proj/.*"),
                nnx_utils.PathRegex(r"coarse_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"coarse_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"coarse_action_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"coarse_action_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"coarse_action_out_proj/.*"),
            ]
        )
    if stage in {"final", "dual"}:
        filters.extend(
            [
                nnx_utils.PathRegex(r"PaliGemma/llm/.*_2(?:/.*)?"),
                nnx_utils.PathRegex(r"action_in_proj/.*"),
                nnx_utils.PathRegex(r"time_mlp_in/.*"),
                nnx_utils.PathRegex(r"time_mlp_out/.*"),
                nnx_utils.PathRegex(r"action_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"action_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"action_out_proj/.*"),
                nnx_utils.PathRegex(r"pact_flow_scheduler/.*"),
            ]
        )
    if not filters:
        raise ValueError(f"stage must be coarse, final, or dual; got {stage!r}.")
    return nnx.Any(*filters)


def _validate_args(args: Args) -> None:
    if args.stage not in {"coarse", "final", "dual"}:
        raise ValueError("--stage must be coarse, final, or dual.")
    if args.variant not in {"b6", "ir"}:
        raise ValueError("--variant must be b6 or ir.")
    if args.variant == "ir" and args.stage == "coarse":
        raise ValueError("IR alignment acts on final actions; use --variant b6 for coarse-only training.")
    for name in (
        "train_steps",
        "batch_size",
        "warmup_steps",
        "log_interval",
        "checkpoint_interval",
        "fsdp_devices",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.learning_rate <= 0 or args.decay_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and peak learning rate must be positive.")
    if min(
        args.coarse_loss_weight,
        args.final_loss_weight,
        args.ir_loss_weight,
        args.multi_time_flow_loss_weight,
        args.multi_time_response_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.variant == "ir" and args.ir_loss_weight == 0:
        raise ValueError("IR variant requires a positive --ir-loss-weight.")
    if not 0 < args.multi_time_timestep < 1:
        raise ValueError("--multi-time-timestep must be in (0, 1).")
    if args.multi_time_response_loss_weight > 0 and args.multi_time_flow_loss_weight <= 0:
        raise ValueError(
            "--multi-time-response-loss-weight requires a positive --multi-time-flow-loss-weight."
        )
    if args.multi_time_response_loss_weight > 0 and args.stage == "coarse":
        raise ValueError("Multi-time response alignment acts on final actions; use final or dual stage.")
    if args.joint_coupled_training and args.stage == "coarse":
        raise ValueError("--joint-coupled-training requires final or dual stage.")
    if args.joint_coupled_training and args.multi_time_response_loss_weight <= 0:
        raise ValueError(
            "--joint-coupled-training requires positive --multi-time-response-loss-weight "
            "because coupled endpoint paths share the same coarse noise at t=1."
        )
    if args.ofp_sc:
        if args.stage != "final":
            raise ValueError("--ofp-sc is a final-only training path; use --stage final.")
        if args.variant != "b6":
            raise ValueError(
                "--ofp-sc does not include endpoint IR alignment; use --variant b6."
            )
        if args.joint_coupled_training or args.use_student_coarse:
            raise ValueError(
                "--ofp-sc currently conditions on exported teacher EAR; do not combine it with "
                "--joint-coupled-training or --use-student-coarse."
            )
        if args.multi_time_flow_loss_weight > 0 or args.multi_time_response_loss_weight > 0:
            raise ValueError(
                "--ofp-sc replaces the fixed multi-time objective with all-time flow anchoring "
                "and nested-interval consistency; leave multi-time weights at zero."
            )
        if args.ofp_flow_anchor_loss_weight <= 0:
            raise ValueError("--ofp-flow-anchor-loss-weight must be positive.")
        if args.ofp_self_consistency_loss_weight <= 0:
            raise ValueError("--ofp-self-consistency-loss-weight must be positive.")
        if args.ofp_endpoint_anchor_loss_weight < 0:
            raise ValueError("--ofp-endpoint-anchor-loss-weight must be non-negative.")
        if not 0.0 < args.ofp_ema_decay < 1.0:
            raise ValueError("--ofp-ema-decay must be in (0, 1).")
        if not 0.0 < args.ofp_min_interval < 1.0:
            raise ValueError("--ofp-min-interval must be in (0, 1).")
        if args.ofp_contraction_power <= 0:
            raise ValueError("--ofp-contraction-power must be positive.")
        if not 0.0 <= args.ofp_interval_condition_strength <= 1.0:
            raise ValueError("--ofp-interval-condition-strength must be in [0, 1].")
        if args.ofp_interval_condition_mode not in {"half_concat", "time_blend"}:
            raise ValueError(
                "--ofp-interval-condition-mode must be half_concat or time_blend."
            )
    elif args.ofp_endpoint_anchor_loss_weight != 0:
        raise ValueError("--ofp-endpoint-anchor-loss-weight requires --ofp-sc.")
    if not args.ofp_sc and args.ofp_interval_condition_strength != 1.0:
        raise ValueError("--ofp-interval-condition-strength requires --ofp-sc.")
    if not args.ofp_sc and args.ofp_interval_condition_mode != "half_concat":
        raise ValueError("--ofp-interval-condition-mode requires --ofp-sc.")
    if args.ofp_adapter_only and (not args.ofp_sc or args.stage != "final"):
        raise ValueError("--ofp-adapter-only requires --ofp-sc with --stage final.")
    if args.adaptive_final_time_warp:
        if args.stage != "final" or args.variant != "ir":
            raise ValueError(
                "--adaptive-final-time-warp requires --stage final --variant ir."
            )
        if args.resume_sidecar_params is None:
            raise ValueError(
                "--adaptive-final-time-warp requires --resume-sidecar-params for the final_ir anchor."
            )
        if args.ofp_sc or args.ofp_adapter_only:
            raise ValueError(
                "--adaptive-final-time-warp cannot be combined with OFP training."
            )
        if args.joint_coupled_training or args.use_student_coarse:
            raise ValueError(
                "--adaptive-final-time-warp uses exported teacher EAR and cannot use coupled/student coarse training."
            )
        if args.multi_time_flow_loss_weight > 0 or args.multi_time_response_loss_weight > 0:
            raise ValueError(
                "--adaptive-final-time-warp supports endpoint and IR losses only; leave multi-time weights at zero."
            )
    if args.pact_flow_scheduler:
        if args.stage != "final" or args.variant != "b6":
            raise ValueError(
                "--pact-flow-scheduler requires --stage final --variant b6."
            )
        if args.resume_sidecar_params is None:
            raise ValueError(
                "--pact-flow-scheduler requires --resume-sidecar-params for the endpoint-student anchor."
            )
        if args.adaptive_final_time_warp:
            raise ValueError(
                "--pact-flow-scheduler and --adaptive-final-time-warp are mutually exclusive."
            )
        if args.ofp_sc or args.ofp_adapter_only:
            raise ValueError("--pact-flow-scheduler cannot be combined with OFP training.")
        if args.joint_coupled_training or args.use_student_coarse:
            raise ValueError(
                "--pact-flow-scheduler uses exported teacher EAR and cannot use coupled/student coarse training."
            )
        if args.multi_time_flow_loss_weight > 0 or args.multi_time_response_loss_weight > 0:
            raise ValueError(
                "--pact-flow-scheduler owns heterogeneous/scalar flow sampling; "
                "leave legacy multi-time weights at zero."
            )
        probabilities = (
            args.pact_heterogeneous_flow_probability,
            args.pact_scalar_flow_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("PACT flow probabilities must each be in [0, 1].")
        if sum(probabilities) > 1.0:
            raise ValueError(
                "PACT heterogeneous and scalar flow probabilities must sum to at most 1."
            )
        if args.pact_schedule_smoothness_loss_weight < 0:
            raise ValueError("--pact-schedule-smoothness-loss-weight must be non-negative.")
        if args.pact_plan_anchor_loss_weight < 0:
            raise ValueError("--pact-plan-anchor-loss-weight must be non-negative.")
        if args.pact_uncertainty_loss_weight < 0:
            raise ValueError("--pact-uncertainty-loss-weight must be non-negative.")
    elif (
        args.pact_heterogeneous_flow_probability != 0.25
        or args.pact_scalar_flow_probability != 0.25
        or args.pact_plan_anchor_loss_weight != 1.0
        or args.pact_uncertainty_loss_weight != 0.01
        or args.pact_schedule_smoothness_loss_weight != 0.01
    ):
        raise ValueError(
            "PACT-specific probability/smoothness overrides require --pact-flow-scheduler."
        )


def _check_audit_gate(inputs: Sequence[str], args: Args) -> None:
    needs_response_audit = args.variant == "ir" or args.multi_time_response_loss_weight > 0
    if not needs_response_audit or args.allow_failed_audit:
        return
    summaries = []
    if args.causal_audit_summary is not None:
        candidate = pathlib.Path(args.causal_audit_summary)
        if not candidate.exists():
            raise FileNotFoundError(f"Causal audit summary not found: {candidate}")
        summaries.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
    else:
        for item in inputs:
            path = pathlib.Path(item)
            candidate = path / "audit_summary.json" if path.is_dir() else path.parent / "audit_summary.json"
            if candidate.exists():
                summaries.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
    failed = [str(path) for path, summary in summaries if not summary.get("ear_causal_audit_pass", False)]
    if failed:
        raise ValueError(
            "EAR causal audit failed for the requested IR dataset. "
            f"Refusing IR training for {failed}; use --allow-failed-audit only for a diagnostic run."
        )
    if not summaries:
        raise ValueError(
            "Interventional-response training requires a causal audit summary with "
            "ear_causal_audit_pass=true. "
            "Pass --causal-audit-summary explicitly, or use --allow-failed-audit only for a diagnostic run."
        )


def _split_indices(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
    require_semantic_intervention: bool,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.ones((len(arrays["dataset_index"]),), dtype=np.bool_)
    if require_semantic_intervention:
        semantic = arrays["intervention_valid"].copy()
        semantic &= arrays["intervention_ids"] != endpoint_dataset.INTERVENTION_IDS["null"]
        eligible &= np.any(semantic, axis=-1)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 2:
        raise ValueError("Endpoint dataset has fewer than two eligible records.")

    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups[eligible_indices])
    rng = np.random.default_rng(seed)
    if unique_groups.size >= 2:
        rng.shuffle(unique_groups)
        validation_count = max(1, round(unique_groups.size * validation_fraction))
        validation_groups = unique_groups[:validation_count]
        validation_mask = eligible & np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(eligible & ~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        shuffled = eligible_indices.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(shuffled.size * validation_fraction))
        validation_count = min(validation_count, shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
        LOGGER.warning(
            "Episode metadata did not identify multiple groups; using a record-level train/validation split."
        )
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Train/validation split produced an empty partition.")
    return train_indices, validation_indices


def _load_norm_stats(
    train_config: config_lib.TrainConfig,
    data_config: config_lib.DataConfig,
    checkpoint_dir: pathlib.Path,
) -> dict[str, Any]:
    if data_config.norm_stats is not None:
        return data_config.norm_stats
    if data_config.asset_id is None:
        raise ValueError("The data config needs asset_id to load checkpoint normalization stats.")
    return checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)


def _with_norm_stats(
    data_config: config_lib.DataConfig,
    norm_stats: dict[str, Any],
) -> config_lib.DataConfig:
    """Replace norm stats without dropping factory-attached runtime fields."""

    updated = dataclasses.replace(data_config, norm_stats=norm_stats)
    declared_fields = {field.name for field in dataclasses.fields(data_config)}
    for name, value in vars(data_config).items():
        if name not in declared_fields:
            object.__setattr__(updated, name, value)
    return updated


def _load_resume_params(path: str | None) -> tuple[dict[str, Any] | None, set[tuple[Any, ...]]]:
    if path is None:
        return None, set()
    resolved = download.maybe_download(path)
    params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(resolved, restore_type=np.ndarray)
    )
    flat = traverse_util.flatten_dict(params)
    disallowed = [
        _path_text(key)
        for key in flat
        if _COARSE_PATH.fullmatch(_path_text(key)) is None
        and _FINAL_PATH.fullmatch(_path_text(key)) is None
    ]
    if disallowed:
        raise ValueError(f"Resume sidecar contains disallowed parameters: {disallowed[:5]}")
    return params, set(flat)


@dataclasses.dataclass(frozen=True)
class _BaseAndSidecarLoader:
    base_params_path: str
    sidecar_params: dict[str, Any] | None
    adaptive_final_time_warp: bool = False
    pact_flow_scheduler: bool = False

    def load(self, params: Any) -> Any:
        if self.adaptive_final_time_warp or self.pact_flow_scheduler:
            flat_reference = traverse_util.flatten_dict(params)
            new_reference = {
                path: value
                for path, value in flat_reference.items()
                if (
                    self.adaptive_final_time_warp
                    and _ADAPTIVE_FINAL_TIME_WARP_PATH.fullmatch(_path_text(path)) is not None
                )
                or (
                    self.pact_flow_scheduler
                    and _PACT_FLOW_SCHEDULER_PATH.fullmatch(_path_text(path)) is not None
                )
            }
            if not new_reference:
                raise ValueError("Requested endpoint extension is missing from the model config.")
            base_reference = traverse_util.unflatten_dict(
                {
                    path: value
                    for path, value in flat_reference.items()
                    if path not in new_reference
                }
            )
            loaded = weight_loaders.CheckpointWeightLoader(self.base_params_path).load(
                base_reference
            )
            flat_loaded = traverse_util.flatten_dict(loaded)
            # ``init_train_state`` supplies ShapeDtypeStruct leaves here. The
            # adaptive gate is contractually zero initialized. PACT may contain
            # non-zero hidden-layer initialization, so retain its abstract
            # leaves: ``_load_weights_and_validate`` removes them and the real
            # model initialization survives when partial params are merged.
            # Never broaden the global checkpoint loader's missing policy.
            for path, expected in new_reference.items():
                if _ADAPTIVE_FINAL_TIME_WARP_PATH.fullmatch(_path_text(path)) is not None:
                    flat_loaded[path] = jax.numpy.zeros(expected.shape, dtype=expected.dtype)
                else:
                    flat_loaded[path] = expected
            loaded = traverse_util.unflatten_dict(flat_loaded)
        else:
            loaded = weight_loaders.CheckpointWeightLoader(self.base_params_path).load(params)
        if self.sidecar_params is not None:
            loaded = policy_config.merge_acot_endpoint_student_params(loaded, self.sidecar_params)
        return loaded


def _choose_interventions(
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    coarse = []
    actions = []
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    for row_index in row_indices:
        ids = arrays["intervention_ids"][row_index]
        valid = arrays["intervention_valid"][row_index] & (ids != null_id)
        candidates = np.flatnonzero(valid)
        if not candidates.size:
            candidates = np.flatnonzero(arrays["intervention_valid"][row_index])
        if not candidates.size:
            # Coarse-only clean-label exports intentionally contain no
            # interventions. The coarse stage never consumes these tensors,
            # so clean targets are a shape-correct neutral fallback. Final/dual
            # stages are filtered to records with a valid semantic intervention.
            coarse.append(arrays["clean_coarse"][row_index])
            actions.append(arrays["clean_actions"][row_index])
            continue
        selected = int(candidates[0] if deterministic else rng.choice(candidates))
        coarse.append(arrays["intervention_coarse"][row_index, selected])
        actions.append(arrays["intervention_actions"][row_index, selected])
    return np.asarray(coarse, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _make_batch(
    observation_dataset: data_loader.Dataset,
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic_intervention: bool,
) -> dict[str, Any]:
    items = [observation_dataset[int(arrays["dataset_index"][index])] for index in row_indices]
    if any(item is None for item in items):
        raise ValueError("Observation dataset returned None for an endpoint record.")
    collated = data_loader._collate_fn(items)  # noqa: SLF001 - reuse the canonical OpenPI collation.
    collated = jax.tree.map(jax.numpy.asarray, collated)
    observation = model_lib.Observation.from_dict(collated)
    intervention_coarse, intervention_actions = _choose_interventions(
        arrays,
        row_indices,
        rng,
        deterministic=deterministic_intervention,
    )
    return {
        "observation": observation,
        "teacher_coarse": np.asarray(arrays["clean_coarse"][row_indices], dtype=np.float32),
        "teacher_actions": np.asarray(arrays["clean_actions"][row_indices], dtype=np.float32),
        "coarse_noise": np.asarray(arrays["coarse_noise"][row_indices], dtype=np.float32),
        "action_noise": np.asarray(arrays["action_noise"][row_indices], dtype=np.float32),
        "intervention_coarse": intervention_coarse,
        "teacher_intervention_actions": intervention_actions,
    }


def _endpoint_train_step(
    state: training_utils.TrainState,
    batch: dict[str, Any],
    *,
    trainable_filter: nnx.filterlib.Filter,
    stage: str,
    use_student_coarse: bool,
    coarse_loss_weight: float,
    final_loss_weight: float,
    ir_loss_weight: float,
    multi_time_flow_loss_weight: float,
    multi_time_response_loss_weight: float,
    multi_time_timestep: float,
    joint_coupled_training: bool,
    adaptive_final_time_warp: bool,
    pact_flow_scheduler: bool,
    pact_seed: int,
    pact_heterogeneous_flow_probability: float,
    pact_scalar_flow_probability: float,
    pact_plan_anchor_loss_weight: float,
    pact_uncertainty_loss_weight: float,
    pact_schedule_smoothness_loss_weight: float,
    ofp_sc: bool,
    ofp_seed: int,
    ofp_train_steps: int,
    ofp_flow_anchor_loss_weight: float,
    ofp_self_consistency_loss_weight: float,
    ofp_endpoint_anchor_loss_weight: float,
    ofp_min_interval: float,
    ofp_contraction_power: float,
    ofp_interval_condition_strength: float,
    ofp_interval_condition_mode: str,
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    ema_teacher = None
    if ofp_sc:
        if state.ema_params is None or state.ema_decay is None:
            raise ValueError("OFP-SC requires EMA parameters in TrainState.")
        ema_teacher = nnx.merge(state.model_def, state.ema_params)
        ema_teacher.eval()

    def loss_fn(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        if pact_flow_scheduler:
            pact_rng = jax.random.fold_in(jax.random.key(pact_seed), state.step)
            return candidate.compute_pact_flow_distillation_loss(
                batch["observation"],
                batch["teacher_coarse"],
                batch["teacher_actions"],
                batch["action_noise"],
                pact_rng,
                heterogeneous_flow_probability=pact_heterogeneous_flow_probability,
                scalar_flow_probability=pact_scalar_flow_probability,
                plan_anchor_loss_weight=pact_plan_anchor_loss_weight,
                uncertainty_loss_weight=pact_uncertainty_loss_weight,
                schedule_smoothness_loss_weight=pact_schedule_smoothness_loss_weight,
                compute_diagnostics=False,
            )
        if ofp_sc:
            assert ema_teacher is not None
            ofp_rng = jax.random.fold_in(jax.random.key(ofp_seed), state.step)
            training_progress = jax.numpy.asarray(state.step, dtype=jax.numpy.float32) / max(
                ofp_train_steps - 1, 1
            )
            return candidate.compute_ofp_sc_loss(
                ema_teacher,
                batch["observation"],
                batch["teacher_coarse"],
                batch["teacher_actions"],
                batch["action_noise"],
                ofp_rng,
                training_progress=training_progress,
                flow_anchor_loss_weight=ofp_flow_anchor_loss_weight,
                self_consistency_loss_weight=ofp_self_consistency_loss_weight,
                endpoint_anchor_loss_weight=ofp_endpoint_anchor_loss_weight,
                min_interval=ofp_min_interval,
                contraction_power=ofp_contraction_power,
                interval_condition_strength=ofp_interval_condition_strength,
                interval_condition_mode=ofp_interval_condition_mode,
                compute_endpoint_metrics=False,
            )
        return candidate.compute_endpoint_distillation_loss(
            batch["observation"],
            batch["teacher_coarse"],
            batch["teacher_actions"],
            batch["coarse_noise"],
            batch["action_noise"],
            batch["intervention_coarse"],
            batch["teacher_intervention_actions"],
            stage=stage,
            use_student_coarse=use_student_coarse,
            coarse_loss_weight=coarse_loss_weight,
            final_loss_weight=final_loss_weight,
            ir_loss_weight=ir_loss_weight,
            multi_time_flow_loss_weight=multi_time_flow_loss_weight,
            multi_time_response_loss_weight=multi_time_response_loss_weight,
            multi_time_timestep=multi_time_timestep,
            joint_coupled_training=joint_coupled_training,
            adaptive_final_time_warp=adaptive_final_time_warp,
            compute_ir_metrics=False,
            compute_multi_time_metrics=False,
        )

    diff_state = nnx.DiffState(0, trainable_filter)
    (_, metrics), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=diff_state,
        has_aux=True,
    )(model)
    params = state.params.filter(trainable_filter)
    updates, optimizer_state = state.tx.update(gradients, state.opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    nnx.update(model, updated_params)
    next_params = nnx.state(model)
    next_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=next_params,
        opt_state=optimizer_state,
    )
    if ofp_sc:
        assert state.ema_params is not None
        assert state.ema_decay is not None
        next_state = dataclasses.replace(
            next_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1.0 - state.ema_decay) * new,
                state.ema_params,
                next_params,
            ),
        )
    return next_state, {**metrics, "gradient_norm": optax.global_norm(gradients)}


def _endpoint_validation_step(
    state: training_utils.TrainState,
    batch: dict[str, Any],
    *,
    stage: str,
    use_student_coarse: bool,
    coarse_loss_weight: float,
    final_loss_weight: float,
    ir_loss_weight: float,
    multi_time_flow_loss_weight: float,
    multi_time_response_loss_weight: float,
    multi_time_timestep: float,
    joint_coupled_training: bool,
    adaptive_final_time_warp: bool,
    pact_flow_scheduler: bool,
    pact_seed: int,
    pact_heterogeneous_flow_probability: float,
    pact_scalar_flow_probability: float,
    pact_plan_anchor_loss_weight: float,
    pact_uncertainty_loss_weight: float,
    pact_schedule_smoothness_loss_weight: float,
    ofp_sc: bool,
    ofp_seed: int,
    ofp_train_steps: int,
    ofp_flow_anchor_loss_weight: float,
    ofp_self_consistency_loss_weight: float,
    ofp_endpoint_anchor_loss_weight: float,
    ofp_min_interval: float,
    ofp_contraction_power: float,
    ofp_interval_condition_strength: float,
    ofp_interval_condition_mode: str,
) -> dict[str, jax.Array]:
    if pact_flow_scheduler:
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        pact_rng = jax.random.fold_in(jax.random.key(pact_seed + 1), state.step)
        _, metrics = model.compute_pact_flow_distillation_loss(
            batch["observation"],
            batch["teacher_coarse"],
            batch["teacher_actions"],
            batch["action_noise"],
            pact_rng,
            heterogeneous_flow_probability=pact_heterogeneous_flow_probability,
            scalar_flow_probability=pact_scalar_flow_probability,
            plan_anchor_loss_weight=pact_plan_anchor_loss_weight,
            uncertainty_loss_weight=pact_uncertainty_loss_weight,
            schedule_smoothness_loss_weight=pact_schedule_smoothness_loss_weight,
            compute_diagnostics=True,
        )
        return metrics
    if ofp_sc:
        if state.ema_params is None:
            raise ValueError("OFP-SC validation requires EMA parameters in TrainState.")
        # OFP-SC deploys and saves EMA weights, so validation must score that
        # exact parameter source rather than the transient online student.
        model = nnx.merge(state.model_def, state.ema_params)
        model.eval()
        ema_teacher = nnx.merge(state.model_def, state.ema_params)
        ema_teacher.eval()
        ofp_rng = jax.random.fold_in(jax.random.key(ofp_seed + 1), state.step)
        training_progress = jax.numpy.asarray(state.step, dtype=jax.numpy.float32) / max(
            ofp_train_steps - 1, 1
        )
        _, metrics = model.compute_ofp_sc_loss(
            ema_teacher,
            batch["observation"],
            batch["teacher_coarse"],
            batch["teacher_actions"],
            batch["action_noise"],
            ofp_rng,
            training_progress=training_progress,
            flow_anchor_loss_weight=ofp_flow_anchor_loss_weight,
            self_consistency_loss_weight=ofp_self_consistency_loss_weight,
            endpoint_anchor_loss_weight=ofp_endpoint_anchor_loss_weight,
            min_interval=ofp_min_interval,
            contraction_power=ofp_contraction_power,
            interval_condition_strength=ofp_interval_condition_strength,
            interval_condition_mode=ofp_interval_condition_mode,
            compute_endpoint_metrics=True,
        )
        return metrics
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    _, metrics = model.compute_endpoint_distillation_loss(
        batch["observation"],
        batch["teacher_coarse"],
        batch["teacher_actions"],
        batch["coarse_noise"],
        batch["action_noise"],
        batch["intervention_coarse"],
        batch["teacher_intervention_actions"],
        stage=stage,
        use_student_coarse=use_student_coarse,
        coarse_loss_weight=coarse_loss_weight,
        final_loss_weight=final_loss_weight,
        ir_loss_weight=ir_loss_weight,
        multi_time_flow_loss_weight=multi_time_flow_loss_weight,
        multi_time_response_loss_weight=multi_time_response_loss_weight,
        multi_time_timestep=multi_time_timestep,
        joint_coupled_training=joint_coupled_training,
        adaptive_final_time_warp=adaptive_final_time_warp,
        compute_ir_metrics=stage in {"final", "dual"},
        compute_multi_time_metrics=(
            multi_time_flow_loss_weight > 0 or multi_time_response_loss_weight > 0
        ),
    )
    return metrics


def _full_validation_index_batches(
    indices: np.ndarray,
    *,
    batch_size: int,
    device_count: int,
) -> list[tuple[np.ndarray, int]]:
    """Build sharding-compatible batches with equal-repeat padding.

    The returned count is the number of real validation records represented by
    the batch.  Any padding repeats every represented record equally, so a
    batch mean remains the exact mean of those real records.
    """

    batches: list[tuple[np.ndarray, int]] = []
    offset = 0
    while offset + batch_size <= indices.size:
        selected = indices[offset : offset + batch_size]
        batches.append((selected, int(selected.size)))
        offset += batch_size

    remaining = indices[offset:]
    shardable_count = (remaining.size // device_count) * device_count
    if shardable_count:
        selected = remaining[:shardable_count]
        batches.append((selected, int(selected.size)))
        remaining = remaining[shardable_count:]

    divisors = [
        value
        for value in range(1, device_count + 1)
        if device_count % value == 0
    ]
    offset = 0
    while offset < remaining.size:
        actual_count = max(value for value in divisors if value <= remaining.size - offset)
        selected = remaining[offset : offset + actual_count]
        repeats = device_count // actual_count
        batches.append((np.repeat(selected, repeats), int(actual_count)))
        offset += actual_count
    return batches


def _save_sidecar(
    state: training_utils.TrainState,
    target: pathlib.Path,
    *,
    stage: str,
    resume_paths: set[tuple[Any, ...]],
    overwrite: bool,
    use_ema: bool,
    ofp_adapter_only: bool,
    adaptive_final_time_warp: bool,
) -> int:
    source_params = state.ema_params if use_ema else state.params
    if source_params is None:
        raise ValueError("Requested an EMA sidecar, but TrainState has no EMA parameters.")
    flat = traverse_util.flatten_dict(source_params.to_pure_dict())
    selected = {
        path: (
            value.astype(jax.numpy.bfloat16)
            if jax.numpy.issubdtype(value.dtype, jax.numpy.floating)
            else value
        )
        for path, value in flat.items()
        if path in resume_paths
        or _matches_stage(
            path,
            stage,
            ofp_adapter_only=ofp_adapter_only,
            adaptive_final_time_warp=adaptive_final_time_warp,
        )
    }
    if not selected:
        raise ValueError("No endpoint-student parameters matched the save filter.")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Sidecar target already exists: {target}")
    item = {"params": traverse_util.unflatten_dict(selected)}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=overwrite)
    return int(sum(np.prod(value.shape) for value in selected.values()))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    _check_audit_gate(args.dataset, args)
    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    final_params_path = output_dir / "final" / "params"
    if not args.overwrite:
        if final_params_path.exists():
            raise FileExistsError(f"Final sidecar already exists: {final_params_path}")
        if metrics_path.exists() and metrics_path.stat().st_size:
            raise FileExistsError(
                f"Metrics already exist in {output_dir}; choose a new output directory or pass --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {args.batch_size} must be divisible by device count {jax.device_count()}."
        )

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(
        arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        require_semantic_intervention=(
            args.stage in {"final", "dual"}
            and not args.ofp_sc
            and not args.pact_flow_scheduler
        ),
    )
    train_config_base = config_lib.get_config(args.config_name)
    model_config = train_config_base.model
    if args.adaptive_final_time_warp:
        if not hasattr(model_config, "adaptive_final_time_warp"):
            raise ValueError("Adaptive final time warp requires ACOTConfig.")
        model_config = dataclasses.replace(
            model_config,
            adaptive_final_time_warp=True,
        )
        train_config_base = dataclasses.replace(train_config_base, model=model_config)
    if args.pact_flow_scheduler:
        if not hasattr(model_config, "pact_flow_scheduler"):
            raise ValueError("PACT flow scheduling requires ACOTConfig.")
        model_config = dataclasses.replace(
            model_config,
            pact_flow_scheduler=True,
        )
        train_config_base = dataclasses.replace(train_config_base, model=model_config)
    expected_shapes = {
        "clean_coarse": (model_config.coarse_action_horizon, model_config.action_dim),
        "clean_actions": (model_config.action_horizon, model_config.action_dim),
    }
    for field, expected in expected_shapes.items():
        if arrays[field].shape[1:] != expected:
            raise ValueError(f"{field} shape {arrays[field].shape[1:]} does not match model {expected}.")

    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    base_params_path = checkpoint_dir / "params"
    if not base_params_path.exists():
        raise FileNotFoundError(f"Base checkpoint params not found: {base_params_path}")
    data_config = train_config_base.data.create(train_config_base.assets_dirs, model_config)
    norm_stats = _load_norm_stats(train_config_base, data_config, checkpoint_dir)
    data_config = _with_norm_stats(data_config, norm_stats)
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    observation_dataset = data_loader.transform_dataset(raw_dataset, data_config)

    resume_params, resume_paths = _load_resume_params(args.resume_sidecar_params)
    resume_has_adaptive_gate = any(
        _ADAPTIVE_FINAL_TIME_WARP_PATH.fullmatch(_path_text(path)) is not None
        for path in resume_paths
    )
    resume_has_pact_scheduler = any(
        _PACT_FLOW_SCHEDULER_PATH.fullmatch(_path_text(path)) is not None
        for path in resume_paths
    )
    if args.pact_flow_scheduler and resume_has_adaptive_gate:
        raise ValueError(
            "PACT training cannot resume from a sidecar containing adaptive_final_time_warp_gate."
        )
    if args.adaptive_final_time_warp and resume_has_pact_scheduler:
        raise ValueError(
            "Adaptive final time-warp training cannot resume from a sidecar containing pact_flow_scheduler."
        )
    trainable_filter = _train_filter(
        args.stage,
        ofp_adapter_only=args.ofp_adapter_only,
        adaptive_final_time_warp=args.adaptive_final_time_warp,
    )
    ir_weight = args.ir_loss_weight if args.variant == "ir" else 0.0
    objective_name = "endpoint"
    if args.ofp_sc:
        objective_name = (
            "ofp_sc_endpoint_anchor"
            if args.ofp_endpoint_anchor_loss_weight > 0
            else "ofp_sc"
        )
    elif args.adaptive_final_time_warp:
        objective_name = "adaptive_final_time_warp_ir"
    elif args.pact_flow_scheduler:
        objective_name = "pact_flow_distillation"
    train_config = dataclasses.replace(
        train_config_base,
        weight_loader=_BaseAndSidecarLoader(
            str(base_params_path),
            resume_params,
            adaptive_final_time_warp=args.adaptive_final_time_warp,
            pact_flow_scheduler=args.pact_flow_scheduler,
        ),
        freeze_filter=nnx.Not(trainable_filter),
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.learning_rate,
            decay_steps=max(args.train_steps, args.warmup_steps + 1),
            decay_lr=args.decay_learning_rate,
        ),
        optimizer=optimizer_lib.AdamW(
            weight_decay=args.weight_decay,
            clip_gradient_norm=args.gradient_clip_norm,
        ),
        ema_decay=args.ofp_ema_decay if args.ofp_sc else None,
        batch_size=args.batch_size,
        num_train_steps=args.train_steps,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
    )

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = train_lib.init_train_state(
        train_config,
        jax.random.key(args.seed),
        mesh,
        resume=False,
    )
    jax.block_until_ready(state)
    trainable_params = state.params.filter(trainable_filter)
    trainable_parameter_count = training_utils.count_parameters(trainable_params)
    trainable_scope = (
        "adaptive_final_time_warp_gate"
        if args.adaptive_final_time_warp
        else "pact_final_expert_and_scheduler"
        if args.pact_flow_scheduler
        else "ofp_final_adapter"
        if args.ofp_adapter_only
        else args.stage
    )
    LOGGER.info(
        "Initialized endpoint student: objective=%s stage=%s variant=%s trainable_scope=%s "
        "train=%s validation=%s trainable_params=%s",
        objective_name,
        args.stage,
        args.variant,
        trainable_scope,
        train_indices.size,
        validation_indices.size,
        trainable_parameter_count,
    )

    train_step = jax.jit(
        functools.partial(
            _endpoint_train_step,
            trainable_filter=trainable_filter,
            stage=args.stage,
            use_student_coarse=args.use_student_coarse,
            coarse_loss_weight=args.coarse_loss_weight,
            final_loss_weight=args.final_loss_weight,
            ir_loss_weight=ir_weight,
            multi_time_flow_loss_weight=args.multi_time_flow_loss_weight,
            multi_time_response_loss_weight=args.multi_time_response_loss_weight,
            multi_time_timestep=args.multi_time_timestep,
            joint_coupled_training=args.joint_coupled_training,
            adaptive_final_time_warp=args.adaptive_final_time_warp,
            pact_flow_scheduler=args.pact_flow_scheduler,
            pact_seed=args.seed,
            pact_heterogeneous_flow_probability=args.pact_heterogeneous_flow_probability,
            pact_scalar_flow_probability=args.pact_scalar_flow_probability,
            pact_plan_anchor_loss_weight=args.pact_plan_anchor_loss_weight,
            pact_uncertainty_loss_weight=args.pact_uncertainty_loss_weight,
            pact_schedule_smoothness_loss_weight=args.pact_schedule_smoothness_loss_weight,
            ofp_sc=args.ofp_sc,
            ofp_seed=args.seed,
            ofp_train_steps=args.train_steps,
            ofp_flow_anchor_loss_weight=args.ofp_flow_anchor_loss_weight,
            ofp_self_consistency_loss_weight=args.ofp_self_consistency_loss_weight,
            ofp_endpoint_anchor_loss_weight=args.ofp_endpoint_anchor_loss_weight,
            ofp_min_interval=args.ofp_min_interval,
            ofp_contraction_power=args.ofp_contraction_power,
            ofp_interval_condition_strength=args.ofp_interval_condition_strength,
            ofp_interval_condition_mode=args.ofp_interval_condition_mode,
        ),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )
    validation_step = jax.jit(
        functools.partial(
            _endpoint_validation_step,
            stage=args.stage,
            use_student_coarse=args.use_student_coarse,
            coarse_loss_weight=args.coarse_loss_weight,
            final_loss_weight=args.final_loss_weight,
            ir_loss_weight=ir_weight,
            multi_time_flow_loss_weight=args.multi_time_flow_loss_weight,
            multi_time_response_loss_weight=args.multi_time_response_loss_weight,
            multi_time_timestep=args.multi_time_timestep,
            joint_coupled_training=args.joint_coupled_training,
            adaptive_final_time_warp=args.adaptive_final_time_warp,
            pact_flow_scheduler=args.pact_flow_scheduler,
            pact_seed=args.seed,
            pact_heterogeneous_flow_probability=args.pact_heterogeneous_flow_probability,
            pact_scalar_flow_probability=args.pact_scalar_flow_probability,
            pact_plan_anchor_loss_weight=args.pact_plan_anchor_loss_weight,
            pact_uncertainty_loss_weight=args.pact_uncertainty_loss_weight,
            pact_schedule_smoothness_loss_weight=args.pact_schedule_smoothness_loss_weight,
            ofp_sc=args.ofp_sc,
            ofp_seed=args.seed,
            ofp_train_steps=args.train_steps,
            ofp_flow_anchor_loss_weight=args.ofp_flow_anchor_loss_weight,
            ofp_self_consistency_loss_weight=args.ofp_self_consistency_loss_weight,
            ofp_endpoint_anchor_loss_weight=args.ofp_endpoint_anchor_loss_weight,
            ofp_min_interval=args.ofp_min_interval,
            ofp_contraction_power=args.ofp_contraction_power,
            ofp_interval_condition_strength=args.ofp_interval_condition_strength,
            ofp_interval_condition_mode=args.ofp_interval_condition_mode,
        ),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    last_train_metrics: dict[str, float] = {}
    last_validation_metrics: dict[str, float] = {}

    metrics_mode = "w" if args.overwrite else "a"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            batch = _make_batch(
                observation_dataset,
                arrays,
                sampled,
                rng,
                deterministic_intervention=False,
            )
            batch = jax.device_put(batch, data_sharding)
            with sharding.set_mesh(mesh):
                state, train_metrics = train_step(state, batch)

            should_log = step == 1 or step % args.log_interval == 0 or step == args.train_steps
            if should_log:
                validation_sample = rng.choice(
                    validation_indices,
                    size=args.batch_size,
                    replace=validation_indices.size < args.batch_size,
                )
                validation_batch = _make_batch(
                    observation_dataset,
                    arrays,
                    validation_sample,
                    rng,
                    deterministic_intervention=True,
                )
                validation_batch = jax.device_put(validation_batch, data_sharding)
                with sharding.set_mesh(mesh):
                    validation_metrics = validation_step(state, validation_batch)
                last_train_metrics = {
                    f"train/{name}": float(value)
                    for name, value in jax.device_get(train_metrics).items()
                }
                last_validation_metrics = {
                    f"validation/{name}": float(value)
                    for name, value in jax.device_get(validation_metrics).items()
                }
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **last_train_metrics,
                    **last_validation_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "step=%s %s",
                    step,
                    " ".join(
                        f"{name}={value:.6f}"
                        for name, value in record.items()
                        if name.startswith(("train/", "validation/"))
                    ),
                )

            if step % args.checkpoint_interval == 0 and step != args.train_steps:
                saved_params = _save_sidecar(
                    state,
                    output_dir / f"step_{step:06d}" / "params",
                    stage=args.stage,
                    resume_paths=resume_paths,
                    overwrite=args.overwrite,
                    use_ema=args.ofp_sc,
                    ofp_adapter_only=args.ofp_adapter_only,
                    adaptive_final_time_warp=args.adaptive_final_time_warp,
                )
                LOGGER.info("Saved step %s delta sidecar with %s parameters.", step, saved_params)

    full_validation_sums: dict[str, float] = {}
    full_validation_count = 0
    full_validation_rng = np.random.default_rng(args.seed)
    for validation_sample, represented_count in _full_validation_index_batches(
        validation_indices,
        batch_size=args.batch_size,
        device_count=jax.device_count(),
    ):
        validation_batch = _make_batch(
            observation_dataset,
            arrays,
            validation_sample,
            full_validation_rng,
            deterministic_intervention=True,
        )
        validation_batch = jax.device_put(validation_batch, data_sharding)
        with sharding.set_mesh(mesh):
            validation_metrics = validation_step(state, validation_batch)
        for name, value in jax.device_get(validation_metrics).items():
            full_validation_sums[name] = full_validation_sums.get(
                name, 0.0
            ) + float(value) * represented_count
        full_validation_count += represented_count
    if full_validation_count != validation_indices.size:
        raise RuntimeError(
            "Full validation represented "
            f"{full_validation_count} records, expected {validation_indices.size}."
        )
    full_validation_values = {
        name: value / full_validation_count
        for name, value in full_validation_sums.items()
    }
    for name in tuple(full_validation_values):
        if not name.endswith("_rmse"):
            continue
        mse_name = f"{name[:-5]}_mse"
        if mse_name in full_validation_values:
            full_validation_values[name] = float(
                np.sqrt(max(full_validation_values[mse_name], 0.0))
            )
    full_validation_metrics = {
        f"validation/{name}": value
        for name, value in full_validation_values.items()
    }

    saved_params = _save_sidecar(
        state,
        final_params_path,
        stage=args.stage,
        resume_paths=resume_paths,
        overwrite=args.overwrite,
        use_ema=args.ofp_sc,
        ofp_adapter_only=args.ofp_adapter_only,
        adaptive_final_time_warp=args.adaptive_final_time_warp,
    )
    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": list(args.dataset),
        "stage": args.stage,
        "variant": args.variant,
        "training_objective": objective_name,
        "resume_sidecar_params": args.resume_sidecar_params,
        "causal_audit_summary": args.causal_audit_summary,
        "multi_time_flow_loss_weight": args.multi_time_flow_loss_weight,
        "multi_time_response_loss_weight": args.multi_time_response_loss_weight,
        "multi_time_timestep": args.multi_time_timestep,
        "joint_coupled_training": args.joint_coupled_training,
        "ofp_sc": args.ofp_sc,
        "ofp_flow_anchor_loss_weight": args.ofp_flow_anchor_loss_weight,
        "ofp_self_consistency_loss_weight": args.ofp_self_consistency_loss_weight,
        "ofp_endpoint_anchor_loss_weight": args.ofp_endpoint_anchor_loss_weight,
        "ofp_ema_decay": args.ofp_ema_decay if args.ofp_sc else None,
        "ofp_min_interval": args.ofp_min_interval,
        "ofp_contraction_power": args.ofp_contraction_power,
        "ofp_interval_condition_strength": args.ofp_interval_condition_strength,
        "ofp_interval_condition_mode": args.ofp_interval_condition_mode,
        "ofp_adapter_only": args.ofp_adapter_only,
        "ofp_self_guidance": False,
        "adaptive_final_time_warp": args.adaptive_final_time_warp,
        "adaptive_final_time_warp_center": 0.05 if args.adaptive_final_time_warp else None,
        "adaptive_final_time_warp_radius": 0.05 if args.adaptive_final_time_warp else None,
        "pact_flow_scheduler": args.pact_flow_scheduler,
        "pact_heterogeneous_flow_probability": (
            args.pact_heterogeneous_flow_probability if args.pact_flow_scheduler else None
        ),
        "pact_scalar_flow_probability": (
            args.pact_scalar_flow_probability if args.pact_flow_scheduler else None
        ),
        "pact_plan_anchor_loss_weight": (
            args.pact_plan_anchor_loss_weight if args.pact_flow_scheduler else None
        ),
        "pact_uncertainty_loss_weight": (
            args.pact_uncertainty_loss_weight if args.pact_flow_scheduler else None
        ),
        "pact_endpoint_probability": (
            1.0
            - args.pact_heterogeneous_flow_probability
            - args.pact_scalar_flow_probability
            if args.pact_flow_scheduler
            else None
        ),
        "pact_schedule_smoothness_loss_weight": (
            args.pact_schedule_smoothness_loss_weight if args.pact_flow_scheduler else None
        ),
        "trainable_scope": trainable_scope,
        "trainable_parameter_count": trainable_parameter_count,
        "saved_parameter_source": "ema" if args.ofp_sc else "online",
        "validation_parameter_source": "ema" if args.ofp_sc else "online",
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "completed_steps": args.train_steps,
        "saved_parameter_count": saved_params,
        "final_params_path": str(final_params_path.resolve()),
        "last_train_metrics": last_train_metrics,
        "last_validation_metrics": last_validation_metrics,
        "full_validation_metrics": full_validation_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "frozen_contract": "VLM, IAR, and reasoning fusion frozen; only selected EAR/final local branches train.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Training complete; final delta sidecar: %s", final_params_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
