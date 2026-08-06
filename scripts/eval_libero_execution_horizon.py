"""Closed-loop LIBERO evaluation for Budgeted Event V2-P execution horizons."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import time
from typing import Any

import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import rl_selector
from openpi.execution_horizon import v2

LEGACY_MODES = (
    "original",
    "fixed_h9",
    "exact_batched_mc_v2",
    "v2_distilled",
    "v2_value_refined",
)
SELECTOR_MODES = ("q_guided_selector", "sft_selector", "ppo_selector")
MODES = (*LEGACY_MODES, *SELECTOR_MODES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit episode/state IDs. Overrides --num-trials-per-task.",
    )
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(LEGACY_MODES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument(
        "--final-denoising-steps",
        type=int,
        default=None,
        help=(
            "Optional independent NFE for the final action expert. When unset, "
            "the policy keeps its existing final-sampling behavior."
        ),
    )
    parser.add_argument(
        "--final-time-warp-alpha",
        type=float,
        default=0.0,
        help=(
            "Endpoint-directed final-expert time calibration in [0,1). "
            "Alpha=0 preserves the legacy sampler; alpha>0 uses the legacy "
            "fast path at effective time (1-alpha)*t."
        ),
    )
    parser.add_argument(
        "--final-token-time-warp-alpha",
        type=float,
        nargs=10,
        default=None,
        metavar="ALPHA",
        help=(
            "Opt-in per-action-token final time calibration. Provide exactly "
            "10 values in [0,1), one for each action token."
        ),
    )
    parser.add_argument(
        "--pact-flow-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to the learned PACT per-token flow-time scheduler. The served "
            "model must include the PACT head; formal PACT evaluation fixes EAR, "
            "final generation, and execution horizon to 1/1/10."
        ),
    )
    parser.add_argument(
        "--final-endpoint-condition-strength",
        type=float,
        default=0.0,
        help=(
            "Static endpoint-time embedding strength in [0,1]. This runs the "
            "endpoint-only half-concat flow in one final suffix call."
        ),
    )
    parser.add_argument(
        "--final-midpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the shared-compute explicit-midpoint final solver: one endpoint "
            "velocity plus one midpoint velocity, with a second-order update."
        ),
    )
    parser.add_argument(
        "--adaptive-final-time-warp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the opt-in zero-init prefix/state-conditioned one-step final "
            "time-warp gate from the endpoint sidecar."
        ),
    )
    parser.add_argument(
        "--compact-alpha-router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the served state32+absolute-step outcome router. The server "
            "must load its NPZ separately; this path hard-routes alpha in {0,.05} "
            "and forces direct final NFE1."
        ),
    )
    parser.add_argument(
        "--harp-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the separately loaded HARP temporal residual after direct "
            "IR NFE1. This path requires EAR/final NFE1 and leaves gripper unchanged."
        ),
    )
    parser.add_argument(
        "--harp-gripper-event",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the separately loaded ES-HARP gripper event student after matched "
            "direct IR NFE1. Only action dimension seven may change."
        ),
    )
    parser.add_argument(
        "--final-hybrid-mode",
        choices=("none", "control_nfe2", "gripper_nfe2"),
        default="none",
        help=(
            "Opt-in final-expert ceiling diagnostic. Reuse one prefix/EAR/IAR/noise, "
            "run direct NFE1 and legacy NFE2, then take either the first six control "
            "dimensions or the remaining gripper dimensions from NFE2."
        ),
    )
    parser.add_argument(
        "--contextual-fusion-mode",
        choices=(
            "compiler",
            "expert",
            "control_compiler",
            "gripper_compiler",
            "blend50",
            "semantic_gate",
            "spectral_compiler_gripper",
            "spectral_expert_gripper",
            "phase_chunk_compiler_expert",
            "phase_chunk_expert_compiler",
            "phase_compiler_expert",
            "phase_expert_compiler",
        ),
        default="compiler",
        help="Opt-in fusion diagnostic when the server loads a contextual compiler.",
    )
    parser.add_argument(
        "--contextual-fusion-translation-tau",
        type=float,
        default=0.20,
        help="Normalized translation-group disagreement threshold for semantic_gate.",
    )
    parser.add_argument(
        "--contextual-fusion-rotation-tau",
        type=float,
        default=0.20,
        help="Normalized rotation-group disagreement threshold for semantic_gate.",
    )
    parser.add_argument(
        "--contextual-fusion-gripper-tau",
        type=float,
        default=0.15,
        help="Normalized gripper margin below which semantic_gate treats a step as an event.",
    )
    parser.add_argument(
        "--contextual-fusion-gate-width",
        type=float,
        default=0.05,
        help="Positive sigmoid width for semantic_gate continuous-control gates.",
    )
    parser.add_argument(
        "--contextual-fusion-high-disagreement-source",
        "--contextual-fusion-high-source",
        dest="contextual_fusion_high_disagreement_source",
        choices=("expert", "compiler"),
        default="expert",
        help="Branch used at high-disagreement/contact-event steps by semantic_gate.",
    )
    parser.add_argument(
        "--contextual-fusion-switch-step",
        type=int,
        default=400,
        help=(
            "Absolute environment step where a phase_* diagnostic switches from "
            "its first branch to its second branch."
        ),
    )
    parser.add_argument(
        "--selective-gripper-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the deployed direct A1, but conditionally run one exact midpoint "
            "suffix and replace only its gripper dimensions."
        ),
    )
    parser.add_argument(
        "--selective-gripper-tau",
        type=float,
        default=0.15,
        help=(
            "Normalized A1 gripper uncertainty threshold. Refinement triggers when "
            "the chunk minimum absolute gripper value is below tau or its sign changes."
        ),
    )
    parser.add_argument(
        "--selective-refinement-mode",
        choices=("gripper", "full"),
        default="gripper",
        help=(
            "When selective refinement triggers, take only the NFE2 gripper or the "
            "complete direct-consistent second-half action chunk."
        ),
    )
    parser.add_argument(
        "--ofp-interval-flow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the opt-in OFP-SC final interval map instead of the legacy final sampler.",
    )
    parser.add_argument(
        "--ofp-warm-start-time",
        type=float,
        default=1.0,
        help="Previous-chunk noising time in (0,1]; 1 disables its influence.",
    )
    parser.add_argument(
        "--ofp-interval-condition-strength",
        type=float,
        default=1.0,
        help=(
            "OFP interval-conditioning alpha in [0,1]. Its meaning is selected by "
            "--ofp-interval-condition-mode; must match training."
        ),
    )
    parser.add_argument(
        "--ofp-interval-condition-mode",
        choices=("half_concat", "time_blend"),
        default="half_concat",
        help=(
            "OFP interval embedding mode. time_blend embeds t + alpha * (r - t) "
            "with one posemb call; must match training."
        ),
    )
    parser.add_argument("--original-horizon", type=int, default=5)
    parser.add_argument("--fixed-horizon", type=int, default=9)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--v2-min-horizon", type=int, default=3)
    parser.add_argument("--v2-risk-threshold", type=float, default=1.5)
    parser.add_argument("--v2-final-weight", type=float, default=0.5)
    parser.add_argument("--v2-action-cot-weight", type=float, default=0.5)
    parser.add_argument("--v2-final-risk-threshold", type=float, default=None)
    parser.add_argument("--v2-action-cot-risk-threshold", type=float, default=None)
    parser.add_argument("--v2-target-average-horizon", type=float, default=9.0)
    parser.add_argument("--v2-initial-budget", type=float, default=6.0)
    parser.add_argument("--v2-budget-capacity", type=float, default=12.0)
    parser.add_argument("--value-candidates", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--q-min-success-probability", type=float, default=0.90)
    parser.add_argument("--q-max-timeout-probability", type=float, default=0.20)
    parser.add_argument("--q-risk-slack-steps", type=int, default=0)
    parser.add_argument("--q-guided-selector-params", default=None)
    parser.add_argument("--ppo-selector-params", default=None)
    parser.add_argument(
        "--selector-sample-modes",
        nargs="*",
        choices=("sft_selector", "ppo_selector"),
        default=[],
        help="Sample the actor for online collection; normal evaluation stays deterministic.",
    )
    parser.add_argument("--selector-temperature", type=float, default=1.0)
    parser.add_argument("--selector-min-success-probability", type=float, default=0.5)
    parser.add_argument("--selector-reference-slack", type=float, default=0.05)
    parser.add_argument("--selector-q-tie-margin", type=float, default=0.03)
    parser.add_argument(
        "--record-selector-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist frozen selector features in decisions.csv for an online RL update.",
    )
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume a matching interrupted evaluation from its per-episode CSV journal.",
    )
    return parser


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _request(
    client: websocket_policy.WebsocketClientPolicy,
    element: dict[str, Any],
    *,
    mode: str,
    seed: int,
    previous_actions: np.ndarray | None,
    previous_horizon: int,
    budget_fraction: float,
    episode_progress: float,
    absolute_decision_step: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        **element,
        "policy_seed": np.asarray(seed, dtype=np.int64),
        "profile_policy_timing": np.asarray(1, dtype=np.bool_),
        "action_cot_denoising_steps": np.asarray(args.action_cot_denoising_steps, dtype=np.int32),
        # Always expose rollout phase. Ordinary policies ignore it, compact
        # routers and phase fusion diagnostics consume the same protocol field.
        "action_cot_absolute_decision_step": np.asarray(
            absolute_decision_step, dtype=np.int32
        ),
    }
    if args.final_denoising_steps is not None:
        request["action_cot_final_denoising_steps"] = np.asarray(
            args.final_denoising_steps,
            dtype=np.int32,
        )
    if args.final_time_warp_alpha > 0.0:
        request["action_cot_final_time_warp_alpha"] = np.asarray(
            args.final_time_warp_alpha,
            dtype=np.float32,
        )
    if args.final_token_time_warp_alpha is not None:
        request["action_cot_final_token_time_warp_alpha"] = np.asarray(
            args.final_token_time_warp_alpha,
            dtype=np.float32,
        )
    if args.pact_flow_scheduler:
        request["action_cot_pact_flow_scheduler"] = np.asarray(
            True,
            dtype=np.bool_,
        )
    if args.final_endpoint_condition_strength > 0.0:
        request["action_cot_final_endpoint_condition_strength"] = np.asarray(
            args.final_endpoint_condition_strength,
            dtype=np.float32,
        )
    if args.final_midpoint:
        request["action_cot_final_midpoint"] = np.asarray(True, dtype=np.bool_)
    if args.adaptive_final_time_warp:
        request["action_cot_adaptive_final_time_warp"] = np.asarray(
            True,
            dtype=np.bool_,
        )
    if args.compact_alpha_router:
        request["action_cot_compact_alpha_router"] = np.asarray(True, dtype=np.bool_)
    if args.harp_residual:
        request["action_cot_harp_residual"] = np.asarray(True, dtype=np.bool_)
    if args.harp_gripper_event:
        request["action_cot_harp_gripper_event"] = np.asarray(True, dtype=np.bool_)
    if args.final_hybrid_mode != "none":
        request["action_cot_final_hybrid_mode"] = args.final_hybrid_mode
    if args.contextual_fusion_mode != "compiler":
        request["action_cot_contextual_fusion_mode"] = args.contextual_fusion_mode
    if args.contextual_fusion_mode in {
        "phase_compiler_expert",
        "phase_expert_compiler",
    }:
        request["action_cot_contextual_fusion_switch_step"] = np.asarray(
            args.contextual_fusion_switch_step, dtype=np.int32
        )
    if args.contextual_fusion_mode == "semantic_gate":
        request.update(
            {
                "action_cot_contextual_fusion_translation_tau": np.asarray(
                    args.contextual_fusion_translation_tau, dtype=np.float32
                ),
                "action_cot_contextual_fusion_rotation_tau": np.asarray(
                    args.contextual_fusion_rotation_tau, dtype=np.float32
                ),
                "action_cot_contextual_fusion_gripper_tau": np.asarray(
                    args.contextual_fusion_gripper_tau, dtype=np.float32
                ),
                "action_cot_contextual_fusion_gate_width": np.asarray(
                    args.contextual_fusion_gate_width, dtype=np.float32
                ),
                "action_cot_contextual_fusion_high_disagreement_source": (
                    args.contextual_fusion_high_disagreement_source
                ),
            }
        )
    if args.selective_gripper_refinement:
        request.update(
            {
                "action_cot_selective_gripper_refinement": np.asarray(True, dtype=np.bool_),
                "action_cot_selective_gripper_tau": np.asarray(
                    args.selective_gripper_tau,
                    dtype=np.float32,
                ),
                "action_cot_selective_refinement_mode": args.selective_refinement_mode,
            }
        )
    if args.ofp_interval_flow:
        request.update(
            {
                "action_cot_ofp_interval_flow": np.asarray(1, dtype=np.bool_),
                "action_cot_ofp_warm_start_actions": (
                    np.asarray(previous_actions, dtype=np.float32)
                    if previous_actions is not None
                    else np.zeros((10, 7), dtype=np.float32)
                ),
                "action_cot_ofp_warm_start_valid": np.asarray(previous_actions is not None),
                "action_cot_ofp_warm_start_time": np.asarray(
                    args.ofp_warm_start_time,
                    dtype=np.float32,
                ),
                "action_cot_ofp_interval_condition_strength": np.asarray(
                    args.ofp_interval_condition_strength,
                    dtype=np.float32,
                ),
                "action_cot_ofp_interval_condition_mode": args.ofp_interval_condition_mode,
            }
        )
    if mode == "exact_batched_mc_v2":
        request["batched_mc_samples"] = np.asarray(args.teacher_samples, dtype=np.int32)
    if mode in {"v2_distilled", "v2_value_refined", *SELECTOR_MODES}:
        request.update(
            {
                "run_execution_horizon_predictor": np.asarray(1, dtype=np.bool_),
                "execution_horizon_previous_actions": (
                    np.asarray(previous_actions, dtype=np.float32)
                    if previous_actions is not None
                    else np.zeros((10, 7), dtype=np.float32)
                ),
                "execution_horizon_previous_h": np.asarray(previous_horizon, dtype=np.int32),
                "execution_horizon_budget_balance": np.asarray(budget_fraction, dtype=np.float32),
                "execution_horizon_episode_progress": np.asarray(episode_progress, dtype=np.float32),
                "execution_horizon_previous_valid": np.asarray(previous_actions is not None),
            }
        )
    started = time.perf_counter()
    result = client.infer(request)
    wall_ms = (time.perf_counter() - started) * 1000.0
    policy_timing = result.get("policy_timing", {})
    server_timing = result.get("server_timing", {})

    def _result_scalar(name: str) -> float:
        return float(np.asarray(result.get(name, np.nan)).item())

    pact_tau_json = ""
    pact_tau_logits_json = ""
    pact_tau_mean = float("nan")
    pact_tau_std = float("nan")
    pact_tau_min = float("nan")
    pact_tau_max = float("nan")
    if args.pact_flow_scheduler:
        if "pact_flow_tau" not in result:
            raise KeyError("PACT response is missing pact_flow_tau diagnostics.")
        pact_tau = np.asarray(result["pact_flow_tau"], dtype=np.float32).reshape((-1,))
        if pact_tau.shape != (10,):
            raise ValueError(f"PACT tau must have shape (10,), got {pact_tau.shape}.")
        if not np.all(np.isfinite(pact_tau)) or np.any(pact_tau < 0.0) or np.any(pact_tau > 1.0):
            raise ValueError("PACT tau values must be finite and in [0, 1].")
        pact_tau_json = json.dumps(pact_tau.tolist(), separators=(",", ":"))
        pact_tau_mean = float(np.mean(pact_tau))
        pact_tau_std = float(np.std(pact_tau))
        pact_tau_min = float(np.min(pact_tau))
        pact_tau_max = float(np.max(pact_tau))
        if "pact_flow_tau_logits" in result:
            pact_tau_logits = np.asarray(
                result["pact_flow_tau_logits"], dtype=np.float32
            ).reshape((-1,))
            if pact_tau_logits.shape != (10,) or not np.all(np.isfinite(pact_tau_logits)):
                raise ValueError(
                    "PACT tau logits must contain exactly 10 finite values when returned."
                )
            pact_tau_logits_json = json.dumps(
                pact_tau_logits.tolist(), separators=(",", ":")
            )

    return result, {
        "wall_ms": wall_ms,
        "policy_ms": float(policy_timing.get("infer_ms", np.nan)),
        "server_ms": float(server_timing.get("infer_ms", wall_ms)),
        "predictor_ms": float(policy_timing.get("execution_horizon_predictor_ms", np.nan)),
        "batched_teacher_ms": float(policy_timing.get("batched_mc_teacher_ms", np.nan)),
        "action_expert_ms": float(policy_timing.get("action_expert_ms", np.nan)),
        "pact_flow_tau_json": pact_tau_json,
        "pact_flow_tau_logits_json": pact_tau_logits_json,
        "pact_flow_tau_mean": pact_tau_mean,
        "pact_flow_tau_std": pact_tau_std,
        "pact_flow_tau_min": pact_tau_min,
        "pact_flow_tau_max": pact_tau_max,
        "compact_alpha_router_ms": float(
            policy_timing.get("compact_alpha_router_ms", np.nan)
        ),
        "contextual_fusion_ms": float(
            policy_timing.get("contextual_fusion_ms", np.nan)
        ),
        "contextual_fusion_translation_disagreement_mean": _result_scalar(
            "contextual_fusion_translation_disagreement_mean"
        ),
        "contextual_fusion_rotation_disagreement_mean": _result_scalar(
            "contextual_fusion_rotation_disagreement_mean"
        ),
        "contextual_fusion_gripper_conflict_rate": _result_scalar(
            "contextual_fusion_gripper_conflict_rate"
        ),
        "contextual_fusion_expert_gate_rate": _result_scalar(
            "contextual_fusion_expert_gate_rate"
        ),
        "contextual_fusion_high_disagreement_source_expert": _result_scalar(
            "contextual_fusion_high_disagreement_source_expert"
        ),
        "contextual_fusion_phase_selected_expert": _result_scalar(
            "contextual_fusion_phase_selected_expert"
        ),
        "contextual_fusion_absolute_decision_step": _result_scalar(
            "contextual_fusion_absolute_decision_step"
        ),
        "contextual_fusion_switch_step": _result_scalar(
            "contextual_fusion_switch_step"
        ),
        "harp_residual_ms": float(policy_timing.get("harp_residual_ms", np.nan)),
        "harp_gripper_event_ms": float(
            policy_timing.get("harp_gripper_event_ms", np.nan)
        ),
        "selective_gripper_verifier_ms": float(
            policy_timing.get("selective_gripper_verifier_ms", np.nan)
        ),
        "selective_gripper_refinement_ms": float(
            policy_timing.get("selective_gripper_refinement_ms", np.nan)
        ),
        "selective_gripper_triggered": float(
            np.asarray(result.get("selective_gripper_triggered", np.nan)).item()
        ),
        "selective_gripper_trigger_min_abs": float(
            np.asarray(result.get("selective_gripper_trigger_min_abs", np.nan)).item()
        ),
        "selective_gripper_trigger_sign_transition": float(
            np.asarray(
                result.get("selective_gripper_trigger_sign_transition", np.nan)
            ).item()
        ),
    }


def _risk_config(args: argparse.Namespace) -> v2.V2RiskConfig:
    return v2.V2RiskConfig(
        risk_threshold=args.v2_risk_threshold,
        final_weight=args.v2_final_weight,
        action_cot_weight=args.v2_action_cot_weight,
        final_risk_threshold=args.v2_final_risk_threshold,
        action_cot_risk_threshold=args.v2_action_cot_risk_threshold,
    )


def _load_selectors(args: argparse.Namespace) -> dict[str, rl_selector.FrozenFeatureSelector]:
    selectors: dict[str, rl_selector.FrozenFeatureSelector] = {}
    needs_q_selector = any(mode in {"q_guided_selector", "sft_selector"} for mode in args.modes)
    if needs_q_selector:
        if args.q_guided_selector_params is None:
            raise ValueError("q_guided_selector/sft_selector requires --q-guided-selector-params.")
        selector = rl_selector.FrozenFeatureSelector.load(args.q_guided_selector_params)
        selectors["q_guided_selector"] = selector
        selectors["sft_selector"] = selector
    if "ppo_selector" in args.modes:
        if args.ppo_selector_params is None:
            raise ValueError("ppo_selector requires --ppo-selector-params.")
        selectors["ppo_selector"] = rl_selector.FrozenFeatureSelector.load(args.ppo_selector_params)
    return selectors


def _select_horizon(
    mode: str,
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    budget_state: v2.EpisodeBudgetState,
    selector: rl_selector.FrozenFeatureSelector | None = None,
    selector_rng: np.random.Generator | None = None,
) -> tuple[int, dict[str, Any]]:
    if mode == "original":
        return args.original_horizon, {"raw_horizon": args.original_horizon, "budget_limited": 0.0}
    if mode == "fixed_h9":
        return args.fixed_horizon, {"raw_horizon": args.fixed_horizon, "budget_limited": 0.0}
    if mode in SELECTOR_MODES:
        if selector is None:
            raise ValueError(f"{mode} requires a selector sidecar.")
        feature = rl_selector.build_selector_feature(result)
        policy = "q" if mode == "q_guided_selector" else "actor"
        decision = selector.decide(
            feature,
            policy=policy,
            minimum_success_probability=args.selector_min_success_probability,
            reference_slack=args.selector_reference_slack,
            q_tie_margin=args.selector_q_tie_margin,
            sample=mode in args.selector_sample_modes,
            temperature=args.selector_temperature,
            rng=selector_rng,
        )
        selector_info = decision.as_json_dict()
        if not args.record_selector_features:
            selector_info.pop("selector_feature")
        return decision.horizon, {
            "raw_horizon": decision.horizon,
            "budget_limited": 0.0,
            "selector_policy": policy,
            "selector_sampled": mode in args.selector_sample_modes,
            **selector_info,
        }

    risk_config = _risk_config(args)
    entropy_candidates = list(range(args.v2_min_horizon, 11))
    if mode == "exact_batched_mc_v2":
        risk = v2.risk_targets_from_normalized_mc(
            result["mc_coarse_actions_normalized"],
            result["mc_actions_normalized"],
            config=risk_config,
        )
        final_risk = np.asarray(risk["final_risk"])
        action_cot_risk = np.asarray(risk["action_cot_risk"])
        fused_risk = np.asarray(risk["fused_risk"])
    else:
        final_risk = np.asarray(result["execution_horizon_final_risk"], dtype=np.float64)
        action_cot_risk = np.asarray(result["execution_horizon_action_cot_risk"], dtype=np.float64)
        fused_risk = np.asarray(result["execution_horizon_fused_risk"], dtype=np.float64)

    entropy_raw_horizon, event_mask = v2.distilled_raw_horizon(
        final_risk,
        action_cot_risk,
        fused_risk,
        candidates=entropy_candidates,
        config=risk_config,
    )
    raw_horizon = entropy_raw_horizon
    candidates = entropy_candidates
    q_info: dict[str, Any] = {}
    if mode == "v2_value_refined":
        success_probability = _sigmoid(result["execution_horizon_success_logits"])
        timeout_probability = _sigmoid(result["execution_horizon_timeout_logits"])
        candidates = sorted(set(args.value_candidates))
        raw_horizon, filters = v2.value_refined_raw_horizon(
            entropy_raw_horizon=entropy_raw_horizon,
            success_probability=success_probability,
            timeout_probability=timeout_probability,
            fused_risk=fused_risk,
            config=v2.ValueRefinementConfig(
                minimum_success_probability=args.q_min_success_probability,
                maximum_timeout_probability=args.q_max_timeout_probability,
                risk_threshold=args.v2_risk_threshold,
                risk_slack_steps=args.q_risk_slack_steps,
                candidates=tuple(candidates),
            ),
        )
        q_info = {
            "success_probability": success_probability.tolist(),
            "timeout_probability": timeout_probability.tolist(),
            "q_eligible": np.asarray(filters["eligible"], dtype=np.int8).tolist(),
            "predicted_remaining_calls": np.asarray(
                result["execution_horizon_remaining_calls"], dtype=np.float64
            ).tolist(),
            "predicted_remaining_steps": np.asarray(
                result["execution_horizon_remaining_steps"], dtype=np.float64
            ).tolist(),
        }

    final_horizon, budget_info = v2.apply_episode_budget(
        raw_horizon,
        candidates,
        config=v2.EpisodeBudgetConfig(
            target_average_horizon=args.v2_target_average_horizon,
            capacity=args.v2_budget_capacity,
        ),
        state=budget_state,
    )
    raw_h_prediction = None
    if "execution_horizon_raw_h_logits" in result:
        raw_h_prediction = int(np.argmax(result["execution_horizon_raw_h_logits"]) + 1)
    return final_horizon, {
        "raw_horizon": raw_horizon,
        "entropy_raw_horizon": entropy_raw_horizon,
        "raw_h_prediction": raw_h_prediction,
        "event_mask": np.asarray(event_mask, dtype=np.int8).tolist(),
        "final_risk": final_risk.tolist(),
        "action_cot_risk": action_cot_risk.tolist(),
        "fused_risk": fused_risk.tolist(),
        **budget_info,
        **q_info,
    }


def _warmup(
    client: websocket_policy.WebsocketClientPolicy,
    task_suite,
    args: argparse.Namespace,
) -> None:
    if args.warmup_requests <= 0:
        return
    task = task_suite.get_task(args.task_start)
    states = task_suite.get_task_init_states(args.task_start)
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        observation = env.set_init_state(states[args.initial_state_offset % len(states)])
        absolute_step = 0
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            absolute_step += 1
            if done:
                break
        element = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        for mode in args.modes:
            for repeat in range(args.warmup_requests):
                warmup_steps = [absolute_step]
                if (
                    args.contextual_fusion_mode
                    in {"phase_compiler_expert", "phase_expert_compiler"}
                    and args.contextual_fusion_switch_step > 0
                ):
                    # Compile both single-branch paths before timing. Deployment
                    # requests still execute only the branch selected at their
                    # actual absolute step.
                    warmup_steps = [
                        args.contextual_fusion_switch_step - 1,
                        args.contextual_fusion_switch_step,
                    ]
                for warmup_step in warmup_steps:
                    _request(
                        client,
                        element,
                        mode=mode,
                        seed=args.seed + repeat,
                        previous_actions=None,
                        previous_horizon=10,
                        budget_fraction=args.v2_initial_budget / args.v2_budget_capacity,
                        episode_progress=0.0,
                        absolute_decision_step=warmup_step,
                        args=args,
                    )
    finally:
        libero_eval._safe_close_env(env)


def _run_episode(
    *,
    mode: str,
    task_id: int,
    episode: int,
    task_suite,
    client: websocket_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
    selector: rl_selector.FrozenFeatureSelector | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_started = time.perf_counter()
    task = task_suite.get_task(task_id)
    states = task_suite.get_task_init_states(task_id)
    state_id = (args.initial_state_offset + episode) % len(states)
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    timings: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    horizons: list[int] = []
    compact_alpha_router_scores: list[float] = []
    compact_alpha_router_alphas: list[float] = []
    selective_gripper_decisions = 0
    selective_gripper_triggers = 0
    policy_calls = 0
    sampled_chunks = 0
    step = 0
    success = False
    previous_actions: np.ndarray | None = None
    previous_horizon = 10
    budget_state = v2.EpisodeBudgetState(balance=min(args.v2_initial_budget, args.v2_budget_capacity))
    max_steps = libero_eval._max_steps(args.task_suite_name)
    try:
        env.reset()
        observation = env.set_init_state(states[state_id])
        environment_horizon = libero_eval._env_horizon(env)
        episode_step_limit = max_steps + args.num_steps_wait
        if environment_horizon is not None:
            episode_step_limit = min(episode_step_limit, environment_horizon)
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
                break

        while not success and step < episode_step_limit:
            element = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
            request_seed = args.seed + task_id * 1_000_000 + episode * 10_000 + step
            result, timing = _request(
                client,
                element,
                mode=mode,
                seed=request_seed,
                previous_actions=previous_actions,
                previous_horizon=previous_horizon,
                budget_fraction=budget_state.balance / args.v2_budget_capacity,
                episode_progress=step / max(episode_step_limit, 1),
                absolute_decision_step=step,
                args=args,
            )
            policy_calls += 1
            sampled_chunks += args.teacher_samples if mode == "exact_batched_mc_v2" else 1
            timings.append(timing)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            horizon, selector_info = _select_horizon(
                mode,
                result,
                args=args,
                budget_state=budget_state,
                selector=selector,
                selector_rng=np.random.default_rng(request_seed + 991),
            )
            selected_horizon = horizon
            horizon = min(horizon, len(action_chunk), episode_step_limit - step)
            if horizon <= 0:
                break
            horizons.append(horizon)
            compact_router_info: dict[str, Any] = {}
            if args.compact_alpha_router:
                required_router_outputs = {
                    "compact_alpha_router_score",
                    "compact_alpha_router_selected_alpha",
                    "compact_alpha_router_absolute_decision_step",
                }
                missing_router_outputs = sorted(required_router_outputs.difference(result))
                if missing_router_outputs:
                    raise KeyError(
                        "Compact alpha router response is missing outputs: "
                        f"{missing_router_outputs}"
                    )
                router_score = float(np.asarray(result["compact_alpha_router_score"]).item())
                router_alpha = float(
                    np.asarray(result["compact_alpha_router_selected_alpha"]).item()
                )
                returned_step = int(
                    np.asarray(result["compact_alpha_router_absolute_decision_step"]).item()
                )
                if returned_step != step:
                    raise ValueError(
                        f"Compact alpha router returned absolute step {returned_step}, requested {step}."
                    )
                if not np.isfinite(router_score):
                    raise ValueError("Compact alpha router returned a non-finite score.")
                if not (np.isclose(router_alpha, 0.0) or np.isclose(router_alpha, 0.05)):
                    raise ValueError(
                        f"Compact alpha router returned unsupported alpha {router_alpha}."
                    )
                router_alpha = 0.05 if np.isclose(router_alpha, 0.05) else 0.0
                compact_alpha_router_scores.append(router_score)
                compact_alpha_router_alphas.append(router_alpha)
                compact_router_info = {
                    "compact_alpha_router_score": router_score,
                    "compact_alpha_router_selected_alpha": router_alpha,
                    "compact_alpha_router_ms": timing["compact_alpha_router_ms"],
                }
            selective_gripper_info: dict[str, Any] = {}
            if args.selective_gripper_refinement:
                triggered = int(timing["selective_gripper_triggered"])
                sign_transition = int(
                    timing["selective_gripper_trigger_sign_transition"]
                )
                trigger_min_abs = timing["selective_gripper_trigger_min_abs"]
                if triggered not in {0, 1} or sign_transition not in {0, 1}:
                    raise ValueError("Selective gripper diagnostics must be binary.")
                if not np.isfinite(trigger_min_abs):
                    raise ValueError("Selective gripper trigger minimum must be finite.")
                selective_gripper_decisions += 1
                selective_gripper_triggers += triggered
                selective_gripper_info = {
                    "selective_gripper_triggered": triggered,
                    "selective_gripper_trigger_min_abs": trigger_min_abs,
                    "selective_gripper_trigger_sign_transition": sign_transition,
                    "selective_gripper_verifier_ms": timing[
                        "selective_gripper_verifier_ms"
                    ],
                    "selective_gripper_refinement_ms": timing[
                        "selective_gripper_refinement_ms"
                    ],
                }
            contextual_fusion_info: dict[str, Any] = {}
            if args.contextual_fusion_mode == "semantic_gate":
                contextual_fusion_info = {
                    "contextual_fusion_ms": timing["contextual_fusion_ms"],
                    "contextual_fusion_translation_disagreement_mean": timing[
                        "contextual_fusion_translation_disagreement_mean"
                    ],
                    "contextual_fusion_rotation_disagreement_mean": timing[
                        "contextual_fusion_rotation_disagreement_mean"
                    ],
                    "contextual_fusion_gripper_conflict_rate": timing[
                        "contextual_fusion_gripper_conflict_rate"
                    ],
                    "contextual_fusion_expert_gate_rate": timing[
                        "contextual_fusion_expert_gate_rate"
                    ],
                    "contextual_fusion_high_disagreement_source_expert": timing[
                        "contextual_fusion_high_disagreement_source_expert"
                    ],
                }
            elif args.contextual_fusion_mode in {
                "phase_compiler_expert",
                "phase_expert_compiler",
            }:
                returned_step = int(timing["contextual_fusion_absolute_decision_step"])
                returned_switch = int(timing["contextual_fusion_switch_step"])
                if returned_step != step or returned_switch != args.contextual_fusion_switch_step:
                    raise ValueError(
                        "Phase fusion response does not match requested step/switch: "
                        f"got {returned_step}/{returned_switch}, requested "
                        f"{step}/{args.contextual_fusion_switch_step}."
                    )
                contextual_fusion_info = {
                    "contextual_fusion_phase_selected_expert": timing[
                        "contextual_fusion_phase_selected_expert"
                    ],
                    "contextual_fusion_absolute_decision_step": returned_step,
                    "contextual_fusion_switch_step": returned_switch,
                }
            pact_flow_info: dict[str, Any] = {}
            if args.pact_flow_scheduler:
                pact_flow_info = {
                    "pact_flow_tau_json": timing["pact_flow_tau_json"],
                    "pact_flow_tau_logits_json": timing[
                        "pact_flow_tau_logits_json"
                    ],
                    "pact_flow_tau_mean": timing["pact_flow_tau_mean"],
                    "pact_flow_tau_std": timing["pact_flow_tau_std"],
                    "pact_flow_tau_min": timing["pact_flow_tau_min"],
                    "pact_flow_tau_max": timing["pact_flow_tau_max"],
                    # This synchronized final-stage timing already contains the
                    # scheduler and its one final expert call. No scheduler-only
                    # online dispatch or device barrier is introduced.
                    "action_expert_ms": timing["action_expert_ms"],
                }
            decisions.append(
                {
                    "mode": mode,
                    "task_id": task_id,
                    "episode": episode,
                    "initial_state_id": state_id,
                    "environment_step": step,
                    "selected_horizon": selected_horizon,
                    "execution_horizon": horizon,
                    "wall_ms": timing["wall_ms"],
                    "policy_ms": timing["policy_ms"],
                    "server_ms": timing["server_ms"],
                    "predictor_ms": timing["predictor_ms"],
                    "batched_teacher_ms": timing["batched_teacher_ms"],
                    "harp_residual_ms": timing["harp_residual_ms"],
                    "harp_gripper_event_ms": timing["harp_gripper_event_ms"],
                    "selector_json": json.dumps(selector_info, separators=(",", ":")),
                    **compact_router_info,
                    **selective_gripper_info,
                    **contextual_fusion_info,
                    **pact_flow_info,
                }
            )
            previous_actions = action_chunk
            previous_horizon = horizon
            for action in action_chunk[:horizon]:
                try:
                    observation, _, done, _ = env.step(np.asarray(action).tolist())
                except Exception as exc:
                    if not libero_eval._is_terminated_episode_error(exc):
                        raise
                    done = libero_eval._env_success(env)
                step += 1
                if done or libero_eval._env_success(env):
                    success = True
                    break
    finally:
        libero_eval._safe_close_env(env)
    episode_elapsed_ms = (time.perf_counter() - episode_started) * 1000.0

    def total(field: str) -> float:
        values = [timing[field] for timing in timings if np.isfinite(timing[field])]
        return float(np.sum(values)) if values else float("nan")

    def mean_timing(field: str) -> float:
        values = [timing[field] for timing in timings if np.isfinite(timing[field])]
        return float(np.mean(values)) if values else float("nan")

    histogram = collections.Counter(horizons)
    row = {
        "mode": mode,
        "task_suite": args.task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "episode": episode,
        "initial_state_id": state_id,
        "success": int(success),
        "timeout": int(not success),
        "steps": step,
        "policy_calls": policy_calls,
        "sampled_action_chunks": sampled_chunks,
        "avg_h": float(np.mean(horizons)) if horizons else float("nan"),
        "h_distribution_json": json.dumps(dict(sorted(histogram.items()))),
        "actual_wall_total_ms": total("wall_ms"),
        # Explicit alias for the legacy ``actual_wall_total_ms`` name.  This is
        # client-observed policy-RPC wait, not full simulator episode time.
        "policy_rpc_wall_total_ms": total("wall_ms"),
        "actual_episode_elapsed_total_ms": episode_elapsed_ms,
        "actual_policy_total_ms": total("policy_ms"),
        "actual_server_total_ms": total("server_ms"),
        "actual_predictor_total_ms": total("predictor_ms"),
        "actual_batched_teacher_total_ms": total("batched_teacher_ms"),
        "actual_contextual_fusion_total_ms": total("contextual_fusion_ms"),
        "contextual_fusion_translation_disagreement_mean": mean_timing(
            "contextual_fusion_translation_disagreement_mean"
        ),
        "contextual_fusion_rotation_disagreement_mean": mean_timing(
            "contextual_fusion_rotation_disagreement_mean"
        ),
        "contextual_fusion_gripper_conflict_rate": mean_timing(
            "contextual_fusion_gripper_conflict_rate"
        ),
        "contextual_fusion_expert_gate_rate": mean_timing(
            "contextual_fusion_expert_gate_rate"
        ),
        "contextual_fusion_phase_expert_rate": mean_timing(
            "contextual_fusion_phase_selected_expert"
        ),
        "actual_harp_residual_total_ms": total("harp_residual_ms"),
        "actual_harp_gripper_event_total_ms": total("harp_gripper_event_ms"),
    }
    if args.pact_flow_scheduler:
        row.update(
            {
                "actual_action_expert_total_ms": total("action_expert_ms"),
                "pact_flow_tau_mean": mean_timing("pact_flow_tau_mean"),
                "pact_flow_tau_std": mean_timing("pact_flow_tau_std"),
                "pact_flow_tau_min": mean_timing("pact_flow_tau_min"),
                "pact_flow_tau_max": mean_timing("pact_flow_tau_max"),
            }
        )
    if args.compact_alpha_router:
        alpha_histogram = collections.Counter(
            f"{alpha:.2f}" for alpha in compact_alpha_router_alphas
        )
        row.update(
            {
                "compact_alpha_router_decisions": len(compact_alpha_router_scores),
                "compact_alpha_router_score_sum": float(np.sum(compact_alpha_router_scores)),
                "compact_alpha_router_score_mean": (
                    float(np.mean(compact_alpha_router_scores))
                    if compact_alpha_router_scores
                    else float("nan")
                ),
                "compact_alpha_router_alpha_distribution_json": json.dumps(
                    dict(sorted(alpha_histogram.items()))
                ),
                "actual_compact_alpha_router_total_ms": total(
                    "compact_alpha_router_ms"
                ),
            }
        )
    if args.selective_gripper_refinement:
        row.update(
            {
                "selective_gripper_decisions": selective_gripper_decisions,
                "selective_gripper_triggers": selective_gripper_triggers,
                "selective_gripper_trigger_rate": (
                    selective_gripper_triggers / selective_gripper_decisions
                    if selective_gripper_decisions
                    else float("nan")
                ),
                "actual_selective_gripper_verifier_total_ms": total(
                    "selective_gripper_verifier_ms"
                ),
                "actual_selective_gripper_refinement_total_ms": total(
                    "selective_gripper_refinement_ms"
                ),
            }
        )
    return row, decisions


def _aggregate(rows: list[dict[str, Any]], mode: str, task_id: int | None = None) -> dict[str, Any]:
    subset = [row for row in rows if row["mode"] == mode and (task_id is None or row["task_id"] == task_id)]
    all_horizons: list[int] = []
    for row in subset:
        histogram = json.loads(row["h_distribution_json"])
        for horizon, count in histogram.items():
            all_horizons.extend([int(horizon)] * int(count))

    def mean(field: str) -> float:
        values = [float(row.get(field, float("nan"))) for row in subset]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    def finite_count(field: str) -> int:
        return int(sum(np.isfinite(float(row.get(field, float("nan")))) for row in subset))

    def outcome_mean(field: str, success_value: int) -> float:
        values = [float(row.get(field, float("nan"))) for row in subset if int(row["success"]) == success_value]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    def per_call(field: str) -> float:
        pairs = [(float(row.get(field, float("nan"))), int(row["policy_calls"])) for row in subset]
        pairs = [(value, calls) for value, calls in pairs if np.isfinite(value) and calls > 0]
        if not pairs:
            return float("nan")
        return float(sum(value for value, _ in pairs) / sum(calls for _, calls in pairs))

    histogram = collections.Counter(all_horizons)
    result = {
        "mode": mode,
        "task_id": task_id if task_id is not None else "overall",
        "episodes": len(subset),
        "success_rate": mean("success"),
        "timeout_rate": mean("timeout"),
        "calls_per_episode": mean("policy_calls"),
        "successful_calls_per_episode": outcome_mean("policy_calls", 1),
        "timeout_calls_per_episode": outcome_mean("policy_calls", 0),
        "sampled_action_chunks_per_episode": mean("sampled_action_chunks"),
        "avg_h": float(np.mean(all_horizons)) if all_horizons else float("nan"),
        "h_distribution": dict(sorted(histogram.items())),
        "actual_wall_ms_per_episode": mean("actual_wall_total_ms"),
        "policy_rpc_wall_ms_per_episode": mean("policy_rpc_wall_total_ms"),
        "policy_rpc_wall_ms_per_call": per_call("policy_rpc_wall_total_ms"),
        "successful_policy_rpc_wall_ms_per_episode": outcome_mean("policy_rpc_wall_total_ms", 1),
        "timeout_policy_rpc_wall_ms_per_episode": outcome_mean("policy_rpc_wall_total_ms", 0),
        "actual_episode_elapsed_ms_per_episode": mean("actual_episode_elapsed_total_ms"),
        "actual_episode_elapsed_episodes": finite_count("actual_episode_elapsed_total_ms"),
        "successful_episode_elapsed_ms_per_episode": outcome_mean("actual_episode_elapsed_total_ms", 1),
        "timeout_episode_elapsed_ms_per_episode": outcome_mean("actual_episode_elapsed_total_ms", 0),
        "actual_policy_ms_per_episode": mean("actual_policy_total_ms"),
        "policy_ms_per_call": per_call("actual_policy_total_ms"),
        "successful_policy_ms_per_episode": outcome_mean("actual_policy_total_ms", 1),
        "timeout_policy_ms_per_episode": outcome_mean("actual_policy_total_ms", 0),
        "actual_server_ms_per_episode": mean("actual_server_total_ms"),
        "server_ms_per_call": per_call("actual_server_total_ms"),
        "predictor_ms_per_episode": mean("actual_predictor_total_ms"),
        "predictor_ms_per_call": per_call("actual_predictor_total_ms"),
        "batched_teacher_ms_per_episode": mean("actual_batched_teacher_total_ms"),
        "contextual_fusion_ms_per_episode": mean(
            "actual_contextual_fusion_total_ms"
        ),
        "contextual_fusion_ms_per_call": per_call(
            "actual_contextual_fusion_total_ms"
        ),
        "contextual_fusion_translation_disagreement_mean": mean(
            "contextual_fusion_translation_disagreement_mean"
        ),
        "contextual_fusion_rotation_disagreement_mean": mean(
            "contextual_fusion_rotation_disagreement_mean"
        ),
        "contextual_fusion_gripper_conflict_rate": mean(
            "contextual_fusion_gripper_conflict_rate"
        ),
        "contextual_fusion_expert_gate_rate": mean(
            "contextual_fusion_expert_gate_rate"
        ),
        "contextual_fusion_phase_expert_rate": mean(
            "contextual_fusion_phase_expert_rate"
        ),
        "harp_residual_ms_per_episode": mean("actual_harp_residual_total_ms"),
        "harp_residual_ms_per_call": per_call("actual_harp_residual_total_ms"),
        "harp_gripper_event_ms_per_episode": mean(
            "actual_harp_gripper_event_total_ms"
        ),
        "harp_gripper_event_ms_per_call": per_call(
            "actual_harp_gripper_event_total_ms"
        ),
    }
    if any("actual_action_expert_total_ms" in row for row in subset):
        result.update(
            {
                "action_expert_ms_per_episode": mean(
                    "actual_action_expert_total_ms"
                ),
                "action_expert_ms_per_call": per_call(
                    "actual_action_expert_total_ms"
                ),
                "pact_flow_tau_mean": mean("pact_flow_tau_mean"),
                "pact_flow_tau_std": mean("pact_flow_tau_std"),
                "pact_flow_tau_min": mean("pact_flow_tau_min"),
                "pact_flow_tau_max": mean("pact_flow_tau_max"),
            }
        )
    if any("compact_alpha_router_alpha_distribution_json" in row for row in subset):
        alpha_histogram: collections.Counter[str] = collections.Counter()
        score_sum = 0.0
        score_count = 0
        for row in subset:
            alpha_histogram.update(
                {
                    str(alpha): int(count)
                    for alpha, count in json.loads(
                        row.get("compact_alpha_router_alpha_distribution_json", "{}")
                    ).items()
                }
            )
            count = int(row.get("compact_alpha_router_decisions", 0))
            score_sum += float(row.get("compact_alpha_router_score_sum", 0.0))
            score_count += count
        result.update(
            {
                "compact_alpha_router_decisions": score_count,
                "compact_alpha_router_score_mean": (
                    score_sum / score_count if score_count else float("nan")
                ),
                "compact_alpha_router_alpha_distribution": dict(
                    sorted(alpha_histogram.items())
                ),
                "compact_alpha_router_ms_per_episode": mean(
                    "actual_compact_alpha_router_total_ms"
                ),
                "compact_alpha_router_ms_per_call": per_call(
                    "actual_compact_alpha_router_total_ms"
                ),
            }
        )
    if any("selective_gripper_decisions" in row for row in subset):
        selective_decisions = sum(
            int(row.get("selective_gripper_decisions", 0)) for row in subset
        )
        selective_triggers = sum(
            int(row.get("selective_gripper_triggers", 0)) for row in subset
        )
        result.update(
            {
                "selective_gripper_decisions": selective_decisions,
                "selective_gripper_triggers": selective_triggers,
                "selective_gripper_trigger_rate": (
                    selective_triggers / selective_decisions
                    if selective_decisions
                    else float("nan")
                ),
                "selective_gripper_verifier_ms_per_episode": mean(
                    "actual_selective_gripper_verifier_total_ms"
                ),
                "selective_gripper_verifier_ms_per_call": per_call(
                    "actual_selective_gripper_verifier_total_ms"
                ),
                "selective_gripper_refinement_ms_per_episode": mean(
                    "actual_selective_gripper_refinement_total_ms"
                ),
                "selective_gripper_refinement_ms_per_call": per_call(
                    "actual_selective_gripper_refinement_total_ms"
                ),
            }
        )
    return result


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _coerce_rollout_row(row: dict[str, str]) -> dict[str, Any]:
    integers = {
        "task_id",
        "episode",
        "initial_state_id",
        "success",
        "timeout",
        "steps",
        "policy_calls",
        "sampled_action_chunks",
        "compact_alpha_router_decisions",
        "selective_gripper_decisions",
        "selective_gripper_triggers",
    }
    floats = {
        "avg_h",
        "actual_wall_total_ms",
        "policy_rpc_wall_total_ms",
        "actual_episode_elapsed_total_ms",
        "actual_policy_total_ms",
        "actual_server_total_ms",
        "actual_predictor_total_ms",
        "actual_batched_teacher_total_ms",
        "actual_action_expert_total_ms",
        "pact_flow_tau_mean",
        "pact_flow_tau_std",
        "pact_flow_tau_min",
        "pact_flow_tau_max",
        "actual_contextual_fusion_total_ms",
        "contextual_fusion_translation_disagreement_mean",
        "contextual_fusion_rotation_disagreement_mean",
        "contextual_fusion_gripper_conflict_rate",
        "contextual_fusion_expert_gate_rate",
        "contextual_fusion_phase_expert_rate",
        "actual_harp_residual_total_ms",
        "actual_harp_gripper_event_total_ms",
        "compact_alpha_router_score_sum",
        "compact_alpha_router_score_mean",
        "actual_compact_alpha_router_total_ms",
        "selective_gripper_trigger_rate",
        "actual_selective_gripper_verifier_total_ms",
        "actual_selective_gripper_refinement_total_ms",
    }
    converted = {
        key: int(value) if key in integers else float(value) if key in floats else value for key, value in row.items()
    }
    converted.setdefault("policy_rpc_wall_total_ms", converted.get("actual_wall_total_ms", float("nan")))
    converted.setdefault("actual_episode_elapsed_total_ms", float("nan"))
    return converted


def _run_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key not in {"output_dir", "resume"}}


def _prepare_journal(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]]]:
    rollout_path = output_dir / "rollout_rows.csv"
    decisions_path = output_dir / "decisions.csv"
    signature_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    signature = _run_signature(args)

    if summary_path.exists():
        if args.resume:
            print(summary_path.read_text(), flush=True)
            return [], set()
        raise FileExistsError(f"Evaluation is already complete: {summary_path}")

    if not args.resume:
        existing = [path for path in (rollout_path, decisions_path, signature_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing evaluation journal: {existing}")
        signature_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
        return [], set()

    if not signature_path.exists():
        raise FileNotFoundError(f"Cannot safely resume without the configuration signature: {signature_path}")
    saved_signature = json.loads(signature_path.read_text())
    if saved_signature != signature:
        raise ValueError(
            "Resume configuration differs from the saved evaluation configuration. "
            f"saved={saved_signature}, requested={signature}"
        )

    rows = [_coerce_rollout_row(row) for row in _read_csv(rollout_path)]
    completed: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (str(row["mode"]), int(row["task_id"]), int(row["episode"]))
        if key in completed:
            raise ValueError(f"Duplicate completed rollout row while resuming: {key}")
        completed.add(key)

    # Decisions are journaled before their rollout completion row.  If the
    # process died in that small window, discard the incomplete episode's
    # decisions.  Also deduplicate a previously interrupted append.
    cleaned_decisions: list[dict[str, Any]] = []
    seen_decisions: set[tuple[str, int, int, int]] = set()
    for row in _read_csv(decisions_path):
        episode_key = (str(row["mode"]), int(row["task_id"]), int(row["episode"]))
        decision_key = (*episode_key, int(row["environment_step"]))
        if episode_key in completed and decision_key not in seen_decisions:
            cleaned_decisions.append(row)
            seen_decisions.add(decision_key)
    decisions_path.unlink(missing_ok=True)
    _write_csv(decisions_path, cleaned_decisions)
    return rows, completed


def main(args: argparse.Namespace) -> None:
    if args.action_cot_denoising_steps <= 0:
        raise ValueError("action_cot_denoising_steps must be positive.")
    if args.contextual_fusion_switch_step < 0:
        raise ValueError("contextual_fusion_switch_step must be non-negative.")
    for name in (
        "contextual_fusion_translation_tau",
        "contextual_fusion_rotation_tau",
        "contextual_fusion_gripper_tau",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if (
        not np.isfinite(args.contextual_fusion_gate_width)
        or args.contextual_fusion_gate_width <= 0.0
    ):
        raise ValueError("contextual_fusion_gate_width must be finite and positive.")
    if args.final_denoising_steps is not None and args.final_denoising_steps <= 0:
        raise ValueError("final_denoising_steps must be positive when set.")
    if not 0.0 <= args.final_time_warp_alpha < 1.0:
        raise ValueError("final_time_warp_alpha must be in [0, 1).")
    if args.final_token_time_warp_alpha is not None:
        token_alpha = np.asarray(args.final_token_time_warp_alpha, dtype=np.float64)
        if token_alpha.shape != (10,):
            raise ValueError("final_token_time_warp_alpha must contain exactly 10 values.")
        if not np.all(np.isfinite(token_alpha)) or np.any(token_alpha < 0.0) or np.any(token_alpha >= 1.0):
            raise ValueError("Every final_token_time_warp_alpha value must be finite and in [0, 1).")
        incompatible_token_time_warp = {
            "pact_flow_scheduler": args.pact_flow_scheduler,
            "final_time_warp_alpha": args.final_time_warp_alpha > 0.0,
            "final_denoising_steps": args.final_denoising_steps is not None,
            "final_endpoint_condition_strength": args.final_endpoint_condition_strength > 0.0,
            "final_midpoint": args.final_midpoint,
            "adaptive_final_time_warp": args.adaptive_final_time_warp,
            "ofp_interval_flow": args.ofp_interval_flow,
            "harp_residual": args.harp_residual,
            "harp_gripper_event": args.harp_gripper_event,
            "final_hybrid_mode": args.final_hybrid_mode != "none",
            "selective_gripper_refinement": args.selective_gripper_refinement,
            "compact_alpha_router": args.compact_alpha_router,
            "exact_batched_mc_v2": "exact_batched_mc_v2" in args.modes,
        }
        conflicts = sorted(
            name for name, enabled in incompatible_token_time_warp.items() if enabled
        )
        if conflicts:
            raise ValueError(
                "final_token_time_warp_alpha owns direct final one-step timing and "
                f"cannot be combined with: {', '.join(conflicts)}."
            )
    if args.pact_flow_scheduler:
        incompatible_pact_modes = {
            "final_token_time_warp_alpha": (
                args.final_token_time_warp_alpha is not None
            ),
            "final_time_warp_alpha": args.final_time_warp_alpha != 0.0,
            "final_denoising_steps": args.final_denoising_steps is not None,
            "final_endpoint_condition_strength": (
                args.final_endpoint_condition_strength != 0.0
            ),
            "final_midpoint": args.final_midpoint,
            "adaptive_final_time_warp": args.adaptive_final_time_warp,
            "compact_alpha_router": args.compact_alpha_router,
            "harp_residual": args.harp_residual,
            "harp_gripper_event": args.harp_gripper_event,
            "final_hybrid_mode": args.final_hybrid_mode != "none",
            "selective_gripper_refinement": args.selective_gripper_refinement,
            "ofp_interval_flow": args.ofp_interval_flow,
            "contextual_fusion": args.contextual_fusion_mode != "compiler",
            "exact_batched_mc_v2": "exact_batched_mc_v2" in args.modes,
        }
        pact_conflicts = sorted(
            name for name, enabled in incompatible_pact_modes.items() if enabled
        )
        if pact_conflicts:
            raise ValueError(
                "pact_flow_scheduler is a standalone one-call final mode and cannot "
                f"be combined with: {', '.join(pact_conflicts)}."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError(
                "pact_flow_scheduler requires endpoint-student EAR NFE1."
            )
        if list(args.modes) != ["fixed_h9"] or args.fixed_horizon != 10:
            raise ValueError(
                "Formal PACT evaluation requires --modes fixed_h9 --fixed-horizon 10."
            )
    if not 0.0 <= args.final_endpoint_condition_strength <= 1.0:
        raise ValueError("final_endpoint_condition_strength must be in [0, 1].")
    if args.final_endpoint_condition_strength > 0.0:
        incompatible_endpoint_modes = (
            args.final_time_warp_alpha > 0.0
            or args.final_denoising_steps is not None
            or args.adaptive_final_time_warp
            or args.compact_alpha_router
            or args.harp_residual
            or args.harp_gripper_event
            or args.final_hybrid_mode != "none"
            or args.selective_gripper_refinement
            or args.ofp_interval_flow
            or "exact_batched_mc_v2" in args.modes
        )
        if incompatible_endpoint_modes:
            raise ValueError(
                "final_endpoint_condition_strength is a standalone one-step final mode."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError(
                "final_endpoint_condition_strength requires endpoint-student EAR NFE1."
            )
    if args.final_midpoint:
        incompatible_midpoint_modes = (
            args.final_denoising_steps is not None
            or args.final_endpoint_condition_strength > 0.0
            or args.adaptive_final_time_warp
            or args.compact_alpha_router
            or args.harp_residual
            or args.harp_gripper_event
            or args.final_hybrid_mode != "none"
            or args.selective_gripper_refinement
            or args.ofp_interval_flow
            or "exact_batched_mc_v2" in args.modes
        )
        if incompatible_midpoint_modes:
            raise ValueError("final_midpoint is a standalone two-NFE final mode.")
        if args.action_cot_denoising_steps != 1:
            raise ValueError("final_midpoint requires endpoint-student EAR NFE1.")
    if not np.isfinite(args.selective_gripper_tau) or not 0.0 <= args.selective_gripper_tau <= 1.0:
        raise ValueError("selective_gripper_tau must be finite and in [0, 1].")
    if args.final_time_warp_alpha > 0.0 and args.ofp_interval_flow:
        raise ValueError("final_time_warp_alpha and ofp_interval_flow are mutually exclusive.")
    if args.adaptive_final_time_warp and (
        args.final_time_warp_alpha > 0.0 or args.ofp_interval_flow
    ):
        raise ValueError(
            "adaptive_final_time_warp is mutually exclusive with fixed time warp and OFP interval flow."
        )
    if args.adaptive_final_time_warp and "exact_batched_mc_v2" in args.modes:
        raise ValueError(
            "adaptive_final_time_warp cannot be evaluated with exact_batched_mc_v2; "
            "pass --modes without that batched-teacher mode."
        )
    if args.final_denoising_steps is not None and args.ofp_interval_flow:
        raise ValueError("final_denoising_steps cannot be combined with OFP interval flow.")
    if args.final_denoising_steps is not None and args.adaptive_final_time_warp:
        raise ValueError("final_denoising_steps cannot be combined with adaptive final time warp.")
    if args.final_denoising_steps is not None and "exact_batched_mc_v2" in args.modes:
        raise ValueError(
            "final_denoising_steps cannot be evaluated with exact_batched_mc_v2; "
            "pass --modes without that batched-teacher mode."
        )
    if args.final_hybrid_mode != "none":
        if args.final_denoising_steps is not None:
            raise ValueError(
                "final_hybrid_mode owns both final NFE1 and NFE2; do not set final_denoising_steps."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError("final_hybrid_mode requires endpoint-student EAR NFE1.")
        if (
            args.harp_residual
            or args.harp_gripper_event
            or args.selective_gripper_refinement
            or args.compact_alpha_router
            or args.adaptive_final_time_warp
            or args.ofp_interval_flow
        ):
            raise ValueError(
                "final_hybrid_mode cannot be combined with HARP, compact routing, "
                "selective gripper refinement, adaptive time warp, or OFP inference."
            )
        if list(args.modes) != ["fixed_h9"] or args.fixed_horizon != 10:
            raise ValueError(
                "The final hybrid ceiling diagnostic requires --modes fixed_h9 --fixed-horizon 10."
            )
        if (
            args.task_suite_name != "libero_10"
            or args.task_start != 8
            or args.max_tasks != 1
            or args.initial_state_offset != 0
            or args.episode_ids != [36, 37]
        ):
            raise ValueError(
                "The final hybrid ceiling diagnostic is restricted to libero_10 Task8 "
                "with initial-state offset 0 and --episode-ids 36 37."
            )
    if args.selective_gripper_refinement:
        if args.final_denoising_steps is not None:
            raise ValueError(
                "selective_gripper_refinement owns direct A1 and conditional NFE2; "
                "do not set final_denoising_steps."
            )
        if args.final_hybrid_mode != "none":
            raise ValueError(
                "selective_gripper_refinement cannot be combined with final_hybrid_mode."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError("selective_gripper_refinement requires endpoint-student EAR NFE1.")
        if (
            args.harp_residual
            or getattr(args, "harp_gripper_event", False)
            or args.compact_alpha_router
            or args.adaptive_final_time_warp
            or args.ofp_interval_flow
        ):
            raise ValueError(
                "selective_gripper_refinement cannot be combined with HARP, a gripper-event "
                "student, compact routing, adaptive time warp, or OFP inference."
            )
        if list(args.modes) != ["fixed_h9"] or args.fixed_horizon != 10:
            raise ValueError(
                "Selective gripper refinement requires --modes fixed_h9 --fixed-horizon 10."
            )
    elif args.selective_refinement_mode != "gripper":
        raise ValueError(
            "selective_refinement_mode is only meaningful with --selective-gripper-refinement."
        )
    if args.compact_alpha_router:
        if args.final_denoising_steps is not None:
            raise ValueError(
                "compact_alpha_router forces direct final NFE1; do not set final_denoising_steps."
            )
        if args.final_time_warp_alpha > 0.0:
            raise ValueError(
                "compact_alpha_router selects alpha itself; do not set final_time_warp_alpha."
            )
        if args.adaptive_final_time_warp or args.ofp_interval_flow:
            raise ValueError(
                "compact_alpha_router cannot be combined with adaptive time warp or OFP inference."
            )
        if list(args.modes) != ["fixed_h9"] or args.fixed_horizon != 10:
            raise ValueError(
                "The compact Task8 formal protocol requires --modes fixed_h9 --fixed-horizon 10."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError(
                "The compact router endpoint-student protocol requires --action-cot-denoising-steps 1."
            )
        expected_formal_episodes = list(range(10, 30))
        if (
            args.task_suite_name != "libero_10"
            or args.task_start != 8
            or args.max_tasks != 1
            or args.initial_state_offset != 0
            or args.episode_ids != expected_formal_episodes
        ):
            raise ValueError(
                "The compact outcome router is Task8-specific and its one-shot formal evaluation "
                "requires libero_10 Task8, initial-state offset 0, and explicit episode IDs 10..29."
            )
    if args.harp_residual:
        if args.final_denoising_steps is not None:
            raise ValueError("harp_residual forces direct final NFE1; do not set final_denoising_steps.")
        if args.action_cot_denoising_steps != 1:
            raise ValueError("harp_residual requires endpoint-student EAR NFE1.")
        if args.adaptive_final_time_warp or args.ofp_interval_flow:
            raise ValueError(
                "harp_residual cannot be combined with adaptive time warp or OFP inference."
            )
        if args.compact_alpha_router:
            raise ValueError("harp_residual cannot be combined with compact_alpha_router.")
        if "exact_batched_mc_v2" in args.modes:
            raise ValueError("harp_residual cannot be evaluated with exact_batched_mc_v2.")
        if args.harp_gripper_event:
            raise ValueError("Continuous HARP and ES-HARP gripper event are separate evaluations.")
    if args.harp_gripper_event:
        if args.final_denoising_steps is not None:
            raise ValueError(
                "harp_gripper_event forces direct final NFE1; do not set final_denoising_steps."
            )
        if args.action_cot_denoising_steps != 1:
            raise ValueError("harp_gripper_event requires endpoint-student EAR NFE1.")
        if abs(args.final_time_warp_alpha - 0.05) > 1e-7:
            raise ValueError("harp_gripper_event requires matched --final-time-warp-alpha 0.05.")
        if (
            args.final_hybrid_mode != "none"
            or args.selective_gripper_refinement
            or args.compact_alpha_router
            or args.adaptive_final_time_warp
            or args.ofp_interval_flow
        ):
            raise ValueError(
                "harp_gripper_event cannot be combined with hybrid/selective refinement, "
                "compact routing, adaptive time warp, or OFP."
            )
        if "exact_batched_mc_v2" in args.modes:
            raise ValueError("harp_gripper_event cannot be evaluated with exact_batched_mc_v2.")
    if not 0.0 < args.ofp_warm_start_time <= 1.0:
        raise ValueError("ofp_warm_start_time must be in (0, 1].")
    if not 0.0 <= args.ofp_interval_condition_strength <= 1.0:
        raise ValueError("ofp_interval_condition_strength must be in [0, 1].")
    if args.v2_budget_capacity <= 0 or args.v2_initial_budget > args.v2_budget_capacity:
        raise ValueError("Invalid V2 budget configuration.")
    if args.selector_temperature <= 0:
        raise ValueError("selector_temperature must be positive.")
    invalid_sample_modes = sorted(set(args.selector_sample_modes).difference(args.modes))
    if invalid_sample_modes:
        raise ValueError(f"selector_sample_modes were not requested in --modes: {invalid_sample_modes}")
    if args.episode_ids is not None and (not args.episode_ids or len(set(args.episode_ids)) != len(args.episode_ids)):
        raise ValueError("episode_ids must be non-empty and unique when provided.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, completed = _prepare_journal(output_dir, args)
    if (output_dir / "summary.json").exists():
        return
    selectors = _load_selectors(args)
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    _warmup(client, task_suite, args)
    task_end = min(task_suite.n_tasks, args.task_start + args.max_tasks)
    episode_ids = args.episode_ids if args.episode_ids is not None else list(range(args.num_trials_per_task))
    for mode in args.modes:
        for task_id in range(args.task_start, task_end):
            for episode in episode_ids:
                episode_key = (mode, task_id, episode)
                if episode_key in completed:
                    continue
                row, episode_decisions = _run_episode(
                    mode=mode,
                    task_id=task_id,
                    episode=episode,
                    task_suite=task_suite,
                    client=client,
                    args=args,
                    selector=selectors.get(mode),
                )
                _append_csv(output_dir / "decisions.csv", episode_decisions)
                _append_csv(output_dir / "rollout_rows.csv", [row])
                rows.append(row)
                completed.add(episode_key)
                print(json.dumps(row, sort_keys=True), flush=True)

    per_task = [_aggregate(rows, mode, task_id) for mode in args.modes for task_id in range(args.task_start, task_end)]
    overall = {mode: _aggregate(rows, mode) for mode in args.modes}
    flat_per_task = []
    for item in per_task:
        flat_item = {
            **item,
            "h_distribution": json.dumps(item["h_distribution"], sort_keys=True),
        }
        if "compact_alpha_router_alpha_distribution" in flat_item:
            flat_item["compact_alpha_router_alpha_distribution"] = json.dumps(
                flat_item["compact_alpha_router_alpha_distribution"],
                sort_keys=True,
            )
        flat_per_task.append(flat_item)
    _write_csv(output_dir / "per_task_summary.csv", flat_per_task)
    summary = {
        "status": "complete",
        "paired_initial_states": True,
        "task_suite": args.task_suite_name,
        "num_tasks": task_end - args.task_start,
        "num_trials_per_task": len(episode_ids),
        "episode_ids": episode_ids,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "teacher_samples": args.teacher_samples,
        "timing_semantics": (
            "actual synchronized policy/server/client-RPC-wall totals plus full episode elapsed time; "
            "predictor and batched sampling are included"
        ),
        "config": vars(args),
        "overall": overall,
        "per_task": per_task,
        "outputs": {
            "rollout_rows": str(output_dir / "rollout_rows.csv"),
            "decisions": str(output_dir / "decisions.csv"),
            "per_task_summary": str(output_dir / "per_task_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
