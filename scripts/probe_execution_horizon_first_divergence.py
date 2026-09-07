"""Probe a single H change at reproduced A/history divergence states."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import copy
import json
import pathlib
from typing import Any

import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
import eval_libero_execution_horizon as evaluator
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy
import replay_execution_horizon_branches as replay

from openpi.execution_horizon.feedback import FeedbackSelector
from openpi.execution_horizon.initial_states import InitialStateBank

BASE_MODE = "ordered_transformer"
HISTORY_MODE = "ordered_feedback_history"
COMPARISON_ATOL = 1e-5
RESTORE_ATOL = 1e-8
REPEAT_SEED_STRIDE = 20_000_000
CONTINUATION_SEED_OFFSET = 100_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=pathlib.Path, required=True)
    parser.add_argument("--analysis-summary", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-config", type=pathlib.Path, required=True)
    parser.add_argument("--history-params", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def load_analysis_cases(path: pathlib.Path) -> list[dict[str, Any]]:
    if path.is_file():
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            raise ValueError("First-divergence probes require a completed passive analysis.")
        directory = path.parent / "episodes"
    else:
        directory = path / "episodes" if (path / "episodes").is_dir() else path
    cases = [json.loads(item.read_text()) for item in sorted(directory.glob("task*_ep*.json"))]
    if not cases or len(cases) > 9:
        raise ValueError("This diagnostic is bounded to the one-to-nine previously discordant cases.")
    identities = [(int(case["task_id"]), int(case["episode_id"])) for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Passive analysis contains duplicate task/episode cases.")
    return cases


def eligibility_reasons(case: dict[str, Any]) -> list[str]:
    reasons = []
    if case["reproduction"]["original_pair_reproduced"] is not True:
        reasons.append("original_pair_not_reproduced")
    alignment = case["alignment"]
    first = alignment["first_shared_decision_with_h_difference"]
    if first is None:
        reasons.append("no_shared_decision_with_h_difference")
        return reasons
    if int(alignment["common_prefix_through_step"]) < int(first["step"]):
        reasons.append("decision_grids_diverged_before_the_selected_root")
    prefix = alignment["before_first_h_or_grid_difference"]
    if not prefix["shared_decisions"] or prefix["matching_request_seeds"] != prefix["shared_decisions"]:
        reasons.append("pre_divergence_policy_seeds_do_not_match")
    for key in (
        "generated_chunk_max_abs_difference", "decision_physics_max_abs_difference",
        "decision_raw_proprio_max_abs_difference", "executed_command_max_abs_difference",
        "eef_position_difference_m", "eef_rotation_difference_rad", "gripper_qpos_max_abs_difference",
    ):
        value = prefix[key]["max"]
        if value is None or not np.isfinite(value) or value > COMPARISON_ATOL:
            reasons.append(f"pre_divergence_{key}_exceeds_tolerance_or_is_missing")
    return reasons


def _baseline_args(path: pathlib.Path, output: pathlib.Path) -> argparse.Namespace:
    saved = json.loads(path.read_text())
    args = evaluator.build_parser().parse_args(["--output-dir", str(output)])
    for name, value in saved.items():
        setattr(args, name, value)
    if (
        args.model_action_horizon != 25
        or args.action_cot_denoising_steps != 10
        or args.final_denoising_steps != 10
        or args.initial_state_bank is None
    ):
        raise ValueError("The saved protocol must use H25, NFE10/10 and an explicit initial-state bank.")
    args.output_dir = str(output)
    args.trace_output_dir = None
    args.resume = False
    args.modes = [HISTORY_MODE]
    return args


def _trace_root(case: dict[str, Any], directory: pathlib.Path) -> dict[str, Any]:
    task, episode = int(case["task_id"]), int(case["episode_id"])
    step = int(case["alignment"]["first_shared_decision_with_h_difference"]["step"])
    filename = f"task{task:02d}_ep{episode:06d}.npz"
    traces = {}
    for mode in (BASE_MODE, HISTORY_MODE):
        path = directory / mode / filename
        with np.load(path, allow_pickle=False) as archive:
            traces[mode] = {name: archive[name] for name in (
                "decision_steps", "selected_h", "generated_action_chunks", "decision_physics_state",
                "previous_h", "episode_progress", "request_seeds",
            )}
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["task_id"] != task or metadata["episode_id"] != episode or metadata["mode"] != mode:
            raise ValueError("Trace identity differs from its passive-analysis case.")
        if metadata["success"] != case["reproduction"]["trace_success"][mode] or metadata.get("error") is not None:
            raise ValueError("Trace outcome differs from the completed passive analysis.")
    indices = {}
    for mode, arrays in traces.items():
        selected = np.flatnonzero(arrays["decision_steps"] == step)
        if len(selected) != 1:
            raise ValueError("The first differing decision must appear exactly once in each trace.")
        indices[mode] = int(selected[0])
    a, history = traces[BASE_MODE], traces[HISTORY_MODE]
    ai, hi = indices[BASE_MODE], indices[HISTORY_MODE]
    saved_actions = np.asarray(history["generated_action_chunks"][hi], dtype=np.float32)
    if saved_actions.shape != (25, 7):
        raise ValueError("A saved root action chunk must have shape (25, 7).")
    first = case["alignment"]["first_shared_decision_with_h_difference"]
    if (
        int(a["selected_h"][ai]) != int(first["a_selected_h"])
        or int(history["selected_h"][hi]) != int(first["history_selected_h"])
        or int(a["request_seeds"][ai]) != int(history["request_seeds"][hi])
        or not np.allclose(a["generated_action_chunks"][ai], saved_actions, rtol=0, atol=COMPARISON_ATOL)
        or not np.allclose(
            a["decision_physics_state"][ai], history["decision_physics_state"][hi], rtol=0, atol=COMPARISON_ATOL
        )
    ):
        raise ValueError("The saved traces do not retain the agreed shared root/action/seed/H values.")
    return {
        "task_id": task, "episode_id": episode, "step": step,
        "physics_state": np.asarray(history["decision_physics_state"][hi], dtype=np.float64).copy(),
        "saved_actions": saved_actions.copy(),
        "previous_actions": None if hi == 0 else np.asarray(history["generated_action_chunks"][hi - 1]).copy(),
        "previous_h": int(history["previous_h"][hi]), "episode_progress": float(history["episode_progress"][hi]),
        "root_request_seed": int(history["request_seeds"][hi]),
        "history_h": int(history["selected_h"][hi]), "a_h": int(a["selected_h"][ai]),
    }


def _continuation_seed(root_seed: int, repeat: int, call_index: int) -> int:
    seed = root_seed + CONTINUATION_SEED_OFFSET + repeat * REPEAT_SEED_STRIDE + call_index
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("Paired continuation seed falls outside uint32.")
    return seed


def _budget_fraction(args: argparse.Namespace) -> float:
    return min(args.v2_initial_budget, args.v2_budget_capacity) / args.v2_budget_capacity


def _run_forced_branch(
    *, env: Any, snapshot: collector.SimulatorSnapshot, root: dict[str, Any],
    forced_h: int, repeat: int, episode_step_limit: int, task_description: str,
    root_observation_cache: dict[str, Any], client: Any, selector: FeedbackSelector,
    args: argparse.Namespace,
) -> dict[str, Any]:
    observation = collector._restore_snapshot(env, snapshot)
    step = root["step"]
    initial_step = step
    previous_cache = copy.deepcopy(root_observation_cache)
    action_chunk = root["saved_actions"].copy()
    selected_h = forced_h
    calls = 1
    success = False
    first_actual_h = 0
    seeds = []
    selected_horizons = [forced_h]
    while step < episode_step_limit:
        decision_step = step
        for action in action_chunk[:min(selected_h, episode_step_limit - step)]:
            try:
                observation, _, done, _ = env.step(np.asarray(action).tolist())
            except Exception as exc:
                if not libero_eval._is_terminated_episode_error(exc):
                    raise
                done = libero_eval._env_success(env)
            step += 1
            if calls == 1:
                first_actual_h += 1
            if done or libero_eval._env_success(env):
                success = True
                break
        if success or step >= episode_step_limit:
            break
        actual_h = step - decision_step
        seed = _continuation_seed(root["root_request_seed"], repeat, len(seeds))
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        result, _ = evaluator._request(
            client, policy_input, mode=HISTORY_MODE, seed=seed,
            previous_actions=action_chunk, previous_horizon=actual_h,
            budget_fraction=_budget_fraction(args), episode_progress=step / episode_step_limit,
            absolute_decision_step=step, args=args,
        )
        inputs, previous_cache = evaluator._feedback_inputs(result, previous_cache, step=step, previous_h=actual_h)
        selected_h, _ = selector.decide(inputs)
        action_chunk = np.asarray(result["actions"], dtype=np.float32)
        if action_chunk.shape != (25, 7) or selected_h not in (5, 10, 15, 20, 25):
            raise ValueError("History continuation returned an unsupported action chunk or H.")
        selected_horizons.append(selected_h)
        seeds.append(seed)
        calls += 1
    return {
        "repeat": repeat, "forced_h": forced_h, "actual_first_h": first_actual_h,
        "success": bool(success), "steps": step - initial_step, "final_step": step,
        "calls": calls, "continuation_calls": len(seeds), "continuation_seeds": seeds,
        "selected_horizons": selected_horizons,
    }


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def probe_case(
    case: dict[str, Any], *, trace_dir: pathlib.Path, task_suite: Any, bank: InitialStateBank,
    selector: FeedbackSelector, client: Any, args: argparse.Namespace, repeats: int, output: pathlib.Path,
) -> dict[str, Any]:
    root = _trace_root(case, trace_dir)
    result: dict[str, Any] = {
        "task_id": root["task_id"], "episode_id": root["episode_id"], "root_step": root["step"],
        "old_relation": case["old_relation"], "history_h": root["history_h"], "a_h": root["a_h"],
        "root_request_seed": root["root_request_seed"], "status": "preparing", "branches": [],
    }
    task = task_suite.get_task(root["task_id"])
    bank.validate_presets(root["task_id"], task_suite.get_task_init_states(root["task_id"]))
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        env.set_init_state(bank.state(root["task_id"], root["episode_id"]))
        for _ in range(args.num_steps_wait):
            _, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            if done:
                break
        snapshot = replay._saved_snapshot(env, root["physics_state"], root["step"])
        observation = collector._restore_snapshot(env, snapshot)
        restored = np.asarray(collector._simulator(env).get_state().flatten(), dtype=np.float64)
        restore_difference = float(np.max(np.abs(restored - root["physics_state"])))
        result["restored_physics_max_abs_difference"] = restore_difference
        if not np.isfinite(restore_difference) or restore_difference > RESTORE_ATOL:
            result.update(status="skipped", skip_reason="restored_physics_differs_from_saved_root")
            return result
        limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        environment_horizon = libero_eval._env_horizon(env)
        if environment_horizon is not None:
            limit = min(limit, environment_horizon)
        if min(root["history_h"], limit - root["step"]) == min(root["a_h"], limit - root["step"]):
            result.update(status="skipped", skip_reason="forced_horizons_have_identical_executable_prefix")
            return result
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        regenerated, timing = evaluator._request(
            client, policy_input, mode=HISTORY_MODE, seed=root["root_request_seed"],
            previous_actions=root["previous_actions"], previous_horizon=root["previous_h"],
            budget_fraction=_budget_fraction(args), episode_progress=root["episode_progress"],
            absolute_decision_step=root["step"], args=args,
        )
        regenerated_actions = np.asarray(regenerated["actions"], dtype=np.float32)
        action_difference = (
            float(np.max(np.abs(regenerated_actions - root["saved_actions"])))
            if regenerated_actions.shape == root["saved_actions"].shape else None
        )
        result.update(root_regeneration_policy_calls=1, root_regeneration_rpc_ms=float(timing["wall_ms"]),
                      regenerated_action_max_abs_difference=action_difference)
        if action_difference is None or not np.isfinite(action_difference) or action_difference > COMPARISON_ATOL:
            result.update(status="skipped", skip_reason="regenerated_root_actions_differ_from_saved_actions")
            return result
        _, cache = evaluator._feedback_inputs(regenerated, None, step=root["step"], previous_h=root["previous_h"])
        result["status"] = "running"
        _write_json(output, result)
        for repeat in range(repeats):
            order = ("history_h", "a_h") if repeat % 2 == 0 else ("a_h", "history_h")
            for name in order:
                outcome = _run_forced_branch(
                    env=env, snapshot=snapshot, root=root, forced_h=root[name], repeat=repeat,
                    episode_step_limit=limit, task_description=task_description,
                    root_observation_cache=cache, client=client, selector=selector, args=args,
                )
                result["branches"].append({"first_h_source": name, **outcome})
                _write_json(output, result)
        result["paired"] = []
        for repeat in range(repeats):
            pair = {row["first_h_source"]: row for row in result["branches"] if row["repeat"] == repeat}
            history, swap = pair["history_h"], pair["a_h"]
            result["paired"].append({
                "repeat": repeat, "history_h_success": history["success"], "a_h_success": swap["success"],
                "success_delta_a_h_minus_history_h": int(swap["success"]) - int(history["success"]),
                "steps_delta_a_h_minus_history_h": swap["steps"] - history["steps"],
                "calls_delta_a_h_minus_history_h": swap["calls"] - history["calls"],
            })
        result["status"] = "complete"
        return result
    finally:
        libero_eval._safe_close_env(env)


def main(args: argparse.Namespace) -> None:
    if not 1 <= args.repeats <= 3:
        raise ValueError("This diagnostic permits one to three paired repeats per case.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory for this bounded diagnostic.")
    cases = load_analysis_cases(args.analysis_summary)
    eval_args = _baseline_args(args.baseline_config, output)
    planned = [{"case": case, "skip_reasons": eligibility_reasons(case)} for case in cases]
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        name: str(value) if isinstance(value, pathlib.Path) else value for name, value in vars(args).items()
    }
    _write_json(output / "run_config.json", configuration)
    eligible = [item for item in planned if not item["skip_reasons"]]
    results = []
    if eligible:
        selector = FeedbackSelector.load(args.history_params)
        if selector.variant != "history":
            raise ValueError("Both branches must continue with the saved history selector.")
        bank = InitialStateBank(eval_args.initial_state_bank)
        task_suite = libero_eval.benchmark.get_benchmark_dict()[eval_args.task_suite_name]()
        client = websocket_policy.WebsocketClientPolicy(
            eval_args.host, eval_args.port, api_key=eval_args.policy_api_key, ping_interval=None, ping_timeout=None,
        )
    for item in planned:
        case = item["case"]
        case_output = output / f"task{case['task_id']:02d}_ep{case['episode_id']:06d}.json"
        if item["skip_reasons"]:
            result = {
                "task_id": case["task_id"], "episode_id": case["episode_id"], "status": "skipped",
                "skip_reasons": item["skip_reasons"], "old_relation": case["old_relation"],
            }
        else:
            result = probe_case(
                case, trace_dir=args.trace_dir, task_suite=task_suite, bank=bank, selector=selector,
                client=client, args=eval_args, repeats=args.repeats, output=case_output,
            )
        _write_json(case_output, result)
        results.append(result)
        print(json.dumps({name: result[name] for name in ("task_id", "episode_id", "status")}), flush=True)
    pairs = [pair for result in results for pair in result.get("paired", [])]
    summary = {
        "status": "complete", "diagnostic_only": True, "cases": results,
        "completed_cases": sum(result["status"] == "complete" for result in results),
        "skipped_cases": sum(result["status"] == "skipped" for result in results),
        "executed_branches": sum(len(result.get("branches", [])) for result in results),
        "paired_repeats": len(pairs),
        "success_delta_a_h_minus_history_h": sum(pair["success_delta_a_h_minus_history_h"] for pair in pairs),
        "first_h_swap_rescues": sum(pair["success_delta_a_h_minus_history_h"] == 1 for pair in pairs),
        "first_h_swap_regressions": sum(pair["success_delta_a_h_minus_history_h"] == -1 for pair in pairs),
        "comparison_absolute_tolerance": COMPARISON_ATOL, "restored_physics_absolute_tolerance": RESTORE_ATOL,
        "continuation_seed_formula": "root_request_seed + 100000000 + repeat * 20000000 + continuation_call_index",
        "scope": (
            "Change only the first differing H; both branches execute the saved history chunk and then use history."
        ),
        "limitations": [
            "Cases were selected from earlier outcome disagreements; this is not a model success-rate evaluation.",
            "Root regeneration is one shared feature RPC per case and is not a speed measurement.",
            "Branch calls count the fixed root chunk once plus actual history continuation RPCs.",
            "Paired seeds align by continuation call index after H changes, not by absolute environment time.",
            "Both branches restore bank-initialization RNG; passive traces did not save the original root RNG.",
            "No critic, training, weight update, or action-chunk replacement is performed.",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"status": "complete", "summary": str(output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
