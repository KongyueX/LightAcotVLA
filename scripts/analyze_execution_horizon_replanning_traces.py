"""Compare passive A/history traces at common environment times."""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

BASE_MODE = "ordered_transformer"
HISTORY_MODE = "ordered_feedback_history"
MODES = (BASE_MODE, HISTORY_MODE)


@dataclasses.dataclass
class Trace:
    path: pathlib.Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-rows", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser


def discordant_episodes(path: pathlib.Path) -> list[dict[str, Any]]:
    indexed: dict[str, dict[tuple[int, int], dict[str, str]]] = {mode: {} for mode in MODES}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] in indexed:
                key = (int(row["task_id"]), int(row["episode"]))
                if key in indexed[row["mode"]]:
                    raise ValueError(f"Duplicate baseline episode {row['mode']}/{key}.")
                indexed[row["mode"]][key] = row
    if indexed[BASE_MODE].keys() != indexed[HISTORY_MODE].keys():
        raise ValueError("Baseline A and history rows must cover identical episode keys.")
    result = []
    for (task, episode), a in sorted(indexed[BASE_MODE].items()):
        history = indexed[HISTORY_MODE][task, episode]
        if a["initial_state_id"] != history["initial_state_id"]:
            raise ValueError(f"Baseline initial states differ at task{task}/episode{episode}.")
        a_success, h_success = bool(int(a["success"])), bool(int(history["success"]))
        if a_success != h_success:
            result.append({
                "task_id": task, "episode_id": episode, "task_name": a["task_name"],
                "old_relation": "rescue" if h_success else "regression",
                "old_success": {BASE_MODE: a_success, HISTORY_MODE: h_success},
            })
    if not result:
        raise ValueError("Baseline rows contain no A/history discordant episodes.")
    return result


def load_trace(path: pathlib.Path, *, mode: str, task_id: int, episode_id: int) -> Trace:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("error") is not None:
        raise ValueError(f"Trace {path} ended with an execution error: {metadata['error']}.")
    for field, expected in (("mode", mode), ("task_id", task_id), ("episode_id", episode_id)):
        if metadata.get(field) != expected:
            raise ValueError(f"Trace {path} has unexpected {field}: {metadata.get(field)!r}.")
    if not isinstance(metadata.get("success"), bool):
        raise ValueError(f"Trace {path} must record a boolean final success.")
    decision_steps = arrays["decision_steps"]
    state_steps = arrays["state_steps"]
    if decision_steps.ndim != 1 or state_steps.ndim != 1:
        raise ValueError("Trace time axes must be vectors.")
    if np.any(np.diff(decision_steps) <= 0) or np.any(np.diff(state_steps) <= 0):
        raise ValueError("Trace steps must be strictly increasing.")
    n, t = len(decision_steps), len(state_steps)
    for name in (
        "selected_h", "execution_h", "generated_action_chunks", "decision_physics_state",
        "decision_proprio", "decision_eef_pos", "decision_eef_quat", "decision_gripper_qpos", "request_seeds",
    ):
        if len(arrays[name]) != n:
            raise ValueError(f"Trace {name} does not align with decision_steps.")
    for name in ("state_decision_index", "executed_actions", "eef_pos", "eef_quat", "gripper_qpos"):
        if len(arrays[name]) != t:
            raise ValueError(f"Trace {name} does not align with state_steps.")
    if arrays["generated_action_chunks"].shape != (n, 25, 7) or arrays["executed_actions"].shape != (t, 7):
        raise ValueError("Trace actions must use raw [N,25,7] plans and [T,7] executed commands.")
    return Trace(path.resolve(), arrays, metadata)


def _stats(values: Any) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)), "min": float(values.min()), "max": float(values.max()),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("Measured EEF quaternion has zero norm.")
    similarity = np.clip(abs(float(np.dot(left, right))) / denominator, 0.0, 1.0)
    return float(2.0 * np.arccos(similarity))


def _state_lookup(trace: Trace, name: str) -> dict[int, np.ndarray]:
    a = trace.arrays
    result = {int(step): np.asarray(value, dtype=np.float64) for step, value in zip(a["state_steps"], a[name], strict=True)}
    result[int(np.asarray(a["initial_step"]).item())] = np.asarray(a[f"initial_{name}"], dtype=np.float64)
    return result


def _raw_command_difference(new: np.ndarray, old: np.ndarray) -> dict[str, float]:
    difference = np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
    return {
        "translation_command_delta_l2": float(np.linalg.norm(difference[:3])),
        "rotation_command_delta_l2": float(np.linalg.norm(difference[3:6])),
        "gripper_command_delta_abs": float(abs(difference[6])),
    }


def trace_metrics(trace: Trace) -> dict[str, Any]:
    a = trace.arrays
    states = {int(step): index for index, step in enumerate(a["state_steps"])}
    positions = _state_lookup(trace, "eef_pos")
    quaternions = _state_lookup(trace, "eef_quat")
    commands = np.asarray(a["executed_actions"], dtype=np.float64)
    chunks = np.asarray(a["generated_action_chunks"], dtype=np.float64)
    executed_changes: dict[str, list[dict[str, float]]] = {"boundary": [], "within_chunk": []}
    eef_changes: dict[str, list[float]] = {"boundary": [], "within_chunk": []}
    for row, step_value in enumerate(a["state_steps"]):
        step = int(step_value)
        decision = int(a["state_decision_index"][row])
        if decision < 0:
            continue
        boundary = decision > 0 and step == int(a["decision_steps"][decision]) + 1
        group = "boundary" if boundary else "within_chunk"
        if step - 1 in positions:
            eef_changes[group].append(float(np.linalg.norm(positions[step] - positions[step - 1])))
        if step - 1 in states:
            previous_row = states[step - 1]
            if int(a["state_decision_index"][previous_row]) >= 0:
                executed_changes[group].append(_raw_command_difference(commands[row], commands[previous_row]))

    boundaries = []
    for index in range(1, len(a["decision_steps"])):
        step = int(a["decision_steps"][index])
        consumed = step - int(a["decision_steps"][index - 1])
        overlap = min(max(25 - consumed, 0), 25)
        entry: dict[str, Any] = {
            "step": step, "previous_selected_h": int(a["selected_h"][index - 1]),
            "previous_execution_h": int(a["execution_h"][index - 1]),
            "selected_h": int(a["selected_h"][index]), "execution_h": int(a["execution_h"][index]),
            "overlap_steps": overlap,
            "overlap_translation_command_rmse": None, "overlap_rotation_command_rmse": None,
            "overlap_gripper_command_mae": None,
        }
        if overlap:
            difference = chunks[index, :overlap] - chunks[index - 1, consumed : consumed + overlap]
            entry.update({
                "overlap_translation_command_rmse": float(np.sqrt(np.mean(difference[:, :3] ** 2))),
                "overlap_rotation_command_rmse": float(np.sqrt(np.mean(difference[:, 3:6] ** 2))),
                "overlap_gripper_command_mae": float(np.mean(abs(difference[:, 6]))),
            })
        if step in states and step + 1 in states:
            old_command, new_command = commands[states[step]], commands[states[step + 1]]
            entry.update(_raw_command_difference(new_command, old_command))
            entry["translation_command_direction_cosine"] = _cosine(old_command[:3], new_command[:3])
            entry["gripper_nonzero_sign_flip"] = bool(old_command[6] * new_command[6] < 0)
            displacement = positions[step + 1] - positions[step]
            entry["eef_next_step_displacement_m"] = float(np.linalg.norm(displacement))
            entry["eef_next_step_rotation_rad"] = _quaternion_angle(quaternions[step], quaternions[step + 1])
            if step - 1 in positions:
                preceding = positions[step] - positions[step - 1]
                entry["eef_previous_step_displacement_m"] = float(np.linalg.norm(preceding))
                entry["eef_displacement_direction_cosine"] = _cosine(preceding, displacement)
        boundaries.append(entry)

    boundary_summary = {}
    for name in (
        "overlap_translation_command_rmse", "overlap_rotation_command_rmse", "overlap_gripper_command_mae",
        "translation_command_delta_l2", "rotation_command_delta_l2", "gripper_command_delta_abs",
        "eef_next_step_displacement_m", "eef_next_step_rotation_rad", "eef_previous_step_displacement_m",
        "translation_command_direction_cosine", "eef_displacement_direction_cosine",
    ):
        boundary_summary[name] = _stats(entry[name] for entry in boundaries if entry.get(name) is not None)
    flips = [entry["gripper_nonzero_sign_flip"] for entry in boundaries if "gripper_nonzero_sign_flip" in entry]
    boundary_summary["gripper_nonzero_sign_flip_count"] = sum(flips)
    boundary_summary["gripper_nonzero_sign_flip_fraction"] = float(np.mean(flips)) if flips else None
    boundary_summary["no_overlap_boundary_count"] = sum(entry["overlap_steps"] == 0 for entry in boundaries)
    executed_summary = {
        group: {
            name: _stats(entry[name] for entry in entries)
            for name in ("translation_command_delta_l2", "rotation_command_delta_l2", "gripper_command_delta_abs")
        }
        for group, entries in executed_changes.items()
    }
    plan_differences = np.diff(chunks, axis=1)
    planned_summary = {
        "per_call_mean_translation_command_delta_l2": _stats(np.linalg.norm(plan_differences[..., :3], axis=-1).mean(axis=1)),
        "per_call_mean_rotation_command_delta_l2": _stats(np.linalg.norm(plan_differences[..., 3:6], axis=-1).mean(axis=1)),
        "per_call_mean_gripper_command_delta_abs": _stats(abs(plan_differences[..., 6]).mean(axis=1)),
    }
    episode_metrics = {
        "boundary_translation_command_delta_mean": boundary_summary["translation_command_delta_l2"]["mean"],
        "within_translation_command_delta_mean": executed_summary["within_chunk"]["translation_command_delta_l2"]["mean"],
        "overlap_translation_command_rmse_mean": boundary_summary["overlap_translation_command_rmse"]["mean"],
        "boundary_eef_displacement_mean_m": boundary_summary["eef_next_step_displacement_m"]["mean"],
        "within_eef_displacement_mean_m": _stats(eef_changes["within_chunk"])["mean"],
        "boundary_gripper_sign_flip_fraction": boundary_summary["gripper_nonzero_sign_flip_fraction"],
    }
    top_boundaries = sorted(
        (entry for entry in boundaries if "translation_command_delta_l2" in entry),
        key=lambda entry: entry["translation_command_delta_l2"], reverse=True,
    )[:5]
    return {
        "trace": str(trace.path), "success": trace.metadata["success"],
        "decision_count": len(a["decision_steps"]), "actual_execution_steps": int(np.sum(a["execution_h"])),
        "boundary_count": len(boundaries), "boundary_summary": boundary_summary,
        "executed_command_differences": executed_summary, "generated_plan_differences": planned_summary,
        "episode_metrics": episode_metrics,
        "largest_boundary_command_changes": top_boundaries,
        "example_order": "Five largest raw translation-command changes; ranking does not label anomalies or causes.",
    }


def _aligned_states(left: Trace, right: Trace, *, through_step: int | None = None) -> dict[str, Any]:
    lookups = {name: (_state_lookup(left, name), _state_lookup(right, name)) for name in ("eef_pos", "eef_quat", "gripper_qpos")}
    shared = sorted(lookups["eef_pos"][0].keys() & lookups["eef_pos"][1].keys())
    if through_step is not None:
        shared = [step for step in shared if step <= through_step]
    position_differences = [np.linalg.norm(lookups["eef_pos"][0][step] - lookups["eef_pos"][1][step]) for step in shared]
    rotation_differences = [_quaternion_angle(lookups["eef_quat"][0][step], lookups["eef_quat"][1][step]) for step in shared]
    gripper_differences = [np.max(abs(lookups["gripper_qpos"][0][step] - lookups["gripper_qpos"][1][step])) for step in shared]
    command_maps = [
        {int(step): command for step, command, decision in zip(
            trace.arrays["state_steps"], trace.arrays["executed_actions"], trace.arrays["state_decision_index"], strict=True,
        ) if int(decision) >= 0}
        for trace in (left, right)
    ]
    command_steps = sorted(command_maps[0].keys() & command_maps[1].keys())
    if through_step is not None:
        command_steps = [step for step in command_steps if step <= through_step]
    command_differences = [np.max(abs(command_maps[0][step] - command_maps[1][step])) for step in command_steps]
    return {
        "shared_state_steps": len(shared), "first_state_step": shared[0] if shared else None,
        "last_state_step": shared[-1] if shared else None,
        "eef_position_difference_m": _stats(position_differences),
        "eef_rotation_difference_rad": _stats(rotation_differences),
        "gripper_qpos_max_abs_difference": _stats(gripper_differences),
        "executed_command_max_abs_difference": _stats(command_differences),
    }


def compare_alignment(left: Trace, right: Trace) -> dict[str, Any]:
    indices = [
        {int(step): index for index, step in enumerate(trace.arrays["decision_steps"])}
        for trace in (left, right)
    ]
    shared = sorted(indices[0].keys() & indices[1].keys())
    mismatches = [step for step in shared if left.arrays["selected_h"][indices[0][step]] != right.arrays["selected_h"][indices[1][step]]]
    first_h = mismatches[0] if mismatches else None
    common_end = min(int(left.arrays["state_steps"][-1]), int(right.arrays["state_steps"][-1]))
    grid_difference = sorted(step for step in indices[0].keys() ^ indices[1].keys() if step < common_end)
    first_grid = grid_difference[0] if grid_difference else None
    cutoffs = [value for value in (first_h, first_grid) if value is not None]
    through = min(cutoffs) if cutoffs else common_end
    prefix_decisions = [step for step in shared if step <= through]
    plan_differences, physics_differences, proprio_differences = [], [], []
    matching_seeds = 0
    for step in prefix_decisions:
        left_index, right_index = indices[0][step], indices[1][step]
        plan_differences.append(np.max(abs(left.arrays["generated_action_chunks"][left_index] - right.arrays["generated_action_chunks"][right_index])))
        physics_differences.append(np.max(abs(left.arrays["decision_physics_state"][left_index] - right.arrays["decision_physics_state"][right_index])))
        proprio_differences.append(np.max(abs(left.arrays["decision_proprio"][left_index] - right.arrays["decision_proprio"][right_index])))
        matching_seeds += int(left.arrays["request_seeds"][left_index] == right.arrays["request_seeds"][right_index])
    first_record = None
    if first_h is not None:
        ia, ih = indices[0][first_h], indices[1][first_h]
        first_record = {
            "step": first_h, "a_selected_h": int(left.arrays["selected_h"][ia]),
            "history_selected_h": int(right.arrays["selected_h"][ih]),
            "a_request_seed": int(left.arrays["request_seeds"][ia]),
            "history_request_seed": int(right.arrays["request_seeds"][ih]),
            "generated_chunk_max_abs_difference": float(np.max(abs(left.arrays["generated_action_chunks"][ia] - right.arrays["generated_action_chunks"][ih]))),
            "decision_physics_max_abs_difference": float(np.max(abs(left.arrays["decision_physics_state"][ia] - right.arrays["decision_physics_state"][ih]))),
        }
        for label, trace, index in (("a", left, ia), ("history", right, ih)):
            for key in ("anchor_continuation_logits", "anchor_probabilities", "ordered_continuation_logits", "ordered_probabilities"):
                if key in trace.arrays:
                    first_record[f"{label}_{key}"] = np.asarray(trace.arrays[key][index]).tolist()
    return {
        "alignment": "Equal absolute environment steps, never equal call indices after the decision grids diverge.",
        "common_environment_end_step": common_end, "shared_decision_steps": len(shared),
        "first_shared_decision_with_h_difference": first_record,
        "first_decision_grid_difference_step": first_grid,
        "common_prefix_through_step": through,
        "before_first_h_or_grid_difference": {
            **_aligned_states(left, right, through_step=through),
            "shared_decisions": len(prefix_decisions), "matching_request_seeds": matching_seeds,
            "generated_chunk_max_abs_difference": _stats(plan_differences),
            "decision_physics_max_abs_difference": _stats(physics_differences),
            "decision_raw_proprio_max_abs_difference": _stats(proprio_differences),
            "scope": "Includes observations and the already generated plans at the first differing decision, before its actions execute.",
        },
        "full_common_environment_time": _aligned_states(left, right),
    }


def analyze_episode(case: dict[str, Any], a: Trace, history: Trace) -> dict[str, Any]:
    replay_success = {BASE_MODE: a.metadata["success"], HISTORY_MODE: history.metadata["success"]}
    reproduced = {mode: replay_success[mode] == case["old_success"][mode] for mode in MODES}
    if replay_success[BASE_MODE] != replay_success[HISTORY_MODE]:
        relation = "rescue" if replay_success[HISTORY_MODE] else "regression"
    else:
        relation = "both_success" if replay_success[BASE_MODE] else "both_failure"
    return {
        **case, "reproduction": {
            "trace_success": replay_success, "matches_original_by_mode": reproduced,
            "original_pair_reproduced": all(reproduced.values()), "trace_relation": relation,
        },
        "traces": {BASE_MODE: trace_metrics(a), HISTORY_MODE: trace_metrics(history)},
        "alignment": compare_alignment(a, history),
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def group_metrics(selected: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        names = selected[0]["traces"][mode]["episode_metrics"] if selected else ()
        return {
            name: _stats(ep["traces"][mode]["episode_metrics"][name] for ep in selected
                         if ep["traces"][mode]["episode_metrics"][name] is not None)
            for name in names
        }
    old_groups = {}
    for relation in ("rescue", "regression"):
        selected = [ep for ep in episodes if ep["old_relation"] == relation]
        old_groups[relation] = {
            "episodes": len(selected), "original_pairs_reproduced": sum(ep["reproduction"]["original_pair_reproduced"] for ep in selected),
            "metrics": {mode: group_metrics(selected, mode) for mode in MODES},
        }
    outcome_groups = {
        mode: {
            name: {
                "episodes": sum(ep["traces"][mode]["success"] is success for ep in episodes),
                "metrics": group_metrics([ep for ep in episodes if ep["traces"][mode]["success"] is success], mode),
            }
            for name, success in (("trace_success", True), ("trace_failure", False))
        }
        for mode in MODES
    }
    return {
        "episode_count": len(episodes),
        "original_pairs_reproduced": sum(ep["reproduction"]["original_pair_reproduced"] for ep in episodes),
        "original_discordance_groups": old_groups, "trace_outcome_groups": outcome_groups,
        "aggregation": "Each episode contributes one mean per metric; episodes with no applicable boundary contribute no value.",
        "limitations": [
            "Cases were selected for earlier A/history outcome disagreement, so these traces are not a new success-rate evaluation.",
            "Raw action differences describe command changes, not measured physical velocity or safety.",
            "No unexecuted suffix after H25 means no overlap evidence, not perfect plan consistency.",
            "Boundary changes and outcome associations are diagnostic clues; passive traces do not identify a causal failure mechanism.",
        ],
        "minimal_causal_followup": (
            "Only after a case reproduces and its pre-divergence traces agree: restore the first differing-decision snapshot, "
            "reuse the already generated action chunk, and change only history's H at that decision to A's H; then resume "
            "history unchanged. Compare against unmodified history with paired continuation seeds. Not run by this analyzer."
        ),
    }


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def episode_markdown(report: dict[str, Any]) -> str:
    reproduction = report["reproduction"]
    relation = {"rescue": "救回", "regression": "退化", "both_success": "均成功", "both_failure": "均失败"}
    first = report["alignment"]["first_shared_decision_with_h_difference"]
    prefix = report["alignment"]["before_first_h_or_grid_difference"]
    lines = [
        f"# Task{report['task_id']} / episode{report['episode_id']}", "", report["task_name"], "",
        f"旧结果：{relation[report['old_relation']]}；本次轨迹：{relation[reproduction['trace_relation']]}；"
        f"两模式原成败均复现：{'是' if reproduction['original_pair_reproduced'] else '否'}。", "",
    ]
    if first:
        lines.append(f"首个共有决策点的H分歧在环境step {first['step']}：A选择{first['a_selected_h']}，history选择{first['history_selected_h']}。")
    else:
        lines.append("共有决策点未记录到H分歧；需结合决策时间表差异与提前结束记录解释。")
    lines += [
        f"分歧前共同时间段的EEF位置最大差 {_number(prefix['eef_position_difference_m']['max'])} m，"
        f"已执行命令最大绝对差 {_number(prefix['executed_command_max_abs_difference']['max'])}；"
        f"同一步生成chunk最大绝对差 {_number(prefix['generated_chunk_max_abs_difference']['max'])}。", "",
        "| 指标 | A | history |", "| --- | ---: | ---: |",
    ]
    labels = {
        "boundary_translation_command_delta_mean": "边界原始平移命令差均值",
        "within_translation_command_delta_mean": "chunk内原始平移命令差均值",
        "overlap_translation_command_rmse_mean": "重叠suffix原始平移命令RMSE均值",
        "boundary_eef_displacement_mean_m": "边界后一步EEF位移均值（m）",
        "within_eef_displacement_mean_m": "其他执行步EEF位移均值（m）",
        "boundary_gripper_sign_flip_fraction": "边界夹爪非零命令反号比例",
    }
    for key, label in labels.items():
        values = [_number(report["traces"][mode]["episode_metrics"][key]) for mode in MODES]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", "边界原始平移命令变化最大的记录（用于定位，无异常阈值）：", "",
              "| 模式 | 环境step | 命令差 | 重叠步数 | EEF下一步位移（m） | 夹爪反号 |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for mode in MODES:
        label = "A" if mode == BASE_MODE else "history"
        for entry in report["traces"][mode]["largest_boundary_command_changes"]:
            lines.append(f"| {label} | {entry['step']} | {_number(entry['translation_command_delta_l2'])} | "
                         f"{entry['overlap_steps']} | {_number(entry.get('eef_next_step_displacement_m'))} | "
                         f"{'是' if entry.get('gripper_nonzero_sign_flip') else '否'} |")
    lines += ["", "原始命令差不代表物理速度；单个跳变和终局成败的关联不能确定失败原因。", ""]
    return "\n".join(lines)


def summary_markdown(summary: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
    lines = [
        "# A与短历史选择器的重规划轨迹诊断", "",
        f"已分析{summary['episode_count']}组旧成败不一致episode，"
        f"其中{summary['original_pairs_reproduced']}组在两模式下均复现原成败。", "",
        "| Task/episode | 旧关系 | 轨迹A成功 | 轨迹history成功 | 原结果复现 | 首个共有H分歧step | 详情 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for report in episodes:
        reproduced = report["reproduction"]
        first = report["alignment"]["first_shared_decision_with_h_difference"]
        filename = f"task{report['task_id']:02d}_ep{report['episode_id']:06d}.md"
        lines.append(f"| {report['task_id']}/{report['episode_id']} | {'救回' if report['old_relation']=='rescue' else '退化'} | "
                     f"{int(reproduced['trace_success'][BASE_MODE])} | {int(reproduced['trace_success'][HISTORY_MODE])} | "
                     f"{'是' if reproduced['original_pair_reproduced'] else '否'} | {first['step'] if first else '—'} | "
                     f"[轨迹指标](episodes/{filename}) |")
    lines += ["", "## Episode等权比较", "",
              "每条episode先计算指标均值，再在旧救回组和旧退化组内等权汇总；无有效重叠的边界不按零计入。", "",
              "| 旧组别 | 模式 | episode数 | 边界命令差均值 | suffix命令RMSE均值 | 边界EEF位移均值（m） |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for relation in ("rescue", "regression"):
        group = summary["original_discordance_groups"][relation]
        for mode in MODES:
            metrics = group["metrics"][mode]
            values = [_number(metrics.get(key, {}).get("mean")) for key in (
                "boundary_translation_command_delta_mean", "overlap_translation_command_rmse_mean", "boundary_eef_displacement_mean_m",
            )]
            lines.append(f"| {'救回' if relation=='rescue' else '退化'} | {'A' if mode==BASE_MODE else 'history'} | "
                         f"{group['episodes']} | {' | '.join(values)} |")
    lines += ["", "## Limitations", "",
              "这些episode按旧结果挑选，不能用来估计总体成功率；原始命令差不是实测物理速度。H25后没有未执行suffix，只表示没有重叠比较证据。数值不设异常阈值，被动轨迹不能证明某个跳变导致失败。", "",
              "若需要因果介入，最小下一项是在已复现且分歧前状态/计划一致的一条case上，从首个分歧snapshot复用同一动作chunk，仅把history当次H改为A当次H，再恢复history，配对continuation seeds比较；本分析未执行该介入。", ""]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    cases = discordant_episodes(args.baseline_rows)
    output = args.output_dir.resolve()
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in cases:
        filename = f"task{case['task_id']:02d}_ep{case['episode_id']:06d}"
        traces = [load_trace(args.trace_dir / mode / f"{filename}.npz", mode=mode,
                             task_id=case["task_id"], episode_id=case["episode_id"]) for mode in MODES]
        report = analyze_episode(case, *traces)
        (episode_dir / f"{filename}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (episode_dir / f"{filename}.md").write_text(episode_markdown(report))
        reports.append(report)
    summary = {
        "status": "complete", "baseline_rows": str(args.baseline_rows.resolve()),
        "trace_dir": str(args.trace_dir.resolve()), **aggregate_episodes(reports),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(summary_markdown(summary, reports))
    print(json.dumps({"status": "complete", "episode_count": len(reports),
                      "original_pairs_reproduced": summary["original_pairs_reproduced"], "report": str(output / "report.md")}))


if __name__ == "__main__":
    main(build_parser().parse_args())
