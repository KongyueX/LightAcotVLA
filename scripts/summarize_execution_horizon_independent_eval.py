"""Summarize a complete, paired H5/ordered-predictor evaluation without model calls."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import pathlib
from typing import Any

import audit_fixed_h_paired as paired
import numpy as np

REFERENCE = "original"
CANDIDATE = "ordered_transformer"
MODES = (REFERENCE, CANDIDATE)
PROGRESS_BINS = ("[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]")
METRIC_LABELS = {
    "policy_calls": ("Calls", 1.0),
    "actual_policy_total_ms": ("Policy (s)", 1000.0),
    "policy_rpc_wall_total_ms": ("RPC (s)", 1000.0),
    "actual_episode_elapsed_total_ms": ("Full episode (s)", 1000.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_rollouts(
    rows: list[dict[str, str]], summary: dict[str, Any]
) -> tuple[dict[str, dict[tuple[int, int], dict[str, str]]], list[tuple[int, int]]]:
    if summary.get("status") != "complete":
        raise ValueError("Evaluation summary is not complete.")
    if int(summary["config"]["original_horizon"]) != 5:
        raise ValueError("The original-mode reference must be Fixed H5.")
    runs: dict[str, dict[tuple[int, int], dict[str, str]]] = {mode: {} for mode in MODES}
    required = {"mode", "task_id", "episode", "initial_state_id", "success", "timeout", *paired._METRICS}
    for row in rows:
        if missing := required.difference(row):
            raise ValueError(f"Rollout row is missing columns: {sorted(missing)}.")
        mode = row["mode"]
        if mode not in runs:
            raise ValueError(f"Unexpected evaluation mode: {mode!r}.")
        key = (int(row["task_id"]), int(row["episode"]))
        if key in runs[mode]:
            raise ValueError(f"Duplicate rollout key for {mode}: {key}.")
        int(row["initial_state_id"])
        if any(not np.isfinite(float(row[field])) for field in (*paired._METRICS, "success", "timeout")):
            raise ValueError(f"Missing/nonfinite primary outcome or timing at {mode}/{key}.")
        runs[mode][key] = row
    keys = paired._validate_pairing(runs)
    task_start = int(summary["config"].get("task_start", 0))
    tasks = range(task_start, task_start + int(summary["num_tasks"]))
    episodes = [int(value) for value in summary["episode_ids"]]
    if len(episodes) != int(summary["num_trials_per_task"]):
        raise ValueError("Summary episode_ids and num_trials_per_task disagree.")
    expected = {(task, episode) for task in tasks for episode in episodes}
    if set(keys) != expected:
        raise ValueError("Actual rollout keys do not match the complete summary's expected task/episode grid.")
    for mode in MODES:
        if int(summary["overall"][mode]["episodes"]) != len(keys):
            raise ValueError(f"Summary episode count disagrees with rollout rows for {mode}.")
    if not keys:
        raise ValueError("The complete evaluation has no paired episodes.")
    return runs, keys


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _run_summary(rows: dict, keys: list[tuple[int, int]]) -> dict[str, Any]:
    result = paired._run_summary(rows, keys)
    overhead = [(key, _finite(rows[key].get("actual_predictor_total_ms"))) for key in keys]
    available = [(key, value) for key, value in overhead if value is not None]
    total_calls = sum(float(rows[key]["policy_calls"]) for key, _ in available)
    total_policy = sum(float(rows[key]["actual_policy_total_ms"]) for key, _ in available)
    total_overhead = sum(value for _, value in available)
    result["predictor_overhead"] = {
        "episodes_available": len(available),
        "status": "complete" if len(available) == len(keys) else "missing_or_partial",
        "mean_ms_per_episode": _mean([value for _, value in available]),
        "mean_ms_per_call": total_overhead / total_calls if total_calls else None,
        "fraction_of_policy_time": total_overhead / total_policy if total_policy else None,
    }
    return result


def _decision_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decisions": len(items),
        "episodes": len({item["key"] for item in items}),
        "selected_horizon_counts": dict(sorted(Counter(item["selected_horizon"] for item in items).items())),
        "execution_horizon_counts": dict(sorted(Counter(item["execution_horizon"] for item in items).items())),
    }
    for field in ("selected_probability", "selected_vs_best_other_margin", "entropy_nats", "budget_fraction"):
        by_episode: dict[tuple[int, int], list[float]] = defaultdict(list)
        for item in items:
            if item.get(field) is not None:
                by_episode[item["key"]].append(item[field])
        result[field] = {
            "count": sum(len(values) for values in by_episode.values()),
            "episodes": len(by_episode),
            "mean": _mean([float(np.mean(values)) for values in by_episode.values()]),
        }
    long_by_episode: dict[tuple[int, int], list[float]] = defaultdict(list)
    for item in items:
        long_by_episode[item["key"]].append(float(item["selected_horizon"] >= 15))
    result["long_h_fraction"] = _mean([float(np.mean(values)) for values in long_by_episode.values()])
    return result


def _decision_diagnostics(
    rows: list[dict[str, str]] | None,
    candidate: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    expected_calls = int(sum(float(row["policy_calls"]) for row in candidate.values()))
    if rows is None:
        return {"status": "missing", "reason": "decisions.csv is absent", "expected_calls": expected_calls}
    items = []
    missing: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    continuation_logits: dict[str, list[float]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row["mode"] != CANDIDATE:
            continue
        key = (int(row["task_id"]), int(row["episode"]))
        if key not in candidate:
            raise ValueError(f"Decision has no matching ordered rollout: {key}.")
        if int(row["initial_state_id"]) != int(candidate[key]["initial_state_id"]):
            raise ValueError(f"Decision initial_state_id differs from its rollout at {key}.")
        decision_key = (*key, int(row["environment_step"]))
        if decision_key in seen:
            raise ValueError(f"Duplicate ordered decision: {decision_key}.")
        seen.add(decision_key)
        info = json.loads(row.get("selector_json") or "{}")
        item = {
            "key": key,
            "outcome": "success" if int(candidate[key]["success"]) else "failure",
            "selected_horizon": int(row["selected_horizon"]),
            "execution_horizon": int(row["execution_horizon"]),
        }
        for field in ("previous_horizon", "episode_progress", "budget_fraction"):
            item[field] = _finite(info.get(field))
            if item[field] is None:
                missing[field] += 1
        previous = item["previous_horizon"]
        if previous is not None:
            transitions[f"{int(previous)}->{item['selected_horizon']}"] += 1
        progress = item["episode_progress"]
        if progress is not None and 0.0 <= progress <= 1.0:
            item["progress_bin"] = PROGRESS_BINS[min(int(progress * 4), 3)]
        else:
            item["progress_bin"] = "missing"
        probabilities = info.get("ordered_horizon_probability")
        candidates = info.get("candidate_horizons")
        if probabilities is None or candidates is None:
            missing["ordered_horizon_probability_or_candidates"] += 1
        else:
            probability = np.asarray(probabilities, dtype=np.float64)
            horizons = [int(value) for value in candidates]
            if (
                probability.shape != (len(horizons),)
                or len(horizons) < 2
                or item["selected_horizon"] not in horizons
                or not np.all(np.isfinite(probability))
                or np.any(probability < 0)
                or not np.isclose(np.sum(probability), 1.0, atol=1e-4)
            ):
                missing["invalid_ordered_probability"] += 1
            else:
                index = horizons.index(item["selected_horizon"])
                item["selected_probability"] = float(probability[index])
                item["selected_vs_best_other_margin"] = float(
                    probability[index] - np.max(np.delete(probability, index))
                )
                positive = probability[probability > 0]
                item["entropy_nats"] = float(-np.sum(positive * np.log(positive)))
            logits = info.get("ordered_continuation_logits")
            if (
                logits is not None
                and len(logits) == len(horizons) - 1
                and all(_finite(value) is not None for value in logits)
            ):
                for left, right, value in zip(horizons[:-1], horizons[1:], logits, strict=True):
                    continuation_logits[f"{left}->{right}"].append(float(value))
            else:
                missing["ordered_continuation_logits"] += 1
        items.append(item)
    group = _decision_group(items)
    group.update(
        status="complete" if len(items) == expected_calls and not missing else "missing_or_partial",
        expected_calls=expected_calls,
        missing_fields=dict(missing),
        previous_execution_to_selected_horizon_counts=dict(sorted(transitions.items())),
        continuation_logits={
            name: {"count": len(values), "mean": _mean(values)} for name, values in continuation_logits.items()
        },
        by_episode_outcome={
            outcome: _decision_group([item for item in items if item["outcome"] == outcome])
            for outcome in ("success", "failure")
        },
        by_progress={
            name: _decision_group([item for item in items if item["progress_bin"] == name])
            for name in (*PROGRESS_BINS, "missing")
        },
        by_selected_horizon={
            str(horizon): {
                outcome: _decision_group([
                    item for item in items if item["selected_horizon"] == horizon and item["outcome"] == outcome
                ])
                for outcome in ("success", "failure")
            }
            for horizon in sorted({item["selected_horizon"] for item in items})
        },
        interpretation=(
            "Confidence and long-H fractions average calls within each episode, then episodes equally; "
            "associations only, not success probabilities."
        ),
    )
    return group


def _state_keys(rows: dict, keys: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [
        {"task_id": task, "episode": episode, "initial_state_id": int(rows[(task, episode)]["initial_state_id"])}
        for task, episode in keys
    ]


def _directions(analysis: dict[str, Any]) -> list[str]:
    directions = []
    losses = sorted(analysis["paired"]["per_task"].items(), key=lambda item: item[1]["regressions"], reverse=True)
    if losses and losses[0][1]["regressions"]:
        task, values = losses[0]
        directions.append(
            f"Task {task} 有 {values['regressions']} 个相对 H5 退化初始状态：优先查看已列出的轨迹，"
            "区分接触、抓取与后期恢复，再决定针对性补采；不先增加层数。"
        )
    else:
        directions.append(
            "本批没有相对 H5 的退化状态；保留当前模型，后续先验证跨种子的可重复性，不因 H 分布单一改架构。"
        )
    diagnostics = analysis["decisions"]
    if "by_episode_outcome" in diagnostics:
        success_long = diagnostics["by_episode_outcome"]["success"]["long_h_fraction"]
        failure_long = diagnostics["by_episode_outcome"]["failure"]["long_h_fraction"]
        if success_long is not None and failure_long is not None:
            directions.append(
                f"最终失败/成功 episode 的长 H（≥15）调用占比均值为 {failure_long:.1%}/{success_long:.1%}；"
                "这是 episode 等权相关性，后续需对相同状态做反事实分支才能判断缩短 H 是否救回，不能直接调阈值。"
            )
    overhead = analysis["runs"][CANDIDATE]["predictor_overhead"]
    if overhead["mean_ms_per_call"] is not None:
        directions.append(
            f"Predictor 每次调用平均 {overhead['mean_ms_per_call']:.3f} ms；"
            "将其与 RPC/整局时间差结合判断是否值得优化 sidecar 开销，不以 calls 单项代替实际提速。"
        )
    else:
        directions.append(
            "缺少 predictor 开销诊断；当前只使用已记录的 policy/RPC/整局时间判断收益，不推测 sidecar 成本。"
        )
    return directions[:3]


def analyze(eval_dir: pathlib.Path, *, samples: int, seed: int) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    runs, keys = _split_rollouts(_read_csv(eval_dir / "rollout_rows.csv"), summary)
    audit = paired._pairwise_audit(
        runs[REFERENCE], runs[CANDIDATE], keys, samples=samples, seed=seed, noninferiority_margin=0.01
    )
    per_task = {}
    for task in sorted({key[0] for key in keys}):
        task_keys = [key for key in keys if key[0] == task]
        per_task[str(task)] = {
            "runs": {mode: _run_summary(runs[mode], task_keys) for mode in MODES},
            "paired": paired._pairwise_audit(
                runs[REFERENCE], runs[CANDIDATE], task_keys,
                samples=samples, seed=seed + task + 100, noninferiority_margin=0.01,
            ),
        }
    both = [key for key in keys if int(runs[REFERENCE][key]["success"]) and int(runs[CANDIDATE][key]["success"])]
    rescue = [key for key in keys if not int(runs[REFERENCE][key]["success"]) and int(runs[CANDIDATE][key]["success"])]
    regression = [key for key in keys if int(runs[REFERENCE][key]["success"]) and not int(runs[CANDIDATE][key]["success"])]
    decisions_path = eval_dir / "decisions.csv"
    analysis = {
        "status": "complete",
        "eval_dir": str(eval_dir.resolve()),
        "source_summary": summary,
        "paired_key": ["task_id", "episode"],
        "initial_state_id_verified": True,
        "bootstrap": {
            "samples": samples,
            "seed": seed,
            "method": "equal-weight task / task-by-initial-state hierarchical cluster resampling",
            "ci95": "two-sided 2.5/97.5 percentiles; lcb95/ucb95 are one-sided 5/95 percentiles",
        },
        "runs": {mode: _run_summary(runs[mode], keys) for mode in MODES},
        "paired": audit,
        "per_task": per_task,
        "both_success": {
            "episodes": len(both),
            "interpretation": "Descriptive outcome-selected subset, not a replacement for the all-episode comparison.",
            "runs": {mode: _run_summary(runs[mode], both) for mode in MODES} if both else {},
            "paired": paired._pairwise_audit(
                runs[REFERENCE], runs[CANDIDATE], both,
                samples=samples, seed=seed + 10000, noninferiority_margin=0.01,
            ) if both else None,
        },
        "rescue_states": _state_keys(runs[REFERENCE], rescue),
        "regression_states": _state_keys(runs[REFERENCE], regression),
        "decisions": _decision_diagnostics(_read_csv(decisions_path) if decisions_path.exists() else None, runs[CANDIDATE]),
    }
    analysis["improvement_directions"] = _directions(analysis)
    return analysis


def _number(value: float | None, *, scale: float = 1.0) -> str:
    return "missing" if value is None else f"{value / scale:.3f}"


def _comparison_table(runs: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    reference, candidate = runs[REFERENCE], runs[CANDIDATE]
    success_ci = audit["success_delta_cluster_bootstrap"]["ci95"]
    lines = [
        "| Metric | Fixed H5 | A ordered | A − H5 [paired 95% CI] |",
        "| --- | ---: | ---: | ---: |",
        f"| Success | {reference['success_count']}/{reference['episodes']} ({reference['success_rate']:.1%}) "
        f"| {candidate['success_count']}/{candidate['episodes']} ({candidate['success_rate']:.1%}) "
        f"| {audit['success_rate_delta'] * 100:+.2f} pp [{success_ci[0] * 100:.2f}, {success_ci[1] * 100:.2f}] |",
    ]
    for metric, (label, scale) in METRIC_LABELS.items():
        difference = audit["metric_delta_candidate_minus_reference"][metric]
        lower, upper = difference["ci95"]
        lines.append(
            f"| {label} | {_number(reference['means'][metric], scale=scale)} "
            f"| {_number(candidate['means'][metric], scale=scale)} | {_number(difference['mean'], scale=scale)} "
            f"[{_number(lower, scale=scale)}, {_number(upper, scale=scale)}] |"
        )
    return lines


def report_markdown(analysis: dict[str, Any]) -> str:
    summary, audit = analysis["source_summary"], analysis["paired"]
    lines = [
        "# Independent execution-horizon evaluation", "",
        f"{audit['paired_episodes']} paired states; {summary['num_tasks']} tasks. Only this bank/run is compared.", "",
    ]
    for key in ("initial_state_bank", "initial_state_bank_sha256", "initial_state_identity_mode", "timing_semantics"):
        lines.append(f"- {key}: {summary.get(key, 'missing')}")
    lines += [
        "", "## All episodes", "", *_comparison_table(analysis["runs"], audit), "",
        f"Rescues: {audit['rescues']}; regressions: {audit['regressions']}; "
        f"exact paired McNemar two-sided p={audit['exact_mcnemar_two_sided_p']:.5g}.", "",
    ]
    for metric, (label, _) in METRIC_LABELS.items():
        reduction = audit["speedups"][metric]["candidate_reduction_fraction"]
        lines.append(
            f"- {label} reduction vs H5: {reduction:.2%}." if reduction is not None else f"- {label} reduction: missing."
        )
    overhead = analysis["runs"][CANDIDATE]["predictor_overhead"]
    lines += [
        f"- Predictor: {_number(overhead['mean_ms_per_call'])} ms/call; "
        f"{_number(overhead['mean_ms_per_episode'])} ms/episode ({overhead['episodes_available']} episodes recorded).",
        "", "## Per task", "",
    ]
    for task, result in analysis["per_task"].items():
        lines += [
            f"### Task {task}", "", *_comparison_table(result["runs"], result["paired"]), "",
            f"Rescues/regressions: {result['paired']['rescues']}/{result['paired']['regressions']}.", "",
        ]
    both = analysis["both_success"]
    lines += [
        "## Both-success episodes (descriptive)", "",
        f"{both['episodes']} paired states selected by both outcomes; not a substitute for all episodes.", "",
    ]
    if both["paired"] is not None:
        lines += _comparison_table(both["runs"], both["paired"]) + [""]
    lines += ["## Ordered decisions", ""]
    diagnostics = analysis["decisions"]
    lines.append(f"Diagnostics status: {diagnostics['status']}.")
    if "selected_horizon_counts" in diagnostics:
        lines += [
            "", f"Selected H: `{json.dumps(diagnostics['selected_horizon_counts'], sort_keys=True)}`.", "",
            "Execution H (after evaluator clipping): "
            f"`{json.dumps(diagnostics['execution_horizon_counts'], sort_keys=True)}`.", "",
            "Previous executed H → selected H: "
            f"`{json.dumps(diagnostics['previous_execution_to_selected_horizon_counts'], sort_keys=True)}`.", "",
            f"Missing diagnostics: `{json.dumps(diagnostics['missing_fields'], sort_keys=True)}`.", "",
            "| Episode outcome / selected H | Calls | Mean selected probability | Mean margin | Entropy (nats) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for horizon, groups in diagnostics["by_selected_horizon"].items():
            for outcome, group in groups.items():
                lines.append(
                    f"| {outcome} / H{horizon} | {group['decisions']} "
                    f"| {_number(group['selected_probability']['mean'])} "
                    f"| {_number(group['selected_vs_best_other_margin']['mean'])} "
                    f"| {_number(group['entropy_nats']['mean'])} |"
                )
        lines += ["", "| Progress | Calls | Selected H counts |", "| --- | ---: | --- |"]
        for name, group in diagnostics["by_progress"].items():
            lines.append(
                f"| {name} | {group['decisions']} | `{json.dumps(group['selected_horizon_counts'], sort_keys=True)}` |"
            )
    lines += ["", "## Improvement directions", ""]
    lines += [f"{index}. {direction}" for index, direction in enumerate(analysis["improvement_directions"], 1)]
    lines += ["", "## Paired state lists", ""]
    for label, field in (("Rescues", "rescue_states"), ("Regressions", "regression_states")):
        states = [(row["task_id"], row["episode"], row["initial_state_id"]) for row in analysis[field]]
        lines += [f"{label} (task, episode, initial_state_id): `{states}`", ""]
    lines += [
        "## Limitations", "",
        "Bootstrap CIs resample tasks and task-by-state clusters; they do not measure across-training-seed variability. "
        "Both-success timing is outcome-selected. Confidence, margin, entropy and long-H fractions first average "
        "within each episode, then weight episodes equally; H counts are call counts. These associations with eventual "
        "episode success are not causal effects. Ordered probabilities describe H selection, not success; "
        "no ECE/Brier/false-long metric is inferred without counterfactual labels. "
        "Complete experiment configuration and bank provenance are retained in analysis.json/source_summary.", "",
    ]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    analysis = analyze(args.eval_dir, samples=args.bootstrap_samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(report_markdown(analysis), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "paired_episodes": analysis["paired"]["paired_episodes"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
