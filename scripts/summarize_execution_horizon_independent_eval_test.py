from __future__ import annotations

import csv
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import summarize_execution_horizon_independent_eval as summarizer


def _synthetic() -> tuple[dict, list[dict], list[dict]]:
    summary = {
        "status": "complete",
        "num_tasks": 2,
        "num_trials_per_task": 2,
        "episode_ids": [100, 101],
        "config": {
            "task_start": 0, "original_horizon": 5, "host": "127.0.0.1", "port": 8040,
            "model_action_horizon": 25,
        },
        "overall": {mode: {"episodes": 4} for mode in summarizer.MODES},
        "initial_state_bank": "/synthetic/bank",
        "initial_state_identity_mode": "synthetic",
    }
    rows = []
    decisions = []
    for index, (task, episode) in enumerate(((0, 100), (0, 101), (1, 100), (1, 101))):
        for mode, outcomes in ((summarizer.REFERENCE, (1, 1, 0, 1)), (summarizer.CANDIDATE, (1, 0, 1, 1))):
            reference = mode == summarizer.REFERENCE
            rows.append({
                "mode": mode,
                "task_id": str(task),
                "episode": str(episode),
                "initial_state_id": str(episode + 1000),
                "success": str(outcomes[index]),
                "timeout": str(1 - outcomes[index]),
                "policy_calls": "4" if reference else "2",
                "actual_policy_total_ms": "400" if reference else "200",
                "policy_rpc_wall_total_ms": "480" if reference else "240",
                "actual_episode_elapsed_total_ms": "1000" if reference else "750",
                "actual_predictor_total_ms": "0" if reference else "6",
            })
        for call in range(2):
            info = {
                "candidate_horizons": [5, 10, 15, 20, 25],
                "ordered_horizon_probability": [.05, .05, .1, .2, .6] if call == 0 else [.6, .2, .1, .05, .05],
                "ordered_continuation_logits": [0.1, 0.2, 0.3, 0.4],
                "previous_horizon": 5 if call == 0 else 20,
                "episode_progress": index / 4 if call == 0 else 1.0,
                "budget_fraction": 0.5,
            }
            decisions.append({
                "mode": summarizer.CANDIDATE,
                "task_id": str(task),
                "episode": str(episode),
                "initial_state_id": str(episode + 1000),
                "environment_step": str(10 + 20 * call),
                "selected_horizon": "25" if call == 0 else "5",
                "execution_horizon": "20" if call == 0 else "5",
                "selector_json": json.dumps(info),
            })
    return summary, rows, decisions


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("separate_original", [False, True])
def test_complete_paired_analysis_and_ordered_diagnostics(
    tmp_path: pathlib.Path, separate_original: bool
) -> None:
    summary, rows, decisions = _synthetic()
    if separate_original:
        summary["config"].update(original_port=8041, original_model_action_horizon=10)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    _write_csv(tmp_path / "rollout_rows.csv", rows)
    _write_csv(tmp_path / "decisions.csv", decisions)
    analysis = summarizer.analyze(tmp_path, samples=20, seed=7)

    assert analysis["paired"]["paired_episodes"] == 4
    assert analysis["paired"]["rescues"] == analysis["paired"]["regressions"] == 1
    assert analysis["both_success"]["episodes"] == 2
    assert analysis["rescue_states"] == [{"task_id": 1, "episode": 100, "initial_state_id": 1100}]
    assert analysis["regression_states"] == [{"task_id": 0, "episode": 101, "initial_state_id": 1101}]
    assert analysis["per_task"]["0"]["runs"][summarizer.CANDIDATE]["means"]["policy_calls"] == 2
    assert analysis["paired"]["metric_delta_candidate_minus_reference"]["policy_rpc_wall_total_ms"]["ci95"] == [-240, -240]
    assert analysis["runs"][summarizer.CANDIDATE]["predictor_overhead"]["mean_ms_per_call"] == 3
    assert analysis["source_summary"]["initial_state_bank"] == "/synthetic/bank"
    diagnostics = analysis["decisions"]
    assert diagnostics["status"] == "complete"
    assert diagnostics["selected_horizon_counts"] == {5: 4, 25: 4}
    assert diagnostics["execution_horizon_counts"] == {5: 4, 20: 4}
    assert diagnostics["previous_execution_to_selected_horizon_counts"] == {"5->25": 4, "20->5": 4}
    assert diagnostics["by_episode_outcome"]["failure"]["decisions"] == 2
    assert diagnostics["by_progress"]["[0.75,1]"]["decisions"] == 5
    assert diagnostics["by_selected_horizon"]["25"]["failure"]["selected_vs_best_other_margin"]["mean"] == pytest.approx(.4)
    assert diagnostics["entropy_nats"]["mean"] > 0
    report = summarizer.report_markdown(analysis)
    assert "All episodes" in report and "paired 95% CI" in report and "Both-success" in report
    assert "Selected H:" in report and "Execution H" in report
    systems = analysis["systems"]
    assert systems[summarizer.CANDIDATE]["model_action_horizon"] == 25
    if separate_original:
        assert systems[summarizer.REFERENCE]["label"] == "Original ACoT-VLA"
        assert systems[summarizer.REFERENCE]["port"] == 8041
        assert systems[summarizer.REFERENCE]["model_action_horizon"] == 10
        assert "New H25+predictor" in report
        assert "Fixed H5" not in report
        assert "systems as a whole" in report
        assert "50999" not in report
    else:
        assert systems[summarizer.REFERENCE]["label"] == "Same-policy Fixed H5"
        assert systems[summarizer.REFERENCE]["model_action_horizon"] == 25
        assert "Same-policy Fixed H5" in report


def test_pairing_and_complete_grid_must_match_summary() -> None:
    summary, rows, _ = _synthetic()
    rows[1]["initial_state_id"] = "9999"
    with pytest.raises(ValueError, match="mismatched initial_state_id"):
        summarizer._split_rollouts(rows, summary)
    summary, rows, _ = _synthetic()
    with pytest.raises(ValueError, match="expected task/episode grid"):
        summarizer._split_rollouts(rows[:-2], summary)
    summary["status"] = "running"
    with pytest.raises(ValueError, match="not complete"):
        summarizer._split_rollouts(rows, summary)


def test_missing_diagnostics_are_not_inferred() -> None:
    summary, rows, decisions = _synthetic()
    runs, _ = summarizer._split_rollouts(rows, summary)
    for row in decisions:
        row["selector_json"] = "{}"
    diagnostics = summarizer._decision_diagnostics(decisions, runs[summarizer.CANDIDATE])
    assert diagnostics["status"] == "missing_or_partial"
    assert diagnostics["missing_fields"]["ordered_horizon_probability_or_candidates"] == 8
    assert diagnostics["entropy_nats"] == {"count": 0, "episodes": 0, "mean": None}
    assert diagnostics["previous_execution_to_selected_horizon_counts"] == {}
    assert summarizer._decision_diagnostics(None, runs[summarizer.CANDIDATE])["status"] == "missing"


def test_confidence_is_episode_balanced_not_call_weighted() -> None:
    items = [
        {"key": (0, 0), "selected_horizon": 25, "execution_horizon": 25, "entropy_nats": 0.1},
        *[
            {"key": (0, 1), "selected_horizon": 5, "execution_horizon": 5, "entropy_nats": 0.9}
            for _ in range(9)
        ],
    ]
    group = summarizer._decision_group(items)
    assert group["entropy_nats"]["mean"] == pytest.approx(0.5)
    assert group["entropy_nats"]["count"] == 10
    assert group["entropy_nats"]["episodes"] == 2
    assert group["long_h_fraction"] == pytest.approx(0.5)


def test_historical_reference_filters_other_modes_and_preserves_two_runs(tmp_path: pathlib.Path) -> None:
    current_dir = tmp_path / "current"
    reference_dir = tmp_path / "historical"
    current_dir.mkdir()
    reference_dir.mkdir()
    summary, rows, decisions = _synthetic()
    summary["episode_ids"] = [0, 1]
    summary["task_suite"] = "libero_10"
    summary["config"].update(seed=7, action_cot_denoising_steps=10, final_denoising_steps=10, num_steps_wait=10)
    for row in [*rows, *decisions]:
        row["episode"] = str(int(row["episode"]) - 100)
    current_summary = json.loads(json.dumps(summary))
    current_summary["overall"] = {summarizer.CANDIDATE: {"episodes": 4}}
    historical_summary = json.loads(json.dumps(summary))
    del historical_summary["episode_ids"]
    historical_summary["overall"] = {summarizer.REFERENCE: {"episodes": 4}, "fixed_h9": {"episodes": 4}}
    historical_summary["config"].update(port=8000, model_action_horizon=10)
    del historical_summary["config"]["final_denoising_steps"]
    original_rows = [row for row in rows if row["mode"] == summarizer.REFERENCE]
    _write_csv(reference_dir / "rollout_rows.csv", [*original_rows, *[{**row, "mode": "fixed_h9"} for row in original_rows]])
    _write_csv(current_dir / "rollout_rows.csv", [row for row in rows if row["mode"] == summarizer.CANDIDATE])
    _write_csv(current_dir / "decisions.csv", decisions)
    (reference_dir / "summary.json").write_text(json.dumps(historical_summary))
    (current_dir / "summary.json").write_text(json.dumps(current_summary))

    analysis = summarizer.analyze(current_dir, samples=20, seed=7, reference_eval_dir=reference_dir)
    assert analysis["paired"]["paired_episodes"] == 4
    assert analysis["paired"]["rescues"] == analysis["paired"]["regressions"] == 1
    assert analysis["reference_eval_dir"] == str(reference_dir.resolve())
    assert analysis["reference_source_summary"] == historical_summary
    assert analysis["source_summary"] == current_summary
    assert analysis["reference_protocol_comparison"]["matched_recorded_fields"]["seed"] == 7
    assert analysis["reference_protocol_comparison"]["unavailable_recorded_fields"]["final_denoising_steps"] == {
        "reference": None, "current": 10,
    }
    assert analysis["systems"][summarizer.REFERENCE]["label"] == "Original ACoT-VLA (historical)"
    assert analysis["systems"][summarizer.REFERENCE]["model_action_horizon"] == 10
    assert analysis["systems"][summarizer.CANDIDATE]["label"] == "Current H25+predictor"
    assert analysis["paired"]["gates"]["strict_engineering_go"] is None
    report = summarizer.report_markdown(analysis)
    assert "cross-run historical time reference" in report
    assert "hardware/software/load differences" in report
    assert "Only this bank/run is compared" not in report

    current_summary["config"]["seed"] = 42
    with pytest.raises(ValueError, match="protocol mismatch for seed"):
        summarizer._historical_protocol(historical_summary, current_summary)
