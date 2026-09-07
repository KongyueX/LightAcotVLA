# ruff: noqa: SLF001
from __future__ import annotations

import argparse
import csv
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
analyzer = importlib.import_module("analyze_execution_horizon_replanning_traces")


def _trace(*, mode=analyzer.BASE_MODE, decision_steps=(2, 7, 12), horizons=(5, 5, 5), success=True, offsets=None):
    decision_steps = np.asarray(decision_steps, dtype=np.int32)
    horizons = np.asarray(horizons, dtype=np.int32)
    n = len(decision_steps)
    final = int(decision_steps[-1] + horizons[-1])
    state_steps = np.arange(1, final + 1, dtype=np.int32)
    executed_h = np.diff(np.append(decision_steps, final)).astype(np.int32)
    chunks = np.zeros((n, 25, 7), dtype=np.float64)
    chunks[:, :, 0] = (decision_steps[:, None] + np.arange(25)[None, :]) / 100.0
    chunks[:, :, 6] = -1
    if offsets is not None:
        chunks[:, :, 0] += np.asarray(offsets)[:, None]
    decision_index = np.searchsorted(decision_steps, state_steps - 1, side="right") - 1
    executed_actions = np.zeros((len(state_steps), 7))
    executed_actions[:, 6] = -1
    for row, step in enumerate(state_steps):
        index = decision_index[row]
        if index >= 0:
            executed_actions[row] = chunks[index, step - decision_steps[index] - 1]
    positions = np.zeros((len(state_steps), 3))
    positions[:, 0] = state_steps * 0.001
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(state_steps), 1))
    gripper = np.zeros((len(state_steps), 2))
    arrays = {
        "decision_steps": decision_steps, "selected_h": horizons, "execution_h": executed_h,
        "generated_action_chunks": chunks, "decision_physics_state": np.column_stack([decision_steps, decision_steps / 1000]),
        "decision_proprio": np.column_stack([decision_steps / 1000, np.zeros(n)]),
        "decision_eef_pos": np.column_stack([decision_steps / 1000, np.zeros((n, 2))]),
        "decision_eef_quat": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        "decision_gripper_qpos": np.zeros((n, 2)), "request_seeds": decision_steps + 7,
        "state_steps": state_steps, "state_decision_index": decision_index,
        "executed_actions": executed_actions, "eef_pos": positions, "eef_quat": quaternions,
        "gripper_qpos": gripper, "initial_step": np.asarray(0), "initial_eef_pos": np.zeros(3),
        "initial_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]), "initial_gripper_qpos": np.zeros(2),
        "anchor_continuation_logits": np.zeros((n, 4)), "anchor_probabilities": np.full((n, 5), 0.2),
        "ordered_continuation_logits": np.zeros((n, 4)), "ordered_probabilities": np.full((n, 5), 0.2),
    }
    return analyzer.Trace(pathlib.Path(f"/trace/{mode}/task03_ep000339.npz"), arrays,
                          {"mode": mode, "task_id": 3, "episode_id": 339, "success": success})


def _case():
    return {"task_id": 3, "episode_id": 339, "task_name": "test task", "old_relation": "regression",
            "old_success": {analyzer.BASE_MODE: True, analyzer.HISTORY_MODE: False}}


def test_overlapping_plans_align_by_consumed_steps_not_full_chunk():
    report = analyzer.trace_metrics(_trace())
    overlap = report["boundary_summary"]["overlap_translation_command_rmse"]
    assert overlap["count"] == 2
    assert overlap["max"] == pytest.approx(0.0)
    assert report["boundary_summary"]["no_overlap_boundary_count"] == 0
    assert report["episode_metrics"]["boundary_translation_command_delta_mean"] == pytest.approx(0.01)


def test_h25_has_no_overlap_evidence_instead_of_zero_error():
    report = analyzer.trace_metrics(_trace(decision_steps=(2, 27), horizons=(25, 5)))
    overlap = report["boundary_summary"]["overlap_translation_command_rmse"]
    assert overlap["count"] == 0
    assert overlap["mean"] is None
    assert report["boundary_summary"]["no_overlap_boundary_count"] == 1
    assert report["largest_boundary_command_changes"][0]["overlap_steps"] == 0


def test_command_jump_is_separate_from_measured_eef_motion():
    trace = _trace(decision_steps=(2, 7), horizons=(5, 5), offsets=(0, 4))
    trace.arrays["generated_action_chunks"][1, :, 6] = 1
    trace.arrays["executed_actions"][trace.arrays["state_decision_index"] == 1, 6] = 1
    report = analyzer.trace_metrics(trace)
    boundary = report["largest_boundary_command_changes"][0]
    assert boundary["step"] == 7
    assert boundary["translation_command_delta_l2"] == pytest.approx(4.01)
    assert boundary["eef_next_step_displacement_m"] == pytest.approx(0.001)
    assert boundary["gripper_nonzero_sign_flip"]
    assert report["boundary_summary"]["gripper_nonzero_sign_flip_fraction"] == 1.0
    assert "velocity" not in report["episode_metrics"]


def test_different_call_grids_align_on_environment_step_and_preserve_pre_divergence_checks():
    a = _trace()
    history = _trace(mode=analyzer.HISTORY_MODE, decision_steps=(2, 7, 17), horizons=(5, 10, 5))
    report = analyzer.compare_alignment(a, history)
    first = report["first_shared_decision_with_h_difference"]
    assert first["step"] == 7
    assert first["a_selected_h"] == 5
    assert first["history_selected_h"] == 10
    assert report["first_decision_grid_difference_step"] == 12
    prefix = report["before_first_h_or_grid_difference"]
    assert prefix["shared_decisions"] == 2
    assert prefix["matching_request_seeds"] == 2
    assert prefix["generated_chunk_max_abs_difference"]["max"] == 0.0
    assert prefix["executed_command_max_abs_difference"]["max"] == 0.0
    assert prefix["eef_position_difference_m"]["max"] == 0.0
    assert report["full_common_environment_time"]["last_state_step"] == 17


def test_preexisting_plan_difference_is_visible_before_first_h_change():
    a = _trace()
    history = _trace(mode=analyzer.HISTORY_MODE, decision_steps=(2, 7, 17), horizons=(5, 10, 5))
    history.arrays["generated_action_chunks"][0, 24, 0] += 0.3
    history.arrays["decision_physics_state"][0, 1] += 0.02
    report = analyzer.compare_alignment(a, history)
    prefix = report["before_first_h_or_grid_difference"]
    assert prefix["generated_chunk_max_abs_difference"]["max"] == pytest.approx(0.3)
    assert prefix["decision_physics_max_abs_difference"]["max"] == pytest.approx(0.02)
    assert report["first_shared_decision_with_h_difference"]["generated_chunk_max_abs_difference"] == 0.0


def test_quaternion_sign_does_not_create_rotation_difference():
    q = np.asarray([0.0, 0.0, 0.0, 1.0])
    assert analyzer._quaternion_angle(q, -q) == 0.0
    assert analyzer._cosine(np.zeros(3), np.ones(3)) is None


def test_old_discordance_and_trace_reproduction_are_separate():
    report = analyzer.analyze_episode(_case(), _trace(success=True), _trace(mode=analyzer.HISTORY_MODE, success=True))
    assert report["old_relation"] == "regression"
    assert not report["reproduction"]["original_pair_reproduced"]
    assert report["reproduction"]["trace_relation"] == "both_success"
    assert report["reproduction"]["matches_original_by_mode"][analyzer.BASE_MODE]
    assert not report["reproduction"]["matches_original_by_mode"][analyzer.HISTORY_MODE]


def test_aggregate_weights_episode_means_equally():
    first = analyzer.analyze_episode(_case(), _trace(), _trace(mode=analyzer.HISTORY_MODE, success=False))
    second = analyzer.analyze_episode(_case(), _trace(), _trace(mode=analyzer.HISTORY_MODE, success=False))
    for mode in analyzer.MODES:
        first["traces"][mode]["episode_metrics"]["boundary_translation_command_delta_mean"] = 1.0
        first["traces"][mode]["boundary_count"] = 1
        second["traces"][mode]["episode_metrics"]["boundary_translation_command_delta_mean"] = 9.0
        second["traces"][mode]["boundary_count"] = 100
    summary = analyzer.aggregate_episodes([first, second])
    group = summary["original_discordance_groups"]["regression"]
    assert group["metrics"][analyzer.BASE_MODE]["boundary_translation_command_delta_mean"]["mean"] == 5.0
    assert summary["original_pairs_reproduced"] == 2


def test_cli_writes_per_episode_reports_and_reproduction_summary(tmp_path):
    trace_dir = tmp_path / "traces"
    for trace in (_trace(), _trace(mode=analyzer.HISTORY_MODE, success=False)):
        directory = trace_dir / trace.metadata["mode"]
        directory.mkdir(parents=True)
        path = directory / "task03_ep000339.npz"
        np.savez_compressed(path, **trace.arrays)
        path.with_suffix(".json").write_text(json.dumps(trace.metadata))
    rows = tmp_path / "rows.csv"
    with rows.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "task_id", "episode", "initial_state_id", "success", "task_name"])
        writer.writeheader()
        for mode, success in ((analyzer.BASE_MODE, 1), (analyzer.HISTORY_MODE, 0)):
            writer.writerow({"mode": mode, "task_id": 3, "episode": 339, "initial_state_id": 339,
                             "success": success, "task_name": "test task"})
    output = tmp_path / "analysis"
    analyzer.main(argparse.Namespace(trace_dir=trace_dir, baseline_rows=rows, output_dir=output))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["episode_count"] == 1
    assert summary["original_pairs_reproduced"] == 1
    assert (output / "episodes" / "task03_ep000339.json").is_file()
    assert "单个跳变" in (output / "episodes" / "task03_ep000339.md").read_text()
    assert "Episode等权" in (output / "report.md").read_text()


def test_runtime_error_trace_is_not_interpreted_as_a_task_failure(tmp_path):
    trace = _trace(success=False)
    trace.metadata["error"] = "RuntimeError: policy unavailable"
    path = tmp_path / "task03_ep000339.npz"
    np.savez_compressed(path, **trace.arrays)
    path.with_suffix(".json").write_text(json.dumps(trace.metadata))
    with pytest.raises(ValueError, match="execution error"):
        analyzer.load_trace(path, mode=analyzer.BASE_MODE, task_id=3, episode_id=339)
