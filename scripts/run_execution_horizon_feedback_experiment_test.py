import argparse
import pathlib
from types import SimpleNamespace

import pytest
import run_execution_horizon_feedback_experiment as runner


def test_diagnostic_roots_are_part_of_disjoint_training_split():
    assert sum(len(runner.diagnostic_episodes(task)) for task in range(10)) == 40
    assert all(set(runner.diagnostic_episodes(task)) <= set(runner.TRAIN_EPISODES) for task in range(10))
    roles = [runner.TRAIN_EPISODES, runner.EARLY_EPISODES, runner.DEV_EPISODES, runner.FINAL_EPISODES]
    assert [10 * len(role) for role in roles] == [300, 60, 100, 200]
    combined = [episode for role in roles for episode in role]
    assert len(set(combined)) == len(combined)
    assert min(combined) == 300


def test_eval_command_interleaves_current_a_and_both_residuals():
    args = argparse.Namespace(python="python", code_dir=pathlib.Path("/code"), output_dir=pathlib.Path("/out"), host="localhost", port=8040)
    command = runner.eval_command(args, "development", runner.DEV_EPISODES, ["current", "history"])
    start = command.index("--modes") + 1
    assert command[start:start + 3] == ["ordered_transformer", "ordered_feedback_current", "ordered_feedback_history"]
    assert "--interleave-modes" in command
    assert command[command.index("--feedback-history-params") + 1] == "/out/training_history/checkpoint.npz"
    assert command[command.index("--final-denoising-steps") + 1] == "10"


def test_candidate_selection_keeps_a_on_tie_and_prioritizes_success():
    def row(success, rpc):
        return {"success_count": success, "means": {"policy_rpc_wall_total_ms": rpc}}
    runs = {runner.ANCHOR: row(93, 1900), "ordered_feedback_current": row(92, 1700), "ordered_feedback_history": row(93, 1900)}
    assert runner.select_development_candidate({"runs": runs}) is None
    runs["ordered_feedback_history"] = row(94, 2100)
    assert runner.select_development_candidate({"runs": runs}) == "history"


def test_completed_training_recovers_missing_exit_without_training(tmp_path, monkeypatch):
    destination = tmp_path / "training_current"
    runner.write_json(destination / "summary.json", {"status": "complete"})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("completed stage must not rerun"))
    runner.run_stage(SimpleNamespace(output_dir=tmp_path, code_dir=tmp_path), "train_current", ["python", "--output-dir", str(destination)])
    assert (tmp_path / "train_current.exit").read_text().strip() == "0"


def test_incomplete_training_is_preserved_before_new_attempt(tmp_path, monkeypatch):
    destination = tmp_path / "training_history"
    destination.mkdir()
    (destination / "training_log.jsonl").write_text("partial\n")
    def start(*args, **kwargs):
        assert not destination.exists()
        assert len(list(tmp_path.glob("training_history.interrupted_*"))) == 1
        return SimpleNamespace(pid=99999999, wait=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    runner.run_stage(SimpleNamespace(output_dir=tmp_path, code_dir=tmp_path), "train_history", ["python", "--output-dir", str(destination)])
    assert (tmp_path / "train_history.exit").read_text().strip() == "0"


def test_completed_run_retries_notification(tmp_path, monkeypatch):
    runner.write_json(tmp_path / "summary.json", {"status": "complete"})
    calls = []
    monkeypatch.setattr(runner, "notify_completion", lambda output, summary: calls.append(summary))
    runner.main(SimpleNamespace(output_dir=tmp_path, code_dir=tmp_path))
    assert calls == [{"status": "complete"}]


def test_failure_updates_visible_stage_status(tmp_path, monkeypatch):
    def fail(args):
        runner.write_json(tmp_path / "status.json", {"status": "running", "stage": "development"})
        raise RuntimeError("evaluation interrupted")
    monkeypatch.setattr(runner, "execute", fail)
    monkeypatch.setattr(runner, "notify", lambda *args: None)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.main(SimpleNamespace(output_dir=tmp_path, code_dir=tmp_path))
    status = runner.json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "development"
