import argparse
import csv
import json

import run_execution_horizon_replanning_diagnosis as runner


def test_selects_only_discordant_pairs_and_preserves_relative_order(tmp_path):
    config = {"task_start": 0, "episode_ids": [336, 337], "interleave_modes": True}
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    (tmp_path / "summary.json").write_text(json.dumps({"status": "complete"}))
    rows = []
    for task, episode, anchor, history in ((0, 336, 1, 1), (0, 337, 1, 0), (1, 336, 0, 1)):
        for mode, success in zip(runner.MODES, (anchor, history), strict=True):
            rows.append({"task_id": task, "episode": episode, "initial_state_id": episode, "mode": mode, "success": success})
    with (tmp_path / "rollout_rows.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, cases = runner.select_cases(tmp_path)
    assert len(cases) == 2
    assert cases[0]["category"] == "regression"
    assert cases[0]["mode_order"] == list(reversed(runner.MODES))
    assert cases[1]["category"] == "rescue"
    assert cases[1]["mode_order"] == list(runner.MODES)


def test_trace_replay_uses_reference_protocol_and_creates_journal_first(tmp_path):
    args = argparse.Namespace(python="python", code_dir=tmp_path, output_dir=tmp_path)
    config = {"seed": 7, "initial_state_bank": "/bank", "model_action_horizon": 25,
              "feedback_history_params": "/history.npz", "final_denoising_steps": 10}
    case = {"task_id": 8, "episode": 343, "mode_order": list(runner.MODES)}
    command = runner.replay_command(args, config, case)
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--feedback-history-params") + 1] == "/history.npz"
    assert command[command.index("--initial-state-bank") + 1] == "/bank"
    assert "--resume" not in command
    journal = tmp_path / "replays/task08_ep000343/run_config.json"
    journal.parent.mkdir(parents=True)
    journal.write_text("{}")
    assert "--resume" in runner.replay_command(args, config, case)
