from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_ordered_horizon_smdp_pilot as pilot


def _args(tmp_path: pathlib.Path):
    return pilot.build_parser().parse_args([
        "--output-dir", str(tmp_path / "pilot"), "--code-dir", str(tmp_path / "code"),
        "--train-state-bank", str(tmp_path / "bank"), "--reference-eval-dir", str(tmp_path / "reference"),
        "--anchor-predictor-dir", str(tmp_path / "anchor"),
    ])


def _rows(episodes: tuple[int, ...], successes: int, rpc_seconds: float) -> dict:
    result = {}
    for index, (task, episode) in enumerate((task, episode) for task in range(10) for episode in episodes):
        result[(task, episode)] = {
            "task_id": str(task), "episode": str(episode), "initial_state_id": str(episode),
            "success": str(int(index < successes)), "policy_calls": "20", "actual_policy_total_ms": "1000",
            "policy_rpc_wall_total_ms": str(rpc_seconds * 1000), "actual_episode_elapsed_total_ms": "10000",
            "h_distribution_json": '{"10":20}',
        }
    return result


def test_bank_and_commands_keep_train_separate_from_validation(tmp_path: pathlib.Path) -> None:
    args = _args(tmp_path)
    args.train_state_bank.mkdir()
    tasks = []
    for task in range(10):
        file = f"task{task:02d}.npz"
        np.savez(args.train_state_bank / file, episode_ids=np.arange(300))
        tasks.append({"task_id": task, "file": file})
    (args.train_state_bank / "manifest.json").write_text(json.dumps({
        "status": "complete", "task_suite": "libero_10", "tasks": tasks,
    }))
    bank = pilot.validate_train_bank(args.train_state_bank)
    assert not bank["validation_or_final_ids_in_training"]
    assert bank["available_ids_per_task"] == {str(task): 300 for task in range(10)}
    checkpoint = tmp_path / "selector.npz"
    train = pilot.build_eval_command(
        args, checkpoint=checkpoint, output_dir=tmp_path / "train", episodes=pilot.TRAIN_EPISODES[0], sample=True,
    )
    validation = pilot.build_eval_command(
        args, checkpoint=checkpoint, output_dir=tmp_path / "val", episodes=pilot.VALIDATION_EPISODES, sample=False,
    )
    assert "--ordered-smdp-sample" in train and "--initial-state-bank" in train
    assert "--record-ordered-diagnostics" in train
    assert "--ordered-smdp-sample" not in validation and "--initial-state-bank" not in validation
    assert train[train.index("--ordered-smdp-params") + 1] == str(checkpoint)
    assert train[train.index("--episode-ids") + 1:train.index("--initial-state-offset")] == [str(x) for x in range(100, 110)]
    command = pilot.build_train_command(args, checkpoint=checkpoint, rollout_dir=tmp_path / "train", output_dir=tmp_path / "trained")
    assert command[command.index("--input-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--epochs") + 1] == "4"
    env = pilot._environment(args, cpu=True)
    assert env["JAX_PLATFORMS"] == "cpu" and env["CUDA_VISIBLE_DEVICES"] == ""


def test_validation_criterion_is_lexicographic_with_rpc_cap() -> None:
    baseline = {"success_count": 93, "mean_rpc_seconds": 1.833}
    assert pilot.eligible({"success_count": 94, "mean_rpc_seconds": 2.9}, baseline)
    assert pilot.eligible({"success_count": 93, "mean_rpc_seconds": 1.7}, baseline)
    assert not pilot.eligible({"success_count": 94, "mean_rpc_seconds": 3.01}, baseline)
    assert not pilot.eligible({"success_count": 92, "mean_rpc_seconds": 1.0}, baseline)
    assert not pilot.eligible(dict(baseline), baseline)


@pytest.mark.parametrize("has_winner", [False, True])
def test_two_round_bound_and_single_conditional_final(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, has_winner: bool,
) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    calls = []
    monkeypatch.setattr(pilot, "validate_train_bank", lambda _path: {"status": "complete"})

    def read_eval(directory, *, mode, episodes, seed):
        if directory == args.reference_eval_dir:
            successes, rpc = (93, 1.833) if episodes == pilot.VALIDATION_EPISODES else (180, 1.957)
        elif directory.name == "validation":
            if directory.parent.name == "round01":
                successes, rpc = (93, 1.7) if has_winner else (93, 1.833)
            else:
                successes, rpc = (94, 2.4) if has_winner else (94, 3.1)
        elif directory.name == "final":
            successes, rpc = 185, 2.0
        else:
            successes, rpc = 80, 2.5
        return _rows(episodes, successes, rpc), {"status": "complete", "config": {"seed": seed}}

    def run_stage(_args, parent, name, command, *, cpu):
        calls.append((parent, name, command, cpu))
        if name == "initialize":
            (parent / "selector.npz").write_bytes(b"test")
        if name == "train":
            training = pathlib.Path(command[command.index("--output-dir") + 1])
            training.mkdir()
            (training / "selector.npz").write_bytes(b"test")
            (training / "summary.json").write_text('{"status":"complete"}')

    monkeypatch.setattr(pilot, "_read_eval", read_eval)
    monkeypatch.setattr(pilot, "_run_stage", run_stage)
    result = pilot._run(args)
    assert len(result["rounds"]) == 2
    assert [name for _, name, _, _ in calls].count("collect") == 2
    assert [name for _, name, _, _ in calls].count("train") == 2
    assert [name for _, name, _, _ in calls].count("validate") == 2
    assert [name for _, name, _, _ in calls].count("final_eval") == int(has_winner)
    assert all(cpu for _, name, _, cpu in calls if name in ("initialize", "train"))
    assert result["rounds"][1]["input_checkpoint"] == result["rounds"][0]["selector_checkpoint"]
    if has_winner:
        assert result["selected_round"] == 2
        assert result["final"]["success_count"] == 185
        assert result["final_vs_A"]["rescues"] == 5
        assert result["final_vs_A"]["regressions"] == 0
    else:
        assert result["selected_system"] == "A"
        assert not result["final_run"]
        assert not (args.output_dir / "final").exists()


def test_existing_output_is_not_overwritten(tmp_path: pathlib.Path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    marker = args.output_dir / "existing"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        pilot.main(args)
    assert marker.read_text() == "keep"
