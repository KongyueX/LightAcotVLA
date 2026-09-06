from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_last_block_horizon_smdp_pilot as pilot


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


def test_new_bank_ids_commands_and_client_cpu_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert bank["round_episode_ids"] == [list(range(120, 130)), list(range(130, 140)), list(range(140, 150))]
    assert not bank["validation_or_final_ids_in_training"]
    checkpoint = tmp_path / "checkpoint"
    collect = pilot.build_eval_command(
        args, checkpoint=checkpoint, output_dir=tmp_path / "collection",
        episodes=pilot.TRAIN_EPISODES[0], sample=True,
    )
    validation = pilot.build_eval_command(
        args, checkpoint=checkpoint, output_dir=tmp_path / "validation",
        episodes=pilot.VALIDATION_EPISODES, sample=False,
    )
    assert collect[collect.index("--modes") + 1] == "ordered_smdp_last_block"
    assert collect[collect.index("--last-block-smdp-params") + 1] == str(checkpoint)
    assert "--last-block-smdp-sample" in collect and "--initial-state-bank" in collect
    assert "--last-block-smdp-sample" not in validation and "--initial-state-bank" not in validation
    assert "--ordered-smdp-params" not in collect
    train = pilot.build_train_command(
        args, checkpoint=checkpoint, rollout_dir=tmp_path / "collection", output_dir=tmp_path / "training",
    )
    assert train[2].endswith("train_last_block_horizon_smdp.py")
    assert train[train.index("--critic-warmup-epochs") + 1] == "10"
    assert train[train.index("--epochs") + 1] == "4"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    client_env = pilot._environment(args, trainer=False)
    train_env = pilot._environment(args, trainer=True)
    assert client_env["JAX_PLATFORMS"] == train_env["JAX_PLATFORMS"] == "cpu"
    assert client_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert train_env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("updates", [(False, False, False), (True, False, True)])
def test_three_round_bound_and_unchanged_actor_validation_reuse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, updates: tuple[bool, bool, bool],
) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    calls = []
    monkeypatch.setattr(pilot, "validate_train_bank", lambda _path: {"status": "complete"})

    def read_eval(directory, *, mode, episodes, seed):
        if directory == args.reference_eval_dir:
            successes, rpc = (93, 1.833) if episodes == pilot.VALIDATION_EPISODES else (180, 1.957)
        elif directory.name == "validation":
            successes, rpc = 94, 2.4 if directory.parent.name == "round01" else 2.6
        elif directory.name == "final":
            successes, rpc = 185, 2.0
        else:
            successes, rpc = 80, 2.5
        return _rows(episodes, successes, rpc), {"status": "complete", "config": {"seed": seed}}

    def run_stage(_args, parent, name, command, *, trainer):
        calls.append((parent, name, command, trainer))
        if name == "initialize":
            checkpoint = parent / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "params").mkdir()
            (checkpoint / "metadata.json").write_text("{}")
        elif name == "train":
            training = pathlib.Path(command[command.index("--output-dir") + 1])
            checkpoint = training / "checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "params").mkdir()
            (checkpoint / "metadata.json").write_text("{}")
            changed = updates[int(parent.name[-2:]) - 1]
            (training / "summary.json").write_text(json.dumps({
                "status": "complete", "selector_params": str(checkpoint),
                "actor_changed": changed, "accepted_epochs": 1 if changed else 0,
            }))

    monkeypatch.setattr(pilot.base, "_read_eval", read_eval)
    monkeypatch.setattr(pilot, "_run_stage", run_stage)
    result = pilot._run(args)
    assert len(result["rounds"]) == 3
    assert sum(name == "collect" for _, name, _, _ in calls) == 3
    assert sum(name == "train" for _, name, _, _ in calls) == 3
    assert sum(name == "validate" for _, name, _, _ in calls) == sum(updates)
    assert sum(name == "final_eval" for _, name, _, _ in calls) == int(any(updates))
    assert [row["validation_reused"] for row in result["rounds"]] == [not value for value in updates]
    assert result["rounds"][2]["input_checkpoint"] == result["rounds"][1]["selector_checkpoint"]
    assert (args.output_dir / "round02/validation_reused.json").is_file()
    assert not (args.output_dir / "round02/validation").exists()
    if any(updates):
        assert result["selected_round"] == 1
        assert result["rounds"][1]["validation_source"] == result["rounds"][0]["validation_source"]
        assert result["rounds"][1]["validation_source_checkpoint"] == result["rounds"][0]["selector_checkpoint"]
        assert result["final_vs_A"]["rescues"] == 5
        assert result["final_vs_A"]["regressions"] == 0
    else:
        assert all(row["validation_source"] == str(args.reference_eval_dir) for row in result["rounds"])
        assert result["selected_system"] == "A"
        assert not result["final_run"]


def test_existing_output_is_preserved(tmp_path: pathlib.Path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    marker = args.output_dir / "existing"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        pilot.main(args)
    assert marker.read_text() == "keep"
