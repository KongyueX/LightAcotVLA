from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_fixed_cost_horizon_smdp_pilot as pilot


def _args(tmp_path: pathlib.Path):
    return pilot.build_parser().parse_args([
        "--output-dir", str(tmp_path / "new"), "--code-dir", str(tmp_path / "code"),
        "--train-state-bank", str(tmp_path / "bank"), "--reference-eval-dir", str(tmp_path / "reference"),
        "--anchor-predictor-dir", str(tmp_path / "anchor"),
        "--reuse-first-round-dir", str(tmp_path / "old/round01"),
        "--original-reference-eval-dir", str(tmp_path / "original"),
    ])


def _rows(episodes: tuple[int, ...], successes: int, rpc: float) -> dict:
    return {
        (task, episode): {
            "success": str(int(index < successes)), "initial_state_id": str(episode),
            "policy_calls": "20", "actual_policy_total_ms": "1000",
            "policy_rpc_wall_total_ms": str(rpc * 1000), "actual_episode_elapsed_total_ms": "10000",
        }
        for index, (task, episode) in enumerate((task, episode) for task in range(10) for episode in episodes)
    }


def test_train_command_changes_only_dual_rate_and_requires_positive_source_cost(tmp_path: pathlib.Path) -> None:
    args = _args(tmp_path)
    checkpoint, collection, output = tmp_path / "checkpoint", tmp_path / "collection", tmp_path / "training"
    normal = pilot.last_block.build_train_command(
        args, checkpoint=checkpoint, rollout_dir=collection, output_dir=output,
    )
    fixed = pilot.build_train_command(args, checkpoint=checkpoint, rollout_dir=collection, output_dir=output)
    assert fixed == [*normal, "--dual-learning-rate", "0"]
    assert fixed[fixed.index("--epochs") + 1] == "4"
    assert fixed[fixed.index("--critic-warmup-epochs") + 1] == "10"
    checkpoint.mkdir()
    metadata = checkpoint / "metadata.json"
    metadata.write_text('{"cost_multiplier":0.02}')
    pilot._check_multiplier(checkpoint)
    metadata.write_text('{"cost_multiplier":0}')
    with pytest.raises(ValueError, match="cost_multiplier=0.02"):
        pilot._check_multiplier(checkpoint)


@pytest.mark.parametrize("outcome", ["unchanged", "not_better", "winner"])
def test_one_update_no_collection_and_conditional_final(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    checkpoint = tmp_path / "old/step0/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text('{"cost_multiplier":0.02}')
    collection = args.reuse_first_round_dir / "collection"
    collection.mkdir(parents=True)
    old_log = args.reuse_first_round_dir / "train.log"
    old_log.write_text("keep old training evidence")
    monkeypatch.setattr(pilot.last_block, "_reuse_first_round", lambda _args: (
        checkpoint, collection, _rows(pilot.last_block.TRAIN_EPISODES[0], 87, 2.0), {"status": "complete"},
    ))
    calls = []

    def run_stage(_args, parent, name, command, *, trainer):
        calls.append(name)
        if name == "train":
            assert trainer
            assert command[command.index("--input-checkpoint") + 1] == str(checkpoint)
            assert command[command.index("--rollout-dir") + 1] == str(collection)
            training_dir = pathlib.Path(command[command.index("--output-dir") + 1])
            output_checkpoint = training_dir / "checkpoint"
            (output_checkpoint / "params").mkdir(parents=True)
            (output_checkpoint / "metadata.json").write_text('{"cost_multiplier":0.02}')
            (training_dir / "summary.json").write_text(json.dumps({
                "status": "complete", "selector_params": str(output_checkpoint),
                "actor_changed": outcome != "unchanged", "accepted_epochs": 0 if outcome == "unchanged" else 4,
                "cost_multiplier_before": 0.02, "cost_multiplier_after": 0.02,
            }))
        else:
            assert "--last-block-smdp-sample" not in command
            assert "--initial-state-bank" not in command

    def read_eval(directory, *, mode, episodes, seed):
        if directory == args.reference_eval_dir:
            successes, rpc = (93, 1.833) if episodes == pilot.last_block.VALIDATION_EPISODES else (180, 1.957)
        elif directory == args.original_reference_eval_dir:
            assert mode == "original"
            successes, rpc = 187, 6.073
        elif directory.name == "validation":
            successes, rpc = (94, 2.0) if outcome == "winner" else (92, 2.0)
        else:
            successes, rpc = 185, 2.0
        return _rows(episodes, successes, rpc), {"status": "complete"}

    monkeypatch.setattr(pilot.last_block, "_run_stage", run_stage)
    monkeypatch.setattr(pilot.base, "_read_eval", read_eval)
    result = pilot._run(args)
    assert calls == {
        "unchanged": ["train"], "not_better": ["train", "validate"],
        "winner": ["train", "validate", "final_eval"],
    }[outcome]
    assert result["maximum_updates"] == 1 and result["no_collection"]
    assert result["final_run"] == (outcome == "winner")
    assert old_log.read_text() == "keep old training evidence"
    if outcome == "winner":
        assert result["final_vs_A"]["rescues"] == 5
        assert result["final_vs_historical_original"]["regressions"] == 2
    else:
        assert result["selected_system"] == "A"
        assert not (args.output_dir / "final").exists()
