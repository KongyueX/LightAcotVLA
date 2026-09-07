import argparse
from fractions import Fraction

import numpy as np
import run_execution_horizon_greedy_selection as runner


def test_replays_original_update_count_and_preserves_optimization_inputs(tmp_path):
    args = argparse.Namespace(python="python", code_dir=tmp_path, output_dir=tmp_path)
    source = {"status": "complete", "variant": "history", "updates": 650, "args": {
        "variant": "history", "seed": 7, "learning_rate": 1e-4, "batch_size": 64,
        "train_dir": "/old/train", "validation_dir": "/old/early", "a_predictor_dir": "/A",
        "output_dir": "/old/output", "max_updates": 2000, "patience": 8, "log_every": 25,
    }}
    command = runner.train_command(args, source, "history")
    for option, expected in (("--max-updates", "650"), ("--patience", "1000"),
                             ("--selection-metric", "greedy"), ("--train-dir", "/old/train"),
                             ("--learning-rate", "0.0001"), ("--seed", "7")):
        assert command[command.index(option) + 1] == expected


def test_success_fraction_precedes_rpc_and_keeps_exact_ties():
    def summary(numerator, denominator, rpc):
        return {"best_greedy_validation": {
            "greedy_success_delta_fraction": {"numerator": numerator, "denominator": denominator},
            "greedy_rpc_delta_seconds": rpc,
        }}
    assert runner.greedy_score(summary(1, 300, 0.1)) > runner.greedy_score(summary(0, 1, -2.0))
    assert runner.greedy_score(summary(1, 300, -0.02)) == runner.greedy_score(summary(2, 600, -0.02))
    assert runner.greedy_score(summary(0, 1, 0)) == (Fraction(0), 0.0)


def test_evaluation_reuses_the_existing_bank_and_runs_only_one_candidate(tmp_path):
    args = argparse.Namespace(python="python", code_dir=tmp_path, output_dir=tmp_path / "new",
                              source_dir=tmp_path / "old", host="localhost", port=8040)
    command = runner.evaluation_command(args, "development", (336, 337), "history")
    assert command[command.index("--initial-state-bank") + 1] == str(tmp_path / "old/initial_state_bank_0_365")
    start = command.index("--modes") + 1
    assert command[start:start + 2] == ["ordered_transformer", "ordered_feedback_history"]
    assert "--resume" not in command


def test_equivalent_weights_ignore_report_metadata_and_detect_parameter_changes(tmp_path):
    model = runner.FeedbackSelector.initialize(np.zeros((2048, 256)), np.zeros(256), variant="history")
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    model.save(left)
    model.metadata["selection_metric"] = "greedy"
    model.save(right)
    assert runner.same_effective_model(left, right)
    model.params["output_b"][0] = 0.1
    model.save(right)
    assert not runner.same_effective_model(left, right)
