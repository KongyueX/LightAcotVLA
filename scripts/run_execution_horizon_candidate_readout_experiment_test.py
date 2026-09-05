from __future__ import annotations

import pathlib

import run_execution_horizon_candidate_readout_experiment as candidate
import run_execution_horizon_dynamic_finetune as dynamic


def test_candidate_readout_keeps_dynamic_baseline_training_protocol() -> None:
    arguments = {
        "python": pathlib.Path("/venv/python"),
        "train_script": pathlib.Path("/code/train_execution_horizon_predictor.py"),
        "data_dirs": [pathlib.Path("/round4/task0"), pathlib.Path("/round4/task1")],
        "output_dir": pathlib.Path("/candidate/predictor"),
        "resume_params": pathlib.Path("/theta0/params"),
        "split_manifest": pathlib.Path("/round4/split_manifest.json"),
    }
    baseline = dynamic.build_train_command(**arguments)
    default_command = candidate.build_train_command(**arguments)
    assert default_command == [*baseline, "--ordered-readout", "candidate", "--resume-candidate-readout"]
    assert default_command[default_command.index("--temporal-layers") + 1] == "2"

    deeper = candidate.build_train_command(**arguments, temporal_layers=4)
    expected = list(default_command)
    expected[expected.index("--temporal-layers") + 1] = "4"
    assert deeper == expected

    residual = candidate.build_train_command(**arguments, ordered_readout="candidate_residual")
    assert residual == [
        *baseline, "--ordered-readout", "candidate_residual", "--resume-candidate-readout",
        "--train-candidate-readout-only",
    ]
