from __future__ import annotations

import json
import pathlib

import pytest

import run_execution_horizon_dynamic_finetune as dynamic


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _collection(tmp_path: pathlib.Path, *, status: str = "complete") -> pathlib.Path:
    data_dirs = []
    for index in range(2):
        data_dir = tmp_path / f"dynamic-{index}"
        data_dir.mkdir()
        data_dirs.append(str(data_dir))
    manifest = tmp_path / "split_manifest.json"
    start = 0
    split = {}
    for field, count in dynamic.EXPECTED_SPLITS.items():
        split[field] = list(range(start, start + count))
        start += count
    _write_json(manifest, split)
    summary = tmp_path / "collection_summary.json"
    _write_json(
        summary,
        {
            "status": status,
            "num_roots": 180,
            "data_dirs": data_dirs,
            "split_manifest": str(manifest),
        },
    )
    return summary


def test_dynamic_collection_builds_only_requested_finetune_command(tmp_path: pathlib.Path) -> None:
    summary_path = _collection(tmp_path)
    _, manifest, data_dirs = dynamic.load_collection(summary_path)
    command = dynamic.build_train_command(
        python=pathlib.Path("/venv/python"),
        train_script=pathlib.Path("/code/train_execution_horizon_predictor.py"),
        data_dirs=data_dirs,
        output_dir=pathlib.Path("/output"),
        resume_params=pathlib.Path("/resume/params"),
        split_manifest=manifest,
    )

    dataset_end = command.index("--output-dir")
    assert command[command.index("--dataset") + 1 : dataset_end] == [str(path) for path in data_dirs]
    expected = {
        "--train-steps": "5000",
        "--batch-size": "128",
        "--learning-rate": "1e-4",
        "--log-interval": "25",
        "--checkpoint-interval": "0",
        "--early-stopping-patience-logs": "8",
        "--loss-ordered-listwise": "2",
        "--loss-danger-rescue": "1",
        "--loss-paired-elapsed": "0.5",
        "--loss-success": "0.5",
        "--loss-raw-h-classification": "0",
        "--loss-raw-h-ordinal": "0",
        "--loss-survival": "0",
        "--loss-final-risk": "0",
        "--loss-calls-advantage": "0",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value
    for flag in ("--paired-distribution-heads", "--ordered-continuation-head"):
        assert flag in command
    assert "--ordered-listwise-elapsed-mode" not in command
    assert "--ordered-listwise-elapsed-floor-seconds" not in command


def test_dynamic_paired_noise_cli_changes_only_elapsed_target_flags() -> None:
    required = [
        "--collection-summary", "/round5/summary.json", "--resume-predictor-dir", "/A",
        "--output-dir", "/output", "--policy-server-tmux", "h25_server",
    ]
    default = dynamic.build_parser().parse_args(required)
    assert default.ordered_listwise_elapsed_mode == "root_minmax"
    assert default.ordered_listwise_elapsed_floor_seconds == 1.0
    args = dynamic.build_parser().parse_args([
        *required, "--ordered-listwise-elapsed-mode", "paired_noise",
        "--ordered-listwise-elapsed-floor-seconds", "1.0",
    ])
    common = {
        "python": pathlib.Path("/python"), "train_script": pathlib.Path("/train.py"),
        "data_dirs": [pathlib.Path("/round5/data")], "output_dir": pathlib.Path("/output"),
        "resume_params": pathlib.Path("/A/params"), "split_manifest": pathlib.Path("/round5/split.json"),
    }
    original = dynamic.build_train_command(**common)
    paired = dynamic.build_train_command(
        **common, ordered_listwise_elapsed_mode=args.ordered_listwise_elapsed_mode,
        ordered_listwise_elapsed_floor_seconds=args.ordered_listwise_elapsed_floor_seconds,
    )
    assert paired[:-4] == original
    assert paired[-4:] == [
        "--ordered-listwise-elapsed-mode", "paired_noise", "--ordered-listwise-elapsed-floor-seconds", "1.0",
    ]


@pytest.mark.parametrize("mode, floor", [("unknown", 1.0), ("paired_noise", 0.0), ("paired_noise", float("nan"))])
def test_dynamic_elapsed_target_rejects_invalid_arguments(mode, floor) -> None:
    with pytest.raises(ValueError, match="ordered_listwise_elapsed"):
        dynamic._validate_elapsed_target(mode, floor)  # noqa: SLF001


def test_dynamic_collection_must_be_complete_before_server_stop(tmp_path: pathlib.Path) -> None:
    summary_path = _collection(tmp_path, status="running")

    with pytest.raises(ValueError, match="must be complete"):
        dynamic.load_collection(summary_path)
