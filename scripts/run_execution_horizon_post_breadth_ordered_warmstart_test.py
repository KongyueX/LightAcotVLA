from __future__ import annotations

import json
import pathlib

import run_execution_horizon_post_breadth_ordered_warmstart as post


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_load_snapshot_and_build_ordered_training_command(tmp_path: pathlib.Path) -> None:
    breadth = tmp_path / "collection" / "breadth_first_min10"
    breadth.mkdir(parents=True)
    data_dirs = []
    for index in range(2):
        data = tmp_path / f"round3-{index}"
        data.mkdir()
        data_dirs.append(str(data))
    realized = {
        str(task): {
            role: {"roots": 10, "episodes": list(range(10)), "data_dirs": data_dirs}
            for role in ("train", "calibration", "dev_audit")
        }
        for task in range(10)
    }
    _write_json(
        breadth / "summary.json",
        {
            "status": "complete",
            "purpose": "breadth_first_warm_start",
            "split_manifest": str(breadth / "split_manifest.json"),
            "resumable_to_full_round3": True,
            "full_round3_outputs_written": False,
            "num_roots": 300,
            "realized_by_task": realized,
            "data_dirs": data_dirs,
        },
    )
    _write_json(
        breadth / "split_manifest.json",
        {
            "train_group_ids": [1],
            "early_stop_group_ids": [2],
            "calibration_group_ids": [3],
            "dev_audit_group_ids": [4],
        },
    )

    summary, manifest, loaded_data = post.load_breadth_snapshot(breadth)
    command = post.build_train_command(
        python=pathlib.Path("/venv/python"),
        train_script=pathlib.Path("/code/train_execution_horizon_predictor.py"),
        datasets=[pathlib.Path("/base"), *loaded_data],
        output_dir=pathlib.Path("/output"),
        resume_params=pathlib.Path("/resume/params"),
        split_manifest=manifest,
    )

    assert summary["status"] == "complete"
    assert command[:3] == ["/venv/python", "/code/train_execution_horizon_predictor.py", "--dataset"]
    assert command[command.index("--resume-params") + 1] == "/resume/params"
    assert command[command.index("--input-split-manifest") + 1] == str(manifest)
    assert command[command.index("--seed") + 1] == "7"
    assert "--paired-distribution-heads" in command
    assert "--ordered-continuation-head" in command
    assert command[command.index("--loss-ordered-listwise") + 1] == "2"
    assert command[command.index("--loss-raw-h-ordinal") + 1] == "0"
    assert "--resume-legacy-paired-heads" not in command
    assert not any("audit" in value or "calibrate" in value for value in command[:2])


def test_resume_predictor_contract_is_strict_and_reuses_existing_params(tmp_path: pathlib.Path) -> None:
    predictor = tmp_path / "predictor"
    (predictor / "params").mkdir(parents=True)
    _write_json(
        predictor / "predictor_config.json",
        {
            "temporal_backbone": "transformer",
            "temporal_layers": 2,
            "hidden_dim": 256,
            "num_heads": 4,
            "feed_forward_multiplier": 4,
            "reference_horizon": 10,
            "visual_num_queries": 4,
            "paired_distribution_heads": True,
            "paired_advantage_heads": False,
            "candidate_horizons": [5, 10, 15, 20, 25],
        },
    )
    _write_json(predictor / "summary.json", {"status": "complete", "training_seed": 7})

    assert post.validate_resume_predictor(predictor) == (predictor / "params").resolve()
