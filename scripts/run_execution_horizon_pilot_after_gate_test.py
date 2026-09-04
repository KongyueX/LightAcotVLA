from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from openpi.execution_horizon import initial_states

_SCRIPT = pathlib.Path(__file__).with_name("run_execution_horizon_pilot_after_gate.py")
sys.path.insert(0, str(_SCRIPT.parent))
pilot = importlib.import_module("run_execution_horizon_pilot_after_gate")


def _make_bank(
    path: pathlib.Path,
    *,
    generated_states: int,
    changed_prefix: bool = False,
) -> initial_states.InitialStateBank:
    path.mkdir()
    states = np.asarray(
        [[0, episode + 1, episode + 2, 0, 0] for episode in range(2 + generated_states)],
        dtype=np.float64,
    )
    if changed_prefix:
        states[2, 1] += 0.5
    seeds = np.asarray([-1, -1, *range(100, 100 + generated_states)], dtype=np.int64)
    target = path / "task00.npz"
    np.savez_compressed(target, states=states, episode_ids=np.arange(len(states)), generation_seeds=seeds)
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "fingerprint_method": initial_states.FINGERPRINT_METHOD,
        "task_suite": "libero_10",
        "max_tasks": 1,
        "generated_per_task": generated_states,
        "tasks": [
            {
                "task_id": 0,
                "file": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "preset_count": 2,
                "state_dim": 5,
                "nq": 2,
                "fingerprints": [initial_states.fingerprint(state, 2) for state in states],
            }
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return initial_states.InitialStateBank(path)


def test_aggregate_artifact_cannot_be_overridden_from_pilot_cli() -> None:
    args = pilot.build_parser().parse_args(
        [
            "--post-summary",
            "/tmp/post.json",
            "--output-dir",
            "/tmp/controller",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--pilot-output-dir",
            "/tmp/pilot",
        ]
    )

    assert not hasattr(args, "aggregate_calibration_json")


def test_eval_resume_config_is_bound_to_aggregate_and_pilot_shape(tmp_path: pathlib.Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text("{}")
    config_path = tmp_path / "run_config.json"
    config = {
        "modes": ["hierarchical_transformer"],
        "hierarchical_aggregate_calibration_json": str(aggregate),
        "hierarchical_calibration_json": None,
        "seed": 42,
        "num_trials_per_task": 20,
        "model_action_horizon": 25,
        "max_tasks": 10,
        "task_suite_name": "libero_10",
    }
    config_path.write_text(json.dumps(config))

    assert pilot._verify_eval_run_config(  # noqa: SLF001
        config_path,
        aggregate_path=aggregate,
        seed=42,
        num_trials_per_task=20,
    ) == config

    config["seed"] = 7
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seed"):
        pilot._verify_eval_run_config(  # noqa: SLF001
            config_path,
            aggregate_path=aggregate,
            seed=42,
            num_trials_per_task=20,
        )


def test_pilot_binding_is_immutable(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pilot_binding.json"
    expected = {"schema_version": 1, "artifact": "abc"}

    pilot._ensure_pilot_binding(path, expected)  # noqa: SLF001
    pilot._ensure_pilot_binding(path, expected)  # noqa: SLF001

    with pytest.raises(ValueError, match="binding differs"):
        pilot._ensure_pilot_binding(path, {**expected, "artifact": "changed"})  # noqa: SLF001


def test_pilot_accepts_legacy_single_bank_isolation(tmp_path: pathlib.Path) -> None:
    bank = _make_bank(tmp_path / "bank", generated_states=2)
    isolation = bank.audit_partitions({"development": {(0, 2)}, "final": {(0, 3)}})

    pilot._verify_initial_state_isolation(isolation)  # noqa: SLF001

    with pytest.raises(ValueError, match="changed after the offline audit"):
        pilot._verify_initial_state_isolation(  # noqa: SLF001
            {**isolation, "initial_state_bank_sha256": "0" * 64}
        )


def test_pilot_revalidates_multi_bank_lineage(tmp_path: pathlib.Path) -> None:
    parent = _make_bank(tmp_path / "parent", generated_states=1)
    child = _make_bank(tmp_path / "child", generated_states=3)
    isolation = initial_states.audit_partitions_across_banks(
        {"development": (child, {(0, 4)}), "final": (parent, {(0, 2)})}
    )
    isolation["bank_lineage"] = initial_states.audit_bank_prefix(parent, child)

    pilot._verify_initial_state_isolation(isolation)  # noqa: SLF001

    changed_parent = dict(isolation["bank_lineage"]["parent"])
    changed_parent["initial_state_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent differs"):
        pilot._verify_initial_state_isolation(  # noqa: SLF001
            {
                **isolation,
                "bank_lineage": {**isolation["bank_lineage"], "parent": changed_parent},
            }
        )


def test_pilot_rejects_live_child_that_no_longer_matches_parent_prefix(tmp_path: pathlib.Path) -> None:
    parent = _make_bank(tmp_path / "parent", generated_states=1)
    changed_child = _make_bank(tmp_path / "changed_child", generated_states=3, changed_prefix=True)
    isolation = initial_states.audit_partitions_across_banks(
        {"development": (changed_child, {(0, 4)}), "final": (parent, {(0, 2)})}
    )
    isolation["bank_lineage"] = {
        "status": "complete",
        "schema_version": 1,
        "parent": parent.metadata(),
        "child": changed_child.metadata(),
    }

    with pytest.raises(ValueError, match="changed parent states"):
        pilot._verify_initial_state_isolation(isolation)  # noqa: SLF001


def test_pilot_rejects_unknown_or_incomplete_multi_bank_isolation() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        pilot._verify_initial_state_isolation(  # noqa: SLF001
            {"status": "complete", "pairwise_initial_state_overlap": 0, "schema_version": 3}
        )
    with pytest.raises(ValueError, match="exactly development and final"):
        pilot._verify_initial_state_isolation(  # noqa: SLF001
            {
                "status": "complete",
                "pairwise_initial_state_overlap": 0,
                "schema_version": 2,
                "partition_banks": {},
            }
        )
