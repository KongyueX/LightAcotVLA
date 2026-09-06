from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("eval_libero_execution_horizon.py")
sys.path.insert(0, str(_SCRIPT.parent))
evaluator = importlib.import_module("eval_libero_execution_horizon")


def test_default_eval_modes_and_horizon_remain_legacy() -> None:
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/eval"])

    assert args.modes == list(evaluator.LEGACY_MODES)
    assert args.model_action_horizon == 10
    assert args.fixed_horizon == 9
    assert args.hierarchical_calibration_json is None
    assert args.hierarchical_aggregate_calibration_json is None
    assert args.initial_state_bank is None
    assert args.interleave_modes is False
    assert args.record_ordered_diagnostics is False


def test_disabled_aggregate_calibration_preserves_legacy_resume_signature() -> None:
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/eval"])

    signature = evaluator._run_signature(args)  # noqa: SLF001

    assert "hierarchical_aggregate_calibration_json" not in signature
    assert "initial_state_bank" not in signature
    assert "interleave_modes" not in signature
    assert "record_ordered_diagnostics" not in signature


@pytest.fixture
def frozen_bank(tmp_path):
    path = tmp_path / "bank"
    path.mkdir()
    states = np.asarray([[0, index, index + 1, 0, 0] for index in range(1, 5)], dtype=np.float64)
    state_file = path / "task00.npz"
    np.savez_compressed(
        state_file,
        states=states,
        episode_ids=np.arange(4),
        generation_seeds=np.asarray([-1, -1, 101, 102]),
    )
    state_lib = evaluator.horizon_initial_states
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "fingerprint_method": state_lib.FINGERPRINT_METHOD,
        "task_suite": "libero_10",
        "max_tasks": 1,
        "generated_per_task": 2,
        "tasks": [{
            "task_id": 0,
            "file": state_file.name,
            "sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
            "preset_count": 2,
            "state_dim": 5,
            "nq": 2,
            "fingerprints": [state_lib.fingerprint(state, 2) for state in states],
        }],
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    suite = SimpleNamespace(
        get_task_init_states=lambda task: states[:2],
        get_task=lambda task: SimpleNamespace(name=f"task{task}"),
    )
    return path, states, suite


def test_frozen_bank_exact_state_is_used_for_warmup_and_episode(frozen_bank, monkeypatch) -> None:
    path, states, suite = frozen_bank
    args = evaluator.build_parser().parse_args([
        "--output-dir", "/tmp/eval", "--initial-state-bank", str(path),
        "--initial-state-offset", "1", "--episode-ids", "2", "--modes", "original",
    ])
    bank = evaluator._prepare_initial_state_bank(args, suite, 1, [2])  # noqa: SLF001
    state_id, state = evaluator._resolve_initial_state(suite, 0, 2, args, bank)  # noqa: SLF001
    assert state_id == 3
    np.testing.assert_array_equal(state, states[3])
    signature = evaluator._run_signature(args)  # noqa: SLF001
    assert signature["initial_state_bank"] == str(path.resolve())
    assert signature["initial_state_bank_sha256"] == bank.sha256
    assert signature["initial_state_identity_mode"] == evaluator.horizon_initial_states.FINGERPRINT_METHOD

    restored = []
    env = SimpleNamespace(
        reset=lambda: None,
        set_init_state=lambda value: restored.append(value.copy()) or {},
        step=lambda action: ({}, 0, True, {}),
    )
    monkeypatch.setattr(evaluator.libero_eval, "_get_libero_env", lambda *unused: (env, "task"))
    monkeypatch.setattr(evaluator.libero_eval, "_safe_close_env", lambda unused: None)
    monkeypatch.setattr(evaluator.libero_eval, "_env_horizon", lambda unused: 1000)
    monkeypatch.setattr(evaluator.libero_eval, "_observation_to_policy_input", lambda *unused: {})
    monkeypatch.setattr(evaluator, "_request", lambda *unused, **kwargs: None)
    args.num_steps_wait = 0
    args.warmup_requests = 1
    evaluator._warmup(None, suite, args, bank)  # noqa: SLF001
    np.testing.assert_array_equal(restored[-1], states[3])
    args.num_steps_wait = 1
    row, _ = evaluator._run_episode(  # noqa: SLF001
        mode="original", task_id=0, episode=2, task_suite=suite, client=None, args=args, initial_state_bank=bank,
    )
    assert row["initial_state_id"] == 3
    np.testing.assert_array_equal(restored[-1], states[3])


def test_frozen_bank_prevalidates_ids_and_presets_and_legacy_still_cycles(frozen_bank) -> None:
    path, states, suite = frozen_bank
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/eval", "--initial-state-bank", str(path)])
    with pytest.raises(ValueError, match="modulo is forbidden"):
        evaluator._prepare_initial_state_bank(args, suite, 1, [4])  # noqa: SLF001
    with pytest.raises(ValueError, match="has no task1"):
        evaluator._prepare_initial_state_bank(args, suite, 2, [3])  # noqa: SLF001
    changed_suite = SimpleNamespace(get_task_init_states=lambda task: states[:2] + 1)
    with pytest.raises(ValueError, match="preset states differ"):
        evaluator._prepare_initial_state_bank(args, changed_suite, 1, [3])  # noqa: SLF001
    legacy_id, legacy_state = evaluator._resolve_initial_state(  # noqa: SLF001
        suite, 0, 3, SimpleNamespace(initial_state_offset=0)
    )
    assert legacy_id == 1
    np.testing.assert_array_equal(legacy_state, states[1])


def test_interleaved_schedule_alternates_pairs_and_preserves_resume_order(tmp_path) -> None:
    args = evaluator.build_parser().parse_args([
        "--output-dir", str(tmp_path), "--modes", "original", "ordered_transformer", "--interleave-modes",
    ])
    episodes = [50, 51, 52]
    schedule = list(evaluator._episode_schedule(args, 2, episodes))  # noqa: SLF001
    assert schedule[:6] == [
        ("original", 0, 50), ("ordered_transformer", 0, 50),
        ("ordered_transformer", 0, 51), ("original", 0, 51),
        ("original", 0, 52), ("ordered_transformer", 0, 52),
    ]
    assert schedule[6:8] == [("ordered_transformer", 1, 50), ("original", 1, 50)]
    assert len(schedule) == len(set(schedule)) == 12
    evaluator._prepare_journal(tmp_path, args)  # noqa: SLF001
    evaluator._append_csv(tmp_path / "rollout_rows.csv", [  # noqa: SLF001
        {"mode": mode, "task_id": task, "episode": episode} for mode, task, episode in schedule[:3]
    ])
    args.resume = True
    _, completed = evaluator._prepare_journal(tmp_path, args)  # noqa: SLF001
    remaining = [key for key in evaluator._episode_schedule(args, 2, episodes) if key not in completed]  # noqa: SLF001
    assert remaining == schedule[3:]
    args.interleave_modes = False
    with pytest.raises(ValueError, match="Resume configuration differs"):
        evaluator._prepare_journal(tmp_path, args)  # noqa: SLF001
    legacy = list(evaluator._episode_schedule(SimpleNamespace(task_start=0, modes=args.modes), 2, episodes))  # noqa: SLF001
    assert legacy == [(mode, task, episode) for mode in args.modes for task in range(2) for episode in episodes]


def test_enabled_aggregate_calibration_is_bound_into_resume_signature() -> None:
    args = evaluator.build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/eval",
            "--modes",
            "hierarchical_transformer",
            "--hierarchical-aggregate-calibration-json",
            "/tmp/aggregate.json",
        ]
    )

    signature = evaluator._run_signature(args)  # noqa: SLF001

    assert signature["hierarchical_aggregate_calibration_json"] == "/tmp/aggregate.json"


def test_ordered_transformer_selects_model_output_without_calibration() -> None:
    args = evaluator.build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/eval",
            "--modes",
            "ordered_transformer",
            "--model-action-horizon",
            "25",
        ]
    )
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(20, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25], dtype=np.int32),
    }

    selected, info = evaluator._select_horizon(  # noqa: SLF001
        evaluator.ORDERED_MODE,
        result,
        args=args,
        budget_state=SimpleNamespace(),
    )

    assert selected == 20
    assert info == {
        "raw_horizon": 20,
        "budget_limited": 0.0,
        "selector_policy": "ordered_transformer",
    }
    assert args.hierarchical_calibration_json is None
    assert args.hierarchical_aggregate_calibration_json is None


def test_ordered_diagnostics_record_existing_vectors_without_changing_selection() -> None:
    args = evaluator.build_parser().parse_args([
        "--output-dir", "/tmp/eval", "--model-action-horizon", "25", "--record-ordered-diagnostics",
    ])
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(20, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
        "execution_horizon_ordered_horizon_probability": np.asarray([0.1, 0.1, 0.1, 0.6, 0.1]),
        "execution_horizon_ordered_continuation_logits": np.asarray([2.0, 2.0, 2.0, -2.0]),
    }
    selected, info = evaluator._select_horizon(  # noqa: SLF001
        evaluator.ORDERED_MODE, result, args=args, budget_state=SimpleNamespace()
    )
    assert selected == 20
    assert info["ordered_horizon_probability"] == [0.1, 0.1, 0.1, 0.6, 0.1]
    assert info["ordered_continuation_logits"] == [2.0, 2.0, 2.0, -2.0]
    assert info["candidate_horizons"] == [5, 10, 15, 20, 25]
    assert evaluator._run_signature(args)["record_ordered_diagnostics"] is True  # noqa: SLF001
    result.pop("execution_horizon_ordered_continuation_logits")
    selected, info = evaluator._select_horizon(  # noqa: SLF001
        evaluator.ORDERED_MODE, result, args=args, budget_state=SimpleNamespace()
    )
    assert selected == 20
    assert "ordered_continuation_logits" not in info


@pytest.mark.parametrize(
    ("selected", "candidates", "model_horizon", "message"),
    [
        (20, [5, 10, 15, 25], 25, "not in candidate_horizons"),
        (25, [5, 10, 15, 20, 25], 20, "model_action_horizon=20"),
    ],
)
def test_ordered_transformer_rejects_invalid_selected_horizon(
    selected: int,
    candidates: list[int],
    model_horizon: int,
    message: str,
) -> None:
    args = SimpleNamespace(model_action_horizon=model_horizon)
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(selected, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray(candidates, dtype=np.int32),
    }

    with pytest.raises(ValueError, match=message):
        evaluator._select_horizon(  # noqa: SLF001
            evaluator.ORDERED_MODE,
            result,
            args=args,
            budget_state=SimpleNamespace(),
        )
