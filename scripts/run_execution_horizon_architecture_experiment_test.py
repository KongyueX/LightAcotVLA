# ruff: noqa: SLF001
from __future__ import annotations

from collections import Counter
import fcntl
import json
from types import SimpleNamespace

import eval_libero_execution_horizon as evaluator
import numpy as np
import pytest
import run_execution_horizon_architecture_experiment as runner


def _args(tmp_path):
    return runner.build_parser().parse_args([
        "--code-dir", str(tmp_path / "code"), "--source-dir", str(tmp_path / "source"),
        "--output-dir", str(tmp_path / "output"), "--a-predictor-dir", str(tmp_path / "A"),
        "--policy-dir", str(tmp_path / "policy"), "--python", "python",
    ])


def _eval_args(args, phase="development", variants=None):
    command = runner.evaluation_command(args, phase, variants or ["visual_query", "expert_hidden"])
    return evaluator.build_parser().parse_args(command[2:])


def _row(success, rpc):
    return {
        "success_count": success, "episodes": 100,
        "means": {
            "policy_rpc_wall_total_ms": rpc, "actual_policy_total_ms": rpc * 0.9,
            "actual_episode_elapsed_total_ms": rpc + 5000,
        },
    }


def _analysis(query, hidden, *, anchor=(93, 1900)):
    return {"runs": {
        runner.MODES["A"]: _row(*anchor),
        runner.MODES["visual_query"]: _row(*query),
        runner.MODES["expert_hidden"]: _row(*hidden),
    }}


def test_development_schedule_pairs_all_three_modes_on_the_same_100_states(tmp_path):
    args = _eval_args(_args(tmp_path))
    expected_modes = ["ordered_transformer", "ordered_visual_query", "ordered_expert_hidden"]
    assert args.modes == expected_modes
    assert args.episode_ids == list(range(336, 346))
    assert (args.port, args.visual_query_port, args.expert_hidden_port) == (8050, 8051, 8052)
    assert args.model_action_horizon == 25
    assert (args.seed, args.action_cot_denoising_steps, args.final_denoising_steps) == (7, 10, 10)
    assert args.num_steps_wait == 10
    assert args.resize_size == 224
    scheduled = list(evaluator._episode_schedule(args, args.max_tasks, args.episode_ids))
    assert len(scheduled) == 300
    assert Counter(mode for mode, _, _ in scheduled) == dict.fromkeys(expected_modes, 100)
    for index in range(0, len(scheduled), 3):
        group = scheduled[index:index + 3]
        assert len({(task, episode) for _, task, episode in group}) == 1
        assert {mode for mode, _, _ in group} == set(expected_modes)


@pytest.mark.parametrize("phase", ["development", "final"])
def test_evaluation_resumes_only_after_its_journal_exists(tmp_path, phase):
    args = _args(tmp_path)
    output = args.output_dir / phase
    output.mkdir(parents=True)
    fresh = _eval_args(args, phase, ["expert_hidden"])
    assert not fresh.resume
    assert evaluator._prepare_journal(output, fresh) == ([], set())
    resumed = _eval_args(args, phase, ["expert_hidden"])
    assert resumed.resume
    assert evaluator._prepare_journal(output, resumed) == ([], set())


def test_native_modes_use_their_own_endpoints_and_server_selected_h(tmp_path):
    args = _eval_args(_args(tmp_path))
    clients = {}
    received = {}
    for mode, horizon in zip(args.modes, (5, 15, 25), strict=True):
        probability = np.zeros(5, dtype=np.float32)
        probability[(horizon // 5) - 1] = 1
        response = {
            "execution_horizon_ordered_horizon_probability": probability,
            "execution_horizon_ordered_selected_h": np.asarray(horizon),
            "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
            "policy_timing": {"infer_ms": 20.0, "execution_horizon_predictor_ms": 2.0},
        }

        def infer(request, *, _mode=mode, _response=response):
            received[_mode] = request
            return _response

        clients[mode] = SimpleNamespace(infer=infer)
    architecture_clients = {mode: clients[mode] for mode in evaluator.ARCHITECTURE_MODES}
    for mode, expected_port, expected_h in zip(args.modes, (8050, 8051, 8052), (5, 15, 25), strict=True):
        client, mode_args = evaluator._mode_runtime(
            mode, args, clients[evaluator.ORDERED_MODE], architecture_clients=architecture_clients,
        )
        assert client is clients[mode]
        assert mode_args.port == expected_port
        if mode in evaluator.ARCHITECTURE_MODES:
            assert mode_args.modes == [mode]
        result, timing = evaluator._request(
            client, {"state": np.zeros(32)}, mode=mode, seed=123, previous_actions=None,
            previous_horizon=10, budget_fraction=0.5, episode_progress=0.1,
            absolute_decision_step=10, args=mode_args,
        )
        horizon, _ = evaluator._select_horizon(mode, result, args=mode_args, budget_state=None)
        assert horizon == expected_h
        assert timing["policy_ms"] == 20
        assert timing["predictor_ms"] == 2
        request = received[mode]
        assert bool(request["run_execution_horizon_predictor"])
        assert "execution_horizon_export_architecture_cache" not in request
        assert "export_execution_horizon_prefix_tokens" not in request
        assert "batched_mc_samples" not in request
    assert args.port == 8050
    assert len(args.modes) == 3


@pytest.mark.parametrize("mode", evaluator.ARCHITECTURE_MODES)
def test_missing_architecture_endpoint_does_not_fall_back_to_a(tmp_path, mode):
    with pytest.raises(ValueError, match="own configured sidecar endpoint"):
        evaluator._mode_runtime(mode, _eval_args(_args(tmp_path)), object(), architecture_clients={})


@pytest.mark.parametrize(("query", "hidden", "expected"), [
    ((92, 100), (93, 1900), None),
    ((94, 2500), (93, 100), "visual_query"),
    ((93, 1800), (93, 1700), "expert_hidden"),
])
def test_winner_prioritizes_success_then_rpc_and_keeps_a_on_tie(query, hidden, expected):
    assert runner.winner(_analysis(query, hidden)) == expected


@pytest.mark.parametrize("selected", [None, "visual_query", "expert_hidden"])
def test_complete_development_is_unconditional_and_only_winner_enters_final(tmp_path, monkeypatch, selected):
    args = _args(tmp_path)
    stages = []
    servers = []
    stopped = []
    development = _analysis((92, 2000), (92, 2000))
    if selected:
        development["runs"][runner.MODES[selected]] = _row(94, 2200)

    def run_stage(arguments, name, command, *, gpu=False):
        stages.append((name, command, gpu))
        if name.startswith("train_"):
            destination = command[command.index("--output-dir") + 1]
            runner.feedback.write_json(runner.pathlib.Path(destination) / "summary.json", {
                "status": "complete", "best_step": 0,
                "best_greedy_validation": {"greedy_success_delta": 0.0, "greedy_rpc_delta_seconds": 0.0},
            })

    def summarize(directory, episodes, variants):
        if directory.name == "development":
            assert tuple(episodes) == tuple(range(336, 346))
            assert variants == ["visual_query", "expert_hidden"]
            return development
        assert selected
        assert variants == [selected]
        assert tuple(episodes) == tuple(range(346, 366))
        return {"runs": {mode: development["runs"][mode] for mode in (runner.MODES["A"], runner.MODES[selected])}}

    monkeypatch.setattr(runner, "run_stage", run_stage)
    monkeypatch.setattr(runner, "summarize", summarize)
    monkeypatch.setattr(runner, "ensure_server", lambda arguments, name, port: servers.append((name, port)))
    monkeypatch.setattr(runner, "stop_servers", lambda arguments: stopped.append(arguments.output_dir))
    monkeypatch.setattr(runner.feedback, "notify", lambda *args: None)
    runner.main(args)
    expected = ["features_train", "features_early_stop", "train_visual_query", "train_expert_hidden", "development"]
    assert [name for name, _, _ in stages] == expected + (["final"] if selected else [])
    assert servers == [("A", 8050), ("visual_query", 8051), ("expert_hidden", 8052)]
    for name, command, gpu in stages:
        assert gpu is name.startswith("train_")
        if gpu:
            assert command[command.index("--max-updates") + 1] == "650"
            assert command[command.index("--seed") + 1] == "7"
            assert command[command.index("--learning-rate") + 1] == "0.0001"
            assert command[command.index("--batch-size") + 1] == "64"
        if name == "final":
            final_args = evaluator.build_parser().parse_args(command[2:])
            assert final_args.modes == [runner.MODES["A"], runner.MODES[selected]]
            assert final_args.episode_ids == list(range(346, 366))
            assert (final_args.visual_query_port is not None) is (selected == "visual_query")
            assert (final_args.expert_hidden_port is not None) is (selected == "expert_hidden")
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["development_winner"] == selected
    assert (summary["final"] is not None) is (selected is not None)
    assert len(stopped) == 1


@pytest.mark.parametrize(("name", "gpu", "expected_platform"), [
    ("features_train", False, "cpu"), ("development", False, "cpu"),
    ("train_visual_query", True, "cuda"),
])
def test_stage_process_uses_cpu_for_rollouts_and_gpu_for_training(tmp_path, monkeypatch, name, gpu, expected_platform):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    command = ["python", "stage.py", "--output-dir", str(tmp_path / "stage_output")]
    seen = []

    def start(actual_command, **kwargs):
        seen.append((actual_command, kwargs))
        return SimpleNamespace(pid=999999, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    runner.run_stage(args, name, command, gpu=gpu)
    assert seen[0][0] == command
    assert seen[0][1]["env"]["JAX_PLATFORMS"] == expected_platform
    assert seen[0][1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert (args.output_dir / f"{name}.exit").read_text().strip() == "0"


@pytest.mark.parametrize("completion", ["stage_exit", "training_summary"])
def test_completed_training_is_not_repeated_without_overall_summary(tmp_path, monkeypatch, completion):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    destination = args.output_dir / "training_visual_query"
    if completion == "stage_exit":
        (args.output_dir / "train_visual_query.exit").write_text("0\n")
    else:
        runner.feedback.write_json(destination / "summary.json", {"status": "complete"})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Completed training reran."))
    runner.run_stage(args, "train_visual_query", ["python", "train.py", "--output-dir", str(destination)], gpu=True)
    assert not (args.output_dir / "summary.json").exists()
    assert (args.output_dir / "train_visual_query.exit").read_text().strip() == "0"


def test_main_retains_live_child_record_before_preparing_or_restarting(tmp_path, monkeypatch):
    args = _args(tmp_path)
    previous = {"status": "running", "stage": "train_visual_query", "child_pid": 456,
                "command": ["python", "train.py", "--output-dir", str(args.output_dir / "training_visual_query")]}
    status_path = args.output_dir / "status.json"
    runner.feedback.write_json(status_path, previous)
    monkeypatch.setattr(runner, "live_command", lambda pid: previous["command"] if pid == 456 else None)
    monkeypatch.setattr(runner, "execute", lambda args: pytest.fail("A live child must block controller restart."))
    with pytest.raises(RuntimeError, match="still running in PID 456"):
        runner.main(args)
    assert json.loads(status_path.read_text()) == previous


def test_stage_rejects_active_original_command_when_resume_flag_is_added(tmp_path, monkeypatch):
    args = _args(tmp_path)
    command = ["python", "eval.py", "--output-dir", str(args.output_dir / "development")]
    runner.feedback.write_json(args.output_dir / "status.json", {
        "stage": "development", "child_pid": 456, "command": command,
    })
    monkeypatch.setattr(runner, "live_command", lambda pid: command)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Active evaluation duplicated."))
    with pytest.raises(RuntimeError, match="still running"):
        runner.run_stage(args, "development", [*command, "--resume"])


def test_single_controller_lock_prevents_overlapping_runs(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    executed = []
    monkeypatch.setattr(runner, "execute", lambda args: executed.append(args.output_dir))
    with (args.output_dir / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main(args)
    assert not executed
    runner.main(args)
    assert executed == [args.output_dir]


def test_stopping_servers_only_signals_this_runs_matching_processes(tmp_path, monkeypatch):
    args = _args(tmp_path)
    processes = {}
    for index, name in enumerate(runner.MODES):
        pid = 1000 + index
        command = runner.server_command(args, name, args.port + index)
        runner.feedback.write_json(args.output_dir / f"server_{name}.json", {"pid": pid, "command": command})
        processes[pid] = command if name == "A" else ["python", "unrelated.py"] if name == "visual_query" else None
    processes[2000] = ["python", "serve_policy.py"]
    signals = []
    monkeypatch.setattr(runner, "live_command", lambda pid: processes[pid])
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    runner.stop_servers(args)
    assert signals == [(1000, runner.signal.SIGTERM)]
    assert (args.output_dir / "server_A.stopped.json").exists()
    assert not (args.output_dir / "server_visual_query.stopped.json").exists()
    assert not (args.output_dir / "server_expert_hidden.stopped.json").exists()
