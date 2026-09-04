"""Run the hierarchical H25 closed-loop pilot only after both offline gates pass."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time
from typing import Any

import execution_horizon_aggregate_risk_common as aggregate_common

from openpi.execution_horizon import hierarchical
from openpi.execution_horizon import initial_states

PILOT_BINDING_SCHEMA_VERSION = 1
PILOT_MODEL_ACTION_HORIZON = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pilot-output-dir", required=True)
    parser.add_argument("--policy-config", default="acot_libero_long_chunk_h25")
    parser.add_argument("--policy-server-tmux", default="h25_dynamic_pilot_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--server-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    parser.add_argument("--require-initial-state-isolation", action="store_true")
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"Artifact is not complete: {path}")
    return payload


def _require_digest(payload: dict[str, Any], name: str, expected: str) -> None:
    value = payload.get(name)
    if value != expected:
        raise ValueError(f"Post-summary {name} does not match the live frozen artifact: {value!r} != {expected!r}.")


def _load_frozen_aggregate_binding(
    post_summary: dict[str, Any],
    *,
    post_summary_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
) -> tuple[
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    hierarchical.AggregateSelectorCalibration,
    dict[str, Any],
]:
    """Load and cryptographically bind every object used by the pilot server/client."""

    aggregate_value = post_summary.get("aggregate_calibration_json")
    if not aggregate_value:
        raise ValueError("Dual-gate post-summary does not name a frozen aggregate calibration artifact.")
    aggregate_path = pathlib.Path(aggregate_value).resolve()
    if not aggregate_path.is_file():
        raise FileNotFoundError(aggregate_path)
    aggregate_digest_before = aggregate_common.file_digest(aggregate_path)
    artifact = hierarchical.AggregateSelectorCalibration.load(aggregate_path)
    aggregate_digest = aggregate_common.file_digest(aggregate_path)
    if aggregate_digest_before != aggregate_digest:
        raise RuntimeError("Aggregate calibration changed while being loaded for the pilot.")
    if not artifact.aggregate_gate_passed or artifact.selected_rule is None:
        raise ValueError("Dynamic pilot requires an aggregate artifact with a uniquely selected passing rule.")
    if post_summary.get("provenance_verified") is not True:
        raise ValueError("Dual-gate post-summary does not attest aggregate provenance verification.")
    if aggregate_common.jsonable(post_summary.get("selected_rule")) != aggregate_common.jsonable(
        artifact.selected_rule
    ):
        raise ValueError("Post-summary selected_rule differs from the frozen aggregate artifact.")

    predictor_dir = pathlib.Path(post_summary["predictor_dir"]).resolve()
    predictor_config_path = predictor_dir / "predictor_config.json"
    params_path = predictor_dir / "params"
    pointwise_value = post_summary.get("pointwise_calibration_json")
    pointwise_path = (
        pathlib.Path(pointwise_value).resolve() if pointwise_value is not None else predictor_dir / "calibration.json"
    )
    for required in (predictor_dir / "summary.json", predictor_config_path, params_path, pointwise_path):
        if not required.exists():
            raise FileNotFoundError(required)

    predictor_config_digest = aggregate_common.json_file_digest(predictor_config_path)
    params_digest = aggregate_common.params_tree_digest(params_path)
    pointwise_digest = aggregate_common.json_file_digest(pointwise_path)
    checkpoint_digest = aggregate_common.params_tree_digest(checkpoint_dir)
    post_summary_digest = aggregate_common.file_digest(post_summary_path)

    frozen = artifact.provenance
    if predictor_config_digest != frozen.predictor_config_digest:
        raise ValueError("Pilot predictor_config.json differs from aggregate provenance.")
    if params_digest != frozen.params_digest:
        raise ValueError("Pilot predictor params differ from aggregate provenance.")
    if pointwise_digest != frozen.pointwise_calibration_digest:
        raise ValueError("Pilot pointwise calibration differs from aggregate provenance.")
    pointwise = hierarchical.HierarchicalCalibration.load(pointwise_path)
    if aggregate_common.jsonable(dataclasses.asdict(pointwise)) != aggregate_common.jsonable(
        dataclasses.asdict(artifact.pointwise_calibration)
    ):
        raise ValueError("Pilot pointwise calibration differs from the copy embedded in aggregate calibration.")
    _require_digest(post_summary, "aggregate_calibration_sha256", aggregate_digest)
    _require_digest(post_summary, "predictor_config_digest", predictor_config_digest)
    _require_digest(post_summary, "params_digest", params_digest)
    _require_digest(post_summary, "pointwise_calibration_digest", pointwise_digest)
    _require_digest(post_summary, "checkpoint_digest", checkpoint_digest)

    recorded_checkpoint = post_summary.get("checkpoint_dir")
    if recorded_checkpoint is None or pathlib.Path(recorded_checkpoint).resolve() != checkpoint_dir:
        raise ValueError("Pilot checkpoint-dir differs from the checkpoint frozen in post-summary.")
    _complete_json(predictor_dir / "summary.json")
    predictor_config = json.loads(predictor_config_path.read_text())
    if not bool(predictor_config.get("paired_distribution_heads", False)):
        raise ValueError("Aggregate-risk pilot requires paired_distribution_heads predictor parameters.")

    binding = {
        "schema_version": PILOT_BINDING_SCHEMA_VERSION,
        "post_summary": str(post_summary_path),
        "post_summary_sha256": post_summary_digest,
        "aggregate_calibration_json": str(aggregate_path),
        "aggregate_calibration_sha256": aggregate_digest,
        "predictor_dir": str(predictor_dir),
        "predictor_config_digest": predictor_config_digest,
        "params_digest": params_digest,
        "pointwise_calibration_json": str(pointwise_path),
        "pointwise_calibration_digest": pointwise_digest,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_digest": checkpoint_digest,
    }
    return predictor_dir, pointwise_path, aggregate_path, artifact, binding


def _ensure_pilot_binding(path: pathlib.Path, expected: dict[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise ValueError(f"Pilot binding differs from the frozen run identity: {observed} != {expected}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        observed = json.loads(path.read_text())
        if observed != expected:
            raise ValueError("A concurrent pilot created a different immutable binding.") from None


def _verify_eval_run_config(
    path: pathlib.Path,
    *,
    aggregate_path: pathlib.Path,
    seed: int,
    num_trials_per_task: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot validate pilot resume without {path}.")
    config = json.loads(path.read_text())
    expected_scalars = {
        "seed": seed,
        "num_trials_per_task": num_trials_per_task,
        "model_action_horizon": PILOT_MODEL_ACTION_HORIZON,
        "max_tasks": 10,
        "task_suite_name": "libero_10",
    }
    for name, expected in expected_scalars.items():
        if config.get(name) != expected:
            raise ValueError(
                f"Pilot run_config {name} differs from the frozen pilot: "
                f"{config.get(name)!r} != {expected!r}."
            )
    if config.get("modes") != ["hierarchical_transformer"]:
        raise ValueError("Pilot run_config must contain only hierarchical_transformer mode.")
    configured_aggregate = config.get("hierarchical_aggregate_calibration_json")
    if configured_aggregate is None or pathlib.Path(configured_aggregate).resolve() != aggregate_path:
        raise ValueError("Pilot run_config uses a different aggregate calibration artifact.")
    if config.get("hierarchical_calibration_json") is not None:
        raise ValueError("Aggregate pilot run_config must not also contain legacy pointwise selector deployment.")
    return config


def _tmux_present(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _stop_tmux(name: str) -> None:
    if _tmux_present(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=True)


def _wait_for_server(host: str, port: int, tmux_name: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _tmux_present(tmux_name):
            raise RuntimeError(f"Policy server tmux {tmux_name!r} exited before becoming ready.")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(5.0)
    raise TimeoutError(f"Policy server did not become ready at {host}:{port} within {timeout_seconds}s.")


def _notify(message: str, output_dir: pathlib.Path, code_dir: pathlib.Path) -> None:
    if not os.environ.get("FEISHU_WEBHOOK_URL") or not os.environ.get("FEISHU_SIGNING_SECRET"):
        return
    completed = subprocess.run(
        [
            sys.executable,
            str(code_dir / "watch_experiment_feishu.py"),
            "--output-dir",
            str(output_dir),
            "--test-message",
            message,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode:
        print(
            f"Warning: Feishu notification exited with code {completed.returncode}; experiment continues.",
            file=sys.stderr,
        )


def _find_pilot_summary(pilot_output_dir: pathlib.Path) -> pathlib.Path:
    summaries = sorted(pilot_output_dir.rglob("summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected exactly one pilot summary under {pilot_output_dir}, found {summaries}.")
    return summaries[0]


def _verify_pilot_summary(summary: dict[str, Any], expected_episodes: int) -> None:
    if int(summary.get("num_tasks", -1)) != 10:
        raise ValueError("Dynamic pilot summary does not contain ten tasks.")
    if int(summary.get("num_trials_per_task", -1)) * 10 != expected_episodes:
        raise ValueError("Dynamic pilot summary trial count does not match the requested pilot.")
    overall = summary.get("overall")
    if isinstance(overall, dict):
        hierarchical = overall.get("hierarchical_transformer")
    elif isinstance(overall, list):
        matching = [row for row in overall if row.get("mode") == "hierarchical_transformer"]
        hierarchical = matching[0] if len(matching) == 1 else None
    else:
        hierarchical = None
    if not isinstance(hierarchical, dict) or int(hierarchical.get("episodes", -1)) != expected_episodes:
        raise ValueError("Dynamic pilot summary does not contain the expected hierarchical episodes.")


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0 or args.server_timeout_seconds <= 0:
        raise ValueError("Poll and server timeout values must be positive.")
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must lie in [1, 65535].")

    post_summary_path = pathlib.Path(args.post_summary).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    pilot_output_dir = pathlib.Path(args.pilot_output_dir).resolve()
    code_dir = pathlib.Path(__file__).resolve().parent
    status_path = output_dir / "status.json"
    result_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_result = _complete_json(result_path) if result_path.exists() else None
    _write_json(status_path, {"status": "waiting_for_dual_offline_gate", "post_summary": str(post_summary_path)})
    while not post_summary_path.exists():
        time.sleep(args.poll_seconds)
    post_summary = _complete_json(post_summary_path)
    post_summary_sha256 = aggregate_common.file_digest(post_summary_path)
    if not bool(post_summary.get("dual_official_gate", False)):
        result = {
            "status": "complete",
            "pilot_started": False,
            "reason": "dual_official_gate_not_met",
            "post_summary": str(post_summary_path),
            "post_summary_sha256": post_summary_sha256,
        }
        if existing_result is not None:
            if existing_result != result:
                raise ValueError("Existing no-pilot result is bound to a different post-summary.")
        else:
            _write_json(result_path, result)
        _write_json(status_path, {"status": "offline_gate_not_met", "pilot_started": False})
        _notify(
            "【H25 predictor门控】冻结候选未同时通过development与fresh-final的预注册95%门, 动态10x20 pilot未启动; 将按失败模式继续安全迭代。",
            output_dir,
            code_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    for name in ("development_official", "final_official"):
        audit = post_summary.get(name)
        if not isinstance(audit, dict) or not bool(audit.get("offline_engineering_gate", False)):
            raise ValueError(f"dual_official_gate conflicts with {name} audit.")
    if args.require_initial_state_isolation:
        isolation = post_summary.get("initial_state_isolation") or {}
        if isolation.get("status") != "complete" or isolation.get("pairwise_initial_state_overlap") != 0:
            raise ValueError("Dynamic pilot requires verified initial-state isolation.")
        bank = initial_states.InitialStateBank(isolation["initial_state_bank"])
        if bank.sha256 != isolation.get("initial_state_bank_sha256"):
            raise ValueError("Initial-state bank changed after the offline audit.")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    (
        predictor_dir,
        calibration_path,
        aggregate_calibration_path,
        _,
        pilot_binding,
    ) = _load_frozen_aggregate_binding(
        post_summary,
        post_summary_path=post_summary_path,
        checkpoint_dir=checkpoint_dir,
    )
    pilot_binding.update(
        {
            "seed": args.seed,
            "num_trials_per_task": args.num_trials_per_task,
            "model_action_horizon": PILOT_MODEL_ACTION_HORIZON,
            "policy_config": args.policy_config,
        }
    )
    pilot_output_dir.mkdir(parents=True, exist_ok=True)
    existing_summaries = sorted(pilot_output_dir.rglob("summary.json"))
    if len(existing_summaries) > 1:
        raise RuntimeError(f"Multiple pilot summaries found under {pilot_output_dir}: {existing_summaries}")
    expected_episodes = 10 * args.num_trials_per_task
    binding_path = pilot_output_dir / "pilot_binding.json"
    run_config_path = pilot_output_dir / "run_config.json"
    if existing_summaries:
        if not binding_path.is_file():
            raise FileNotFoundError("Completed pilot has no immutable pilot_binding.json.")
        _ensure_pilot_binding(binding_path, pilot_binding)
        _verify_eval_run_config(
            run_config_path,
            aggregate_path=aggregate_calibration_path,
            seed=args.seed,
            num_trials_per_task=args.num_trials_per_task,
        )
        pilot_summary = _complete_json(existing_summaries[0])
        _verify_pilot_summary(pilot_summary, expected_episodes)
        result = {
            "status": "complete",
            "pilot_started": True,
            "post_summary": str(post_summary_path),
            "predictor_dir": str(predictor_dir),
            "calibration_path": str(calibration_path),
            "aggregate_calibration_path": str(aggregate_calibration_path),
            "pilot_binding": pilot_binding,
            "pilot_binding_path": str(binding_path),
            "pilot_summary": str(existing_summaries[0]),
            "pilot_summary_status": pilot_summary["status"],
            "expected_episodes": expected_episodes,
            "resumed_completed_pilot": True,
        }
        if existing_result is not None:
            if existing_result.get("pilot_binding") != pilot_binding:
                raise ValueError("Existing controller result is bound to a different pilot identity.")
            if pathlib.Path(existing_result.get("pilot_summary", "")).resolve() != existing_summaries[0].resolve():
                raise ValueError("Existing controller result references a different pilot summary.")
            result = existing_result
        else:
            _write_json(result_path, result)
        _write_json(status_path, {"status": "complete", "pilot_summary": str(existing_summaries[0])})
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if existing_result is not None:
        raise ValueError("Controller result exists but the bound pilot summary is absent.")
    resume_pilot = run_config_path.exists()
    if resume_pilot:
        if not binding_path.is_file():
            raise FileNotFoundError("Resumable pilot journal predates the required immutable pilot binding.")
        _ensure_pilot_binding(binding_path, pilot_binding)
        _verify_eval_run_config(
            run_config_path,
            aggregate_path=aggregate_calibration_path,
            seed=args.seed,
            num_trials_per_task=args.num_trials_per_task,
        )
    else:
        existing_entries = set(pilot_output_dir.iterdir())
        if existing_entries and existing_entries != {binding_path}:
            raise FileExistsError(f"Pilot output is non-empty but has no resume signature: {pilot_output_dir}")
        _ensure_pilot_binding(binding_path, pilot_binding)

    _write_json(
        status_path,
        {
            "status": "starting_dynamic_pilot_server",
            "predictor_dir": str(predictor_dir),
            "pilot_output_dir": str(pilot_output_dir),
        },
    )
    _notify(
        "【H25 predictor门控】冻结候选已同时通过development与fresh-final的预注册95%长H门, 开始动态10 tasks x 20闭环pilot。",
        output_dir,
        code_dir,
    )
    if _tmux_present(args.policy_server_tmux):
        raise RuntimeError(f"Scoped pilot server tmux already exists: {args.policy_server_tmux}")
    server_log = output_dir / "policy_server.log"
    # A new tmux session inherits the tmux server's environment, not this process's.
    # Pin the same source snapshot and forward only non-secret runtime settings.
    runtime_environment = {
        key: os.environ[key]
        for key in (
            "HF_HOME",
            "LD_LIBRARY_PATH",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "JAX_PLATFORMS",
        )
        if key in os.environ
    }
    source_paths = [str(code_dir), str(code_dir.parent / "src"), str(code_dir.parent / "packages/openpi-client/src")]
    if os.environ.get("PYTHONPATH"):
        source_paths.append(os.environ["PYTHONPATH"])
    runtime_environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    server_command = [
        "env",
        *[f"{key}={value}" for key, value in sorted(runtime_environment.items())],
        sys.executable,
        str(code_dir / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(args.port),
        "policy:checkpoint",
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        str(checkpoint_dir),
        "--policy.execution-horizon-predictor-params",
        str(predictor_dir),
    ]
    shell_command = f"{shlex.join(server_command)} >{shlex.quote(str(server_log))} 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.policy_server_tmux, "-c", str(code_dir.parent), shell_command],
        check=True,
    )
    try:
        _wait_for_server(args.host, args.port, args.policy_server_tmux, args.server_timeout_seconds)
        _write_json(
            status_path,
            {
                "status": "dynamic_pilot_running",
                "predictor_dir": str(predictor_dir),
                "pilot_output_dir": str(pilot_output_dir),
            },
        )
        eval_command = [
            sys.executable,
            str(code_dir / "eval_libero_execution_horizon.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output-dir",
            str(pilot_output_dir),
            "--modes",
            "hierarchical_transformer",
            "--model-action-horizon",
            str(PILOT_MODEL_ACTION_HORIZON),
            "--task-suite-name",
            "libero_10",
            "--max-tasks",
            "10",
            "--num-trials-per-task",
            str(args.num_trials_per_task),
            "--seed",
            str(args.seed),
            "--action-cot-denoising-steps",
            "10",
            "--teacher-samples",
            "20",
        ]
        eval_command.extend(("--hierarchical-aggregate-calibration-json", str(aggregate_calibration_path)))
        if resume_pilot:
            eval_command.append("--resume")
        with (output_dir / "dynamic_pilot.log").open("a" if resume_pilot else "w", encoding="utf-8") as log:
            subprocess.run(eval_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        _stop_tmux(args.policy_server_tmux)

    pilot_summary_path = _find_pilot_summary(pilot_output_dir)
    pilot_summary = _complete_json(pilot_summary_path)
    _verify_pilot_summary(pilot_summary, expected_episodes)
    result = {
        "status": "complete",
        "pilot_started": True,
        "post_summary": str(post_summary_path),
        "predictor_dir": str(predictor_dir),
        "calibration_path": str(calibration_path),
        "aggregate_calibration_path": str(aggregate_calibration_path),
        "pilot_binding": pilot_binding,
        "pilot_binding_path": str(binding_path),
        "pilot_summary": str(pilot_summary_path),
        "pilot_summary_status": pilot_summary["status"],
        "expected_episodes": expected_episodes,
    }
    _write_json(result_path, result)
    _write_json(status_path, {"status": "complete", "pilot_summary": str(pilot_summary_path)})
    _notify(
        f"【H25 predictor完成】通过双重95%门控的分层Transformer已完成动态10x{args.num_trials_per_task} pilot; 下一步审计success、实际elapsed及H分布。",
        output_dir,
        code_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
