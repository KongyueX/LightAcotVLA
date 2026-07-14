"""Capture and compare deterministic base-policy actions across code snapshots."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

DEFAULT_CASES = ("0:0", "4:16", "6:4", "6:14", "8:0", "8:3", "9:0", "9:13")


def _capture_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("capture", help="Capture deterministic first-decision action chunks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-label", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)


def _compare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare", help="Compare two captures on identical policy inputs.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _capture_parser(subparsers)
    _compare_parser(subparsers)
    return parser


def _parse_cases(values: list[str]) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for value in values:
        try:
            task_id, episode = (int(part) for part in value.split(":", maxsplit=1))
        except ValueError as exc:
            raise ValueError(f"Invalid case {value!r}; expected TASK_ID:EPISODE.") from exc
        if task_id < 0 or episode < 0:
            raise ValueError(f"Case values must be non-negative, got {value!r}.")
        cases.append((task_id, episode))
    if len(cases) != len(set(cases)):
        raise ValueError("--cases contains duplicate task/episode pairs.")
    return cases


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode())
        return
    array = np.asarray(value)
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())


def _input_digest(element: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(element):
        digest.update(key.encode())
        digest.update(b"\0")
        _update_digest(digest, element[key])
        digest.update(b"\0")
    return digest.hexdigest()


def _capture(args: argparse.Namespace) -> None:
    cases = _parse_cases(args.cases)
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    records: list[dict[str, Any]] = []
    for task_id, episode in cases:
        if task_id >= task_suite.n_tasks:
            raise ValueError(f"Task {task_id} is outside suite {args.task_suite_name!r}.")
        task = task_suite.get_task(task_id)
        states = task_suite.get_task_init_states(task_id)
        state_id = episode % len(states)
        env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
        try:
            env.reset()
            observation = env.set_init_state(states[state_id])
            step = 0
            for _ in range(args.num_steps_wait):
                observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
                step += 1
                if done:
                    break
            element = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        finally:
            libero_eval._safe_close_env(env)

        request_seed = args.seed + task_id * 1_000_000 + episode * 10_000 + step
        result = client.infer(
            {
                **element,
                "policy_seed": np.asarray(request_seed, dtype=np.int64),
                "profile_policy_timing": np.asarray(1, dtype=np.bool_),
                "action_cot_denoising_steps": np.asarray(args.action_cot_denoising_steps, dtype=np.int32),
            }
        )
        records.append(
            {
                "task_id": task_id,
                "episode": episode,
                "state_id": state_id,
                "environment_step": step,
                "request_seed": request_seed,
                "input_digest": _input_digest(element),
                "actions": np.asarray(result["actions"], dtype=np.float32),
                "coarse_actions": np.asarray(result["coarse_actions"], dtype=np.float32),
            }
        )
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "episode": episode,
                    "request_seed": request_seed,
                    "input_digest": records[-1]["input_digest"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "snapshot_label": args.snapshot_label,
        "task_suite_name": args.task_suite_name,
        "seed": args.seed,
        "resize_size": args.resize_size,
        "num_steps_wait": args.num_steps_wait,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
    }
    np.savez_compressed(
        output,
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        task_id=np.asarray([record["task_id"] for record in records], dtype=np.int16),
        episode=np.asarray([record["episode"] for record in records], dtype=np.int16),
        state_id=np.asarray([record["state_id"] for record in records], dtype=np.int16),
        environment_step=np.asarray([record["environment_step"] for record in records], dtype=np.int16),
        request_seed=np.asarray([record["request_seed"] for record in records], dtype=np.int64),
        input_digest=np.asarray([record["input_digest"] for record in records]),
        actions=np.stack([record["actions"] for record in records]),
        coarse_actions=np.stack([record["coarse_actions"] for record in records]),
    )


def _load_capture(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as capture:
        return {key: np.asarray(capture[key]) for key in capture.files}


def _compare(args: argparse.Namespace) -> None:
    left = _load_capture(args.left)
    right = _load_capture(args.right)
    identity_fields = (
        "task_id",
        "episode",
        "state_id",
        "environment_step",
        "request_seed",
        "input_digest",
    )
    mismatched_identity = [field for field in identity_fields if not np.array_equal(left[field], right[field])]
    if mismatched_identity:
        raise ValueError(f"Captures do not contain identical policy inputs: {mismatched_identity}")

    rows = []
    for index, (task_id, episode) in enumerate(zip(left["task_id"], left["episode"], strict=True)):
        row: dict[str, Any] = {"task_id": int(task_id), "episode": int(episode)}
        for field in ("actions", "coarse_actions"):
            difference = np.abs(left[field][index].astype(np.float64) - right[field][index].astype(np.float64))
            row[f"{field}_exact_equal"] = bool(np.array_equal(left[field][index], right[field][index]))
            row[f"{field}_max_abs"] = float(np.max(difference))
            row[f"{field}_mean_abs"] = float(np.mean(difference))
        rows.append(row)

    summary = {
        "status": "complete",
        "scope": "same first-decision LIBERO observations and deterministic policy seeds; not a full-rollout audit",
        "left": str(pathlib.Path(args.left).resolve()),
        "right": str(pathlib.Path(args.right).resolve()),
        "cases": len(rows),
        "input_digests_equal": True,
        "actions_all_exact_equal": all(row["actions_exact_equal"] for row in rows),
        "coarse_actions_all_exact_equal": all(row["coarse_actions_exact_equal"] for row in rows),
        "actions_max_abs": max(row["actions_max_abs"] for row in rows),
        "coarse_actions_max_abs": max(row["coarse_actions_max_abs"] for row in rows),
        "rows": rows,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def main(args: argparse.Namespace) -> None:
    if args.command == "capture":
        _capture(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
