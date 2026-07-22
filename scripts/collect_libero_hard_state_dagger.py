"""Collect high-confidence correction trajectories from failed LIBERO rollouts.

The frozen student is first replayed with fixed H9. For episodes that time out,
the collector selects late snapshots along the visited trajectory and samples
multiple action chunks from each snapshot. A correction is accepted only when
the same initial chunk reaches terminal success under every independently
seeded fixed-H9 continuation. The successful trajectory is written as a small
LeRobot dataset and a root-balanced manifest for conservative SFT replay.

This is DAgger-style hindsight relabeling, not a deployable selector: terminal
outcomes are used offline to choose correction labels and are never available
to the policy at evaluation time.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections import Counter
import copy
import dataclasses
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import collect_action_chunk_candidate_oracle as candidate_audit
import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy


@dataclasses.dataclass
class HardRoot:
    snapshot: collector.SimulatorSnapshot
    step: int
    decision_index: int
    request_seed: int
    progress: float


@dataclasses.dataclass
class BranchResult:
    success: bool
    steps: int
    calls: int
    frames: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/task89_hard_state_dagger_round1")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser(),
    )
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument(
        "--task-episode",
        action="append",
        required=True,
        help="Failed rollout to aggregate, formatted TASK_ID:EPISODE_ID; repeat as needed.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--student-horizon", type=int, default=9)
    parser.add_argument(
        "--student-action-cot-denoising-steps",
        type=int,
        default=10,
        help="Deployment-time Action-CoT denoising steps used to reproduce the frozen student.",
    )
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-selection", choices=("farthest", "first"), default="farthest")
    parser.add_argument("--candidate-indices", nargs="+", type=int, default=None)
    parser.add_argument("--candidate-horizon", type=int, default=3)
    parser.add_argument("--continuation-horizon", type=int, default=9)
    parser.add_argument("--confidence-repeats", type=int, default=2)
    parser.add_argument("--repeat-seed-stride", type=int, default=20_000_000)
    parser.add_argument("--teacher-seed-offset", type=int, default=300_000_000)
    parser.add_argument("--hard-root-progress", nargs="+", type=float, default=(0.55, 0.70, 0.85))
    parser.add_argument("--max-corrections-per-episode", type=int, default=2)
    parser.add_argument("--min-trajectory-steps", type=int, default=29)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_task_episodes(values: list[str]) -> list[tuple[int, int]]:
    result = []
    seen = set()
    for value in values:
        try:
            task_text, episode_text = value.split(":", maxsplit=1)
            pair = (int(task_text), int(episode_text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid --task-episode {value!r}; expected TASK_ID:EPISODE_ID") from error
        if pair[0] < 0 or pair[1] < 0:
            raise ValueError(f"Task and episode IDs must be non-negative, got {pair}")
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    return result


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "student_horizon": args.student_horizon,
        "student_action_cot_denoising_steps": args.student_action_cot_denoising_steps,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "candidate_count": args.candidate_count,
        "candidate_horizon": args.candidate_horizon,
        "continuation_horizon": args.continuation_horizon,
        "confidence_repeats": args.confidence_repeats,
        "repeat_seed_stride": args.repeat_seed_stride,
        "max_corrections_per_episode": args.max_corrections_per_episode,
        "min_trajectory_steps": args.min_trajectory_steps,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.candidate_count > args.teacher_samples:
        raise ValueError("candidate-count cannot exceed teacher-samples")
    if args.student_horizon > 10 or args.candidate_horizon > 10 or args.continuation_horizon > 10:
        raise ValueError("All action horizons must be at most 10")
    if not args.hard_root_progress or any(not 0.0 < value < 1.0 for value in args.hard_root_progress):
        raise ValueError("hard-root-progress values must lie strictly between 0 and 1")


def _training_frame(observation: dict[str, Any], action: np.ndarray, task_description: str) -> dict[str, Any]:
    return {
        "image": np.ascontiguousarray(observation["agentview_image"][::-1, ::-1], dtype=np.uint8),
        "wrist_image": np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8),
        "state": np.asarray(
            np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    libero_eval._quat2axisangle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ),
            dtype=np.float32,
        ),
        "actions": np.asarray(action, dtype=np.float32),
        "task": str(task_description),
    }


def _replay_student(
    env: Any,
    observation: dict[str, Any],
    *,
    task_id: int,
    episode_id: int,
    task_description: str,
    episode_step_limit: int,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> tuple[bool, int, list[HardRoot]]:
    step = args.num_steps_wait
    decision_index = 0
    done = False
    roots = []
    while not done and step < episode_step_limit:
        request_seed = args.seed + task_id * 1_000_000 + episode_id * 10_000 + step
        roots.append(
            HardRoot(
                snapshot=collector._capture_snapshot(env),
                step=step,
                decision_index=decision_index,
                request_seed=request_seed,
                progress=step / max(episode_step_limit, 1),
            )
        )
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        result = collector._policy_request(client, policy_input, seed=request_seed, args=args, profile=True)
        actions = np.asarray(result["actions"], dtype=np.float32)[: args.student_horizon]
        for action in actions:
            if step >= episode_step_limit:
                break
            try:
                observation, _, done, _ = env.step(np.asarray(action).tolist())
            except Exception as exc:
                if not libero_eval._is_terminated_episode_error(exc):
                    raise
                done = libero_eval._env_success(env)
            step += 1
            if done or libero_eval._env_success(env):
                done = True
                break
        decision_index += 1
    return bool(done or libero_eval._env_success(env)), step, roots


def _select_hard_roots(roots: list[HardRoot], episode_step_limit: int, args: argparse.Namespace) -> list[HardRoot]:
    eligible = [root for root in roots if episode_step_limit - root.step >= args.min_trajectory_steps]
    selected = []
    for target in args.hard_root_progress:
        remaining = [root for root in eligible if root.decision_index not in {item.decision_index for item in selected}]
        if not remaining:
            break
        selected.append(min(remaining, key=lambda root: abs(root.progress - target)))
    return sorted(selected, key=lambda root: root.progress, reverse=True)


def _run_teacher_branch(
    env: Any,
    hard_root: HardRoot,
    initial_actions: np.ndarray,
    *,
    branch_seed: int,
    task_description: str,
    episode_step_limit: int,
    record: bool,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> BranchResult:
    observation = collector._restore_snapshot(env, hard_root.snapshot)
    action_plan = np.asarray(initial_actions, dtype=np.float32)[: args.candidate_horizon]
    steps = 0
    calls = 1
    continuation_index = 0
    frames = []
    done = False
    while hard_root.step + steps < episode_step_limit:
        for action in action_plan:
            if hard_root.step + steps >= episode_step_limit:
                break
            if record:
                frames.append(_training_frame(observation, action, task_description))
            try:
                observation, _, done, _ = env.step(np.asarray(action).tolist())
            except Exception as exc:
                if not libero_eval._is_terminated_episode_error(exc):
                    raise
                done = libero_eval._env_success(env)
            steps += 1
            if done or libero_eval._env_success(env):
                done = True
                break
        if done or hard_root.step + steps >= episode_step_limit:
            break

        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        continuation_seed = branch_seed + 100_000 + continuation_index
        result = collector._policy_request(client, policy_input, seed=continuation_seed, args=args, profile=True)
        action_plan = np.asarray(result["actions"], dtype=np.float32)[: args.continuation_horizon]
        calls += 1
        continuation_index += 1

    return BranchResult(
        success=bool(done or libero_eval._env_success(env)),
        steps=steps,
        calls=calls,
        frames=frames,
    )


def _search_correction(
    env: Any,
    hard_root: HardRoot,
    *,
    task_description: str,
    episode_step_limit: int,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> tuple[BranchResult | None, dict[str, Any]]:
    observation = collector._restore_snapshot(env, hard_root.snapshot)
    policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
    teacher_seed = hard_root.request_seed + args.teacher_seed_offset
    result = collector._policy_request(client, policy_input, seed=teacher_seed, args=args, teacher=True)
    candidates = np.asarray(result["mc_actions"], dtype=np.float32)
    normalized = np.asarray(result["mc_actions_normalized"], dtype=np.float32)
    selected_indices = candidate_audit._candidate_indices(normalized, args)
    stable = []
    candidate_records = []
    for rank, candidate_index in enumerate(selected_indices):
        repeat_records = []
        recorded_trajectory = None
        for repeat_index in range(args.confidence_repeats):
            if repeat_index > 0 and repeat_records and not all(item["success"] for item in repeat_records):
                break
            branch_seed = teacher_seed + repeat_index * args.repeat_seed_stride
            outcome = _run_teacher_branch(
                env,
                hard_root,
                candidates[candidate_index],
                branch_seed=branch_seed,
                task_description=task_description,
                episode_step_limit=episode_step_limit,
                record=repeat_index == 0,
                args=args,
                client=client,
            )
            if repeat_index == 0 and outcome.success:
                recorded_trajectory = outcome
            repeat_records.append(
                {
                    "repeat_index": repeat_index,
                    "continuation_seed": branch_seed,
                    "success": outcome.success,
                    "steps": outcome.steps,
                    "calls": outcome.calls,
                }
            )
        accepted = (
            len(repeat_records) == args.confidence_repeats
            and all(item["success"] for item in repeat_records)
            and recorded_trajectory is not None
            and len(recorded_trajectory.frames) >= args.min_trajectory_steps
        )
        candidate_records.append(
            {
                "rank": rank,
                "candidate_index": candidate_index,
                "accepted": accepted,
                "repeat_outcomes": repeat_records,
            }
        )
        if accepted:
            stable.append((recorded_trajectory, rank, candidate_index))

    if not stable:
        return None, {
            "teacher_seed": teacher_seed,
            "selected_candidate_indices": selected_indices,
            "candidates": candidate_records,
            "accepted": False,
        }
    trajectory, rank, candidate_index = min(stable, key=lambda item: (item[0].steps, item[1]))
    return trajectory, {
        "teacher_seed": teacher_seed,
        "selected_candidate_indices": selected_indices,
        "candidates": candidate_records,
        "accepted": True,
        "selected_rank": rank,
        "selected_candidate_index": candidate_index,
        "teacher_confidence": 1.0,
        "trajectory_steps": trajectory.steps,
        "trajectory_calls": trajectory.calls,
    }


def _dataset_features() -> dict[str, dict[str, Any]]:
    return {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def _prepare_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_root = args.lerobot_home / args.dataset_repo_id
    for path in (args.output_dir, dataset_root):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
            shutil.rmtree(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    return args.output_dir, dataset_root


def _write_training_manifest(path: Path, correction_records: list[dict[str, Any]], dataset_repo_id: str) -> None:
    corrections_by_root = Counter(record["root_id"] for record in correction_records)
    with path.open("w", encoding="utf-8") as handle:
        for record in correction_records:
            frame_count = int(record["trajectory_steps"])
            root_count = corrections_by_root[record["root_id"]]
            entry = {
                "repo_id": dataset_repo_id,
                "episode_index": int(record["correction_episode_index"]),
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "split": "train",
                "task": record["task_description"],
                "phase": "hard_state_correction",
                "source": "dagger_hindsight_success",
                "root_id": record["root_id"],
                "teacher_confidence": float(record["teacher_confidence"]),
                "weight": float(record["teacher_confidence"]) / (frame_count * root_count),
            }
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    task_episodes = _parse_task_episodes(args.task_episode)
    student_args = copy.copy(args)
    student_args.action_cot_denoising_steps = args.student_action_cot_denoising_steps
    output_dir, dataset_root = _prepare_outputs(args)
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=10,
        features=_dataset_features(),
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
    )
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = libero_eval._max_steps(args.task_suite_name)
    collection_path = output_dir / "collection_records.jsonl"
    correction_records = []
    attempted_roots = 0
    base_failures = 0
    started = time.monotonic()
    with collection_path.open("w", encoding="utf-8") as collection_writer:
        for task_id, episode_id in task_episodes:
            if task_id >= task_suite.n_tasks:
                raise ValueError(f"Task {task_id} is outside suite with {task_suite.n_tasks} tasks")
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            initial_state_id = episode_id % len(initial_states)
            env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
            try:
                env.reset()
                observation = env.set_init_state(initial_states[initial_state_id])
                environment_horizon = libero_eval._env_horizon(env)
                episode_step_limit = max_steps + args.num_steps_wait
                if environment_horizon is not None:
                    episode_step_limit = min(episode_step_limit, environment_horizon)
                done = False
                for _ in range(args.num_steps_wait):
                    observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
                    if done:
                        break
                base_success, base_steps, roots = _replay_student(
                    env,
                    observation,
                    task_id=task_id,
                    episode_id=episode_id,
                    task_description=task_description,
                    episode_step_limit=episode_step_limit,
                    args=student_args,
                    client=client,
                )
                if base_success:
                    record = {
                        "schema_version": 1,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "initial_state_id": initial_state_id,
                        "task_description": task_description,
                        "base_success": True,
                        "base_steps": base_steps,
                        "corrections": [],
                        "skip_reason": "base_replay_succeeded",
                    }
                    collection_writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    collection_writer.flush()
                    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
                    continue

                base_failures += 1
                selected_roots = _select_hard_roots(roots, episode_step_limit, args)
                episode_corrections = []
                for hard_root in selected_roots:
                    if len(episode_corrections) >= args.max_corrections_per_episode:
                        break
                    attempted_roots += 1
                    trajectory, teacher_record = _search_correction(
                        env,
                        hard_root,
                        task_description=task_description,
                        episode_step_limit=episode_step_limit,
                        args=args,
                        client=client,
                    )
                    root_id = f"task:{task_id}:initial_state:{initial_state_id}"
                    correction = {
                        "root_id": root_id,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "initial_state_id": initial_state_id,
                        "task_description": task_description,
                        "decision_index": hard_root.decision_index,
                        "decision_step": hard_root.step,
                        "episode_progress": hard_root.progress,
                        **teacher_record,
                    }
                    if trajectory is not None:
                        correction_episode_index = dataset.meta.total_episodes
                        for frame in trajectory.frames:
                            dataset.add_frame(frame)
                        dataset.save_episode()
                        correction["correction_episode_index"] = correction_episode_index
                        correction_records.append(correction)
                        episode_corrections.append(correction)
                    collection_writer.write(json.dumps(correction, ensure_ascii=False, sort_keys=True) + "\n")
                    collection_writer.flush()
                    print(
                        json.dumps(
                            {
                                "task": task_id,
                                "episode": episode_id,
                                "initial_state": initial_state_id,
                                "root_step": hard_root.step,
                                "accepted": trajectory is not None,
                                "correction_episodes": len(correction_records),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                libero_eval._safe_close_env(env)

    dataset.stop_image_writer()
    manifest_path = output_dir / "training_manifest.jsonl"
    _write_training_manifest(manifest_path, correction_records, args.dataset_repo_id)
    summary = {
        "status": "complete",
        "schema_version": 1,
        "task_episodes": task_episodes,
        "base_failures": base_failures,
        "attempted_hard_roots": attempted_roots,
        "accepted_correction_episodes": len(correction_records),
        "accepted_correction_frames": sum(int(record["trajectory_steps"]) for record in correction_records),
        "accepted_roots": len({record["root_id"] for record in correction_records}),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "collection_records": str(collection_path),
        "training_manifest": str(manifest_path),
        "confidence_repeats": args.confidence_repeats,
        "student_action_cot_denoising_steps": args.student_action_cot_denoising_steps,
        "teacher_action_cot_denoising_steps": args.action_cot_denoising_steps,
        "min_trajectory_steps": args.min_trajectory_steps,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
