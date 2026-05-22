import collections
import csv
import dataclasses
import json
import logging
import math
import pathlib
import re

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


import matplotlib.pyplot as plt
import os


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    exp_name: str = ("debug")
    resume_id: int = 0
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "./libero_videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    save_video: bool = True
    timing_out_dir: str | None = None


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220 * 3
    elif args.task_suite_name == "libero_object":
        max_steps = 280 * 3
    elif args.task_suite_name == "libero_goal":
        max_steps = 300 * 3
    elif args.task_suite_name == "libero_10":
        max_steps = 520 * 3
    elif args.task_suite_name == "libero_90":
        max_steps = 400 * 3
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    video_out_path = args.video_out_path
    video_out_path_per_task = pathlib.Path(video_out_path) / args.exp_name / args.task_suite_name
    video_out_path_per_task_success = video_out_path_per_task / "success"
    video_out_path_per_task_failure = video_out_path_per_task / "failure"
    timing_root = _get_timing_root(args)
    timing_stats = _new_timing_stats()

    if args.save_video:
        pathlib.Path(video_out_path_per_task_success).mkdir(parents=True, exist_ok=True)
        pathlib.Path(video_out_path_per_task_failure).mkdir(parents=True, exist_ok=True)
    if timing_root is not None:
        timing_root.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            logging.warning(
                "Requested %d trials for task %d, but LIBERO provides %d initial states. "
                "Initial states will be reused modulo %d.",
                args.num_trials_per_task,
                task_id,
                len(initial_states),
                len(initial_states),
            )

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            episode_timing_records = []
            episode_infer_idx = 0
            done = False

            # Set initial states
            init_state_idx = episode_idx % len(initial_states)
            obs = env.set_init_state(initial_states[init_state_idx])

            # resume logic
            if total_episodes < args.resume_id:
                task_episodes += 1
                total_episodes += 1
                continue


            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        ret_result = client.infer(element)
                        action_chunk = ret_result["actions"]

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                        _record_timing_if_available(
                            timing_stats=timing_stats,
                            episode_timing_records=episode_timing_records,
                            args=args,
                            task_id=task_id,
                            task_description=str(task_description),
                            episode_idx=episode_idx,
                            init_state_idx=init_state_idx,
                            infer_idx=episode_infer_idx,
                            response=ret_result,
                        )
                        episode_infer_idx += 1

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Caught exception while running task %d trial %d", task_id, episode_idx)
                    break

            task_episodes += 1
            total_episodes += 1
            _finish_episode_timing_records(
                episode_timing_records,
                success=done,
                episode_steps=t,
            )
            _write_episode_timing_records(
                timing_root=timing_root,
                task_id=task_id,
                task_description=str(task_description),
                episode_idx=episode_idx,
                records=episode_timing_records,
            )

            # Save a replay video of the episode
            if args.save_video:
                suffix = "success" if done else "failure"
                if suffix == "failure":
                    imageio.mimwrite(
                        video_out_path_per_task_failure / f"rollout_{task_id}_{episode_idx}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                if suffix == "success":
                    imageio.mimwrite(
                        video_out_path_per_task_success / f"rollout_{task_id}_{episode_idx}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    _write_timing_summary(timing_root=timing_root, args=args, timing_stats=timing_stats)


def _get_timing_root(args: Args) -> pathlib.Path | None:
    if args.timing_out_dir is None:
        return None
    return pathlib.Path(args.timing_out_dir) / args.exp_name / args.task_suite_name


def _new_timing_stats():
    return collections.defaultdict(
        lambda: {
            "task_id": None,
            "task_description": None,
            "infer_idx": None,
            "num_trials_with_this_inference": 0,
            "server_infer_ms_sum": 0.0,
            "server_infer_ms_count": 0,
            "policy_infer_ms_sum": 0.0,
            "policy_infer_ms_count": 0,
        }
    )


def _make_timing_record(
    *,
    args: Args,
    task_id: int,
    task_description: str,
    episode_idx: int,
    init_state_idx: int,
    infer_idx: int,
    response: dict,
) -> dict:
    return {
        "task_suite_name": args.task_suite_name,
        "exp_name": args.exp_name,
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "init_state_idx": init_state_idx,
        "infer_idx": infer_idx,
        "server_timing": {
            "infer_ms": _as_float(response.get("server_timing", {}).get("infer_ms")),
        },
        "policy_timing": {
            "infer_ms": _as_float(response.get("policy_timing", {}).get("infer_ms")),
        },
    }


def _record_timing_if_available(
    *,
    timing_stats,
    episode_timing_records: list[dict],
    args: Args,
    task_id: int,
    task_description: str,
    episode_idx: int,
    init_state_idx: int,
    infer_idx: int,
    response: dict,
) -> None:
    try:
        timing_record = _make_timing_record(
            args=args,
            task_id=task_id,
            task_description=task_description,
            episode_idx=episode_idx,
            init_state_idx=init_state_idx,
            infer_idx=infer_idx,
            response=response,
        )
        episode_timing_records.append(timing_record)
        _accumulate_timing(timing_stats, timing_record)
    except Exception:
        logging.exception("Failed to record timing for task %d trial %d infer %d", task_id, episode_idx, infer_idx)


def _accumulate_timing(timing_stats, record: dict) -> None:
    key = (record["task_id"], record["infer_idx"])
    bucket = timing_stats[key]
    bucket["task_id"] = record["task_id"]
    bucket["task_description"] = record["task_description"]
    bucket["infer_idx"] = record["infer_idx"]
    bucket["num_trials_with_this_inference"] += 1

    server_infer_ms = record["server_timing"]["infer_ms"]
    if server_infer_ms is not None:
        bucket["server_infer_ms_sum"] += server_infer_ms
        bucket["server_infer_ms_count"] += 1

    policy_infer_ms = record["policy_timing"]["infer_ms"]
    if policy_infer_ms is not None:
        bucket["policy_infer_ms_sum"] += policy_infer_ms
        bucket["policy_infer_ms_count"] += 1


def _finish_episode_timing_records(records: list[dict], *, success: bool, episode_steps: int) -> None:
    for record in records:
        record["success"] = bool(success)
        record["episode_steps"] = int(episode_steps)
        record["num_episode_inferences"] = len(records)


def _write_episode_timing_records(
    *,
    timing_root: pathlib.Path | None,
    task_id: int,
    task_description: str,
    episode_idx: int,
    records: list[dict],
) -> None:
    if timing_root is None:
        return

    task_name = _sanitize_filename(task_description)
    out_dir = timing_root / "episodes" / f"task_{task_id:02d}_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_{episode_idx:03d}.jsonl"
    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_timing_summary(*, timing_root: pathlib.Path | None, args: Args, timing_stats) -> None:
    if timing_root is None:
        return

    rows = []
    for _, bucket in sorted(timing_stats.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append(
            {
                "task_suite_name": args.task_suite_name,
                "exp_name": args.exp_name,
                "task_id": bucket["task_id"],
                "task_description": bucket["task_description"],
                "infer_idx": bucket["infer_idx"],
                "num_trials_requested": args.num_trials_per_task,
                "num_trials_with_this_inference": bucket["num_trials_with_this_inference"],
                "server_infer_ms_mean": _mean_or_none(
                    bucket["server_infer_ms_sum"], bucket["server_infer_ms_count"]
                ),
                "policy_infer_ms_mean": _mean_or_none(
                    bucket["policy_infer_ms_sum"], bucket["policy_infer_ms_count"]
                ),
            }
        )

    jsonl_path = timing_root / "summary_by_inference_index.jsonl"
    csv_path = timing_root / "summary_by_inference_index.csv"

    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    fieldnames = [
        "task_suite_name",
        "exp_name",
        "task_id",
        "task_description",
        "infer_idx",
        "num_trials_requested",
        "num_trials_with_this_inference",
        "server_infer_ms_mean",
        "policy_infer_ms_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in fieldnames})

    logging.info("Wrote timing summary to %s and %s", jsonl_path, csv_path)


def _mean_or_none(total: float, count: int) -> float | None:
    if count == 0:
        return None
    return total / count


def _as_float(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    return float(array.reshape(()))


def _sanitize_filename(value: str, *, max_len: int = 96) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return (value or "unknown_task")[:max_len]


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
