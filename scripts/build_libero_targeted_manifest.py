"""Build a reproducible Task8/9 key-phase manifest from a LeRobot dataset.

The generated JSONL contains episode-local frame ranges around gripper state
transitions plus the terminal phase of each target-task demonstration. It is
consumed by the ``mixture`` frame sampler; it does not modify the source data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_TARGET_TASKS = (
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="your_hf_username/libero")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser(),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-task",
        action="append",
        dest="target_tasks",
        help="Case-insensitive task substring; may be passed multiple times.",
    )
    parser.add_argument("--transition-before", type=int, default=20)
    parser.add_argument("--transition-after", type=int, default=30)
    parser.add_argument("--tail-frames", type=int, default=80)
    parser.add_argument("--split", default="train")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _task_matches(task: str, patterns: tuple[str, ...]) -> bool:
    normalized_task = _normalize_text(task)
    return any(_normalize_text(pattern) in normalized_task for pattern in patterns)


def _merge_intervals(
    intervals: list[tuple[int, int, str]],
) -> list[tuple[int, int, tuple[str, ...]]]:
    merged: list[tuple[int, int, set[str]]] = []
    for start, end, phase in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, {phase}))
            continue
        previous_start, previous_end, phases = merged[-1]
        phases.add(phase)
        merged[-1] = (previous_start, max(previous_end, end), phases)
    return [(start, end, tuple(sorted(phases))) for start, end, phases in merged]


def _gripper_transitions(actions: list[list[float]]) -> list[int]:
    if not actions:
        return []
    states = [float(action[-1]) >= 0.0 for action in actions]
    return [index for index in range(1, len(states)) if states[index] != states[index - 1]]


def main() -> None:
    args = _parse_args()
    if args.transition_before < 0 or args.transition_after < 0 or args.tail_frames < 0:
        raise ValueError("Frame-window sizes must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output}")

    target_tasks = tuple(args.target_tasks or DEFAULT_TARGET_TASKS)
    repo_path = args.lerobot_home / args.repo_id
    meta_path = repo_path / "meta"
    with (meta_path / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = _read_jsonl(meta_path / "episodes.jsonl")

    manifest_entries: list[dict[str, Any]] = []
    task_episode_counts: Counter[str] = Counter()
    task_frame_counts: Counter[str] = Counter()
    chunks_size = int(info["chunks_size"])
    data_path_template = str(info["data_path"])

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task = str(episode["tasks"][0])
        if not _task_matches(task, target_tasks):
            continue

        parquet_path = repo_path / data_path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        action_column = pq.read_table(parquet_path, columns=["actions"])["actions"]
        actions = action_column.to_pylist()
        episode_length = len(actions)
        if episode_length != int(episode["length"]):
            raise ValueError(
                f"Episode {episode_index} metadata length {episode['length']} "
                f"does not match parquet length {episode_length}"
            )

        intervals = [
            (
                max(0, transition - args.transition_before),
                min(episode_length - 1, transition + args.transition_after),
                "gripper_transition",
            )
            for transition in _gripper_transitions(actions)
        ]
        if args.tail_frames:
            intervals.append(
                (
                    max(0, episode_length - args.tail_frames),
                    episode_length - 1,
                    "terminal_phase",
                )
            )

        task_episode_counts[task] += 1
        for start_frame, end_frame, phases in _merge_intervals(intervals):
            frame_count = end_frame - start_frame + 1
            task_frame_counts[task] += frame_count
            manifest_entries.append(
                {
                    "repo_id": args.repo_id,
                    "episode_index": episode_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "split": args.split,
                    "task": task,
                    "phase": "+".join(phases),
                    "source": "expert_targeted",
                    "root_id": f"{args.repo_id}:episode:{episode_index}",
                    "teacher_confidence": 1.0,
                    "weight": 1.0,
                }
            )

    if not manifest_entries:
        raise ValueError(f"No episodes matched target tasks: {target_tasks!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "repo_id": args.repo_id,
        "target_tasks": target_tasks,
        "transition_before": args.transition_before,
        "transition_after": args.transition_after,
        "tail_frames": args.tail_frames,
        "manifest_entries": len(manifest_entries),
        "target_episodes": sum(task_episode_counts.values()),
        "manifest_frames": sum(task_frame_counts.values()),
        "episodes_by_task": dict(task_episode_counts),
        "manifest_frames_by_task": dict(task_frame_counts),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
