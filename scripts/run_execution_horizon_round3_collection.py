#!/usr/bin/env python3
"""Collect Round-3 on-policy execution-horizon roots from fresh episode groups.

This controller is intentionally development-only. It pre-registers every
primary and same-role fallback episode before starting a collector, keeps
``dev_audit`` labels out of selection logic, and resumes through immutable
attempt directories rather than overwriting interrupted output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any

import collect_execution_horizon_counterfactuals as horizon_collector
import execution_horizon_aggregate_risk_common as aggregate_common
import h5py
import numpy as np

from openpi.execution_horizon import hierarchical
from openpi.execution_horizon import initial_states as horizon_initial_states
from openpi.execution_horizon import splits as horizon_splits

PROTOCOL_NAME = "h25_round3_fresh_groups_v1"
SELECTION_SEED = 20260904
COLLECTOR_SEED = 37007
ACQUISITION_TRAINING_SEED = 7
TASK_IDS = tuple(range(10))
EPISODE_START = 100
EPISODE_END = 299
ROOT_CALL_OFFSET_CYCLE = 12
MINIMUM_CALL_OFFSET = 3
MAXIMUM_CALL_OFFSET = 10
ROLE_TARGETS = {"train": 10, "calibration": 10, "dev_audit": 40}
CANDIDATE_HORIZONS = (5, 10, 15, 20, 25)
REFERENCE_HORIZON = 10
SOURCE_ITERATION = 3
ROOT_SEED_TASK_STRIDE = 250_000_000
BRANCH_REPEATS = 3
BRANCH_REPEAT_SEED_STRIDE = 20_000_000
FIXED_CONTINUATION_HORIZON = 5
MAX_POLICY_SEED = horizon_collector.MAX_POLICY_SEED
# libero_10 uses 520 * 3 rollout steps plus the 10-step reset wait.
MAXIMUM_EPISODE_STEP_FOR_SEED_AUDIT = 1_570


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True, help="Immutable checkout/snapshot containing the collector.")
    parser.add_argument("--code-commit", required=True, help="Expected commit identity for --code-dir.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-initial-state-bank", required=True)
    parser.add_argument("--initial-state-bank", required=True)
    parser.add_argument("--base-split-manifest", required=True)
    parser.add_argument("--predictor-dir", required=True)
    parser.add_argument("--aggregate-calibration-json", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--collector-script", default=None)
    parser.add_argument("--episodes-per-attempt", type=int, default=10)
    return parser


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_once_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    normalized = json.loads(json.dumps(payload))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Existing immutable JSON differs from the requested protocol: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")


def eligible_episodes(
    *,
    episode_start: int = EPISODE_START,
    episode_end: int = EPISODE_END,
    cycle: int = ROOT_CALL_OFFSET_CYCLE,
    minimum_offset: int = MINIMUM_CALL_OFFSET,
    maximum_offset: int = MAXIMUM_CALL_OFFSET,
) -> tuple[int, ...]:
    if episode_start < 0 or episode_end < episode_start or cycle <= 0:
        raise ValueError("Invalid episode range or root-call-offset cycle.")
    if not 0 <= minimum_offset <= maximum_offset < cycle:
        raise ValueError("Call offsets must define a non-empty interval within the cycle.")
    return tuple(
        episode
        for episode in range(episode_start, episode_end + 1)
        if minimum_offset <= episode % cycle <= maximum_offset
    )


def validate_root_seed_namespace(
    *,
    task_stride: int = ROOT_SEED_TASK_STRIDE,
    maximum_episode_step: int = MAXIMUM_EPISODE_STEP_FOR_SEED_AUDIT,
) -> tuple[tuple[int, int], ...]:
    """Use the collector's exact scheme to prove the full Round3 namespace."""

    if task_stride <= 0 or not 0 <= maximum_episode_step < 10_000:
        raise ValueError("Invalid task stride or maximum episode step for seed audit.")
    episodes = eligible_episodes()
    horizon_collector._validate_seed_namespace(  # noqa: SLF001
        base_seed=COLLECTOR_SEED,
        task_ids=list(TASK_IDS),
        episode_ids=list(episodes),
        maximum_episode_step=maximum_episode_step,
        maximum_continuation_calls=maximum_episode_step,
        task_stride=task_stride,
        branch_repeats=BRANCH_REPEATS,
        branch_repeat_seed_stride=BRANCH_REPEAT_SEED_STRIDE,
        teacher_samples=20,
    )
    intervals = tuple(
        (
            COLLECTOR_SEED + task * task_stride,
            COLLECTOR_SEED + (task + 1) * task_stride - 1,
        )
        for task in TASK_IDS
    )
    if intervals[-1][1] > MAX_POLICY_SEED:
        raise ValueError(f"Round3 seed namespace exceeds uint32: {intervals[-1][1]}.")
    return intervals


def _round3_seed_metadata() -> dict[str, int | str]:
    return horizon_collector._seed_scheme_metadata(  # noqa: SLF001
        argparse.Namespace(
            root_seed_task_stride=ROOT_SEED_TASK_STRIDE,
            branch_repeat_seed_stride=BRANCH_REPEAT_SEED_STRIDE,
        )
    )


def _take_balanced_primary(
    pools: dict[int, list[int]],
    *,
    count: int,
    offset_rotation: int,
) -> list[int]:
    offsets = list(range(MINIMUM_CALL_OFFSET, MAXIMUM_CALL_OFFSET + 1))
    rotation = offset_rotation % len(offsets)
    offsets = offsets[rotation:] + offsets[:rotation]
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for offset in offsets:
            if pools[offset] and len(selected) < count:
                selected.append(pools[offset].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Could select only {len(selected)}/{count} balanced episodes.")
    return selected


def build_task_plan(task_id: int, *, selection_seed: int = SELECTION_SEED) -> dict[str, Any]:
    """Pre-register disjoint primary/fallback pools without outcome access."""

    if task_id not in TASK_IDS or selection_seed < 0:
        raise ValueError("Unsupported task or negative selection seed.")
    candidates = eligible_episodes()
    rng = np.random.default_rng(selection_seed + task_id * 100_003)
    pools = {offset: [episode for episode in candidates if episode % ROOT_CALL_OFFSET_CYCLE == offset] for offset in range(3, 11)}
    for values in pools.values():
        rng.shuffle(values)

    role_order = tuple(ROLE_TARGETS)
    primary: dict[str, list[int]] = {}
    for role_index, role in enumerate(role_order):
        primary[role] = sorted(
            _take_balanced_primary(
                pools,
                count=ROLE_TARGETS[role],
                offset_rotation=task_id + 3 * role_index,
            )
        )

    remaining = [episode for values in pools.values() for episode in values]
    rng.shuffle(remaining)
    audit_fallback_count = 50
    train_fallback_count = 13 if task_id % 2 == 0 else 12
    calibration_fallback_count = len(remaining) - audit_fallback_count - train_fallback_count
    fallback_counts = {
        "train": train_fallback_count,
        "calibration": calibration_fallback_count,
        "dev_audit": audit_fallback_count,
    }
    fallback: dict[str, list[int]] = {}
    cursor = 0
    for role in role_order:
        count = fallback_counts[role]
        fallback[role] = sorted(remaining[cursor : cursor + count])
        cursor += count
    if cursor != len(remaining):
        raise AssertionError("Fallback allocation did not consume every eligible episode.")

    roles = {
        role: {
            "target_roots": ROLE_TARGETS[role],
            "primary": primary[role],
            "primary_call_offsets": [episode % ROOT_CALL_OFFSET_CYCLE for episode in primary[role]],
            "fallback": fallback[role],
            "fallback_call_offsets": [episode % ROOT_CALL_OFFSET_CYCLE for episode in fallback[role]],
        }
        for role in role_order
    }
    return {"task_id": task_id, "roles": roles}


def build_selection(*, selection_seed: int = SELECTION_SEED) -> dict[str, Any]:
    validate_root_seed_namespace()
    tasks = {str(task): build_task_plan(task, selection_seed=selection_seed) for task in TASK_IDS}
    selection = {
        "status": "complete",
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "selection_seed": selection_seed,
        "selection_semantics": (
            "Outcome-blind task-stratified episode assignment frozen before collection. A missing root may be "
            "replaced only by the next unused fallback episode preassigned to the same role."
        ),
        "task_ids": list(TASK_IDS),
        "episode_start": EPISODE_START,
        "episode_end": EPISODE_END,
        "root_call_offset_cycle": ROOT_CALL_OFFSET_CYCLE,
        "minimum_call_offset": MINIMUM_CALL_OFFSET,
        "maximum_call_offset": MAXIMUM_CALL_OFFSET,
        **_round3_seed_metadata(),
        "targets_per_task": ROLE_TARGETS,
        "total_target_roots": len(TASK_IDS) * sum(ROLE_TARGETS.values()),
        "tasks": tasks,
    }
    validate_selection(selection)
    return selection


def validate_selection(selection: Mapping[str, Any]) -> None:
    if selection.get("protocol") != PROTOCOL_NAME or selection.get("status") != "complete":
        raise ValueError("Round3 selection has an unsupported protocol or incomplete status.")
    all_groups: set[tuple[int, int]] = set()
    for task in TASK_IDS:
        task_plan = selection["tasks"][str(task)]
        if int(task_plan["task_id"]) != task:
            raise ValueError("Selection task identity mismatch.")
        task_episodes: set[int] = set()
        for role, target in ROLE_TARGETS.items():
            values = task_plan["roles"][role]
            if int(values["target_roots"]) != target or len(values["primary"]) != target:
                raise ValueError(f"Task {task} role {role} has the wrong target/primary size.")
            episodes = [int(value) for value in (*values["primary"], *values["fallback"])]
            if len(episodes) != len(set(episodes)) or task_episodes.intersection(episodes):
                raise ValueError(f"Task {task} selection assigns an episode more than once.")
            if any(episode not in eligible_episodes() for episode in episodes):
                raise ValueError(f"Task {task} role {role} contains an ineligible episode.")
            task_episodes.update(episodes)
            all_groups.update((task, episode) for episode in episodes)
        if task_episodes != set(eligible_episodes()):
            raise ValueError(f"Task {task} selection does not preassign every eligible fallback episode.")
    expected = len(TASK_IDS) * len(eligible_episodes())
    if len(all_groups) != expected:
        raise ValueError("Selection contains duplicate task-qualified episode groups.")


def split_for_episode(selection: Mapping[str, Any], task_id: int, episode_id: int) -> str:
    roles = selection["tasks"][str(task_id)]["roles"]
    matches = [
        role
        for role in ROLE_TARGETS
        if episode_id in roles[role]["primary"] or episode_id in roles[role]["fallback"]
    ]
    if len(matches) != 1:
        raise KeyError(f"Episode {(task_id, episode_id)} has {len(matches)} split assignments.")
    return matches[0]


def next_fallback_batch(
    selection: Mapping[str, Any],
    *,
    task_id: int,
    role: str,
    attempted_episodes: set[int],
    deficit: int,
    maximum_batch: int,
) -> tuple[int, ...]:
    if role not in ROLE_TARGETS or deficit <= 0 or maximum_batch <= 0:
        raise ValueError("Invalid fallback role, deficit, or batch size.")
    fallback = selection["tasks"][str(task_id)]["roles"][role]["fallback"]
    available = [int(episode) for episode in fallback if int(episode) not in attempted_episodes]
    count = min(deficit, maximum_batch, len(available))
    if not count:
        raise RuntimeError(f"Exhausted same-role fallback episodes for task{task_id}/{role}.")
    return tuple(available[:count])


def _chunks(values: Sequence[int], size: int) -> list[tuple[int, ...]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [tuple(int(value) for value in values[start : start + size]) for start in range(0, len(values), size)]


def build_collector_command(
    *,
    python: pathlib.Path,
    collector_script: pathlib.Path,
    data_dir: pathlib.Path,
    initial_state_bank: pathlib.Path,
    aggregate_calibration_json: pathlib.Path,
    host: str,
    port: int,
    task_id: int,
    episodes: Sequence[int],
) -> list[str]:
    if not episodes or task_id not in TASK_IDS or port <= 0:
        raise ValueError("Collector command requires a valid task, port, and episode batch.")
    if any(episode not in eligible_episodes() for episode in episodes):
        raise ValueError("Collector command contains an episode outside the preregistered Round3 range.")
    return [
        str(python),
        str(collector_script),
        "--host",
        host,
        "--port",
        str(port),
        "--output-dir",
        str(data_dir),
        "--task-suite-name",
        "libero_10",
        "--task-start",
        str(task_id),
        "--max-tasks",
        "1",
        "--episode-ids",
        *[str(value) for value in episodes],
        "--initial-state-bank",
        str(initial_state_bank),
        "--num-steps-wait",
        "10",
        "--resize-size",
        "224",
        "--seed",
        str(COLLECTOR_SEED),
        "--root-seed-task-stride",
        str(ROOT_SEED_TASK_STRIDE),
        "--teacher-samples",
        "20",
        "--action-cot-denoising-steps",
        "10",
        "--root-stride-calls",
        "1",
        "--root-call-offset-cycle",
        str(ROOT_CALL_OFFSET_CYCLE),
        "--max-roots-per-episode",
        "1",
        "--records-per-shard",
        "10",
        "--source-iteration",
        str(SOURCE_ITERATION),
        "--candidate-horizons",
        *[str(value) for value in CANDIDATE_HORIZONS],
        "--reference-horizon",
        str(REFERENCE_HORIZON),
        "--model-action-horizon",
        "25",
        "--model-coarse-horizon",
        "15",
        "--model-action-dim",
        "32",
        "--model-state-dim",
        "32",
        "--prefix-feature-dim",
        "2048",
        "--prefix-token-count",
        "1024",
        "--source-policy",
        "current_student",
        "--continuation-policy",
        "fixed_h",
        "--fixed-continuation-horizon",
        str(FIXED_CONTINUATION_HORIZON),
        "--branch-repeats",
        str(BRANCH_REPEATS),
        "--repeat-branch-horizons",
        *[str(value) for value in CANDIDATE_HORIZONS],
        "--branch-repeat-seed-stride",
        str(BRANCH_REPEAT_SEED_STRIDE),
        "--student-mode",
        "hierarchical_transformer",
        "--hierarchical-aggregate-calibration-json",
        str(aggregate_calibration_json),
        "--debug-failure-videos",
        "0",
    ]


def _server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except OSError:
        return False


def _validate_code_snapshot(code_dir: pathlib.Path, code_commit: str, collector_script: pathlib.Path) -> None:
    if not code_dir.is_dir() or not collector_script.is_file():
        raise FileNotFoundError(f"Missing code snapshot or collector script under {code_dir}.")
    if not code_commit or any(character not in "0123456789abcdef" for character in code_commit.lower()):
        raise ValueError("code_commit must be a non-empty hexadecimal Git identity.")
    marker = code_dir / "code_commit"
    if marker.exists() and marker.read_text().strip() != code_commit:
        raise ValueError("Code snapshot marker differs from --code-commit.")
    if not marker.exists() and code_commit[:7].lower() not in code_dir.name.lower():
        raise ValueError("Code snapshot has neither a matching code_commit marker nor commit-bearing directory name.")
    controller_code_dir = pathlib.Path(__file__).resolve().parent.parent
    imported_collector = pathlib.Path(horizon_collector.__file__).resolve()
    if controller_code_dir != code_dir or imported_collector != collector_script:
        raise ValueError("Round3 controller and collector must execute from the same immutable --code-dir snapshot.")


def _validate_acquisition(
    predictor_dir: pathlib.Path,
    aggregate_path: pathlib.Path,
) -> dict[str, Any]:
    summary = json.loads((predictor_dir / "summary.json").read_text())
    config_path = predictor_dir / "predictor_config.json"
    params_path = predictor_dir / "params"
    pointwise_path = predictor_dir / "calibration.json"
    artifact = json.loads(aggregate_path.read_text())
    loaded_artifact = hierarchical.AggregateSelectorCalibration.load(aggregate_path)
    if int(summary.get("training_seed", -1)) != ACQUISITION_TRAINING_SEED:
        raise ValueError("Round3 acquisition requires the seed7 predictor.")
    config = json.loads(config_path.read_text())
    if (
        config.get("temporal_backbone") != "transformer"
        or not config.get("paired_distribution_heads")
        or tuple(config.get("candidate_horizons", ())) != CANDIDATE_HORIZONS
    ):
        raise ValueError("Acquisition predictor is not the expected paired H25 Transformer.")
    if (
        artifact.get("selected_rule") is None
        or artifact.get("source_split") != "calibration"
        or not loaded_artifact.aggregate_gate_passed
    ):
        raise ValueError("Aggregate acquisition artifact has no calibration-passing frozen selected rule.")
    provenance = artifact.get("provenance", {})
    live = {
        "predictor_config_digest": aggregate_common.json_file_digest(config_path),
        "params_digest": aggregate_common.params_tree_digest(params_path),
        "pointwise_calibration_digest": aggregate_common.json_file_digest(pointwise_path),
    }
    mismatches = {name: (provenance.get(name), value) for name, value in live.items() if provenance.get(name) != value}
    if mismatches:
        raise ValueError(f"Aggregate acquisition provenance differs from seed7 predictor: {mismatches}.")
    return {
        "predictor_dir": str(predictor_dir),
        "predictor_training_seed": ACQUISITION_TRAINING_SEED,
        "predictor_config_digest": live["predictor_config_digest"],
        "params_digest": live["params_digest"],
        "pointwise_calibration_digest": live["pointwise_calibration_digest"],
        "aggregate_calibration_json": str(aggregate_path),
        "aggregate_calibration_sha256": _file_digest(aggregate_path),
        "acquisition_only": True,
        "deployable_gate_claim": False,
    }


def validate_bank_lineage(
    parent_path: pathlib.Path,
    child_path: pathlib.Path,
) -> dict[str, Any]:
    """Require the Round3 bank to preserve old episodes 0-99 exactly."""

    parent = horizon_initial_states.InitialStateBank(parent_path)
    child = horizon_initial_states.InitialStateBank(child_path)
    lineage = horizon_initial_states.audit_bank_prefix(parent, child)
    parent_counts = {task: len(parent.states[task]) for task in TASK_IDS}
    child_counts = {task: len(child.states[task]) for task in TASK_IDS}
    if set(parent.states) != set(TASK_IDS) or any(count != EPISODE_START for count in parent_counts.values()):
        raise ValueError("Parent bank must exactly cover episodes 0-99 for all ten Round3 tasks.")
    if set(child.states) != set(TASK_IDS) or any(count <= EPISODE_END for count in child_counts.values()):
        raise ValueError("Child bank must cover every Round3 episode through 299 for all ten tasks.")
    return {
        **lineage,
        "round3_contract": {
            "parent_episode_range": [0, EPISODE_START - 1],
            "child_required_episode_range": [0, EPISODE_END],
            "parent_episode_counts_by_task": {str(task): count for task, count in parent_counts.items()},
            "child_episode_counts_by_task": {str(task): count for task, count in child_counts.items()},
        },
    }


def _expected_root_seed(task: int, episode: int, step: int) -> int:
    return horizon_collector._root_seed(  # noqa: SLF001
        COLLECTOR_SEED,
        task,
        episode,
        step,
        task_stride=ROOT_SEED_TASK_STRIDE,
    )


def _validate_completed_attempt(
    attempt_dir: pathlib.Path,
    *,
    expected_config: Mapping[str, Any],
) -> set[tuple[int, int, int, int]] | None:
    config_path = attempt_dir / "run_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    if config != expected_config:
        raise ValueError(f"Attempt config differs from the frozen protocol: {attempt_dir}")
    exit_path = attempt_dir / "collector.exit"
    summary_path = attempt_dir / "data" / "summary.json"
    if not exit_path.exists() or exit_path.read_text().strip() != "0" or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        return None
    metadata = summary.get("metadata", {})
    expected_metadata = {
        "source_policy": "current_student",
        "continuation_policy": "fixed_h",
        "fixed_continuation_horizon": FIXED_CONTINUATION_HORIZON,
        "student_mode": "hierarchical_transformer",
        "source_iteration": SOURCE_ITERATION,
        "root_call_offset_cycle": ROOT_CALL_OFFSET_CYCLE,
        **_round3_seed_metadata(),
        "branch_repeats": BRANCH_REPEATS,
        "branch_repeat_seed_stride": BRANCH_REPEAT_SEED_STRIDE,
        "candidate_horizons": list(CANDIDATE_HORIZONS),
        "reference_horizon": REFERENCE_HORIZON,
        "episode_ids": list(config["episodes"]),
        "initial_state_bank_sha256": config["initial_state_bank_sha256"],
    }
    mismatches = {name: (metadata.get(name), value) for name, value in expected_metadata.items() if metadata.get(name) != value}
    if mismatches:
        raise ValueError(f"Completed collector metadata mismatch in {attempt_dir}: {mismatches}.")
    expected_task = int(config["task_id"])
    expected_episodes = {int(value) for value in config["episodes"]}
    roots: set[tuple[int, int, int, int]] = set()
    for shard in sorted((attempt_dir / "data").glob("shard-*.h5")):
        with h5py.File(shard, "r") as handle:
            candidates = np.asarray(handle["candidate_horizons"])
            if not np.all(candidates == np.asarray(CANDIDATE_HORIZONS)[None, :]):
                raise ValueError(f"Candidate mismatch in {shard}.")
            if not np.all(np.asarray(handle["trial_count"]) == BRANCH_REPEATS):
                raise ValueError(f"Trial-count mismatch in {shard}.")
            if not np.all(np.asarray(handle["trial_valid"])):
                raise ValueError(f"Invalid paired trial in {shard}.")
            if not np.all(np.asarray(handle["source_iteration"]) == SOURCE_ITERATION):
                raise ValueError(f"Source-iteration mismatch in {shard}.")
            for task, episode, step, root_seed in zip(
                handle["task_id"][:],
                handle["episode_id"][:],
                handle["decision_step"][:],
                handle["root_seed"][:],
                strict=True,
            ):
                root = (int(task), int(episode), int(step), int(root_seed))
                if root[0] != expected_task or root[1] not in expected_episodes:
                    raise ValueError(f"Unexpected root {root} in {shard}.")
                if root[3] != _expected_root_seed(*root[:3]):
                    raise ValueError(f"Root seed does not match the Round3 namespace: {root}.")
                if root in roots:
                    raise ValueError(f"Duplicate exact root in attempt {attempt_dir}: {root}.")
                roots.add(root)
            for field in ("prefix_feature", "state", "coarse_actions", "final_actions", "trial_elapsed"):
                if not np.all(np.isfinite(np.asarray(handle[field]))):
                    raise ValueError(f"Non-finite {field} in {shard}.")
    if len(roots) != int(summary.get("num_records", -1)) or len({root[1] for root in roots}) != len(roots):
        raise ValueError(f"Completed attempt has inconsistent or multi-root episodes: {attempt_dir}.")
    return roots


def _attempt_config(
    *,
    protocol_identity: Mapping[str, Any],
    role: str,
    task_id: int,
    episodes: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "protocol_identity_sha256": _json_digest(protocol_identity),
        "code_commit": protocol_identity["code_commit"],
        "collector_script_sha256": protocol_identity["collector_script_sha256"],
        "initial_state_bank_sha256": protocol_identity["initial_state_bank_sha256"],
        "aggregate_calibration_sha256": protocol_identity["aggregate_calibration_sha256"],
        "role": role,
        "task_id": task_id,
        "episodes": [int(value) for value in episodes],
    }


def _scan_role_attempts(
    role_dir: pathlib.Path,
    *,
    protocol_identity: Mapping[str, Any],
    role: str,
    task_id: int,
) -> tuple[set[int], set[tuple[int, int, int, int]], int]:
    attempted: set[int] = set()
    roots: set[tuple[int, int, int, int]] = set()
    maximum_attempt = -1
    if not role_dir.exists():
        return attempted, roots, maximum_attempt
    for attempt_dir in sorted(role_dir.glob("attempt-*")):
        try:
            attempt_index = int(attempt_dir.name.removeprefix("attempt-"))
        except ValueError as error:
            raise ValueError(f"Unexpected attempt directory name: {attempt_dir}") from error
        maximum_attempt = max(maximum_attempt, attempt_index)
        config_path = attempt_dir / "run_config.json"
        if not config_path.exists():
            continue
        stored = json.loads(config_path.read_text())
        expected = _attempt_config(
            protocol_identity=protocol_identity,
            role=role,
            task_id=task_id,
            episodes=stored.get("episodes", ()),
        )
        completed = _validate_completed_attempt(attempt_dir, expected_config=expected)
        if completed is None:
            continue
        episodes = {int(value) for value in stored["episodes"]}
        if attempted.intersection(episodes) or roots.intersection(completed):
            raise ValueError(f"Completed attempts overlap under {role_dir}.")
        attempted.update(episodes)
        roots.update(completed)
    return attempted, roots, maximum_attempt


def _run_attempt(
    *,
    role_dir: pathlib.Path,
    attempt_index: int,
    protocol_identity: Mapping[str, Any],
    role: str,
    task_id: int,
    episodes: Sequence[int],
    python: pathlib.Path,
    collector_script: pathlib.Path,
    initial_state_bank: pathlib.Path,
    aggregate_calibration_json: pathlib.Path,
    host: str,
    port: int,
) -> set[tuple[int, int, int, int]]:
    attempt_dir = role_dir / f"attempt-{attempt_index:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    data_dir = attempt_dir / "data"
    config = _attempt_config(
        protocol_identity=protocol_identity,
        role=role,
        task_id=task_id,
        episodes=episodes,
    )
    _write_once_json(attempt_dir / "run_config.json", config)
    command = build_collector_command(
        python=python,
        collector_script=collector_script,
        data_dir=data_dir,
        initial_state_bank=initial_state_bank,
        aggregate_calibration_json=aggregate_calibration_json,
        host=host,
        port=port,
        task_id=task_id,
        episodes=episodes,
    )
    with (attempt_dir / "collector.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
    (attempt_dir / "collector.exit").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    roots = _validate_completed_attempt(attempt_dir, expected_config=config)
    if roots is None:
        raise RuntimeError(f"Collector exited zero without a complete auditable attempt: {attempt_dir}")
    return roots


def build_four_way_split_manifest(
    base_manifest: Mapping[str, Any],
    *,
    realized_groups: Mapping[str, Sequence[int]],
    initial_state_bank: pathlib.Path,
    initial_state_bank_sha256: str,
    bank_lineage_path: pathlib.Path,
    bank_lineage_sha256: str,
    selection_path: pathlib.Path,
    selection_sha256: str,
) -> dict[str, Any]:
    """Append Round3 groups while quarantining every previously inspected role."""

    base_groups = horizon_splits.validate_manifest(base_manifest)
    if "dev_audit" in base_groups:
        raise ValueError("Round3 requires a base manifest with no previously exposed dev_audit split.")
    early_name = "early_stop" if "early_stop" in base_groups else "validation"
    required = {"train", early_name, "calibration"}
    if not required.issubset(base_groups):
        raise ValueError(f"Base split manifest lacks required groups: {sorted(required.difference(base_groups))}.")
    new = {name: {int(value) for value in realized_groups[name]} for name in ROLE_TARGETS}
    if any(len(new[name]) != len(TASK_IDS) * ROLE_TARGETS[name] for name in ROLE_TARGETS):
        raise ValueError("Realized Round3 groups do not match the preregistered role targets.")
    if any(set(base_groups[name]).intersection(set().union(*new.values())) for name in required):
        raise ValueError("Round3 groups overlap the base development manifest.")
    return {
        "split_schema_version": horizon_splits.FOUR_WAY_SPLIT_SCHEMA_VERSION,
        "split_roles": dict(horizon_splits.FOUR_WAY_SPLIT_ROLES),
        "split_seed": SELECTION_SEED,
        "split_semantics": (
            "Old train and previously inspected calibration groups are training-only; old validation groups "
            "are early-stop-only. Round3 calibration and dev_audit use new episode groups, and dev_audit must "
            "not be used for training, checkpoint/seed selection, calibration, or threshold fitting."
        ),
        "train_group_ids": sorted(
            {int(value) for value in base_groups["train"]}
            .union(int(value) for value in base_groups["calibration"])
            .union(new["train"])
        ),
        "early_stop_group_ids": sorted(int(value) for value in base_groups[early_name]),
        "calibration_group_ids": sorted(new["calibration"]),
        "dev_audit_group_ids": sorted(new["dev_audit"]),
        "development_initial_state_bank": str(initial_state_bank),
        "development_initial_state_bank_sha256": initial_state_bank_sha256,
        "development_initial_state_bank_lineage_json": str(bank_lineage_path),
        "development_initial_state_bank_lineage_sha256": bank_lineage_sha256,
        "round3_selection_manifest": str(selection_path),
        "round3_selection_sha256": selection_sha256,
    }


def _protocol_identity(
    args: argparse.Namespace,
    collector_script: pathlib.Path,
    *,
    bank_lineage_path: pathlib.Path,
    bank_lineage_sha256: str,
) -> dict[str, Any]:
    code_dir = pathlib.Path(args.code_dir).resolve()
    bank = pathlib.Path(args.initial_state_bank).resolve()
    base_split = pathlib.Path(args.base_split_manifest).resolve()
    predictor = pathlib.Path(args.predictor_dir).resolve()
    aggregate_path = pathlib.Path(args.aggregate_calibration_json).resolve()
    bank_manifest = bank / "manifest.json"
    if not bank_manifest.is_file() or not base_split.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError("Round3 bank, base split, or aggregate calibration input is missing.")
    acquisition = _validate_acquisition(predictor, aggregate_path)
    return {
        "protocol": PROTOCOL_NAME,
        "code_dir": str(code_dir),
        "code_commit": args.code_commit,
        "collector_script": str(collector_script),
        "collector_script_sha256": _file_digest(collector_script),
        "parent_initial_state_bank": str(pathlib.Path(args.parent_initial_state_bank).resolve()),
        "initial_state_bank": str(bank),
        "initial_state_bank_sha256": _file_digest(bank_manifest),
        "bank_lineage_json": str(bank_lineage_path),
        "bank_lineage_sha256": bank_lineage_sha256,
        "base_split_manifest": str(base_split),
        "base_split_manifest_sha256": aggregate_common.json_file_digest(base_split),
        "host": args.host,
        "port": args.port,
        "selection_seed": SELECTION_SEED,
        "collector_seed": COLLECTOR_SEED,
        "source_policy": "current_student",
        "source_selector": "aggregate_seed7",
        "continuation_policy": "fixed_h5",
        "source_iteration": SOURCE_ITERATION,
        **_round3_seed_metadata(),
        "candidate_horizons": list(CANDIDATE_HORIZONS),
        "branch_repeats": BRANCH_REPEATS,
        **acquisition,
    }


def main(args: argparse.Namespace) -> None:
    if args.episodes_per_attempt <= 0 or args.port <= 0:
        raise ValueError("episodes_per_attempt and port must be positive.")
    output = pathlib.Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "controller.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another Round3 controller is active for {output}.") from error

        code_dir = pathlib.Path(args.code_dir).resolve()
        collector_script = (
            pathlib.Path(args.collector_script).resolve()
            if args.collector_script is not None
            else code_dir / "scripts" / "collect_execution_horizon_counterfactuals.py"
        )
        _validate_code_snapshot(code_dir, args.code_commit, collector_script)
        selection = build_selection()
        selection_path = output / "selection.json"
        _write_once_json(selection_path, selection)
        selection_sha256 = _file_digest(selection_path)
        bank_lineage_path = output / "bank_lineage.json"
        bank_lineage = validate_bank_lineage(
            pathlib.Path(args.parent_initial_state_bank).resolve(),
            pathlib.Path(args.initial_state_bank).resolve(),
        )
        _write_once_json(bank_lineage_path, bank_lineage)
        bank_lineage_sha256 = _file_digest(bank_lineage_path)
        protocol_identity = _protocol_identity(
            args,
            collector_script,
            bank_lineage_path=bank_lineage_path,
            bank_lineage_sha256=bank_lineage_sha256,
        )
        protocol_identity = {**protocol_identity, "selection_sha256": selection_sha256}
        _write_once_json(output / "run_config.json", protocol_identity)
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            expected_role_counts = {role: len(TASK_IDS) * count for role, count in ROLE_TARGETS.items()}
            if (
                summary.get("status") != "complete"
                or summary.get("protocol_identity_sha256") != _json_digest(protocol_identity)
                or summary.get("roots_by_role") != expected_role_counts
                or int(summary.get("num_roots", -1)) != sum(expected_role_counts.values())
                or not (output / "split_manifest.json").is_file()
                or not (output / "controller.exit").is_file()
                or (output / "controller.exit").read_text().strip() != "0"
            ):
                raise ValueError("Existing Round3 summary is incomplete or belongs to a different protocol.")
            horizon_splits.load_manifest(output / "split_manifest.json", require_four_way=True)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if not _server_ready(args.host, args.port):
            raise ConnectionError(f"Policy server is not reachable at {args.host}:{args.port}.")

        bank = pathlib.Path(args.initial_state_bank).resolve()
        aggregate_path = pathlib.Path(args.aggregate_calibration_json).resolve()
        python = pathlib.Path(args.python).resolve()
        if not python.is_file():
            raise FileNotFoundError(python)
        realized_groups: dict[str, list[int]] = {role: [] for role in ROLE_TARGETS}
        data_dirs: list[str] = []
        realized_by_task: dict[str, Any] = {}
        started = time.monotonic()
        try:
            for task_id in TASK_IDS:
                realized_by_task[str(task_id)] = {}
                for role, target in ROLE_TARGETS.items():
                    role_dir = output / f"task{task_id:02d}" / role
                    role_dir.mkdir(parents=True, exist_ok=True)
                    attempted, roots, maximum_attempt = _scan_role_attempts(
                        role_dir,
                        protocol_identity=protocol_identity,
                        role=role,
                        task_id=task_id,
                    )
                    if any(split_for_episode(selection, task_id, episode) != role for episode in attempted):
                        raise ValueError(f"Completed Task{task_id}/{role} attempt used another role's episode.")
                    plan = selection["tasks"][str(task_id)]["roles"][role]
                    for batch in _chunks(plan["primary"], args.episodes_per_attempt):
                        pending = tuple(episode for episode in batch if episode not in attempted)
                        if not pending:
                            continue
                        maximum_attempt += 1
                        new_roots = _run_attempt(
                            role_dir=role_dir,
                            attempt_index=maximum_attempt,
                            protocol_identity=protocol_identity,
                            role=role,
                            task_id=task_id,
                            episodes=pending,
                            python=python,
                            collector_script=collector_script,
                            initial_state_bank=bank,
                            aggregate_calibration_json=aggregate_path,
                            host=args.host,
                            port=args.port,
                        )
                        attempted.update(pending)
                        roots.update(new_roots)
                    while len(roots) < target:
                        fallback = next_fallback_batch(
                            selection,
                            task_id=task_id,
                            role=role,
                            attempted_episodes=attempted,
                            deficit=target - len(roots),
                            maximum_batch=args.episodes_per_attempt,
                        )
                        maximum_attempt += 1
                        new_roots = _run_attempt(
                            role_dir=role_dir,
                            attempt_index=maximum_attempt,
                            protocol_identity=protocol_identity,
                            role=role,
                            task_id=task_id,
                            episodes=fallback,
                            python=python,
                            collector_script=collector_script,
                            initial_state_bank=bank,
                            aggregate_calibration_json=aggregate_path,
                            host=args.host,
                            port=args.port,
                        )
                        attempted.update(fallback)
                        roots.update(new_roots)
                    if len(roots) != target:
                        raise ValueError(f"Task{task_id}/{role} produced {len(roots)} roots; expected {target}.")
                    episodes = sorted(root[1] for root in roots)
                    if any(split_for_episode(selection, task_id, episode) != role for episode in episodes):
                        raise ValueError(f"Task{task_id}/{role} used an episode from another split.")
                    realized_groups[role].extend(
                        task_id * horizon_splits.GROUP_ID_TASK_MULTIPLIER + episode for episode in episodes
                    )
                    completed_dirs = [
                        str(path / "data")
                        for path in sorted(role_dir.glob("attempt-*"))
                        if (path / "collector.exit").exists()
                        and (path / "collector.exit").read_text().strip() == "0"
                        and (path / "data" / "summary.json").exists()
                        and any((path / "data").glob("shard-*.h5"))
                    ]
                    data_dirs.extend(completed_dirs)
                    realized_by_task[str(task_id)][role] = {
                        "target_roots": target,
                        "realized_roots": len(roots),
                        "episodes": episodes,
                        "attempted_episodes": sorted(attempted),
                        "data_dirs": completed_dirs,
                    }
                    _write_json(
                        output / "status.json",
                        {
                            "status": "collecting",
                            "task_id": task_id,
                            "role": role,
                            "completed_roots": sum(len(values) for values in realized_groups.values()),
                            "target_roots": len(TASK_IDS) * sum(ROLE_TARGETS.values()),
                        },
                    )

            base_manifest = json.loads(pathlib.Path(args.base_split_manifest).resolve().read_text())
            bank_manifest_sha256 = _file_digest(bank / "manifest.json")
            four_way = build_four_way_split_manifest(
                base_manifest,
                realized_groups=realized_groups,
                initial_state_bank=bank,
                initial_state_bank_sha256=bank_manifest_sha256,
                bank_lineage_path=bank_lineage_path,
                bank_lineage_sha256=bank_lineage_sha256,
                selection_path=selection_path,
                selection_sha256=selection_sha256,
            )
            horizon_splits.validate_manifest(four_way, require_four_way=True)
            _write_once_json(output / "split_manifest.json", four_way)
            summary = {
                "status": "complete",
                "schema_version": 1,
                "protocol": PROTOCOL_NAME,
                "protocol_identity_sha256": _json_digest(protocol_identity),
                "selection_manifest": str(selection_path),
                "selection_sha256": selection_sha256,
                "bank_lineage_json": str(bank_lineage_path),
                "bank_lineage_sha256": bank_lineage_sha256,
                "split_manifest": str(output / "split_manifest.json"),
                "num_roots": sum(len(values) for values in realized_groups.values()),
                "roots_by_role": {role: len(values) for role, values in realized_groups.items()},
                "source_policy": "current_student",
                "source_selector": "aggregate_seed7",
                "branch_continuation_policy": "fixed_h5",
                "source_iteration": SOURCE_ITERATION,
                "data_dirs": sorted(set(data_dirs)),
                "realized_by_task": realized_by_task,
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_once_json(output / "summary.json", summary)
            _write_json(
                output / "status.json",
                {"status": "complete", "completed_roots": summary["num_roots"], "target_roots": 600},
            )
            if not (output / "controller.exit").exists():
                (output / "controller.exit").write_text("0\n")
            print(json.dumps(summary, indent=2, sort_keys=True))
        except Exception as error:
            failure_dir = output / "failures"
            failure_dir.mkdir(exist_ok=True)
            failure_path = failure_dir / f"failure-{time.time_ns()}.json"
            _write_once_json(failure_path, {"status": "failed", "error": repr(error)})
            _write_json(output / "status.json", {"status": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main(build_parser().parse_args())
