#!/usr/bin/env python3
"""Resume an existing Round-3 collection to breadth-first warm-start coverage.

This entrypoint deliberately reuses the frozen protocol identity and collector
from an existing Round-3 output.  It writes a separate warm-start snapshot and
leaves the original ``summary.json`` / ``split_manifest.json`` absent so the
original controller can later resume to its full target.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import pathlib
import sys
from typing import Any

MINIMUM_ROOTS_PER_TASK_ROLE = 10
SNAPSHOT_DIRECTORY_NAME = "breadth_first_min10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-output-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--episodes-per-attempt", type=int, default=10)
    return parser


def _load_frozen_controller(output: pathlib.Path) -> tuple[Any, dict[str, Any]]:
    protocol = json.loads((output / "run_config.json").read_text())
    code_dir = pathlib.Path(protocol["code_dir"])
    sys.path[:0] = [str(code_dir / "scripts"), str(code_dir / "src"), str(code_dir)]
    controller = importlib.import_module("run_execution_horizon_round3_collection")
    if pathlib.Path(controller.__file__).resolve() != (code_dir / "scripts" / f"{controller.__name__}.py").resolve():
        raise RuntimeError("Imported Round-3 controller does not belong to the frozen code snapshot.")
    return controller, protocol


def _assert_clean_boundary(output: pathlib.Path) -> None:
    unfinished = []
    for attempt in sorted(output.glob("task*/*/attempt-*")):
        if not (attempt / "run_config.json").exists():
            unfinished.append(str(attempt))
            continue
        exit_path = attempt / "collector.exit"
        if not exit_path.exists():
            unfinished.append(str(attempt))
    if unfinished:
        raise RuntimeError(
            "Breadth-first resume requires an attempt boundary; unfinished attempts: " + ", ".join(unfinished)
        )


def _completed_state(controller: Any, output: pathlib.Path, protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    selection = json.loads((output / "selection.json").read_text())
    controller.validate_selection(selection)
    for task in controller.TASK_IDS:
        result[task] = {}
        for role in controller.ROLE_TARGETS:
            role_dir = output / f"task{task:02d}" / role
            role_dir.mkdir(parents=True, exist_ok=True)
            attempted, roots, maximum_attempt = controller._scan_role_attempts(  # noqa: SLF001
                role_dir,
                protocol_identity=protocol,
                role=role,
                task_id=task,
            )
            result[task][role] = {
                "role_dir": role_dir,
                "attempted": attempted,
                "roots": roots,
                "maximum_attempt": maximum_attempt,
            }
    return result


def _collect_to_minimum(
    controller: Any,
    *,
    output: pathlib.Path,
    protocol: dict[str, Any],
    state: dict[int, dict[str, Any]],
    python: pathlib.Path,
    episodes_per_attempt: int,
) -> None:
    selection = json.loads((output / "selection.json").read_text())
    collector_script = pathlib.Path(protocol["collector_script"])
    bank = pathlib.Path(protocol["initial_state_bank"])
    aggregate = pathlib.Path(protocol["aggregate_calibration_json"])

    # Breadth first: every task gets train/calibration/dev_audit coverage before
    # any role is expanded beyond the common minimum.
    for task in controller.TASK_IDS:
        for role in controller.ROLE_TARGETS:
            item = state[task][role]
            attempted: set[int] = item["attempted"]
            roots: set[tuple[int, int, int, int]] = item["roots"]
            maximum_attempt = int(item["maximum_attempt"])
            if len(roots) >= MINIMUM_ROOTS_PER_TASK_ROLE:
                continue

            plan = selection["tasks"][str(task)]["roles"][role]
            primary = [int(value) for value in plan["primary"] if int(value) not in attempted]
            while len(roots) < MINIMUM_ROOTS_PER_TASK_ROLE and primary:
                count = min(
                    MINIMUM_ROOTS_PER_TASK_ROLE - len(roots),
                    episodes_per_attempt,
                    len(primary),
                )
                episodes = tuple(primary[:count])
                primary = primary[count:]
                maximum_attempt += 1
                new_roots = controller._run_attempt(  # noqa: SLF001
                    role_dir=item["role_dir"],
                    attempt_index=maximum_attempt,
                    protocol_identity=protocol,
                    role=role,
                    task_id=task,
                    episodes=episodes,
                    python=python,
                    collector_script=collector_script,
                    initial_state_bank=bank,
                    aggregate_calibration_json=aggregate,
                    host=protocol["host"],
                    port=int(protocol["port"]),
                )
                attempted.update(episodes)
                roots.update(new_roots)

            while len(roots) < MINIMUM_ROOTS_PER_TASK_ROLE:
                episodes = controller.next_fallback_batch(
                    selection,
                    task_id=task,
                    role=role,
                    attempted_episodes=attempted,
                    deficit=MINIMUM_ROOTS_PER_TASK_ROLE - len(roots),
                    maximum_batch=episodes_per_attempt,
                )
                maximum_attempt += 1
                new_roots = controller._run_attempt(  # noqa: SLF001
                    role_dir=item["role_dir"],
                    attempt_index=maximum_attempt,
                    protocol_identity=protocol,
                    role=role,
                    task_id=task,
                    episodes=episodes,
                    python=python,
                    collector_script=collector_script,
                    initial_state_bank=bank,
                    aggregate_calibration_json=aggregate,
                    host=protocol["host"],
                    port=int(protocol["port"]),
                )
                attempted.update(episodes)
                roots.update(new_roots)
            item["maximum_attempt"] = maximum_attempt


def _snapshot_payloads(
    controller: Any,
    *,
    output: pathlib.Path,
    protocol: dict[str, Any],
    state: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = output / "selection.json"
    lineage_path = output / "bank_lineage.json"
    base_manifest = json.loads(pathlib.Path(protocol["base_split_manifest"]).read_text())
    base_groups = controller.horizon_splits.validate_manifest(base_manifest)
    early_name = "early_stop" if "early_stop" in base_groups else "validation"

    by_role: dict[str, set[int]] = {role: set() for role in controller.ROLE_TARGETS}
    by_task: dict[str, Any] = {}
    data_dirs: list[str] = []
    for task in controller.TASK_IDS:
        by_task[str(task)] = {}
        for role in controller.ROLE_TARGETS:
            roots = state[task][role]["roots"]
            episodes = sorted(root[1] for root in roots)
            if len(roots) < MINIMUM_ROOTS_PER_TASK_ROLE or len(episodes) != len(set(episodes)):
                raise ValueError(f"Task{task}/{role} does not meet breadth-first coverage.")
            if any(controller.split_for_episode(json.loads(selection_path.read_text()), task, episode) != role for episode in episodes):
                raise ValueError(f"Task{task}/{role} contains an episode assigned to another role.")
            by_role[role].update(task * controller.horizon_splits.GROUP_ID_TASK_MULTIPLIER + episode for episode in episodes)
            completed_dirs = [
                str(attempt / "data")
                for attempt in sorted(state[task][role]["role_dir"].glob("attempt-*"))
                if (attempt / "collector.exit").exists()
                and (attempt / "collector.exit").read_text().strip() == "0"
                and (attempt / "data" / "summary.json").exists()
            ]
            data_dirs.extend(completed_dirs)
            by_task[str(task)][role] = {"roots": len(roots), "episodes": episodes, "data_dirs": completed_dirs}

    new_groups = set().union(*by_role.values())
    old_groups = {
        int(value)
        for name in ("train", early_name, "calibration")
        for value in base_groups[name]
    }
    if old_groups.intersection(new_groups):
        raise ValueError("Breadth-first groups overlap the base development split.")

    split_manifest = {
        "split_schema_version": controller.horizon_splits.FOUR_WAY_SPLIT_SCHEMA_VERSION,
        "split_roles": dict(controller.horizon_splits.FOUR_WAY_SPLIT_ROLES),
        "split_seed": controller.SELECTION_SEED,
        "split_semantics": (
            "Round-3 breadth-first warm-start snapshot. Old train and calibration groups are training-only; "
            "old validation is early-stop-only; new calibration and dev_audit remain disjoint."
        ),
        "train_group_ids": sorted(
            {int(value) for value in base_groups["train"]}
            | {int(value) for value in base_groups["calibration"]}
            | {int(value) for value in by_role["train"]}
        ),
        "early_stop_group_ids": sorted(int(value) for value in base_groups[early_name]),
        "calibration_group_ids": sorted(int(value) for value in by_role["calibration"]),
        "dev_audit_group_ids": sorted(int(value) for value in by_role["dev_audit"]),
        "development_initial_state_bank": protocol["initial_state_bank"],
        "development_initial_state_bank_sha256": protocol["initial_state_bank_sha256"],
        "development_initial_state_bank_lineage_json": str(lineage_path),
        "development_initial_state_bank_lineage_sha256": protocol["bank_lineage_sha256"],
        "round3_selection_manifest": str(selection_path),
        "round3_selection_sha256": protocol["selection_sha256"],
    }
    controller.horizon_splits.validate_manifest(split_manifest, require_four_way=True)
    summary = {
        "status": "complete",
        "schema_version": 1,
        "purpose": "breadth_first_warm_start",
        "source_protocol": protocol["protocol"],
        "source_protocol_identity_sha256": controller._json_digest(protocol),  # noqa: SLF001
        "minimum_roots_per_task_role": MINIMUM_ROOTS_PER_TASK_ROLE,
        "num_roots": sum(len(values) for values in by_role.values()),
        "roots_by_role": {role: len(values) for role, values in by_role.items()},
        "realized_by_task": by_task,
        "data_dirs": sorted(set(data_dirs)),
        "split_manifest": str(output / SNAPSHOT_DIRECTORY_NAME / "split_manifest.json"),
        "resumable_to_full_round3": True,
        "full_round3_outputs_written": False,
    }
    return split_manifest, summary


def main(args: argparse.Namespace) -> None:
    if args.episodes_per_attempt <= 0:
        raise ValueError("episodes_per_attempt must be positive.")
    output = pathlib.Path(args.existing_output_dir).resolve()
    controller, protocol = _load_frozen_controller(output)
    snapshot = output / SNAPSHOT_DIRECTORY_NAME
    if (snapshot / "summary.json").exists():
        print((snapshot / "summary.json").read_text(), end="")
        return

    with (output / "controller.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Original Round-3 controller is still active; switch only at an attempt boundary.") from error
        _assert_clean_boundary(output)
        if not controller._server_ready(protocol["host"], int(protocol["port"])):  # noqa: SLF001
            raise ConnectionError("The frozen acquisition policy server is not reachable.")
        python = controller._executable_path(args.python)  # noqa: SLF001
        state = _completed_state(controller, output, protocol)
        _collect_to_minimum(
            controller,
            output=output,
            protocol=protocol,
            state=state,
            python=python,
            episodes_per_attempt=args.episodes_per_attempt,
        )
        # Rescan so the frozen snapshot includes only complete immutable attempts.
        state = _completed_state(controller, output, protocol)
        split_manifest, summary = _snapshot_payloads(controller, output=output, protocol=protocol, state=state)
        snapshot.mkdir(parents=True, exist_ok=False)
        controller._write_once_json(snapshot / "split_manifest.json", split_manifest)  # noqa: SLF001
        controller._write_once_json(snapshot / "summary.json", summary)  # noqa: SLF001
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
