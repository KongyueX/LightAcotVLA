"""Extend selected counterfactual roots from their saved simulator snapshots.

The ordinary counterfactual collector reaches a root by replaying the whole
source trajectory.  That is appropriate for fresh roots, but it is not safe
for extending an existing label: small simulator or policy replay drift can
change both the root action and the continuation outcome.  This collector
instead treats the closed HDF5 row as the source of truth:

* restore the exact saved MuJoCo physics state;
* reconstruct the original executable action chunk by unnormalizing the
  stored candidate-0 final action;
* preserve every existing paired trial byte-for-byte; and
* collect only the missing continuation seeds.

The resulting overlay therefore remains compatible with strict exact-root
consolidation while adding statistically independent continuation trials for
the action representation that is actually used to train the predictor.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import pathlib
import time
from typing import Any

import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
import h5py
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import v2
from openpi.shared import normalize

_IDENTITY_FIELDS = ("task_id", "episode_id", "decision_step", "root_seed")
_PHYSICS_RESTORE_ATOL = 1e-12
_LABEL_FIELDS = {
    "hazard_event_count",
    "hazard_at_risk_count",
    "branch_success",
    "branch_timeout",
    "remaining_steps",
    "remaining_calls",
    "branch_valid",
    "success_count",
    "timeout_count",
    "trial_count",
    "remaining_steps_mean",
    "remaining_steps_variance",
    "remaining_calls_mean",
    "remaining_calls_variance",
    "elapsed_mean",
    "elapsed_variance",
    "trial_success",
    "trial_timeout",
    "trial_remaining_steps",
    "trial_remaining_calls",
    "trial_elapsed",
    "trial_valid",
    "dangerous_long_count",
    "paired_trial_count",
}


@dataclasses.dataclass(frozen=True)
class SourceRow:
    shard: pathlib.Path
    row: int
    shape: horizon_dataset.DatasetShape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--norm-stats-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--prefix-token-count", type=int, default=0)
    parser.add_argument("--fixed-continuation-horizon", type=int, default=5)
    parser.add_argument("--branch-repeat-seed-stride", type=int, default=20_000_000)
    parser.add_argument("--physical-action-dim", type=int, default=7)
    parser.add_argument("--source-iteration", type=int, default=1)
    parser.add_argument("--records-per-shard", type=int, default=1)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--use-quantile-norm", action="store_true")
    parser.add_argument("--debug-video-stride", type=int, default=5)
    parser.add_argument("--v2-min-horizon", type=int, default=3)
    parser.add_argument("--v2-budget-capacity", type=float, default=12.0)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _shape(handle: h5py.File, shard: pathlib.Path) -> horizon_dataset.DatasetShape:
    version = int(handle.attrs["schema_version"])
    if version != horizon_dataset.SCHEMA_VERSION:
        raise ValueError(f"Snapshot relabeling requires schema v2, found {version} in {shard}.")
    return horizon_dataset.DatasetShape(**json.loads(handle.attrs["shape_json"]))


def _root_key(handle: h5py.File, row: int) -> tuple[int, int, int, int]:
    return tuple(int(np.asarray(handle[name][row]).item()) for name in _IDENTITY_FIELDS)


def _index_input(path: pathlib.Path) -> dict[tuple[int, int, int, int], SourceRow]:
    result: dict[tuple[int, int, int, int], SourceRow] = {}
    for shard in horizon_dataset.discover_shards((path,)):
        with h5py.File(shard, "r") as handle:
            shape = _shape(handle, shard)
            for row in range(int(handle["task_id"].shape[0])):
                key = _root_key(handle, row)
                if key in result:
                    raise ValueError(f"Duplicate exact root in source input {path}: {key}.")
                result[key] = SourceRow(shard=shard, row=row, shape=shape)
    return result


def _read_record(source: SourceRow) -> dict[str, Any]:
    record: dict[str, Any] = {}
    with h5py.File(source.shard, "r") as handle:
        for field in horizon_dataset._fixed_specs(source.shape):
            record[field] = np.asarray(handle[field][source.row])
        record["physics_state"] = np.asarray(handle["physics_state"][source.row], dtype=np.float64)
    return record


def _replacement_scalar(value: Any, *, replacement: int | bool) -> Any:
    if isinstance(value, np.ndarray):
        return np.full_like(value, replacement)
    if isinstance(value, np.generic):
        return value.dtype.type(replacement)
    try:
        return type(value)(replacement)
    except (TypeError, ValueError):
        return replacement


def _saved_snapshot(
    env: Any,
    physics_state: np.ndarray,
    decision_step: int,
) -> collector.SimulatorSnapshot:
    template = collector._capture_snapshot(env)
    scalar_attributes = []
    for owner, name, value in template.scalar_attributes:
        replacement = value
        if name in ("timestep", "_timestep"):
            replacement = _replacement_scalar(value, replacement=decision_step)
        elif name in ("done", "_done"):
            replacement = _replacement_scalar(value, replacement=False)
        scalar_attributes.append((owner, name, replacement))
    return collector.SimulatorSnapshot(
        physics_state=np.asarray(physics_state, dtype=np.float64).copy(),
        scalar_attributes=scalar_attributes,
        random_states=copy.deepcopy(template.random_states),
    )


def _prepare_root(
    task_suite: Any,
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[Any, str, collector.SimulatorSnapshot, int]:
    task_id = int(record["task_id"])
    episode_id = int(record["episode_id"])
    decision_step = int(record["decision_step"])
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    env.reset()
    observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
    for _ in range(args.num_steps_wait):
        observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
        if done:
            break
    snapshot = _saved_snapshot(env, record["physics_state"], decision_step)
    collector._restore_snapshot(env, snapshot)
    restored = np.asarray(collector._simulator(env).get_state().flatten(), dtype=np.float64)
    if not np.allclose(
        restored,
        snapshot.physics_state,
        rtol=0.0,
        atol=_PHYSICS_RESTORE_ATOL,
        equal_nan=True,
    ):
        maximum_error = float(np.max(np.abs(restored - snapshot.physics_state)))
        raise ValueError(
            "MuJoCo snapshot restore exceeded the declared tolerance; "
            f"max_abs={maximum_error:.9g}, atol={_PHYSICS_RESTORE_ATOL:.9g}."
        )
    environment_horizon = libero_eval._env_horizon(env)
    episode_step_limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
    if environment_horizon is not None:
        episode_step_limit = min(episode_step_limit, environment_horizon)
    return env, task_description, snapshot, episode_step_limit


def _action_stats(args: argparse.Namespace) -> Any:
    stats = normalize.load(pathlib.Path(args.norm_stats_dir))
    if "actions" not in stats:
        raise KeyError(f"Normalization stats under {args.norm_stats_dir} do not contain 'actions'.")
    return stats["actions"]


def _unnormalize_primary_actions(
    normalized: np.ndarray,
    *,
    stats: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    normalized = np.asarray(normalized, dtype=np.float32)[..., : args.physical_action_dim]
    if args.use_quantile_norm:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile action normalization requested but q01/q99 are unavailable.")
        low = np.asarray(stats.q01, dtype=np.float32)[..., : args.physical_action_dim]
        high = np.asarray(stats.q99, dtype=np.float32)[..., : args.physical_action_dim]
        raw = (normalized + 1.0) / 2.0 * (high - low + 1e-6) + low
        roundtrip = (raw - low) / (high - low + 1e-6) * 2.0 - 1.0
    else:
        mean = np.asarray(stats.mean, dtype=np.float32)[..., : args.physical_action_dim]
        std = np.asarray(stats.std, dtype=np.float32)[..., : args.physical_action_dim]
        raw = normalized * (std + 1e-6) + mean
        roundtrip = (raw - mean) / (std + 1e-6)
    maximum_roundtrip_error = float(np.max(np.abs(roundtrip - normalized)))
    if maximum_roundtrip_error > 2e-6:
        raise ValueError(f"Action normalization roundtrip failed; max_abs={maximum_roundtrip_error:.9g}.")
    return raw.astype(np.float32), maximum_roundtrip_error


def _base_branches(
    record: dict[str, Any],
    *,
    candidates: tuple[int, ...],
    root_seed: int,
    repeat_seed_stride: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    stored_candidates = tuple(int(value) for value in np.asarray(record["candidate_horizons"]))
    if stored_candidates != candidates:
        raise ValueError(f"Candidate mismatch: stored={stored_candidates}, requested={candidates}.")
    valid = np.asarray(record["trial_valid"], dtype=np.bool_)
    counts = np.sum(valid, axis=-1, dtype=np.int64)
    if not np.all(counts == counts[:1]) or int(counts[0]) <= 0:
        raise ValueError(f"Existing trials are not paired across candidates: {counts.tolist()}.")
    existing_trials = int(counts[0])
    if not np.all(valid[:, :existing_trials]) or np.any(valid[:, existing_trials:]):
        raise ValueError("Existing trial validity must be a contiguous paired prefix.")
    result: list[list[dict[str, Any]]] = []
    for candidate_index, _ in enumerate(candidates):
        outcomes = [
            {
                "repeat_index": repeat_index,
                "policy_seed": root_seed + repeat_index * repeat_seed_stride,
                "success": bool(record["trial_success"][candidate_index, repeat_index]),
                "timeout": bool(record["trial_timeout"][candidate_index, repeat_index]),
                "remaining_steps": int(record["trial_remaining_steps"][candidate_index, repeat_index]),
                "remaining_calls": int(record["trial_remaining_calls"][candidate_index, repeat_index]),
                "elapsed_seconds": float(record["trial_elapsed"][candidate_index, repeat_index]),
                "source": "preserved_base_trial",
            }
            for repeat_index in range(existing_trials)
        ]
        result.append(outcomes)
    return result, existing_trials


def _fake_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "mc_actions_normalized": np.asarray(record["final_actions"], dtype=np.float32)[None, ...],
        "mc_coarse_actions_normalized": np.asarray(record["coarse_actions"], dtype=np.float32)[None, ...],
        "execution_horizon_prefix_feature": np.asarray(record["prefix_feature"], dtype=np.float32),
        "execution_horizon_state_normalized": np.asarray(record["state"], dtype=np.float32),
        "execution_horizon_prefix_tokens": np.asarray(record.get("prefix_tokens", []), dtype=np.float32),
        "execution_horizon_prefix_mask": np.asarray(record.get("prefix_token_mask", []), dtype=np.bool_),
    }


def _rebuilt_record(
    base: dict[str, Any],
    *,
    branches: list[list[dict[str, Any]]],
    snapshot: collector.SimulatorSnapshot,
    target_shape: horizon_dataset.DatasetShape,
    reference_horizon: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    event_positions = np.flatnonzero(np.asarray(base["event_mask"], dtype=np.bool_))
    risk = {
        "event_index": int(event_positions[0]) if event_positions.size else target_shape.action_horizon,
        "final_risk": np.asarray(base["final_risk"]),
        "action_cot_risk": np.asarray(base["action_cot_risk"]),
        "fused_risk": np.asarray(base["fused_risk"]),
        "event_mask": np.asarray(base["event_mask"]),
    }
    rebuilt = collector._root_record(
        result=_fake_result(base),
        risk=risk,
        branches=branches,
        snapshot=snapshot,
        task_id=int(base["task_id"]),
        episode_id=int(base["episode_id"]),
        decision_step=int(base["decision_step"]),
        root_seed=int(base["root_seed"]),
        previous_actions_normalized=np.asarray(base["previous_actions"], dtype=np.float32),
        previous_h=int(base["previous_h"]),
        previous_valid=bool(base["previous_valid"]),
        budget_balance=float(base["budget_balance"]),
        episode_progress=float(base["episode_progress"]),
        source_iteration=args.source_iteration,
        v2_min_horizon=args.v2_min_horizon,
        shape=target_shape,
        reference_horizon=reference_horizon,
    )
    # Static predictor inputs and exact-root identity are copied from the base
    # row, never from a replayed observation or policy request.
    for field in horizon_dataset._fixed_specs(target_shape):
        if field in _LABEL_FIELDS or field == "source_iteration":
            continue
        rebuilt[field] = np.asarray(base[field]).copy()
    rebuilt["physics_state"] = np.asarray(base["physics_state"], dtype=np.float64).copy()
    rebuilt["source_iteration"] = args.source_iteration
    return rebuilt


def _completed_roots(output_dir: pathlib.Path) -> set[tuple[int, int, int, int]]:
    result: set[tuple[int, int, int, int]] = set()
    for shard in sorted(output_dir.glob("shard-*.h5")):
        with h5py.File(shard, "r") as handle:
            for row in range(int(handle["task_id"].shape[0])):
                key = _root_key(handle, row)
                if key in result:
                    raise ValueError(f"Duplicate exact root in existing snapshot relabel output: {key}.")
                result.add(key)
    return result


def _selection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_path = pathlib.Path(args.selection_manifest).resolve()
    payload = json.loads(selection_path.read_text())
    records = list(payload.get("records", []))
    if payload.get("status") != "complete" or len(records) != int(payload.get("num_selected_roots", -1)):
        raise ValueError("Selection manifest is incomplete or internally inconsistent.")
    if args.max_roots:
        records = records[: args.max_roots]
    if not records:
        raise ValueError("Selection produced no roots.")
    return payload, records


def main(args: argparse.Namespace) -> None:
    if (
        args.records_per_shard <= 0
        or args.branch_repeat_seed_stride <= 0
        or args.physical_action_dim <= 0
        or args.v2_budget_capacity <= 0
        or args.max_roots < 0
    ):
        raise ValueError("Shard size, seed stride, action dimension and budget must be positive; max_roots nonnegative.")
    selection, selected_records = _selection(args)
    candidates = tuple(int(value) for value in selection["candidate_horizons"])
    reference_horizon = int(selection["reference_horizon"])
    target_trials = int(selection["target_trials"])
    if reference_horizon not in candidates or target_trials <= 0:
        raise ValueError("Selection manifest has invalid candidates/reference/target trials.")
    if not 1 <= args.fixed_continuation_horizon <= max(candidates):
        raise ValueError("fixed_continuation_horizon must lie within the candidate action horizon.")

    source_cache: dict[pathlib.Path, dict[tuple[int, int, int, int], SourceRow]] = {}
    source_rows: list[SourceRow] = []
    base_records: list[dict[str, Any]] = []
    expected_shape: horizon_dataset.DatasetShape | None = None
    seen: set[tuple[int, int, int, int]] = set()
    for selected in selected_records:
        source_path = pathlib.Path(selected["source_input"]).resolve()
        if source_path not in source_cache:
            source_cache[source_path] = _index_input(source_path)
        key = tuple(int(selected[name]) for name in _IDENTITY_FIELDS)
        if key in seen:
            raise ValueError(f"Selection contains duplicate exact root {key}.")
        seen.add(key)
        source = source_cache[source_path].get(key)
        if source is None:
            raise KeyError(f"Selected root {key} is absent from declared source {source_path}.")
        if expected_shape is None:
            expected_shape = source.shape
        elif dataclasses.replace(expected_shape, max_trials=source.shape.max_trials) != source.shape:
            raise ValueError("Selected source datasets have incompatible shapes.")
        source_rows.append(source)
        base_records.append(_read_record(source))
    assert expected_shape is not None
    if expected_shape.candidate_horizons != candidates:
        raise ValueError("Selection candidates differ from the source dataset shape.")
    if target_trials <= expected_shape.max_trials:
        raise ValueError("Target trial count must exceed the source dataset max_trials.")
    target_shape = dataclasses.replace(expected_shape, max_trials=target_trials)

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    existing = _completed_roots(output_dir)
    unexpected = existing.difference(seen)
    if unexpected:
        raise ValueError(f"Existing output contains roots outside this selection: {sorted(unexpected)[:3]}.")

    args.continuation_policy = "fixed_h"
    stats = _action_stats(args)
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    metadata = {
        "operation": "saved-snapshot missing-trial extension",
        "selection_manifest": str(pathlib.Path(args.selection_manifest).resolve()),
        "norm_stats_dir": str(pathlib.Path(args.norm_stats_dir).resolve()),
        "base_trials_preserved": True,
        "root_policy_replayed": False,
        "primary_action_source": "unnormalized stored candidate-0 final_actions",
        "candidate_horizons": candidates,
        "reference_horizon": reference_horizon,
        "target_trials": target_trials,
        "branch_repeat_seed_stride": args.branch_repeat_seed_stride,
        "fixed_continuation_horizon": args.fixed_continuation_horizon,
        "source_iteration": args.source_iteration,
    }
    started = time.monotonic()
    maximum_roundtrip_error = 0.0
    repeated_path = output_dir / "repeated_branch_outcomes.jsonl"
    repeated_writer = repeated_path.open("a", encoding="utf-8")
    try:
        with horizon_dataset.ShardedCounterfactualWriter(
            output_dir,
            shape=target_shape,
            records_per_shard=args.records_per_shard,
            metadata=metadata,
        ) as writer:
            for selected, base, source in zip(selected_records, base_records, source_rows, strict=True):
                key = tuple(int(base[name]) for name in _IDENTITY_FIELDS)
                if key in existing:
                    continue
                branches, existing_trials = _base_branches(
                    base,
                    candidates=candidates,
                    root_seed=int(base["root_seed"]),
                    repeat_seed_stride=args.branch_repeat_seed_stride,
                )
                if existing_trials >= target_trials:
                    raise ValueError(f"Root {key} already has {existing_trials} trials; target is {target_trials}.")
                if int(selected["target_trials"]) != target_trials:
                    raise ValueError(f"Selected root {key} has a conflicting target trial count.")
                primary_actions, roundtrip_error = _unnormalize_primary_actions(
                    base["final_actions"],
                    stats=stats,
                    args=args,
                )
                maximum_roundtrip_error = max(maximum_roundtrip_error, roundtrip_error)
                env, task_description, snapshot, episode_step_limit = _prepare_root(
                    task_suite,
                    base,
                    args=args,
                )
                try:
                    root_budget_state = v2.EpisodeBudgetState(
                        balance=float(base["budget_balance"]) * args.v2_budget_capacity
                    )
                    for repeat_index in range(existing_trials, target_trials):
                        schedule_rng = np.random.default_rng(
                            int(base["root_seed"])
                            + repeat_index * args.branch_repeat_seed_stride
                            + 17
                        )
                        for candidate_index in schedule_rng.permutation(len(candidates)):
                            forced_horizon = candidates[int(candidate_index)]
                            branch_seed = int(base["root_seed"]) + repeat_index * args.branch_repeat_seed_stride
                            (
                                success,
                                timeout,
                                remaining_steps,
                                remaining_calls,
                                elapsed_seconds,
                                _,
                            ) = collector._run_branch(
                                env,
                                snapshot,
                                primary_actions,
                                forced_horizon=forced_horizon,
                                root_step=int(base["decision_step"]),
                                episode_step_limit=episode_step_limit,
                                root_seed=branch_seed,
                                task_description=task_description,
                                args=args,
                                client=client,
                                root_budget_state=root_budget_state,
                                capture_video=False,
                            )
                            branches[int(candidate_index)].append(
                                {
                                    "repeat_index": repeat_index,
                                    "policy_seed": branch_seed,
                                    "success": success,
                                    "timeout": timeout,
                                    "remaining_steps": remaining_steps,
                                    "remaining_calls": remaining_calls,
                                    "elapsed_seconds": elapsed_seconds,
                                    "source": "saved_snapshot_new_trial",
                                }
                            )
                finally:
                    libero_eval._safe_close_env(env)

                rebuilt = _rebuilt_record(
                    base,
                    branches=branches,
                    snapshot=snapshot,
                    target_shape=target_shape,
                    reference_horizon=reference_horizon,
                    args=args,
                )
                writer.append(rebuilt)
                writer.flush()
                repeated_writer.write(
                    json.dumps(
                        {
                            "task_id": key[0],
                            "episode_id": key[1],
                            "decision_step": key[2],
                            "root_seed": key[3],
                            "source_shard": str(source.shard),
                            "source_row": source.row,
                            "existing_trials_preserved": existing_trials,
                            "target_trials": target_trials,
                            "outcomes": {
                                str(horizon): outcomes
                                for horizon, outcomes in zip(candidates, branches, strict=True)
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                repeated_writer.flush()
                os.fsync(repeated_writer.fileno())
                existing.add(key)
                _write_json(
                    status_path,
                    {
                        "status": "collecting",
                        "completed_roots": len(existing),
                        "total_roots": len(selected_records),
                        "last_root": key,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
    finally:
        repeated_writer.close()

    if existing != seen:
        raise RuntimeError(f"Snapshot relabel output is incomplete: {len(existing)}/{len(seen)} roots.")
    summary = {
        "status": "complete",
        "selection_manifest": str(pathlib.Path(args.selection_manifest).resolve()),
        "num_roots": len(existing),
        "target_trials": target_trials,
        "preserved_trials_per_candidate": expected_shape.max_trials,
        "new_trials_per_candidate": target_trials - expected_shape.max_trials,
        "candidate_horizons": candidates,
        "reference_horizon": reference_horizon,
        "group_data_dirs": [str(output_dir)],
        "maximum_action_normalization_roundtrip_error": maximum_roundtrip_error,
        "root_policy_replayed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(status_path, {"status": "complete", "completed_roots": len(existing), "total_roots": len(existing)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
