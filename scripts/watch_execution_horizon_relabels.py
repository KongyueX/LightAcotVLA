"""Audit each closed snapshot-relabel shard once without touching the experiment.

The watcher is read-only with respect to source data and running controllers.
It writes a separate audit journal and can notify Feishu on task completion or
failure. A final success requires both every selected root and a zero collector
exit; a quiet or slow collector is never restarted by this script.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
import pathlib
import time
from typing import Any

import consolidate_execution_horizon_counterfactuals as consolidate
import numpy as np
import watch_experiment_feishu as feishu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--relabel-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--collector-pid", type=int)
    parser.add_argument("--collector-exit-file", type=pathlib.Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--notify", action="store_true")
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _notify(message: str, *, enabled: bool) -> bool:
    if not enabled:
        return False
    return feishu._try_send_feishu(
        message,
        webhook_url=os.environ["FEISHU_WEBHOOK_URL"],
        secret=os.environ["FEISHU_SIGNING_SECRET"],
    )


def _collector_identity(pid: int | None) -> str | None:
    if pid is None:
        return None
    try:
        process = pathlib.Path(f"/proc/{pid}")
        command = (process / "cmdline").read_bytes()
        if b"collect_execution_horizon_snapshot_relabels.py" not in command:
            return None
        # starttime is field 22; split after comm, which can contain spaces.
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        return f"{pid}:{fields[19]}"
    except FileNotFoundError:
        return None


def _verify_statistics(record: dict[str, Any], *, reference_horizon: int, target_trials: int) -> None:
    valid = np.asarray(record["trial_valid"], dtype=np.bool_)
    success = np.asarray(record["trial_success"], dtype=np.bool_)
    timeout = np.asarray(record["trial_timeout"], dtype=np.bool_)
    if valid.shape[-1] != target_trials or not np.all(valid):
        raise ValueError("Closed overlay does not contain the required number of valid trials.")
    if np.any(success & timeout):
        raise ValueError("A trial is labelled as both success and timeout.")
    for name, expected in (
        ("trial_count", valid.sum(axis=-1)),
        ("success_count", (success & valid).sum(axis=-1)),
        ("timeout_count", (timeout & valid).sum(axis=-1)),
    ):
        np.testing.assert_array_equal(record[name], expected, err_msg=name)
    for raw_name, stem in (
        ("trial_remaining_steps", "remaining_steps"),
        ("trial_remaining_calls", "remaining_calls"),
        ("trial_elapsed", "elapsed"),
    ):
        values = np.asarray(record[raw_name], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Non-finite or negative valid outcomes in {raw_name}.")
        variance = values.var(axis=-1, ddof=1) if target_trials > 1 else np.zeros(values.shape[0])
        for suffix, expected in (("mean", values.mean(axis=-1)), ("variance", variance)):
            field = f"{stem}_{suffix}"
            np.testing.assert_allclose(record[field], expected, rtol=1e-5, atol=1e-5, err_msg=field)
    candidates = tuple(int(value) for value in record["candidate_horizons"])
    reference = candidates.index(reference_horizon)
    paired = np.zeros(len(candidates), dtype=np.int64)
    dangerous = np.zeros_like(paired)
    for index, horizon in enumerate(candidates):
        if horizon > reference_horizon:
            paired[index] = target_trials
            dangerous[index] = np.sum(success[reference] & ~success[index])
    np.testing.assert_array_equal(record["paired_trial_count"], paired, err_msg="paired_trial_count")
    np.testing.assert_array_equal(record["dangerous_long_count"], dangerous, err_msg="dangerous_long_count")


def _audit_row(
    base: consolidate.RowRef,
    overlay: consolidate.RowRef,
    *,
    reference_horizon: int,
    target_trials: int,
) -> dict[str, Any]:
    consolidate._compatible_shape(base.shape, overlay.shape)
    consolidate._verify_overlay(base, overlay)
    before = consolidate._read_record(base, base.shape)
    after = consolidate._read_record(overlay, overlay.shape)
    # Snapshot relabeling promises exact static inputs, stronger than ordinary
    # consolidation tolerance. Include elapsed in the preserved paired trials.
    for field in (*consolidate._STATIC_FLOAT_FIELDS, *consolidate._STATIC_EXACT_FIELDS, "physics_state"):
        if field in before:
            np.testing.assert_array_equal(after[field], before[field], err_msg=f"preserved {field}")
    for field in consolidate._TRIAL_FIELDS:
        np.testing.assert_array_equal(
            after[field][..., : base.shape.max_trials], before[field], err_msg=f"preserved {field}"
        )
    _verify_statistics(after, reference_horizon=reference_horizon, target_trials=target_trials)
    return {
        "root": list(consolidate._root_identity(overlay)),
        "shard": overlay.shard.name,
        "row": overlay.row,
        "success_count": np.asarray(after["success_count"]).tolist(),
        "timeout_count": np.asarray(after["timeout_count"]).tolist(),
    }


def _run(args: argparse.Namespace, state: dict[str, Any]) -> None:
    selection_bytes = args.selection_manifest.read_bytes()
    selection = json.loads(selection_bytes)
    selected = selection["records"]
    keys = [tuple(int(row[field]) for field in consolidate._IDENTITY_FIELDS) for row in selected]
    if selection.get("status") != "complete" or len(keys) != selection["num_selected_roots"]:
        raise ValueError("Selection manifest is incomplete.")
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Selection is empty or contains duplicate roots.")
    selected_by_key = dict(zip(keys, selected, strict=True))
    sources = sorted({pathlib.Path(row["source_input"]).resolve() for row in selected})
    base_rows, _ = consolidate._index_inputs(sources)
    candidates = tuple(int(value) for value in selection["candidate_horizons"])
    if any(base_rows[key].shape.candidate_horizons != candidates for key in keys):
        raise ValueError("Selection candidates differ from the source datasets.")
    expected_tasks = Counter(key[0] for key in keys)
    identity = _collector_identity(args.collector_pid)
    previous_status = args.output_dir / "status.json"
    previous = json.loads(previous_status.read_text()) if previous_status.exists() else {}
    selection_digest = hashlib.sha256(selection_bytes).hexdigest()
    if previous.get("selection_sha256", selection_digest) != selection_digest:
        raise ValueError("Audit output belongs to a different selection manifest.")
    state.update(
        selection_sha256=selection_digest,
        total_roots=len(keys),
        target_trials=int(selection["target_trials"]),
        collector_identity=identity,
        audited_roots=[],
        audited_shards={},
    )
    audited: set[tuple[int, ...]] = set()
    notified_tasks = set(previous.get("notified_task_ids", []))
    missing_since: float | None = None
    while True:
        shards = sorted(args.relabel_dir.glob("shard-*.h5"))
        names = {shard.name for shard in shards}
        if set(state["audited_shards"]).difference(names):
            raise ValueError("An already audited shard disappeared.")
        for shard in shards:
            stat = shard.stat()
            fingerprint = [stat.st_size, stat.st_mtime_ns]
            previous = state["audited_shards"].get(shard.name)
            if previous is not None:
                if previous["stat"] != fingerprint:
                    raise ValueError(f"Already audited shard changed: {shard.name}.")
                continue
            rows, _ = consolidate._index_inputs([shard])
            for key, overlay in rows.items():
                if key not in selected_by_key or key in audited:
                    raise ValueError(f"Unexpected or duplicate overlay root: {key}.")
                base = base_rows[key]
                if base.source_input != pathlib.Path(selected_by_key[key]["source_input"]).resolve():
                    raise ValueError(f"Source input mismatch for root {key}.")
                try:
                    result = _audit_row(
                        base,
                        overlay,
                        reference_horizon=int(selection["reference_horizon"]),
                        target_trials=int(selection["target_trials"]),
                    )
                except Exception as exc:
                    raise ValueError(f"Root {key} failed strict audit: {exc}") from exc
                audited.add(key)
                state["audited_roots"].append(result)
            with shard.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if [shard.stat().st_size, shard.stat().st_mtime_ns] != fingerprint:
                raise ValueError(f"Closed shard changed during audit: {shard.name}.")
            state["audited_shards"][shard.name] = {"stat": fingerprint, "sha256": digest}
            print(f"AUDIT_OK {len(audited)}/{len(keys)} {shard.name}", flush=True)
        observed_tasks = Counter(key[0] for key in audited)
        complete_tasks = [task for task, count in expected_tasks.items() if observed_tasks[task] == count]
        state.update(
            status="snapshot_checked" if args.once else "watching",
            checked_roots=len(audited),
            checked_at_unix=time.time(),
            completed_task_ids=sorted(complete_tasks),
            notified_task_ids=sorted(notified_tasks),
        )
        for task in complete_tasks:
            if task not in notified_tasks and _notify(
                f"H25 dense审计进度: Task{task}的{expected_tasks[task]}个roots全部通过严格审计, "
                f"累计{len(audited)}/{len(keys)}. 采集与训练协议未改变, 后续接力继续等待完整数据。",
                enabled=args.notify and not args.once,
            ):
                notified_tasks.add(task)
                state["notified_task_ids"] = sorted(notified_tasks)
                _write_json(args.output_dir / "status.json", state)
                print(f"FEISHU_OK task{task}", flush=True)
        if args.once:
            _write_json(args.output_dir / "status.json", state)
            return
        if args.collector_exit_file.exists():
            code = int(args.collector_exit_file.read_text().strip())
            if code != 0:
                raise RuntimeError(f"Collector exited with code {code}.")
            if audited != set(keys):
                raise ValueError(f"Collector exited zero with only {len(audited)}/{len(keys)} audited roots.")
            summary = json.loads((args.relabel_dir / "summary.json").read_text())
            if (
                summary.get("status") != "complete"
                or summary.get("num_roots") != len(keys)
                or summary.get("target_trials") != selection["target_trials"]
            ):
                raise ValueError("Collector summary does not satisfy the completion contract.")
            state.update(status="complete", collector_exit_code=0)
            _write_json(args.output_dir / "summary.json", state)
            _write_json(args.output_dir / "status.json", state)
            if _notify(
                f"H25 dense全部{len(keys)}个roots严格审计通过, collector exit0, fresh接力可继续。", enabled=args.notify
            ):
                print("FEISHU_OK complete", flush=True)
            return
        current_identity = _collector_identity(args.collector_pid)
        alive = identity is not None and current_identity == identity
        state["collector_alive"] = alive
        if alive:
            missing_since = None
        else:
            missing_since = time.monotonic() if missing_since is None else missing_since
            if time.monotonic() - missing_since >= 2 * args.poll_seconds:
                raise RuntimeError("Original collector handle is gone and no exit file appeared after two polls.")
        _write_json(args.output_dir / "status.json", state)
        time.sleep(args.poll_seconds)


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive.")
    if not args.once and (args.collector_pid is None or args.collector_exit_file is None):
        raise ValueError("Monitoring requires collector-pid and collector-exit-file.")
    if args.notify and not all(os.environ.get(name) for name in ("FEISHU_WEBHOOK_URL", "FEISHU_SIGNING_SECRET")):
        raise ValueError("Feishu notification was requested but credentials are not configured.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"status": "starting", "watcher_pid": os.getpid()}
    with (args.output_dir / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _run(args, state)
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", checked_at_unix=time.time())
            _write_json(args.output_dir / "status.json", state)
            if _notify(f"H25 snapshot增量审计异常: {type(exc).__name__}: {exc}", enabled=args.notify):
                print("FEISHU_OK failure", flush=True)
            raise


if __name__ == "__main__":
    main(build_parser().parse_args())
