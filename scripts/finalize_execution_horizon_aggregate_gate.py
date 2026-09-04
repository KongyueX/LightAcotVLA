"""Run one frozen aggregate selector on fresh final data and seal the pilot gate.

This is deliberately a separate command from development calibration/audit.
It validates a completed passing development audit before it claims or opens
the explicitly named fresh-final summary.  Final labels are evaluated exactly
once with the already-selected rule; this command never fits or sweeps a rule.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
import re
from typing import Any

import execution_horizon_aggregate_risk_common as common
import numpy as np

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import hierarchical
from openpi.execution_horizon import initial_states
from openpi.execution_horizon import splits as horizon_splits

SCHEMA_VERSION = 1
_FINAL_COMPONENT = re.compile(r"(?:^|[._-])(final|holdout|test)(?:[._-]|$)", re.IGNORECASE)
_IDENTITY_FIELDS = ("task_id", "episode_id", "decision_step", "root_seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-audit-json", required=True)
    parser.add_argument("--fresh-final-summary", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-state-isolation-json", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-initialization-seed", type=int, default=None)
    parser.add_argument("--expected-final-roots", type=int, default=100)
    parser.add_argument("--expected-task-start", type=int, default=0)
    parser.add_argument("--expected-max-tasks", type=int, default=10)
    parser.add_argument("--expected-episode-start", type=int, default=90)
    parser.add_argument("--expected-episode-end", type=int, default=99)
    parser.add_argument("--expected-branch-repeats", type=int, default=3)
    return parser


def _json_snapshot(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"JSON artifact changed while being read: {path}.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload, hashlib.sha256(raw).hexdigest()


def _require_development_gate(audit: Mapping[str, Any]) -> None:
    split_name = audit.get("split_name")
    split_role = audit.get("split_role")
    split_contract_valid = (split_name, split_role) in {
        ("validation", "validation"),
        ("dev_audit", "development_audit"),
    }
    checks = {
        "status=complete": audit.get("status") == "complete",
        "offline_engineering_gate=true": audit.get("offline_engineering_gate") is True,
        "provenance_verified=true": audit.get("provenance_verified") is True,
        "aggregate_calibration_gate_passed=true": audit.get("aggregate_calibration_gate_passed") is True,
        "rule_frozen_before_audit=true": audit.get("rule_frozen_before_audit") is True,
        "audit_not_used_for_threshold_fit": audit.get("audit_split_used_for_threshold_fit") is False,
        "recognized_independent_audit_split": split_contract_valid,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Development aggregate audit is not eligible to unlock final data: {failed}.")


def _artifact_snapshot(path: pathlib.Path) -> tuple[hierarchical.AggregateSelectorCalibration, str]:
    before = common.file_digest(path)
    artifact = hierarchical.AggregateSelectorCalibration.load(path)
    after = common.file_digest(path)
    if before != after:
        raise RuntimeError("Aggregate calibration changed while being loaded.")
    if not artifact.aggregate_gate_passed or artifact.selected_rule is None:
        raise ValueError("Final audit requires one uniquely frozen aggregate rule that passed calibration.")
    return artifact, after


def _manifest_audit_groups(audit: Mapping[str, Any], split_manifest: pathlib.Path) -> tuple[int, ...]:
    split_name = str(audit["split_name"])
    split_role = str(audit["split_role"])
    _, values = common.load_split_manifest(
        split_manifest,
        split_name=split_name,
        required_role=split_role,
    )
    return tuple(int(value) for value in values)


def _state_groups(group_ids: Sequence[int]) -> set[tuple[int, int]]:
    return {
        (
            int(group // horizon_splits.GROUP_ID_TASK_MULTIPLIER),
            int(group % horizon_splits.GROUP_ID_TASK_MULTIPLIER),
        )
        for group in group_ids
    }


def _development_bank_from_manifest(
    manifest: Mapping[str, Any],
    development_groups: set[tuple[int, int]],
) -> initial_states.InitialStateBank | None:
    """Load the child/superset bank bound by a four-way development split.

    Legacy three-way manifests intentionally keep the historical single-bank
    behavior. A schema-v2 four-way manifest, however, must bind the bank that
    can resolve its new episode IDs before the one-shot final claim is made.
    """

    bank_value = manifest.get("development_initial_state_bank")
    bank_sha256 = manifest.get("development_initial_state_bank_sha256")
    if horizon_splits.is_four_way_manifest(manifest) and (not bank_value or not bank_sha256):
        raise ValueError("Four-way finalization requires a frozen development initial-state bank binding.")
    if bank_value is None and bank_sha256 is None:
        return None
    if not isinstance(bank_value, str) or not bank_value or not isinstance(bank_sha256, str):
        raise ValueError("Development initial-state bank path and SHA-256 must be provided together.")
    bank = initial_states.InitialStateBank(bank_value)
    if bank.sha256 != bank_sha256:
        raise ValueError("Development initial-state bank changed after the split manifest was frozen.")
    identities = {bank.identity(task, episode) for task, episode in development_groups}
    if len(identities) != len(development_groups):
        raise ValueError("Development split reuses at least one initial-state pose.")
    return bank


def _audit_initial_state_isolation(
    *,
    final_bank: initial_states.InitialStateBank,
    development_bank: initial_states.InitialStateBank | None,
    development_groups: set[tuple[int, int]],
    final_groups: set[tuple[int, int]],
) -> dict[str, Any]:
    """Audit legacy single-bank or schema-v2 parent/child isolation."""

    if development_bank is None:
        return final_bank.audit_partitions({"development": development_groups, "final": final_groups})
    lineage = initial_states.audit_bank_prefix(final_bank, development_bank)
    isolation = initial_states.audit_partitions_across_banks(
        {
            "development": (development_bank, development_groups),
            "final": (final_bank, final_groups),
        }
    )
    isolation["bank_lineage"] = lineage
    return isolation


def _validate_development_identity(
    audit: Mapping[str, Any],
    *,
    artifact: hierarchical.AggregateSelectorCalibration,
    aggregate_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, tuple[str, ...], tuple[int, ...]]:
    if pathlib.Path(str(audit.get("aggregate_calibration_json", ""))).resolve() != aggregate_path:
        raise ValueError("Development audit references a different aggregate calibration artifact.")
    predictor_dir = pathlib.Path(str(audit["predictor_dir"])).resolve()
    params_path = pathlib.Path(str(audit["params"])).resolve()
    pointwise_path = pathlib.Path(str(audit["pointwise_calibration_json"])).resolve()
    split_manifest = pathlib.Path(str(audit["split_manifest"])).resolve()
    development_inputs = tuple(str(pathlib.Path(value).resolve()) for value in audit["development_dataset_inputs"])
    if params_path != predictor_dir / "params":
        raise ValueError("Development audit did not use the predictor_dir/params sidecar deployed by the server.")
    if common.jsonable(audit.get("selected_rule")) != common.jsonable(artifact.selected_rule):
        raise ValueError("Development audit selected_rule differs from the frozen aggregate artifact.")
    if common.jsonable(audit.get("provenance")) != common.jsonable(dataclasses.asdict(artifact.provenance)):
        raise ValueError("Development audit provenance differs from the frozen aggregate artifact.")

    calibration_groups = tuple(sorted({int(value) for value in audit["calibration_group_ids"]}))
    audit_groups = tuple(sorted({int(value) for value in audit["selected_group_ids"]}))
    if calibration_groups != artifact.provenance.calibration_group_ids:
        raise ValueError("Development audit calibration groups differ from aggregate provenance.")
    if not audit_groups or set(calibration_groups).intersection(audit_groups):
        raise ValueError("Development calibration/audit groups are empty or overlapping.")
    if _manifest_audit_groups(audit, split_manifest) != audit_groups:
        raise ValueError("Development audit groups differ from the frozen split manifest.")

    live = common.provenance_values(
        predictor_dir=predictor_dir,
        params_path=params_path,
        pointwise_calibration_json=pointwise_path,
        split_manifest=split_manifest,
        development_dataset=development_inputs,
        calibration_group_ids=calibration_groups,
    )
    common.verify_provenance(dataclasses.asdict(artifact.provenance), live)
    pointwise = hierarchical.HierarchicalCalibration.load(pointwise_path)
    if common.jsonable(dataclasses.asdict(pointwise)) != common.jsonable(
        dataclasses.asdict(artifact.pointwise_calibration)
    ):
        raise ValueError("Pointwise calibration differs from the copy embedded in aggregate calibration.")
    return predictor_dir, params_path, pointwise_path, split_manifest, development_inputs, audit_groups


def _write_exclusive_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(common.jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_fresh_final_summary(
    summary: Mapping[str, Any],
    *,
    expected_roots: int,
    expected_task_start: int,
    expected_max_tasks: int,
    expected_episode_start: int,
    expected_episode_end: int,
    expected_branch_repeats: int,
) -> tuple[str, ...]:
    expected = {
        "status": "complete",
        "num_roots": expected_roots,
        "task_start": expected_task_start,
        "max_tasks": expected_max_tasks,
        "episode_start": expected_episode_start,
        "episode_end": expected_episode_end,
        "branch_repeats": expected_branch_repeats,
    }
    mismatches = {name: (summary.get(name), value) for name, value in expected.items() if summary.get(name) != value}
    if mismatches:
        raise ValueError(f"Fresh-final summary differs from the pre-registered contract: {mismatches}.")
    inputs = tuple(str(pathlib.Path(value).resolve()) for value in summary.get("data_dirs", ()))
    if not inputs or len(inputs) != len(set(inputs)):
        raise ValueError("Fresh-final summary must name non-empty unique data_dirs.")
    return inputs


def _require_explicit_final_paths(summary_path: pathlib.Path, inputs: Sequence[str]) -> None:
    if not any(_FINAL_COMPONENT.search(part) for part in summary_path.parts):
        raise ValueError("fresh-final summary path is not explicitly marked final/holdout/test.")
    for value in inputs:
        path = pathlib.Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        if not any(_FINAL_COMPONENT.search(part) for part in path.parts):
            raise ValueError(f"Final dataset path is not explicitly marked final/holdout/test: {path}.")


def _dataset_fingerprint(inputs: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    shards = horizon_dataset.discover_shards(inputs)
    if not shards:
        raise FileNotFoundError("Fresh-final inputs contain no HDF5 shards.")
    records = []
    seen: set[pathlib.Path] = set()
    for shard in shards:
        path = pathlib.Path(shard).resolve()
        if path in seen:
            raise ValueError(f"Fresh-final inputs contain duplicate shard {path}.")
        seen.add(path)
        records.append({"path": str(path), "size": path.stat().st_size, "sha256": common.file_digest(path)})
    canonical = json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest(), records


def _final_groups_and_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_task_start: int,
    expected_max_tasks: int,
    expected_episode_start: int,
    expected_episode_end: int,
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    count = len(np.asarray(arrays["task_id"]))
    indices = np.arange(count, dtype=np.int64)
    identities = np.stack([np.asarray(arrays[name], dtype=np.uint64) for name in _IDENTITY_FIELDS], axis=1)
    if np.unique(identities, axis=0).shape[0] != count:
        raise ValueError("Fresh-final data contains duplicate counterfactual root identities.")
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    observed = {(int(task_id), int(episode_id)) for task_id, episode_id in zip(task, episode, strict=True)}
    expected = {
        (task_id, episode_id)
        for task_id in range(expected_task_start, expected_task_start + expected_max_tasks)
        for episode_id in range(expected_episode_start, expected_episode_end + 1)
    }
    if count != len(expected) or observed != expected:
        raise ValueError(
            "Fresh-final roots do not exactly cover the pre-registered task/episode grid: "
            f"rows={count}, unique_groups={len(observed)}, expected={len(expected)}."
        )
    groups = common.episode_group_ids(arrays)
    return indices, groups, observed


def _row(predictions: Mapping[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[index] for name, value in predictions.items()}


def _official_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_roots": int(report["num_roots"]),
        "long_h_coverage": float(report["long_h_coverage"]),
        "selected_h_distribution": report["selected_h_distribution"],
        "success_advantage": report["success_advantage_vs_reference_cluster_bootstrap"],
        "elapsed_advantage": report["elapsed_advantage_vs_reference_cluster_bootstrap"],
        "calls_advantage": report["calls_advantage_vs_reference_cluster_bootstrap"],
        "false_long_rate": float(report["false_long_rate"]),
        "false_long_upper_95": float(report["false_long_upper_95"]),
        "gate_checks": report["gate_checks"],
        "offline_engineering_gate": bool(report["offline_engineering_gate"]),
    }


def main(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.expected_final_roots <= 0 or args.expected_max_tasks <= 0:
        raise ValueError("batch_size, expected_final_roots, and expected_max_tasks must be positive.")
    if args.expected_episode_start < 0 or args.expected_episode_end < args.expected_episode_start:
        raise ValueError("Invalid expected final episode range.")
    expected_grid = args.expected_max_tasks * (args.expected_episode_end - args.expected_episode_start + 1)
    if expected_grid != args.expected_final_roots:
        raise ValueError("expected_final_roots must equal the exact task-by-episode grid size.")

    development_audit_path = pathlib.Path(args.development_audit_json).resolve()
    # Keep the final path lexical until the passing development gate has been
    # verified and the one-shot claim has been created. Path.resolve() can
    # inspect symlink metadata even before file contents are opened.
    final_summary_input = pathlib.Path(args.fresh_final_summary).absolute()
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    claim_path = output_dir.with_name(output_dir.name + ".one_shot_claim.json")
    incomplete_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or claim_path.exists() or incomplete_dir.exists():
        raise FileExistsError("Final gate output/claim already exists; one-shot final data may not be reopened.")

    # This entire block is development-only.  In particular, final_summary_path
    # is not stat'ed or read until after the passing gate and immutable claim.
    development_audit, development_audit_sha256 = _json_snapshot(development_audit_path)
    _require_development_gate(development_audit)
    aggregate_path = pathlib.Path(development_audit["aggregate_calibration_json"]).resolve()
    artifact, aggregate_sha256 = _artifact_snapshot(aggregate_path)
    (
        predictor_dir,
        params_path,
        pointwise_path,
        split_manifest,
        development_inputs,
        audit_groups,
    ) = _validate_development_identity(
        development_audit,
        artifact=artifact,
        aggregate_path=aggregate_path,
    )
    manifest, split_manifest_raw_sha256 = _json_snapshot(split_manifest)
    split_manifest_digest = common.json_file_digest(split_manifest)
    if split_manifest_digest != artifact.provenance.split_manifest_digest:
        raise ValueError("Frozen split manifest digest differs from aggregate calibration provenance.")
    manifest_groups = horizon_splits.validate_manifest(
        manifest,
        require_four_way=horizon_splits.is_four_way_manifest(manifest),
    )
    declared_development_group_ids = {
        int(group) for values in manifest_groups.values() for group in np.asarray(values, dtype=np.uint64)
    }
    development_state_groups = _state_groups(tuple(declared_development_group_ids))
    development_bank = _development_bank_from_manifest(manifest, development_state_groups)
    if not (checkpoint_dir / "params").exists():
        raise FileNotFoundError(checkpoint_dir / "params")
    checkpoint_digest = common.params_tree_digest(checkpoint_dir)
    if not any(_FINAL_COMPONENT.search(part) for part in final_summary_input.parts):
        raise ValueError("fresh-final summary path is not explicitly marked final/holdout/test.")
    supplied_isolation: dict[str, Any] | None = None
    supplied_isolation_path: pathlib.Path | None = None
    supplied_isolation_sha256: str | None = None
    if args.initial_state_isolation_json is not None:
        supplied_isolation_path = pathlib.Path(args.initial_state_isolation_json).resolve()
        supplied_isolation, supplied_isolation_sha256 = _json_snapshot(supplied_isolation_path)
        if (
            supplied_isolation.get("status") != "complete"
            or supplied_isolation.get("pairwise_initial_state_overlap") != 0
        ):
            raise ValueError("Provided initial-state isolation audit is incomplete or overlapping.")

    claim = {
        "status": "claimed",
        "schema_version": SCHEMA_VERSION,
        "semantics": "This immutable claim precedes the first read of fresh-final metadata or labels.",
        "development_audit_json": str(development_audit_path),
        "development_audit_sha256": development_audit_sha256,
        "aggregate_calibration_json": str(aggregate_path),
        "aggregate_calibration_sha256": aggregate_sha256,
        "predictor_config_digest": artifact.provenance.predictor_config_digest,
        "params_digest": artifact.provenance.params_digest,
        "pointwise_calibration_digest": artifact.provenance.pointwise_calibration_digest,
        "split_manifest": str(split_manifest),
        "split_manifest_digest": split_manifest_digest,
        "split_manifest_raw_sha256": split_manifest_raw_sha256,
        "development_initial_state_bank": (
            development_bank.metadata() if development_bank is not None else None
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_digest": checkpoint_digest,
        "initial_state_isolation_json": (
            str(supplied_isolation_path) if supplied_isolation_path is not None else None
        ),
        "initial_state_isolation_sha256": supplied_isolation_sha256,
        "fresh_final_summary": str(final_summary_input),
        "final_contract": {
            "expected_final_roots": args.expected_final_roots,
            "expected_task_start": args.expected_task_start,
            "expected_max_tasks": args.expected_max_tasks,
            "expected_episode_start": args.expected_episode_start,
            "expected_episode_end": args.expected_episode_end,
            "expected_branch_repeats": args.expected_branch_repeats,
        },
    }
    _write_exclusive_json(claim_path, claim)

    # The one-shot claim now exists.  Only the frozen rule is applied below;
    # no calibration, checkpoint choice, threshold grid, or alternative rule is evaluated.
    final_summary_path = final_summary_input.resolve()
    final_summary, final_summary_sha256 = _json_snapshot(final_summary_path)
    final_inputs = _validate_fresh_final_summary(
        final_summary,
        expected_roots=args.expected_final_roots,
        expected_task_start=args.expected_task_start,
        expected_max_tasks=args.expected_max_tasks,
        expected_episode_start=args.expected_episode_start,
        expected_episode_end=args.expected_episode_end,
        expected_branch_repeats=args.expected_branch_repeats,
    )
    _require_explicit_final_paths(final_summary_path, final_inputs)
    fingerprint_before, shard_records = _dataset_fingerprint(final_inputs)
    arrays = horizon_dataset.load_counterfactual_arrays(final_inputs)
    fingerprint_after, _ = _dataset_fingerprint(final_inputs)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("Fresh-final shards changed while being audited.")
    indices, final_group_ids, final_state_groups = _final_groups_and_indices(
        arrays,
        expected_task_start=args.expected_task_start,
        expected_max_tasks=args.expected_max_tasks,
        expected_episode_start=args.expected_episode_start,
        expected_episode_end=args.expected_episode_end,
    )

    final_groups = {int(value) for value in final_group_ids}
    overlap = sorted(final_groups.intersection(declared_development_group_ids))
    if overlap:
        raise ValueError(f"Fresh-final groups overlap declared development groups: {overlap[:10]}.")
    if final_groups.intersection(artifact.provenance.calibration_group_ids):
        raise ValueError("Fresh-final groups overlap aggregate calibration groups.")
    if final_groups.intersection(audit_groups):
        raise ValueError("Fresh-final groups overlap development audit groups.")

    audit_split_name = str(development_audit["split_name"])
    audit_group_key = f"{audit_split_name}_group_ids"
    if audit_group_key not in manifest:
        raise KeyError(f"Frozen split manifest is missing {audit_group_key!r}.")
    manifest_audit_groups = _manifest_audit_groups(development_audit, split_manifest)
    if manifest_audit_groups != audit_groups:
        raise ValueError("Development audit groups differ from the frozen split manifest.")
    manifest_calibration_groups = tuple(sorted(int(value) for value in manifest["calibration_group_ids"]))
    if manifest_calibration_groups != artifact.provenance.calibration_group_ids:
        raise ValueError("Aggregate calibration groups differ from the frozen split manifest.")

    final_bank_value = final_summary.get("initial_state_bank")
    final_bank_sha256 = final_summary.get("initial_state_bank_sha256")
    if not final_bank_value or not final_bank_sha256:
        raise ValueError("Fresh-final summary lacks frozen initial-state bank provenance.")
    final_bank = initial_states.InitialStateBank(final_bank_value)
    if final_bank.sha256 != final_bank_sha256:
        raise ValueError("Fresh-final initial-state bank changed after collection.")
    verified_final_state_groups = initial_states.dataset_groups(list(final_inputs), final_bank)
    if verified_final_state_groups != final_state_groups:
        raise ValueError("Fresh-final HDF5 provenance does not match its exact task/episode grid.")
    isolation = _audit_initial_state_isolation(
        final_bank=final_bank,
        development_bank=development_bank,
        development_groups=development_state_groups,
        final_groups=final_state_groups,
    )
    if supplied_isolation is not None:
        if development_bank is None:
            if supplied_isolation.get("initial_state_bank_sha256") != final_bank.sha256:
                raise ValueError("Initial-state isolation bank changed after its audit.")
        else:
            supplied_banks = supplied_isolation.get("partition_banks")
            expected_banks = isolation["partition_banks"]
            if common.jsonable(supplied_banks) != common.jsonable(expected_banks):
                raise ValueError("Cross-bank initial-state isolation bindings changed after their audit.")
            if supplied_isolation.get("partition_group_counts") != isolation["partition_group_counts"]:
                raise ValueError("Cross-bank initial-state isolation group counts changed after their audit.")
        assert supplied_isolation_path is not None
        isolation["supplied_isolation_json"] = str(supplied_isolation_path)
        isolation["supplied_isolation_sha256"] = supplied_isolation_sha256

    inference_seed, predictor_summary_path = common.resolve_inference_initialization_seed(
        predictor_dir,
        args.inference_initialization_seed,
    )
    predictor_config, predictions, restored_params = common.predict_split(
        predictor_dir,
        arrays,
        indices,
        params=params_path,
        batch_size=args.batch_size,
        inference_initialization_seed=inference_seed,
    )
    if restored_params.resolve() != params_path or not getattr(predictor_config, "paired_distribution_heads", False):
        raise ValueError("Final audit did not restore the frozen paired-distribution predictor sidecar.")
    if tuple(predictor_config.candidate_horizons) != artifact.candidate_horizons:
        raise ValueError("Fresh-final predictor and aggregate artifact candidate horizons differ.")
    if int(predictor_config.reference_horizon) != artifact.reference_horizon:
        raise ValueError("Fresh-final predictor and aggregate artifact reference horizons differ.")

    decisions = [artifact.apply(_row(predictions, index)) for index in range(indices.size)]
    selected = np.asarray([decision.selected_horizon for decision in decisions], dtype=np.int64)
    labels = common.split_labels(arrays, indices)
    metrics = common.selection_metrics(
        labels,
        selected_horizons=selected,
        candidate_horizons=artifact.candidate_horizons,
        reference_horizon=artifact.reference_horizon,
        cluster_ids=final_group_ids,
        bootstrap_samples=artifact.search_config.bootstrap_samples,
        seed=artifact.search_config.bootstrap_seed,
        success_noninferiority_margin=artifact.search_config.success_noninferiority_margin,
        false_long_upper_bound=artifact.search_config.false_long_upper_bound,
    )
    final_official = {
        "status": "complete",
        "semantics": "One-shot evaluation of the calibration-frozen selector; final labels performed no selection.",
        "rule_frozen_before_final": True,
        "final_used_for_selection": False,
        "provenance_verified": True,
        "aggregate_calibration_json": str(aggregate_path),
        "aggregate_calibration_sha256": aggregate_sha256,
        "fresh_final_summary": str(final_summary_path),
        "fresh_final_summary_sha256": final_summary_sha256,
        "final_dataset_fingerprint": fingerprint_before,
        "final_dataset_inputs": list(final_inputs),
        "final_shards": shard_records,
        **metrics,
        "constraint_elimination": common.decision_diagnostics(decisions),
        "audit_statistics_config": {
            "confidence_level": artifact.search_config.confidence_level,
            "bootstrap_samples": artifact.search_config.bootstrap_samples,
            "bootstrap_seed": artifact.search_config.bootstrap_seed,
            "source": "frozen aggregate calibration artifact",
        },
    }
    development_official = _official_summary(development_audit)
    final_official_summary = _official_summary(final_official)
    dual_gate = bool(
        development_official["offline_engineering_gate"]
        and final_official_summary["offline_engineering_gate"]
    )
    summary = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "semantics": "Development passed before one-shot final access; the same frozen selector was used unchanged.",
        "dual_official_gate": dual_gate,
        "development_official": development_official,
        "final_official": final_official_summary,
        "development_audit_json": str(development_audit_path),
        "development_audit_sha256": development_audit_sha256,
        "aggregate_calibration_json": str(aggregate_path),
        "aggregate_calibration_sha256": aggregate_sha256,
        "predictor_dir": str(predictor_dir),
        "predictor_config_digest": artifact.provenance.predictor_config_digest,
        "params_digest": artifact.provenance.params_digest,
        "pointwise_calibration_json": str(pointwise_path),
        "pointwise_calibration_digest": artifact.provenance.pointwise_calibration_digest,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_digest": checkpoint_digest,
        "provenance_verified": True,
        "selected_rule": common.jsonable(artifact.selected_rule),
        "fresh_final_summary": str(final_summary_path),
        "fresh_final_summary_sha256": final_summary_sha256,
        "final_dataset_fingerprint": fingerprint_before,
        "one_shot_claim": str(claim_path),
        "initial_state_isolation": isolation,
        "deployment": {
            "pilot_allowed": dual_gate,
            "reason": "dual_official_gate_passed" if dual_gate else "fresh_final_official_gate_not_met",
        },
    }

    incomplete_dir.mkdir(parents=False)
    common.write_json(incomplete_dir / "final_official_audit.json", final_official)
    common.write_json(incomplete_dir / "summary.json", summary)
    incomplete_dir.replace(output_dir)
    print(json.dumps(common.jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
