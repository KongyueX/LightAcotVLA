"""Shared, development-only helpers for aggregate execution-horizon calibration.

This module deliberately lives under ``scripts``.  It composes the public
aggregate-calibration API with the existing predictor restore/inference helper
without changing the deployed selector or its legacy audit scripts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
import re
from typing import Any

import numpy as np

from openpi.execution_horizon import dataset as horizon_dataset

_IDENTITY_FIELDS = ("task_id", "episode_id", "decision_step", "root_seed")
_NON_DEVELOPMENT_COMPONENT = re.compile(r"(?:^|[._-])(final|test|holdout)(?:[._-]|$)", re.IGNORECASE)
_FINAL_OR_HOLDOUT_COMPONENT = re.compile(r"(?:^|[._-])(final|holdout)(?:[._-]|$)", re.IGNORECASE)
_TEST_COMPONENT = re.compile(r"(?:^|[._-])test(?:[._-]|$)", re.IGNORECASE)
_PYTEST_RUN_COMPONENT = re.compile(r"^pytest-(?:\d+|current)$")
_PYTEST_CASE_COMPONENT = re.compile(r"^test_.+\d+$")
_CONSTRAINT_FIELDS = (
    "success_pass",
    "elapsed_pass",
    "danger_pass",
    "faster_pass",
    "hazard_pass",
    "ood_pass",
)


def jsonable(value: Any) -> Any:
    """Convert dataclass/numpy-rich diagnostics to JSON-compatible values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="surrogateescape")
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_file_digest(path: pathlib.Path | str) -> str:
    """Hash parsed JSON canonically so formatting-only edits do not change identity."""

    source = pathlib.Path(path).resolve()
    return _sha256_bytes(_canonical_json_bytes(json.loads(source.read_text())))


def file_digest(path: pathlib.Path | str) -> str:
    """Stream a regular file into SHA-256 and reject concurrent mutation."""

    unresolved = pathlib.Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"Expected one non-symlink regular file for digesting: {unresolved}.")
    source = unresolved.resolve()
    if not source.is_file():
        raise ValueError(f"Expected one non-symlink regular file for digesting: {source}.")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while computing provenance digest: {source}.")
    return digest.hexdigest()


def params_tree_digest(path: pathlib.Path | str) -> str:
    """Hash a checkpoint file/tree by sorted relative names and exact bytes."""

    unresolved = pathlib.Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"Checkpoint provenance refuses a symlink root: {unresolved}.")
    root = unresolved.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        entries = [{"kind": "file", "path": ".", "size": root.stat().st_size, "sha256": file_digest(root)}]
    else:
        entries = []
        for entry in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ValueError(f"Checkpoint provenance refuses symlink entry {relative!r}.")
            if entry.is_dir():
                entries.append({"kind": "directory", "path": relative})
            elif entry.is_file():
                entries.append(
                    {
                        "kind": "file",
                        "path": relative,
                        "size": entry.stat().st_size,
                        "sha256": file_digest(entry),
                    }
                )
            else:
                raise ValueError(f"Unsupported checkpoint entry type at {entry}.")
        if not entries:
            raise ValueError(f"Checkpoint tree is empty: {root}.")
    return _sha256_bytes(_canonical_json_bytes({"schema": "params-tree-v1", "entries": entries}))


def resolve_inference_initialization_seed(
    predictor_dir: pathlib.Path | str,
    requested_seed: int | None,
) -> tuple[int, pathlib.Path]:
    """Bind graph initialization to the predictor's recorded training seed."""

    summary_path = pathlib.Path(predictor_dir).resolve() / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise ValueError(f"Predictor summary is not complete: {summary_path}.")
    if "training_seed" not in summary:
        raise KeyError(f"Predictor summary has no training_seed: {summary_path}.")
    training_seed = int(summary["training_seed"])
    seed = training_seed if requested_seed is None else int(requested_seed)
    if seed != training_seed:
        raise ValueError(f"inference_initialization_seed={seed} differs from predictor training_seed={training_seed}.")
    return seed, summary_path


def development_dataset_fingerprint(paths: Sequence[pathlib.Path | str]) -> str:
    """Hash every development shard byte plus its canonical manifest identity."""

    inputs = validate_development_dataset_paths(paths)
    sources: list[dict[str, Any]] = []
    seen_shards: set[pathlib.Path] = set()
    for input_index, raw_path in enumerate(inputs):
        source = pathlib.Path(raw_path)
        if source.is_dir():
            manifest_path = source / "manifest.json"
            manifest_digest = json_file_digest(manifest_path) if manifest_path.exists() else None
            shards = sorted(source.glob("shard-*.h5"))
            kind = "directory"
        else:
            manifest_digest = None
            shards = [source]
            kind = "file"
        if not shards:
            raise FileNotFoundError(f"No HDF5 shards found for development input {source}.")
        shard_records = []
        for shard_index, shard in enumerate(shards):
            resolved_shard = shard.resolve()
            if resolved_shard in seen_shards:
                raise ValueError(f"Development inputs contain duplicate shard {resolved_shard}.")
            seen_shards.add(resolved_shard)
            shard_records.append(
                {
                    "index": shard_index,
                    "logical_name": shard.name,
                    "size": resolved_shard.stat().st_size,
                    "sha256": file_digest(resolved_shard),
                }
            )
        sources.append(
            {
                "index": input_index,
                "kind": kind,
                "logical_name": source.name,
                "manifest_digest": manifest_digest,
                "shards": shard_records,
            }
        )
    return _sha256_bytes(_canonical_json_bytes({"schema": "development-dataset-v1", "sources": sources}))


def provenance_values(
    *,
    predictor_dir: pathlib.Path | str,
    params_path: pathlib.Path | str,
    pointwise_calibration_json: pathlib.Path | str,
    split_manifest: pathlib.Path | str,
    development_dataset: Sequence[pathlib.Path | str],
    calibration_group_ids: Sequence[int],
) -> dict[str, Any]:
    """Compute every field required by AggregateSelectorProvenance."""

    predictor_config_path = pathlib.Path(predictor_dir).resolve() / "predictor_config.json"
    groups = tuple(sorted({int(value) for value in calibration_group_ids}))
    if not groups:
        raise ValueError("calibration_group_ids must not be empty.")
    return {
        "predictor_config_digest": json_file_digest(predictor_config_path),
        "params_digest": params_tree_digest(params_path),
        "pointwise_calibration_digest": json_file_digest(pointwise_calibration_json),
        "split_manifest_digest": json_file_digest(split_manifest),
        "development_dataset_fingerprint": development_dataset_fingerprint(development_dataset),
        "calibration_group_ids": groups,
    }


def verify_provenance(frozen: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    """Require every aggregate provenance field to match exactly."""

    if set(frozen) != set(live):
        raise ValueError(
            "Aggregate calibration provenance fields differ: " f"frozen={sorted(frozen)}, audit={sorted(live)}."
        )
    mismatches = {
        name: {"frozen": frozen[name], "audit": live[name]} for name in sorted(frozen) if frozen[name] != live[name]
    }
    if mismatches:
        raise ValueError(f"Aggregate calibration provenance mismatch: {mismatches}.")


def write_json(path: pathlib.Path | str, payload: Mapping[str, Any]) -> pathlib.Path:
    """Atomically write a JSON artifact."""

    target = pathlib.Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def _manifest_role(manifest: Mapping[str, Any], split_name: str) -> str | None:
    roles = manifest.get("split_roles", {})
    if roles and not isinstance(roles, Mapping):
        raise ValueError("split_manifest split_roles must be a mapping when present.")
    if split_name in {"train", "calibration", "validation"}:
        if split_name in roles and str(roles[split_name]) != split_name:
            raise ValueError(f"split_roles may not override canonical split {split_name!r}.")
        return split_name
    if split_name in roles:
        return str(roles[split_name])
    return None


def _validated_manifest_groups(name: str, values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size:
        raise ValueError(f"split_manifest {name!r} must contain a non-empty one-dimensional group list.")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"split_manifest {name!r} must contain integer group IDs.")
    if np.any(raw < 0):
        raise ValueError(f"split_manifest {name!r} must contain non-negative group IDs.")
    groups = raw.astype(np.uint64)
    if np.unique(groups).size != groups.size:
        raise ValueError(f"split_manifest {name!r} contains duplicate groups.")
    return groups


def load_split_manifest(
    path: pathlib.Path | str,
    *,
    split_name: str,
    required_role: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Load one explicit development split and prove its declared role.

    Custom names are supported only when ``split_roles`` declares their role.
    This prevents a file named ``final_group_ids`` from being passed to a
    development audit merely by changing a CLI flag.
    """

    manifest_path = validate_development_path(path, description="split manifest")
    manifest = json.loads(manifest_path.read_text())
    role = _manifest_role(manifest, split_name)
    if role != required_role:
        raise ValueError(
            f"split {split_name!r} has role {role!r}; this command requires an explicit {required_role!r} split."
        )
    lowered = split_name.lower()
    if _NON_DEVELOPMENT_COMPONENT.search(lowered):
        raise ValueError(f"Non-development split {split_name!r} is forbidden by this command.")
    key = f"{split_name}_group_ids"
    if key not in manifest:
        raise KeyError(f"split_manifest is missing {key!r}.")
    groups = np.sort(_validated_manifest_groups(key, manifest[key]))

    # Validate every declared group list, not only the requested one.  The
    # aggregate rule is invalid if calibration and validation overlap.
    declared: dict[str, np.ndarray] = {}
    for manifest_key, values in manifest.items():
        if not manifest_key.endswith("_group_ids") or manifest_key == "bootstrap_train_group_counts":
            continue
        name = manifest_key.removesuffix("_group_ids")
        candidate = _validated_manifest_groups(manifest_key, values)
        declared[name] = candidate
    names = sorted(declared)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = np.intersect1d(declared[left], declared[right])
            if overlap.size:
                raise ValueError(
                    f"split_manifest groups overlap between {left!r} and {right!r}: " f"{overlap[:10].tolist()}."
                )
    return manifest, groups


def validate_development_dataset_paths(paths: Sequence[pathlib.Path | str]) -> tuple[str, ...]:
    """Reject paths visibly labelled as final/test/holdout before opening them."""

    if not paths:
        raise ValueError("At least one development dataset path is required.")
    resolved = tuple(str(validate_development_path(path, description="dataset")) for path in paths)
    if len(resolved) != len(set(resolved)):
        raise ValueError("Development dataset paths must not contain duplicates.")
    return resolved


def validate_development_path(path: pathlib.Path | str, *, description: str) -> pathlib.Path:
    """Reject a visibly final/test/holdout path before opening it."""

    resolved = pathlib.Path(path).resolve()
    # Only the input itself and its immediate container describe the supplied
    # partition. Scanning arbitrary ancestors would reject ordinary pytest
    # and CI temporary directories. A generic pytest ``test_*`` parent is not
    # evidence that a leaf artifact belongs to the held-out test partition.
    semantic_parts = resolved.parts[-2:]
    partition_match = any(_FINAL_OR_HOLDOUT_COMPONENT.search(part) for part in semantic_parts)
    pytest_case_parent = bool(
        _PYTEST_CASE_COMPONENT.fullmatch(resolved.parent.name)
        and _PYTEST_RUN_COMPONENT.fullmatch(resolved.parent.parent.name)
    )
    test_match = bool(
        _TEST_COMPONENT.search(resolved.name)
        or (_TEST_COMPONENT.search(resolved.parent.name) and not pytest_case_parent)
    )
    if partition_match or test_match:
        raise ValueError(f"Development-only command refuses final/test/holdout {description} path: {resolved}.")
    return resolved


def episode_group_ids(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    task = np.asarray(arrays["task_id"], dtype=np.uint64)
    episode = np.asarray(arrays["episode_id"], dtype=np.uint64)
    return task * np.uint64(1_000_000_000) + episode


def load_development_split(
    dataset_paths: Sequence[pathlib.Path | str],
    *,
    split_manifest: pathlib.Path | str,
    split_name: str,
    required_role: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any], tuple[str, ...]]:
    """Load development arrays and resolve an explicit episode-disjoint split."""

    inputs = validate_development_dataset_paths(dataset_paths)
    manifest, requested_groups = load_split_manifest(
        split_manifest,
        split_name=split_name,
        required_role=required_role,
    )
    arrays = horizon_dataset.load_counterfactual_arrays(inputs)
    groups = episode_group_ids(arrays)
    indices = np.flatnonzero(np.isin(groups, requested_groups))
    if not indices.size:
        raise ValueError(f"No development roots matched split {split_name!r}.")
    matched_groups = np.unique(groups[indices])
    missing_groups = np.setdiff1d(requested_groups, matched_groups)
    if missing_groups.size:
        raise ValueError(
            f"Development dataset is missing {missing_groups.size} groups declared by split {split_name!r}: "
            f"{missing_groups[:10].tolist()}."
        )
    identities = np.stack([np.asarray(arrays[field], dtype=np.uint64) for field in _IDENTITY_FIELDS], axis=1)
    selected_identities = identities[indices]
    if np.unique(selected_identities, axis=0).shape[0] != selected_identities.shape[0]:
        raise ValueError(f"Split {split_name!r} contains duplicate counterfactual root identities.")
    return arrays, indices, groups[indices], manifest, inputs


def predict_split(
    predictor_dir: pathlib.Path | str,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    params: pathlib.Path | str | None,
    batch_size: int,
    inference_initialization_seed: int,
) -> tuple[Any, dict[str, np.ndarray], pathlib.Path]:
    """Restore the existing predictor and return predictions for one split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    # Reuse the established restore/inference path without editing the legacy
    # calibrator.  The import is local so pure metric tests do not initialize JAX.
    import calibrate_execution_horizon_predictor as legacy_calibrator

    directory = pathlib.Path(predictor_dir).resolve()
    config = legacy_calibrator._load_config(directory / "predictor_config.json")  # noqa: SLF001
    candidate_rows = np.asarray(arrays["candidate_horizons"], dtype=np.int64)
    expected = np.asarray(config.candidate_horizons, dtype=np.int64)
    if candidate_rows.ndim != 2 or not np.all(candidate_rows == expected[None, :]):
        raise ValueError("Development dataset and predictor candidate horizons differ.")
    params_path = pathlib.Path(params).resolve() if params is not None else directory / "params"
    module = legacy_calibrator._restore(  # noqa: SLF001
        config,
        params_path,
        inference_initialization_seed,
    )
    predictions = legacy_calibrator._predict(module, dict(arrays), indices, batch_size)  # noqa: SLF001
    if len(next(iter(predictions.values()))) != len(indices):
        raise ValueError("Predictor output row count does not match the requested split.")
    return config, predictions, params_path


def split_labels(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    required = (
        "success_count",
        "trial_count",
        "elapsed_mean",
        "remaining_calls_mean",
        "dangerous_long_count",
        "paired_trial_count",
        "trial_valid",
        "trial_success",
    )
    missing = sorted(set(required).difference(arrays))
    if missing:
        raise KeyError(f"Counterfactual dataset is missing aggregate-risk labels: {missing}.")
    return {name: np.asarray(arrays[name])[indices] for name in required}


def distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def cluster_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Equal-cluster bootstrap over task-by-episode initial-state groups."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.uint64)
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    if not values.size:
        return {"mean": None, "ci95": [None, None], "num_roots": 0, "num_clusters": 0}
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    unique_groups = np.unique(groups)
    cluster_means = np.asarray([np.mean(values[groups == group]) for group in unique_groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((samples,), dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        sampled = rng.integers(0, cluster_means.size, size=(count, cluster_means.size))
        means[start : start + count] = np.mean(cluster_means[sampled], axis=1)
    return {
        "mean": float(np.mean(cluster_means)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "num_roots": int(values.size),
        "num_clusters": int(cluster_means.size),
    }


def wilson_upper(successes: float, trials: float, z: float = 1.96) -> float:
    if trials <= 0:
        return 1.0
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    margin = z / denominator * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
    return float(min(1.0, center + margin))


def validate_raw_paired_labels(
    labels: Mapping[str, np.ndarray],
    *,
    candidate_horizons: Sequence[int],
    reference_horizon: int,
) -> None:
    """Recompute every count used by aggregate gates from raw paired trials."""

    candidates = tuple(int(value) for value in candidate_horizons)
    trial_count = np.asarray(labels["trial_count"])
    success_count = np.asarray(labels["success_count"])
    expected_shape = trial_count.shape
    if expected_shape != success_count.shape or len(expected_shape) != 2 or expected_shape[1] != len(candidates):
        raise ValueError("Count labels must have shape [roots, candidate_horizons].")
    trial_valid_raw = np.asarray(labels["trial_valid"])
    trial_success_raw = np.asarray(labels["trial_success"])
    expected_trial_prefix = (*expected_shape,)
    if (
        trial_valid_raw.ndim != 3
        or trial_success_raw.shape != trial_valid_raw.shape
        or trial_valid_raw.shape[:2] != expected_trial_prefix
    ):
        raise ValueError("trial_valid/trial_success must have shape [roots, candidate_horizons, trials].")
    if not np.all(np.isin(trial_valid_raw, (False, True))) or not np.all(np.isin(trial_success_raw, (False, True))):
        raise ValueError("Raw paired trial labels must be binary.")
    trial_valid = trial_valid_raw.astype(np.bool_)
    trial_success = trial_success_raw.astype(np.bool_)
    recomputed_trials = np.sum(trial_valid, axis=-1, dtype=np.int64)
    recomputed_success = np.sum(trial_valid & trial_success, axis=-1, dtype=np.int64)
    if not np.array_equal(trial_count, recomputed_trials):
        raise ValueError("trial_count does not match raw trial_valid outcomes.")
    if not np.array_equal(success_count, recomputed_success):
        raise ValueError("success_count does not match raw paired success outcomes.")

    dangerous = np.asarray(labels["dangerous_long_count"])
    paired = np.asarray(labels["paired_trial_count"])
    if dangerous.shape != expected_shape or paired.shape != expected_shape:
        raise ValueError("Paired danger counts must have all-candidate width for script auditing.")
    recomputed_dangerous = np.zeros(expected_shape, dtype=np.int64)
    recomputed_paired = np.zeros(expected_shape, dtype=np.int64)
    reference_index = candidates.index(reference_horizon)
    for candidate_index, horizon in enumerate(candidates):
        if horizon <= reference_horizon:
            continue
        valid = trial_valid[:, reference_index] & trial_valid[:, candidate_index]
        recomputed_paired[:, candidate_index] = np.sum(valid, axis=-1, dtype=np.int64)
        recomputed_dangerous[:, candidate_index] = np.sum(
            valid & trial_success[:, reference_index] & ~trial_success[:, candidate_index],
            axis=-1,
            dtype=np.int64,
        )
    if not np.array_equal(paired, recomputed_paired):
        raise ValueError("paired_trial_count does not match raw paired trial_valid outcomes.")
    if not np.array_equal(dangerous, recomputed_dangerous):
        raise ValueError("dangerous_long_count does not match raw paired success outcomes.")


def selection_metrics(
    labels: Mapping[str, np.ndarray],
    *,
    selected_horizons: np.ndarray,
    candidate_horizons: Sequence[int],
    reference_horizon: int,
    cluster_ids: np.ndarray,
    bootstrap_samples: int,
    seed: int,
    success_noninferiority_margin: float,
    false_long_upper_bound: float,
) -> dict[str, Any]:
    """Evaluate the frozen selector with the same four preregistered gates."""

    candidates = tuple(int(value) for value in candidate_horizons)
    if reference_horizon not in candidates:
        raise ValueError("reference_horizon must be one of candidate_horizons.")
    selected_horizons = np.asarray(selected_horizons, dtype=np.int64)
    if selected_horizons.ndim != 1:
        raise ValueError("selected_horizons must be one-dimensional.")
    unknown = sorted(set(selected_horizons.tolist()).difference(candidates))
    if unknown:
        raise ValueError(f"Selector returned horizons absent from candidate_horizons: {unknown}.")
    candidate_to_index = {horizon: index for index, horizon in enumerate(candidates)}
    selected_indices = np.asarray([candidate_to_index[int(value)] for value in selected_horizons], dtype=np.int64)
    row = np.arange(selected_horizons.size)
    reference_index = candidate_to_index[reference_horizon]

    trials = np.asarray(labels["trial_count"], dtype=np.float64)
    successes = np.asarray(labels["success_count"], dtype=np.float64)
    success_rate = successes / np.maximum(trials, 1.0)
    elapsed = np.asarray(labels["elapsed_mean"], dtype=np.float64)
    calls = np.asarray(labels["remaining_calls_mean"], dtype=np.float64)
    expected_shape = (selected_horizons.size, len(candidates))
    shaped_values = (
        ("trial_count", trials),
        ("success_count", successes),
        ("elapsed_mean", elapsed),
        ("remaining_calls_mean", calls),
    )
    for name, values in shaped_values:
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite on every audited root and candidate.")
    if np.any(trials <= 0):
        raise ValueError("trial_count must be positive on every audited root and candidate.")
    validate_raw_paired_labels(
        labels,
        candidate_horizons=candidates,
        reference_horizon=reference_horizon,
    )

    selected_valid = trials[row, selected_indices] > 0
    reference_valid = trials[:, reference_index] > 0
    if not np.all(selected_valid & reference_valid):
        raise ValueError("Selected/reference outcomes must have positive trial_count on every audited root.")
    success_advantage = success_rate[row, selected_indices] - success_rate[:, reference_index]
    elapsed_advantage = elapsed[row, selected_indices] - elapsed[:, reference_index]
    calls_advantage = calls[row, selected_indices] - calls[:, reference_index]
    success_interval = cluster_bootstrap_interval(
        success_advantage,
        cluster_ids,
        seed=seed + 1,
        samples=bootstrap_samples,
    )
    elapsed_interval = cluster_bootstrap_interval(
        elapsed_advantage,
        cluster_ids,
        seed=seed + 2,
        samples=bootstrap_samples,
    )
    calls_interval = cluster_bootstrap_interval(
        calls_advantage,
        cluster_ids,
        seed=seed + 3,
        samples=bootstrap_samples,
    )

    selected_long = selected_horizons > reference_horizon
    dangerous = np.asarray(labels["dangerous_long_count"], dtype=np.float64)
    paired = np.asarray(labels["paired_trial_count"], dtype=np.float64)
    dangerous_total = float(np.sum(dangerous[row[selected_long], selected_indices[selected_long]]))
    paired_total = float(np.sum(paired[row[selected_long], selected_indices[selected_long]]))
    false_long_upper_95 = wilson_upper(dangerous_total, paired_total)
    gates = {
        "nonzero_long_coverage": bool(np.any(selected_long)),
        "success_noninferiority": bool(
            success_interval["ci95"][0] is not None and success_interval["ci95"][0] >= -success_noninferiority_margin
        ),
        "elapsed_improvement": bool(elapsed_interval["ci95"][1] is not None and elapsed_interval["ci95"][1] < 0.0),
        "false_long_control": bool(false_long_upper_95 <= false_long_upper_bound),
    }
    return {
        "num_roots": int(selected_horizons.size),
        "selected_h_distribution": distribution(selected_horizons),
        "long_h_coverage": float(np.mean(selected_long)),
        "short_h_intervention_rate": float(np.mean(selected_horizons < reference_horizon)),
        "counterfactual_success_rate": float(np.mean(success_rate[row, selected_indices])),
        "mean_elapsed_seconds": float(np.mean(elapsed[row, selected_indices])),
        "mean_remaining_calls": float(np.mean(calls[row, selected_indices])),
        "success_advantage_vs_reference_cluster_bootstrap": success_interval,
        "elapsed_advantage_vs_reference_cluster_bootstrap": elapsed_interval,
        "calls_advantage_vs_reference_cluster_bootstrap": calls_interval,
        "false_long_count": int(dangerous_total),
        "false_long_paired_trials": int(paired_total),
        "false_long_rate": float(dangerous_total / max(paired_total, 1.0)),
        "false_long_upper_95": false_long_upper_95,
        "gate_checks": gates,
        "offline_engineering_gate": bool(all(gates.values())),
    }


def decision_diagnostics(decisions: Sequence[Any]) -> dict[str, Any]:
    """Summarize independent and first-failure rejection counts per long H."""

    if not decisions:
        raise ValueError("At least one aggregate selector decision is required.")
    long_horizons = tuple(int(value) for value in decisions[0].long_horizons)
    per_h: dict[str, Any] = {}
    for position, horizon in enumerate(long_horizons):
        rejected_by = {}
        for field in _CONSTRAINT_FIELDS:
            passed = np.asarray([getattr(decision, field)[position] for decision in decisions], dtype=np.bool_)
            rejected_by[field.removesuffix("_pass")] = int(np.sum(~passed))
        eligible = np.asarray([decision.long_eligible[position] for decision in decisions], dtype=np.bool_)
        first_failure: dict[str, int] = {}
        rejection_reason_counts: dict[str, int] = {}
        for decision in decisions:
            reasons = decision.rejection_reasons[position]
            if isinstance(reasons, str):
                reasons = (reasons,) if reasons else ()
            reasons = tuple(str(value) for value in reasons)
            for reason in reasons:
                rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
            if reasons:
                first_failure[reasons[0]] = first_failure.get(reasons[0], 0) + 1
        per_h[str(horizon)] = {
            "roots_considered": len(decisions),
            "eligible_count": int(np.sum(eligible)),
            "eligible_coverage": float(np.mean(eligible)),
            "selected_count": int(sum(int(decision.selected_horizon) == horizon for decision in decisions)),
            "rejected_by_constraint_independent": rejected_by,
            "first_rejection_count": first_failure,
            "all_rejection_reason_counts": rejection_reason_counts,
            "scores": {
                "success": _score_summary(decisions, "success_score", position),
                "elapsed": _score_summary(decisions, "elapsed_score", position),
                "danger_probability": _score_summary(decisions, "danger_probability", position),
                "faster_probability": _score_summary(decisions, "faster_probability", position),
                "long_event_probability": _score_summary(decisions, "long_event_probability", position),
            },
        }
    return {
        "decision_reason_distribution": _string_distribution([str(decision.reason) for decision in decisions]),
        "per_long_horizon": per_h,
    }


def _score_summary(decisions: Sequence[Any], field: str, position: int) -> dict[str, float | None]:
    values = np.asarray([getattr(decision, field)[position] for decision in decisions], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    quantiles = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "mean": float(np.mean(values)),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def _string_distribution(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))
