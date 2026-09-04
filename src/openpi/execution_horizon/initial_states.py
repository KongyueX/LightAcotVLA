"""Explicit, fingerprinted LIBERO initial states for genuinely disjoint episodes.

Legacy evaluation may cycle through the benchmark's finite preset list. A bank
instead maps every episode ID to one frozen state; it never applies modulo.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import h5py
import numpy as np

from openpi.execution_horizon import dataset

FINGERPRINT_METHOD = "qpos_float64_round8_v1"


def fingerprint(state: np.ndarray, nq: int) -> str:
    """Identify initial poses, ignoring time and velocity-only differences."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or nq <= 0 or state.size <= nq or not np.all(np.isfinite(state)):
        raise ValueError("Initial state must be a finite flattened MuJoCo state with valid nq.")
    qpos = np.round(state[1 : 1 + nq], decimals=8).astype("<f8")
    qpos[qpos == 0] = 0.0  # Canonicalize negative zero before hashing.
    return hashlib.sha256(qpos.tobytes()).hexdigest()


class InitialStateBank:
    def __init__(self, directory: str | pathlib.Path) -> None:
        self.directory = pathlib.Path(directory).resolve()
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        self.sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes)
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("schema_version") != 1
            or self.manifest.get("fingerprint_method") != FINGERPRINT_METHOD
        ):
            raise ValueError("Initial-state bank is incomplete or has an unsupported schema.")
        self.tasks: dict[int, dict[str, Any]] = {}
        self.states: dict[int, np.ndarray] = {}
        self.generation_seeds: dict[int, np.ndarray] = {}
        self.fingerprints: dict[int, list[str]] = {}
        for entry in self.manifest["tasks"]:
            task_id = int(entry["task_id"])
            if task_id in self.tasks or entry["file"] != f"task{task_id:02d}.npz":
                raise ValueError("Bank contains duplicate tasks or an unexpected task filename.")
            path = self.directory / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Initial-state bank checksum mismatch: {path}.")
            with np.load(path, allow_pickle=False) as archive:
                states = np.asarray(archive["states"], dtype=np.float64)
                episode_ids = np.asarray(archive["episode_ids"], dtype=np.int64)
                seeds = np.asarray(archive["generation_seeds"], dtype=np.int64)
            preset_count = int(entry["preset_count"])
            expected_count = preset_count + int(self.manifest["generated_per_task"])
            if states.shape != (expected_count, int(entry["state_dim"])):
                raise ValueError(f"Invalid state shape for task{task_id}.")
            if not np.array_equal(episode_ids, np.arange(expected_count)) or seeds.shape != episode_ids.shape:
                raise ValueError(f"Bank episode IDs must be explicit and contiguous for task{task_id}.")
            identities = [fingerprint(state, int(entry["nq"])) for state in states]
            if identities != entry["fingerprints"] or len(set(identities)) != expected_count:
                raise ValueError(f"Duplicate or inconsistent initial poses in task{task_id} bank.")
            if np.any(seeds[:preset_count] != -1) or np.any(seeds[preset_count:] < 0):
                raise ValueError(f"Invalid generation provenance for task{task_id}.")
            self.tasks[task_id] = entry
            self.states[task_id] = states
            self.generation_seeds[task_id] = seeds
            self.fingerprints[task_id] = identities
        if not self.tasks or len(self.tasks) != int(self.manifest["max_tasks"]):
            raise ValueError("Initial-state bank task count is inconsistent.")

    def metadata(self) -> dict[str, Any]:
        return {
            "initial_state_bank": str(self.directory),
            "initial_state_bank_sha256": self.sha256,
            "initial_state_identity_mode": FINGERPRINT_METHOD,
        }

    def validate_presets(self, task_id: int, preset_states: np.ndarray) -> None:
        count = int(self.tasks[task_id]["preset_count"])
        expected = np.asarray(preset_states, dtype=np.float64)
        if not np.array_equal(self.states[task_id][:count], expected):
            raise ValueError(f"Current LIBERO preset states differ from the bank provenance for task{task_id}.")

    def state(self, task_id: int, episode_id: int) -> np.ndarray:
        if task_id not in self.states or not 0 <= episode_id < len(self.states[task_id]):
            raise ValueError(f"Initial-state bank has no task{task_id}/episode{episode_id}; modulo is forbidden.")
        return self.states[task_id][episode_id].copy()

    def identity(self, task_id: int, episode_id: int) -> tuple[int, str]:
        self.state(task_id, episode_id)
        return task_id, self.fingerprints[task_id][episode_id]

    def audit_partitions(self, partitions: dict[str, set[tuple[int, int]]]) -> dict[str, Any]:
        identities: dict[str, set[tuple[int, str]]] = {}
        for name, groups in partitions.items():
            identities[name] = {self.identity(task, episode) for task, episode in groups}
            if len(identities[name]) != len(groups):
                raise ValueError(f"Multiple episode IDs reuse the same initial pose within {name}.")
            for previous in identities.keys() - {name}:
                if identities[name] & identities[previous]:
                    raise ValueError(f"Initial-state overlap between {previous} and {name}.")
        return {
            "status": "complete",
            **self.metadata(),
            "partition_group_counts": {name: len(groups) for name, groups in partitions.items()},
            "pairwise_initial_state_overlap": 0,
            "semantics": "Task-qualified frozen initial-pose identities, not merely distinct episode numbers.",
        }


def audit_bank_prefix(parent: InitialStateBank, child: InitialStateBank) -> dict[str, Any]:
    """Prove that ``child`` extends ``parent`` without changing any old state."""

    if parent.manifest["task_suite"] != child.manifest["task_suite"]:
        raise ValueError("Initial-state bank lineage task suites differ.")
    if set(parent.tasks) != set(child.tasks):
        raise ValueError("Initial-state bank lineage task sets differ.")
    prefix_counts: dict[str, int] = {}
    new_counts: dict[str, int] = {}
    strictly_extended = False
    for task_id in sorted(parent.tasks):
        parent_states = parent.states[task_id]
        count = len(parent_states)
        if len(child.states[task_id]) < count:
            raise ValueError(f"Child initial-state bank is shorter than its parent for task{task_id}.")
        if int(parent.tasks[task_id]["preset_count"]) != int(child.tasks[task_id]["preset_count"]):
            raise ValueError(f"Initial-state bank lineage preset count differs for task{task_id}.")
        if int(parent.tasks[task_id]["state_dim"]) != int(child.tasks[task_id]["state_dim"]):
            raise ValueError(f"Initial-state bank lineage state dimension differs for task{task_id}.")
        if int(parent.tasks[task_id]["nq"]) != int(child.tasks[task_id]["nq"]):
            raise ValueError(f"Initial-state bank lineage nq differs for task{task_id}.")
        if not np.array_equal(child.states[task_id][:count], parent_states):
            raise ValueError(f"Child initial-state bank changed parent states for task{task_id}.")
        if not np.array_equal(child.generation_seeds[task_id][:count], parent.generation_seeds[task_id]):
            raise ValueError(f"Child initial-state bank changed parent generation seeds for task{task_id}.")
        if child.fingerprints[task_id][:count] != parent.fingerprints[task_id]:
            raise ValueError(f"Child initial-state bank changed parent fingerprints for task{task_id}.")
        prefix_counts[str(task_id)] = count
        new_count = len(child.states[task_id]) - count
        new_counts[str(task_id)] = new_count
        strictly_extended |= new_count > 0
    if not strictly_extended:
        raise ValueError("Child initial-state bank does not extend its parent with any new episodes.")
    return {
        "status": "complete",
        "schema_version": 1,
        "semantics": "The child bank exactly preserves every parent state, seed, and fingerprint as a prefix.",
        "parent": parent.metadata(),
        "child": child.metadata(),
        "prefix_episode_counts_by_task": prefix_counts,
        "new_episode_counts_by_task": new_counts,
    }


def audit_partitions_across_banks(
    partitions: dict[str, tuple[InitialStateBank, set[tuple[int, int]]]],
) -> dict[str, Any]:
    """Audit task-qualified pose isolation when partitions use different banks."""

    if not partitions:
        raise ValueError("At least one initial-state partition is required.")
    suites = {str(bank.manifest["task_suite"]) for bank, _ in partitions.values()}
    if len(suites) != 1:
        raise ValueError("Initial-state partition banks use different task suites.")
    identities: dict[str, set[tuple[int, str]]] = {}
    for name, (bank, groups) in partitions.items():
        if not name or name in identities:
            raise ValueError("Initial-state partition names must be unique and non-empty.")
        identities[name] = {bank.identity(task, episode) for task, episode in groups}
        if len(identities[name]) != len(groups):
            raise ValueError(f"Multiple episode IDs reuse the same initial pose within {name}.")
        for previous in identities.keys() - {name}:
            if identities[name] & identities[previous]:
                raise ValueError(f"Initial-state overlap between {previous} and {name}.")
    return {
        "status": "complete",
        "schema_version": 2,
        "partition_banks": {name: bank.metadata() for name, (bank, _) in partitions.items()},
        "partition_group_counts": {name: len(groups) for name, (_, groups) in partitions.items()},
        "pairwise_initial_state_overlap": 0,
        "semantics": "Task-qualified frozen initial-pose identities compared across explicitly bound banks.",
    }


def dataset_groups(
    data_dirs: list[str], bank: InitialStateBank, *, allow_legacy_presets: bool = False
) -> set[tuple[int, int]]:
    """Read identity columns and provenance only, without loading visual features."""
    groups: set[tuple[int, int]] = set()
    for shard in dataset.discover_shards(data_dirs):
        with h5py.File(shard, "r") as handle:
            metadata = json.loads(handle.attrs["metadata_json"])
            if metadata.get("task_suite") != bank.manifest["task_suite"]:
                raise ValueError(f"Task suite differs from the initial-state bank: {shard}.")
            legacy = metadata.get("initial_state_bank_sha256") is None
            if legacy and not allow_legacy_presets:
                raise ValueError(f"Fresh data lacks explicit initial-state provenance: {shard}.")
            if not legacy and (
                metadata.get("initial_state_bank_sha256") != bank.sha256
                or metadata.get("initial_state_identity_mode") != FINGERPRINT_METHOD
            ):
                raise ValueError(f"Dataset initial-state bank fingerprint mismatch: {shard}.")
            for task, episode in zip(handle["task_id"][:], handle["episode_id"][:], strict=True):
                key = (int(task), int(episode))
                bank.identity(*key)
                if legacy and key[1] >= int(bank.tasks[key[0]]["preset_count"]):
                    raise ValueError(f"Legacy episode {key} can alias a preset via modulo.")
                groups.add(key)
    return groups
