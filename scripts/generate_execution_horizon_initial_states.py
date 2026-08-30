"""Freeze fresh LIBERO reset states without policy calls or GPU rendering.

Preset episodes are copied exactly; subsequent episode IDs are new reset draws,
not aliases of the finite preset list. Sampling never uses rollout outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np

from openpi.execution_horizon import initial_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--expected-preset-count", type=int, default=50)
    parser.add_argument("--generated-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-attempts-per-state", type=int, default=20)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    if (
        args.task_start < 0
        or args.max_tasks <= 0
        or args.generated_per_task <= 0
        or args.expected_preset_count <= 0
        or args.seed < 0
        or not 1 <= args.max_attempts_per_state <= 100
    ):
        raise ValueError("Invalid bank size, task range, seed, or attempt limit.")
    output = args.output_dir.resolve()
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    if (output / "manifest.json").exists():
        bank = initial_states.InitialStateBank(output)
        if bank.manifest["generation_config"] != config:
            raise ValueError("Existing initial-state bank belongs to a different generation request.")
        print("INITIAL_STATE_BANK_ALREADY_COMPLETE", bank.sha256, flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an incomplete initial-state bank: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # Import LIBERO only in the CPU generator; the bank reader needs no simulator.
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.utils.errors import RandomizationError

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.task_start + args.max_tasks > suite.n_tasks:
        raise ValueError("Requested task range exceeds the LIBERO suite.")
    entries = []
    started = time.monotonic()
    for task_id in range(args.task_start, args.task_start + args.max_tasks):
        _write_json(
            output / "status.json", {"status": "generating", "task_id": task_id, "completed_tasks": len(entries)}
        )
        task = suite.get_task(task_id)
        presets = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
        if presets.ndim != 2 or len(presets) != args.expected_preset_count:
            raise ValueError(f"Unexpected preset state count/shape for task{task_id}: {presets.shape}.")
        np.random.seed(args.seed + task_id * 100_000)
        environment = ControlEnv(
            bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        try:
            nq = int(environment.env.sim.model.nq)
            states = [state.copy() for state in presets]
            fingerprints = [initial_states.fingerprint(state, nq) for state in states]
            seen = set(fingerprints)
            if len(seen) != len(states):
                raise ValueError(f"The official task{task_id} presets already contain duplicate initial poses.")
            seeds = [-1] * len(states)
            for episode_id in range(len(presets), len(presets) + args.generated_per_task):
                for attempt in range(args.max_attempts_per_state):
                    seed = args.seed + task_id * 100_000 + episode_id * 100 + attempt
                    environment.seed(seed)
                    try:
                        # The wrapper retries forever; bound placement retries here instead.
                        environment.env.reset()
                    except RandomizationError:
                        continue
                    state = np.asarray(environment.env.sim.get_state().flatten(), dtype=np.float64)
                    if state.shape != presets.shape[1:]:
                        raise ValueError(f"Reset/preset physics shapes differ for task{task_id}.")
                    identity = initial_states.fingerprint(state, nq)
                    if identity in seen or environment.env._check_success():  # noqa: SLF001
                        continue
                    states.append(state)
                    fingerprints.append(identity)
                    seen.add(identity)
                    seeds.append(seed)
                    break
                else:
                    raise RuntimeError(f"Could not sample a new valid task{task_id}/episode{episode_id} initial pose.")
        finally:
            environment.close()
        path = output / f"task{task_id:02d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                states=np.stack(states),
                episode_ids=np.arange(len(states), dtype=np.int64),
                generation_seeds=np.asarray(seeds, dtype=np.int64),
            )
        temporary.replace(path)
        entries.append(
            {
                "task_id": task_id,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "preset_count": len(presets),
                "state_dim": int(presets.shape[1]),
                "nq": nq,
                "fingerprints": fingerprints,
            }
        )
        print(f"INITIAL_STATES_OK task{task_id} presets={len(presets)} fresh={args.generated_per_task}", flush=True)
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "fingerprint_method": initial_states.FINGERPRINT_METHOD,
        "task_suite": args.task_suite_name,
        "max_tasks": args.max_tasks,
        "generated_per_task": args.generated_per_task,
        "generation_config": config,
        "tasks": entries,
        "elapsed_seconds": time.monotonic() - started,
        "semantics": "Official presets plus fresh CPU reset draws; fresh draws are not the official fixed-state benchmark.",
    }
    _write_json(output / "manifest.json", manifest)
    bank = initial_states.InitialStateBank(output)
    _write_json(output / "status.json", {"status": "complete", **bank.metadata()})
    print("INITIAL_STATE_BANK_COMPLETE", bank.sha256, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
