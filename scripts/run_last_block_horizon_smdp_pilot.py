"""Run at most three critic-warmup/last-Transformer-block PPO development rounds."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any

import numpy as np
import run_ordered_horizon_smdp_pilot as base

MODE = "ordered_smdp_last_block"
TRAIN_EPISODES = (tuple(range(120, 130)), tuple(range(130, 140)), tuple(range(140, 150)))
VALIDATION_EPISODES = base.VALIDATION_EPISODES
FINAL_EPISODES = base.FINAL_EPISODES


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--reuse-first-round-dir", type=pathlib.Path,
        help="Reuse a completed round01 collection and its original step0 checkpoint in a fresh output directory.",
    )
    return parser


def validate_train_bank(bank_dir: pathlib.Path) -> dict[str, Any]:
    """Confirm existing bank ID coverage without loading observations or hashing states."""
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("task_suite") != "libero_10":
        raise ValueError("Training bank must be complete and use libero_10.")
    training_ids = set().union(*TRAIN_EPISODES)
    if training_ids & (set(VALIDATION_EPISODES) | set(FINAL_EPISODES)):
        raise ValueError("Training IDs overlap validation/final IDs.")
    entries = {int(entry["task_id"]): entry for entry in manifest["tasks"]}
    counts = {}
    for task_id in range(10):
        with np.load(bank_dir / entries[task_id]["file"], allow_pickle=False) as archive:
            ids = set(int(value) for value in archive["episode_ids"])
        if not training_ids <= ids:
            raise ValueError(f"Bank task {task_id} lacks requested IDs 120–149.")
        counts[str(task_id)] = len(ids)
    return {
        "directory": str(bank_dir), "status": "complete", "task_suite": "libero_10",
        "round_episode_ids": [list(ids) for ids in TRAIN_EPISODES],
        "available_ids_per_task": counts, "validation_or_final_ids_in_training": False,
    }


def build_eval_command(
    args: argparse.Namespace, *, checkpoint: pathlib.Path, output_dir: pathlib.Path,
    episodes: tuple[int, ...], sample: bool,
) -> list[str]:
    command = base.build_eval_command(
        args, checkpoint=checkpoint, output_dir=output_dir, episodes=episodes, sample=sample,
    )
    replacements = {
        base.MODE: MODE,
        "--ordered-smdp-params": "--last-block-smdp-params",
        "--ordered-smdp-sample": "--last-block-smdp-sample",
    }
    return [replacements.get(value, value) for value in command]


def build_train_command(
    args: argparse.Namespace, *, checkpoint: pathlib.Path, rollout_dir: pathlib.Path, output_dir: pathlib.Path,
) -> list[str]:
    command = base.build_train_command(
        args, checkpoint=checkpoint, rollout_dir=rollout_dir, output_dir=output_dir,
    )
    old_script = str(args.code_dir / "scripts/train_ordered_horizon_smdp.py")
    command[command.index(old_script)] = str(args.code_dir / "scripts/train_last_block_horizon_smdp.py")
    command.extend(["--critic-warmup-epochs", "10"])
    return command


def _environment(args: argparse.Namespace, *, trainer: bool) -> dict[str, str]:
    env = base._environment(args, cpu=trainer)
    # Client selector JAX runs on CPU; leave CUDA/EGL visibility unchanged for simulation.
    env["JAX_PLATFORMS"] = "cpu"
    return env


def _run_stage(
    args: argparse.Namespace, parent: pathlib.Path, name: str, command: list[str], *, trainer: bool,
) -> None:
    base._status(args.output_dir, name, stage_dir=str(parent), command=command)
    base._write_json(parent / f"{name}.command.json", {"command": command, "jax_platforms": "cpu"})
    with (parent / f"{name}.log").open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command, cwd=args.code_dir, env=_environment(args, trainer=trainer),
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            code = completed.returncode
        except OSError as error:
            log.write(str(error) + "\n")
            code = 127
    (parent / f"{name}.exit").write_text(f"{code}\n", encoding="utf-8")
    if code:
        raise subprocess.CalledProcessError(code, command)


def _reuse_first_round(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path, dict, dict]:
    old_round = args.reuse_first_round_dir.resolve()
    if old_round.name != "round01" or (old_round / "collect.exit").read_text().strip() != "0":
        raise ValueError("Only a successfully completed round01 collection can be reused.")
    old_protocol = json.loads((old_round.parent / "run_config.json").read_text())
    for name in ("anchor_predictor_dir", "reference_eval_dir"):
        if pathlib.Path(old_protocol[name]).resolve() != getattr(args, name).resolve():
            raise ValueError(f"Reused collection has a different {name}.")
    if (
        old_protocol["seed"] != args.seed
        or old_protocol["maximum_rounds"] != 3
        or old_protocol["training_bank"]["round_episode_ids"] != [list(ids) for ids in TRAIN_EPISODES]
        or old_protocol["validation_episode_ids"] != list(VALIDATION_EPISODES)
        or old_protocol["final_episode_ids"] != list(FINAL_EPISODES)
    ):
        raise ValueError("Reused collection has a different three-round seed/episode protocol.")
    collection_dir = old_round / "collection"
    rows, summary = base._read_eval(
        collection_dir, mode=MODE, episodes=TRAIN_EPISODES[0], seed=args.seed,
    )
    config = json.loads((collection_dir / "run_config.json").read_text())
    expected = {
        "modes": [MODE], "last_block_smdp_sample": True, "episode_ids": list(TRAIN_EPISODES[0]),
        "num_trials_per_task": 10, "task_start": 0, "max_tasks": 10, "seed": args.seed,
        "model_action_horizon": 25, "action_cot_denoising_steps": 10, "final_denoising_steps": 10,
        "num_steps_wait": 10, "resize_size": 224, "warmup_requests": 1, "initial_state_offset": 0,
    }
    for name, value in expected.items():
        if config.get(name) != value or summary["config"].get(name) != value:
            raise ValueError(f"Reused first-round collection has incompatible {name}.")
    for source in (config, summary["config"]):
        if pathlib.Path(source["initial_state_bank"]).resolve() != args.train_state_bank.resolve():
            raise ValueError("Reused first-round collection uses a different training bank.")
    checkpoint = pathlib.Path(config["last_block_smdp_params"]).resolve()
    if (
        pathlib.Path(summary["config"]["last_block_smdp_params"]).resolve() != checkpoint
        or checkpoint != (old_round.parent / "step0/checkpoint").resolve()
    ):
        raise ValueError("Reused first-round collection is not bound to its original step0 checkpoint.")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if (
        not (checkpoint / "params").is_dir()
        or pathlib.Path(metadata["anchor_predictor_dir"]).resolve() != args.anchor_predictor_dir.resolve()
        or metadata["seed"] != args.seed
    ):
        raise ValueError("Reused step0 checkpoint does not match the requested A anchor/seed.")
    return checkpoint, collection_dir, rows, summary


def _run(args: argparse.Namespace) -> dict[str, Any]:
    bank = validate_train_bank(args.train_state_bank)
    baseline_rows, reference_summary = base._read_eval(
        args.reference_eval_dir, mode=base.ANCHOR_MODE, episodes=VALIDATION_EPISODES, seed=args.seed,
    )
    baseline = base.summarize(baseline_rows)
    reused_first = _reuse_first_round(args) if args.reuse_first_round_dir is not None else None
    protocol = {
        "experiment": "critic_mc_warmup_existing_last_transformer_block_smdp_ppo",
        "maximum_rounds": 3, "critic_warmup_epochs": 10, "ppo_epochs_per_round": 4,
        "frozen_vla": True, "new_transformer_layers": 0,
        "actor_scope": "A existing last Transformer block and ordered head; earlier encoder stays frozen",
        "cached_inputs": "fresh last-block input tokens and context; not old pooled-feature-only trajectories",
        "code_dir": str(args.code_dir), "server": {"host": args.host, "port": args.port},
        "anchor_predictor_dir": str(args.anchor_predictor_dir), "seed": args.seed,
        "reference_eval_dir": str(args.reference_eval_dir),
        "original_reference_eval_dir": str(args.original_reference_eval_dir) if args.original_reference_eval_dir else None,
        "training_bank": bank, "validation_episode_ids": list(VALIDATION_EPISODES),
        "final_episode_ids": list(FINAL_EPISODES), "maximum_validation_rpc_seconds": 3.0,
        "selection": "lexicographic (validation success_count, -mean_rpc_seconds), strictly above A",
        "unchanged_actor_validation": "reuse the input actor's existing validation; retain its updated critic",
        "all_results_development": True,
        "reuse_first_round_dir": str(args.reuse_first_round_dir) if args.reuse_first_round_dir is not None else None,
        "caveat": "Validation/final presets were previously used in development; this is not independent confirmation.",
    }
    base._write_json(args.output_dir / "run_config.json", protocol)
    step0 = args.output_dir / "step0"
    step0.mkdir()
    if reused_first is None:
        checkpoint = step0 / "checkpoint"
        initialize = (
            "from openpi.execution_horizon.last_block_smdp import LastBlockHorizonSelector; "
            f"selector=LastBlockHorizonSelector.initialize_from_predictor({str(args.anchor_predictor_dir)!r}, "
            f"seed={args.seed}); selector.save({str(checkpoint)!r}); print('INITIALIZED_EXISTING_LAST_BLOCK')"
        )
        _run_stage(args, step0, "initialize", [str(args.python), "-c", initialize], trainer=True)
    else:
        checkpoint = reused_first[0]
    base._write_json(step0 / "summary.json", {
        "status": "complete", "selector_checkpoint": str(checkpoint), "validation": baseline,
        "validation_reused": True, "validation_source": str(args.reference_eval_dir),
        "validation_source_checkpoint": str(args.anchor_predictor_dir),
        "initialization_reused": reused_first is not None,
        "reference_source_summary": reference_summary,
    })
    validation_rows = baseline_rows
    validation_source = args.reference_eval_dir
    validation_source_checkpoint = args.anchor_predictor_dir
    rounds = []
    for round_index, episodes in enumerate(TRAIN_EPISODES, start=1):
        round_dir = args.output_dir / f"round{round_index:02d}"
        round_dir.mkdir()
        input_checkpoint = checkpoint
        collection_reused = round_index == 1 and reused_first is not None
        if collection_reused:
            _, collection_dir, collection_rows, collection_summary = reused_first
            base._write_json(round_dir / "collection_reused.json", {
                "source_collection_dir": str(collection_dir), "input_checkpoint": str(input_checkpoint),
                "episodes": list(episodes), "records": len(collection_rows),
            })
        else:
            collection_dir = round_dir / "collection"
            _run_stage(args, round_dir, "collect", build_eval_command(
                args, checkpoint=input_checkpoint, output_dir=collection_dir, episodes=episodes, sample=True,
            ), trainer=False)
            collection_rows, collection_summary = base._read_eval(
                collection_dir, mode=MODE, episodes=episodes, seed=args.seed,
            )
        training_dir = round_dir / "training"
        _run_stage(args, round_dir, "train", build_train_command(
            args, checkpoint=input_checkpoint, rollout_dir=collection_dir, output_dir=training_dir,
        ), trainer=True)
        training = json.loads((training_dir / "summary.json").read_text(encoding="utf-8"))
        if training.get("status") != "complete" or not isinstance(training.get("actor_changed"), bool):
            raise ValueError(f"Trainer must complete and report actor_changed: {training_dir}")
        checkpoint = pathlib.Path(training["selector_params"])
        if not (checkpoint / "params").is_dir() or not (checkpoint / "metadata.json").is_file():
            raise FileNotFoundError(f"Last-block checkpoint directory is incomplete: {checkpoint}")
        actor_changed = training["actor_changed"]
        if actor_changed and int(training["accepted_epochs"]) == 0:
            raise ValueError("Trainer reports actor changes without an accepted PPO epoch.")
        if actor_changed:
            validation_source = round_dir / "validation"
            _run_stage(args, round_dir, "validate", build_eval_command(
                args, checkpoint=checkpoint, output_dir=validation_source,
                episodes=VALIDATION_EPISODES, sample=False,
            ), trainer=False)
            validation_rows, _ = base._read_eval(
                validation_source, mode=MODE, episodes=VALIDATION_EPISODES, seed=args.seed,
            )
            validation_source_checkpoint = checkpoint
        else:
            base._write_json(round_dir / "validation_reused.json", {
                "reason": "Actor unchanged; updated critic alone does not require new greedy validation.",
                "source_eval_dir": str(validation_source),
                "source_actor_checkpoint": str(validation_source_checkpoint),
                "input_checkpoint": str(input_checkpoint), "output_checkpoint": str(checkpoint),
            })
        validation = base.summarize(validation_rows)
        result = {
            "round": round_index, "status": "complete", "input_checkpoint": str(input_checkpoint),
            "selector_checkpoint": str(checkpoint), "collection": base.summarize(collection_rows),
            "collection_reused": collection_reused, "collection_dir": str(collection_dir),
            "collection_source_summary": collection_summary, "training_summary": training,
            "actor_changed": actor_changed, "validation_reused": not actor_changed,
            "validation_source": str(validation_source),
            "validation_source_checkpoint": str(validation_source_checkpoint),
            "validation": validation, "validation_vs_A": base.compare(baseline_rows, validation_rows),
            "eligible": base.eligible(validation, baseline),
        }
        base._write_json(round_dir / "summary.json", result)
        rounds.append(result)
    candidates = [result for result in rounds if result["eligible"] and result["actor_changed"]]
    result = {
        "status": "complete", **protocol, "baseline_validation": baseline, "rounds": rounds,
        "final_run": False, "selected_system": "A", "selected_checkpoint": None,
    }
    if not candidates:
        result["conclusion"] = "No updated actor exceeded A under validation/RPC criteria; retain A and skip final."
        return result
    best = max(candidates, key=lambda item: (
        item["validation"]["success_count"], -item["validation"]["mean_rpc_seconds"],
    ))
    selected_checkpoint = pathlib.Path(best["selector_checkpoint"])
    final_dir = args.output_dir / "final"
    _run_stage(args, args.output_dir, "final_eval", build_eval_command(
        args, checkpoint=selected_checkpoint, output_dir=final_dir, episodes=FINAL_EPISODES, sample=False,
    ), trainer=False)
    final_rows, _ = base._read_eval(final_dir, mode=MODE, episodes=FINAL_EPISODES, seed=args.seed)
    anchor_rows, _ = base._read_eval(
        args.reference_eval_dir, mode=base.ANCHOR_MODE, episodes=FINAL_EPISODES, seed=args.seed,
    )
    result.update(
        final_run=True, selected_system=MODE, selected_checkpoint=str(selected_checkpoint), selected_round=best["round"],
        final=base.summarize(final_rows), final_vs_A=base.compare(anchor_rows, final_rows),
        conclusion="One validation-selected last-block actor completed the sole development final; no further rounds.",
        legacy_A_pilot_caveat=(
            "Earlier A pilot reported 189/200; the reference here is episode0–19 from the current 1000-episode A run. "
            "Identical task/episode IDs do not imply byte-identical simulator trajectories across runs."
        ),
    )
    if args.original_reference_eval_dir is not None:
        original_rows, original_summary = base._read_eval(
            args.original_reference_eval_dir, mode="original", episodes=FINAL_EPISODES, seed=args.seed,
        )
        result["final_vs_historical_original"] = base.compare(original_rows, final_rows)
        result["historical_original_source_summary"] = original_summary
    return result


def main(args: argparse.Namespace) -> None:
    for name in ("output_dir", "code_dir", "python", "train_state_bank", "reference_eval_dir", "anchor_predictor_dir"):
        setattr(args, name, getattr(args, name).absolute())
    if args.original_reference_eval_dir is not None:
        args.original_reference_eval_dir = args.original_reference_eval_dir.absolute()
    if args.reuse_first_round_dir is not None:
        args.reuse_first_round_dir = args.reuse_first_round_dir.absolute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        summary = _run(args)
        base._write_json(args.output_dir / "summary.json", summary)
        base._status(args.output_dir, "complete", selected_system=summary["selected_system"], final_run=summary["final_run"])
    except Exception as error:
        base._status(args.output_dir, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
