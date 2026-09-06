"""Train one last-block PPO candidate with fixed positive RPC cost on the original A batch."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import run_last_block_horizon_smdp_pilot as last_block

base = last_block.base
COST_MULTIPLIER = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--reuse-first-round-dir", type=pathlib.Path, required=True)
    return parser


def build_train_command(
    args: argparse.Namespace, *, checkpoint: pathlib.Path, rollout_dir: pathlib.Path, output_dir: pathlib.Path,
) -> list[str]:
    return [
        *last_block.build_train_command(
            args, checkpoint=checkpoint, rollout_dir=rollout_dir, output_dir=output_dir,
        ),
        "--dual-learning-rate", "0",
    ]


def _check_multiplier(checkpoint: pathlib.Path) -> None:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if float(metadata["cost_multiplier"]) != COST_MULTIPLIER:
        raise ValueError("Fixed-cost experiment requires checkpoint cost_multiplier=0.02.")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    input_checkpoint, collection_dir, collection_rows, collection_summary = last_block._reuse_first_round(args)
    _check_multiplier(input_checkpoint)
    baseline_rows, reference_summary = base._read_eval(
        args.reference_eval_dir, mode=base.ANCHOR_MODE, episodes=last_block.VALIDATION_EPISODES, seed=args.seed,
    )
    baseline = base.summarize(baseline_rows)
    protocol = {
        "experiment": "fixed_positive_cost_last_block_smdp_ppo", "maximum_updates": 1, "no_collection": True,
        "cost_multiplier": COST_MULTIPLIER, "dual_learning_rate": 0.0,
        "critic_warmup_epochs": 10, "ppo_epochs": 4,
        "frozen_vla": True, "new_transformer_layers": 0,
        "actor_scope": "A existing last Transformer block and ordered head; earlier encoder stays frozen",
        "source_round_dir": str(args.reuse_first_round_dir), "input_checkpoint": str(input_checkpoint),
        "collection_dir": str(collection_dir), "collection_reused": True,
        "collection": base.summarize(collection_rows), "collection_source_summary": collection_summary,
        "anchor_predictor_dir": str(args.anchor_predictor_dir), "code_dir": str(args.code_dir), "seed": args.seed,
        "server": {"host": args.host, "port": args.port},
        "reference_eval_dir": str(args.reference_eval_dir), "reference_source_summary": reference_summary,
        "original_reference_eval_dir": str(args.original_reference_eval_dir) if args.original_reference_eval_dir else None,
        "validation_episode_ids": list(last_block.VALIDATION_EPISODES),
        "final_episode_ids": list(last_block.FINAL_EPISODES), "maximum_validation_rpc_seconds": 3.0,
        "selection": "lexicographic (validation success_count, -mean_rpc_seconds), strictly above A",
        "all_results_development": True,
        "caveat": "This reuses A's original on-policy batch; validation/final presets have prior development use.",
    }
    base._write_json(args.output_dir / "run_config.json", protocol)
    training_dir = args.output_dir / "training"
    last_block._run_stage(args, args.output_dir, "train", build_train_command(
        args, checkpoint=input_checkpoint, rollout_dir=collection_dir, output_dir=training_dir,
    ), trainer=True)
    training = json.loads((training_dir / "summary.json").read_text(encoding="utf-8"))
    if training.get("status") != "complete" or not isinstance(training.get("actor_changed"), bool):
        raise ValueError("Trainer must complete and report actor_changed.")
    if any(float(training[name]) != COST_MULTIPLIER for name in ("cost_multiplier_before", "cost_multiplier_after")):
        raise ValueError("Trainer changed the fixed positive RPC cost multiplier.")
    checkpoint = pathlib.Path(training["selector_params"])
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(checkpoint / "params")
    _check_multiplier(checkpoint)
    actor_changed = training["actor_changed"]
    if actor_changed and int(training["accepted_epochs"]) == 0:
        raise ValueError("Trainer reports actor changes without an accepted PPO epoch.")
    validation_rows = baseline_rows
    validation_source = args.reference_eval_dir
    if actor_changed:
        validation_source = args.output_dir / "validation"
        last_block._run_stage(args, args.output_dir, "validate", last_block.build_eval_command(
            args, checkpoint=checkpoint, output_dir=validation_source,
            episodes=last_block.VALIDATION_EPISODES, sample=False,
        ), trainer=False)
        validation_rows, _ = base._read_eval(
            validation_source, mode=last_block.MODE, episodes=last_block.VALIDATION_EPISODES, seed=args.seed,
        )
    else:
        base._write_json(args.output_dir / "validation_reused.json", {
            "reason": "Actor unchanged; reuse A validation rather than measure the same actor again.",
            "source_eval_dir": str(args.reference_eval_dir), "source_actor_checkpoint": str(args.anchor_predictor_dir),
        })
    validation = base.summarize(validation_rows)
    eligible = actor_changed and base.eligible(validation, baseline)
    result = {
        "status": "complete", **protocol, "training_summary": training, "selector_checkpoint": str(checkpoint),
        "actor_changed": actor_changed, "baseline_validation": baseline, "validation": validation,
        "validation_source": str(validation_source), "validation_reused": not actor_changed,
        "validation_vs_A": base.compare(baseline_rows, validation_rows), "eligible": eligible,
        "final_run": False, "selected_system": "A", "selected_checkpoint": None,
    }
    if not eligible:
        result["conclusion"] = "The sole fixed-cost candidate did not exceed A under validation/RPC criteria; retain A."
        return result
    final_dir = args.output_dir / "final"
    last_block._run_stage(args, args.output_dir, "final_eval", last_block.build_eval_command(
        args, checkpoint=checkpoint, output_dir=final_dir, episodes=last_block.FINAL_EPISODES, sample=False,
    ), trainer=False)
    final_rows, _ = base._read_eval(final_dir, mode=last_block.MODE, episodes=last_block.FINAL_EPISODES, seed=args.seed)
    anchor_rows, _ = base._read_eval(
        args.reference_eval_dir, mode=base.ANCHOR_MODE, episodes=last_block.FINAL_EPISODES, seed=args.seed,
    )
    result.update(
        final_run=True, selected_system=last_block.MODE, selected_checkpoint=str(checkpoint),
        final=base.summarize(final_rows), final_vs_A=base.compare(anchor_rows, final_rows),
        conclusion="The sole fixed-cost candidate completed one development final evaluation; no additional updates.",
        legacy_A_pilot_caveat=(
            "Earlier A pilot reported 189/200; this comparison uses episode0–19 from the current A1000 run. "
            "Identical task/episode IDs do not imply byte-identical simulator trajectories across runs."
        ),
    )
    if args.original_reference_eval_dir is not None:
        original_rows, original_summary = base._read_eval(
            args.original_reference_eval_dir, mode="original", episodes=last_block.FINAL_EPISODES, seed=args.seed,
        )
        result["final_vs_historical_original"] = base.compare(original_rows, final_rows)
        result["historical_original_source_summary"] = original_summary
    return result


def main(args: argparse.Namespace) -> None:
    for name in (
        "output_dir", "code_dir", "python", "train_state_bank", "reference_eval_dir", "anchor_predictor_dir",
        "reuse_first_round_dir",
    ):
        setattr(args, name, getattr(args, name).absolute())
    if args.original_reference_eval_dir is not None:
        args.original_reference_eval_dir = args.original_reference_eval_dir.absolute()
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
