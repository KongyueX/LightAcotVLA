"""Train one zero-initialized architecture adapter on fixed paired A labels."""
# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
from fractions import Fraction
import json
import pathlib
import time
from typing import Any, Literal

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import train_execution_horizon_feedback as feedback_train
import tyro

from openpi.models import execution_horizon_predictor as predictor_lib
from openpi.models import model as model_lib

VARIANT_ROOTS = {
    "visual_query": ("visual_query_conditioner",),
    "expert_hidden": ("expert_feature_in", "expert_feature_out"),
}
INPUT_NAMES = (
    "prefix_feature", "state", "coarse_actions", "final_actions", "previous_actions", "previous_h",
    "budget_balance", "episode_progress", "previous_valid", "prefix_tokens", "prefix_mask", "expert_hidden",
)


@dataclasses.dataclass(frozen=True)
class Args:
    train_dir: str
    validation_dir: str
    a_predictor_dir: str
    output_dir: str
    variant: Literal["visual_query", "expert_hidden"]
    seed: int = 7
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_updates: int = 650
    log_every: int = 25


def initialize_model(anchor_dir: str, variant: str, seed: int = 7):
    if variant not in VARIANT_ROOTS:
        raise ValueError(f"Unknown architecture variant: {variant}.")
    anchor = pathlib.Path(anchor_dir).resolve()
    config = predictor_lib.ExecutionHorizonPredictorConfig(**json.loads((anchor / "predictor_config.json").read_text()))
    if config.ordered_readout != "global" or config.temporal_layers != 2 or config.visual_num_queries != 4:
        raise ValueError("Architecture comparison requires the original two-layer, four-slot global A.")
    config = dataclasses.replace(
        config, visual_query_conditioning=variant == "visual_query",
        expert_feature_dim=1024 if variant == "expert_hidden" else 0, expert_feature_projection_dim=64,
    )
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(seed))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(anchor / "params", dtype=jnp.float32))
    loaded = loaded.get("execution_horizon_predictor", loaded)
    state = nnx.state(module)
    expected, actual = traverse_util.flatten_dict(state.to_pure_dict()), traverse_util.flatten_dict(loaded)
    roots = VARIANT_ROOTS[variant]
    shared = {path for path in expected if path[0] not in roots}
    if shared != set(actual) or any(np.shape(expected[path]) != np.shape(actual[path]) for path in shared):
        raise ValueError("The saved A tree must match every shared architecture parameter exactly.")
    merged = {path: actual[path] if path in shared else value for path, value in expected.items()}
    state.replace_by_pure_dict(traverse_util.unflatten_dict(merged))
    module = nnx.merge(nnx.graphdef(module), state)
    graphdef, trainable, frozen = nnx.split(module, nnx.All(nnx.Param, lambda path, value: path[0] in roots), ...)
    return config, graphdef, trainable, frozen


def read_roots(directory: str) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    paths = sorted(pathlib.Path(directory).resolve().glob("task*_ep*.npz"))
    if not paths:
        raise ValueError(f"No architecture roots in {directory}.")
    records, identities = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            record = {name: archive[name] for name in archive.files}
        if int(record.get("architecture_cache_schema", 0)) != 1:
            raise ValueError(f"Missing architecture cache in {path}.")
        for name in INPUT_NAMES:
            if "input_" + name not in record:
                raise ValueError(f"Missing {name} in {path}.")
        record.update(feedback_train.paired_advantages(record))
        records.append(record)
        identities.append([int(record[name]) for name in ("task_id", "episode_id", "root_step")])
    if len({tuple(row[:2]) for row in identities}) != len(identities):
        raise ValueError("Each task/episode must contribute exactly one architecture root.")
    return records, {
        "roots": len(records), "identities": identities, "source_dir": str(pathlib.Path(directory).resolve())
    }


def batch_records(records: list[dict[str, np.ndarray]], prefix_length: int) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {name: [] for name in INPUT_NAMES}
    labels = ("advantage", "success_delta", "rpc_delta_seconds", "paired_count", "continuation_logits")
    columns.update({name: [] for name in labels})
    for record in records:
        for name in INPUT_NAMES:
            value = np.asarray(record["input_" + name])
            if name == "prefix_tokens":
                value = np.pad(value, ((0, prefix_length - len(value)), (0, 0)))
            elif name == "prefix_mask":
                value = np.pad(value, (0, prefix_length - len(value)))
            columns[name].append(value)
        for name in labels:
            columns[name].append(record[name])
    return {name: np.stack(values) for name, values in columns.items()}


def apply_model(graphdef: Any, params: nnx.State, frozen: nnx.State, batch: dict[str, jax.Array], variant: str):
    inputs = {name: batch[name] for name in INPUT_NAMES if name != "expert_hidden" or variant == "expert_hidden"}
    return nnx.merge(graphdef, params, frozen)(**inputs)


def objective(prediction: dict[str, jax.Array], batch: dict[str, jax.Array]):
    log_probabilities = prediction["ordered_horizon_log_probability"]
    probabilities = prediction["ordered_horizon_probability"]
    anchor_log, anchor_probability = predictor_lib.ordered_continuation_distribution(batch["continuation_logits"])
    advantage = jnp.mean(jnp.sum(probabilities * batch["advantage"], axis=-1))
    kl = jnp.mean(jnp.sum(anchor_probability * (anchor_log - log_probabilities), axis=-1))
    loss = -advantage + feedback_train.ANCHOR_KL_WEIGHT * kl
    return loss, {"loss": loss, "expected_paired_advantage": advantage, "anchor_kl": kl}


def greedy_metrics(probabilities: np.ndarray, data: dict[str, np.ndarray]):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5 or not len(probabilities):
        raise ValueError("Greedy selection requires a non-empty (roots, 5) probability matrix.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite architecture probabilities cannot participate in checkpoint selection.")
    index = np.argmax(probabilities, axis=-1)
    row = np.arange(len(index))
    counts = data["paired_count"][row, index].astype(np.int64)
    net = np.rint(data["success_delta"][row, index] * counts).astype(np.int64)
    fraction = sum((Fraction(int(n), int(d)) for n, d in zip(net, counts, strict=True)), Fraction(0)) / len(row)
    rpc = float(np.mean(data["rpc_delta_seconds"][row, index]))
    return (fraction, -rpc), {
        "roots": len(row), "success_delta": float(fraction),
        "success_delta_fraction": {"numerator": fraction.numerator, "denominator": fraction.denominator},
        "rpc_delta_seconds": rpc, "paired_net_success_count": int(net.sum()),
        "selected_h": [int((5, 10, 15, 20, 25)[i]) for i in index],
    }


def save_sidecar(target: pathlib.Path, config, graphdef, params, frozen):
    target.mkdir(parents=True, exist_ok=True)
    (target / "predictor_config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    state = nnx.state(nnx.merge(graphdef, params, frozen))
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target / "params", {"params": {"execution_horizon_predictor": state.to_pure_dict()}})


def main(args: Args) -> None:
    if args.batch_size <= 0 or args.max_updates <= 0 or args.log_every <= 0:
        raise ValueError("batch_size, max_updates and log_every must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    output = pathlib.Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Architecture training output must be empty.")
    train_records, train_summary = read_roots(args.train_dir)
    val_records, val_summary = read_roots(args.validation_dir)
    if {tuple(row[:2]) for row in train_summary["identities"]} & {tuple(row[:2]) for row in val_summary["identities"]}:
        raise ValueError("Training and early-stop episodes overlap.")
    length = max(len(record["input_prefix_mask"]) for record in train_records + val_records)
    train_np, val_np = batch_records(train_records, length), batch_records(val_records, length)
    del train_records, val_records
    config, graphdef, params, frozen = initialize_model(args.a_predictor_dir, args.variant, args.seed)
    initial_params = jax.tree.map(lambda value: np.asarray(value).copy(), params)
    frozen_before = jax.tree.map(lambda value: np.asarray(value).copy(), frozen)
    train_data = jax.tree.map(jnp.asarray, train_np)
    val_data = jax.tree.map(jnp.asarray, val_np)
    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(params)

    def loss(p, batch):
        return objective(apply_model(graphdef, p, frozen, batch, args.variant), batch)

    @jax.jit
    def update(p, optimizer_state, batch):
        (value, _), gradients = jax.value_and_grad(loss, has_aux=True)(p, batch)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, p)
        return optax.apply_updates(p, updates), optimizer_state, value

    infer = jax.jit(lambda p, batch: apply_model(graphdef, p, frozen, batch, args.variant))
    evaluate = jax.jit(lambda p, batch: loss(p, batch)[1])
    initial_prediction = jax.device_get(infer(params, val_data))
    if not np.allclose(
        initial_prediction["ordered_continuation_logits"], val_np["continuation_logits"], rtol=0, atol=1e-4
    ):
        raise ValueError("Zero-initialized adapter does not reproduce cached A logits.")
    anchor_h = np.asarray((5, 10, 15, 20, 25))[np.argmax(np.asarray(predictor_lib.ordered_continuation_distribution(
        jnp.asarray(val_np["continuation_logits"])
    )[1]), axis=-1)]
    if not np.array_equal(initial_prediction["ordered_selected_h"], anchor_h):
        raise ValueError("Zero-initialized adapter changes A's validation argmax.")
    best_score, initial_greedy = greedy_metrics(initial_prediction["ordered_horizon_probability"], val_np)
    best_greedy, best_step = initial_greedy, 0
    best_params = jax.tree.map(lambda value: np.asarray(value).copy(), params)
    output.mkdir(parents=True, exist_ok=True)
    rng, started = np.random.default_rng(args.seed), time.monotonic()
    with (output / "training_log.jsonl").open("w") as log:
        def record(step, value):
            row = {"step": step, "validation": value, "best_step": best_step}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        record(0, {"greedy": initial_greedy})
        for step in range(1, args.max_updates + 1):
            indices = rng.choice(
                len(train_np["state"]), size=min(args.batch_size, len(train_np["state"])), replace=False
            )
            batch = {name: value[indices] for name, value in train_data.items()}
            params, optimizer_state, value = update(params, optimizer_state, batch)
            if not np.isfinite(float(value)):
                raise ValueError("Non-finite architecture training loss.")
            if step % args.log_every and step != args.max_updates:
                continue
            prediction = jax.device_get(infer(params, val_data))
            score, greedy = greedy_metrics(prediction["ordered_horizon_probability"], val_np)
            metrics = {name: float(value) for name, value in jax.device_get(evaluate(params, val_data)).items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("Non-finite architecture validation metrics.")
            if score > best_score:
                best_score, best_step, best_greedy = score, step, greedy
                best_params = jax.tree.map(lambda value: np.asarray(value).copy(), params)
            record(step, {**metrics, "greedy": greedy})
    unchanged = all(np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(frozen_before), jax.tree.leaves(frozen), strict=True
    ))
    if not unchanged:
        raise ValueError("Frozen A weights changed.")
    save_sidecar(output, config, graphdef, best_params, frozen)
    save_sidecar(output / "last", config, graphdef, params, frozen)
    summary = {
        "status": "complete", "variant": args.variant, "args": dataclasses.asdict(args),
        "best_step": best_step, "updates": args.max_updates, "selection_metric": "greedy_success_then_rpc",
        "initial_greedy_validation": initial_greedy, "best_greedy_validation": best_greedy,
        "step0_included": True, "frozen_A_unchanged": unchanged,
        "trainable_parameters": sum(value.size for value in jax.tree.leaves(params)),
        "selected_adapter_changed": any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(initial_params), jax.tree.leaves(best_params), strict=True,
        )),
        "train": train_summary, "validation": val_summary, "training_seconds": time.monotonic() - started,
        "device": str(jax.devices()[0]), "rpc_weight": feedback_train.RPC_WEIGHT,
        "anchor_kl_weight": feedback_train.ANCHOR_KL_WEIGHT,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
