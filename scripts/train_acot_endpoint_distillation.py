"""Train one-step EAR/final endpoint students from compact HDF5 labels.

This trainer keeps the VLM, IAR, and reasoning-fusion modules frozen. Only the
selected coarse/final action-expert LLM branch plus its local
input/time/output projections is optimized. It writes a small delta sidecar,
not a full ACoT-VLA checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import json
import logging
import pathlib
import re
import time
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import endpoint_dataset
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import optimizer as optimizer_lib
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders

try:
    import train as train_lib
except ImportError:  # pragma: no cover - supports python -m scripts.train_acot_endpoint_distillation
    from scripts import train as train_lib


LOGGER = logging.getLogger("train_acot_endpoint_distillation")

_COARSE_PATH = re.compile(
    r"^(?:"
    r"PaliGemma/llm/.*_1(?:/.*)?|"
    r"coarse_action_in_proj/.*|"
    r"coarse_time_mlp_in/.*|"
    r"coarse_time_mlp_out/.*|"
    r"coarse_action_time_mlp_in/.*|"
    r"coarse_action_time_mlp_out/.*|"
    r"coarse_action_out_proj/.*"
    r")$"
)
_FINAL_PATH = re.compile(
    r"^(?:"
    r"PaliGemma/llm/.*_2(?:/.*)?|"
    r"action_in_proj/.*|"
    r"time_mlp_in/.*|"
    r"time_mlp_out/.*|"
    r"action_time_mlp_in/.*|"
    r"action_time_mlp_out/.*|"
    r"action_out_proj/.*"
    r")$"
)


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    resume_sidecar_params: str | None = None
    causal_audit_summary: str | None = None
    stage: str = "final"
    variant: str = "ir"
    seed: int = 7
    train_steps: int = 1_000
    batch_size: int = 8
    learning_rate: float = 3e-6
    decay_learning_rate: float = 3e-7
    warmup_steps: int = 50
    weight_decay: float = 1e-10
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    log_interval: int = 25
    checkpoint_interval: int = 250
    coarse_loss_weight: float = 1.0
    final_loss_weight: float = 1.0
    ir_loss_weight: float = 0.5
    use_student_coarse: bool = False
    fsdp_devices: int = 1
    overwrite: bool = False
    allow_failed_audit: bool = False


def _path_text(path: tuple[Any, ...]) -> str:
    return "/".join(map(str, path))


def _matches_stage(path: tuple[Any, ...], stage: str) -> bool:
    text = _path_text(path)
    return (
        (stage in {"coarse", "dual"} and _COARSE_PATH.fullmatch(text) is not None)
        or (stage in {"final", "dual"} and _FINAL_PATH.fullmatch(text) is not None)
    )


def _train_filter(stage: str) -> nnx.filterlib.Filter:
    filters: list[nnx.filterlib.Filter] = []
    if stage in {"coarse", "dual"}:
        filters.extend(
            [
                nnx_utils.PathRegex(r"PaliGemma/llm/.*_1(?:/.*)?"),
                nnx_utils.PathRegex(r"coarse_action_in_proj/.*"),
                nnx_utils.PathRegex(r"coarse_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"coarse_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"coarse_action_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"coarse_action_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"coarse_action_out_proj/.*"),
            ]
        )
    if stage in {"final", "dual"}:
        filters.extend(
            [
                nnx_utils.PathRegex(r"PaliGemma/llm/.*_2(?:/.*)?"),
                nnx_utils.PathRegex(r"action_in_proj/.*"),
                nnx_utils.PathRegex(r"time_mlp_in/.*"),
                nnx_utils.PathRegex(r"time_mlp_out/.*"),
                nnx_utils.PathRegex(r"action_time_mlp_in/.*"),
                nnx_utils.PathRegex(r"action_time_mlp_out/.*"),
                nnx_utils.PathRegex(r"action_out_proj/.*"),
            ]
        )
    if not filters:
        raise ValueError(f"stage must be coarse, final, or dual; got {stage!r}.")
    return nnx.Any(*filters)


def _validate_args(args: Args) -> None:
    if args.stage not in {"coarse", "final", "dual"}:
        raise ValueError("--stage must be coarse, final, or dual.")
    if args.variant not in {"b6", "ir"}:
        raise ValueError("--variant must be b6 or ir.")
    if args.variant == "ir" and args.stage == "coarse":
        raise ValueError("IR alignment acts on final actions; use --variant b6 for coarse-only training.")
    for name in (
        "train_steps",
        "batch_size",
        "warmup_steps",
        "log_interval",
        "checkpoint_interval",
        "fsdp_devices",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.learning_rate <= 0 or args.decay_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and peak learning rate must be positive.")
    if min(args.coarse_loss_weight, args.final_loss_weight, args.ir_loss_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.variant == "ir" and args.ir_loss_weight == 0:
        raise ValueError("IR variant requires a positive --ir-loss-weight.")


def _check_audit_gate(inputs: Sequence[str], args: Args) -> None:
    if args.variant != "ir" or args.allow_failed_audit:
        return
    summaries = []
    if args.causal_audit_summary is not None:
        candidate = pathlib.Path(args.causal_audit_summary)
        if not candidate.exists():
            raise FileNotFoundError(f"Causal audit summary not found: {candidate}")
        summaries.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
    else:
        for item in inputs:
            path = pathlib.Path(item)
            candidate = path / "audit_summary.json" if path.is_dir() else path.parent / "audit_summary.json"
            if candidate.exists():
                summaries.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
    failed = [str(path) for path, summary in summaries if not summary.get("ear_causal_audit_pass", False)]
    if failed:
        raise ValueError(
            "EAR causal audit failed for the requested IR dataset. "
            f"Refusing IR training for {failed}; use --allow-failed-audit only for a diagnostic run."
        )
    if not summaries:
        raise ValueError(
            "IR training requires a causal audit summary with ear_causal_audit_pass=true. "
            "Pass --causal-audit-summary explicitly, or use --allow-failed-audit only for a diagnostic run."
        )


def _split_indices(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
    require_semantic_intervention: bool,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.ones((len(arrays["dataset_index"]),), dtype=np.bool_)
    if require_semantic_intervention:
        semantic = arrays["intervention_valid"].copy()
        semantic &= arrays["intervention_ids"] != endpoint_dataset.INTERVENTION_IDS["null"]
        eligible &= np.any(semantic, axis=-1)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 2:
        raise ValueError("Endpoint dataset has fewer than two eligible records.")

    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups[eligible_indices])
    rng = np.random.default_rng(seed)
    if unique_groups.size >= 2:
        rng.shuffle(unique_groups)
        validation_count = max(1, round(unique_groups.size * validation_fraction))
        validation_groups = unique_groups[:validation_count]
        validation_mask = eligible & np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(eligible & ~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        shuffled = eligible_indices.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(shuffled.size * validation_fraction))
        validation_count = min(validation_count, shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
        LOGGER.warning(
            "Episode metadata did not identify multiple groups; using a record-level train/validation split."
        )
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Train/validation split produced an empty partition.")
    return train_indices, validation_indices


def _load_norm_stats(
    train_config: config_lib.TrainConfig,
    data_config: config_lib.DataConfig,
    checkpoint_dir: pathlib.Path,
) -> dict[str, Any]:
    if data_config.norm_stats is not None:
        return data_config.norm_stats
    if data_config.asset_id is None:
        raise ValueError("The data config needs asset_id to load checkpoint normalization stats.")
    return checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)


def _with_norm_stats(
    data_config: config_lib.DataConfig,
    norm_stats: dict[str, Any],
) -> config_lib.DataConfig:
    """Replace norm stats without dropping factory-attached runtime fields."""

    updated = dataclasses.replace(data_config, norm_stats=norm_stats)
    declared_fields = {field.name for field in dataclasses.fields(data_config)}
    for name, value in vars(data_config).items():
        if name not in declared_fields:
            object.__setattr__(updated, name, value)
    return updated


def _load_resume_params(path: str | None) -> tuple[dict[str, Any] | None, set[tuple[Any, ...]]]:
    if path is None:
        return None, set()
    resolved = download.maybe_download(path)
    params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(resolved, restore_type=np.ndarray)
    )
    flat = traverse_util.flatten_dict(params)
    disallowed = [
        _path_text(key)
        for key in flat
        if _COARSE_PATH.fullmatch(_path_text(key)) is None
        and _FINAL_PATH.fullmatch(_path_text(key)) is None
    ]
    if disallowed:
        raise ValueError(f"Resume sidecar contains disallowed parameters: {disallowed[:5]}")
    return params, set(flat)


@dataclasses.dataclass(frozen=True)
class _BaseAndSidecarLoader:
    base_params_path: str
    sidecar_params: dict[str, Any] | None

    def load(self, params: Any) -> Any:
        loaded = weight_loaders.CheckpointWeightLoader(self.base_params_path).load(params)
        if self.sidecar_params is not None:
            loaded = policy_config.merge_acot_endpoint_student_params(loaded, self.sidecar_params)
        return loaded


def _choose_interventions(
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    coarse = []
    actions = []
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    for row_index in row_indices:
        ids = arrays["intervention_ids"][row_index]
        valid = arrays["intervention_valid"][row_index] & (ids != null_id)
        candidates = np.flatnonzero(valid)
        if not candidates.size:
            candidates = np.flatnonzero(arrays["intervention_valid"][row_index])
        if not candidates.size:
            # Coarse-only clean-label exports intentionally contain no
            # interventions. The coarse stage never consumes these tensors,
            # so clean targets are a shape-correct neutral fallback. Final/dual
            # stages are filtered to records with a valid semantic intervention.
            coarse.append(arrays["clean_coarse"][row_index])
            actions.append(arrays["clean_actions"][row_index])
            continue
        selected = int(candidates[0] if deterministic else rng.choice(candidates))
        coarse.append(arrays["intervention_coarse"][row_index, selected])
        actions.append(arrays["intervention_actions"][row_index, selected])
    return np.asarray(coarse, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _make_batch(
    observation_dataset: data_loader.Dataset,
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic_intervention: bool,
) -> dict[str, Any]:
    items = [observation_dataset[int(arrays["dataset_index"][index])] for index in row_indices]
    if any(item is None for item in items):
        raise ValueError("Observation dataset returned None for an endpoint record.")
    collated = data_loader._collate_fn(items)  # noqa: SLF001 - reuse the canonical OpenPI collation.
    collated = jax.tree.map(jax.numpy.asarray, collated)
    observation = model_lib.Observation.from_dict(collated)
    intervention_coarse, intervention_actions = _choose_interventions(
        arrays,
        row_indices,
        rng,
        deterministic=deterministic_intervention,
    )
    return {
        "observation": observation,
        "teacher_coarse": np.asarray(arrays["clean_coarse"][row_indices], dtype=np.float32),
        "teacher_actions": np.asarray(arrays["clean_actions"][row_indices], dtype=np.float32),
        "coarse_noise": np.asarray(arrays["coarse_noise"][row_indices], dtype=np.float32),
        "action_noise": np.asarray(arrays["action_noise"][row_indices], dtype=np.float32),
        "intervention_coarse": intervention_coarse,
        "teacher_intervention_actions": intervention_actions,
    }


def _endpoint_train_step(
    state: training_utils.TrainState,
    batch: dict[str, Any],
    *,
    trainable_filter: nnx.filterlib.Filter,
    stage: str,
    use_student_coarse: bool,
    coarse_loss_weight: float,
    final_loss_weight: float,
    ir_loss_weight: float,
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        return candidate.compute_endpoint_distillation_loss(
            batch["observation"],
            batch["teacher_coarse"],
            batch["teacher_actions"],
            batch["coarse_noise"],
            batch["action_noise"],
            batch["intervention_coarse"],
            batch["teacher_intervention_actions"],
            stage=stage,
            use_student_coarse=use_student_coarse,
            coarse_loss_weight=coarse_loss_weight,
            final_loss_weight=final_loss_weight,
            ir_loss_weight=ir_loss_weight,
            compute_ir_metrics=False,
        )

    diff_state = nnx.DiffState(0, trainable_filter)
    (_, metrics), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=diff_state,
        has_aux=True,
    )(model)
    params = state.params.filter(trainable_filter)
    updates, optimizer_state = state.tx.update(gradients, state.opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    nnx.update(model, updated_params)
    next_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=optimizer_state,
    )
    return next_state, {**metrics, "gradient_norm": optax.global_norm(gradients)}


def _endpoint_validation_step(
    state: training_utils.TrainState,
    batch: dict[str, Any],
    *,
    stage: str,
    use_student_coarse: bool,
    coarse_loss_weight: float,
    final_loss_weight: float,
    ir_loss_weight: float,
) -> dict[str, jax.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    _, metrics = model.compute_endpoint_distillation_loss(
        batch["observation"],
        batch["teacher_coarse"],
        batch["teacher_actions"],
        batch["coarse_noise"],
        batch["action_noise"],
        batch["intervention_coarse"],
        batch["teacher_intervention_actions"],
        stage=stage,
        use_student_coarse=use_student_coarse,
        coarse_loss_weight=coarse_loss_weight,
        final_loss_weight=final_loss_weight,
        ir_loss_weight=ir_loss_weight,
        compute_ir_metrics=stage in {"final", "dual"},
    )
    return metrics


def _save_sidecar(
    state: training_utils.TrainState,
    target: pathlib.Path,
    *,
    stage: str,
    resume_paths: set[tuple[Any, ...]],
    overwrite: bool,
) -> int:
    flat = traverse_util.flatten_dict(state.params.to_pure_dict())
    selected = {
        path: (
            value.astype(jax.numpy.bfloat16)
            if jax.numpy.issubdtype(value.dtype, jax.numpy.floating)
            else value
        )
        for path, value in flat.items()
        if path in resume_paths or _matches_stage(path, stage)
    }
    if not selected:
        raise ValueError("No endpoint-student parameters matched the save filter.")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Sidecar target already exists: {target}")
    item = {"params": traverse_util.unflatten_dict(selected)}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=overwrite)
    return int(sum(np.prod(value.shape) for value in selected.values()))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    _check_audit_gate(args.dataset, args)
    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    final_params_path = output_dir / "final" / "params"
    if not args.overwrite:
        if final_params_path.exists():
            raise FileExistsError(f"Final sidecar already exists: {final_params_path}")
        if metrics_path.exists() and metrics_path.stat().st_size:
            raise FileExistsError(
                f"Metrics already exist in {output_dir}; choose a new output directory or pass --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {args.batch_size} must be divisible by device count {jax.device_count()}."
        )

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(
        arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        require_semantic_intervention=args.stage in {"final", "dual"},
    )
    train_config_base = config_lib.get_config(args.config_name)
    model_config = train_config_base.model
    expected_shapes = {
        "clean_coarse": (model_config.coarse_action_horizon, model_config.action_dim),
        "clean_actions": (model_config.action_horizon, model_config.action_dim),
    }
    for field, expected in expected_shapes.items():
        if arrays[field].shape[1:] != expected:
            raise ValueError(f"{field} shape {arrays[field].shape[1:]} does not match model {expected}.")

    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    base_params_path = checkpoint_dir / "params"
    if not base_params_path.exists():
        raise FileNotFoundError(f"Base checkpoint params not found: {base_params_path}")
    data_config = train_config_base.data.create(train_config_base.assets_dirs, model_config)
    norm_stats = _load_norm_stats(train_config_base, data_config, checkpoint_dir)
    data_config = _with_norm_stats(data_config, norm_stats)
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    observation_dataset = data_loader.transform_dataset(raw_dataset, data_config)

    resume_params, resume_paths = _load_resume_params(args.resume_sidecar_params)
    trainable_filter = _train_filter(args.stage)
    ir_weight = args.ir_loss_weight if args.variant == "ir" else 0.0
    train_config = dataclasses.replace(
        train_config_base,
        weight_loader=_BaseAndSidecarLoader(str(base_params_path), resume_params),
        freeze_filter=nnx.Not(trainable_filter),
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.learning_rate,
            decay_steps=max(args.train_steps, args.warmup_steps + 1),
            decay_lr=args.decay_learning_rate,
        ),
        optimizer=optimizer_lib.AdamW(
            weight_decay=args.weight_decay,
            clip_gradient_norm=args.gradient_clip_norm,
        ),
        ema_decay=None,
        batch_size=args.batch_size,
        num_train_steps=args.train_steps,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
    )

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = train_lib.init_train_state(
        train_config,
        jax.random.key(args.seed),
        mesh,
        resume=False,
    )
    jax.block_until_ready(state)
    trainable_params = state.params.filter(trainable_filter)
    LOGGER.info(
        "Initialized endpoint student: stage=%s variant=%s train=%s validation=%s trainable_params=%s",
        args.stage,
        args.variant,
        train_indices.size,
        validation_indices.size,
        training_utils.count_parameters(trainable_params),
    )

    train_step = jax.jit(
        functools.partial(
            _endpoint_train_step,
            trainable_filter=trainable_filter,
            stage=args.stage,
            use_student_coarse=args.use_student_coarse,
            coarse_loss_weight=args.coarse_loss_weight,
            final_loss_weight=args.final_loss_weight,
            ir_loss_weight=ir_weight,
        ),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )
    validation_step = jax.jit(
        functools.partial(
            _endpoint_validation_step,
            stage=args.stage,
            use_student_coarse=args.use_student_coarse,
            coarse_loss_weight=args.coarse_loss_weight,
            final_loss_weight=args.final_loss_weight,
            ir_loss_weight=ir_weight,
        ),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    last_train_metrics: dict[str, float] = {}
    last_validation_metrics: dict[str, float] = {}

    metrics_mode = "w" if args.overwrite else "a"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            batch = _make_batch(
                observation_dataset,
                arrays,
                sampled,
                rng,
                deterministic_intervention=False,
            )
            batch = jax.device_put(batch, data_sharding)
            with sharding.set_mesh(mesh):
                state, train_metrics = train_step(state, batch)

            should_log = step == 1 or step % args.log_interval == 0 or step == args.train_steps
            if should_log:
                validation_sample = rng.choice(
                    validation_indices,
                    size=args.batch_size,
                    replace=validation_indices.size < args.batch_size,
                )
                validation_batch = _make_batch(
                    observation_dataset,
                    arrays,
                    validation_sample,
                    rng,
                    deterministic_intervention=True,
                )
                validation_batch = jax.device_put(validation_batch, data_sharding)
                with sharding.set_mesh(mesh):
                    validation_metrics = validation_step(state, validation_batch)
                last_train_metrics = {
                    f"train/{name}": float(value)
                    for name, value in jax.device_get(train_metrics).items()
                }
                last_validation_metrics = {
                    f"validation/{name}": float(value)
                    for name, value in jax.device_get(validation_metrics).items()
                }
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **last_train_metrics,
                    **last_validation_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "step=%s %s",
                    step,
                    " ".join(
                        f"{name}={value:.6f}"
                        for name, value in record.items()
                        if name.startswith(("train/", "validation/"))
                    ),
                )

            if step % args.checkpoint_interval == 0 and step != args.train_steps:
                saved_params = _save_sidecar(
                    state,
                    output_dir / f"step_{step:06d}" / "params",
                    stage=args.stage,
                    resume_paths=resume_paths,
                    overwrite=args.overwrite,
                )
                LOGGER.info("Saved step %s delta sidecar with %s parameters.", step, saved_params)

    saved_params = _save_sidecar(
        state,
        final_params_path,
        stage=args.stage,
        resume_paths=resume_paths,
        overwrite=args.overwrite,
    )
    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": list(args.dataset),
        "stage": args.stage,
        "variant": args.variant,
        "resume_sidecar_params": args.resume_sidecar_params,
        "causal_audit_summary": args.causal_audit_summary,
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "completed_steps": args.train_steps,
        "saved_parameter_count": saved_params,
        "final_params_path": str(final_params_path.resolve()),
        "last_train_metrics": last_train_metrics,
        "last_validation_metrics": last_validation_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "frozen_contract": "VLM, IAR, and reasoning fusion frozen; only selected EAR/final local branches train.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Training complete; final delta sidecar: %s", final_params_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
