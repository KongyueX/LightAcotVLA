"""Export exact same-context IR NFE1/NFE2 pairs for the HARP pilot.

The existing endpoint HDF5 labels do not contain a deployed final-IR NFE2
target.  This GPU-only exporter therefore reloads the frozen base checkpoint
plus the exact endpoint-student sidecar and generates both endpoints directly:

* one canonical transformed observation and VLM prefix;
* one IAR and one student EAR generated from the stored coarse-flow noise;
* one stored final-flow noise shared by both final calls;
* deployed direct NFE1 ``A1`` and same-model NFE2 ``A2``.

The pair file is standalone and contains all causal inputs needed by the tiny
HARP residual trainer.  No teacher action, future frame, or outcome label is
used as a model input.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import endpoint_dataset
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader

try:
    import train_acot_endpoint_distillation as endpoint_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_acot_endpoint_distillation as endpoint_trainer


LOGGER = logging.getLogger("export_harp_ir_pairs")
PAIR_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    batch_size: int = 8
    seed: int = 7
    overwrite: bool = False


def _gpu() -> jax.Device:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("HARP pair export is GPU-only; no JAX GPU device was found.")
    LOGGER.info("Using JAX device %s", devices[0])
    return devices[0]


def _prepare_output(args: Args) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = pathlib.Path(args.output_dir).resolve()
    target = output_dir / "harp_ir_pairs.h5"
    temporary = target.with_suffix(".h5.tmp")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        target.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return target, temporary


def _load_model_and_data(args: Args) -> tuple[Any, data_loader.Dataset, dict[str, np.ndarray], pathlib.Path]:
    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_config = config_lib.get_config(args.config_name)
    model_config = train_config.model
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    base_params_path = checkpoint_dir / "params"
    if not base_params_path.exists():
        raise FileNotFoundError(f"Base checkpoint params not found: {base_params_path}")
    base_params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(base_params_path, dtype=jnp.bfloat16)
    )
    sidecar_path = pathlib.Path(download.maybe_download(args.endpoint_student_params))
    sidecar_params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(sidecar_path, dtype=jnp.bfloat16)
    )
    merged_params = policy_config.merge_acot_endpoint_student_params(base_params, sidecar_params)
    model = model_config.load(merged_params)
    model.eval()

    if arrays["coarse_noise"].shape[1:] != (
        model_config.coarse_action_horizon,
        model_config.action_dim,
    ):
        raise ValueError(
            f"coarse_noise {arrays['coarse_noise'].shape[1:]} does not match the model."
        )
    if arrays["action_noise"].shape[1:] != (
        model_config.action_horizon,
        model_config.action_dim,
    ):
        raise ValueError(
            f"action_noise {arrays['action_noise'].shape[1:]} does not match the model."
        )

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = endpoint_trainer._load_norm_stats(  # noqa: SLF001
        train_config, data_config, checkpoint_dir
    )
    data_config = endpoint_trainer._with_norm_stats(data_config, norm_stats)  # noqa: SLF001
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    observation_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    return model, observation_dataset, arrays, sidecar_path


def _create_datasets(
    handle: h5py.File,
    arrays: dict[str, np.ndarray],
    *,
    state_shape: tuple[int, ...],
    ear_shape: tuple[int, ...],
    iar_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    chunk_size: int,
) -> None:
    count = len(arrays["dataset_index"])
    for name, dtype in (
        ("dataset_index", np.uint32),
        ("task_id", np.int16),
        ("episode_id", np.int32),
        ("frame_id", np.int32),
        ("policy_seed", np.uint32),
    ):
        handle.create_dataset(
            name,
            data=np.asarray(arrays[name], dtype=dtype),
            compression="lzf",
            shuffle=True,
        )
    specifications = {
        "state": (state_shape, np.float16),
        "ear": (ear_shape, np.float16),
        "iar": (iar_shape, np.float16),
        "action_noise": (action_shape, np.float16),
        "action_nfe1": (action_shape, np.float16),
        "action_nfe2": (action_shape, np.float16),
    }
    for name, (shape, dtype) in specifications.items():
        handle.create_dataset(
            name,
            shape=(count, *shape),
            dtype=dtype,
            chunks=(min(chunk_size, count), *shape),
            compression="lzf",
            shuffle=True,
        )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    device = _gpu()
    target, temporary = _prepare_output(args)
    model, observation_dataset, arrays, sidecar_path = _load_model_and_data(args)
    count = len(arrays["dataset_index"])
    if count < 2:
        raise ValueError("HARP pair export requires at least two records.")

    prefix_fn = nnx_utils.module_jit(model.sample_actions_profile_prefix)
    implicit_fn = nnx_utils.module_jit(model.sample_actions_profile_implicit)
    coarse_fn = nnx_utils.module_jit(model.sample_actions_profile_coarse)
    pair_fn = nnx_utils.module_jit(model.sample_actions_profile_harp_pair)

    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    squared_error_sum = 0.0
    squared_error_count = 0
    gripper_squared_error_sum = 0.0
    gripper_squared_error_count = 0
    handle: h5py.File | None = None
    try:
        handle = h5py.File(temporary, "w")
        handle.attrs["schema_version"] = PAIR_SCHEMA_VERSION
        handle.attrs["contract"] = (
            "same transformed observation, prefix, NFE1 EAR, IAR, coarse/action noise; "
            "deployed direct IR NFE1 versus same-model IR NFE2"
        )
        handle.attrs["endpoint_student_params"] = str(sidecar_path)
        handle.attrs["checkpoint_dir"] = str(pathlib.Path(args.checkpoint_dir))
        handle.attrs["source_dataset_json"] = json.dumps(list(args.dataset))
        handle.attrs["config_name"] = args.config_name
        handle.attrs["seed"] = args.seed

        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            selected = np.arange(start, stop, dtype=np.int64)
            real_count = selected.size
            if real_count < args.batch_size:
                selected = np.pad(selected, (0, args.batch_size - real_count), mode="edge")
            batch = endpoint_trainer._make_batch(  # noqa: SLF001
                observation_dataset,
                arrays,
                selected,
                rng,
                deterministic_intervention=True,
            )
            observation = jax.device_put(batch["observation"], device)
            coarse_noise = jax.device_put(
                jnp.asarray(batch["coarse_noise"], dtype=jnp.float32), device
            )
            action_noise = jax.device_put(
                jnp.asarray(batch["action_noise"], dtype=jnp.float32), device
            )
            prefix_state = prefix_fn(jax.random.fold_in(jax.random.key(args.seed), start), observation)
            # Stored endpoint noise is the source of truth.  Overriding both
            # generated fields leaves observation/prefix/cache identical.
            prefix_state = {
                **prefix_state,
                "ref_action_noise": coarse_noise,
                "expert_action_noise": action_noise,
            }
            implicit = implicit_fn(prefix_state)["implicit_action_reason"]
            if implicit is None:
                raise ValueError("The selected ACoT config does not expose IAR tokens.")
            coarse = coarse_fn(
                prefix_state,
                num_steps=1,
                action_cot_denoising_steps=1,
                dynamic_denoising_steps=False,
            )["explicit_action_reason"]
            if coarse is None:
                raise ValueError("The selected ACoT config does not expose an explicit EAR.")
            pair = pair_fn(prefix_state, coarse, implicit)
            action_nfe1 = pair["action_nfe1"]
            action_nfe2 = pair["action_nfe2"]
            state = jnp.asarray(prefix_state["observation"].state, dtype=jnp.float32)
            state, coarse, implicit, action_noise_device, action_nfe1, action_nfe2 = jax.device_get(
                (state, coarse, implicit, action_noise, action_nfe1, action_nfe2)
            )

            if "state" not in handle:
                _create_datasets(
                    handle,
                    arrays,
                    state_shape=tuple(state.shape[1:]),
                    ear_shape=tuple(coarse.shape[1:]),
                    iar_shape=tuple(implicit.shape[1:]),
                    action_shape=tuple(action_nfe1.shape[1:]),
                    chunk_size=args.batch_size,
                )
            destination = slice(start, stop)
            handle["state"][destination] = np.asarray(state[:real_count], dtype=np.float16)
            handle["ear"][destination] = np.asarray(coarse[:real_count], dtype=np.float16)
            handle["iar"][destination] = np.asarray(implicit[:real_count], dtype=np.float16)
            handle["action_noise"][destination] = np.asarray(
                action_noise_device[:real_count], dtype=np.float16
            )
            handle["action_nfe1"][destination] = np.asarray(
                action_nfe1[:real_count], dtype=np.float16
            )
            handle["action_nfe2"][destination] = np.asarray(
                action_nfe2[:real_count], dtype=np.float16
            )

            continuous_delta = np.asarray(
                action_nfe2[:real_count, :, :6] - action_nfe1[:real_count, :, :6],
                dtype=np.float64,
            )
            squared_error_sum += float(np.sum(np.square(continuous_delta)))
            squared_error_count += continuous_delta.size
            if action_nfe1.shape[-1] > 6:
                gripper_delta = np.asarray(
                    action_nfe2[:real_count, :, 6] - action_nfe1[:real_count, :, 6],
                    dtype=np.float64,
                )
                gripper_squared_error_sum += float(np.sum(np.square(gripper_delta)))
                gripper_squared_error_count += gripper_delta.size
            handle.flush()
            LOGGER.info("Exported exact HARP pairs for %s/%s records", stop, count)

        elapsed = time.monotonic() - started
        continuous_mse = squared_error_sum / max(squared_error_count, 1)
        gripper_mse = gripper_squared_error_sum / max(gripper_squared_error_count, 1)
        handle.attrs["continuous_nfe1_nfe2_mse"] = continuous_mse
        handle.attrs["gripper_nfe1_nfe2_mse"] = gripper_mse
        handle.attrs["elapsed_seconds"] = elapsed
        handle.flush()
        handle.close()
        handle = None
        temporary.replace(target)
    finally:
        if handle is not None:
            handle.close()

    summary = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pair_path": str(target),
        "records": count,
        "endpoint_student_params": str(sidecar_path),
        "continuous_nfe1_nfe2_mse": continuous_mse,
        "gripper_nfe1_nfe2_mse": gripper_mse,
        "elapsed_seconds": elapsed,
        "contract": (
            "exact deployed IR NFE1 and same-model NFE2 share observation, prefix, "
            "student EAR NFE1, IAR, and stored flow noises"
        ),
        "gripper_policy": "exported for audit; HARP serving never modifies gripper",
    }
    (target.parent / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Exact HARP pair export complete: %s", target)


if __name__ == "__main__":
    main(tyro.cli(Args))
