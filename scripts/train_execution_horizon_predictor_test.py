from __future__ import annotations

import importlib
import json
import pathlib
import sys

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("train_execution_horizon_predictor.py")
sys.path.insert(0, str(_SCRIPT.parent))
trainer = importlib.import_module("train_execution_horizon_predictor")


def _predictor_config(*, legacy_paired: bool = False, paired_distribution: bool = False):
    return trainer.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16,
        state_dim=4,
        action_dim=7,
        physical_action_dim=7,
        coarse_horizon=15,
        action_horizon=25,
        hidden_dim=32,
        temporal_layers=2,
        temporal_backbone="transformer",
        num_heads=4,
        candidate_horizons=(5, 10, 15, 20, 25),
        reference_horizon=10,
        visual_num_queries=4,
        paired_advantage_heads=legacy_paired,
        paired_distribution_heads=paired_distribution,
    )


def _arrays() -> dict[str, np.ndarray]:
    task_ids = np.repeat(np.arange(3, dtype=np.uint8), 8)
    episode_ids = np.tile(np.repeat(np.arange(4, dtype=np.uint32), 2), 3)
    return {"task_id": task_ids, "episode_id": episode_ids}


def test_transformer_split_is_episode_disjoint_and_independent_of_training_seed(tmp_path) -> None:
    common = {
        "dataset": (str(tmp_path),),
        "output_dir": str(tmp_path / "out"),
        "temporal_backbone": "transformer",
        "validation_fraction": 0.25,
        "calibration_fraction": 0.25,
        "split_seed": 42,
    }
    first = trainer._split_indices(_arrays(), trainer.Args(seed=7, **common))  # noqa: SLF001
    second = trainer._split_indices(_arrays(), trainer.Args(seed=11, **common))  # noqa: SLF001

    for first_partition, second_partition in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_partition, second_partition)

    arrays = _arrays()
    group_ids = arrays["task_id"].astype(np.uint64) * np.uint64(1_000_000_000)
    group_ids += arrays["episode_id"].astype(np.uint64)
    partition_groups = [set(group_ids[indices].tolist()) for indices in first]
    assert partition_groups[0].isdisjoint(partition_groups[1])
    assert partition_groups[0].isdisjoint(partition_groups[2])
    assert partition_groups[1].isdisjoint(partition_groups[2])


def test_transformer_split_can_stratify_every_partition_by_task(tmp_path) -> None:
    arrays = _arrays()
    args = trainer.Args(
        dataset=(str(tmp_path),),
        output_dir=str(tmp_path / "out"),
        temporal_backbone="transformer",
        validation_fraction=0.25,
        calibration_fraction=0.25,
        split_seed=42,
        stratify_splits_by_task=True,
    )

    partitions = trainer._split_indices(arrays, args)  # noqa: SLF001

    for indices in partitions:
        assert set(arrays["task_id"][indices].tolist()) == {0, 1, 2}
    group_ids = arrays["task_id"].astype(np.uint64) * np.uint64(1_000_000_000)
    group_ids += arrays["episode_id"].astype(np.uint64)
    partition_groups = [set(group_ids[indices].tolist()) for indices in partitions]
    assert partition_groups[0].isdisjoint(partition_groups[1])
    assert partition_groups[0].isdisjoint(partition_groups[2])
    assert partition_groups[1].isdisjoint(partition_groups[2])


def test_transformer_split_accepts_explicit_four_way_manifest(tmp_path) -> None:
    arrays = _arrays()
    manifest = {
        "split_schema_version": 2,
        "split_roles": {
            "train": "train",
            "early_stop": "early_stop",
            "calibration": "calibration",
            "dev_audit": "development_audit",
        },
        "train_group_ids": [0, 1_000_000_000, 2_000_000_000],
        "early_stop_group_ids": [1, 1_000_000_001, 2_000_000_001],
        "calibration_group_ids": [2, 1_000_000_002, 2_000_000_002],
        "dev_audit_group_ids": [3, 1_000_000_003, 2_000_000_003],
    }
    manifest_path = tmp_path / "four_way_split.json"
    manifest_path.write_text(json.dumps(manifest))
    args = trainer.Args(
        dataset=(str(tmp_path),),
        output_dir=str(tmp_path / "out"),
        temporal_backbone="transformer",
        input_split_manifest=str(manifest_path),
    )

    train, early_stop, calibration = trainer._split_indices(arrays, args)  # noqa: SLF001
    groups = arrays["task_id"].astype(np.uint64) * np.uint64(1_000_000_000)
    groups += arrays["episode_id"].astype(np.uint64)

    assert set(groups[train].tolist()) == set(manifest["train_group_ids"])
    assert set(groups[early_stop].tolist()) == set(manifest["early_stop_group_ids"])
    assert set(groups[calibration].tolist()) == set(manifest["calibration_group_ids"])
    assert set(groups[train]).isdisjoint(groups[early_stop])
    assert set(groups[train]).isdisjoint(groups[calibration])
    assert set(groups[early_stop]).isdisjoint(groups[calibration])
    used_for_fit_or_selection = set(groups[np.concatenate((train, early_stop, calibration))].tolist())
    assert used_for_fit_or_selection.isdisjoint(manifest["dev_audit_group_ids"])


def test_paired_distribution_cli_and_loss_weights_are_explicit() -> None:
    args = trainer.tyro.cli(
        trainer.Args,
        args=[
            "--dataset",
            "dataset.h5",
            "--output-dir",
            "output",
            "--temporal-backbone",
            "transformer",
            "--paired-distribution-heads",
            "--loss-danger-rescue",
            "1",
            "--loss-paired-elapsed",
            "1",
            "--resume-params",
            "legacy/params",
            "--resume-legacy-paired-heads",
            "--input-split-manifest",
            "four-way.json",
        ],
    )

    trainer._validate_paired_args(args)  # noqa: SLF001
    weights = trainer._loss_weights(args)  # noqa: SLF001
    assert args.paired_distribution_heads is True
    assert args.resume_legacy_paired_heads is True
    assert args.input_split_manifest == "four-way.json"
    assert weights.success_advantage == 0.0
    assert weights.elapsed_advantage == 0.0
    assert weights.danger_rescue == 1.0
    assert weights.paired_elapsed == 1.0


def test_paired_distribution_fails_fast_without_both_likelihood_losses(tmp_path) -> None:
    args = trainer.Args(
        dataset=(str(tmp_path / "missing"),),
        output_dir=str(tmp_path / "out"),
        temporal_backbone="transformer",
        paired_distribution_heads=True,
        loss_danger_rescue=1.0,
        loss_paired_elapsed=0.0,
    )

    with pytest.raises(ValueError, match="positive --loss-danger-rescue and --loss-paired-elapsed"):
        trainer.main(args)
    assert not (tmp_path / "out").exists()


def test_resume_legacy_paired_heads_migrates_interleaved_categorical_logits(monkeypatch) -> None:
    legacy = trainer.ExecutionHorizonPredictor(_predictor_config(legacy_paired=True), rngs=nnx.Rngs(7))
    danger_kernel = jnp.arange(legacy.danger_logits_head.kernel.value.size, dtype=jnp.float32).reshape(
        legacy.danger_logits_head.kernel.value.shape
    )
    rescue_kernel = -danger_kernel - 1.0
    danger_bias = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    rescue_bias = jnp.asarray([-1.0, -2.0, -3.0], dtype=jnp.float32)
    legacy.danger_logits_head.kernel.value = danger_kernel
    legacy.rescue_logits_head.kernel.value = rescue_kernel
    legacy.danger_logits_head.bias.value = danger_bias
    legacy.rescue_logits_head.bias.value = rescue_bias
    legacy.elapsed_advantage_log_scale_head.kernel.value = jnp.full_like(
        legacy.elapsed_advantage_log_scale_head.kernel.value,
        123.0,
    )
    loaded = {"execution_horizon_predictor": nnx.state(legacy, nnx.Param).to_pure_dict()}
    monkeypatch.setattr(trainer.model_lib, "restore_params", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(trainer.model_lib, "convert_str_keys_to_int", lambda value: value)

    target = trainer.ExecutionHorizonPredictor(_predictor_config(paired_distribution=True), rngs=nnx.Rngs(11))
    initial_raw_scale_kernel = np.asarray(target.elapsed_advantage_log_scale_head.kernel.value).copy()
    restored, report = trainer._restore_predictor(  # noqa: SLF001
        target,
        "legacy/params",
        resume_legacy_paired_heads=True,
    )
    migrated_kernel = np.asarray(restored.paired_outcome_logits_head.kernel.value).reshape(32, 3, 3)
    migrated_bias = np.asarray(restored.paired_outcome_logits_head.bias.value).reshape(3, 3)

    np.testing.assert_array_equal(migrated_kernel[..., 0], danger_kernel)
    np.testing.assert_array_equal(migrated_kernel[..., 1], 0.0)
    np.testing.assert_array_equal(migrated_kernel[..., 2], rescue_kernel)
    np.testing.assert_array_equal(migrated_bias[..., 0], danger_bias)
    np.testing.assert_array_equal(migrated_bias[..., 1], 0.0)
    np.testing.assert_array_equal(migrated_bias[..., 2], rescue_bias)
    np.testing.assert_array_equal(
        restored.elapsed_advantage_head.kernel.value,
        legacy.elapsed_advantage_head.kernel.value,
    )
    np.testing.assert_array_equal(
        restored.elapsed_advantage_log_scale_head.kernel.value,
        initial_raw_scale_kernel,
    )
    assert not np.array_equal(
        restored.elapsed_advantage_log_scale_head.kernel.value,
        legacy.elapsed_advantage_log_scale_head.kernel.value,
    )
    assert report["mode"] == "legacy_paired_to_distribution"
    assert report["tie_logits_initialized_to_zero"] is True
    assert "elapsed_advantage_log_scale_head/kernel" in report["reinitialized_parameter_leaves"]


def test_default_resume_remains_strict_and_restores_the_full_tree(monkeypatch) -> None:
    source = trainer.ExecutionHorizonPredictor(_predictor_config(legacy_paired=True), rngs=nnx.Rngs(7))
    loaded = {"execution_horizon_predictor": nnx.state(source, nnx.Param).to_pure_dict()}
    monkeypatch.setattr(trainer.model_lib, "restore_params", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(trainer.model_lib, "convert_str_keys_to_int", lambda value: value)
    target = trainer.ExecutionHorizonPredictor(_predictor_config(legacy_paired=True), rngs=nnx.Rngs(11))

    restored, report = trainer._restore_predictor(target, "strict/params")  # noqa: SLF001
    source_state = nnx.state(source, nnx.Param).flat_state()
    restored_state = nnx.state(restored, nnx.Param).flat_state()

    assert source_state.keys() == restored_state.keys()
    for path in source_state:
        np.testing.assert_array_equal(source_state[path].value, restored_state[path].value)
    assert report == {"mode": "strict", "enabled": False}


def test_resume_legacy_paired_heads_rejects_shared_shape_mismatch() -> None:
    legacy = trainer.ExecutionHorizonPredictor(_predictor_config(legacy_paired=True), rngs=nnx.Rngs(7))
    loaded = nnx.state(legacy, nnx.Param).to_pure_dict()
    loaded["elapsed_advantage_head"]["kernel"] = jnp.zeros((1, 1), dtype=jnp.float32)
    target = trainer.ExecutionHorizonPredictor(_predictor_config(paired_distribution=True), rngs=nnx.Rngs(11))
    _, state = nnx.split(target)

    with pytest.raises(ValueError, match="shared parameter shape mismatches"):
        trainer._migrate_legacy_paired_state(target, state, loaded)  # noqa: SLF001
