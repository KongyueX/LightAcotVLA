from __future__ import annotations

import importlib
import json
import pathlib
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("train_execution_horizon_predictor.py")
sys.path.insert(0, str(_SCRIPT.parent))
trainer = importlib.import_module("train_execution_horizon_predictor")


def _predictor_config(
    *,
    legacy_paired: bool = False,
    paired_distribution: bool = False,
    ordered_continuation: bool = False,
    ordered_readout: str = "global",
    temporal_layers: int = 2,
):
    return trainer.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16,
        state_dim=4,
        action_dim=7,
        physical_action_dim=7,
        coarse_horizon=15,
        action_horizon=25,
        hidden_dim=32,
        temporal_layers=temporal_layers,
        temporal_backbone="transformer",
        num_heads=4,
        candidate_horizons=(5, 10, 15, 20, 25),
        reference_horizon=10,
        visual_num_queries=4,
        paired_advantage_heads=legacy_paired,
        paired_distribution_heads=paired_distribution,
        ordered_continuation_head=ordered_continuation,
        ordered_readout=ordered_readout,
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
            "--ordered-continuation-head",
            "--loss-danger-rescue",
            "1",
            "--loss-paired-elapsed",
            "1",
            "--loss-ordered-listwise",
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
    assert args.ordered_continuation_head is True
    assert args.input_split_manifest == "four-way.json"
    assert weights.success_advantage == 0.0
    assert weights.elapsed_advantage == 0.0
    assert weights.danger_rescue == 1.0
    assert weights.paired_elapsed == 1.0
    assert weights.ordered_listwise == 1.0


def test_ordered_listwise_requires_explicit_continuation_head(tmp_path) -> None:
    args = trainer.Args(
        dataset=(str(tmp_path / "missing"),),
        output_dir=str(tmp_path / "out"),
        temporal_backbone="transformer",
        loss_ordered_listwise=1.0,
    )

    with pytest.raises(ValueError, match="requires --ordered-continuation-head"):
        trainer.main(args)
    assert not (tmp_path / "out").exists()


def test_ordered_elapsed_target_cli_defaults_and_paired_noise() -> None:
    default = trainer.Args(dataset=("dataset.h5",), output_dir="output")
    assert default.ordered_listwise_elapsed_mode == "root_minmax"
    assert default.ordered_listwise_elapsed_floor_seconds == 1.0
    args = trainer.tyro.cli(
        trainer.Args,
        args=[
            "--dataset", "dataset.h5", "--output-dir", "output",
            "--ordered-listwise-elapsed-mode", "paired_noise",
            "--ordered-listwise-elapsed-floor-seconds", "1.5",
        ],
    )
    trainer._validate_paired_args(args)  # noqa: SLF001
    assert args.ordered_listwise_elapsed_mode == "paired_noise"
    assert args.ordered_listwise_elapsed_floor_seconds == 1.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordered_listwise_elapsed_mode", "unknown"),
        ("ordered_listwise_elapsed_floor_seconds", 0.0),
        ("ordered_listwise_elapsed_floor_seconds", -1.0),
        ("ordered_listwise_elapsed_floor_seconds", float("nan")),
        ("ordered_listwise_elapsed_floor_seconds", float("inf")),
    ],
)
def test_ordered_elapsed_target_rejects_invalid_mode_or_floor(field, value) -> None:
    args = trainer.Args(dataset=("dataset.h5",), output_dir="output", **{field: value})
    with pytest.raises(ValueError, match=field):
        trainer._validate_paired_args(args)  # noqa: SLF001


def test_ordered_continuation_strict_resume_keeps_existing_parameter_tree(monkeypatch) -> None:
    source = trainer.ExecutionHorizonPredictor(_predictor_config(), rngs=nnx.Rngs(7))
    loaded = {"execution_horizon_predictor": nnx.state(source, nnx.Param).to_pure_dict()}
    monkeypatch.setattr(trainer.model_lib, "restore_params", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(trainer.model_lib, "convert_str_keys_to_int", lambda value: value)
    target = trainer.ExecutionHorizonPredictor(
        _predictor_config(ordered_continuation=True),
        rngs=nnx.Rngs(11),
    )

    restored, report = trainer._restore_predictor(target, "strict/params")  # noqa: SLF001

    assert report == {"mode": "strict", "enabled": False}
    assert restored.config.ordered_continuation_head is True


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


@pytest.mark.parametrize("target_depth", [2, 4])
@pytest.mark.parametrize("readout", ["candidate", "candidate_residual"])
def test_candidate_readout_resume_preserves_shared_weights_and_initializes_new_parameters(
    monkeypatch, target_depth, readout
) -> None:
    source = trainer.ExecutionHorizonPredictor(
        _predictor_config(ordered_continuation=True, paired_distribution=True), rngs=nnx.Rngs(7)
    )
    source_state = nnx.state(source, nnx.Param)
    loaded = {"execution_horizon_predictor": source_state.to_pure_dict()}
    monkeypatch.setattr(trainer.model_lib, "restore_params", lambda *_args, **_kwargs: loaded)
    target = trainer.ExecutionHorizonPredictor(
        _predictor_config(
            ordered_continuation=True,
            paired_distribution=True,
            ordered_readout=readout,
            temporal_layers=target_depth,
        ),
        rngs=nnx.Rngs(11),
    )
    initial_readout = nnx.state(target.candidate_readout, nnx.Param).flat_state()

    restored, report = trainer._restore_predictor(target, "global/params", resume_candidate_readout=True)  # noqa: SLF001

    restored_state = nnx.state(restored, nnx.Param).flat_state()
    for path, variable in source_state.flat_state().items():
        np.testing.assert_array_equal(restored_state[path].value, variable.value)
    for path, variable in initial_readout.items():
        np.testing.assert_array_equal(restored_state[("candidate_readout", *path)].value, variable.value)
    tokens = jnp.arange(2 * 25 * 32, dtype=jnp.float32).reshape(2, 25, 32) / 100
    for layer in restored.temporal_layers[2:]:
        np.testing.assert_array_equal(layer(tokens), tokens)
    expected_mode = (
        "global_to_candidate_residual_readout" if readout == "candidate_residual" else "global_to_candidate_readout"
    )
    assert report["mode"] == expected_mode
    assert report["identity_initialized_layers"] == list(range(2, target_depth))


def test_candidate_readout_requires_explicit_migration_and_rejects_incompatible_weights(monkeypatch) -> None:
    source = trainer.ExecutionHorizonPredictor(_predictor_config(ordered_continuation=True), rngs=nnx.Rngs(7))
    loaded = nnx.state(source, nnx.Param).to_pure_dict()
    monkeypatch.setattr(trainer.model_lib, "restore_params", lambda *_args, **_kwargs: loaded)
    target = trainer.ExecutionHorizonPredictor(
        _predictor_config(ordered_continuation=True, ordered_readout="candidate"), rngs=nnx.Rngs(11)
    )
    with pytest.raises(ValueError, match="Strict resume requires an identical parameter tree"):
        trainer._restore_predictor(target, "global/params")  # noqa: SLF001
    loaded["summary_proj"]["kernel"] = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="shared parameter shape mismatches"):
        trainer._restore_predictor(target, "global/params", resume_candidate_readout=True)  # noqa: SLF001


def test_candidate_readout_cli_and_migration_modes_are_explicit() -> None:
    args = trainer.tyro.cli(
        trainer.Args,
        args=[
            "--dataset",
            "dataset.h5",
            "--output-dir",
            "output",
            "--temporal-backbone",
            "transformer",
            "--ordered-continuation-head",
            "--ordered-readout",
            "candidate",
            "--resume-params",
            "global/params",
            "--resume-candidate-readout",
        ],
    )
    trainer._validate_paired_args(args)  # noqa: SLF001
    assert args.ordered_readout == "candidate"
    assert args.resume_candidate_readout
    args = trainer.dataclasses.replace(
        args,
        paired_distribution_heads=True,
        loss_danger_rescue=1.0,
        loss_paired_elapsed=1.0,
        resume_legacy_paired_heads=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._validate_paired_args(args)  # noqa: SLF001


def test_candidate_readout_only_cli_requires_residual_and_ordered_loss() -> None:
    args = trainer.tyro.cli(
        trainer.Args,
        args=[
            "--dataset", "dataset.h5", "--output-dir", "output",
            "--temporal-backbone", "transformer", "--ordered-continuation-head",
            "--ordered-readout", "candidate_residual", "--resume-params", "global/params",
            "--resume-candidate-readout", "--train-candidate-readout-only", "--loss-ordered-listwise", "2",
        ],
    )
    trainer._validate_paired_args(args)  # noqa: SLF001
    assert args.train_candidate_readout_only
    with pytest.raises(ValueError, match="requires --ordered-readout candidate_residual"):
        trainer._validate_paired_args(trainer.dataclasses.replace(args, ordered_readout="candidate"))  # noqa: SLF001
    with pytest.raises(ValueError, match="positive --loss-ordered-listwise"):
        trainer._validate_paired_args(trainer.dataclasses.replace(args, loss_ordered_listwise=0))  # noqa: SLF001


def test_candidate_readout_only_update_preserves_frozen_parameters_with_weight_decay() -> None:
    module = trainer.ExecutionHorizonPredictor(
        _predictor_config(ordered_continuation=True, ordered_readout="candidate_residual"), rngs=nnx.Rngs(7)
    )
    args = trainer.Args(dataset=("data.h5",), output_dir="out", train_candidate_readout_only=True)
    trainable_filter = trainer._trainable_filter(args)  # noqa: SLF001
    optimizer = optax.adamw(1e-3, weight_decay=1.0)
    state = nnx.state(module)
    before = {path: np.asarray(variable.value).copy() for path, variable in state.flat_state().items()}
    optimizer_state = optimizer.init(state.filter(trainable_filter))
    inputs = {
        "prefix_feature": jnp.zeros((1, 16)),
        "prefix_tokens": jnp.zeros((1, 4, 16)),
        "prefix_mask": jnp.ones((1, 4), dtype=jnp.bool_),
        "state": jnp.zeros((1, 4)),
        "coarse_actions": jnp.zeros((1, 15, 7)),
        "final_actions": jnp.zeros((1, 25, 7)),
        "previous_actions": jnp.zeros((1, 25, 7)),
        "previous_h": jnp.asarray([10]),
        "budget_balance": jnp.asarray([0.5]),
        "episode_progress": jnp.asarray([0.25]),
        "previous_valid": jnp.asarray([True]),
    }

    def loss_fn(candidate):
        logits = candidate(**inputs)["ordered_continuation_logits"]
        return jnp.mean(jax.nn.softplus(-logits)), {}

    updated, _, metrics = trainer._update_predictor(  # noqa: SLF001
        module, loss_fn, optimizer, optimizer_state, trainable_filter
    )
    updated_flat = updated.flat_state()
    for path, value in before.items():
        if path[0] != "candidate_readout":
            np.testing.assert_array_equal(updated_flat[path].value, value)
    assert float(metrics["gradient_norm"]) > 0
    assert any(
        not np.array_equal(updated_flat[path].value, value)
        for path, value in before.items()
        if path[0] == "candidate_readout"
    )
