from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
import execution_horizon_aggregate_risk_common as common  # noqa: E402


def _write_manifest(path: pathlib.Path, **updates) -> pathlib.Path:
    payload = {
        "train_group_ids": [1],
        "calibration_group_ids": [2],
        "validation_group_ids": [3],
        **updates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_manifest_enforces_role_and_disjoint_groups(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path / "split_manifest.json")
    _, groups = common.load_split_manifest(
        manifest_path,
        split_name="calibration",
        required_role="calibration",
    )
    np.testing.assert_array_equal(groups, np.asarray([2], dtype=np.uint64))

    with pytest.raises(ValueError, match="requires an explicit 'calibration' split"):
        common.load_split_manifest(
            manifest_path,
            split_name="validation",
            required_role="calibration",
        )

    overlapping_path = _write_manifest(
        tmp_path / "overlapping.json",
        calibration_group_ids=[2, 3],
    )
    with pytest.raises(ValueError, match="overlap"):
        common.load_split_manifest(
            overlapping_path,
            split_name="calibration",
            required_role="calibration",
        )

    disguised_final = _write_manifest(
        tmp_path / "development" / "split_manifest.json",
        fresh_final_group_ids=[4],
        split_roles={"fresh_final": "validation"},
    )
    with pytest.raises(ValueError, match="Non-development split"):
        common.load_split_manifest(
            disguised_final,
            split_name="fresh_final",
            required_role="validation",
        )

    canonical_override = _write_manifest(
        tmp_path / "canonical_override.json",
        split_roles={"calibration": "validation"},
    )
    with pytest.raises(ValueError, match="may not override canonical split"):
        common.load_split_manifest(
            canonical_override,
            split_name="calibration",
            required_role="calibration",
        )


def test_development_path_guard_runs_before_dataset_open(tmp_path) -> None:
    final_path = tmp_path / "fresh_final_e90_99" / "shard-00000.h5"
    with pytest.raises(ValueError, match="refuses final/test/holdout"):
        common.validate_development_dataset_paths((final_path,))
    dotted_manifest = tmp_path / "development" / "split.final.json"
    with pytest.raises(ValueError, match="refuses final/test/holdout"):
        common.load_split_manifest(
            dotted_manifest,
            split_name="validation",
            required_role="validation",
        )
    dotted_test_shard = tmp_path / "fresh.test" / "shard-00000.h5"
    with pytest.raises(ValueError, match="refuses final/test/holdout"):
        common.validate_development_dataset_paths((dotted_test_shard,))


def test_digests_bind_params_and_full_shard_bytes(tmp_path) -> None:
    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    first_json.write_text('{"b": 2, "a": 1}\n')
    second_json.write_text('{\n  "a": 1,\n  "b": 2\n}\n')
    assert common.json_file_digest(first_json) == common.json_file_digest(second_json)

    params = tmp_path / "params"
    params.mkdir()
    (params / "metadata").write_bytes(b"metadata-v1")
    (params / "weights").write_bytes(b"weights-v1")
    params_before = common.params_tree_digest(params)
    (params / "weights").write_bytes(b"weights-v2")
    assert common.params_tree_digest(params) != params_before
    params_link = tmp_path / "params-link"
    params_link.symlink_to(params, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink root"):
        common.params_tree_digest(params_link)

    development = tmp_path / "development"
    development.mkdir()
    manifest = development / "manifest.json"
    manifest.write_text('{"schema_version": 2, "num_records": 1}\n')
    shard = development / "shard-00000.h5"
    shard.write_bytes(b"entire-shard-v1")
    first_fingerprint = common.development_dataset_fingerprint((development,))
    manifest.write_text('{\n  "num_records": 1,\n  "schema_version": 2\n}\n')
    assert common.development_dataset_fingerprint((development,)) == first_fingerprint
    shard.write_bytes(b"entire-shard-v2")
    assert common.development_dataset_fingerprint((development,)) != first_fingerprint

    frozen = {
        "predictor_config_digest": "0" * 64,
        "params_digest": "1" * 64,
    }
    live = {**frozen, "params_digest": "2" * 64}
    with pytest.raises(ValueError, match="params_digest"):
        common.verify_provenance(frozen, live)


def test_atomic_json_writer_replaces_temporary_file_once(tmp_path) -> None:
    output = tmp_path / "result.json"
    returned = common.write_json(output, {"status": "complete", "value": np.int64(3)})
    assert returned == output.resolve()
    assert json.loads(output.read_text()) == {"status": "complete", "value": 3}
    assert not output.with_suffix(".json.tmp").exists()


def test_json_normalization_treats_artifact_tuple_roundtrip_as_equal() -> None:
    file_loaded = {"candidate_horizons": [5, 10, 15], "residual_quantiles": [0.1, 0.2]}
    artifact_loaded = {"candidate_horizons": (5, 10, 15), "residual_quantiles": (0.1, 0.2)}
    assert common.jsonable(file_loaded) == common.jsonable(artifact_loaded)


def test_raw_trial_recomputation_rejects_aggregate_count_tampering() -> None:
    valid = np.ones((1, 3, 3), dtype=np.bool_)
    success = np.asarray([[[True, True, False], [True, True, False], [True, False, False]]])
    labels = {
        "trial_valid": valid,
        "trial_success": success,
        "trial_count": np.full((1, 3), 3),
        "success_count": np.asarray([[2, 2, 1]]),
        "paired_trial_count": np.asarray([[0, 0, 3]]),
        "dangerous_long_count": np.asarray([[0, 0, 1]]),
    }
    common.validate_raw_paired_labels(labels, candidate_horizons=(5, 10, 15), reference_horizon=10)
    labels["dangerous_long_count"][0, 2] = 0
    with pytest.raises(ValueError, match="dangerous_long_count"):
        common.validate_raw_paired_labels(labels, candidate_horizons=(5, 10, 15), reference_horizon=10)


def test_selection_metrics_reports_four_gates_and_cluster_bootstrap() -> None:
    candidates = (5, 10, 15)
    trial_valid = np.ones((4, 3, 100), dtype=np.bool_)
    trial_success = np.zeros((4, 3, 100), dtype=np.bool_)
    trial_success[..., :80] = True
    paired = np.zeros((4, 3), dtype=np.uint16)
    paired[:, 2] = 100
    labels = {
        "trial_count": np.full((4, 3), 100, dtype=np.uint16),
        "success_count": np.asarray(
            [
                [80, 80, 80],
                [80, 80, 80],
                [80, 80, 80],
                [80, 80, 80],
            ],
            dtype=np.uint16,
        ),
        "elapsed_mean": np.asarray(
            [
                [11.0, 10.0, 8.0],
                [11.0, 10.0, 8.0],
                [11.0, 10.0, 8.0],
                [11.0, 10.0, 8.0],
            ],
            dtype=np.float32,
        ),
        "remaining_calls_mean": np.asarray(
            [
                [12.0, 10.0, 7.0],
                [12.0, 10.0, 7.0],
                [12.0, 10.0, 7.0],
                [12.0, 10.0, 7.0],
            ],
            dtype=np.float32,
        ),
        "dangerous_long_count": np.zeros((4, 3), dtype=np.uint16),
        "paired_trial_count": paired,
        "trial_valid": trial_valid,
        "trial_success": trial_success,
    }
    result = common.selection_metrics(
        labels,
        selected_horizons=np.full((4,), 15),
        candidate_horizons=candidates,
        reference_horizon=10,
        cluster_ids=np.asarray([1, 2, 3, 4], dtype=np.uint64),
        bootstrap_samples=500,
        seed=7,
        success_noninferiority_margin=0.01,
        false_long_upper_bound=0.05,
    )

    assert result["selected_h_distribution"] == {"15": 4}
    assert result["long_h_coverage"] == 1.0
    assert result["success_advantage_vs_reference_cluster_bootstrap"]["mean"] == 0.0
    assert result["elapsed_advantage_vs_reference_cluster_bootstrap"]["mean"] == -2.0
    assert result["calls_advantage_vs_reference_cluster_bootstrap"]["mean"] == -3.0
    assert result["gate_checks"] == {
        "nonzero_long_coverage": True,
        "success_noninferiority": True,
        "elapsed_improvement": True,
        "false_long_control": True,
    }
    assert result["offline_engineering_gate"] is True


def test_constraint_diagnostics_counts_each_rejection() -> None:
    common_fields = {
        "long_horizons": (15, 20),
        "success_score": (0.0, -0.1),
        "elapsed_score": (-1.0, -2.0),
        "danger_probability": (0.01, 0.20),
        "faster_probability": (0.9, 0.9),
        "long_event_probability": (0.1, 0.3),
        "short_event_probability": (0.0, 0.0, 0.1),
        "ood_probability": 0.1,
        "elapsed_pass": (True, True),
        "danger_pass": (True, False),
        "faster_pass": (True, True),
        "hazard_pass": (True, False),
        "ood_pass": (True, True),
    }
    decisions = [
        SimpleNamespace(
            **common_fields,
            selected_horizon=15,
            reason="aggregate_long_h",
            success_pass=(True, False),
            long_eligible=(True, False),
            rejection_reasons=((), ("success", "danger", "hazard")),
        ),
        SimpleNamespace(
            **common_fields,
            selected_horizon=10,
            reason="aggregate_reference_fallback",
            success_pass=(False, False),
            long_eligible=(False, False),
            rejection_reasons=(("success",), ("success", "danger", "hazard")),
        ),
    ]

    result = common.decision_diagnostics(decisions)

    assert result["decision_reason_distribution"] == {
        "aggregate_long_h": 1,
        "aggregate_reference_fallback": 1,
    }
    assert result["per_long_horizon"]["15"]["eligible_count"] == 1
    assert result["per_long_horizon"]["15"]["rejected_by_constraint_independent"]["success"] == 1
    assert result["per_long_horizon"]["20"]["first_rejection_count"] == {"success": 2}
