import pathlib

import numpy as np
import pytest

from openpi.action_cot import endpoint_dataset


def _record(shape: endpoint_dataset.EndpointDatasetShape) -> dict[str, np.ndarray | int]:
    return {
        "dataset_index": 4,
        "task_id": 2,
        "episode_id": 3,
        "frame_id": 7,
        "policy_seed": 11,
        "coarse_noise": np.zeros((shape.coarse_horizon, shape.action_dim), dtype=np.float32),
        "action_noise": np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32),
        "clean_coarse": np.ones((shape.coarse_horizon, shape.action_dim), dtype=np.float32),
        "clean_actions": np.ones((shape.action_horizon, shape.action_dim), dtype=np.float32),
        "clean_coarse_env": np.ones((shape.coarse_horizon, shape.env_action_dim), dtype=np.float32),
        "intervention_ids": np.arange(shape.num_interventions, dtype=np.uint8),
        "intervention_valid": np.ones((shape.num_interventions,), dtype=np.bool_),
        "intervention_coarse": np.ones(
            (shape.num_interventions, shape.coarse_horizon, shape.action_dim), dtype=np.float32
        ),
        "intervention_actions": np.ones(
            (shape.num_interventions, shape.action_horizon, shape.action_dim), dtype=np.float32
        ),
        "intervention_coarse_env": np.ones(
            (shape.num_interventions, shape.coarse_horizon, shape.env_action_dim), dtype=np.float32
        ),
        "response_l2": np.arange(shape.num_interventions, dtype=np.float32),
    }


def test_sharded_endpoint_round_trip(tmp_path: pathlib.Path) -> None:
    shape = endpoint_dataset.EndpointDatasetShape(action_dim=8, env_action_dim=7)
    with endpoint_dataset.ShardedEndpointWriter(
        tmp_path,
        shape=shape,
        records_per_shard=1,
        metadata={"purpose": "test"},
    ) as writer:
        writer.append(_record(shape))
        second = _record(shape)
        second["dataset_index"] = 9
        writer.append(second)

    arrays = endpoint_dataset.load_endpoint_arrays((tmp_path,))
    assert arrays["dataset_index"].tolist() == [4, 9]
    assert arrays["clean_coarse"].shape == (2, shape.coarse_horizon, shape.action_dim)
    assert arrays["intervention_actions"].dtype == np.float16


def test_writer_commits_completed_buffer_when_context_raises(tmp_path: pathlib.Path) -> None:
    shape = endpoint_dataset.EndpointDatasetShape(action_dim=8, env_action_dim=7)

    def write_then_raise() -> None:
        with endpoint_dataset.ShardedEndpointWriter(
            tmp_path,
            shape=shape,
            records_per_shard=8,
        ) as writer:
            writer.append(_record(shape))
            raise RuntimeError("later sample failed")

    with pytest.raises(RuntimeError, match="later sample"):
        write_then_raise()

    arrays = endpoint_dataset.load_endpoint_arrays((tmp_path,))
    assert arrays["dataset_index"].tolist() == [4]


@pytest.mark.parametrize("name", endpoint_dataset.INTERVENTION_NAMES)
def test_interventions_preserve_shape(name: str) -> None:
    coarse = np.arange(15 * 7, dtype=np.float32).reshape(15, 7)
    changed = endpoint_dataset.apply_intervention(coarse, name, seed=3)
    assert changed.shape == coarse.shape
    if name == "null":
        np.testing.assert_array_equal(changed, coarse)
    else:
        assert not np.array_equal(changed, coarse)


def test_unknown_intervention_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown intervention"):
        endpoint_dataset.apply_intervention(np.zeros((15, 7), dtype=np.float32), "bad")
