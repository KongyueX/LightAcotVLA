import dataclasses
import json
import pathlib

import numpy as np
import pytest

from openpi.action_cot import multirate_dataset


def _small_shape() -> multirate_dataset.MultirateDatasetShape:
    return multirate_dataset.MultirateDatasetShape(
        window_size=4,
        num_cameras=2,
        image_height=3,
        image_width=5,
        image_channels=3,
        state_dim=6,
        coarse_horizon=3,
        action_dim=6,
        iar_tokens=2,
        iar_dim=7,
    )


def _record(
    shape: multirate_dataset.MultirateDatasetShape,
    *,
    anchor_index: int = 4,
) -> dict[str, np.ndarray | int]:
    return {
        "anchor_index": anchor_index,
        "task_id": 2,
        "episode_id": 3,
        "frame_id": 7,
        "policy_seed": 11,
        "images": np.full(
            (
                shape.window_size,
                shape.num_cameras,
                shape.image_height,
                shape.image_width,
                shape.image_channels,
            ),
            17,
            dtype=np.uint8,
        ),
        "states": np.zeros((shape.window_size, shape.state_dim), dtype=np.float32),
        "fresh_ear": np.ones(
            (shape.window_size, shape.coarse_horizon, shape.action_dim),
            dtype=np.float32,
        ),
        "fresh_iar": np.ones(
            (shape.window_size, shape.iar_tokens, shape.iar_dim),
            dtype=np.float32,
        ),
        "teacher_actions": np.ones((shape.window_size, shape.action_dim), dtype=np.float32),
        "b6_actions": np.full((shape.window_size, shape.action_dim), 0.5, dtype=np.float32),
        "hold_actions": np.zeros((shape.window_size, shape.action_dim), dtype=np.float32),
        "event_mask": np.array([False, True, False, True], dtype=np.bool_),
    }


def test_sharded_multirate_round_trip_and_manifest(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()
    with multirate_dataset.ShardedMultirateWriter(
        tmp_path,
        shape=shape,
        records_per_shard=1,
        metadata={"purpose": "test"},
    ) as writer:
        writer.append(_record(shape))
        writer.append(_record(shape, anchor_index=9))

    arrays = multirate_dataset.load_multirate_arrays((tmp_path,))
    assert arrays["anchor_index"].tolist() == [4, 9]
    assert arrays["images"].shape == (
        2,
        shape.window_size,
        shape.num_cameras,
        shape.image_height,
        shape.image_width,
        shape.image_channels,
    )
    assert arrays["images"].dtype == np.uint8
    assert arrays["states"].dtype == np.float16
    assert arrays["fresh_ear"].dtype == np.float16
    assert arrays["fresh_iar"].dtype == np.float16
    assert arrays["teacher_actions"].dtype == np.float16
    assert arrays["b6_actions"].dtype == np.float16
    assert arrays["hold_actions"].dtype == np.float16
    assert arrays["event_mask"].dtype == np.bool_

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == multirate_dataset.SCHEMA_VERSION
    assert manifest["num_shards"] == 2
    assert manifest["num_records"] == 2
    assert manifest["metadata"] == {"purpose": "test"}
    assert manifest["shards"] == [
        {"file": "shard-00000.h5", "num_records": 1},
        {"file": "shard-00001.h5", "num_records": 1},
    ]


def test_writer_commits_completed_buffer_when_context_raises(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()

    def write_then_raise() -> None:
        with multirate_dataset.ShardedMultirateWriter(
            tmp_path,
            shape=shape,
            records_per_shard=8,
        ) as writer:
            writer.append(_record(shape))
            raise RuntimeError("later sample failed")

    with pytest.raises(RuntimeError, match="later sample"):
        write_then_raise()

    arrays = multirate_dataset.load_multirate_arrays((tmp_path,))
    assert arrays["anchor_index"].tolist() == [4]


def test_writer_resumes_without_overwriting_shards(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()
    kwargs = {
        "shape": shape,
        "records_per_shard": 1,
        "metadata": {"purpose": "resume"},
    }
    with multirate_dataset.ShardedMultirateWriter(tmp_path, **kwargs) as writer:
        writer.append(_record(shape))

    # A stale or missing manifest must not make the committed shard disappear.
    (tmp_path / "manifest.json").unlink()
    with multirate_dataset.ShardedMultirateWriter(tmp_path, **kwargs) as writer:
        writer.append(_record(shape, anchor_index=12))

    assert sorted(path.name for path in tmp_path.glob("shard-*.h5")) == [
        "shard-00000.h5",
        "shard-00001.h5",
    ]
    arrays = multirate_dataset.load_multirate_arrays((tmp_path,))
    assert arrays["anchor_index"].tolist() == [4, 12]


def test_writer_rejects_shape_and_metadata_mismatch_on_resume(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()
    with multirate_dataset.ShardedMultirateWriter(
        tmp_path,
        shape=shape,
        records_per_shard=1,
        metadata={"protocol": "a"},
    ) as writer:
        writer.append(_record(shape))

    with pytest.raises(ValueError, match="shape mismatch"):
        multirate_dataset.ShardedMultirateWriter(
            tmp_path,
            shape=dataclasses_replace(shape, state_dim=shape.state_dim + 1),
            records_per_shard=1,
            metadata={"protocol": "a"},
        )
    with pytest.raises(ValueError, match="metadata differs"):
        multirate_dataset.ShardedMultirateWriter(
            tmp_path,
            shape=shape,
            records_per_shard=1,
            metadata={"protocol": "b"},
        )


def dataclasses_replace(
    shape: multirate_dataset.MultirateDatasetShape,
    **changes: int,
) -> multirate_dataset.MultirateDatasetShape:
    values = {field.name: getattr(shape, field.name) for field in dataclasses.fields(shape)}
    values.update(changes)
    return multirate_dataset.MultirateDatasetShape(**values)


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda record: record.pop("images"), KeyError, "missing required fields"),
        (lambda record: record.update(extra=np.array(1)), KeyError, "unexpected fields"),
        (
            lambda record: record.update(states=np.zeros((4, 5), dtype=np.float32)),
            ValueError,
            "states has shape",
        ),
        (
            lambda record: record.update(event_mask=np.zeros((4,), dtype=np.uint8)),
            TypeError,
            "boolean dtype",
        ),
        (
            lambda record: record.update(images=np.full(record["images"].shape, 256, dtype=np.int16)),
            ValueError,
            "uint8 range",
        ),
        (
            lambda record: record["fresh_iar"].__setitem__((0, 0, 0), np.nan),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_record_validation_is_strict(
    tmp_path: pathlib.Path,
    mutation,
    error: type[Exception],
    message: str,
) -> None:
    shape = _small_shape()
    record = _record(shape)
    mutation(record)

    writer = multirate_dataset.ShardedMultirateWriter(tmp_path, shape=shape)
    with pytest.raises(error, match=message):
        writer.append(record)
    writer.close()


def test_discover_shards_deduplicates_paths(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()
    with multirate_dataset.ShardedMultirateWriter(tmp_path, shape=shape, records_per_shard=1) as writer:
        writer.append(_record(shape))

    shard = tmp_path / "shard-00000.h5"
    assert multirate_dataset.discover_shards((tmp_path, shard)) == [shard.resolve()]


def test_load_selected_fields_avoids_materialising_large_arrays(tmp_path: pathlib.Path) -> None:
    shape = _small_shape()
    with multirate_dataset.ShardedMultirateWriter(tmp_path, shape=shape) as writer:
        writer.append(_record(shape))

    arrays = multirate_dataset.load_multirate_arrays((tmp_path,), fields=("anchor_index",))
    assert list(arrays) == ["anchor_index"]
    assert arrays["anchor_index"].tolist() == [4]

    with pytest.raises(KeyError, match="Unknown multirate dataset fields"):
        multirate_dataset.load_multirate_arrays((tmp_path,), fields=("missing",))


def test_invalid_shape_rejected() -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        multirate_dataset.MultirateDatasetShape(window_size=0)
