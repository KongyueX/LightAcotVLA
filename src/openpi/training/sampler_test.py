from __future__ import annotations

import torch

from openpi.training import sampler


class _EpisodeDataset:
    def __init__(self, lengths: list[int]):
        starts = []
        ends = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }
        self._length = cursor

    def __len__(self) -> int:
        return self._length


def test_episode_split_is_deterministic_and_frame_disjoint():
    dataset = _EpisodeDataset([2, 3, 4, 5, 6])
    train = sampler._episode_split_frame_indices(dataset, "train", 0.2, 7)
    validation = sampler._episode_split_frame_indices(dataset, "validation", 0.2, 7)

    assert train == sampler._episode_split_frame_indices(dataset, "train", 0.2, 7)
    assert validation == sampler._episode_split_frame_indices(dataset, "validation", 0.2, 7)
    assert set(train).isdisjoint(validation)
    assert set(train).union(validation) == set(range(len(dataset)))


def test_episode_split_changes_with_seed_without_frame_leakage():
    dataset = _EpisodeDataset([2, 2, 2, 2, 2, 2])
    validation_seed_7 = sampler._episode_split_frame_indices(dataset, "validation", 0.25, 7)
    validation_seed_8 = sampler._episode_split_frame_indices(dataset, "validation", 0.25, 8)

    assert set(validation_seed_7) != set(validation_seed_8)
    for seed, validation in ((7, validation_seed_7), (8, validation_seed_8)):
        train = sampler._episode_split_frame_indices(dataset, "train", 0.25, seed)
        assert set(train).isdisjoint(validation)
