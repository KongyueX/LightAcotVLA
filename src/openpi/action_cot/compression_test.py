import numpy as np

from openpi.action_cot import compression


def _assert_covers_without_overlap(segments, t_len):
    cursor = 0
    for start, end in segments:
        assert start == cursor
        assert end > start
        assert end <= t_len
        cursor = end
    assert cursor == t_len


def test_segment_fixed_covers_trajectory():
    coarse_actions = np.zeros((30, 21), dtype=np.float32)

    segments = compression.segment_fixed(coarse_actions, chunk_size=5)

    assert segments == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    _assert_covers_without_overlap(segments, t_len=30)


def test_segment_adaptive_covers_without_overlap():
    coarse_actions = np.zeros((30, 4), dtype=np.float32)
    coarse_actions[:8, 0] = np.linspace(0.0, 1.0, 8)
    coarse_actions[8:16, 1] = np.linspace(0.0, 4.0, 8)
    coarse_actions[16:24, 2] = np.linspace(2.0, -2.0, 8)
    coarse_actions[24:, 3] = np.linspace(0.0, 3.0, 6)

    segments = compression.segment_adaptive(coarse_actions, min_len=3, max_len=8, max_segments=8)

    _assert_covers_without_overlap(segments, t_len=30)


def test_entropy_output_has_segment_shape():
    rng = np.random.default_rng(0)
    coarse_samples = rng.normal(size=(4, 30, 21)).astype(np.float32)
    segments = compression.segment_fixed(coarse_samples[0], chunk_size=5)

    entropy = compression.compute_mc_predictive_entropy(coarse_samples, segments)

    assert entropy.shape == (6,)


def test_skip_mask_never_skips_all_segments():
    entropy = np.asarray([-3.0, -2.0, -1.0], dtype=np.float32)

    skip_mask = compression.make_skip_mask(entropy, prune_ratio=1.0, min_keep=1, max_skip_ratio=1.0)

    assert skip_mask.shape == (3,)
    assert np.any(skip_mask == 0)


def test_expand_segment_mask_matches_segment_mask():
    segments = [(0, 2), (2, 5), (5, 6)]
    skip_mask = np.asarray([1, 0, 1], dtype=np.int8)

    frame_mask = compression.expand_segment_mask(skip_mask, segments, t_len=6)

    np.testing.assert_array_equal(frame_mask, np.asarray([1, 1, 0, 0, 0, 1], dtype=np.int8))
