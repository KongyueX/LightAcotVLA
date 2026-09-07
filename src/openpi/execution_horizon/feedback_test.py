from __future__ import annotations

import numpy as np

from openpi.execution_horizon.feedback import FEATURE_DIM
from openpi.execution_horizon.feedback import FeedbackSelector
from openpi.execution_horizon.ordered_smdp import ordered_log_probabilities


def _inputs(*, history_valid: bool = True) -> dict[str, np.ndarray]:
    return {
        "temporal_feature": np.linspace(-1, 1, 256, dtype=np.float32),
        "prefix_feature": np.ones(2048, dtype=np.float32),
        "state": np.ones(32, dtype=np.float32),
        "previous_prefix_feature": np.zeros(2048, dtype=np.float32),
        "previous_state": np.zeros(32, dtype=np.float32),
        "previous_h": np.asarray(25), "elapsed_steps": np.asarray(25),
        "history_valid": np.asarray(history_valid),
        "continuation_logits": np.asarray([1, 2, 3, 4], dtype=np.float32),
    }


def _selector(variant: str = "history") -> FeedbackSelector:
    kernel = np.zeros((2048, 256), dtype=np.float32)
    kernel[:256] = np.eye(256)
    return FeedbackSelector.initialize(kernel, np.zeros(256), variant=variant, a_predictor_dir="/A")


def test_zero_initialized_variants_preserve_a_and_first_call_remains_a_after_training() -> None:
    for variant in ("current", "history"):
        selector = _selector(variant)
        inputs = _inputs()
        result = selector.forward(selector.build_features(inputs), inputs["continuation_logits"], history_valid=True)
        np.testing.assert_array_equal(result["continuation_logits"], inputs["continuation_logits"])
        np.testing.assert_array_equal(
            result["log_probabilities"], ordered_log_probabilities(inputs["continuation_logits"])
        )
        selector.params["output_b"][:] = 7
        inputs = _inputs(history_valid=False)
        feature = selector.build_features(inputs)
        np.testing.assert_array_equal(feature[256:], np.zeros(FEATURE_DIM - 256))
        result = selector.forward(feature, inputs["continuation_logits"], history_valid=False)
        np.testing.assert_array_equal(result["continuation_logits"], inputs["continuation_logits"])


def test_frozen_projection_difference_and_current_control_exclude_history() -> None:
    inputs = _inputs()
    selector = _selector()
    feature = selector.build_features(inputs)
    np.testing.assert_allclose(feature[256:512], 1 / (1 + np.exp(-1)), atol=1e-7)
    np.testing.assert_array_equal(feature[512:544], np.ones(32))
    np.testing.assert_array_equal(feature[544:], [1, 1, 1])
    inputs["elapsed_steps"] = np.asarray(12)
    assert selector.build_features(inputs)[545] == np.float32(12 / 25)
    current = _selector("current")
    first = current.build_features(inputs)
    inputs["previous_state"][:] = 20
    inputs["previous_prefix_feature"][:] = 20
    np.testing.assert_array_equal(current.build_features(inputs), first)
    np.testing.assert_array_equal(first[256:], np.zeros(FEATURE_DIM - 256))


def test_train_normalization_and_npz_roundtrip(tmp_path) -> None:
    selector = _selector()
    first = selector.build_features(_inputs())
    second = first.copy()
    second[0] += 4
    selector.fit_normalization(np.stack([first, second]))
    assert selector.feature_mean[0] == first[0] + 2
    assert selector.feature_std[0] == 2
    np.testing.assert_array_equal(selector.feature_std[1:], np.ones(FEATURE_DIM - 1))
    selector.params["output_w"][0, 2] = 0.125
    path = tmp_path / "checkpoint.npz"
    selector.save(path)
    loaded = FeedbackSelector.load(path)
    np.testing.assert_array_equal(loaded.prefix_kernel, selector.prefix_kernel)
    np.testing.assert_array_equal(loaded.feature_mean, selector.feature_mean)
    np.testing.assert_array_equal(loaded.feature_std, selector.feature_std)
    assert loaded.metadata == selector.metadata
    assert loaded.decide(_inputs()) == selector.decide(_inputs())
