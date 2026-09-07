# ruff: noqa: SLF001

from types import SimpleNamespace

import eval_libero_execution_horizon as evaluator
import numpy as np

from openpi.execution_horizon import v2


def _response(value=0.0):
    return {
        "execution_horizon_prefix_feature": np.full(2048, value, dtype=np.float32),
        "execution_horizon_state_normalized": np.full(32, value, dtype=np.float32),
        "execution_horizon_temporal_feature": np.zeros(256, dtype=np.float32),
        "execution_horizon_ordered_continuation_logits": np.zeros(4, dtype=np.float32),
        "execution_horizon_ordered_horizon_probability": np.array([.5, .25, .125, .0625, .0625]),
        "execution_horizon_candidate_horizons": np.array([5, 10, 15, 20, 25]),
    }


def test_history_uses_previous_observation_and_actual_elapsed_steps():
    response = _response(1.0)
    first, previous = evaluator._feedback_inputs(response, None, step=10, previous_h=10)
    assert not first["history_valid"]
    assert first["elapsed_steps"] == 0
    response["execution_horizon_prefix_feature"][:] = 9
    second, _ = evaluator._feedback_inputs(_response(2.0), previous, step=35, previous_h=25)
    assert second["history_valid"]
    assert second["elapsed_steps"] == 25
    assert second["previous_h"] == 25
    np.testing.assert_array_equal(second["previous_prefix_feature"], np.ones(2048))
    np.testing.assert_array_equal(second["previous_state"], np.ones(32))
    reset, _ = evaluator._feedback_inputs(_response(), None, step=10, previous_h=10)
    assert not reset["history_valid"]


def test_feedback_diagnostics_preserve_candidate_and_anchor_distributions():
    args = evaluator.build_parser().parse_args([
        "--output-dir", "/tmp/feedback", "--modes", evaluator.FEEDBACK_HISTORY_MODE,
        "--record-ordered-diagnostics",
    ])
    probabilities = [0, 0, 0, 0, 1]
    selector = SimpleNamespace(decide=lambda inputs: (25, {
        "ordered_horizon_probability": probabilities, "ordered_continuation_logits": [9] * 4,
    }))
    selected, info = evaluator._select_horizon(
        evaluator.FEEDBACK_HISTORY_MODE, _response(), args=args,
        budget_state=v2.EpisodeBudgetState(balance=6), selector=selector, feedback_inputs={},
    )
    assert selected == 25
    assert info["ordered_horizon_probability"] == probabilities
    assert info["anchor_ordered_horizon_probability"][0] == .5
    assert info["selector_postprocess_ms"] >= 0


def test_unused_feedback_options_preserve_old_resume_signature():
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/feedback"])
    signature = evaluator._run_signature(args)
    assert "feedback_current_params" not in signature
    assert "feedback_history_params" not in signature
    assert args.modes == list(evaluator.LEGACY_MODES)
