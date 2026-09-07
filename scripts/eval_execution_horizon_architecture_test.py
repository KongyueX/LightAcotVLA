# ruff: noqa: SLF001

import eval_libero_execution_horizon as evaluate
import numpy as np
import pytest


def test_architecture_modes_use_their_own_endpoints_without_changing_base_args(tmp_path):
    args = evaluate.build_parser().parse_args([
        "--output-dir", str(tmp_path), "--port", "8050", "--visual-query-port", "8051",
        "--expert-hidden-port", "8052", "--model-action-horizon", "25",
        "--modes", "ordered_transformer", "ordered_visual_query", "ordered_expert_hidden",
    ])
    base, visual, expert = object(), object(), object()
    endpoints = {evaluate.VISUAL_QUERY_MODE: visual, evaluate.EXPERT_HIDDEN_MODE: expert}
    for mode, expected, port in [(evaluate.ORDERED_MODE, base, 8050), (evaluate.VISUAL_QUERY_MODE, visual, 8051),
                                  (evaluate.EXPERT_HIDDEN_MODE, expert, 8052)]:
        client, runtime = evaluate._mode_runtime(mode, args, base, architecture_clients=endpoints)
        assert client is expected
        assert runtime.port == port
        assert runtime.model_action_horizon == 25
    assert args.port == 8050
    with pytest.raises(ValueError, match="endpoint"):
        evaluate._mode_runtime(evaluate.VISUAL_QUERY_MODE, args, base)


@pytest.mark.parametrize("mode", evaluate.ARCHITECTURE_MODES)
def test_architecture_modes_use_served_ordered_h_without_client_selector(tmp_path, mode):
    args = evaluate.build_parser().parse_args(["--output-dir", str(tmp_path), "--model-action-horizon", "25"])
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(15),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
    }
    horizon, info = evaluate._select_horizon(mode, result, args=args, budget_state=None)
    assert horizon == 15
    assert info["raw_horizon"] == 15
