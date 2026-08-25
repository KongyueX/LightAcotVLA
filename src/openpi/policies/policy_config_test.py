import json

import numpy as np
import pytest

from openpi.policies import policy_config


def test_execution_horizon_artifact_resolves_output_or_params_dir(tmp_path) -> None:
    params = tmp_path / "params"
    params.mkdir()
    config = {
        "action_horizon": 20,
        "coarse_horizon": 15,
        "action_dim": 32,
    }
    (tmp_path / "predictor_config.json").write_text(json.dumps(config))

    output_params, output_config = policy_config._execution_horizon_artifact(tmp_path)  # noqa: SLF001
    direct_params, direct_config = policy_config._execution_horizon_artifact(params)  # noqa: SLF001

    assert output_params == params
    assert direct_params == params
    assert output_config == config
    assert direct_config == config


def test_merge_acot_endpoint_student_replaces_only_selected_params() -> None:
    base = {
        "PaliGemma": {
            "llm": {
                "layers_1": {"kernel": np.zeros((2, 3), dtype=np.float32)},
                "layers": {"kernel": np.zeros((2, 3), dtype=np.float32)},
            }
        },
        "action_out_proj": {"kernel": np.zeros((3, 4), dtype=np.float32)},
    }
    student = {
        "PaliGemma": {"llm": {"layers_1": {"kernel": np.ones((2, 3), dtype=np.float16)}}},
        "action_out_proj": {"kernel": np.ones((3, 4), dtype=np.float16)},
    }

    merged = policy_config.merge_acot_endpoint_student_params(base, student)

    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["layers_1"]["kernel"], 1)
    np.testing.assert_array_equal(merged["action_out_proj"]["kernel"], 1)
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["layers"]["kernel"], 0)
    assert merged["action_out_proj"]["kernel"].dtype == np.float32


def test_merge_acot_endpoint_student_rejects_frozen_module() -> None:
    base = {"implicit_action_reasoner": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    student = {"implicit_action_reasoner": {"kernel": np.ones((2, 2), dtype=np.float32)}}

    with pytest.raises(ValueError, match="Disallowed"):
        policy_config.merge_acot_endpoint_student_params(base, student)


def test_merge_acot_endpoint_student_rejects_shape_mismatch() -> None:
    base = {"action_out_proj": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    student = {"action_out_proj": {"kernel": np.zeros((3, 2), dtype=np.float32)}}

    with pytest.raises(ValueError, match="shape mismatch"):
        policy_config.merge_acot_endpoint_student_params(base, student)
