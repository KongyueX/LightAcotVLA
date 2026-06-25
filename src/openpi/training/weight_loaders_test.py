import numpy as np

from openpi.training import weight_loaders


def test_merge_params_converts_numeric_string_keys():
    params = {
        "implicit_action_reasoner": {
            "query_params": {
                0: np.zeros((1, 2), dtype=np.float32),
            },
        },
    }
    loaded = {
        "implicit_action_reasoner": {
            "query_params": {
                "0": np.ones((1, 2), dtype=np.float32),
            },
        },
    }

    merged = weight_loaders._merge_params(loaded, params, missing_regex=".*action_cot_skip_head.*")

    np.testing.assert_array_equal(
        merged["implicit_action_reasoner"]["query_params"][0],
        np.ones((1, 2), dtype=np.float32),
    )
