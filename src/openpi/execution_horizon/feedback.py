"""One-call observation feedback residual on a frozen ordered horizon predictor."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any, Literal

import numpy as np

from openpi.execution_horizon.ordered_smdp import DEFAULT_CANDIDATES
from openpi.execution_horizon.ordered_smdp import ordered_log_probabilities

FEATURE_DIM = 547
HIDDEN_DIM = 64
PARAMETER_NAMES = ("hidden_w", "hidden_b", "output_w", "output_b")
VECTOR_FIELDS = {
    "temporal_feature": 256,
    "prefix_feature": 2048,
    "state": 32,
    "previous_prefix_feature": 2048,
    "previous_state": 32,
    "continuation_logits": 4,
}


def _vector(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape {(width,)}.")
    return array


@dataclasses.dataclass
class FeedbackSelector:
    prefix_kernel: np.ndarray
    prefix_bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    params: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.prefix_kernel = np.asarray(self.prefix_kernel, dtype=np.float32)
        self.prefix_bias = _vector(self.prefix_bias, 256, "prefix_bias")
        self.feature_mean = _vector(self.feature_mean, FEATURE_DIM, "feature_mean")
        self.feature_std = _vector(self.feature_std, FEATURE_DIM, "feature_std")
        if self.prefix_kernel.shape != (2048, 256) or not np.all(np.isfinite(self.prefix_kernel)):
            raise ValueError("prefix_kernel must be finite with shape (2048, 256).")
        if np.any(self.feature_std <= 0):
            raise ValueError("feature_std must be positive.")
        self.metadata = dict(self.metadata)
        if self.variant not in {"current", "history"}:
            raise ValueError("variant must be current or history.")
        if tuple(self.metadata["candidate_horizons"]) != DEFAULT_CANDIDATES:
            raise ValueError("Feedback selection requires H5/10/15/20/25.")
        if set(self.params) != set(PARAMETER_NAMES):
            raise ValueError(f"Feedback parameters must be exactly {PARAMETER_NAMES}.")
        shapes = {
            "hidden_w": (FEATURE_DIM, HIDDEN_DIM), "hidden_b": (HIDDEN_DIM,),
            "output_w": (HIDDEN_DIM, 4), "output_b": (4,),
        }
        for name, shape in shapes.items():
            self.params[name] = np.asarray(self.params[name], dtype=np.float32)
            if self.params[name].shape != shape or not np.all(np.isfinite(self.params[name])):
                raise ValueError(f"{name} must be finite with shape {shape}.")

    @property
    def variant(self) -> str:
        return str(self.metadata["variant"])

    @property
    def candidates(self) -> tuple[int, ...]:
        return DEFAULT_CANDIDATES

    @classmethod
    def initialize(
        cls,
        prefix_kernel: np.ndarray,
        prefix_bias: np.ndarray,
        *,
        variant: Literal["current", "history"] = "history",
        seed: int = 7,
        a_predictor_dir: str = "",
    ) -> FeedbackSelector:
        rng = np.random.default_rng(seed)
        return cls(
            prefix_kernel=prefix_kernel,
            prefix_bias=prefix_bias,
            feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
            feature_std=np.ones(FEATURE_DIM, dtype=np.float32),
            params={
                "hidden_w": rng.normal(0.0, 1.0 / np.sqrt(FEATURE_DIM), (FEATURE_DIM, HIDDEN_DIM)).astype(np.float32),
                "hidden_b": np.zeros(HIDDEN_DIM, dtype=np.float32),
                "output_w": np.zeros((HIDDEN_DIM, 4), dtype=np.float32),
                "output_b": np.zeros(4, dtype=np.float32),
            },
            metadata={
                "variant": variant, "seed": seed, "a_predictor_dir": a_predictor_dir,
                "candidate_horizons": list(DEFAULT_CANDIDATES),
                "feature_dim": FEATURE_DIM, "hidden_dim": HIDDEN_DIM,
                "history_length": 1, "a_frozen": True,
            },
        )

    @classmethod
    def initialize_from_predictor(
        cls,
        a_predictor_dir: str | pathlib.Path,
        variant: Literal["current", "history"] = "history",
        seed: int = 7,
    ) -> FeedbackSelector:
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib

        directory = pathlib.Path(a_predictor_dir).resolve()
        config = json.loads((directory / "predictor_config.json").read_text())
        if (
            config["hidden_dim"] != 256
            or tuple(config["candidate_horizons"]) != DEFAULT_CANDIDATES
            or not config.get("ordered_continuation_head", False)
            or config.get("ordered_readout", "global") != "global"
        ):
            raise ValueError("Feedback initialization requires the 256-dimensional global ordered A predictor.")
        cpu = jax.devices("cpu")[0]
        loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(
            directory / "params", dtype=jnp.float32, sharding=jax.sharding.SingleDeviceSharding(cpu)
        ))
        loaded = loaded.get("execution_horizon_predictor", loaded)
        projection = loaded["prefix_proj"]
        return cls.initialize(
            np.asarray(projection["kernel"]), np.asarray(projection["bias"]),
            variant=variant, seed=seed, a_predictor_dir=str(directory),
        )

    def build_features(self, inputs: Mapping[str, Any]) -> np.ndarray:
        values = {name: _vector(inputs[name], width, name) for name, width in VECTOR_FIELDS.items()}
        valid = bool(np.asarray(inputs["history_valid"]).item())
        previous_h = float(np.asarray(inputs["previous_h"]).item())
        elapsed_steps = float(np.asarray(inputs["elapsed_steps"]).item())
        if valid and (
            previous_h not in DEFAULT_CANDIDATES
            or not np.isfinite(elapsed_steps)
            or elapsed_steps <= 0
            or elapsed_steps != int(elapsed_steps)
        ):
            raise ValueError("Valid history requires a candidate previous_h and positive integer elapsed_steps.")
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        feature[:256] = values["temporal_feature"]
        if valid and self.variant == "history":
            prefix_pair = np.stack([values["prefix_feature"], values["previous_prefix_feature"]])
            projected = prefix_pair @ self.prefix_kernel + self.prefix_bias
            projected = projected * np.exp(-np.logaddexp(np.float32(0.0), -projected))
            feature[256:512] = projected[0] - projected[1]
            feature[512:544] = values["state"] - values["previous_state"]
            feature[544:] = [previous_h / 25.0, elapsed_steps / 25.0, 1.0]
        if not np.all(np.isfinite(feature)):
            raise ValueError("Feedback features must be finite.")
        return feature

    def fit_normalization(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not len(features):
            raise ValueError("Normalization requires a non-empty train feature matrix.")
        if not np.all(np.isfinite(features)):
            raise ValueError("Training features must be finite.")
        self.feature_mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
        self.feature_std = np.where(scale > 0, scale, np.float32(1.0))

    def forward(
        self, feature: np.ndarray, anchor_logits: np.ndarray, *, history_valid: bool
    ) -> dict[str, np.ndarray]:
        feature = _vector(feature, FEATURE_DIM, "feature")
        anchor_logits = _vector(anchor_logits, 4, "anchor_logits")
        normalized = (feature - self.feature_mean) / self.feature_std
        hidden = np.tanh(normalized @ self.params["hidden_w"] + self.params["hidden_b"])
        residual = hidden @ self.params["output_w"] + self.params["output_b"]
        logits = anchor_logits + np.float32(history_valid) * residual
        log_probabilities = ordered_log_probabilities(logits)
        return {
            "continuation_logits": logits,
            "log_probabilities": log_probabilities,
            "probabilities": np.exp(log_probabilities),
        }

    def decide(self, inputs: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        valid = bool(np.asarray(inputs["history_valid"]).item())
        prediction = self.forward(self.build_features(inputs), inputs["continuation_logits"], history_valid=valid)
        selected = self.candidates[int(np.argmax(prediction["probabilities"]))]
        anchor_h = self.candidates[int(np.argmax(ordered_log_probabilities(inputs["continuation_logits"])))]
        return selected, {
            "raw_horizon": anchor_h,
            "selector_policy": f"ordered_feedback_{self.variant}",
            "feedback_variant": self.variant,
            "feedback_history_valid": valid,
            "ordered_continuation_logits": prediction["continuation_logits"].tolist(),
            "ordered_horizon_probability": prediction["probabilities"].tolist(),
            "candidate_horizons": list(self.candidates),
        }

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle, **self.params, prefix_kernel=self.prefix_kernel, prefix_bias=self.prefix_bias,
                feature_mean=self.feature_mean, feature_std=self.feature_std,
                metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> FeedbackSelector:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                prefix_kernel=archive["prefix_kernel"], prefix_bias=archive["prefix_bias"],
                feature_mean=archive["feature_mean"], feature_std=archive["feature_std"],
                params={name: archive[name] for name in PARAMETER_NAMES},
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
