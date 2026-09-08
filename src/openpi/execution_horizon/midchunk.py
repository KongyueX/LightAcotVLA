"""Cheap proprioceptive feedback for interrupting an already generated action chunk."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


PRE_FEATURE_DIM = 441
FEATURE_DIM = 457
HIDDEN_DIM = 64
CHECK_AFTER = 5
PARAMETER_NAMES = ("hidden_w", "hidden_b", "output_w", "output_b")


def _vector(value: Any, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {(width,)}.")
    return result


def proprio_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    position = _vector(observation["robot0_eef_pos"], 3, "eef_pos")
    quaternion = _vector(observation["robot0_eef_quat"], 4, "eef_quat").astype(np.float64)
    gripper = _vector(observation["robot0_gripper_qpos"], 2, "gripper_qpos")
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] ** 2)
    axis_angle = (
        np.zeros(3) if np.isclose(denominator, 0.0)
        else quaternion[:3] * (2.0 * np.arccos(quaternion[3]) / denominator)
    )
    return np.concatenate([position, axis_angle, gripper]).astype(np.float32)


@dataclasses.dataclass
class MidchunkMonitor:
    variant: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    params: dict[str, np.ndarray]
    threshold: float = 0.0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in {"fresh", "masked"}:
            raise ValueError("Midchunk monitor variant must be fresh or masked.")
        self.feature_mean = _vector(self.feature_mean, FEATURE_DIM, "feature_mean")
        self.feature_std = _vector(self.feature_std, FEATURE_DIM, "feature_std")
        if np.any(self.feature_std <= 0) or not np.isfinite(self.threshold):
            raise ValueError("Feature scales must be positive and threshold finite.")
        shapes = {
            "hidden_w": (FEATURE_DIM, HIDDEN_DIM), "hidden_b": (HIDDEN_DIM,),
            "output_w": (HIDDEN_DIM, 1), "output_b": (1,),
        }
        if set(self.params) != set(shapes):
            raise ValueError(f"Midchunk parameters must be exactly {PARAMETER_NAMES}.")
        for name, shape in shapes.items():
            value = np.asarray(self.params[name], dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}.")
            self.params[name] = value
        self.threshold = float(self.threshold)
        self.metadata = dict(self.metadata)

    @classmethod
    def initialize(cls, variant: str = "fresh", seed: int = 7, threshold: float = 0.0) -> MidchunkMonitor:
        rng = np.random.default_rng(seed)
        return cls(
            variant=variant, feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
            feature_std=np.ones(FEATURE_DIM, dtype=np.float32), threshold=threshold,
            params={
                "hidden_w": rng.normal(0, 1 / np.sqrt(FEATURE_DIM), (FEATURE_DIM, HIDDEN_DIM)).astype(np.float32),
                "hidden_b": np.zeros(HIDDEN_DIM, dtype=np.float32),
                "output_w": np.zeros((HIDDEN_DIM, 1), dtype=np.float32),
                "output_b": np.zeros(1, dtype=np.float32),
            },
            metadata={"schema_version": 1, "feature_dim": FEATURE_DIM, "pre_feature_dim": PRE_FEATURE_DIM,
                      "hidden_dim": HIDDEN_DIM, "check_after": CHECK_AFTER, "seed": seed,
                      "input_contract": "A summary, start proprio, original chunk, planned H, clock, fresh proprio"},
        )

    def build_features(self, inputs: Mapping[str, Any]) -> np.ndarray:
        temporal = _vector(inputs["temporal_feature"], 256, "temporal_feature")
        start = _vector(inputs["start_proprio"], 8, "start_proprio")
        current = _vector(inputs["current_proprio"], 8, "current_proprio")
        actions = np.asarray(inputs["chunk_actions"], dtype=np.float32)
        if actions.shape != (25, 7) or not np.all(np.isfinite(actions)):
            raise ValueError("chunk_actions must be finite with shape (25, 7).")
        planned_h = int(inputs["planned_h"])
        progress = float(inputs["episode_progress"])
        if planned_h not in (5, 10, 15, 20, 25) or not np.isfinite(progress):
            raise ValueError("planned_h must be an A horizon and episode_progress finite.")
        feature = np.concatenate([
            temporal, start, actions.reshape(-1), np.asarray([planned_h / 25.0, progress], dtype=np.float32),
            current, current - start,
        ]).astype(np.float32)
        if self.variant == "masked":
            feature[PRE_FEATURE_DIM:] = 0
        return feature

    def fit_normalization(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not len(features):
            raise ValueError("Normalization requires a non-empty (roots,457) training matrix.")
        if not np.all(np.isfinite(features)):
            raise ValueError("Normalization input must be finite.")
        self.feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_std = np.where(scale > 0, scale, np.float32(1))

    def normalized_features(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape[-1:] != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
            raise ValueError("Monitor features must be finite with final dimension 457.")
        normalized = (feature - self.feature_mean) / self.feature_std
        if self.variant == "masked":
            normalized = normalized.copy()
            normalized[..., PRE_FEATURE_DIM:] = 0
        return normalized

    def forward(self, feature: np.ndarray) -> float:
        feature = _vector(feature, FEATURE_DIM, "feature")
        hidden = np.tanh(self.normalized_features(feature) @ self.params["hidden_w"] + self.params["hidden_b"])
        return float((hidden @ self.params["output_w"] + self.params["output_b"])[0])

    def decide(self, inputs: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        gain = self.forward(self.build_features(inputs))
        if not np.isfinite(gain):
            raise ValueError("Midchunk monitor produced a non-finite gain.")
        replan = gain > self.threshold
        return replan, {"replan_gain": gain, "threshold": self.threshold, "replan": replan, "variant": self.variant}

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "variant": self.variant, "threshold": self.threshold}
        with path.open("wb") as stream:
            np.savez_compressed(
                stream, **self.params, feature_mean=self.feature_mean, feature_std=self.feature_std,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> MidchunkMonitor:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                variant=metadata["variant"], threshold=float(metadata["threshold"]), metadata=metadata,
                feature_mean=archive["feature_mean"], feature_std=archive["feature_std"],
                params={name: archive[name] for name in PARAMETER_NAMES},
            )
