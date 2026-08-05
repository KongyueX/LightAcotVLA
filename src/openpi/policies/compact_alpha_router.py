"""Deployable state/step outcome router for one-step ACoT final inference."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


_STATE_DIM = 32
_FEATURE_DIM = _STATE_DIM + 1
_FEATURE_NAME = "state_32+decision_step_1"
_SCORE_SEMANTICS = "score>0 selects alpha=.05; otherwise alpha=0"
_TRAINING_CLASSES = {
    "no_decisive_labels_alpha0_fallback",
    "constant_positive",
    "constant_negative",
    "two_class_balanced_ridge",
}


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> float:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Compact alpha router field {name!r} must be scalar, got {value.shape}.")
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f"Compact alpha router field {name!r} must be finite.")
    return result


def _metadata(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Compact alpha router metadata {name!r} must be scalar, got {value.shape}.")
    return str(value.item())


def _vector(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    value = np.asarray(archive[name], dtype=np.float64)
    if value.shape != (_FEATURE_DIM,):
        raise ValueError(
            f"Compact alpha router field {name!r} has shape {value.shape}; "
            f"expected {(_FEATURE_DIM,)}."
        )
    if np.any(~np.isfinite(value)):
        raise ValueError(f"Compact alpha router field {name!r} contains non-finite values.")
    return value


@dataclasses.dataclass(frozen=True)
class CompactAlphaRouter:
    """Hard router between final time-warp alpha 0 and .05.

    The deployed CPU ``route`` mirrors the ridge trainer. Loading also validates
    and retains the equivalent pre-composed affine for artifact compatibility,
    while static endpoint selection preserves the original action path.
    """

    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_scale: float
    target_mean: float
    weights: np.ndarray
    raw_score_kernel: np.ndarray
    raw_score_bias: float
    ridge_lambda: float
    training_class: str

    def route(self, normalized_state: np.ndarray, absolute_decision_step: int) -> tuple[float, float]:
        state = np.asarray(normalized_state, dtype=np.float64)
        if state.shape != (_STATE_DIM,):
            raise ValueError(
                "Compact alpha router requires transformed normalized state with "
                f"shape {(_STATE_DIM,)}, got {state.shape}."
            )
        if np.any(~np.isfinite(state)):
            raise ValueError("Compact alpha router received non-finite normalized state.")
        if absolute_decision_step < 0:
            raise ValueError("absolute_decision_step must be non-negative.")

        feature = np.concatenate(
            [state, np.asarray([absolute_decision_step], dtype=np.float64)]
        )
        normalized = (feature - self.feature_mean) / self.feature_std * self.feature_scale
        score = float(normalized @ self.weights + self.target_mean)
        if not np.isfinite(score):
            raise ValueError("Compact alpha router produced a non-finite score.")
        return score, 0.05 if score > 0.0 else 0.0


def load_compact_alpha_router(path: pathlib.Path | str) -> CompactAlphaRouter:
    """Load and strictly validate a compact outcome-router NPZ."""

    router_path = pathlib.Path(path)
    if router_path.suffix != ".npz" or not router_path.is_file():
        raise FileNotFoundError(f"Compact alpha router must be an existing NPZ file: {router_path}")

    required = {
        "feature_mean",
        "feature_std",
        "feature_scale",
        "target_mean",
        "weights",
        "raw_score_kernel",
        "raw_score_bias",
        "ridge_lambda",
        "training_class",
        "feature_name",
        "score_semantics",
    }
    with np.load(router_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Compact alpha router is missing required fields: {missing}")

        feature_name = _metadata(archive, "feature_name")
        if feature_name != _FEATURE_NAME:
            raise ValueError(
                f"Compact alpha router feature_name is {feature_name!r}; expected {_FEATURE_NAME!r}."
            )
        score_semantics = _metadata(archive, "score_semantics")
        if score_semantics != _SCORE_SEMANTICS:
            raise ValueError(
                "Compact alpha router score_semantics mismatch: "
                f"{score_semantics!r} != {_SCORE_SEMANTICS!r}."
            )

        feature_mean = _vector(archive, "feature_mean")
        feature_std = _vector(archive, "feature_std")
        if np.any(feature_std <= 0.0):
            raise ValueError("Compact alpha router feature_std must be strictly positive.")
        feature_scale = _scalar(archive, "feature_scale")
        if feature_scale <= 0.0:
            raise ValueError("Compact alpha router feature_scale must be strictly positive.")
        if not np.isclose(feature_scale, 1.0 / np.sqrt(_FEATURE_DIM), rtol=1e-5, atol=1e-7):
            raise ValueError(
                "Compact alpha router feature_scale does not match the 33D trainer contract."
            )
        target_mean = _scalar(archive, "target_mean")
        weights = _vector(archive, "weights")
        ridge_lambda = _scalar(archive, "ridge_lambda")
        if ridge_lambda < 1.0:
            raise ValueError("Compact alpha router ridge_lambda must be at least 1.")
        training_class = _metadata(archive, "training_class")
        if training_class not in _TRAINING_CLASSES:
            raise ValueError(
                f"Compact alpha router training_class is unsupported: {training_class!r}."
            )

        # Verify that the redundant deployment affine is exactly the same
        # model. This catches stale/mixed NPZ fields before serving.
        raw_kernel = _vector(archive, "raw_score_kernel")
        raw_bias = _scalar(archive, "raw_score_bias")
        expected_kernel = feature_scale * weights / feature_std
        expected_bias = target_mean - float(expected_kernel @ feature_mean)
        if not np.allclose(raw_kernel, expected_kernel, rtol=1e-4, atol=1e-6):
            raise ValueError("Compact alpha router raw_score_kernel is inconsistent with ridge fields.")
        if not np.isclose(raw_bias, expected_bias, rtol=1e-4, atol=1e-5):
            raise ValueError("Compact alpha router raw_score_bias is inconsistent with ridge fields.")

    return CompactAlphaRouter(
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_scale=feature_scale,
        target_mean=target_mean,
        weights=weights,
        raw_score_kernel=raw_kernel,
        raw_score_bias=raw_bias,
        ridge_lambda=ridge_lambda,
        training_class=training_class,
    )
