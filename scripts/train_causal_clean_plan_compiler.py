"""Phase-A oracle probe for a causal clean-plan compiler.

The probe deliberately removes every observation-side shortcut.  Its model
receives only a teacher EAR endpoint (clean or semantically intervened) and the
same final-action noise for the matched pair.  A temporal plan encoder
compresses EAR into a small latent token set; action-noise queries then
cross-attend only to those latent tokens to predict the final action endpoint.

This is an oracle/data-path probe, not a deployable observation-conditioned
policy.  Task and episode metadata are used only for episode-held-out splitting
and the same-task shuffled-EAR diagnostic.  They are never model inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import pathlib
import random
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

from openpi.action_cot import endpoint_dataset


@dataclasses.dataclass(frozen=True)
class ProbeArgs:
    dataset: tuple[str, ...]
    output_dir: str
    steps: int = 1_000
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 7
    log_interval: int = 25
    clean_loss_weight: float = 1.0
    intervention_loss_weight: float = 1.0
    response_loss_weight: float = 1.0
    plan_reconstruction_loss_weight: float = 0.25
    model_dim: int = 256
    feedforward_dim: int = 512
    plan_encoder_layers: int = 2
    attention_heads: int = 8
    latent_tokens: int = 4
    active_action_dim: int = 7
    dropout: float = 0.0
    device: str = "auto"
    amp: str = "bfloat16"
    latency_warmup: int = 20
    latency_runs: int = 100
    overwrite: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--intervention-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--plan-reconstruction-loss-weight", type=float, default=0.25)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--plan-encoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--latent-tokens", type=int, default=4)
    parser.add_argument(
        "--active-action-dim",
        type=int,
        default=7,
        help="Informative leading action dimensions; LIBERO endpoint exports use 7 of 32.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args() -> ProbeArgs:
    values = vars(_parser().parse_args())
    values["dataset"] = tuple(values["dataset"])
    return ProbeArgs(**values)


def _validate_args(args: ProbeArgs) -> None:
    for name in (
        "steps",
        "batch_size",
        "eval_batch_size",
        "warmup_steps",
        "log_interval",
        "model_dim",
        "feedforward_dim",
        "plan_encoder_layers",
        "attention_heads",
        "latent_tokens",
        "active_action_dim",
        "latency_runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.latency_warmup < 0:
        raise ValueError("--latency-warmup must be non-negative.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.learning_rate <= 0 or args.final_learning_rate < 0:
        raise ValueError("Learning rates are invalid.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate.")
    if args.gradient_clip_norm <= 0 or args.weight_decay < 0:
        raise ValueError("Gradient clipping and weight decay values are invalid.")
    if min(
        args.clean_loss_weight,
        args.intervention_loss_weight,
        args.response_loss_weight,
        args.plan_reconstruction_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.model_dim % args.attention_heads:
        raise ValueError("--model-dim must be divisible by --attention-heads.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")


def _resolve_device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if resolved == "auto":
        resolved = "cpu"
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _semantic_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    return np.asarray(arrays["intervention_valid"], dtype=np.bool_) & (
        np.asarray(arrays["intervention_ids"]) != null_id
    )


def _split_indices(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the episode-held-out split in train_acot_endpoint_distillation.py."""

    eligible = np.any(_semantic_mask(arrays), axis=-1)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 2:
        raise ValueError("Endpoint dataset has fewer than two semantic-intervention records.")

    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups[eligible_indices])
    rng = np.random.default_rng(seed)
    if unique_groups.size >= 2:
        rng.shuffle(unique_groups)
        validation_count = max(1, round(unique_groups.size * validation_fraction))
        validation_groups = unique_groups[:validation_count]
        validation_mask = eligible & np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(eligible & ~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        shuffled = eligible_indices.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(shuffled.size * validation_fraction))
        validation_count = min(validation_count, shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Train/validation split produced an empty partition.")
    return train_indices, validation_indices


class _CrossFeedForwardBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        values = queries + attended
        return values + self.feedforward(self.feedforward_norm(values))


class _ActionQueryBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_block = _CrossFeedForwardBlock(
            dimension,
            heads,
            feedforward_dim,
            dropout,
        )

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        return self.cross_block(queries + attended, latents)


class CausalCleanPlanCompiler(nn.Module):
    """EAR-only compiler with four plan latents and action-noise queries."""

    def __init__(
        self,
        *,
        action_dim: int,
        plan_horizon: int,
        action_horizon: int,
        model_dim: int,
        feedforward_dim: int,
        plan_encoder_layers: int,
        attention_heads: int,
        latent_tokens: int,
        active_action_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        if not 0 < active_action_dim <= action_dim:
            raise ValueError(f"active_action_dim must be in [1, {action_dim}].")
        self.active_action_dim = active_action_dim
        self.plan_horizon = plan_horizon
        self.action_horizon = action_horizon
        self.plan_input = nn.Linear(action_dim, model_dim)
        self.plan_positions = nn.Parameter(torch.empty(1, plan_horizon, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.plan_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=plan_encoder_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.latent_queries = nn.Parameter(torch.empty(1, latent_tokens, model_dim))
        self.latent_compiler = _CrossFeedForwardBlock(
            model_dim,
            attention_heads,
            feedforward_dim,
            dropout,
        )

        self.noise_input = nn.Linear(action_dim, model_dim)
        self.action_positions = nn.Parameter(torch.empty(1, action_horizon, model_dim))
        self.action_decoder = _ActionQueryBlock(
            model_dim,
            attention_heads,
            feedforward_dim,
            dropout,
        )
        self.action_output = nn.Linear(model_dim, action_dim)

        # This head is optimized during training to keep the four latent tokens
        # plan-complete; endpoint-only inference can skip it.
        self.reconstruction_queries = nn.Parameter(torch.empty(1, plan_horizon, model_dim))
        self.reconstruction_norm = nn.LayerNorm(model_dim)
        self.reconstruction_memory_norm = nn.LayerNorm(model_dim)
        self.reconstruction_attention = nn.MultiheadAttention(
            model_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reconstruction_output = nn.Linear(model_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.plan_positions, std=0.02)
        nn.init.trunc_normal_(self.action_positions, std=0.02)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        nn.init.trunc_normal_(self.reconstruction_queries, std=0.02)

    def compile_plan(self, plan: torch.Tensor) -> torch.Tensor:
        plan_tokens = self.plan_encoder(self.plan_input(plan) + self.plan_positions)
        queries = self.latent_queries.expand(plan.shape[0], -1, -1)
        return self.latent_compiler(queries, plan_tokens)

    def reconstruct_plan(self, latents: torch.Tensor) -> torch.Tensor:
        queries = self.reconstruction_queries.expand(latents.shape[0], -1, -1)
        normalized_latents = self.reconstruction_memory_norm(latents)
        decoded, _ = self.reconstruction_attention(
            self.reconstruction_norm(queries),
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        return self._zero_inactive(self.reconstruction_output(queries + decoded))

    def _zero_inactive(self, values: torch.Tensor) -> torch.Tensor:
        if self.active_action_dim == self.action_dim:
            return values
        return torch.cat(
            (
                values[..., : self.active_action_dim],
                torch.zeros_like(values[..., self.active_action_dim :]),
            ),
            dim=-1,
        )

    def forward(
        self,
        plan: torch.Tensor,
        action_noise: torch.Tensor,
        *,
        return_reconstruction: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latents = self.compile_plan(plan)
        action_queries = self.noise_input(action_noise) + self.action_positions
        action_endpoint = self._zero_inactive(
            self.action_output(self.action_decoder(action_queries, latents))
        )
        reconstruction = self.reconstruct_plan(latents) if return_reconstruction else None
        return action_endpoint, reconstruction


def _choose_interventions(
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> np.ndarray:
    semantic = _semantic_mask(arrays)
    selected = np.empty(row_indices.shape[0], dtype=np.int64)
    for offset, row_index in enumerate(row_indices):
        candidates = np.flatnonzero(semantic[row_index])
        if not candidates.size:
            raise RuntimeError(f"Row {row_index} has no semantic intervention after filtering.")
        selected[offset] = int(candidates[0] if deterministic else rng.choice(candidates))
    return selected


def _as_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    return torch.from_numpy(array).to(device=device, non_blocking=device.type == "cuda")


def _make_batch(
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    intervention_indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "clean_plan": _as_tensor(arrays["clean_coarse"][row_indices], device),
        "clean_actions": _as_tensor(arrays["clean_actions"][row_indices], device),
        "action_noise": _as_tensor(arrays["action_noise"][row_indices], device),
        "intervention_plan": _as_tensor(
            arrays["intervention_coarse"][row_indices, intervention_indices],
            device,
        ),
        "intervention_actions": _as_tensor(
            arrays["intervention_actions"][row_indices, intervention_indices],
            device,
        ),
    }


def _autocast(device: torch.device, amp: str) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda" and amp == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _losses(
    model: CausalCleanPlanCompiler,
    batch: dict[str, torch.Tensor],
    args: ProbeArgs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_prediction, clean_reconstruction = model(batch["clean_plan"], batch["action_noise"])
    intervention_prediction, intervention_reconstruction = model(
        batch["intervention_plan"],
        batch["action_noise"],
    )
    if clean_reconstruction is None or intervention_reconstruction is None:
        raise RuntimeError("Training requires the plan reconstruction head.")
    active = slice(0, args.active_action_dim)
    clean_mse = torch_f.mse_loss(
        clean_prediction[..., active],
        batch["clean_actions"][..., active],
    )
    clean_mse_full = torch_f.mse_loss(clean_prediction, batch["clean_actions"])
    intervention_mse = torch_f.mse_loss(
        intervention_prediction[..., active],
        batch["intervention_actions"][..., active],
    )
    intervention_mse_full = torch_f.mse_loss(
        intervention_prediction,
        batch["intervention_actions"],
    )
    predicted_response = intervention_prediction - clean_prediction
    teacher_response = batch["intervention_actions"] - batch["clean_actions"]
    response_mse = torch_f.mse_loss(
        predicted_response[..., active],
        teacher_response[..., active],
    )
    response_mse_full = torch_f.mse_loss(predicted_response, teacher_response)
    clean_reconstruction_mse = torch_f.mse_loss(
        clean_reconstruction[..., active],
        batch["clean_plan"][..., active],
    )
    clean_reconstruction_mse_full = torch_f.mse_loss(
        clean_reconstruction,
        batch["clean_plan"],
    )
    intervention_reconstruction_mse = torch_f.mse_loss(
        intervention_reconstruction[..., active],
        batch["intervention_plan"][..., active],
    )
    intervention_reconstruction_mse_full = torch_f.mse_loss(
        intervention_reconstruction,
        batch["intervention_plan"],
    )
    reconstruction_mse = 0.5 * (
        clean_reconstruction_mse + intervention_reconstruction_mse
    )
    reconstruction_mse_full = 0.5 * (
        clean_reconstruction_mse_full + intervention_reconstruction_mse_full
    )
    total = (
        args.clean_loss_weight * clean_mse
        + args.intervention_loss_weight * intervention_mse
        + args.response_loss_weight * response_mse
        + args.plan_reconstruction_loss_weight * reconstruction_mse
    )
    return total, {
        "total_loss": total,
        "clean_action_mse_active7": clean_mse,
        "clean_action_mse_full32": clean_mse_full,
        "intervention_action_mse_active7": intervention_mse,
        "intervention_action_mse_full32": intervention_mse_full,
        "response_mse_active7": response_mse,
        "response_mse_full32": response_mse_full,
        "plan_reconstruction_mse_active7": reconstruction_mse,
        "plan_reconstruction_mse_full32": reconstruction_mse_full,
        "clean_plan_reconstruction_mse_active7": clean_reconstruction_mse,
        "clean_plan_reconstruction_mse_full32": clean_reconstruction_mse_full,
        "intervention_plan_reconstruction_mse_active7": intervention_reconstruction_mse,
        "intervention_plan_reconstruction_mse_full32": intervention_reconstruction_mse_full,
    }


def _learning_rate(args: ProbeArgs, step: int) -> float:
    if step <= args.warmup_steps:
        return args.learning_rate * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * cosine


def _index_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, indices.size, batch_size):
        yield indices[start : start + batch_size]


def _predict_clean(
    model: CausalCleanPlanCompiler,
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
    active_action_dim: int,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions: list[np.ndarray] = []
    clean_active_squared_error = 0.0
    clean_active_elements = 0
    clean_full_squared_error = 0.0
    clean_full_elements = 0
    reconstruction_active_squared_error = 0.0
    reconstruction_active_elements = 0
    reconstruction_full_squared_error = 0.0
    reconstruction_full_elements = 0
    with torch.inference_mode():
        for rows in _index_batches(validation_indices, batch_size):
            plans = _as_tensor(arrays["clean_coarse"][rows], device)
            noise = _as_tensor(arrays["action_noise"][rows], device)
            targets = _as_tensor(arrays["clean_actions"][rows], device)
            with _autocast(device, amp):
                endpoint, reconstruction = model(plans, noise)
            if reconstruction is None:
                raise RuntimeError("Validation requires plan reconstruction output.")
            endpoint_float = endpoint.float()
            reconstruction_float = reconstruction.float()
            clean_active_squared_error += float(
                torch_f.mse_loss(
                    endpoint_float[..., :active_action_dim],
                    targets[..., :active_action_dim],
                    reduction="sum",
                )
            )
            clean_active_elements += targets[..., :active_action_dim].numel()
            clean_full_squared_error += float(
                torch_f.mse_loss(endpoint_float, targets, reduction="sum")
            )
            clean_full_elements += targets.numel()
            reconstruction_active_squared_error += float(
                torch_f.mse_loss(
                    reconstruction_float[..., :active_action_dim],
                    plans[..., :active_action_dim],
                    reduction="sum",
                )
            )
            reconstruction_active_elements += plans[..., :active_action_dim].numel()
            reconstruction_full_squared_error += float(
                torch_f.mse_loss(reconstruction_float, plans, reduction="sum")
            )
            reconstruction_full_elements += plans.numel()
            predictions.append(endpoint_float.cpu().numpy())
    clean_active_mse = clean_active_squared_error / clean_active_elements
    clean_full_mse = clean_full_squared_error / clean_full_elements
    return np.concatenate(predictions, axis=0), {
        "clean_action_mse_active7": clean_active_mse,
        "clean_action_rmse_active7": math.sqrt(clean_active_mse),
        "clean_action_mse_full32": clean_full_mse,
        "clean_action_rmse_full32": math.sqrt(clean_full_mse),
        "clean_plan_reconstruction_mse_active7": (
            reconstruction_active_squared_error / reconstruction_active_elements
        ),
        "clean_plan_reconstruction_mse_full32": (
            reconstruction_full_squared_error / reconstruction_full_elements
        ),
    }


def _all_semantic_pairs(
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    validation_offsets: list[int] = []
    interventions: list[int] = []
    semantic = _semantic_mask(arrays)
    for validation_offset, row_index in enumerate(validation_indices):
        for intervention_index in np.flatnonzero(semantic[row_index]):
            rows.append(int(row_index))
            validation_offsets.append(validation_offset)
            interventions.append(int(intervention_index))
    if not rows:
        raise RuntimeError("Held-out split contains no semantic intervention pairs.")
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(validation_offsets, dtype=np.int64),
        np.asarray(interventions, dtype=np.int64),
    )


def _evaluate_interventions(
    model: CausalCleanPlanCompiler,
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    clean_predictions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
    active_action_dim: int,
) -> dict[str, float | int]:
    rows, clean_offsets, intervention_indices = _all_semantic_pairs(
        arrays,
        validation_indices,
    )
    intervention_active_squared_error = 0.0
    intervention_active_elements = 0
    intervention_full_squared_error = 0.0
    intervention_full_elements = 0
    response_active_squared_error = 0.0
    response_active_elements = 0
    response_full_squared_error = 0.0
    response_full_elements = 0
    response_active_cosine_sum = 0.0
    response_full_cosine_sum = 0.0
    response_cosine_count = 0
    reconstruction_active_squared_error = 0.0
    reconstruction_active_elements = 0
    reconstruction_full_squared_error = 0.0
    reconstruction_full_elements = 0
    with torch.inference_mode():
        pair_positions = np.arange(rows.size, dtype=np.int64)
        for positions in _index_batches(pair_positions, batch_size):
            selected_rows = rows[positions]
            selected_interventions = intervention_indices[positions]
            plans = _as_tensor(
                arrays["intervention_coarse"][selected_rows, selected_interventions],
                device,
            )
            noise = _as_tensor(arrays["action_noise"][selected_rows], device)
            targets = _as_tensor(
                arrays["intervention_actions"][selected_rows, selected_interventions],
                device,
            )
            clean_targets = _as_tensor(arrays["clean_actions"][selected_rows], device)
            clean_endpoint = _as_tensor(clean_predictions[clean_offsets[positions]], device)
            with _autocast(device, amp):
                endpoint, reconstruction = model(plans, noise)
            if reconstruction is None:
                raise RuntimeError("Validation requires plan reconstruction output.")
            endpoint = endpoint.float()
            reconstruction = reconstruction.float()
            intervention_active_squared_error += float(
                torch_f.mse_loss(
                    endpoint[..., :active_action_dim],
                    targets[..., :active_action_dim],
                    reduction="sum",
                )
            )
            intervention_active_elements += targets[..., :active_action_dim].numel()
            intervention_full_squared_error += float(
                torch_f.mse_loss(endpoint, targets, reduction="sum")
            )
            intervention_full_elements += targets.numel()
            reconstruction_active_squared_error += float(
                torch_f.mse_loss(
                    reconstruction[..., :active_action_dim],
                    plans[..., :active_action_dim],
                    reduction="sum",
                )
            )
            reconstruction_active_elements += plans[..., :active_action_dim].numel()
            reconstruction_full_squared_error += float(
                torch_f.mse_loss(reconstruction, plans, reduction="sum")
            )
            reconstruction_full_elements += plans.numel()
            predicted_response = endpoint - clean_endpoint
            teacher_response = targets - clean_targets
            response_active_squared_error += float(
                torch_f.mse_loss(
                    predicted_response[..., :active_action_dim],
                    teacher_response[..., :active_action_dim],
                    reduction="sum",
                )
            )
            response_active_elements += teacher_response[..., :active_action_dim].numel()
            response_full_squared_error += float(
                torch_f.mse_loss(predicted_response, teacher_response, reduction="sum")
            )
            response_full_elements += teacher_response.numel()
            response_active_cosines = torch_f.cosine_similarity(
                predicted_response[..., :active_action_dim].flatten(1),
                teacher_response[..., :active_action_dim].flatten(1),
                dim=-1,
                eps=1e-8,
            )
            response_full_cosines = torch_f.cosine_similarity(
                predicted_response.flatten(1),
                teacher_response.flatten(1),
                dim=-1,
                eps=1e-8,
            )
            response_active_cosine_sum += float(response_active_cosines.sum())
            response_full_cosine_sum += float(response_full_cosines.sum())
            response_cosine_count += response_active_cosines.numel()
    intervention_active_mse = intervention_active_squared_error / intervention_active_elements
    intervention_full_mse = intervention_full_squared_error / intervention_full_elements
    response_active_mse = response_active_squared_error / response_active_elements
    response_full_mse = response_full_squared_error / response_full_elements
    return {
        "semantic_intervention_pairs": int(rows.size),
        "intervention_action_mse_active7": intervention_active_mse,
        "intervention_action_rmse_active7": math.sqrt(intervention_active_mse),
        "intervention_action_mse_full32": intervention_full_mse,
        "intervention_action_rmse_full32": math.sqrt(intervention_full_mse),
        "response_mse_active7": response_active_mse,
        "response_rmse_active7": math.sqrt(response_active_mse),
        "response_mse_full32": response_full_mse,
        "response_rmse_full32": math.sqrt(response_full_mse),
        "response_cosine_active7": response_active_cosine_sum / response_cosine_count,
        "response_cosine_full32": response_full_cosine_sum / response_cosine_count,
        "intervention_plan_reconstruction_mse_active7": (
            reconstruction_active_squared_error / reconstruction_active_elements
        ),
        "intervention_plan_reconstruction_mse_full32": (
            reconstruction_full_squared_error / reconstruction_full_elements
        ),
    }


def _same_task_shuffle_sources(
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tasks = np.asarray(arrays["task_id"])[validation_indices]
    episodes = np.asarray(arrays["episode_id"])[validation_indices]
    target_offsets: list[int] = []
    source_offsets: list[int] = []
    for target_offset in range(validation_indices.size):
        candidates = np.flatnonzero(
            (tasks == tasks[target_offset]) & (episodes != episodes[target_offset])
        )
        if not candidates.size:
            candidates = np.flatnonzero(tasks == tasks[target_offset])
            candidates = candidates[candidates != target_offset]
        if not candidates.size:
            continue
        target_offsets.append(target_offset)
        source_offsets.append(int(rng.choice(candidates)))
    return (
        np.asarray(target_offsets, dtype=np.int64),
        np.asarray(source_offsets, dtype=np.int64),
    )


def _evaluate_same_task_shuffle(
    model: CausalCleanPlanCompiler,
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    clean_predictions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
    seed: int,
    active_action_dim: int,
) -> dict[str, float | int | None]:
    target_offsets, source_offsets = _same_task_shuffle_sources(
        arrays,
        validation_indices,
        seed=seed,
    )
    if not target_offsets.size:
        return {
            "same_task_shuffle_records": 0,
            "shuffle_correct_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_active7": None,
            "same_task_shuffled_ear_action_mse_gap_active7": None,
            "shuffle_correct_action_mse_full32": None,
            "same_task_shuffled_ear_action_mse_full32": None,
            "same_task_shuffled_ear_action_mse_gap_full32": None,
        }
    target_rows = validation_indices[target_offsets]
    source_rows = validation_indices[source_offsets]
    target_actions = np.asarray(arrays["clean_actions"][target_rows], dtype=np.float32)
    correct_errors = np.asarray(clean_predictions[target_offsets], dtype=np.float32) - target_actions
    correct_active_squared_error = float(
        np.square(correct_errors[..., :active_action_dim].astype(np.float64)).sum()
    )
    correct_full_squared_error = float(
        np.square(correct_errors.astype(np.float64)).sum()
    )
    shuffled_active_squared_error = 0.0
    shuffled_full_squared_error = 0.0
    active_element_count = target_actions[..., :active_action_dim].size
    full_element_count = target_actions.size
    positions = np.arange(target_rows.size, dtype=np.int64)
    with torch.inference_mode():
        for selected in _index_batches(positions, batch_size):
            shuffled_plans = _as_tensor(arrays["clean_coarse"][source_rows[selected]], device)
            target_noise = _as_tensor(arrays["action_noise"][target_rows[selected]], device)
            targets = _as_tensor(arrays["clean_actions"][target_rows[selected]], device)
            with _autocast(device, amp):
                predictions, _ = model(
                    shuffled_plans,
                    target_noise,
                    return_reconstruction=False,
                )
            shuffled_active_squared_error += float(
                torch_f.mse_loss(
                    predictions.float()[..., :active_action_dim],
                    targets[..., :active_action_dim],
                    reduction="sum",
                )
            )
            shuffled_full_squared_error += float(
                torch_f.mse_loss(predictions.float(), targets, reduction="sum")
            )
    correct_active_mse = correct_active_squared_error / active_element_count
    shuffled_active_mse = shuffled_active_squared_error / active_element_count
    correct_full_mse = correct_full_squared_error / full_element_count
    shuffled_full_mse = shuffled_full_squared_error / full_element_count
    return {
        "same_task_shuffle_records": int(target_offsets.size),
        "shuffle_correct_action_mse_active7": correct_active_mse,
        "same_task_shuffled_ear_action_mse_active7": shuffled_active_mse,
        "same_task_shuffled_ear_action_mse_gap_active7": (
            shuffled_active_mse - correct_active_mse
        ),
        "shuffle_correct_action_mse_full32": correct_full_mse,
        "same_task_shuffled_ear_action_mse_full32": shuffled_full_mse,
        "same_task_shuffled_ear_action_mse_gap_full32": (
            shuffled_full_mse - correct_full_mse
        ),
    }


def _latency_metrics(
    model: CausalCleanPlanCompiler,
    arrays: dict[str, np.ndarray],
    validation_indices: np.ndarray,
    *,
    device: torch.device,
    amp: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    row = validation_indices[:1]
    plan = _as_tensor(arrays["clean_coarse"][row], device)
    noise = _as_tensor(arrays["action_noise"][row], device)

    def infer() -> None:
        with _autocast(device, amp):
            model(plan, noise, return_reconstruction=False)

    durations: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            infer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                infer()
                stop.record()
                stop.synchronize()
                durations.append(float(start.elapsed_time(stop)))
        else:
            for _ in range(runs):
                started = time.perf_counter()
                infer()
                durations.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(durations, dtype=np.float64)
    result: dict[str, Any] = {
        "latency_device": str(device),
        "latency_batch_size": 1,
        "latency_runs": runs,
        "latency_mean_ms": float(values.mean()),
        "latency_p95_ms": float(np.percentile(values, 95)),
    }
    if device.type == "cuda":
        result["gpu_latency_mean_ms"] = result["latency_mean_ms"]
        result["gpu_latency_p95_ms"] = result["latency_p95_ms"]
    else:
        result["gpu_latency_mean_ms"] = None
        result["gpu_latency_p95_ms"] = None
    return result


def _task_counts(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, int]:
    tasks, counts = np.unique(np.asarray(arrays["task_id"])[indices], return_counts=True)
    return {str(int(task)): int(count) for task, count in zip(tasks, counts, strict=True)}


def _episode_group_count(arrays: dict[str, np.ndarray], indices: np.ndarray) -> int:
    keys = np.stack(
        (
            np.asarray(arrays["task_id"])[indices],
            np.asarray(arrays["episode_id"])[indices],
        ),
        axis=-1,
    )
    return int(np.unique(keys, axis=0).shape[0])


def main(args: ProbeArgs) -> None:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = pathlib.Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    existing = [path for path in (metrics_path, summary_path, checkpoint_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Probe outputs already exist: {[str(path) for path in existing]}; "
            "choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = endpoint_dataset.load_endpoint_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(
        arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    action_dim = int(arrays["clean_actions"].shape[-1])
    plan_horizon = int(arrays["clean_coarse"].shape[1])
    action_horizon = int(arrays["clean_actions"].shape[1])
    if arrays["clean_coarse"].shape[-1] != action_dim:
        raise ValueError("EAR and final endpoints must share action_dim.")
    if arrays["action_noise"].shape[1:] != arrays["clean_actions"].shape[1:]:
        raise ValueError("action_noise and clean_actions shapes do not match.")

    model = CausalCleanPlanCompiler(
        action_dim=action_dim,
        plan_horizon=plan_horizon,
        action_horizon=action_horizon,
        model_dim=args.model_dim,
        feedforward_dim=args.feedforward_dim,
        plan_encoder_layers=args.plan_encoder_layers,
        attention_heads=args.attention_heads,
        latent_tokens=args.latent_tokens,
        active_action_dim=args.active_action_dim,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 1_000_000 <= parameter_count <= 3_000_000:
        print(
            f"WARNING: model has {parameter_count:,} parameters; Phase-A target is 1-3M.",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)
    started = time.monotonic()
    print(
        f"Initialized causal clean-plan compiler: train={train_indices.size} "
        f"validation={validation_indices.size} params={parameter_count:,} device={device}",
        flush=True,
    )

    metrics_mode = "w" if args.overwrite else "a"
    last_record: dict[str, Any] = {}
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            learning_rate = _learning_rate(args, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            rows = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            intervention_indices = _choose_interventions(
                arrays,
                rows,
                rng,
                deterministic=False,
            )
            batch = _make_batch(arrays, rows, intervention_indices, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.amp):
                total_loss, train_metrics = _losses(model, batch, args)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip_norm,
            )
            optimizer.step()

            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            if not should_log:
                continue
            validation_rows = validation_rng.choice(
                validation_indices,
                size=min(args.batch_size, validation_indices.size),
                replace=False,
            )
            validation_interventions = _choose_interventions(
                arrays,
                validation_rows,
                validation_rng,
                deterministic=True,
            )
            validation_batch = _make_batch(
                arrays,
                validation_rows,
                validation_interventions,
                device,
            )
            model.eval()
            with torch.inference_mode(), _autocast(device, args.amp):
                _, validation_metrics = _losses(model, validation_batch, args)
            last_record = {
                "phase": "train",
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": learning_rate,
                "train/gradient_norm": float(gradient_norm),
                **{
                    f"train/{name}": float(value.detach().float())
                    for name, value in train_metrics.items()
                },
                **{
                    f"validation_sample/{name}": float(value.detach().float())
                    for name, value in validation_metrics.items()
                },
            }
            metrics_file.write(json.dumps(last_record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"step={step} train_total={last_record['train/total_loss']:.6f} "
                f"val_clean7={last_record['validation_sample/clean_action_mse_active7']:.6f} "
                f"val_response7={last_record['validation_sample/response_mse_active7']:.6f}",
                flush=True,
            )

        model.eval()
        clean_predictions, clean_metrics = _predict_clean(
            model,
            arrays,
            validation_indices,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
            active_action_dim=args.active_action_dim,
        )
        intervention_metrics = _evaluate_interventions(
            model,
            arrays,
            validation_indices,
            clean_predictions,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
            active_action_dim=args.active_action_dim,
        )
        shuffle_metrics = _evaluate_same_task_shuffle(
            model,
            arrays,
            validation_indices,
            clean_predictions,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
            seed=args.seed + 2,
            active_action_dim=args.active_action_dim,
        )
        latency_metrics = _latency_metrics(
            model,
            arrays,
            validation_indices,
            device=device,
            amp=args.amp,
            warmup=args.latency_warmup,
            runs=args.latency_runs,
        )
        full_metrics = {
            **clean_metrics,
            **intervention_metrics,
            **shuffle_metrics,
            **latency_metrics,
        }
        final_record = {
            "phase": "full_validation",
            "step": args.steps,
            "elapsed_seconds": time.monotonic() - started,
            **full_metrics,
        }
        metrics_file.write(json.dumps(final_record, sort_keys=True) + "\n")
        metrics_file.flush()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": dataclasses.asdict(args),
        "model_shape": {
            "action_dim": action_dim,
            "plan_horizon": plan_horizon,
            "action_horizon": action_horizon,
        },
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "completed_steps": args.steps,
    }
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)

    summary = {
        "probe": "phase_a_causal_clean_plan_compiler_oracle",
        "dataset": list(args.dataset),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "args": dataclasses.asdict(args),
        "input_contract": {
            "model_inputs": ["teacher_ear_endpoint", "shared_action_noise"],
            "clean_and_intervention_share_action_noise": True,
            "forbidden_inputs": ["observation", "image", "state", "IAR", "task_id", "episode_id"],
            "metadata_usage": "task_id/episode_id only for split and shuffled-EAR evaluation",
            "semantic_interventions_only": True,
        },
        "architecture": {
            "plan_temporal_encoder": "TransformerEncoder",
            "latent_tokens": args.latent_tokens,
            "action_decoder": "action-noise self-attention then cross-attention to plan latents",
            "training_only_plan_reconstruction_head": True,
            "active_action_dim": args.active_action_dim,
            "inactive_output_dimensions_forced_zero": True,
            "parameter_count": parameter_count,
            "parameter_target_met": 1_000_000 <= parameter_count <= 3_000_000,
        },
        "dataset_records": int(len(arrays["dataset_index"])),
        "eligible_semantic_records": int(train_indices.size + validation_indices.size),
        "train_records": int(train_indices.size),
        "validation_records": int(validation_indices.size),
        "train_episode_groups": _episode_group_count(arrays, train_indices),
        "validation_episode_groups": _episode_group_count(arrays, validation_indices),
        "train_task_counts": _task_counts(arrays, train_indices),
        "validation_task_counts": _task_counts(arrays, validation_indices),
        "completed_steps": args.steps,
        "last_training_record": last_record,
        "full_validation_metrics": full_metrics,
        "device": str(device),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(
        "Full held-out: "
        f"clean_mse_active7={full_metrics['clean_action_mse_active7']:.6f} "
        f"response_mse_active7={full_metrics['response_mse_active7']:.6f} "
        f"response_cosine_active7={full_metrics['response_cosine_active7']:.4f} "
        f"shuffle_gap_active7={full_metrics['same_task_shuffled_ear_action_mse_gap_active7']} "
        f"gpu_latency_mean_ms={full_metrics['gpu_latency_mean_ms']} ",
        flush=True,
    )


if __name__ == "__main__":
    main(_parse_args())
