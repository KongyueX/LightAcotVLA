import dataclasses
import logging
from typing import Any, Tuple, Optional, List, Sequence
import copy
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
import os
from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictor
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig
from openpi.models.pi0 import posemb_sincos, make_attn_mask
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("ACoT_VLA")


class MLP(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, activate: bool = True, rngs: nnx.Rngs, param_dtype=jnp.float32):
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs, param_dtype=param_dtype)
        self.activate = activate

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.activate:
            return self.fc3(nnx.swish(self.fc2(nnx.swish(self.fc1(x)))))
        else:
            return self.fc3(self.fc2(self.fc1(x)))

class LearnableQueryExtractor(nnx.Module):
    def __init__(self, num_queries: int, dim: int, output_dim: int, depth: int, rngs: nnx.Rngs, param_dtype = jnp.float32,
                 heads: int = 8, head_dim: int = 256, group_size: int = 3):
        """
        Args:
            num_queries: num of learnable query for each layer
            dim: input dim D
            depth: num of layers L
            heads: num of heads
            head_dim: dim of each head
            group_size: for group_size layers, we share same params of projector
        """
        self.num_queries = num_queries
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_groups = depth // group_size

        self.query_params = [
            nnx.Param(jax.random.normal(rngs.params(), (num_queries, dim)))
            for _ in range(self.depth)
        ]

        self.k_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.v_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.q_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.out_proj = [
            nnx.Linear(in_features=self.heads * self.head_dim, out_features=output_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]


    def __call__(self, K: jax.Array, V: jax.Array) -> jax.Array:
        """
        K, V: (B, L, T, D)
        Returns: (B, L, D)
        """
        B, L, T, D = K.shape
        outputs = []
        for l in range(L):
            g = l // self.group_size
            Q_l = self.query_params[l][None, :, :]  # (1,Q,D)
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]
            Q_proj = self.q_proj[g](Q_l).reshape(1, self.num_queries, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            K_proj = self.k_proj[g](K_l).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            V_proj = self.v_proj[g](V_l).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            attn = jnp.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / jnp.sqrt(self.head_dim)
            attn = nnx.softmax(attn, axis=-1)
            pooled = jnp.einsum("bhqk,bhkd->bhqd", attn, V_proj)  # (B,H,Q,Hd)

            pooled = pooled.mean(axis=2)  # (B,H,Hd)
            pooled = pooled.transpose(0, 2, 1).reshape(B, self.heads * self.head_dim)
            pooled = self.out_proj[g](pooled)  # (B,D)
            outputs.append(pooled)

        return jnp.stack(outputs, axis=1)  # (B,L,D)


class AttentionPoolingExtractor(nnx.Module):
    def __init__(self, dim: int, output_dim: int, depth: int, rngs: nnx.Rngs, param_dtype = jnp.float32,
        heads: int = 8, head_dim: int = 256, group_size: int = 3):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_groups = depth // group_size

        self.k_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.v_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.q_proj = [
            nnx.Linear(in_features=self.dim, out_features=self.heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.out_proj = [
            nnx.Linear(in_features=self.heads * self.head_dim, out_features=output_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]


    def __call__(self, K: jax.Array, V: jax.Array) -> jax.Array:
        B, L, T, D = K.shape
        outputs = []
        for l in range(L):
            g = l // self.group_size
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]
            Q_l = K_l.mean(axis=1, keepdims=True)  # (B,1,D), AttentionPooling means the query is from the Key tensor

            Q_proj = self.q_proj[g](Q_l).reshape(B, 1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            K_proj = self.k_proj[g](K_l).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            V_proj = self.v_proj[g](V_l).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            attn = jnp.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / jnp.sqrt(self.head_dim)
            attn = nnx.softmax(attn, axis=-1)
            pooled = jnp.einsum("bhqk,bhkd->bhqd", attn, V_proj)  # (B,H,1,Hd)
            # pooling for query
            pooled = pooled.transpose(0, 2, 1, 3).reshape(B, self.heads * self.head_dim)
            pooled = self.out_proj[g](pooled)
            outputs.append(pooled)

        return jnp.stack(outputs, axis=1)


class DownsampleExtractor(nnx.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        downsample_dim: int = 512,  # Attention dim after downsampling
        group_size: int = 3,
        num_queries: int = 1,
        heads: int = 8,
    ):
        
        self.dim = dim
        self.depth = depth
        self.downsample_dim = downsample_dim
        self.group_size = group_size
        self.num_groups = depth // group_size
        self.num_queries = num_queries
        self.heads = heads
        self.head_dim = downsample_dim // heads

        self.query_params = [
            nnx.Param(jax.random.normal(rngs.params(), (num_queries, dim)))
            for _ in range(self.depth)
        ]

        self.q_proj = [
            nnx.Linear(in_features=dim, out_features=self.downsample_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.k_proj = [
            nnx.Linear(in_features=dim, out_features=self.downsample_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        self.v_proj = [
            nnx.Linear(in_features=dim, out_features=self.downsample_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]
        
        self.out_proj = [
            nnx.Linear(in_features=self.downsample_dim, out_features=output_dim, rngs=rngs, param_dtype=param_dtype)
            for _ in range(self.num_groups)
        ]

    def __call__(self, K: jax.Array, V: jax.Array) -> jax.Array:
        """
        K, V: (B, L, T, D)
        Returns: (B, L, output_dim)
        """
        B, L, T, D = K.shape
        outputs = []
        
        for l in range(L):
            g = l // self.group_size
            K_l, V_l = K[:, l, :, :], V[:, l, :, :] # (B, T, D)

            # Q_l: (1, Q, D) -> (1, Q, downsample_dim) -> (1, H, Q, Hd)
            Q_l_learnable = self.query_params[l][None, :, :] # (1, num_queries, D)
            Q_proj = self.q_proj[g](Q_l_learnable)
            Q_proj = Q_proj.reshape(1, self.num_queries, self.heads, self.head_dim).transpose(0, 2, 1, 3) 
            
            # K_l: (B, T, D) -> (B, T, downsample_dim) -> (B, H, T, Hd)
            K_proj = self.k_proj[g](K_l)
            K_proj = K_proj.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3) 
            
            # V_l: (B, T, D) -> (B, T, downsample_dim) -> (B, H, T, Hd)
            V_proj = self.v_proj[g](V_l)
            V_proj = V_proj.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3) 

            Q_proj_batched = jnp.tile(Q_proj, [B, 1, 1, 1]) 
            
            attn = jnp.einsum("bhqd,bhkd->bhqk", Q_proj_batched, K_proj) / jnp.sqrt(self.head_dim)
            attn = nnx.softmax(attn, axis=-1)
            
            pooled = jnp.einsum("bhqk,bhkd->bhqd", attn, V_proj) 
            pooled = pooled.mean(axis=2) if self.num_queries > 1 else pooled.squeeze(axis=2)
            
            pooled = pooled.transpose(0, 2, 1).reshape(B, self.downsample_dim)
            feat = self.out_proj[g](pooled) # (B, out_dim)
            outputs.append(feat)

        return jnp.stack(outputs, axis=1) # (B, L, out_dim)


class UnifiedAttentionModule(nnx.Module):
    def __init__(self, in_dim_1: int, in_dim_2: int, out_dim: int, apply_sigmoid: bool, hidden_dim: int = 128, num_heads: int = 4, *, rngs: nnx.Rngs, param_dtype=jnp.float32):

        self.q_proj = nnx.Linear(in_features=in_dim_1, out_features=hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.kv_proj = nnx.Linear(in_features=in_dim_2, out_features=hidden_dim * 2, rngs=rngs, param_dtype=param_dtype)
        self.attn = nnx.MultiHeadAttention(in_features=hidden_dim, num_heads=num_heads, rngs=rngs, param_dtype=param_dtype)
        self.fc_out = nnx.Linear(in_features=hidden_dim, out_features=out_dim, rngs=rngs, param_dtype=param_dtype)
        self.apply_sigmoid = apply_sigmoid

    def __call__(self, feat_1: jnp.ndarray, feat_2: jnp.ndarray, decode: bool = False) -> jnp.ndarray:
        
        Q = self.q_proj(feat_1)
        KV = self.kv_proj(feat_2)
        K, V = jnp.split(KV, 2, axis=-1)
        
        attn_out = self.attn(Q, K, V, decode=decode)
        output = self.fc_out(attn_out)

        # Apply sigmoid activation if allowed
        if self.apply_sigmoid:
            return nnx.sigmoid(output)
        return output


@dataclasses.dataclass(frozen=True)
class ACOTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    coarse_action_expert_variant: _gemma.Variant = "gemma_300m"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    coarse_action_horizon: int = 50
    action_horizon: int = 30
    max_token_len: int = None  # type: ignore
    pi05: bool = True  # type: ignore

    discrete_state_input: bool = None  # type: ignore

    adopt_explicit_action_reasoner: bool = False  # type: ignore
    adopt_implicit_action_reasoner: bool = False  # type: ignore

    query_based_implicit_extractor: bool = False  # type: ignore
    attention_pooling_implicit_extractor: bool = False  # type: ignore
    downsample_based_implicit_extractor: bool = False  # type: ignore

    action_cot_max_segments: int = 0
    action_cot_skip_loss_weight: float = 0.0
    action_cot_step_values: tuple[int, ...] = (3, 5, 10)
    action_cot_step_loss_weight: float = 0.0
    action_cot_dynamic_steps: bool = False

    # Execution-horizon prediction is a separate problem from selecting the
    # number of Action-CoT denoising iterations above.  It therefore has its
    # own module and checkpoint sidecar.
    execution_horizon_predictor: bool = False
    execution_horizon_hidden_dim: int = 256
    execution_horizon_temporal_layers: int = 3

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.action_cot_skip_loss_weight > 0 and self.action_cot_max_segments <= 0:
            raise ValueError("action_cot_max_segments must be positive when action_cot_skip_loss_weight > 0.")
        if any(value <= 0 for value in self.action_cot_step_values):
            raise ValueError(f"action_cot_step_values must be positive, got {self.action_cot_step_values}.")
        if (self.action_cot_step_loss_weight > 0 or self.action_cot_dynamic_steps) and len(self.action_cot_step_values) < 2:
            raise ValueError("action_cot_step_values must contain at least two values when step head is enabled.")
        if self.execution_horizon_predictor and self.action_horizon != 10:
            raise ValueError("execution_horizon_predictor currently requires action_horizon=10.")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if not self.pi05:
            return _model.ModelType.ACOT_VLA_PI0
        else:
            return _model.ModelType.ACOT_VLA_PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACOT_VLA":
        return ACOT_VLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec


    def get_freeze_filter(self, freeze_llm = False, freeze_llm_embedder = True, freeze_vision = False, freeze_dual_ae = [False, False]) -> nnx.filterlib.Filter:
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        paligemma_base_filter = nnx_utils.PathRegex(".*llm(?!.*_1|.*_2).*") 
        coarse_action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_2.*")
        embedder_filter = nnx_utils.PathRegex(".*llm.*embed.*|.*llm.*embedding.*")
        lora_filter = nnx_utils.PathRegex(".*lora.*")

        freeze_paths = []
        
        if freeze_vision:
            freeze_paths.append(nnx_utils.PathRegex(".*img.*"))
            
        if freeze_llm:
            freeze_paths.append(paligemma_base_filter)
            
        if freeze_dual_ae[0]:
            freeze_paths.append(coarse_action_expert_params_filter)
        if freeze_dual_ae[1]:
            freeze_paths.append(action_expert_params_filter)

        if not freeze_paths:
            return nnx.Nothing

        base_freeze_filter = nnx.Any(*freeze_paths)
        keep_alive_paths = []

        has_lora = "lora" in self.paligemma_variant or "lora" in self.action_expert_variant
        if has_lora:
            keep_alive_paths.append(lora_filter)

        if freeze_llm and not freeze_llm_embedder:
            keep_alive_paths.append(embedder_filter)

        if not keep_alive_paths:
            return base_freeze_filter
        else:
            return nnx.All(
                base_freeze_filter,
                nnx.Not(nnx.Any(*keep_alive_paths))
            )


class ACOT_VLA(_model.BaseModel):
    def __init__(self, config: ACOTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        coarse_action_expert_config = _gemma.get_config(config.coarse_action_expert_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, coarse_action_expert_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=self.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True] if self.pi05 else [False, False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.coarse_action_in_proj = nnx.Linear(config.action_dim, coarse_action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if self.pi05:
            self.coarse_time_mlp_in = nnx.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width, rngs=rngs)
            self.coarse_time_mlp_out = nnx.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width, rngs=rngs)
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.coarse_action_time_mlp_in = nnx.Linear(2 * coarse_action_expert_config.width, coarse_action_expert_config.width, rngs=rngs)
            self.coarse_action_time_mlp_out = nnx.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.coarse_action_out_proj = nnx.Linear(coarse_action_expert_config.width, config.action_dim, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.action_cot_max_segments = config.action_cot_max_segments
        self.action_cot_skip_loss_weight = config.action_cot_skip_loss_weight
        self.action_cot_step_values = tuple(config.action_cot_step_values)
        self.action_cot_step_loss_weight = config.action_cot_step_loss_weight
        self.action_cot_dynamic_steps = config.action_cot_dynamic_steps
        if self.action_cot_max_segments > 0:
            self.action_cot_skip_head = nnx.Linear(
                paligemma_config.width,
                self.action_cot_max_segments,
                rngs=rngs,
            )
        if self.action_cot_step_loss_weight > 0 or self.action_cot_dynamic_steps:
            self.action_cot_step_head = nnx.Linear(
                paligemma_config.width,
                len(self.action_cot_step_values),
                rngs=rngs,
            )
        self.execution_horizon_predictor_enabled = config.execution_horizon_predictor
        if self.execution_horizon_predictor_enabled:
            self.execution_horizon_predictor = ExecutionHorizonPredictor(
                ExecutionHorizonPredictorConfig(
                    prefix_feature_dim=paligemma_config.width,
                    state_dim=config.action_dim,
                    action_dim=config.action_dim,
                    coarse_horizon=config.coarse_action_horizon,
                    action_horizon=config.action_horizon,
                    hidden_dim=config.execution_horizon_hidden_dim,
                    temporal_layers=config.execution_horizon_temporal_layers,
                ),
                rngs=rngs,
            )
        
        self.adopt_explicit_action_reasoner = config.adopt_explicit_action_reasoner
        if self.adopt_explicit_action_reasoner:
            self.explicit_action_reasoner = UnifiedAttentionModule(
                in_dim_1=action_expert_config.width,
                in_dim_2=coarse_action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                rngs=rngs,
            )

        self.adopt_implicit_action_reasoner = config.adopt_implicit_action_reasoner
        self.query_based_implicit_extractor = config.query_based_implicit_extractor
        self.attention_pooling_implicit_extractor = config.attention_pooling_implicit_extractor
        self.downsample_based_implicit_extractor = config.downsample_based_implicit_extractor

        if self.adopt_implicit_action_reasoner:

            if self.query_based_implicit_extractor:
                self.implicit_action_reasoner = LearnableQueryExtractor(
                    num_queries=8,
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    rngs=rngs,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                    group_size=3
                )
            elif self.attention_pooling_implicit_extractor:
                self.implicit_action_reasoner = AttentionPoolingExtractor(
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    rngs=rngs,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                    group_size=3
                )
            elif self.downsample_based_implicit_extractor:
                self.implicit_action_reasoner = DownsampleExtractor(
                    num_queries=1,
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    rngs=rngs,
                    downsample_dim=paligemma_config.head_dim // 2,
                    heads=paligemma_config.num_heads,
                    group_size=3
                )
            else:
                raise ValueError("At least one extractor type must be selected when adopt_implicit_action_reasoner is True.")
            self.implicit_action_reasoner_interact = UnifiedAttentionModule(
                in_dim_1=action_expert_config.width,
                in_dim_2=action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                rngs=rngs,
            )

        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            self.explicit_action_reason_proj = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.implicit_action_reason_proj = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_reasoning_fusion = UnifiedAttentionModule(
                in_dim_1=2 * action_expert_config.width,
                in_dim_2=2 * action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                rngs=rngs,
            )

        elif self.adopt_explicit_action_reasoner or self.adopt_implicit_action_reasoner:
            self.action_reasoning_fusion = MLP(
                input_dim=2 * action_expert_config.width,
                hidden_dim=action_expert_config.width,
                output_dim=action_expert_config.width,
                activate=False,
                rngs=rngs,
            )

        else:
            self.action_reasoning_proj = None

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
        self.coarse_action_horizon = config.coarse_action_horizon


    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation,
        noisy_actions,
        timestep: at.Float[at.Array, " b"],
        explicit_action_reason: Optional[jax.Array] = None,
        implicit_action_reason: Optional[jax.Array] = None,
        suf_type = "reasoner"
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            # state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        if suf_type == "reasoner":
            action_tokens = self.coarse_action_in_proj(noisy_actions)
            time_emb = posemb_sincos(timestep, self.coarse_action_in_proj.out_features, min_period=4e-3, max_period=4.0)

            if self.pi05:
                # time MLP (for adaRMS)
                time_emb = self.coarse_time_mlp_in(time_emb)
                time_emb = nnx.swish(time_emb)
                time_emb = self.coarse_time_mlp_out(time_emb)
                time_emb = nnx.swish(time_emb)
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
            else:
                # mix timestep + action information using an MLP (no adaRMS)
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
                action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
                action_time_tokens = self.coarse_action_time_mlp_in(action_time_tokens)
                action_time_tokens = nnx.swish(action_time_tokens)
                action_time_tokens = self.coarse_action_time_mlp_out(action_time_tokens)
                action_expert_tokens = action_time_tokens
                adarms_cond = None

        elif suf_type == "expert":
            action_tokens = self.action_in_proj(noisy_actions)
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

            if self.pi05:
                # time MLP (for adaRMS)
                time_emb = self.time_mlp_in(time_emb)
                time_emb = nnx.swish(time_emb)
                time_emb = self.time_mlp_out(time_emb)
                time_emb = nnx.swish(time_emb)
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
            else:
                # mix timestep + action information using an MLP (no adaRMS)
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
                action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
                action_time_tokens = self.action_time_mlp_in(action_time_tokens)
                action_time_tokens = nnx.swish(action_time_tokens)
                action_time_tokens = self.action_time_mlp_out(action_time_tokens)
                action_expert_tokens = action_time_tokens
                adarms_cond = None

            if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
                # explicit action reasoner, explicit_action_reason is coarse-grained traj, we encode it to get z^{ex} in the paper
                explicit_action_reason_tokens = self.coarse_action_in_proj(explicit_action_reason)
                # cross attention to get s^{ex} representations in the paper
                aligned_explicit_action_reason_tokens = self.explicit_action_reasoner(
                    action_expert_tokens,
                    explicit_action_reason_tokens
                )
                
                # implicit_action_reason are already tokens, which is z^{im} wirtten in the paper
                implicit_action_reason_tokens = implicit_action_reason
                # cross attention to get s^{im} representations in the paper
                aligned_implicit_action_reason_tokens = self.implicit_action_reasoner_interact(
                    action_expert_tokens,
                    implicit_action_reason_tokens
                )

                # we project s^{ex} and s^{im} for dimension transform, but we did not explitly write it in the paper, we are sorry.
                action_expert_tokens_explicit = jnp.concatenate([action_expert_tokens, aligned_explicit_action_reason_tokens], axis = -1)
                action_expert_tokens_explicit = self.explicit_action_reason_proj(action_expert_tokens_explicit)

                action_expert_tokens_implicit = jnp.concatenate([action_expert_tokens, aligned_implicit_action_reason_tokens], axis = -1)
                action_expert_tokens_implicit = self.implicit_action_reason_proj(action_expert_tokens_implicit)

                # concatenate and self attention fusion 
                action_expert_tokens = jnp.concatenate([action_expert_tokens_explicit, action_expert_tokens_implicit], axis = -1)
                action_expert_tokens = self.action_reasoning_fusion(action_expert_tokens, action_expert_tokens)


            elif self.adopt_explicit_action_reasoner:
                explicit_action_reason_tokens = self.coarse_action_in_proj(explicit_action_reason)
                aligned_explicit_action_reason_tokens = self.explicit_action_reasoner(
                    action_expert_tokens,
                    explicit_action_reason_tokens
                )
                # concatenate and fusion 
                action_expert_tokens = jnp.concatenate([action_expert_tokens, aligned_explicit_action_reason_tokens], axis = -1)
                action_expert_tokens = self.action_reasoning_fusion(action_expert_tokens)

            elif self.adopt_implicit_action_reasoner:
                implicit_action_reason_tokens = implicit_action_reason
                aligned_implicit_action_reason_tokens = self.implicit_action_reasoner_interact(
                    action_expert_tokens,
                    implicit_action_reason_tokens
                )
                # concatenate and fusion 
                action_expert_tokens = jnp.concatenate([action_expert_tokens, aligned_implicit_action_reason_tokens], axis = -1)
                action_expert_tokens = self.action_reasoning_fusion(action_expert_tokens)
            
            else:
                # keep vanilla
                pass

        else:
            raise ValueError(f"Unknown suffix type: {suf_type}")

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        # image/language/state inputs do not attend to action tokens
        if suf_type == "reasoner":
            ar_mask += [True] + ([False] * (action_expert_tokens.shape[1] - 1))
        elif suf_type == "expert":
            ar_mask += [True] + ([False] * (action_expert_tokens.shape[1] - 1))
        else:
            raise ValueError(f"Unknown suffix type: {suf_type}")

        # ar_mask += [True] + ([False] * (action_expert_tokens.shape[1] - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    def _pool_prefix(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
    ) -> jax.Array:
        mask = prefix_mask.astype(prefix_out.dtype)
        return jnp.sum(prefix_out * mask[..., None], axis=1) / jnp.maximum(
            jnp.sum(mask, axis=1, keepdims=True),
            1.0,
        )

    def _action_cot_skip_logits(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
    ) -> jax.Array:
        pooled = self._pool_prefix(prefix_out, prefix_mask)
        return self.action_cot_skip_head(pooled)

    def _action_cot_step_logits(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
    ) -> jax.Array:
        pooled = self._pool_prefix(prefix_out, prefix_mask)
        return self.action_cot_step_head(pooled)

    def _action_cot_skip_loss(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        skip_mask: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        logits = self._action_cot_skip_logits(prefix_out, prefix_mask)
        labels = jnp.asarray(skip_mask, dtype=logits.dtype)
        valid = jnp.asarray(valid_mask, dtype=logits.dtype)
        labels = labels[..., : self.action_cot_max_segments]
        valid = valid[..., : self.action_cot_max_segments]

        bce = jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.sum(bce * valid) / jnp.maximum(jnp.sum(valid), 1.0)

    def _action_cot_step_loss(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        step_label: jax.Array,
    ) -> jax.Array:
        logits = self._action_cot_step_logits(prefix_out, prefix_mask)
        labels = jnp.reshape(jnp.asarray(step_label, dtype=jnp.int32), (-1,))
        labels = jnp.clip(labels, 0, logits.shape[-1] - 1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
        return jnp.mean(nll)

    def _fixed_l5_keep_indices(self, skip_segment: jax.Array) -> jax.Array:
        skip_segment = jnp.reshape(jnp.asarray(skip_segment, dtype=jnp.int32), ())
        skip_segment = jnp.clip(skip_segment, 0, 2)
        keep_indices = jnp.asarray(
            [
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=jnp.int32,
        )
        return keep_indices[skip_segment]

    def _restore_fixed_l5_skip(self, kept_actions: jax.Array, skip_segment: jax.Array) -> jax.Array:
        skip_segment = jnp.reshape(jnp.asarray(skip_segment, dtype=jnp.int32), ())
        skip_segment = jnp.clip(skip_segment, 0, 2)

        def skip_first(actions):
            prefix = jnp.repeat(actions[:, :1, :], 5, axis=1)
            return jnp.concatenate([prefix, actions], axis=1)

        def skip_middle(actions):
            left = actions[:, 4:5, :]
            right = actions[:, 5:6, :]
            alpha = jnp.arange(1, 6, dtype=actions.dtype).reshape(1, 5, 1) / 6.0
            middle = (1.0 - alpha) * left + alpha * right
            return jnp.concatenate([actions[:, :5, :], middle, actions[:, 5:, :]], axis=1)

        def skip_last(actions):
            suffix = jnp.repeat(actions[:, -1:, :], 5, axis=1)
            return jnp.concatenate([actions, suffix], axis=1)

        return jax.lax.switch(skip_segment, (skip_first, skip_middle, skip_last), kept_actions)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        action_cot_skip_mask=None,
        action_cot_skip_valid_mask=None,
        action_cot_step_label=None,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:

        # preprocess_rng, _, time_rng, coarse_action_noise_rng, _, expert_action_noise_rng = jax.random.split(rng, 6)
        preprocess_rng, time_rng, coarse_action_noise_rng, expert_action_noise_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]

        coarse_action_noise = jax.random.normal(coarse_action_noise_rng, coarse_actions.shape)
        expert_action_noise = jax.random.normal(expert_action_noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        # use coarse actions as explicit action reasoning
        x_ref_t = time_expanded * coarse_action_noise + (1.0 - time_expanded) * coarse_actions
        u_ref_t = coarse_action_noise - coarse_actions

        x_expert_t = time_expanded * expert_action_noise + (1.0 - time_expanded) * actions
        u_expert_t = expert_action_noise - actions

        # forward to get kv cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )

        if self.adopt_explicit_action_reasoner:
            # suffix forward to get explicit action reference
            suffix_ref_action_tokens, suffix_ref_action_mask, suffix_ref_action_ar_mask, adarms_ref_action_cond = self.embed_suffix(observation, x_ref_t, time, suf_type = "reasoner")

            input_mask = jnp.concatenate([prefix_mask, suffix_ref_action_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ref_action_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (prefix_ref_action_out, suffix_ref_action_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_ref_action_tokens, None],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_ref_action_cond, None],
            )
            # teacher forcing
            explicit_action_reason = coarse_actions

        else:
            explicit_action_reason = None

        if self.adopt_implicit_action_reasoner:
            K_all, V_all = kv_cache
            K_rearranged = einops.rearrange(K_all, 'L B T 1 D -> B L T D')
            V_rearranged = einops.rearrange(V_all, 'L B T 1 D -> B L T D')
            # implicit action reasoner
            implicit_action_reason = self.implicit_action_reasoner(K_rearranged, V_rearranged)
        else:
            implicit_action_reason = None
        
        # suffix forward to get action prediction
        suffix_expert_tokens, suffix_expert_mask, suffix_expert_ar_mask, adarms_expert_cond = self.embed_suffix(
            observation, x_expert_t, time,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suf_type = "expert"
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_expert_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_expert_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_expert_out, _, suffix_expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suffix_expert_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_expert_cond],
        )


        if self.adopt_explicit_action_reasoner:
            # trainer explicit action reasoner using flow matching
            v_ref_t = self.coarse_action_out_proj(suffix_ref_action_out[:, -self.coarse_action_horizon :])
            v_expert_t = self.action_out_proj(suffix_expert_out[:, -self.action_horizon :])

            action_diff_ref = u_ref_t - v_ref_t
            action_diff_expert = u_expert_t - v_expert_t
            # Since we set the balance factor as 0.5, the following loss is equal.
            flow_loss = jnp.mean(jnp.square(action_diff_ref)) + jnp.mean(jnp.square(action_diff_expert))

        else:
            v_expert_t = self.action_out_proj(suffix_expert_out[:, -self.action_horizon :])
            action_diff_expert = u_expert_t - v_expert_t
            flow_loss = jnp.mean(jnp.square(action_diff_expert))

        total_loss = flow_loss
        if (
            self.action_cot_skip_loss_weight > 0
            and action_cot_skip_mask is not None
            and action_cot_skip_valid_mask is not None
        ):
            skip_loss = self._action_cot_skip_loss(
                prefix_out,
                prefix_mask,
                action_cot_skip_mask,
                action_cot_skip_valid_mask,
            )
            total_loss = total_loss + self.action_cot_skip_loss_weight * skip_loss

        if self.action_cot_step_loss_weight > 0 and action_cot_step_label is not None:
            step_loss = self._action_cot_step_loss(prefix_out, prefix_mask, action_cot_step_label)
            total_loss = total_loss + self.action_cot_step_loss_weight * step_loss

        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        action_cot_denoising_steps: int | at.Int[at.Array, ""] | None = None,
        dynamic_denoising_steps: bool | None = None,
        explicit_action_reason_override: _model.CoarseActions | None = None,
        explicit_action_skip_segment: int | at.Int[at.Array, ""] | None = None,
    ) -> _model.Actions:
        prefix_state = self.sample_actions_profile_prefix(rng, observation)
        implicit_outputs = self.sample_actions_profile_implicit(prefix_state)
        coarse_outputs = self.sample_actions_profile_coarse(
            prefix_state,
            num_steps=num_steps,
            action_cot_denoising_steps=action_cot_denoising_steps,
            dynamic_denoising_steps=dynamic_denoising_steps,
            explicit_action_reason_override=explicit_action_reason_override,
            explicit_action_skip_segment=explicit_action_skip_segment,
        )
        expert_outputs = self.sample_actions_profile_expert(
            prefix_state,
            coarse_outputs["explicit_action_reason"],
            implicit_outputs["implicit_action_reason"],
            num_steps=num_steps,
        )
        if self.adopt_explicit_action_reasoner:
            result = {
                "actions": expert_outputs["actions"],
                "coarse_actions": coarse_outputs["explicit_action_reason"],
                "action_cot_denoising_steps": coarse_outputs["action_cot_denoising_steps"],
            }
            if self.execution_horizon_predictor_enabled:
                result["execution_horizon_prefix_feature"] = jnp.asarray(
                    self._pool_prefix(prefix_state["prefix_out"], prefix_state["prefix_mask"]),
                    dtype=jnp.float32,
                )
            return result
        return expert_outputs

    def _compute_prefix_state(self, observation: _model.Observation) -> dict[str, Any]:
        """Run preprocessing and the VLM prefix exactly once."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        return {
            "observation": observation,
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "prefix_ar_mask": prefix_ar_mask,
            "prefix_out": prefix_out,
            "kv_cache": kv_cache,
        }

    def sample_actions_profile_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> dict[str, Any]:
        prefix_state = self._compute_prefix_state(observation)
        observation = prefix_state["observation"]
        batch_size = observation.state.shape[0]

        ref_action_rng, expert_action_rng = jax.random.split(rng, 2)
        ref_action_noise = jax.random.normal(ref_action_rng, (batch_size, self.coarse_action_horizon, self.action_dim))
        expert_action_noise = jax.random.normal(expert_action_rng, (batch_size, self.action_horizon, self.action_dim))

        result = {
            **prefix_state,
            "ref_action_noise": ref_action_noise,
            "expert_action_noise": expert_action_noise,
        }
        if self.execution_horizon_predictor_enabled:
            result["execution_horizon_prefix_feature"] = jnp.asarray(
                self._pool_prefix(prefix_state["prefix_out"], prefix_state["prefix_mask"]),
                dtype=jnp.float32,
            )
        return result

    def sample_actions_batched_mc(
        self,
        rngs: jax.Array,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        action_cot_denoising_steps: int | at.Int[at.Array, ""] | None = 10,
    ) -> dict[str, jax.Array]:
        """Sample K complete action chunks while sharing one VLM/prefix pass.

        ``rngs`` contains the exact per-sample policy keys.  Consequently a
        batched request with seeds ``s..s+K-1`` has the same flow-noise seeds
        as K legacy requests, without repeating image/language encoding.
        """
        prefix_state = self._compute_prefix_state(observation)
        if prefix_state["observation"].state.shape[0] != 1:
            raise ValueError("Batched MC teacher currently expects one decision observation per request.")
        sample_count = rngs.shape[0]
        split_rngs = jax.vmap(lambda key: jax.random.split(key, 2))(rngs)
        ref_action_noise = jax.vmap(
            lambda key: jax.random.normal(
                key, (self.coarse_action_horizon, self.action_dim), dtype=jnp.float32
            )
        )(split_rngs[:, 0])
        expert_action_noise = jax.vmap(
            lambda key: jax.random.normal(
                key, (self.action_horizon, self.action_dim), dtype=jnp.float32
            )
        )(split_rngs[:, 1])

        def repeat_batch(value: jax.Array) -> jax.Array:
            return jnp.repeat(value, sample_count, axis=0)

        repeated_observation = jax.tree.map(repeat_batch, prefix_state["observation"])
        cache_k, cache_v = prefix_state["kv_cache"]
        batched_prefix_state = {
            "observation": repeated_observation,
            "prefix_tokens": repeat_batch(prefix_state["prefix_tokens"]),
            "prefix_mask": repeat_batch(prefix_state["prefix_mask"]),
            "prefix_ar_mask": prefix_state["prefix_ar_mask"],
            "prefix_out": repeat_batch(prefix_state["prefix_out"]),
            "kv_cache": (
                jnp.repeat(cache_k, sample_count, axis=1),
                jnp.repeat(cache_v, sample_count, axis=1),
            ),
            "ref_action_noise": ref_action_noise,
            "expert_action_noise": expert_action_noise,
        }
        implicit_outputs = self.sample_actions_profile_implicit(prefix_state)
        implicit_action_reason = implicit_outputs["implicit_action_reason"]
        if implicit_action_reason is not None:
            implicit_action_reason = repeat_batch(implicit_action_reason)
        coarse_outputs = self.sample_actions_profile_coarse(
            batched_prefix_state,
            num_steps=num_steps,
            action_cot_denoising_steps=action_cot_denoising_steps,
            dynamic_denoising_steps=False,
        )
        expert_outputs = self.sample_actions_profile_expert(
            batched_prefix_state,
            coarse_outputs["explicit_action_reason"],
            implicit_action_reason,
            num_steps=num_steps,
        )
        all_actions = expert_outputs["actions"]
        all_coarse_actions = coarse_outputs["explicit_action_reason"]
        return {
            # Keep a leading observation batch for Policy.infer's unbatching.
            "actions": all_actions[:1],
            "coarse_actions": all_coarse_actions[:1],
            "action_cot_denoising_steps": coarse_outputs["action_cot_denoising_steps"][:1],
            "mc_actions_normalized": all_actions[None, ...],
            "mc_coarse_actions_normalized": all_coarse_actions[None, ...],
            "execution_horizon_prefix_feature": jnp.asarray(
                self._pool_prefix(prefix_state["prefix_out"], prefix_state["prefix_mask"]),
                dtype=jnp.float32,
            ),
        }

    def predict_execution_horizon(
        self,
        *,
        prefix_feature: jax.Array,
        proprioception: jax.Array,
        coarse_actions: jax.Array,
        final_actions: jax.Array,
        previous_actions: jax.Array,
        previous_h: jax.Array,
        budget_balance: jax.Array,
        episode_progress: jax.Array,
        previous_valid: jax.Array,
    ) -> dict[str, jax.Array]:
        if not self.execution_horizon_predictor_enabled:
            raise ValueError("This model was created without execution_horizon_predictor=True.")
        return self.execution_horizon_predictor(
            prefix_feature=prefix_feature,
            state=proprioception,
            coarse_actions=coarse_actions,
            final_actions=final_actions,
            previous_actions=previous_actions,
            previous_h=previous_h,
            budget_balance=budget_balance,
            episode_progress=episode_progress,
            previous_valid=previous_valid,
        )

    def sample_actions_profile_implicit(self, prefix_state: dict[str, Any]) -> dict[str, Any]:
        if self.adopt_implicit_action_reasoner:
            K_all, V_all = prefix_state["kv_cache"]
            K_rearranged = einops.rearrange(K_all, 'L B T 1 D -> B L T D')
            V_rearranged = einops.rearrange(V_all, 'L B T 1 D -> B L T D')
            implicit_action_reason = self.implicit_action_reasoner(K_rearranged, V_rearranged)
        else:
            implicit_action_reason = None
        return {"implicit_action_reason": implicit_action_reason}

    def sample_actions_profile_coarse(
        self,
        prefix_state: dict[str, Any],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        action_cot_denoising_steps: int | at.Int[at.Array, ""] | None = None,
        dynamic_denoising_steps: bool | None = None,
        explicit_action_reason_override: _model.CoarseActions | None = None,
        explicit_action_skip_segment: int | at.Int[at.Array, ""] | None = None,
    ) -> dict[str, Any]:
        observation = prefix_state["observation"]
        prefix_tokens = prefix_state["prefix_tokens"]
        prefix_mask = prefix_state["prefix_mask"]
        prefix_out = prefix_state["prefix_out"]
        kv_cache = prefix_state["kv_cache"]
        ref_action_noise = prefix_state["ref_action_noise"]
        batch_size = observation.state.shape[0]

        use_dynamic_denoising_steps = (
            self.action_cot_dynamic_steps if dynamic_denoising_steps is None else dynamic_denoising_steps
        )
        if action_cot_denoising_steps is None and use_dynamic_denoising_steps:
            logits = self._action_cot_step_logits(prefix_out, prefix_mask)
            step_values = jnp.asarray(self.action_cot_step_values, dtype=jnp.int32)
            action_cot_denoising_steps = step_values[jnp.argmax(logits, axis=-1)[0]]
        if action_cot_denoising_steps is None:
            action_cot_denoising_steps = num_steps
        action_cot_denoising_steps = jnp.maximum(jnp.asarray(action_cot_denoising_steps, dtype=jnp.float32), 1.0)
        denoising_dt = -1.0 / action_cot_denoising_steps

        def step_explicit_action_reasoner(carry):
            x_t, time, step_idx = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), suf_type = "reasoner"
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond, None],
            )
            v_t = self.coarse_action_out_proj(suffix_out[:, -x_t.shape[1] :])

            return x_t + denoising_dt * v_t, time + denoising_dt, step_idx + 1

        def cond_explicit_action_reasoner(carry):
            x_t, time, _ = carry
            return time >= -denoising_dt / 2

        if explicit_action_reason_override is not None:
            if not self.adopt_explicit_action_reasoner:
                raise ValueError("explicit_action_reason_override requires adopt_explicit_action_reasoner=True.")
            if explicit_action_skip_segment is not None:
                raise ValueError(
                    "explicit_action_skip_segment cannot be combined with explicit_action_reason_override."
                )
            explicit_action_reason = explicit_action_reason_override
        elif self.adopt_explicit_action_reasoner:
            if explicit_action_skip_segment is not None:
                if self.coarse_action_horizon != 15:
                    raise ValueError("explicit_action_skip_segment currently supports coarse_action_horizon=15 only.")
                keep_indices = self._fixed_l5_keep_indices(explicit_action_skip_segment)
                ref_action_noise_kept = jnp.take(ref_action_noise, keep_indices, axis=1)
                explicit_action_reason_kept, _, _ = jax.lax.while_loop(
                    cond_explicit_action_reasoner,
                    step_explicit_action_reasoner,
                    (ref_action_noise_kept, 1.0, 1),
                )
                explicit_action_reason = self._restore_fixed_l5_skip(
                    explicit_action_reason_kept,
                    explicit_action_skip_segment,
                )
            else:
                explicit_action_reason, _, _ = jax.lax.while_loop(
                    cond_explicit_action_reasoner,
                    step_explicit_action_reasoner,
                    (ref_action_noise, 1.0, 1),
                )
        else:
            explicit_action_reason = None

        return {
            "explicit_action_reason": explicit_action_reason,
            "action_cot_denoising_steps": jnp.broadcast_to(jnp.asarray(action_cot_denoising_steps), (batch_size,)),
        }

    def sample_actions_profile_expert(
        self,
        prefix_state: dict[str, Any],
        explicit_action_reason: _model.CoarseActions | None,
        implicit_action_reason: jax.Array | None,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> dict[str, Any]:
        observation = prefix_state["observation"]
        prefix_tokens = prefix_state["prefix_tokens"]
        prefix_mask = prefix_state["prefix_mask"]
        kv_cache = prefix_state["kv_cache"]
        expert_action_noise = prefix_state["expert_action_noise"]
        batch_size = observation.state.shape[0]
        action_dt = -1.0 / num_steps

        def step_expert(carry):
            x_t, time, step_idx = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size),
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
                suf_type = "expert"
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, _, suffix_out), _ = self.PaliGemma.llm(
                [None, None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + action_dt * v_t, time + action_dt, step_idx + 1

        def cond_expert(carry):
            x_t, time, _ = carry
            return time >= -action_dt / 2

        x_0_expert, _, _ = jax.lax.while_loop(cond_expert, step_expert, (expert_action_noise, 1.0, 1))
        return {"actions": x_0_expert}
