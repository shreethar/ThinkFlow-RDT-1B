"""Qwen-KV-conditioned SmolVLA without patching the upstream LeRobot package.

The custom tokens are appended to the VLM-derived source K/V in each Action Expert
cross-attention layer. It is deliberately not a replacement for the VLM source:
with only one replacement key, softmax attention would always equal one and the
query/key interaction would be mathematically inert.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import Iterator, Unpack

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration import KVSmolVLAConfig


def eager_grouped_query_attention(
    attention_mask: Tensor,
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    *,
    appended_token_count: int = 0,
    appended_logit_bias: Tensor | None = None,
) -> Tensor:
    """SmolVLA eager GQA with an optional bias on appended source tokens.

    Shapes are ``Q=[B,Lq,Hq,D]``, ``K/V=[B,Lk,Hkv,D]``, and
    ``mask=[B,Lq,Lk]``. The implementation mirrors LeRobot 0.6 SmolVLA, but
    infers head counts from tensors so it can be unit-tested independently.
    """

    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query_states, key_states, and value_states must all be rank four")
    if key_states.shape != value_states.shape:
        raise ValueError(f"K and V shapes differ: {key_states.shape} versus {value_states.shape}")
    batch_size, query_length, num_query_heads, head_dim = query_states.shape
    key_batch, key_length, num_kv_heads, key_head_dim = key_states.shape
    if key_batch != batch_size or key_head_dim != head_dim:
        raise ValueError("Q and K batch/head dimensions are incompatible")
    if num_query_heads % num_kv_heads:
        raise ValueError("The query head count must be divisible by the KV head count")
    if attention_mask.shape != (batch_size, query_length, key_length):
        raise ValueError(
            f"Expected attention mask {(batch_size, query_length, key_length)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if appended_token_count < 0 or appended_token_count > key_length:
        raise ValueError("appended_token_count is outside the key sequence")

    groups = num_query_heads // num_kv_heads
    key_states = (
        key_states[:, :, :, None, :]
        .expand(batch_size, key_length, num_kv_heads, groups, head_dim)
        .reshape(batch_size, key_length, num_query_heads, head_dim)
    )
    value_states = (
        value_states[:, :, :, None, :]
        .expand(batch_size, key_length, num_kv_heads, groups, head_dim)
        .reshape(batch_size, key_length, num_query_heads, head_dim)
    )

    scores = torch.matmul(
        query_states.float().transpose(1, 2),
        key_states.float().transpose(1, 2).transpose(2, 3),
    )
    scores.mul_(head_dim**-0.5)
    if appended_token_count and appended_logit_bias is not None:
        bias = appended_logit_bias.float()
        if bias.ndim == 0:
            bias = bias.expand(num_query_heads)
        if bias.shape != (num_query_heads,):
            raise ValueError(
                f"Expected one appended logit bias per query head ({num_query_heads}), "
                f"got {tuple(bias.shape)}"
            )
        scores[..., -appended_token_count:] += bias[None, :, None, None]

    scores = torch.where(
        attention_mask[:, None, :, :],
        scores,
        torch.finfo(scores.dtype).min,
    )
    probabilities = nn.functional.softmax(scores, dim=-1).to(dtype=value_states.dtype)
    output = torch.matmul(probabilities, value_states.permute(0, 2, 1, 3))
    return output.permute(0, 2, 1, 3).reshape(
        batch_size, query_length, num_query_heads * head_dim
    )


class QwenKVSmolVLMWithExpertModel(SmolVLMWithExpertModel):
    """SmolVLM + Action Expert with layer-specific Qwen K/V adapters."""

    def __init__(
        self,
        *args,
        external_kv_width: int = 2048,
        external_kv_token_count: int = 1,
        external_kv_logit_bias_init: float = -4.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if external_kv_width <= 0 or external_kv_width % 2:
            raise ValueError("external_kv_width must be positive and even")
        if external_kv_token_count <= 0:
            raise ValueError("external_kv_token_count must be positive")
        self.external_kv_width = external_kv_width
        self.external_kv_token_count = external_kv_token_count
        self.external_half_width = external_kv_width // 2
        self._active_external_kv: Tensor | None = None

        source_kv_width = (
            self.config.text_config.num_key_value_heads * self.config.text_config.head_dim
        )
        cross_layers = [
            idx
            for idx in range(self.num_vlm_layers)
            if not (
                self.self_attn_every_n_layers > 0
                and idx % self.self_attn_every_n_layers == 0
            )
        ]
        self.external_key_projections = nn.ModuleDict(
            {
                str(idx): nn.Linear(self.external_half_width, source_kv_width)
                for idx in cross_layers
            }
        )
        self.external_value_projections = nn.ModuleDict(
            {
                str(idx): nn.Linear(self.external_half_width, source_kv_width)
                for idx in cross_layers
            }
        )
        self.external_logit_biases = nn.ParameterDict(
            {
                str(idx): nn.Parameter(
                    torch.full(
                        (self.num_attention_heads,),
                        float(external_kv_logit_bias_init),
                    )
                )
                for idx in cross_layers
            }
        )

    @contextmanager
    def use_external_kv(self, external_kv: Tensor | None) -> Iterator[None]:
        """Activate one batch of external KV for a synchronous train/sample call."""

        previous = self._active_external_kv
        self._active_external_kv = external_kv
        try:
            yield
        finally:
            self._active_external_kv = previous

    def _project_external_kv(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor] | None:
        external = self._active_external_kv
        if external is None:
            return None
        if external.ndim == 2:
            external = external[:, None, :]
        if external.ndim != 3:
            raise ValueError(
                f"Expected external KV [B,T,{self.external_kv_width}], got {tuple(external.shape)}"
            )
        if external.shape[0] != batch_size or external.shape[-1] != self.external_kv_width:
            raise ValueError(
                f"Expected external KV [B,T,{self.external_kv_width}] with B={batch_size}, "
                f"got {tuple(external.shape)}"
            )
        if external.shape[1] != self.external_kv_token_count:
            raise ValueError(
                f"Expected {self.external_kv_token_count} external KV tokens, "
                f"got {external.shape[1]}"
            )

        key_input, value_input = external.split(self.external_half_width, dim=-1)
        key_projection = self.external_key_projections[str(layer_idx)]
        value_projection = self.external_value_projections[str(layer_idx)]
        key_input = key_input.to(device=device, dtype=key_projection.weight.dtype)
        value_input = value_input.to(device=device, dtype=value_projection.weight.dtype)
        key = key_projection(key_input)
        value = value_projection(value_input)
        kv_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.head_dim
        key = key.view(batch_size, key.shape[1], kv_heads, head_dim)
        value = value.view(batch_size, value.shape[1], kv_heads, head_dim)
        return key, value

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        """Upstream cross-attention plus the appended external source token."""

        attention_interface = self.get_attention_interface()
        att_outputs = []
        if len(inputs_embeds) != 2:
            raise AssertionError(
                f"Unexpected inputs/cache combination: inputs={len(inputs_embeds)}, "
                f"past_key_values={past_key_values is not None}"
            )
        # SmolVLA prefill supplies [prefix_embeds, None]. Flow-matching denoise
        # calls then supply [None, suffix_embeds] and reuse the prefix K/V. Use
        # the actual prefix tensor as the phase signal: ``past_key_values`` is a
        # shared per-layer dictionary, and ``fill_kv_cache`` has varied across
        # LeRobot releases and cannot reliably distinguish these calls.
        prefix_embeds = inputs_embeds[0]
        has_prefix = prefix_embeds is not None
        if has_prefix:
            seq_len = prefix_embeds.shape[1]
            position_id = position_ids[:, :seq_len]
            expert_position_id = position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]
            layer = model_layers[0][layer_idx]
            hidden_states = layer.input_layernorm(prefix_embeds)
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
            query_states = apply_rope(query_states, position_id)
            key_states = apply_rope(key_states, position_id)
            att_outputs.append(
                attention_interface(
                    prefix_attention_mask,
                    batch_size,
                    head_dim,
                    query_states,
                    key_states,
                    value_states,
                )
            )
        else:
            expert_position_id = position_ids

        if has_prefix:
            if use_cache:
                if past_key_values is None:
                    past_key_values = {}
                # The Qwen token is not put in the static cache: its layer adapter is
                # applied at every expert CA invocation and remains differentiable.
                if isinstance(past_key_values, dict):
                    # Compatibility with older LeRobot releases.
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                elif hasattr(past_key_values, "update"):
                    # Newer LeRobot uses Transformers DynamicCache, whose stored
                    # layout is [batch, heads, sequence, head_dim].
                    past_key_values.update(
                        key_states.transpose(1, 2),
                        value_states.transpose(1, 2),
                        layer_idx,
                    )
                else:
                    raise TypeError(
                        "Unsupported SmolVLA prefix cache type: "
                        f"{type(past_key_values).__name__}"
                    )
        else:
            if not use_cache or past_key_values is None:
                raise RuntimeError(
                    "SmolVLA cross-attention received no prefix embeddings and no "
                    "prefix K/V cache"
                )
            if isinstance(past_key_values, dict):
                if layer_idx not in past_key_values:
                    raise KeyError(
                        f"Missing cached prefix K/V for cross-attention layer {layer_idx}"
                    )
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]
            elif hasattr(past_key_values, "layers"):
                if layer_idx >= len(past_key_values.layers):
                    raise KeyError(
                        f"Missing DynamicCache layer {layer_idx}; "
                        f"cache contains {len(past_key_values.layers)} layers"
                    )
                cached_layer = past_key_values.layers[layer_idx]
                cached_keys = getattr(cached_layer, "keys", None)
                cached_values = getattr(cached_layer, "values", None)
                if cached_keys is None or cached_values is None:
                    raise KeyError(
                        f"DynamicCache layer {layer_idx} has no prefix K/V tensors"
                    )
                key_states = cached_keys.transpose(1, 2)
                value_states = cached_values.transpose(1, 2)
            else:
                raise TypeError(
                    "Unsupported SmolVLA prefix cache type: "
                    f"{type(past_key_values).__name__}"
                )

        expert_layer = model_layers[1][layer_idx]
        if expert_layer is None:
            att_outputs.append(None)
            return att_outputs, past_key_values

        expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
        expert_hidden_shape = (
            *expert_hidden_states.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )
        expert_hidden_states = expert_hidden_states.to(
            dtype=expert_layer.self_attn.q_proj.weight.dtype
        )
        expert_query_states = expert_layer.self_attn.q_proj(expert_hidden_states).view(
            expert_hidden_shape
        )

        original_source_length = key_states.shape[1]
        external_pair = self._project_external_kv(
            layer_idx,
            batch_size=batch_size,
            device=key_states.device,
        )
        external_token_count = 0
        if external_pair is not None:
            external_key, external_value = external_pair
            external_token_count = external_key.shape[1]
            # The external tensor already represents K/V. Do not apply SmolVLA's
            # positional RoPE a second time; the learned adapters align its space.
            key_states = torch.cat([key_states, external_key], dim=1)
            value_states = torch.cat([value_states, external_value], dim=1)

        flat_key_states = key_states.to(
            dtype=expert_layer.self_attn.k_proj.weight.dtype
        ).flatten(start_dim=2)
        expert_key_states = expert_layer.self_attn.k_proj(flat_key_states).view(
            *flat_key_states.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )
        flat_value_states = value_states.to(
            dtype=expert_layer.self_attn.v_proj.weight.dtype
        ).flatten(start_dim=2)
        expert_value_states = expert_layer.self_attn.v_proj(flat_value_states).view(
            *flat_value_states.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )

        expert_position_id = expert_position_id - torch.min(
            expert_position_id, dim=1, keepdim=True
        ).values
        suffix_length = inputs_embeds[1].shape[1]
        expert_attention_mask = attention_mask[
            :, -suffix_length:, :original_source_length
        ]
        if external_token_count:
            external_mask = torch.ones(
                batch_size,
                suffix_length,
                external_token_count,
                dtype=torch.bool,
                device=expert_attention_mask.device,
            )
            expert_attention_mask = torch.cat(
                [expert_attention_mask.to(dtype=torch.bool), external_mask], dim=-1
            )

        expert_query_states = apply_rope(expert_query_states, expert_position_id)
        if external_token_count:
            att_output = eager_grouped_query_attention(
                expert_attention_mask,
                expert_query_states,
                expert_key_states,
                expert_value_states,
                appended_token_count=external_token_count,
                appended_logit_bias=self.external_logit_biases[str(layer_idx)],
            )
        else:
            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
        att_outputs.append(att_output)
        return att_outputs, past_key_values


class QwenKVVLAFlowMatching(VLAFlowMatching):
    """Normal SmolVLA flow-matching model with the custom inner transformer."""

    def __init__(self, config: KVSmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        # This mirrors LeRobot's VLAFlowMatching.__init__, replacing only the
        # SmolVLMWithExpertModel constructor. Calling super() and replacing it would
        # temporarily allocate two complete VLMs.
        nn.Module.__init__(self)
        self.config = config
        if config.compile_model:
            raise NotImplementedError("torch.compile is not enabled for the external-KV context path")
        self.vlm_with_expert = QwenKVSmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
            external_kv_width=config.external_kv_width,
            external_kv_token_count=config.external_kv_token_count,
            external_kv_logit_bias_init=config.external_kv_logit_bias_init,
        )
        self.state_proj = nn.Linear(
            config.max_state_dim,
            self.vlm_with_expert.config.text_config.hidden_size,
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size,
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size,
        )
        self.set_requires_grad()
        tokenizer = self.vlm_with_expert.processor.tokenizer
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = config.prefix_length
        self.rtc_processor = rtc_processor


class KVSmolVLAPolicy(SmolVLAPolicy):
    """Drop-in SmolVLA policy requiring an additional ``batch['qwen_kv']``."""

    config_class = KVSmolVLAConfig
    name = "smolvla_qwen_kv"

    def __init__(self, config: KVSmolVLAConfig, **kwargs) -> None:
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = QwenKVVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _external_kv_from_batch(self, batch: dict[str, Tensor]) -> Tensor | None:
        external = batch.get(self.config.external_kv_key)
        if external is None:
            if self.config.external_kv_required:
                raise KeyError(
                    f"Missing required batch key {self.config.external_kv_key!r}; "
                    "expected Qwen [K | V] conditioning"
                )
            return None
        if external.ndim == 2:
            external = external[:, None, :]
        if external.ndim != 3 or external.shape[-1] != self.config.external_kv_width:
            raise ValueError(
                f"Expected {self.config.external_kv_key} [B,T,{self.config.external_kv_width}], "
                f"got {tuple(external.shape)}"
            )
        if external.shape[1] != self.config.external_kv_token_count:
            raise ValueError(
                f"Expected {self.config.external_kv_token_count} Qwen KV tokens for this "
                f"checkpoint, got {external.shape[1]}"
            )
        return external

    @staticmethod
    def _counterfactual_indices(batch: dict[str, Tensor], batch_size: int, device) -> tuple[Tensor, float]:
        """Choose a cyclic donor assignment with maximum task mismatch.

        Cached batches preserve ``qwen_group_id`` from the instruction. Searching
        cyclic shifts gives a derangement without an expensive assignment solver.
        If legacy data lacks the ID, a one-position derangement is still used.
        """

        if batch_size < 2:
            return torch.arange(batch_size, device=device), 0.0
        groups = batch.get("qwen_group_id")
        if groups is None:
            return torch.roll(torch.arange(batch_size, device=device), 1), 0.0
        groups = torch.as_tensor(groups, device=device).reshape(batch_size)
        base = torch.arange(batch_size, device=device)
        best_indices = torch.roll(base, 1)
        best_fraction = float((groups[best_indices] != groups).float().mean().item())
        for shift in range(2, batch_size):
            candidate = torch.roll(base, shift)
            fraction = float((groups[candidate] != groups).float().mean().item())
            if fraction > best_fraction:
                best_indices = candidate
                best_fraction = fraction
            if best_fraction == 1.0:
                break
        return best_indices, best_fraction

    @staticmethod
    def _capture_rng_state() -> tuple[Tensor, list[Tensor] | None]:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng_state(state: tuple[Tensor, list[Tensor] | None]) -> None:
        cpu_state, cuda_state = state
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ):
        external = self._external_kv_from_batch(batch)
        ranking_weight = float(self.config.external_kv_ranking_weight)
        use_ranking = (
            self.training
            and ranking_weight > 0.0
            and reduction == "mean"
            and external is not None
            and external.shape[0] > 1
        )
        if not use_ranking:
            with self.model.vlm_with_expert.use_external_kv(external):
                return super().forward(batch, noise=noise, time=time, reduction=reduction)

        # Replay the exact RNG stream for the counterfactual pass. Consequently
        # both losses use identical diffusion noise, time, and dropout masks;
        # only the external Qwen K/V donor changes.
        rng_before = self._capture_rng_state()
        with self.model.vlm_with_expert.use_external_kv(external):
            matched_loss, loss_dict = super().forward(
                batch, noise=noise, time=time, reduction=reduction
            )
        rng_after = self._capture_rng_state()
        donor_indices, task_mismatch = self._counterfactual_indices(
            batch, external.shape[0], external.device
        )
        counterfactual = external.index_select(0, donor_indices)
        try:
            self._restore_rng_state(rng_before)
            with self.model.vlm_with_expert.use_external_kv(counterfactual):
                counterfactual_loss, _ = super().forward(
                    batch, noise=noise, time=time, reduction=reduction
                )
        finally:
            # One optimizer step should advance randomness exactly as one native
            # forward, independent of whether fusion ranking is enabled.
            self._restore_rng_state(rng_after)

        margin = float(self.config.external_kv_ranking_margin)
        ranking_loss = nn.functional.relu(margin + matched_loss - counterfactual_loss)
        total_loss = matched_loss + ranking_weight * ranking_loss
        loss_dict = dict(loss_dict)
        loss_dict.update(
            {
                # LeRobot's WandB wrapper deliberately accepts Python scalars
                # rather than tensors. These are logging-only copies; the
                # differentiable tensors above still form ``total_loss``.
                "imitation_loss": float(matched_loss.detach().item()),
                "fusion_ranking_loss": float(ranking_loss.detach().item()),
                "fusion_counterfactual_loss": float(
                    counterfactual_loss.detach().item()
                ),
                "fusion_loss_gap": float(
                    (counterfactual_loss - matched_loss).detach().item()
                ),
                "fusion_task_mismatch": float(task_mismatch),
            }
        )
        return total_loss, loss_dict

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        external = self._external_kv_from_batch(batch)
        with self.model.vlm_with_expert.use_external_kv(external):
            return super()._get_action_chunk(batch, noise=noise, **kwargs)
