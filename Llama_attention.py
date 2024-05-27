"""PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Union
from fake_quant import W8A8Linear


import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
import copy

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)


    @torch.no_grad()
    def forward(self, x, position_ids):
        dev = x.device
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)).to(dev)
        position_ids_expanded = (position_ids[:, None, :].float()).to(dev)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # print(hidden_states.shape, n_rep)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_module, config, K_scales, V_scales, group, weight_quant, act_quant, quantize_bmm_input, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.group = group
        self.K_scales = K_scales
        self.V_scales = V_scales
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )


        self.k_proj = W8A8Linear.from_float(org_module.k_proj, weight_quant=weight_quant, act_quant=act_quant)
        self.v_proj = W8A8Linear.from_float(org_module.v_proj, weight_quant=weight_quant, act_quant=act_quant)

        self.q_proj = W8A8Linear.from_float(org_module.q_proj, weight_quant=weight_quant,
                                            act_quant=act_quant, quantize_output=quantize_bmm_input)
        self.o_proj = W8A8Linear.from_float(org_module.o_proj, weight_quant=weight_quant, act_quant=act_quant)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            # print(key_states.shape, value_states.shape)
            # print("key_states", key_states.max())
            # print("value_states", value_states.max())
            # print(self.q_proj.weight.shape, self.k_proj.weight.shape)

            cos, sin = self.rotary_emb(value_states * self.K_scales, position_ids)

            mixed = False

            if mixed:
                query_states1 = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states1 = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                query_states1, key_states1 = apply_rotary_pos_emb(query_states1, key_states1, cos, sin)
                key_states1 = repeat_kv(key_states1, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states1, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    # print("causal_mask_shape", causal_mask.shape)
                    attn_weights = attn_weights + causal_mask.to(attn_weights.device)
                    # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
                    attn_weights = attn_weights.view(bsz * self.num_heads, q_len, q_len)

                heavy_budget = int(attn_weights.shape[-1])
                # print("attn_weights.shape",attn_weights.shape)

                row_vector = torch.tensor([heavy_budget - i if i < heavy_budget else 1 for i in range(heavy_budget)],
                                          dtype=torch.float16)
                row_vector = torch.unsqueeze(row_vector, dim=0).to(attn_weights.device)
                # print("row_vector", row_vector)

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                tmp_attn = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                # print("tmp_attn_shape", tmp_attn.shape)

                accumulated_attention_score = torch.sum(tmp_attn[:, :, :], dim=-2)
                # accumulated_attention_score = torch.sum(tmp_attn[:, heavy_budget - budget:, :], dim=-2)
                accumulated_attention_score = torch.unsqueeze(torch.sum(accumulated_attention_score, dim=0), dim=0)
                accumulated_attention_score = accumulated_attention_score / (row_vector)
                accumulated_attention_score = accumulated_attention_score / (bsz * self.num_heads)
                # _, print_indices = accumulated_attention_score.topk(k=100, dim=-1)
                # print(print_indices)
                # print("accumulated_attention_score_shape", accumulated_attention_score.shape)

                Group = True  ###不要动
                if Group:
                    ratios = self.group
                    num_ranges = len(ratios)
                    start = 1
                    topk_indices_list = []
                    range_size_or = int((heavy_budget / (len(ratios))) / 10)
                    for i, ratio in enumerate(ratios):
                        range_size = int(heavy_budget * ratio)
                        end = min(start + range_size - 1, heavy_budget) if i < num_ranges - 1 else heavy_budget
                        partial_score = accumulated_attention_score[:, start - 1:end]
                        _, topk_indices = partial_score.topk(k=range_size_or, dim=-1)
                        topk_indices_list.append(topk_indices + start - 1)
                        start = end + 1
                    tmp_topk_index = torch.cat(topk_indices_list, dim=-1)
                else:
                    head_sum = torch.sum(tmp_attn[:, :, :], dim=0)
                    average_attention = head_sum / (self.num_heads)
                    _, tmp_topk_index = average_attention[-1].topk(k=int(heavy_budget / 10), dim=-1)

                # print(tmp_topk_index)
                def fake_quant_mix(t, axis, n_bits=8):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s

                def fake_quant(t, axis, n_bits=4):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s
                def fake_quant_v(t, n_bits=4):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max()
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t

                key_states_importance = fake_quant_mix(key_states, axis=-1)
                key_states = fake_quant(key_states, axis=-1)
                key_states[:, tmp_topk_index, :] = key_states_importance[:, tmp_topk_index, :]
                value_states = fake_quant_v(value_states)
            else:
                def fake_quant_k(t, axis, n_bits=8):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s

                def fake_quant_v(t, n_bits=4):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max()
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t

                key_states = fake_quant_k(key_states, axis=-1)
                value_states = fake_quant_v(value_states)


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states * self.K_scales
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states * self.V_scales
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask.to(attn_weights.device)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
