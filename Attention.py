import torch
from torch import nn
from typing import Optional, Tuple
from fake_quant import W8A8Linear
# import matplotlib.pyplot as plt

class QuantOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        group,
        embed_dim: int,
        num_heads: int,
        weight_quant,
        act_quant,
        quantize_bmm_input,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = 0.0
        self.group = group
        self.head_dim = embed_dim // num_heads
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.quantize_bmm_input = quantize_bmm_input

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = True

        self.k_proj = W8A8Linear.from_float(org_module.k_proj, weight_quant=weight_quant, act_quant=act_quant)
        self.v_proj = W8A8Linear.from_float(org_module.v_proj, weight_quant=weight_quant, act_quant=act_quant)
        # 在此处不对KV进行量化，将量化步骤转移到后面 
        self.q_proj = W8A8Linear.from_float(org_module.q_proj, weight_quant=weight_quant,
                                            act_quant=act_quant, quantize_output=quantize_bmm_input)
        self.out_proj = W8A8Linear.from_float(org_module.out_proj, weight_quant=weight_quant, act_quant=act_quant)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            print(key_states.shape)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            print(key_states.shape)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            mixed = True
            if mixed:
                proj_shape = (bsz * self.num_heads, -1, self.head_dim)
                query_states1 = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
                key_states1 = self._shape(key_states, -1, bsz).view(*proj_shape)
                src_len = key_states1.size(1)
                attn_weights = torch.bmm(query_states1, key_states1.transpose(1, 2))
                if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                        f" {attn_weights.size()}")
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
                    attn_weights = (attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask)
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
                    attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

                heavy_budget = int(attn_weights.shape[-1])
                #budget = int(heavy_budget / 10)
                #row_vector = torch.tensor([budget if i < heavy_budget - budget else heavy_budget - i for i in range(heavy_budget)],
                #        dtype=torch.float16)

                row_vector = torch.tensor([heavy_budget - i if i < heavy_budget else 1 for i in range(heavy_budget)], dtype=torch.float16)

                row_vector = torch.unsqueeze(row_vector, dim=0).to(attn_weights.device)

                if attn_weights.dtype == torch.float16:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                else:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1)

                accumulated_attention_score = torch.sum(tmp_attn[:, :, :], dim=-2)
                #accumulated_attention_score = torch.sum(tmp_attn[:, heavy_budget - budget:, :], dim=-2)
                accumulated_attention_score = torch.unsqueeze(torch.sum(accumulated_attention_score, dim=0), dim=0)
                accumulated_attention_score = accumulated_attention_score / (row_vector)
                accumulated_attention_score = accumulated_attention_score / (bsz * self.num_heads)
                # _, print_indices = accumulated_attention_score.topk(k=100, dim=-1)
                # print(print_indices)
                

                Group = True  ###不要动
                if Group:
                    ratios = self.group # 1
                    num_ranges = len(ratios)
                    start = 1
                    topk_indices_list = []
                    range_size_or = int((heavy_budget / (len(ratios))) / 2)
                    for i, ratio in enumerate(ratios):
                        range_size = int(heavy_budget * ratio)
                        end = min(start + range_size - 1, heavy_budget) if i < num_ranges - 1 else heavy_budget
                        partial_score = accumulated_attention_score[:, start - 1:end]
                        _, topk_indices = partial_score.topk(k=range_size_or, dim=-1)
                        topk_indices_list.append(topk_indices + start - 1)
                        start = end + 1
                    tmp_topk_index = torch.cat(topk_indices_list, dim=-1)
                else:
                    # _, tmp_topk_index = accumulated_attention_score.topk(k=int(heavy_budget / 20), dim=-1)
                    # tmp_topk_index = torch.randint(low=0, high=heavy_budget, size=(1, int(heavy_budget / 2))).to(attn_weights.device) # 随机选取
                    head_sum = torch.sum(tmp_attn[:, :, :], dim=0)
                    average_attention = head_sum / (self.num_heads)
                    _, tmp_topk_index = average_attention[-1].topk(k=int(heavy_budget / 10), dim=-1)


                def fake_quant_mix(t, axis, n_bits=8):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s
                def fake_quant(t, axis, n_bits=4):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max(dim=axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t
                def fake_quant_v(t, n_bits=4):
                    t_shape = t.shape
                    t.view(-1, t_shape[-1])
                    scales = t.abs().max()
                    q_max = 2 ** (n_bits - 1) - 1
                    scales.clamp_(min=1e-5).div_(q_max)
                    t.div_(scales).round_().mul_(scales)
                    return t


                # important_states = key_states[:, tmp_topk_index, :]
                # important_states_importance = fake_quant_mix(important_states, axis=-1)
                # key_states[:, tmp_topk_index, :] = important_states_importance
                # total_indices = torch.arange(key_states.size(1))
                # mask = torch.ones_like(total_indices, dtype=bool)
                # mask[tmp_topk_index] = False
                # other_indices = total_indices[mask]
                # key_states[:, other_indices, :] = fake_quant(key_states[:, other_indices, :], axis=-1)

                key_states_importance = fake_quant_mix(key_states, axis=-1)
                key_states = fake_quant(key_states, axis=-1)
                key_states[:, tmp_topk_index, :] = key_states_importance[:, tmp_topk_index, :]
                value_states = fake_quant_v(value_states)

            else:
                def fake_quant_k(t, axis = -1, n_bits=4):
                    xmax = t.abs().amax(axis, keepdim=True)[0]
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s
                def fake_quant_v(t, n_bits=4):
                    xmax = t.abs().amax()
                    q_max = 2 ** (n_bits - 1) - 1
                    scales = xmax / q_max
                    scales.clamp_(min=1e-5)
                    s = (t / scales).round_()
                    s = s.mul_(scales)
                    return s

                #K1 = query_states
                #query_states= fake_quant_k(query_states)
                key_states = fake_quant_v(key_states)
                #K2 = query_states

                def matrix_matrix(arr, brr):
                    dot_product = torch.sum(arr * brr, dim=2)
                    norm_arr = torch.sqrt(torch.sum(arr ** 2, dim=2))
                    norm_brr = torch.sqrt(torch.sum(brr ** 2, dim=2))

                    zero_mask_arr = norm_arr == 0
                    zero_mask_brr = norm_brr == 0
                    norm_arr = torch.where(zero_mask_arr, torch.ones_like(norm_arr) * 1e-7, norm_arr)
                    norm_brr = torch.where(zero_mask_brr, torch.ones_like(norm_brr) * 1e-7, norm_brr)
                    cosine_similarity = dot_product / (norm_arr * norm_brr)

                    # 处理 NaN 的情况
                    cosine_similarity[torch.isnan(cosine_similarity)] = 0
                    return cosine_similarity.mean().item()

                #print(matrix_matrix(K1,K2))

                value_states = fake_quant_v(value_states)
                ##value_states = fake_quant_k(value_states, axis=-1)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value