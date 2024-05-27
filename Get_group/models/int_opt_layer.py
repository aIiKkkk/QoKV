import torch
from torch import nn
from typing import Optional, Tuple, List
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class QuantOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.k_proj = org_module.k_proj
        self.v_proj = org_module.v_proj
        self.q_proj = org_module.q_proj
        self.out_proj = org_module.out_proj

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous())

    @torch.no_grad()
    def forward(
        self,
        i,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self.k_proj(key_value_states)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self.k_proj(hidden_states)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self.v_proj(hidden_states)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
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
                row_vector = torch.tensor([heavy_budget - i if i < heavy_budget else 1 for i in range(heavy_budget)],
                                          dtype=torch.float16)
                row_vector = torch.unsqueeze(row_vector, dim=0).to(attn_weights.device)

                if attn_weights.dtype == torch.float16:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
                else:
                    tmp_attn = nn.functional.softmax(attn_weights, dim=-1)

                accumulated_attention_score = torch.sum(tmp_attn[:, :, :], dim=-2)
                # accumulated_attention_score = torch.sum(tmp_attn[:, heavy_budget - budget:, :], dim=-2)
                accumulated_attention_score = torch.unsqueeze(torch.sum(accumulated_attention_score, dim=0), dim=0)
                accumulated_attention_score = accumulated_attention_score / (row_vector)
                accumulated_attention_score = accumulated_attention_score / (bsz * self.num_heads)
                _, print_indices = accumulated_attention_score.topk(k=1000, dim=-1)


                data = print_indices.reshape(-1, 1)
                # print(data)
                # 设定我们想要的簇数（K）值
                k = 10  # 设置为3个簇
                # 初始化 K-means 模型
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
                # 拟合数据并预测簇标签
                kmeans.fit(data.cpu().numpy())
                # 获取每个簇的最小值和最大值，并计算簇中元素的数量
                clusters = {}
                for label in np.unique(kmeans.labels_):
                    # print(kmeans.labels_)
                    clusters[label] = data[kmeans.labels_ == label]
                    # print(clusters[label])
                    clusters[label] = {
                        'min': torch.min(clusters[label]),
                        'max': torch.max(clusters[label]),
                        'num_elements': clusters[label].cpu().numpy().size
                    }
                # 把clusters按簇的最小值进行排序
                sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['min'])
                # 计算每个簇的密度
                for label, cluster in sorted_clusters:
                    # print(data.cpu().numpy().size)
                    #print(cluster['num_elements'])
                    cluster['density'] = (data.cpu().numpy().size) / cluster['num_elements']
                # 获取总范围和总密度
                total_density = sum(cluster['density'] for _, cluster in sorted_clusters)
                # 计算每个簇的0-1比例
                grouped_ranges = []
                cumulative_sum = 0.0
                for i, (label, cluster) in enumerate(sorted_clusters):
                    proportion = cluster['density'] / total_density
                    start = cumulative_sum
                    end = start + proportion
                    if i == len(sorted_clusters) - 1:
                        end = 1.0
                    cumulative_sum += proportion
                    grouped_ranges.append((end - start))
                # print(grouped_ranges)


            key_states = self._shape(key_states, -1, bsz)  # (bsz, self.num_heads, seq_len, self.head_dim)
            value_states = self._shape(value_states, -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # ([32, 2048, 64])
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:

            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_probs_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output =  torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value, grouped_ranges


class QuantOPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        ori_layer,
        args,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = QuantOPTAttention(
            org_module=ori_layer.self_attn,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            args=args,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.self_attn_layer_norm = ori_layer.self_attn_layer_norm
        self.fc1 = ori_layer.fc1
        self.fc2 = ori_layer.fc2
        self.final_layer_norm = ori_layer.final_layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        i: Optional = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,

    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # Self Attention
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value, grouped_ranges = self.self_attn(
            i = i,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        self.grouped_ranges = grouped_ranges
        # print(self.s.shape)
        hidden_states = nn.functional.dropout(hidden_states, p=0.0, training=False)

        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)

        hidden_states = F.relu(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
