import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
import matplotlib.pyplot as plt

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, K_scales, V_scales, alpha=0.5):  # qproj, kproj,
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

    K_sca = True
    if K_sca:
        # 按layer对KV进行平滑操作
        layers = model.model.decoder.layers
        for i in range(len(layers)):
            for name, m in layers[i].named_modules():
                if isinstance(m, OPTAttention):
                    device, dtype = m.q_proj.weight.device, m.q_proj.weight.dtype

                    K_scale = K_scales[i].to(device=device, dtype=dtype)

                    m.q_proj.bias.mul_(K_scale)
                    m.k_proj.bias.div_(K_scale)

                    K_scale = K_scale.unsqueeze(0)
                    K_scale_T = K_scale.transpose(0, 1)

                    #QOr = m.q_proj.weight
                    #print("QOr", torch.max(QOr))
                    m.q_proj.weight.mul_(K_scale_T)
                    #QNow = m.q_proj.weight
                    #print("QNow", torch.max(QNow))

                    #KOr = m.k_proj.weight
                    #print("KOr", torch.max(KOr))
                    m.k_proj.weight.div_(K_scale_T)
                    #KNow = m.k_proj.weight
                    #print("KNow", torch.max(KNow))

    V_sca = True
    if V_sca:
        # 按layer对KV进行平滑操作
        layers = model.model.decoder.layers
        for i in range(len(layers)):
            for name, m in layers[i].named_modules():
                if isinstance(m, OPTAttention):
                    device, dtype = m.q_proj.weight.device, m.q_proj.weight.dtype

                    V_scale = V_scales[i].to(device=device, dtype=dtype)
                    m.v_proj.bias.div_(V_scale)

                    V_scale = V_scale.unsqueeze(0)
                    V_scale_T = V_scale.transpose(0, 1)

                    #OOr = m.out_proj.weight
                    #print("OOr", torch.max(OOr))
                    m.out_proj.weight.mul_(V_scale)
                    #ONow = m.out_proj.weight
                    #print("ONow", torch.max(ONow))

                    #VOr = m.v_proj.weight
                    #print("VOr", torch.max(VOr))
                    m.v_proj.weight.div_(V_scale_T)
                    #VNow = m.v_proj.weight
                    #print("VNow", torch.max(VNow))

    PrintK = False
    if PrintK:
        fig, axs = plt.subplots(10, 2, figsize=(15, 30))
        for i in range(20):
            row = i // 2
            col = i % 2
            x_axis = range(K_scales[i].shape[0])
            axs[row, col].scatter(x_axis, K_scales[i].cpu())
            axs[row, col].set_title(f'Scatter Plot of K_scales[{i}]')
            axs[row, col].set_xlabel('Index')
            axs[row, col].set_ylabel('Values')
        plt.tight_layout()
        plt.show()

    PrintV = False
    if PrintV:
        fig, axs = plt.subplots(10, 2, figsize=(15, 30))
        for i in range(20):
            row = i // 2
            col = i % 2
            x_axis = range(V_scales[i].shape[0])
            axs[row, col].scatter(x_axis, V_scales[i].cpu())
            axs[row, col].set_title(f'Scatter Plot of V_scales[{i}]')
            axs[row, col].set_xlabel('Index')
            axs[row, col].set_ylabel('Values')
        plt.tight_layout()
        plt.show()
           