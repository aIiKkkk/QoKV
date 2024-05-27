import torch
import os
import torch.nn as nn
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.quant_transformer_layer import quant_layer
from tqdm import tqdm
R_DEBUG_BIT = 0
DEBUG_BREAK_LAYER = -1
@torch.no_grad()
def opt_reorder_quantize(
    lm,
    args,
    dataloader,
):
    print("Starting ...")

    model = lm.model
    dev = lm.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers


    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    # only catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))

        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0] #.cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    act_scales_k = {}
    act_scales_v = {}
    ratio_k_ave = []
    ratio_v_ave = []
    for i in tqdm(range(len(layers))):
        if i == DEBUG_BREAK_LAYER:
            break

        layer = layers[i]#.to(dev)
        dev = layer.self_attn.k_proj.weight.device
        # print(layer.self_attn.k_proj.weight.device)
        qlayer = QuantOPTDecoderLayer(lm.model.config, layer, args)

        outs, act_scale_k, act_scale_v, ratio_k, ratio_v = quant_layer(qlayer, args, outs, inps, attention_mask, dev, i)
        act_scales_k[i] = act_scale_k
        act_scales_v[i] = act_scale_v

        ratio_k_ave.append(ratio_k)
        ratio_v_ave.append(ratio_v)


        layers[i] = qlayer#.to(dev)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # ratio_k_percentage = [f"{x * 100:.2f}%" for x in ratio_k_ave]
    # ratio_v_percentage = [f"{x * 100:.2f}%" for x in ratio_v_ave]

    print("ratio_k:", ratio_k_ave)
    print("ratio_v:", ratio_v_ave)



    if not os.path.exists('act_scales'):
        os.makedirs('act_scales')
    torch.save(act_scales_k, f'act_scales/{args.net}Testk7.pt')
    torch.save(act_scales_v, f'act_scales/{args.net}Testv7.pt')
    del act_scales_k
    del act_scales_v
    del inps, outs
    model.config.use_cache = use_cache
    return model
