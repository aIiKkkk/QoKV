import torch

@torch.no_grad()
def quant_layer(qlayer, args, outs, inps, attention_mask, dev, i):
    smaxk = torch.zeros(inps[0].shape[-1]).to(dev)
    smaxv = torch.zeros(inps[0].shape[-1]).to(dev)
    ratio_k = 0
    ratio_v = 0
    for j in range(args.nsamples):
        outs[j] = qlayer(
            inps[j].unsqueeze(0).to(dev), i=i, attention_mask=attention_mask.to(dev)
        )[0]
        smaxk = torch.max(smaxk, qlayer.best_sca_val_K)
        smaxv = torch.max(smaxv, qlayer.best_sca_val_V)

        ratio_k += qlayer.ratio_k
        ratio_v += qlayer.ratio_v

    act_scale_k = smaxk
    act_scale_v = smaxv
    del smaxk
    del smaxv
    return outs, act_scale_k, act_scale_v, ratio_k/args.nsamples, ratio_v/args.nsamples

def quant_layer_llama(qlayer, args, outs, inps, attention_mask, position_ids, dev, i):
    smaxk = torch.zeros((inps[0].shape[-1])).to(dev)#  // 8).to(dev)
    smaxv = torch.zeros((inps[0].shape[-1])).to(dev)#  // 8
    optimal_group = []
    ratio_k = 0
    ratio_v = 0
    for j in range(args.nsamples):
        outs[j] = qlayer(
            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev), position_ids = position_ids.to(dev)
        )[0]
        smaxk = torch.max(smaxk, qlayer.best_sca_val_K)
        smaxv = torch.max(smaxv, qlayer.best_sca_val_V)
        if not optimal_group:
            optimal_group = qlayer.grouped_ranges.copy()
        else:
            for k in range(len(qlayer.grouped_ranges)):
                optimal_group[k] += qlayer.grouped_ranges[k]

        ratio_k += qlayer.ratio_k
        ratio_v += qlayer.ratio_v
    optimal_group = [x / args.nsamples for x in optimal_group]
    #print(optimal_group)
    act_scale_k = smaxk
    act_scale_v = smaxv
    del smaxk
    del smaxv
    return outs, act_scale_k, act_scale_v, optimal_group, ratio_k/args.nsamples, ratio_v/args.nsamples
