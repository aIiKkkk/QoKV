import torch

@torch.no_grad()
def quant_layer(qlayer, args, outs, inps, attention_mask, dev, i):

    optimal_group = []
    for j in range(args.nsamples):
        outs[j] = qlayer(
            inps[j].unsqueeze(0).to(dev), i=i, attention_mask=attention_mask.to(dev)
        )[0]

        if not optimal_group:
            optimal_group = qlayer.grouped_ranges.copy()
        else:
            for k in range(len(qlayer.grouped_ranges)):
                optimal_group[k] += qlayer.grouped_ranges[k]
    optimal_group = [x / args.nsamples for x in optimal_group]
    print(optimal_group)
    return outs, optimal_group
