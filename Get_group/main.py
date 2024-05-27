import os
import sys

import random
import numpy as np
from models.opt import OPTClass
import torch
import time
from datautils import get_loaders
from quantize.opt_reorder_quantize import opt_reorder_quantize
import datetime
from models.int_opt_layer import QuantOPTAttention
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.opt_reorder_quantize import opt_reorder_quantize
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    # "llama-7b",
    # "llama-13b",
    # "bloom-3b",
]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=net_choices)
    parser.add_argument(
        "--cache_dir", default="./data", type=str, help="OPT model cache_dir"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix",
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ema_minmax",
        choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--abits_importance", type=int, default=16)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--reorder", type=str, default="0", help="like 12345 or 1")
    parser.add_argument(
        "--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--a_dynamic", action="store_true")
    parser.add_argument("--eval_base_ppl", action="store_true")
    parser.add_argument("--act_dist_plot", action="store_true")
    parser.add_argument("--only_quant_kv", action="store_true")
    parser.add_argument(
        "--pack_weight",
        action="store_true",
        help="enable this to reduce memory consumption",
    )
    parser.add_argument(
        "--multigpu", action="store_true", help="at eval, map model to multiple gpus"
    )

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if "opt" in args.net:
        args.model = f"facebook/{args.net}"
        lm = OPTClass(args)
        lm.model.eval()
    #print(lm.model)

    print("=== start search ===")


    if "opt" in args.model:
        cache_dataloader = (
            f"./Scale_get/dataloader_opt_mix_128_mix.cache"
        )
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            print(f"load calibration from {cache_dataloader}")
        else:
            dataloader = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
                cache_dir=args.cache_dir,
            )
            torch.save(dataloader, cache_dataloader)
        lm.model.eval()
    else:
        raise NotImplementedError()

    if "opt" in args.model:
        opt_reorder_quantize(
            lm,
            args,
            dataloader,
        )

if __name__ == "__main__":
    print(sys.argv)
    main()