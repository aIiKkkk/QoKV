import os
import sys

import random
import numpy as np
from models.opt import OPTClass
from transformers import  AutoModelForCausalLM
import torch
import time
from datautils import get_loaders
from lm_evaluation.lm_eval import tasks, evaluator
import datetime
from models.int_opt_layer import QuantOPTAttention
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.opt_reorder_quantize import opt_reorder_quantize
from quantize.llama_reorder_quantize import llama_reorder_quantize
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-3-8b",
    "Llama-3-70b",
]

# tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq

def get_llama(model_name):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(f"./Model_data/{model_name}",
                                                 torch_dtype=torch.float16, device_map='auto')
    model.seqlen = 2048
    return model

class RMSNorm(torch.nn.Module):
    def __init__(self, ori, dev, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dev = dev
        self.weight = nn.Parameter(ori.weight).to(dev)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = (self._norm(x.float()).type_as(x)).to(self.dev)
        return output * self.weight


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model, model_name):
        model.eval()

        for dataset in ["wikitext2", "ptb", "c4"]:
            if "Llama-2" in model_name:
                cache_testloader = f".{dataset}_testloader_Llama-2_all.cache"
            elif "Llama-3" in model_name:
                cache_testloader = f".{dataset}_testloader_Llama-3_all.cache"
            if os.path.exists(cache_testloader):

                testloader = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                # testloader = load_dataset('json', data_files="/data/wangjinguang/dataset/c4-05-10/c4-validation.00000-of-00008.json.gz", split='train')
                # testloader = testloader.shuffle(seed=42)
                trainloader, testloader = get_loaders(dataset, model=model_name,
                                                      cache_dir=f"./Model_data/{model_name}")
                torch.save(testloader, cache_testloader)
                # print(testloader)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            seqlen = 2048
            nsamples = testenc.numel() // seqlen
            model.config.use_cache = False
            nlls = []
            model.model.norm = self.norm = RMSNorm(model.model.norm, model.device)
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
                # print(model.model.layers[0].self_attn.q_proj.weight.device)
                outputs = model.model(batch)
                hidden_states = outputs[0]
                # print(hidden_states.device)
                logits = model.lm_head(hidden_states.to(model.lm_head.weight.device))

                shift_logits = logits[:, :-1, :]
                # print(shift_logits.shape)
                shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                               :, 1:
                               ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            print(dataset, ppl.item())
        return ppl


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
    elif "Llama" in args.net:
        lm = get_llama(args.net)
        lm.eval()

    print("=== start search ===")

    tick = time.time()

    if "opt" in args.net:
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
                model=args.net,
                seqlen=lm.seqlen,
                cache_dir=args.cache_dir,
            )
            torch.save(dataloader, cache_dataloader)
        lm.model.eval()
    elif "Llama" in args.net:
        cache_dataloader = (
            f"./Scale_get/dataloader_{args.net}_mix_128.cache"
        )
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            print(f"load calibration from {cache_dataloader}")
        else:
            dataloader = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.net,
                seqlen=lm.seqlen,
                cache_dir = f"./Model_data/{args.net}"
            )
            torch.save(dataloader, cache_dataloader)
        lm.model.eval()
    else:
        raise NotImplementedError()
    if "opt" in args.net:
        opt_reorder_quantize(
            lm,
            args,
            dataloader,
        )
    elif "Llama" in args.net:
        llama_reorder_quantize(
            lm,
            args,
            dataloader,
        )


if __name__ == "__main__":
    print(sys.argv)
    main()