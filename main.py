import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from smooth import smooth_lm
from smooth_bloom import smooth_bloomlm
from fake_quant import W8A8Linear
from Attention import QuantOPTAttention
from Llama_attention import LlamaAttention
# from smoothquant.Cal_Attention_scale import CalScaleOPTAttention
from tqdm import tqdm
import torch.nn as nn
import os
from pprint import pprint
from opt import OPTClass


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

def quantize_OPT_model(model, group, weight_quant='per_tensor', act_quant='per_token', quantize_bmm_input=True):
    embed_dim = model.model.decoder.layers[0].embed_dim
    num_heads = model.model.decoder.layers[0].self_attn.num_heads
    for i in tqdm(range(len(model.model.decoder.layers))):
        for name, m in model.model.decoder.layers[i].named_modules():
            if isinstance(m, OPTDecoderLayer):
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
                m.self_attn = QuantOPTAttention(m.self_attn, group[i], embed_dim, num_heads, weight_quant, act_quant, quantize_bmm_input)
    return model


def quantize_Llama_model(model, K_scales, V_scales, group, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    # print(model)
    device, dtype = model.model.layers[0].self_attn.q_proj.weight.device, model.model.layers[0].self_attn.q_proj.weight.dtype
    for i in tqdm(range(len(model.model.layers))):
        for name, m in model.model.layers[i].named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.mlp.gate_proj = W8A8Linear.from_float(
                    m.mlp.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
                m.mlp.up_proj = W8A8Linear.from_float(
                    m.mlp.up_proj, weight_quant=weight_quant, act_quant=act_quant)
                m.mlp.down_proj = W8A8Linear.from_float(
                    m.mlp.down_proj, weight_quant=weight_quant, act_quant=act_quant)
                m.self_attn = LlamaAttention(m.self_attn, model.config, K_scales[i].to(device=device, dtype=dtype), V_scales[i].to(device=device, dtype=dtype), group[i],
                                             weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True, layer_idx = i)
    return model


def quantize_model_opt(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    for name, m in tqdm(model.model.named_modules()):
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m,  OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


def quantize_model_llama(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    for i in tqdm(range(len(model.model.layers))):
        model.model.layers[i].self_attn.q_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.q_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.k_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.k_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.v_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.v_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.o_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.o_proj, weight_quant=weight_quant, act_quant=act_quant)


        model.model.layers[i].mlp.gate_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.up_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.up_proj, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.down_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.down_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        for dataset in ["wikitext2", "ptb", "c4"]:
            cache_testloader = f"{dataset}_testloader_opt_all.cache"

            testloader = torch.load(cache_testloader)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            seqlen = 2048
            nsamples = testenc.numel() // seqlen
            model.config.use_cache = False
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.lm_head.weight.device)
                outputs = model.model.decoder(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
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

class Evaluator_Llama:
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
                cache_testloader = f"{dataset}_testloader_Llama-2_all.cache"
            elif "Llama-3" in model_name:
                cache_testloader = f"{dataset}_testloader_Llama-3_all.cache"
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


from datasets import load_from_disk
model_name = "Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(f"./Model_data/{model_name}")
dataset = load_dataset('lambada_openai', split='validation[:1000]')
print("Loaded dataset from {./dataset/lambada_openai}")
if "Llama" in model_name:
    evaluator_PPL = Evaluator_Llama(dataset, tokenizer, 'cuda')
else:
    evaluator_PPL = Evaluator(dataset, tokenizer, 'cuda')


compare_smoothw8a8 = True
if compare_smoothw8a8:
    model = AutoModelForCausalLM.from_pretrained(f"./Model_data/{model_name}",
                                       torch_dtype=torch.float16, device_map='auto')

    act_scales = torch.load(f'./act_scales/{model_name}.pt') #
    K_scales = torch.load(f'./Scale_get/act_scales/{model_name}Testk2.pt')
    V_scales = torch.load(f'./Scale_get/act_scales/{model_name}Testv2.pt')
    smooth_lm(model, act_scales, K_scales, V_scales, 0.8)
    print("Starting quantize_lm")
    group = torch.load(f'./Get_group/act_scales/{model_name}group.pt')
    model_smoothquant_w8a8 =  quantize_Llama_model(model, K_scales, V_scales, group)
    print("Starting evaluate")
    acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_smoothquant_w8a8, model_name)

