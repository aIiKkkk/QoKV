# QoKV: Comprehending and Surpassing the Hurdles of KV Cache Quantization

## Abstract
Large language models (LLMs) have demonstrated outstanding performance in various tasks. However,  the  memory footprint of key-value (KV) cache generated during the model inference, which may be several times the model sizes, poses challenges for the efficient model deployment. Previous researchs have primarily focused on the impediment posed by the outliers in the activation quantifications, with limited attention given to the low-bit quantization of KV cache. This paper provides an in-depth analysis of KV cache and the potential sources of its quantization errors. We find that the attention scores exhibit a strong power-law distribution while showing an additional attention to the initial and recent tokens, and that only a small subset of outlier channels in KV cache span a wide numerical range. Based on these findings, we propose QoKV, an accurate low-bit quantization method designed for the KV cache. QoKV employs a grouping evaluation strategy based on the average attention scores to evaluate the importance of different tokens and uses high-bit quantization to protect the important tokens. In addition, the channel scaling is employed to balance the differences between the different channels, thereby mitigating the  adverse effect  of the outlier channels with large numerical ranges on quantization scales. The proposed framework preserves the model performance under the 4-bit quantization of KV cache while incurring only a small computational overhead. Extensive experimental results on different tasks show that the proposed method outperforms existing methods and can achieve near-floating-point performance while saving 73.75\% of the cache footprint.

## Installation
```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets zstandard
```

### Activation Scales and Calibration
We provide the attention groups for OPT models in [Get_group/](main.py). And We provide the KVscale for OPT models in [Scale_get/](main.py).
