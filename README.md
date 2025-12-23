# Qwentization Lab

Stage1:

1.  **MMLU-Calibrated AWQ:** 4-bit quantization using activation statistics from the MMLU dataset.
2.  **LM Head SVD:** Singular Value Decomposition (SVD) applied to the massive MMLU output layer (`lm_head`) to reduce parameter count and memory footprint.

Stage2:

1. **LORA trainig:** Training of AWQ-quantized via LORA on MMLU aluxuairy dataset part

## Installation

Ensure you have Python 3.10+ and CUDA installed.

```bash
pip install reqs.txt
```

## Repository Structure

*   `quantize_mmlu.py` - Core script for MMLU data loading and AWQ quantization;
*   `apply_svd.py` - Script for dequantizing the LLM head and applying SVD;
*   `train_lora.py` - Script for PEFT training;
*   `mmlu_benchmark.py` - Script for models evaluation on MMLU dev 


## Models of HuggingFace HUB

*   [Qwen3-8B AWQ](https://huggingface.co/akon1te/qwen3-8b-awq) - Best model of Stage1
*   [Qwen3-8B-AWQ-Lora](https://huggingface.co/akon1te/qwen3-8b-awq-mmlu-lora) - Best adapter from Stage2
*   [Qwen3-8B AWQ SVD_lm_head](https://huggingface.co/akon1te/qwen3-8b-awq_lm-head-svd) - SVD of lm_head model


### SVD model loading

> weights = hf_hub_download(repo_id="akon1te/qwen3-8b-awq_lm-head-svd", filename="model.pt")\
base_model = torch.load(weights, weights_only=False)


## Results 

### Stage 0: Baseline eval

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7439** \
VRAM consuming: **17360 MiB** 


### Stage 1: AWQ Quantization (MMLU)

This step converts the generic FP16 model to INT4, using the MMLU validation set to calibrate weights.

*   *Default settings: 4-bit, group_size=128, 128 calibration samples.*
*   Accuracy on dev MMLU (10 samples per category): **0.7298**
*   VRAM consuming: **7406 MiB** 



> **Results:** \
**Compression Ratio:** $17360 / 7406 \approx 2.34$\
**Performance Drop:** 
$(0.7439 − 0.7298) / 0.7439 \approx 0.019$\
**Score:** $2.34 / (1 + 0.019) = 2.30$


---

### Stage 1.1: Tensor Decomposition (SVD)

This step takes the AWQ model, dequantizes the `lm_head` layer, performs SVD, and replaces the layer with two smaller low-rank matrices.


*   Accuracy on dev MMLU (10 samples per category): **0.7018** \
*   VRAM consuming: **6838MiB**


> **Results:** \
**Compression Ratio:** $17360 / 6838 \approx \mathbf{2.54}$ \
**Performance Drop:** 
$(0.7439−0.7018)/0.7439 \approx 0.057$\
**Score:** $2.54 / (1 + 0.057) = 2.54 / 1.057 \approx \mathbf{2.40}$


### Stage 2.1: Train LORA on Qwen3-AWQ

Train on Aluxuary part on MMLU:
- 10000 samples
- Lr= 1e-5

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7368** \


## Немного выводов

На самом деле для первого этапа судя по самоподсчитанным скорам надо было брать модель с SVD, но из-за "фриза" на работе, я не довел до ума тренировку с SVD. 
Там был около нулевой прирост на валидации и с Lora, и Full-FT новой головы, поэтому за базу считаю просто AWQ модель.

