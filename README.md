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
*   [Qwen3-8B-AWQ-SVD](https://huggingface.co/akon1te/qwen3-8b-awq-mmlu-lora) - Best adapter from Stage2


## Usage


### Stage 0: Baseline eval

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7439** \
VRAM consuming: **17360 MiB** 


### Stage 1: AWQ Quantization (MMLU)

This step converts the generic FP16 model to INT4, using the MMLU validation set to calibrate weights.

*   *Default settings: 4-bit, group_size=128, 128 calibration samples.*

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7298** \
VRAM consuming: **7406 MiB** 


---

### Stage 1.1: Tensor Decomposition (SVD)

This step takes the AWQ model, dequantizes the `lm_head` layer, performs SVD, and replaces the layer with two smaller low-rank matrices.

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7018** \
VRAM consuming: **6838MiB**


### Stage 2: Train LORA

Train on Aluxuary part on MMLU:
- 10000 samples
- Lr= 1e-5

> **Results:** \
Accuracy on dev MMLU (10 samples per category): **0.7368** \
VRAM consuming: **7730 MiB**