# Qwentization Lab

Stage1:

1.  **MMLU-Calibrated AWQ:** 4-bit quantization using activation statistics from the MMLU dataset.
2.  **LM Head SVD:** Singular Value Decomposition (SVD) applied to the massive MMLU output layer (`lm_head`) to reduce parameter count and memory footprint.

Stage2:

1. **PEFT trainig:** Training of AWQ-quantized via LORA on MMLU aluxuairy dataset part

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

*   `Qwen3-8B-AWQ` - Best model of Stage1
*   `Qwen3-8B-AWQ-SVD ` - Second model from Stage1


## Usage

### Stage 1: AWQ Quantization (MMLU)

This step converts the generic FP16 model to INT4, using the MMLU validation set to calibrate weights.

**Syntax:**
```bash
./run_quant.sh <SOURCE_MODEL_ID> <OUTPUT_DIR>
```

**Example:**
```bash
./run_quant.sh Qwen/Qwen3-8B ./models/Qwen3-8B-AWQ-MMLU
```
*   *Default settings: 4-bit, group_size=128, 128 calibration samples.*

> **Note:** \
Accuracy on dev MMLU (10 samples per category): **0.7368** \
VRAM consuming: **3280.76 MB**


---

### Stage 1.1: Tensor Decomposition (SVD)

This step takes the AWQ model, dequantizes the `lm_head` layer, performs SVD, and replaces the layer with two smaller low-rank matrices.

**Syntax:**
```bash
./run_svd.sh <AWQ_MODEL_DIR> <OUTPUT_DIR> [RANK_RATIO]
```

**Example:**
```bash
# Reduces the head rank to 80% of original
./run_svd.sh ./models/Qwen3-8B-AWQ-MMLU ./models/Qwen3-8B-AWQ-SVD 0.8
```

> **Note:** \
Accuracy on dev MMLU (10 samples per category): **0.6982** \
VRAM consuming: **5787.29 MB**

