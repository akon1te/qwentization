#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL_PATH> <OUTPUT_PATH>"
    echo "Example: $0 Qwen/Qwen2.5-7B-Instruct ./checkpoints/Qwen2.5-7B-AWQ-MMLU"
    exit 1
fi

MODEL_PATH=$1
OUTPUT_PATH=$2

echo "=========================================="
echo "Starting AWQ Quantization on MMLU Dataset"
echo "Source Model: $MODEL_PATH"
echo "Target Path:  $OUTPUT_PATH"
echo "=========================================="


python quantize_mmlu.py \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --w_bit 4 \
    --group_size 128 \
    --n_samples 128


if [ $? -eq 0 ]; then
    echo "SUCCESS: Model saved to $OUTPUT_PATH"
else
    echo "ERROR: Quantization failed."
    exit 1
fi
