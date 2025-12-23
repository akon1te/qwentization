#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL_PATH> <OUTPUT_DIR>"
    echo "Example: $0 ./models/Qwen3-8B-AWQ-MMLU ./models/Qwen3-8B-AWQ-MMLU-lora"
    exit 1
fi

MODEL_PATH=$1
OUTPUT_DIR=$2

echo "=========================================="
echo "Starting LoRA Fine-Tuning"
echo "Base Model: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

python src/train_lora.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_limit 10000 \
    --batch_size 6 \
    --grad_acc 4 \
    --lr 1e-5 \
    --epochs 1

if [ $? -eq 0 ]; then
    echo "SUCCESS: Adapters saved to $OUTPUT_DIR"
else
    echo "ERROR: Training failed."
    exit 1
fi
