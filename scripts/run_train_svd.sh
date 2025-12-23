#!/bin/bash

# Проверяем аргументы
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <SVD_MODEL_DIR> <OUTPUT_ADAPTER_DIR>"
    echo "Example: $0 ./Qwen3-8B-AWQ-SVD-MMLU ./Qwen3-8B-AWQ-SVD-MMLU-fft"
    exit 1
fi

SVD_MODEL_DIR=$1
OUTPUT_DIR=$2

echo "=========================================="
echo "Starting LoRA Training on SVD+AWQ Model"
echo "Input Dir (must contain model.pt): $SVD_MODEL_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Формируем команду
CMD="python ./src/train_lora_svd.py \
    --model_dir $SVD_MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_limit 20000 \
    --batch_size 8 \
    --grad_acc 4 \
    --lr 1e-5 \
    --epochs 1"

# Если задан HUB_ID, добавляем флаги загрузки
if [ ! -z "$HUB_ID" ]; then
    CMD="$CMD --push_to_hub --hub_model_id $HUB_ID"
fi

# Запуск
eval $CMD

if [ $? -eq 0 ]; then
    echo "SUCCESS: Training finished."
else
    echo "ERROR: Training failed."
    exit 1
fi
