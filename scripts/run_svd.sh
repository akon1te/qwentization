#!/bin/bash
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <MODEL_PATH> <OUTPUT_PATH> [RANK_RATIO]"
    echo "Example: $0 ./Qwen3-8B-AWQ-MMLU ./Qwen3-8B-AWQ-SVD 0.8"
    exit 1
fi

MODEL_PATH=$1
OUTPUT_PATH=$2
RANK_RATIO=${3:-0.8} 

echo "=========================================="
echo "Starting SVD Decomposition on lm_head"
echo "Source Model: $MODEL_PATH"
echo "Target Path:  $OUTPUT_PATH"
echo "Rank Ratio:   $RANK_RATIO"
echo "=========================================="


python apply_svd.py \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --rank_ratio "$RANK_RATIO"


if [ $? -eq 0 ]; then
    echo "SUCCESS: SVD applied. Model saved to $OUTPUT_PATH"
else
    echo "ERROR: Decomposition failed."
    exit 1
fi
