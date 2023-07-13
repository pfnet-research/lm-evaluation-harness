#!/bin/bash
set -eu

PROJECT_DIR=""
PRETRAINED="${PROJECT_DIR}/sft/checkpoints/stablelm-jp-instruct-3b_1.3.0/"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
NUM_FEW_SHOTS="2,3,3,3"
OUTPUT_PATH="models/stablelm-jp-instruct-3b_1.3.0/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --no_cache \
    --output_path $OUTPUT_PATH