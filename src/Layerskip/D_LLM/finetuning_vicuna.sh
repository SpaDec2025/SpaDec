#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=/home/models/vicuna-7b-v1.5
MODEL_PARAM_PATH=vicuna_params.json
DATASET_PATH=/datasets
OUTPUT_PATH=/spadec/models/d-llm

torchrun --nproc_per_node 1 --master_port 9903 finetuning_vicuna.py \
    --model_save_name Vicuna-7b-dllm \
    --llama_model_path $MODEL_PATH \
    --llama_param_path $MODEL_PARAM_PATH \
    --tokenizer_path $MODEL_PATH/tokenizer.model \
    --dataset_path $DATASET_PATH \
    --dataset_name alpaca \
    --max_seq_len 1024 \
    --lora_rank 8 \
    --dynamic_active_target 0.50 \
    --dynamic_router_hdim 512 \
    --dynamic_start_layer 2 \
    --dynamic_reserve_initials 2 \
    --lambda_active 5.0 \
    --batch_size 1 \
    --epochs 15 \
    --warmup_epochs 2 \
    --save_freq 1 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $OUTPUT_PATH
