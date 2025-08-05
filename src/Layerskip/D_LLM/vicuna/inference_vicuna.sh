export CUDA_VISIBLE_DEVICES=4

MODEL_PATH=/home/models/vicuna-7b-v1.5
DYNAMIC_MODEL_PATH=$lora_checkpoint_path

torchrun --nproc_per_node 1 --master_port 29223 example_vicuna.py \
    --llama_ckpt_dir $MODEL_PATH \
    --dynamic_ckpt_dir $DYNAMIC_MODEL_PATH \
    --model_args_path vicuna_params.json \
    --tokenizer_path $MODEL_PATH/tokenizer.model \
    --instructs "['How to start learning guitar and become a master at it?']"
