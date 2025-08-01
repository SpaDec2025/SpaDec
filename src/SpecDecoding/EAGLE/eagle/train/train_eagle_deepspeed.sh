#!/bin/bash

DEEPSPEED_ARGS=" \
    --num_gpus 5 \
    --num_nodes 1 \
    --master_port 9095 \
"

deepspeed $DEEPSPEED_ARGS main_deepspeed.py --deepspeed_config ds_config.json