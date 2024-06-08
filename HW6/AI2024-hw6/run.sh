#!/bin/bash

if [ -z "$4" ] && [ -z "$5" ]; then
    python main.py \
        --exp_name "${1}" \
        --model_name "${2}" \
        --train \
        --wandb_token "${3}" 

elif [ -z "$5" ]; then
    python main.py \
        --exp_name "${1}" \
        --model_name "${2}" \
        --train \
        --wandb_token "${3}" \
        --num_epochs "${4}"
else
    python main.py \
        --exp_name "${1}" \
        --model_name "${2}" \
        --train \
        --wandb_token "${3}" \
        --num_epochs "${4}" \
        --beta "${5}"
fi
