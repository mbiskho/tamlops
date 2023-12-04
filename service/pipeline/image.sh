#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$0 python3 image.py --resolution=$1 --train_batch_size=$2 --num_train_epochs=$3 --max_train_steps=$4 --learning_rate=$5 --gradient_accumulation_steps=$6 --file=$7 --id=$8"

CUDA_VISIBLE_DEVICES=$0 python3 image.py --resolution=$1 --train_batch_size=$2 --num_train_epochs=$3 --max_train_steps=$4 --learning_rate=$5 --gradient_accumulation_steps=$6 --file=$7 --id=$8