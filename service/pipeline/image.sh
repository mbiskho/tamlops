#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$1 python3 image.py --resolution=$2 --train_batch_size=$3 --num_train_epochs=$4 --max_train_steps=$5 --learning_rate=$6 --gradient_accumulation_steps=$7 --file=$8 --id=$9"

CUDA_VISIBLE_DEVICES=$1 python3 image.py --resolution=$2 --train_batch_size=$3 --num_train_epochs=$4 --max_train_steps=$5 --learning_rate=$6 --gradient_accumulation_steps=$7 --file=$8 --id=$9
