#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$0 python3 text.py --per_device_train_batch_size=$1 --per_device_eval_batch_size=$2 --learning_rate=$3 --num_train_epochs=$4 --file=$5 --id=$6"

CUDA_VISIBLE_DEVICES=$0 python3 text.py --per_device_train_batch_size=$1 --per_device_eval_batch_size=$2 --learning_rate=$3 --num_train_epochs=$4 --file=$5 --id=$6