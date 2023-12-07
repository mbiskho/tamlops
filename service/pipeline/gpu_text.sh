#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$1 python3 text.py --per_device_train_batch_size=$2 --per_device_eval_batch_size=$3 --learning_rate=$4 --num_train_epochs=$5 --file=$6 --id=$7"

CUDA_VISIBLE_DEVICES=$1 python3 text.py --per_device_train_batch_size=$2 --per_device_eval_batch_size=$3 --learning_rate=$4 --num_train_epochs=$5 --file=$6 --id=$7