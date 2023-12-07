#!/bin/bash

echo "python3 image.py --resolution=$1 --train_batch_size=$2 --num_train_epochs=$3 --max_train_steps=$4 --learning_rate=$5 --gradient_accumulation_steps=$6 --file=$7 --id=$8"

python3 image.py --resolution=$1 --train_batch_size=$2 --num_train_epochs=$3 --max_train_steps=$4 --learning_rate=$5 --gradient_accumulation_steps=$6 --file=$7 --id=$8