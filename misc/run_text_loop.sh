#!/bin/bash

# Define parameters for the loop
learning_rates=(0.0001 0.001 0.01)
train_sizes=(2 4 8 16)
batch_sizes=(2 4 8 16)
epochs=(2 6 10)
num_datasets=(50 100 200 400 800)

# Loop through combinations of parameters
for lr in "${learning_rates[@]}"; do
    for ts in "${train_sizes[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for ep in "${epochs[@]}"; do
                for nd in "${num_datasets[@]}"; do
                    CUDA_VISIBLE_DEVICES=7 python text_param.py --learning_rate "$lr" --train_size "$ts" --batch_size "$bs" --epoch "$ep" --num_dataset "$nd"
                done
            done
        done
    done
done
