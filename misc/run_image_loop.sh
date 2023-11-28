#!/bin/bash

# Define parameters for the loop
resolution=(256 512 1024)
train_size=(2 4 8 16)
epoch=(2 6 10)
max_train_steps=(100 500 1000)
gradient_accumulation_steps=(1 2 4)
learning_rates=(0.0001 0.001 0.01)
dataset=(50 100 200 400 800 1600)

# Loop through combinations of parameters
for lr in "${learning_rates[@]}"; do
    for ds in "${dataset[@]}"; do
        for ts in "${train_size[@]}"; do
            for ep in "${epoch[@]}"; do
                for res in "${resolution[@]}"; do
                    for mts in "${max_train_steps[@]}"; do
                        for gas in "${gradient_accumulation_steps[@]}"; do
                            CUDA_VISIBLE_DEVICES=0 python image_param.py --learning_rate "$lr" --num_dataset "$ds" --train_batch_size "$ts" --batch_size "$bs" --num_train_epochs "$ep" --resolution "$res" --max_train_steps "$mts" --gradient_accumulation_steps "$gas"
                        done
                    done
                done
            done
        done
    done
done