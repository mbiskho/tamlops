#!/bin/bash

docker push mbiskho/tamlops-trainning
python3 image.py --resolution=1 --train_batch_size=1 --num_train_epochs=1 --max_train_steps=1 --learning_rate=0.001 --gradient_accumulation_steps=1 --file=https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet --id=123

