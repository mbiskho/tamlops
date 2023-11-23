import joblib
import os
from google.cloud import storage
from transformers import T5Tokenizer, T5ForConditionalGeneration
import shutil
from datasets import load_dataset, concatenate_datasets
from random import randrange
from huggingface_hub import HfFolder
import argparse
import logging
import math
import random
import shutil
from pathlib import Path
import accelerate
import datasets
import numpy as np
import gc

# Numpy Error
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)


import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import requests
import json
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datasets import DatasetDict, Dataset
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import time
import csv
import os
import psutil


datas = []

for learning in [0.0001, 0.001, 0.01, 0.1]:
    for dataset in [50, 100, 200, 400, 800, 1600]:
        for train_size in [2, 4, 8 ,16]: 
            for batch_size in [2,4, 8, 16]:
                for epoch in [2, 6 ,10]:
                    datas.append({
                    "file": "https://storage.googleapis.com/training-dataset-tamlops/test_fa9a2823-e86d-4630-9878-33adde1dd4e8.json",
                    "param": {
                    "per_device_train_batch_size": train_size,
                    "per_device_eval_batch_size": batch_size,
                    "learning_rate": learning,
                    "num_train_epochs": epoch,
                    "num_dataset": dataset
                    }
                })


def write_to_csv(data, csv_file_path):
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the data to the CSV file
        writer.writerow(data)

def download_data(url, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Extract filename from URL
    filename = url.split('/')[-1]
    file_path = os.path.join(folder_path, filename)

    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Write the content to the file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded file '{filename}' to '{folder_path}'")
        return file_path
    else:
        print(f"Failed to download file from {url}")
        return None

def load_data(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read and return the content of the file
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    else:
        print(f"File '{file_path}' not found.")
        return None

def remove_file(file_path):
    # Check if the file exists and remove it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' removed.")
    else:
        print(f"File '{file_path}' not found.")


def preprocess_function(sample, padding="max_length"):
    inputs = ["summarize: " + item for item in sample["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
    print(sample)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


for data in datas:
    print("")
    folder = './text_data'
    
    # Start Time of All
    start_all = datetime.now()
    downloaded_file_path = download_data(data['file'], folder)

    # Open the JSON file and load its content into a dictionary
    with open(downloaded_file_path, 'r') as json_file:
        datas = json.load(json_file)

    datas = datas[:data['param']['num_dataset']]
    id = [item["id"] for item in datas]
    dialogue = [item["dialogue"] for item in datas]
    summaries = [item["summary"] for item in datas]

    # ID
    id_train, id_test = train_test_split(id, test_size=0.3, random_state=42)
    id_test, id_validation = train_test_split(id_test, test_size=0.5, random_state=42)

    # Dialogue
    dialogue_train, dialogue_test = train_test_split(dialogue, test_size=0.3, random_state=42)
    dialogue_test, dialogue_validation = train_test_split(dialogue_test, test_size=0.5, random_state=42)

    # Summaries
    summaries_train, summaries_test = train_test_split(summaries, test_size=0.3, random_state=42)
    summaries_test, summaries_validation = train_test_split(summaries_test, test_size=0.5, random_state=42)

    dataset = DatasetDict({
        'train': Dataset.from_dict({"id": id_train, "dialogue": dialogue_train, "summary": summaries_train}),
        'test': Dataset.from_dict({"id": id_test, "dialogue": dialogue_test, "summary": summaries_test}),
        'validation': Dataset.from_dict({"id": id_validation, "dialogue": dialogue_validation, "summary": summaries_validation})
    })

    start_trainning = datetime.now()

    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset['train'], dataset['test']]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset['train'], dataset['test']]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=1
    )
    # Define training args
    repository_id = "mbiskho/text"
    training_args = Seq2SeqTrainingArguments(
        output_dir="/root/tamlops/service/pipeline/example/tmp",
        per_device_train_batch_size=data['param']['per_device_train_batch_size'],
        per_device_eval_batch_size=data['param']['per_device_eval_batch_size'],
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=data['param']['learning_rate'],
        num_train_epochs=data['param']['num_train_epochs'],
        # logging & evaluation strategies
        # logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token="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",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        # compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    print("[!] Flan T5 Has been Trained")
    end_trainning = (datetime.now() - start_trainning)

    print("[!] Time Trainnig: ", end_trainning)
    trainer.push_to_hub()


    # End Time of All
    end_all = (datetime.now() - start_all)


    # TLDR
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
    gpu_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"GPU Usage: {gpu_usage:.2f}GB / {gpu_memory:.2f}GB")

    to_logs = {
        'train_execution': end_trainning,
        'overall_execution': end_all,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_usage,
        'num_train_epochs': data['param']['num_train_epochs'],
        'learning_rate': data['param']['learning_rate'],
        'per_device_eval_batch_size': data['param']['per_device_eval_batch_size'],
        'per_device_train_batch_size': data['param']['per_device_train_batch_size'],
        'num_dataset': data['param']['num_dataset']
    } 

    write_to_csv(to_logs, 'text.csv')
    
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

