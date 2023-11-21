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


# data = {
#     "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
#     "param": {
#       ""
#       "resolution": 10, # 512
#       "train_batch_size": 1, #6
#       "num_train_epochs": 1, #100
#       "max_train_steps": 10,
#       "gradient_accumulation_steps": 1,
#       "learning_rate": 0.0001,

#       "num_dataset": 100
#     }
# }


datas = [
    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },

    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },
    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },

]


logger = get_logger(__name__, log_level="INFO")
# Start Time of All
start_all = datetime.now()

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

def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args['validation_prompts']))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {args['pretrained_model_name_or_path']}
    datasets:
    - {args['dataset_name']}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - text-to-image
    - diffusers
    inference: true
    ---
    """
    model_card = f"""
    # Text-to-image finetuning - {repo_id}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
    image = pipeline(prompt).images[0]
    image.save("my_image.png")
    ```

    ## Training info

    These are the key hyperparameters used during training:

    * Epochs: {args['num_train_epochs']}
    * Learning rate: {args['learning_rate']}
    * Batch size: {args['train_batch_size']}
    * Gradient accumulation steps: {args['gradient_accumulation_steps']}
    * Image resolution: {args['resolution']}
    * Mixed-precision: {args['mixed_precision']}

    """

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

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

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args['pretrained_model_name_or_path'],
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args['revision'],
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args['enable_xformers_memory_efficient_attention']:
        pipeline.enable_xformers_memory_efficient_attention()

    if args['seed'] is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args['seed'])

    images = []
    for i in range(len(args['validation_prompts'])):
        with torch.autocast("cuda"):
            image = pipeline(args['validation_prompts[i]'], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args['validation_prompts[i]']}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

for data in datas:

    args = {
        "input_perturbation": 0.0,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1",
        "revision": None,
        "dataset_name": "lambdalabs/pokemon-blip-captions",
        "dataset_config_name": None,
        "train_data_dir": None,
        "image_column": "image",
        "caption_column": "text",
        "max_train_samples": None,
        "validation_prompts": None,
        "output_dir": "sd-model",
        "cache_dir": None,
        "seed": None,
        "resolution": data['param']['resolution'], #512
        "center_crop": False,
        "random_flip": False,
        "train_batch_size": data['param']['train_batch_size'], #6
        "num_train_epochs": data['param']['num_train_epochs'], #100
        "max_train_steps": data['param']['max_train_steps'],
        "gradient_accumulation_steps": data['param']['gradient_accumulation_steps'],
        "gradient_checkpointing": False,
        "learning_rate": data['param']['learning_rate'] ,
        "scale_lr": False,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "snr_gamma": None,
        "use_8bit_adam": False,
        "allow_tf32": False,
        "use_ema": False,
        "non_ema_revision": None,
        "dataloader_num_workers": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 0.01,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "push_to_hub": True,
        "hub_token": "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",
        "prediction_type": None,
        "hub_model_id": "mbiskho/tamlops-image",
        "logging_dir": "logs",
        "mixed_precision": None,
        "report_to": "tensorboard",
        "local_rank": -1,
        "checkpointing_steps": 500,
        "checkpoints_total_limit": None,
        "resume_from_checkpoint": None,
        "enable_xformers_memory_efficient_attention": False,
        "noise_offset": 0.0,
        "validation_epochs": 5,
        "tracker_project_name": "text2image-fine-tune",
    }

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
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs




    folder = './image_data'

    downloaded_file_path = download_data(data['file'], folder)

    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(downloaded_file_path)

    # Converting
    converted = []
    for index, row in df.iterrows():
        cells = {}
        for column_name, cell_value in row.items():
            cells[column_name] = cell_value
        # Create a BytesIO object using the byte string
        bytes_io = BytesIO(cells['image']['bytes'])

    # Open the BytesIO object as an image using PIL
    image = Image.open(bytes_io)
    cells['image'] = image
    converted.append(cells)


    if args['non_ema_revision'] is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    logging_dir = os.path.join(args['output_dir'], args['logging_dir'])

    accelerator_project_config = ProjectConfiguration(project_dir=args['output_dir'], logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        mixed_precision=args['mixed_precision'],
        log_with=args['report_to'],
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args['seed'] is not None:
        set_seed(args['seed'])

    # Handle the repository creation
    if accelerator.is_main_process:
        if args['output_dir'] is not None:
            os.makedirs(args['output_dir'], exist_ok=True)

        if args['push_to_hub']:
            repo_id = create_repo(
                repo_id=args['hub_model_id'] or Path(args['output_dir']).name, exist_ok=True, token=args['hub_token'],
            ).repo_id
        print("REPO ID: ", repo_id)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args['pretrained_model_name_or_path'], subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args['pretrained_model_name_or_path'], subfolder="tokenizer", revision=args['revision']
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args['pretrained_model_name_or_path'], subfolder="text_encoder", revision=args['revision']
        )
        vae = AutoencoderKL.from_pretrained(
            args['pretrained_model_name_or_path'], subfolder="vae", revision=args['revision']
        )

    unet = UNet2DConditionModel.from_pretrained(
        args['pretrained_model_name_or_path'], subfolder="unet", revision=args['non_ema_revision']
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args['use_ema']:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args['pretrained_model_name_or_path'], subfolder="unet", revision=args['revision']
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args['enable_xformers_memory_efficient_attention']:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args['use_ema']:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args['use_ema']:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()
    if args['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args['scale_lr']:
        args['learning_rate'] = (
            args['learning_rate'] * args['gradient_accumulation_steps'] * args['train_batch_size'] * accelerator.num_processes
        )

    # Initialize the optimizer
    if args['use_8bit_adam']:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args['learning_rate'],
        betas=(args['adam_beta1'], args['adam_beta2']),
        weight_decay=args['adam_weight_decay'],
        eps=args['adam_epsilon'],
    )

    # Get The Dataset
    img = [item['image'] for item in converted[:data['param']['num_dataset']]]
    text = [item['text'] for item in converted[:data['param']['num_dataset']]]
    dataset['train'] = Dataset.from_dict({"text": text, "image": img})


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args['dataset_name'], None)
    if args['image_column'] is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args['image_column']
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args['image_column']}' needs to be one of: {', '.join(column_names)}"
            )
    if args['caption_column'] is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args['caption_column']
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args['caption_column']}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args['resolution'], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args['resolution']) if args['center_crop'] else transforms.RandomCrop(args['resolution']),
            transforms.RandomHorizontalFlip() if args['random_flip'] else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args['max_train_samples'] is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args['train_batch_size'],
        num_workers=args['dataloader_num_workers'],
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args['gradient_accumulation_steps'])
    if args['max_train_steps'] is None:
        args['max_train_steps'] = args['num_train_epochs'] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=args['lr_warmup_steps'] * accelerator.num_processes,
        num_training_steps=args['max_train_steps'] * accelerator.num_processes,
    )

    if torch.cuda.is_available():
        weight_dtype = torch.float32
        torch.device("cuda")
    else:
        weight_dtype = torch.float32
        torch.device("cpu")

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args['gradient_accumulation_steps'])
    if overrode_max_train_steps:
        args['max_train_steps'] = args['num_train_epochs'] * num_update_steps_per_epoch
    args['num_train_epochs'] = math.ceil(args['max_train_steps'] / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = args
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args['tracker_project_name'], tracker_config)

    # Train!
    start_trainning = datetime.now()
    print(accelerator.device)
    total_batch_size = args['train_batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {args['train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {args['max_train_steps']}")
    global_step = 0
    first_epoch = 0

    if args['resume_from_checkpoint']:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args['resume_from_checkpoint'])
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args['output_dir'])
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args['resume_from_checkpoint']}' does not exist. Starting a new training run."
            )
            args['resume_from_checkpoint'] = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args['output_dir'], path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args['max_train_steps']),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    # Start The Epocs
    for epoch in range(first_epoch, args['num_train_epochs']):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args['resume_from_checkpoint']:
                    noise += args['noise_offset'] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args['input_perturbation']:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                if args['input_perturbation']:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                if args['prediction_type'] is not None:
                    noise_scheduler.register_to_config(prediction_type=args['prediction_type'])

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args['snr_gamma'] is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args['snr_gamma'] * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args['train_batch_size'])).mean()
                train_loss += avg_loss.item() / args['gradient_accumulation_steps']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args['max_grad_norm'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args['use_ema']:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args['checkpointing_steps'] == 0:
                    if accelerator.is_main_process:
                        if args['checkpoints_total_limit'] is not None:
                            checkpoints = os.listdir(args['output_dir'])
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args['checkpoints_total_limit']:
                                num_to_remove = len(checkpoints) - args['checkpoints_total_limit'] + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args['output_dir'], removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args['output_dir'], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args['max_train_steps']:
                break


    accelerator.wait_for_everyone()

    end_trainning = (datetime.now() - start_trainning)
    print("[!] Time Trainnig: ", end_trainning)

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args['use_ema']:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args['pretrained_model_name_or_path'],
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args['revision'],
        )
        pipeline.save_pretrained(args['output_dir'])

    save_model_card(args, repo_id, [], repo_folder=args['output_dir'])
    upload_folder(
        repo_id=repo_id,
        folder_path=args['output_dir'],
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
        token=args['hub_token']
    )
    print("[!] Uploaded to Registry")
    accelerator.end_training()
    # End Time of All
    end_all = (datetime.now() - start_all)
    print("[!] Time Overall: ", end_all)


    # ------------- TLDR ---------------
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
        'resolution': data['param']['num_train_epochs'],
        'train_batch_size': data['param']['learning_rate'],
        'num_train_epochs': data['param']['num_train_epochs'],
        'max_train_steps': data['param']['max_train_steps'],
        'learning_rate': data['param']['learning_rate'],
        'gradient_accumulation_steps':  data['param']['gradient_accumulation_steps'],
        'num_dataset': data['param']['num_dataset']
    } 

    write_to_csv(to_logs, 'image.csv')
