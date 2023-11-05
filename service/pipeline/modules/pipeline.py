import joblib
import os
from google.cloud import storage
from transformers import T5Tokenizer, T5ForConditionalGeneration
import shutil
from datasets import load_dataset, concatenate_datasets
from random import randrange  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
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


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

if is_wandb_available():
    import wandb


check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {args.pretrained_model_name_or_path}
    datasets:
    - {args.dataset_name}
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

    This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
    {img_str}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
    prompt = "{args.validation_prompts[0]}"
    image = pipeline(prompt).images[0]
    image.save("my_image.png")
    ```

    ## Training info

    These are the key hyperparameters used during training:

    * Epochs: {args.num_train_epochs}
    * Learning rate: {args.learning_rate}
    * Batch size: {args.train_batch_size}
    * Gradient accumulation steps: {args.gradient_accumulation_steps}
    * Image resolution: {args.resolution}
    * Mixed-precision: {args.mixed_precision}

    """
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
    More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
    """

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images





def train_image(datas):
    args = {
        "non_ema_revision": None,
        "dataset_name":"lambdalabs/pokemon-blip-captions",
        "resolution":1,
        "train_batch_size": 1,
        "max_train_steps": 1 ,# Should be changed,
        "output_dir": "sd-model" 
        

    }
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
# `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
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
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    print(column_names)

def train_text(datas):
    def preprocess_function(sample,padding="max_length"):
        inputs = ["summarize: " + item for item in sample["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    
    # Initial Setup
    folder_path_text = "temp/text"
    folderExist = os.path.exists(folder_path_text)
    if not folderExist:
        os.makedirs(folder_path_text)

    # # Specify the directory where you want to save the tokenizer and model
    output_dir = "temp/text"

    # Load dataset from the hub
    dataset_id = "samsum"
    dataset = load_dataset(dataset_id)
    num_train_samples = 50
    dataset['train'] = dataset['train'].select(range(num_train_samples))

    num_test_samples = 10
    dataset['test'] = dataset['test'].select(range(num_test_samples))
    print(dataset['train'])


    model_id="google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # # The maximum total input sequence length after tokenization. 
    # # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # # The maximum total sequence length for target text after tokenization. 
    # # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")


    # tokenizer.save_pretrained(output_dir)
    # model.save_pretrained(output_dir)

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
    training_args = Seq2SeqTrainingArguments(
        output_dir="/root/tamlops/service/pipeline/example/tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        # learning_rate=5e-5,
        num_train_epochs=1,
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
        # hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
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
    



# train_text()
train_image()