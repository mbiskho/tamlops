from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import io
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
import psutil


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

tokenizer.save_pretrained('/etc/model/text')
model.save_pretrained('/etc/model/text')

scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=torch.float16)

scheduler.save_pretrained('/etc/model/image')
pipe.save_pretrained('/etc/model/image')