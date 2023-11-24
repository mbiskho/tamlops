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

async def inference_image(text):
    try:
        model_id = "stabilityai/stable-diffusion-2-1-base"

        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        prompt = text
        image = pipe(prompt).images[0]  

        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')  
        image_bytes = byte_stream.getvalue()
        return image_bytes
    except:
        print("Error occured happen")
        return None

async def inference_text(text):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

    input_text = text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    try:
        outputs = model.generate(input_ids)
        out = tokenizer.decode(outputs[0])
        print("[!] Text has been generated")
        return out
    except:
        print("Error occurred")
        return ""
