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
import base64

def remove_pad_and_end_tags(text):
    text = text.replace('<pad>', '').replace('</s>', '')
    return text.strip()

async def inference_image(text):
    path = "/etc/model/image"
    scheduler = EulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(path, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = text
    image = pipe(prompt).images[0]  
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')  
    image_bytes = byte_stream.getvalue()
    return image_bytes

async def inference_text(text):
    path = "/etc/model/text"
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path, device_map="auto")

    input_text = text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    try:
        outputs = model.generate(input_ids)
        out = tokenizer.decode(outputs[0])
        print("[!] Text has been generated")
        ans = remove_pad_and_end_tags(out)
        return ans
    except:
        print("Error occurred")
        return ""
    
def inference_image_burst(text, pp):
    path = "/etc/model/image"
    scheduler = EulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(path, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = text
    image = pipe(prompt).images[0]  
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')  
    image_bytes = byte_stream.getvalue()
    return image_bytes

def inference_text_burst(text, pp):
    path = "/etc/model/text"
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path, device_map="auto")

    input_text = text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    try:
        outputs = model.generate(input_ids)
        out = tokenizer.decode(outputs[0])
        print("[!] Text has been generated")
        ans = remove_pad_and_end_tags(out)
        return ans
    except:
        print("Error occurred")
        return ""

