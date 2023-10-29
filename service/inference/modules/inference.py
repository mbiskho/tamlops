from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import torch
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

async def inference_image(req):
    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", 
        torch_dtype=torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)

    try:
        target_runtime = random.uniform(15, 30)  # in seconds

        start_time = time.time()
        prompt = 'A dog sitting. Black and white photography. Leica lens. Hi-res. hd 8k --ar 2:3'
        num_steps = 1
        num_variations = 1
        prompt_guidance = 9
        dimensions = (400, 600)  # (width, height) tuple
        random_seeds = [random.randint(0, 65000) for _ in range(num_variations)]
        images = pipe(
            prompt=num_variations * [prompt],
            num_inference_steps=num_steps,
            guidance_scale=prompt_guidance,
            height=dimensions[0],
            width=dimensions[1],
            generator=[torch.Generator().manual_seed(i) for i in random_seeds]
        )
        print("Image has been generated")
    except:
        print("Error occured happen")
        # return "Error Occured Happen"

async def inference_text(req):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    try:
        outputs = model.generate(input_ids)
        out = tokenizer.decode(outputs[0])
        print("[!] Text has been generated")
        return out
    except:
        print("Error occured")
        return ""
