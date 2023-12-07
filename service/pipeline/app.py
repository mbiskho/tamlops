from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.gpu import get_gpu_info
from modules.cache import get_item, set_item
import subprocess
import os
import threading


app = FastAPI(docs_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_text_nogpu(
        id,
        file,
        per_device_train_batch_size, 
        per_device_eval_batch_size, 
        learning_rate,
        num_train_epochs
):

    command = [
            'sh', 
            'text.sh', 
            f'{per_device_train_batch_size}', 
            f'{per_device_eval_batch_size}', 
            f'{learning_rate}', 
            f'{num_train_epochs}', 
            f'{file}', 
            f'{id}'
    ]
    result = subprocess.run(command)
    return ""

def process_image_nogpu(
        id,
        file,
        resolution,
        train_batch_size,
        num_train_epochs,
        max_train_steps,
        gradient_accumulation_steps,
        learning_rate
):
    command = [
            'sh', 
            'image.sh', 
            f'{resolution}', 
            f'{train_batch_size}', 
            f'{num_train_epochs}', 
            f'{max_train_steps}', 
            f'{learning_rate}', 
            f'{gradient_accumulation_steps}', 
            f'{file}',
            f'{id}'
    ]
    result = subprocess.run(command)
    return ""

def process_text(
        id,
        gpu, 
        file,
        per_device_train_batch_size, 
        per_device_eval_batch_size, 
        learning_rate,
        num_train_epochs
):
    print(f"[!] Using GPU {gpu}")
    command = [
            'sh', 
            'gpu_text.sh', 
            f'{gpu}',
            f'{per_device_train_batch_size}', 
            f'{per_device_eval_batch_size}', 
            f'{learning_rate}', 
            f'{num_train_epochs}', 
            f'{file}', 
            f'{id}'
    ]

    # Run
    value = get_item(gpu)
    if(value == None):
        print("Current Process: 1")
        set_item(gpu, 1)
    else:
        print("Current Process: ", str(value), " added to be", str(int(value) + 1))
        value = int(value) + 1
        set_item(gpu, value)

    result = subprocess.run(command)

    value = get_item(gpu)
    value = int(value) - 1
    set_item(gpu, value)
    print("[!] Release num process")
    print("Remaining Process: ", value)
    return ""

def process_image(
        id,
        gpu, 
        file,
        resolution,
        train_batch_size,
        num_train_epochs,
        max_train_steps,
        gradient_accumulation_steps,
        learning_rate
):
    print(f"[!] Using GPU {gpu}")
    command = [
            'sh', 
            'gpu_image.sh', 
            f'{gpu}',
            f'{resolution}', 
            f'{train_batch_size}', 
            f'{num_train_epochs}', 
            f'{max_train_steps}', 
            f'{learning_rate}', 
            f'{gradient_accumulation_steps}', 
            f'{file}',
            f'{id}'
    ]
    # Run
    value = get_item(gpu)
    if(value == None):
        print("Current Process: 1")
        set_item(gpu, 1)
    else:
        print("Current Process: ", str(value), " added to be", str(int(value) + 1))
        value = int(value) + 1
        set_item(gpu, value)

    result = subprocess.run(command)

    value = get_item(gpu)
    value = int(value) - 1
    set_item(gpu, value)
    print("[!] Release num process")
    print("Remaining Process: ", value)
    return ""


@app.get('/health', response_class=JSONResponse)
async def health_svc():
    return {"error": False, "response": "Iam Healty"}

@app.get("/check-gpu", response_class=JSONResponse)
async def checkgpu(requests: Request):
    result = get_gpu_info()
    return {"error": False, "response": result}

@app.post("/train", response_class=JSONResponse)
async def train(requests: Request):
    req = await requests.json()
    data = req['data']
    typ = data['type']
    gpu = data['gpu']

    if typ == 'image':
        print("[!] Train image")
        
        command = [
            'sh', 
            'gpu_image.sh', 
            f'{gpu}',
            f"{data['param']['resolution']}", 
            f"{data['param']['train_batch_size']}", 
            f"{data['param']['num_train_epochs']}", 
            f"{data['param']['max_train_steps']}", 
            f"{data['param']['learning_rate']}", 
            f"{data['param']['gradient_accumulation_steps']}", 
            f"{data['file']}",
            f"{data['id']}"
        ]
    else:
        print("[!] Train Text")

        command = [
            'sh', 
            'gpu_text.sh', 
            f'{gpu}',
            f"{data['param']['per_device_train_batch_size']}", 
            f"{data['param']['per_device_eval_batch_size']}", 
            f"{data['param']['learning_rate']}", 
            f"{data['param']['num_train_epochs']}", 
            f"{data['file']}", 
            f"{data['id']}"
        ]
    value = get_item(gpu)
    if(value == None):
        print("Current Process: 1")
        set_item(gpu, 1)
    else:
        print("Current Process: ", str(value), " added to be", str(int(value) + 1))
        value = int(value) + 1
        set_item(gpu, value)


    print("[!] Adding num process")

    result = subprocess.run(command)


    value = get_item(gpu)
    value = int(value) - 1
    set_item(gpu, value)
    print("[!] Release num process")
    print("Remaining Process: ", value)

    return {"error": False, "response": "Train has been done"}

@app.post("/train-nogpu", response_class=JSONResponse)
async def train(requests: Request):
    req = await requests.json()
    data = req['data']
    typ = data['type']
    
    if typ == 'image':
        print("[!] Train image") 
        command = [
            'sh', 
            'image.sh', 
            f"{data['param']['resolution']}", 
            f"{data['param']['train_batch_size']}", 
            f"{data['param']['num_train_epochs']}", 
            f"{data['param']['max_train_steps']}", 
            f"{data['param']['learning_rate']}", 
            f"{data['param']['gradient_accumulation_steps']}", 
            f"{data['file']}",
            f"{data['id']}"
        ]
    else:
        print("[!] Train Text")
        command = [
            'sh', 
            'text.sh', 
            f"{data['param']['per_device_train_batch_size']}", 
            f"{data['param']['per_device_eval_batch_size']}", 
            f"{data['param']['learning_rate']}", 
            f"{data['param']['num_train_epochs']}", 
            f"{data['file']}", 
            f"{data['id']}"
        ]
    result = subprocess.run(command)

    return {"error": False, "response": "Train has been done"}


@app.post("/train-burst", response_class=JSONResponse)
async def train(requests: Request):
    req = await requests.json()
    data = req['data']
    
    if data['type'] == 'text':
        th = threading.Thread(target=process_text, args=(data['id'], 
                                data['gpu'], data['file'], 
                                data['param']['per_device_train_batch_size'],
                                data['param']['per_device_eval_batch_size'],
                                data['param']['learning_rate'],
                                data['param']['num_train_epochs'],
        ))
    else:
        th = threading.Thread(target=process_image, args=(data['id'], 
                                data['gpu'], data['file'], 
                                data['param']['resolution'],
                                data['param']['train_batch_size'],
                                data['param']['num_train_epochs'],
                                data['param']['max_train_steps'],
                                data['param']['gradient_accumulation_steps'],
                                data['param']['learning_rate'],
        ))
    th.start()
    return {"error": False, "response": "Iam Healty"}




@app.post("/train-burst-nogpu", response_class=JSONResponse)
async def train(requests: Request):
    req = await requests.json()
    data = req['data']
    
    if data['type'] == 'text':
        th = threading.Thread(target=process_text_nogpu, args=(data['id'], 
                                data['file'],
                                data['param']['per_device_train_batch_size'],
                                data['param']['per_device_eval_batch_size'],
                                data['param']['learning_rate'],
                                data['param']['num_train_epochs'],
        ))
    else:
        th = threading.Thread(target=process_image_nogpu, args=(data['id'], 
                                data['file'],
                                data['param']['resolution'],
                                data['param']['train_batch_size'],
                                data['param']['num_train_epochs'],
                                data['param']['max_train_steps'],
                                data['param']['gradient_accumulation_steps'],
                                data['param']['learning_rate'],
        ))
    th.start()
    return {"error": False, "response": "Iam Healty"}
