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

    used = ""

    if typ == 'image':
        print("[!] Train image")
        used = ""

        if(gpu == "3"):
            print("[!] Using GPU 3")
            used = "3_image.py"
        else:
            print("[!] Using GPU 5")
            used = "5_image.py"

        command = [
            'python3',
            used,
            f"--resolution={data['param']['resolution']}",
            f"--train_batch_size={data['param']['train_batch_size']}",
            f"--num_train_epochs={data['param']['num_train_epochs']}",
            f"--max_train_steps={data['param']['max_train_steps']}",
            f"--learning_rate={data['param']['learning_rate']}",
            f"--gradient_accumulation_steps={data['param']['gradient_accumulation_steps']}", 
            f"--file={data['file']}",
            f"--id={data['id']}"
        ]
    else:
        print("[!] Train Text")
        used = ""
        if(gpu == "3"):
            print("[!] Using GPU 3")
            used = "3_text.py"
        else:
            print("[!] Using GPU 5")
            used = "5_text.py"

        command = [
            'python3',
            used,
            f"--per_device_train_batch_size={data['param']['per_device_train_batch_size']}",
            f"--per_device_eval_batch_size={data['param']['per_device_eval_batch_size']}",
            f"--learning_rate={data['param']['learning_rate']}",
            f"--num_train_epochs={data['param']['num_train_epochs']}",
            f"--file={data['file']}",
            f"--id={data['id']}"
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

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Subprocess output (stdout):", result.stdout)
    print("Subprocess output (stderr):", result.stderr)


    value = get_item(gpu)
    value = int(value) - 1
    set_item(gpu, value)
    print("[!] Release num process")
    print("Remaining Process: ", value)

    return {"error": False, "response": "Train has been done"}


def process_text(
        id,
        gpu, 
        file,
        per_device_train_batch_size, 
        per_device_eval_batch_size, 
        learning_rate,
        num_train_epochs
):
    used = ""
    if(gpu == "0"):
        print("[!] Using GPU 0")
        used = "0_text.py"
    elif(gpu == "1"):
        print("[!] Using GPU 1")
        used = "1_text.py"
    elif(gpu == "2"):
        print("[!] Using GPU 2")
        used = "2_text.py"
    elif(gpu == "3"):
        print("[!] Using GPU 3")
        used = "3_text.py"
    elif(gpu == "4"):
        print("[!] Using GPU 4")
        used = "4_text.py"
    elif(gpu == "5"):
        print("[!] Using GPU 5")
        used = "5_text.py"
    elif(gpu == "6"):
        print("[!] Using GPU 6")
        used = "6_text.py"
    elif(gpu == "7"):
        print("[!] Using GPU 7")
        used = "7_text.py"
    else:
        print("[!] Using GPU 5")
        used = "5_text.py"
    command = [
            'python3',
            used,
            f"--per_device_train_batch_size={per_device_train_batch_size}",
            f"--per_device_eval_batch_size={per_device_eval_batch_size}",
            f"--learning_rate={learning_rate}",
            f"--num_train_epochs={num_train_epochs}",
            f"--file={file}",
            f"--id={id}"
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

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Subprocess output (stdout):", result.stdout)
    print("Subprocess output (stderr):", result.stderr)


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
    used = ""
    if(gpu == "0"):
        print("[!] Using GPU 0")
        used = "0_image.py"
    elif(gpu == "1"):
        print("[!] Using GPU 1")
        used = "1_image.py"
    elif(gpu == "2"):
        print("[!] Using GPU 2")
        used = "2_image.py"
    elif(gpu == "3"):
        print("[!] Using GPU 3")
        used = "3_image.py"
    elif(gpu == "4"):
        print("[!] Using GPU 4")
        used = "4_image.py"
    elif(gpu == "5"):
        print("[!] Using GPU 5")
        used = "5_image.py"
    elif(gpu == "6"):
        print("[!] Using GPU 6")
        used = "6_image.py"
    elif(gpu == "7"):
        print("[!] Using GPU 7")
        used = "7_image.py"
    else:
        print("[!] Using GPU 5")
        used = "5_image.py"

    command = [
            'python3',
            used,
            f"--resolution={resolution}",
            f"--train_batch_size={train_batch_size}",
            f"--num_train_epochs={num_train_epochs}",
            f"--max_train_steps={max_train_steps}",
            f"--learning_rate={learning_rate}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}", 
            f"--file={file}",
            f"--id={id}"
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

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Subprocess output (stdout):", result.stdout)
    print("Subprocess output (stderr):", result.stderr)


    value = get_item(gpu)
    value = int(value) - 1
    set_item(gpu, value)
    print("[!] Release num process")
    print("Remaining Process: ", value)
    return ""

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
