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

    file = ""

    if typ == 'image':
        print("[!] Train image")
        file = ""

        if(gpu == "3"):
            print("[!] Using GPU 3")
            file = "3_image.py"
        else:
            print("[!] Using GPU 5")
            file = "5_image.py"

        command = [
            'python3',
            file,
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
        file = ""
        if(gpu == "3"):
            print("[!] Using GPU 3")
            file = "3_text.py"
        else:
            print("[!] Using GPU 5")
            file = "5_text.py"

        command = [
            'python3',
            file,
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


def process_burst(req):
    data = req['data']
    typ = data['type']
    gpu = data['gpu']

    file = ""

    if typ == 'image':
        print("[!] Train image")
        file = ""

        if(gpu == "3"):
            print("[!] Using GPU 3")
            file = "3_image.py"
        else:
            print("[!] Using GPU 5")
            file = "5_image.py"

        command = [
            'python3',
            file,
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
        file = ""
        if(gpu == "3"):
            print("[!] Using GPU 3")
            file = "3_text.py"
        else:
            print("[!] Using GPU 5")
            file = "5_text.py"

        command = [
            'python3',
            file,
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
    return ""


@app.post("/train-burst", response_class=JSONResponse)
async def train(requests: Request):
    req = await requests.json()
    th = threading.Thread(target=process_burst, args=(req))
    th.start()
    return {"error": False, "response": "Train has been done"}
