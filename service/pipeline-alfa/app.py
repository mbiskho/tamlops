from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.gpu import get_gpu_info
from modules.cache import get_item, set_item
import subprocess
import os


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
    command = []

    if typ == 'image':
        print("[!] Train image")
        command = [
            'python3',
            'image.py',
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
        command = [
            'python3',
            'text.py',
            f"--per_device_train_batch_size={data['param']['per_device_train_batch_size']}",
            f"--per_device_eval_batch_size={data['param']['per_device_eval_batch_size']}",
            f"--learning_rate={data['param']['learning_rate']}",
            f"--num_train_epochs={data['param']['num_train_epochs']}",
            f"--file={data['file']}",
            f"--id={data['id']}"
        ]
        print("[!] Train text")


    value = get_item('process_alfa')
    print("Current Process: ", value, " added to be", value + 1)
    if(value == None):
        set_item('process_alfa', 1)
    else:
        value = int(value) + 1
        set_item('process_alfa', value)


    print("[!] Adding num process")

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Subprocess output (stdout):", result.stdout)
    print("Subprocess output (stderr):", result.stderr)


    value = get_item('process_alfa')
    value = int(value) - 1
    set_item('process_alfa', value)
    print("[!] Release num process")
    print("Remaining Process: ", value)

    return {"error": False, "response": "Train has been done"}
