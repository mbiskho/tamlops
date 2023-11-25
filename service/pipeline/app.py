from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.gpu import get_gpu_info
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
    datas = req['datas']

    for item in datas:
        command = []

        if item['type'] == 'text':
          command = [
            f"CUDA_VISIBLE_DEVICES={item['num_gpu']}",
            'python3',
            'text.py',
            f"--per_device_train_batch_size={item['param']['per_device_train_batch_size']}",
            f"--per_device_eval_batch_size={item['param']['per_device_eval_batch_size']}",
            f"--learning_rate={item['param']['learning_rate']}",
            f"--num_train_epochs={item['param']['num_train_epochs']}",
            f"--file={item['file']}",
            f"--id={item['id']}"
        ]      
        else:
          command = [
            f"CUDA_VISIBLE_DEVICES={item['num_gpu']}",
            'python3',
            'image.py',
            f"--resolution={item['param']['resolution']}",
            f"--train_batch_size={item['param']['train_batch_size']}",
            f"--num_train_epochs={item['param']['num_train_epochs']}",
            f"--max_train_steps={item['param']['max_train_steps']}",
            f"--learning_rate={item['param']['learning_rate']}",
            f"--gradient_accumulation_steps={item['param']['gradient_accumulation_steps']}", 
            f"--file={item['file']}",
            f"--id={item['id']}"
        ]   
        print(command)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Subprocess output (stdout):", result.stdout)
        print("Subprocess output (stderr):", result.stderr)
    return {"error": False, "response": "Train has been done"}
