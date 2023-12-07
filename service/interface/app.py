from fastapi import FastAPI, HTTPException, Request,  Header, FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.database import save_training_db, get_from_db
from modules.gcp import upload_to_gcs
from modules.schedule import schedule_logic_min_min, schedule_logic_fcfs_burst, schedule_logic_max_min, schedule_logic_fcfs_normal, schedule_logic_real_min_min
from modules.request import send_post_request
import json
import requests

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

@app.post("/training")
async def training(file: UploadFile = File(...), type: str = Form(...), params: str = Form(...)):
    try:
        file_url = await upload_to_gcs(file)

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        db_res = await save_training_db(type, file_url, file_size, params)
        print(db_res)
        
        return {
            "error": False,
            "response": f"Training submitted successfully",
            "fileURL": file_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference", response_class=JSONResponse)
async def inference(requ: Request):
    req = await requ.json()
    res = None
    typ = req['type']
    text = req['text']
    URL = ""
    payload = json.dumps({
        "type": typ,
        "text": text
    })
    response = ""
    headers = {
        'Content-Type': 'application/json'
    }

    if typ == "image":
        URL = "http://127.0.0.1:5060/inference-image"
    else:
        URL = "http://127.0.0.1:5060/inference-text"

    try:
        response = requests.request("POST", URL, headers=headers, data=payload) 
        res = response.text
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print("Request exception:", e)

    return {"error": False, "response": res}

@app.post("/inference-burst", response_class=JSONResponse)
async def inference_burst(requ: Request):
    req = await requ.json()
    res = None
    typ = req['type']
    text = req['text']
    URL = ""
    payload = json.dumps({
        "type": typ,
        "text": text
    })
    headers = {
        'Content-Type': 'application/json'
    }

    if typ == "image":
        URL = "http://127.0.0.1:5060/inference-image-burst"
    else:
        URL = "http://127.0.0.1:5065/inference-text-burst"


    response = requests.request("POST", URL, headers=headers, data=payload) 
    res = response.text
    return {"error": False, "response": res}

@app.get('/schedule/real-min-min', response_class=JSONResponse)
async def schedule():
    post_response = await schedule_logic_real_min_min()
    return {"error": False, "response": post_response}

@app.get('/schedule/min-min', response_class=JSONResponse)
async def schedule():
    post_response = await schedule_logic_min_min()
    return {"error": False, "response": post_response}
    
@app.get('/schedule/max-min', response_class=JSONResponse)
async def schedule():
    post_response = await schedule_logic_max_min()
    return {"error": False, "response": post_response}
    
@app.get('/schedule/fcfs-burst', response_class=JSONResponse)
async def schedule():
    await schedule_logic_fcfs_burst()
    return {"error": False, "response": "Schedule Started"}

@app.get('/schedule/fcfs-normal', response_class=JSONResponse)
async def schedule():
    await schedule_logic_fcfs_normal()
    return {"error": False, "response": "Schedule Started"}