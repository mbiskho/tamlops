from fastapi import FastAPI, HTTPException, Request,  Header, FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.database import save_training_db, get_from_db
from modules.gcp import upload_to_gcs
from modules.schedule import schedule_logic_min_min, schedule_logic_fcfs, schedule_logic_max_min
from modules.request import send_post_request
import asyncio

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

    file_url = await upload_to_gcs(file)

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    await save_training_db(type, file_url, file_size, params)
  
    return {"error": False, "response": f"Training submitted successfully", "fileURL": file_url}

@app.post("/inference", response_class=JSONResponse)
async def inference(requests: Request):
     req = await requests.json()
     res = await send_post_request("http://127.0.0.1:3000/inference", req)
     return {"error": False, "response": res}


@app.get('/schedule/min-min', response_class=JSONResponse)
async def schedule():
    try:
        post_response = await schedule_logic_min_min()
        return post_response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get('/schedule/max-min', response_class=JSONResponse)
async def schedule():
    try:
        post_response = await schedule_logic_max_min()
        return post_response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get('/schedule/fcfs', response_class=JSONResponse)
async def schedule():
    asyncio.run(schedule_logic_fcfs())
    return {"error": False, "response": "Schedule Started"}

