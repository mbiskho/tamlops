from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.inference import *

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


@app.get("/pull-model", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()

    return {"error": False, "response": "Output has been made"}


@app.post("/inference", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()

    if req['type'] == "image":
        await inference_image(req['text'])
    if req['type'] == "text":
        await inference_text(req['text'])
    return {"error": False, "response": "Output has been made"}
