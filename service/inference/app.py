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


@app.post("/inference/text", response_class=JSONResponse)
async def text(requests: Request):
    req = await requests.json()
    text = req["text"]
    resp = inference_text(text)
    return {"error": False, "response": resp}


@app.post("/inference/image", response_class=JSONResponse)
async def image(requests: Request):
    req = await requests.json()
    text = req["text"]
    resp = inference_image(text)
    return {"error": False, "response": resp}