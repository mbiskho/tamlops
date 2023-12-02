from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from modules.inference import inference_image, inference_text
from modules.gpu import get_gpu_info

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
     # Pulling Model
    return {"error": False, "response": "Output has been made"}

@app.get("/check-gpu", response_class=JSONResponse)
async def checkgpu(requests: Request):
    result = get_gpu_info()
    return {"error": False, "response": result}

@app.post("/inference-text", response_class=JSONResponse)
async def text(requests: Request):
    req = await requests.json()
    typ = req['type']
    text = req['text']
    print('Request \n', req)
    print("[!] Inference Text")
    print("Text: ", text)
    response = await inference_text(text)
    return response

@app.post("/inference-image", response_class=Response)
async def image(requests: Request):
    req = await requests.json()
    typ = req['type']
    text = req['text']
    print('Request \n', req)
    print("[!] Inference Image")
    print("Text: ", text)
    response = await inference_image(text)
    return Response(content=response, media_type="raw")


