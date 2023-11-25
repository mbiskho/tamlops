from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/check-gpu", response_class=JSONResponse)
async def checkgpu(requests: Request):
    req = await requests.json()
    result = get_gpu_info()
    return {"error": False, "response": result}

@app.post("/train-text", response_class=JSONResponse)
async def text2text(requests: Request):
    req = await requests.json()
    
    return {"error": False, "response": "model is trainned"}


@app.post("/train-image", response_class=JSONResponse)
async def text2image(requests: Request):
    req = await requests.json()



    return {"error": False, "response": "model is trainned"}
