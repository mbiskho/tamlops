from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.database import text_test_db, text_train_db, image_test_db, image_train_db

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


@app.post("/text/training", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()
    await text_train_db(req)

    return {"error": False, "response": "Success submit request"}

@app.post("/text/test", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()
    await text_test_db(req)

    return {"error": False, "response": "Success submit request"}

@app.post("/image/training", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()
    await image_train_db(req)

    return {"error": False, "response": "Success submit request"}

@app.post("/image/test", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()
    await image_test_db(req)

    return {"error": False, "response": "Success submit request"}