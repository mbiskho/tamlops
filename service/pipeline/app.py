from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.pipeline import train_text, train_image
from modules.database import get_data

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


@app.post("/train", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()

    crs = get_data()
    data_list = [document for document in crs]

    image_data = [data for data in data_list if data['type'] == "image"]
    text_data = [data for data in data_list if data['type'] == "text"]
    audio_data = [data for data in data_list if data['type'] == "audio"]

    train_text(text_data)
    train_audio(audio_data)
    train_image(image_data)

    return {"error": False, "response": "model is trainned"}
