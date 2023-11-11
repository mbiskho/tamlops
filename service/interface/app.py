from fastapi import FastAPI, HTTPException, Request,  Header, FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.database import save_training_db
from modules.gcp import upload_to_gcs

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
async def training(file: UploadFile = File(...), type: str = Form(...)):

    file_url = await upload_to_gcs(file)
    await save_training_db(type, file_url)
    
    return {"error": False, "response": f"Training submitted successfully", "fileURL": file_url}