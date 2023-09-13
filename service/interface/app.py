from fastapi import FastAPI, HTTPException, Request,  Header, Form
from fastapi.responses import JSONResponse, HTMLResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware

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