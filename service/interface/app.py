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


@app.post("/", response_class=JSONResponse)
async def prompting(requests: Request):
    req = await requests.json()
    prompt = req['prompt']
    text = prompt.split()

    # Check for specific content generation requests in the extracted entities
    # Keywords for Text Generation
    text_generation_keywords = {
        "paragraph", "short novel", "essay", "article", "content",
        "writing", "text", "script", "literature", "prose",
        "document", "write", "typing", "author", "literary", "creative"
    }

    # Keywords for Image Generation
    image_generation_keywords = {
        "image", "picture", "photo", "artwork", "illustration",
        "graphic", "design", "visual", "draw", "sketch",
        "painting", "create an image", "generate a photo",
        "produce a drawing", "design a graphic"
    }

    # Keywords for Audio Generation
    audio_generation_keywords = {
        "music", "song", "melody", "sound", "audio",
        "tune", "compose", "create music", "produce a song",
        "generate a melody", "make a sound", "music composition",
        "musical", "audio track", "musical composition"
    }

    classification = ""

    for word in text:
        print(word)
        if word in text_generation_keywords:
            classification = "text"
        elif word in image_generation_keywords:
            classification = "image"
        elif word in audio_generation_keywords:
            classification = "audio"

    return {"error": False, "response": classification}
