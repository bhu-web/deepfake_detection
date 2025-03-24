from fastapi import FastAPI
from backend.routes.image_video import router as image_video_router
from backend.routes.audio_text import router as audio_text_router

app = FastAPI()

app.include_router(image_video_router, prefix="/image-video")
app.include_router(audio_text_router, prefix="/audio-text")

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is running!"}
