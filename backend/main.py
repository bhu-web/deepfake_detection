from fastapi import FastAPI
from backend.routes import audio_text  # Remove image_video

app = FastAPI()

app.include_router(audio_text.router, prefix="/audio-text")  # Only this

@app.get("/")
def home():
    return {"message": "Audio-Text Deepfake Detection API is running!"}
