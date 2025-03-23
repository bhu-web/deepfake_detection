from fastapi import FastAPI
from routes import image_video, audio_text

app = FastAPI()

# Include routes
app.include_router(image_video.router, prefix="/image-video")
app.include_router(audio_text.router, prefix="/audio-text")

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is running!"}
