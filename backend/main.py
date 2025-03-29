from fastapi import FastAPI
from backend.routes.image_video import router as image_video_router

app = FastAPI()

app.include_router(image_video_router, prefix="/image-video")

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is running!"}
