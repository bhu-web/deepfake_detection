from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def detect_audio_fake():
    return {"message": "Audio & Text deepfake detection API"}

