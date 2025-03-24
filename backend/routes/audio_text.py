from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def detect_audio_text():
    return {"message": "Detecting deepfake in audio/text"}
