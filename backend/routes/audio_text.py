from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_audio_text_status():
    return {"message": "Audio-Text detection API is working!"}
