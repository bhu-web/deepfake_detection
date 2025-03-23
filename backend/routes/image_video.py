from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def detect_fake():
    return {"message": "Image & Video deepfake detection API"}
