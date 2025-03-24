from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def detect():
    return {"message": "Detecting deepfake in images/videos"}
