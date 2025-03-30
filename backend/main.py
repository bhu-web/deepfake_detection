import uvicorn
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("best_xception_model.keras")

# Define image size (must match training size)
IMG_SIZE = (128, 128)

# Initialize FastAPI app
app = FastAPI()

# Preprocessing function
def preprocess_image(image):
    image = image.resize(IMG_SIZE)  # Resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")  # Load image
    image = preprocess_image(image)  # Preprocess image
    prediction = model.predict(image)[0][0]  # Get prediction

    # Return result
    result = "Fake" if prediction > 0.5 else "Real"
    return {"prediction": result, "confidence": float(prediction)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
