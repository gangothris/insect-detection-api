from fastapi import FastAPI, UploadFile, File
import shutil
import os

# ✅ FIXED IMPORT (IMPORTANT)
from backend.utils.predict import load_model, predict_image

app = FastAPI()

UPLOAD_FOLDER = "backend/data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load model once (not per request)
model = load_model()


@app.get("/")
def home():
    return {"message": "Insect Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run prediction
        detections = predict_image(file_path, model)

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}
