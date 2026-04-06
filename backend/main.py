from fastapi import FastAPI, UploadFile, File
import shutil
import os
from utils.predict import load_model, predict_image

app = FastAPI()

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 LOAD MODEL ONLY ONCE (IMPORTANT)
model = load_model()


@app.get("/")
def home():
    return {"message": "Insect Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 USE SAME MODEL (NO RELOADING)
    detections = predict_image(file_path, model)

    return {"detections": detections}