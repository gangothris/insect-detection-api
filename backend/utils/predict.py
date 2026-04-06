from ultralytics import YOLO
import os

# Advice mapping
ADVICE = {
    "whitefly": "Use yellow sticky traps and neem spray",
    "aphid": "Use soap water spray",
    "thrips": "Use blue sticky traps",
}


# ✅ Load model once
def load_model():
    model_path = os.path.join("backend", "model", "insect_model.pt")
    return YOLO(model_path)


def predict_image(image_path, model):
    results = model(image_path)

    detections = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": name,
                "confidence": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "advice": ADVICE.get(name, "No advice available")
            })

    return detections
