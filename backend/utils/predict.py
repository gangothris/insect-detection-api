from ultralytics import YOLO

# 🔥 LOAD MODEL ONCE (already correct, keep it here)
model = YOLO("model/insect_model.pt")

# Advice mapping
ADVICE = {
    "whitefly": "Use yellow sticky traps and neem spray",
    "aphid": "Use soap water spray",
    "thrips": "Use blue sticky traps",
}

def predict_image(image_path):
    try:
        # 🔥 Run inference (disable unnecessary overhead)
        results = model(image_path, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, "Unknown")

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class": name,
                    "confidence": round(conf, 2),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],
                    "advice": ADVICE.get(name, "No advice available")
                })

        return detections

    except Exception as e:
        # 🔥 Prevent server crash (VERY IMPORTANT)
        return [{
            "class": "error",
            "confidence": 0.0,
            "bbox": [0, 0, 0, 0],
            "advice": f"Error: {str(e)}"
        }]