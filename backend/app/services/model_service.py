from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path: str, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def simple_defect_predict(image_path: str):
    arr = preprocess_image(image_path)
    mean_value = float(np.mean(arr))

    if mean_value < 0.45:
        label = "defective"
        defect_type = "scratch"
        severity = "high"
        confidence = 0.92
    elif mean_value < 0.7:
        label = "defective"
        defect_type = "crack"
        severity = "medium"
        confidence = 0.88
    else:
        label = "normal"
        defect_type = "none"
        severity = "low"
        confidence = 0.95

    return {
        "status": label,
        "defect_type": defect_type,
        "severity": severity,
        "confidence": round(confidence, 2),
        "notes": "Basic heuristic defect detection for demo pipeline. Replace with trained CNN/YOLO model for production.",
    }
