
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.feature_extractor import extract_features


MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "bottle_anomaly_model.joblib"
)


def predict_defect(image_path):

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found!")

    model = joblib.load(MODEL_PATH)

    features = extract_features(image_path)

    prediction = model.predict([features])[0]
    anomaly_score = model.decision_function([features])[0]

    if prediction == 1:
        status = "PASS"
        message = "Bottle appears normal."
    else:
        status = "REVIEW"
        message = "Possible anomaly detected. Manual inspection required."

    return {
        "status": status,
        "message": message,
        "anomaly_score": float(anomaly_score)
    }


if __name__ == "__main__":

    test_image = (
        Path(r"C:\Users\Sucha\Downloads\bottle")
        / "test"
        / "broken_large"
        / "000.png"
    )

    result = predict_defect(test_image)

    print("\nVISIONINSPECT AI - INSPECTION RESULT")
    print("-" * 45)
    print("Image:", test_image.name)
    print("Status:", result["status"])
    print("Message:", result["message"])
    print("Anomaly score:", result["anomaly_score"])
