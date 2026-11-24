
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.feature_extractor import extract_features


DATASET_PATH = Path(r"C:\Users\Sucha\Downloads\bottle")
MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "bottle_anomaly_model.joblib"
)


def evaluate_model():

    model = joblib.load(MODEL_PATH)

    true_labels = []
    predicted_labels = []

    test_folder = DATASET_PATH / "test"

    print("\nVISIONINSPECT AI - MODEL EVALUATION")
    print("-" * 45)

    for category in sorted(test_folder.iterdir()):

        if not category.is_dir():
            continue

        actual_label = 0 if category.name == "good" else 1

        images = sorted([
            path for path in category.iterdir()
            if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        for image_path in images:

            features = extract_features(image_path)
            prediction = model.predict([features])[0]

            # Isolation Forest: -1 means anomaly, 1 means normal
            predicted_label = 1 if prediction == -1 else 0

            true_labels.append(actual_label)
            predicted_labels.append(predicted_label)

        print(f"Evaluated {category.name}: {len(images)} images")

    print("\nTOTAL TEST IMAGES:", len(true_labels))

    print("\nACCURACY:", round(
        accuracy_score(true_labels, predicted_labels), 4
    ))

    print("PRECISION:", round(
        precision_score(true_labels, predicted_labels, zero_division=0), 4
    ))

    print("RECALL:", round(
        recall_score(true_labels, predicted_labels, zero_division=0), 4
    ))

    print("F1 SCORE:", round(
        f1_score(true_labels, predicted_labels, zero_division=0), 4
    ))

    print("\nCONFUSION MATRIX")
    print("Rows = Actual [Normal, Defect]")
    print("Columns = Predicted [Normal, Defect]")
    print(confusion_matrix(true_labels, predicted_labels))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=["Normal", "Defect"],
        zero_division=0
    ))


if __name__ == "__main__":
    evaluate_model()
