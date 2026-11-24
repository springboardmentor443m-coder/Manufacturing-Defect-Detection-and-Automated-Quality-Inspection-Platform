
import sys
from pathlib import Path

import cv2
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

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = Path(r"C:\Users\Sucha\Downloads\bottle")

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "bottle_patch_anomaly_model.joblib"
)

IMAGE_SIZE = 256
PATCH_SIZE = 16

# Initial threshold: percentage of patches flagged
# as anomalous before classifying the whole image.
ANOMALY_THRESHOLD = 0.05


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = image.astype(np.float32) / 255.0

    return image


# --------------------------------------------------
# PATCH EXTRACTION
# --------------------------------------------------

def extract_patches(image):

    patches = []

    height, width, channels = image.shape

    for y in range(0, height, PATCH_SIZE):

        for x in range(0, width, PATCH_SIZE):

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(patch)

    return patches


# --------------------------------------------------
# PATCH FEATURE EXTRACTION
# --------------------------------------------------

def extract_patch_features(patch):

    gray = cv2.cvtColor(
        (patch * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    color_mean = np.mean(patch, axis=(0, 1))

    color_std = np.std(patch, axis=(0, 1))

    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.mean(edges > 0)

    gray_std = np.std(gray) / 255.0

    features = np.concatenate([
        color_mean,
        color_std,
        [edge_density, gray_std]
    ])

    return features


# --------------------------------------------------
# PREDICT ONE IMAGE
# --------------------------------------------------

def predict_image(model, image_path):

    image = preprocess_image(image_path)

    patches = extract_patches(image)

    patch_features = np.array([
        extract_patch_features(patch)
        for patch in patches
    ])

    predictions = model.predict(patch_features)

    anomalous_patches = np.sum(predictions == -1)

    total_patches = len(predictions)

    anomaly_ratio = anomalous_patches / total_patches

    predicted_label = (
        1 if anomaly_ratio >= ANOMALY_THRESHOLD else 0
    )

    return predicted_label, anomaly_ratio


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

def evaluate_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    true_labels = []

    predicted_labels = []

    anomaly_ratios = []

    test_folder = DATASET_PATH / "test"

    print("\nVISIONINSPECT AI - PATCH MODEL EVALUATION")
    print("-" * 50)

    print("Loading patch anomaly model...")
    print("Evaluation threshold:", ANOMALY_THRESHOLD)

    for category in sorted(test_folder.iterdir()):

        if not category.is_dir():
            continue

        actual_label = (
            0 if category.name == "good" else 1
        )

        images = sorted([
            path for path in category.iterdir()
            if path.suffix.lower() in [
                ".png", ".jpg", ".jpeg"
            ]
        ])

        for image_path in images:

            prediction, anomaly_ratio = predict_image(
                model,
                image_path
            )

            true_labels.append(actual_label)

            predicted_labels.append(prediction)

            anomaly_ratios.append(anomaly_ratio)

        print(
            f"Evaluated {category.name}: {len(images)} images"
        )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\nPATCH MODEL PERFORMANCE")
    print("-" * 50)

    print("Total test images:", len(true_labels))

    print(
        "Accuracy:",
        round(accuracy_score(true_labels, predicted_labels), 4)
    )

    print(
        "Precision:",
        round(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0
            ),
            4
        )
    )

    print("\nCONFUSION MATRIX")

    print("Rows = Actual [Normal, Defect]")
    print("Columns = Predicted [Normal, Defect]")

    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
            labels=[0, 1]
        )
    )

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["Normal", "Defect"],
            zero_division=0
        )
    )

    print("\nAVERAGE ANOMALOUS PATCH RATIO")

    print(
        round(float(np.mean(anomaly_ratios)), 4)
    )

    print("\nPATCH MODEL EVALUATION COMPLETED!")


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    evaluate_model()
