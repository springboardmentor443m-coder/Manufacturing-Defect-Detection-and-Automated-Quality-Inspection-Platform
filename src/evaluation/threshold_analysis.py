
from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

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
THRESHOLDS = [0.043, 0.045, 0.047, 0.05]



# -----------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return image.astype(np.float32) / 255.0


# -----------------------------------------
# PATCH EXTRACTION
# -----------------------------------------

def extract_patches(image):

    patches = []

    for y in range(0, IMAGE_SIZE, PATCH_SIZE):
        for x in range(0, IMAGE_SIZE, PATCH_SIZE):

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(patch)

    return patches


# -----------------------------------------
# PATCH FEATURE EXTRACTION
# -----------------------------------------

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

    return np.concatenate([
        color_mean,
        color_std,
        [edge_density, gray_std]
    ])


# -----------------------------------------
# CALCULATE ANOMALY RATIO
# -----------------------------------------

def calculate_anomaly_ratio(model, image_path):

    image = preprocess_image(image_path)

    patches = extract_patches(image)

    features = np.array([
        extract_patch_features(patch)
        for patch in patches
    ])

    predictions = model.predict(features)

    anomalous_patches = np.sum(predictions == -1)

    anomaly_ratio = anomalous_patches / len(patches)

    return anomaly_ratio


# -----------------------------------------
# LOAD TEST DATA
# -----------------------------------------

def load_test_data():

    test_folder = DATASET_PATH / "test"

    image_paths = []
    actual_labels = []

    for category in sorted(test_folder.iterdir()):

        if not category.is_dir():
            continue

        label = 0 if category.name == "good" else 1

        for image_path in sorted(category.iterdir()):

            if image_path.suffix.lower() in [
                ".png", ".jpg", ".jpeg"
            ]:
                image_paths.append(image_path)
                actual_labels.append(label)

    return image_paths, np.array(actual_labels)


# -----------------------------------------
# THRESHOLD ANALYSIS
# -----------------------------------------

def main():

    print("\nVISIONINSPECT AI - THRESHOLD ANALYSIS")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    image_paths, actual_labels = load_test_data()

    print("Total test images:", len(image_paths))
    print("Normal images:", np.sum(actual_labels == 0))
    print("Defective images:", np.sum(actual_labels == 1))

    print("\nCalculating anomaly ratios...")

    anomaly_ratios = []

    for index, image_path in enumerate(image_paths, start=1):

        ratio = calculate_anomaly_ratio(model, image_path)

        anomaly_ratios.append(ratio)

        if index % 10 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print("\nTHRESHOLD COMPARISON")
    print("=" * 75)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
    )

    print("-" * 75)

    results = []

    for threshold in THRESHOLDS:

        predicted_labels = np.array([
            1 if ratio >= threshold else 0
            for ratio in anomaly_ratios
        ])

        accuracy = accuracy_score(
            actual_labels,
            predicted_labels
        )

        precision = precision_score(
            actual_labels,
            predicted_labels,
            zero_division=0
        )

        recall = recall_score(
            actual_labels,
            predicted_labels,
            zero_division=0
        )

        f1 = f1_score(
            actual_labels,
            predicted_labels,
            zero_division=0
        )

        results.append((threshold, accuracy, precision, recall, f1))

        print(
            f"{threshold * 100:<11.1f}%"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    # Select threshold with highest F1 score
    best_result = max(results, key=lambda result: result[4])

    best_threshold = best_result[0]

    best_predictions = np.array([
        1 if ratio >= best_threshold else 0
        for ratio in anomaly_ratios
    ])

    print("\nBEST THRESHOLD BY F1 SCORE")
    print("-" * 50)

    print("Threshold:", best_threshold * 100, "%")
    print("Accuracy:", round(best_result[1], 4))
    print("Precision:", round(best_result[2], 4))
    print("Recall:", round(best_result[3], 4))
    print("F1 Score:", round(best_result[4], 4))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(actual_labels, best_predictions))

    print("\nTHRESHOLD ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
