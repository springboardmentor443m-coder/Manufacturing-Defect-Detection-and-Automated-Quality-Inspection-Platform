
from pathlib import Path

import cv2
import joblib
import numpy as np


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "bottle_patch_anomaly_model.joblib"
)

OUTPUT_FOLDER = PROJECT_ROOT / "artifacts" / "heatmaps"

DATASET_PATH = Path(r"C:\Users\Sucha\Downloads\bottle")

IMAGE_SIZE = 256
PATCH_SIZE = 16

# Provisional threshold selected during experiments
ANOMALY_THRESHOLD = 0.0429


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

    for y in range(0, IMAGE_SIZE, PATCH_SIZE):
        for x in range(0, IMAGE_SIZE, PATCH_SIZE):

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

    return np.concatenate([
        color_mean,
        color_std,
        [edge_density, gray_std]
    ])


# --------------------------------------------------
# GENERATE ANOMALY HEATMAP
# --------------------------------------------------

def generate_heatmap(model, image_path):

    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError(f"Unable to read image: {image_path}")

    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    patches = extract_patches(image)

    patch_features = np.array([
        extract_patch_features(patch)
        for patch in patches
    ])

    # Higher scores indicate more unusual patches
    anomaly_scores = -model.decision_function(
        patch_features
    )

    # Create 16 x 16 anomaly score map
    score_map = anomaly_scores.reshape(
        IMAGE_SIZE // PATCH_SIZE,
        IMAGE_SIZE // PATCH_SIZE
    )

    # Resize heatmap to original processing dimensions
    heatmap = cv2.resize(
        score_map.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize heatmap for visualization
    heatmap_normalized = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Generate coloured heatmap
    colored_heatmap = cv2.applyColorMap(
        heatmap_normalized,
        cv2.COLORMAP_JET
    )

    # Blend heatmap with original image
    overlay = cv2.addWeighted(
        original,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    # Calculate anomalous patches
    patch_predictions = model.predict(patch_features)

    anomalous_count = int(
        np.sum(patch_predictions == -1)
    )

    total_patches = len(patches)

    anomaly_ratio = anomalous_count / total_patches

    # Image-level decision
    status = (
        "REVIEW"
        if anomaly_ratio >= ANOMALY_THRESHOLD
        else "PASS"
    )

    # Save generated images
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    image_name = Path(image_path).stem

    original_path = OUTPUT_FOLDER / f"{image_name}_original.jpg"
    heatmap_path = OUTPUT_FOLDER / f"{image_name}_heatmap.jpg"
    overlay_path = OUTPUT_FOLDER / f"{image_name}_overlay.jpg"

    cv2.imwrite(str(original_path), original)
    cv2.imwrite(str(heatmap_path), colored_heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    # Display inspection report
    print("\nVISIONINSPECT AI - PATCH INSPECTION")
    print("-" * 45)

    print("Image:", Path(image_path).name)
    print("Status:", status)
    print("Anomalous patches:", anomalous_count)
    print("Total patches:", total_patches)
    print("Anomaly ratio:", f"{anomaly_ratio:.8f}")
    print("Decision threshold:", ANOMALY_THRESHOLD)

    print("\nSaved visualizations:")
    print("Original:", original_path)
    print("Heatmap:", heatmap_path)
    print("Overlay:", overlay_path)

    print("\nInspection completed!")

    return status, anomaly_ratio


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Test defective bottle image
    image_path = (
        DATASET_PATH
        / "test"
        / "broken_large"
        / "000.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    generate_heatmap(model, image_path)


if __name__ == "__main__":
    main()
