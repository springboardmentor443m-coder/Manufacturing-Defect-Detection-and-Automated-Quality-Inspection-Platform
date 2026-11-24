import sys
from pathlib import Path

import cv2
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATASET_PATH = Path(r"C:\Users\Sucha\Downloads\bottle")

# Model output directory
MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "bottle_patch_anomaly_model.joblib"
)

# Image processing settings
IMAGE_SIZE = 256
PATCH_SIZE = 16
def extract_patches(image):
    """
    Divide an image into smaller 16x16 patches.
    """

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
def extract_patch_features(patch):
    """
    Extract visual features from one image patch.
    """

    # Convert RGB patch into grayscale
    gray = cv2.cvtColor(
        (patch * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    # Calculate average colour
    color_mean = np.mean(patch, axis=(0, 1))

    # Calculate colour variation
    color_std = np.std(patch, axis=(0, 1))

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Calculate edge density
    edge_density = np.mean(edges > 0)

    # Calculate grayscale texture variation
    gray_std = np.std(gray) / 255.0

    # Combine all features
    features = np.concatenate([
        color_mean,
        color_std,
        [edge_density, gray_std]
    ])

    return features
def preprocess_image(image_path):
    """
    Read and resize an image for patch extraction.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 256 x 256
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize pixel values between 0 and 1
    image = image.astype(np.float32) / 255.0

    return image
def collect_training_features():
    """
    Extract patch features from all normal training images.
    """

    train_folder = DATASET_PATH / "train" / "good"

    image_paths = sorted([
        path for path in train_folder.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if not image_paths:
        raise ValueError("No normal training images found!")

    all_features = []

    print("\nCollecting normal image patches...")
    print("Training images:", len(image_paths))

    for index, image_path in enumerate(image_paths, start=1):

        # Prepare image
        image = preprocess_image(image_path)

        # Divide image into patches
        patches = extract_patches(image)

        # Extract features from every patch
        for patch in patches:
            features = extract_patch_features(patch)
            all_features.append(features)

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    X_train = np.array(all_features)

    print("\nPATCH TRAINING DATA REPORT")
    print("-" * 40)
    print("Total training images:", len(image_paths))
    print("Patches per image:", len(patches))
    print("Total patches:", len(X_train))
    print("Features per patch:", X_train.shape[1])

    return X_train
def train_patch_model():
    """
    Train an anomaly detector using normal image patches.
    """

    print("\nVISIONINSPECT AI - PATCH MODEL TRAINING")
    print("-" * 45)

    # Collect features from normal training images
    X_train = collect_training_features()

    print("\nStarting model training...")

    # Create machine learning pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train the model
    model.fit(X_train)

    # Create model directory
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nPATCH MODEL TRAINING COMPLETED!")
    print("Training patches:", len(X_train))
    print("Features per patch:", X_train.shape[1])
    print("Saved model:", MODEL_PATH)
if __name__ == "__main__":
    train_patch_model()