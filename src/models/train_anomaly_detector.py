
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.feature_extractor import extract_features


DATASET_PATH = Path(r"C:\Users\Sucha\Downloads\bottle")
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "bottle_anomaly_model.joblib"


def train_model():

    # Load only normal training images
    train_folder = DATASET_PATH / "train" / "good"

    image_paths = sorted([
        path for path in train_folder.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if not image_paths:
        raise ValueError("No training images found!")

    print("\nVISIONINSPECT AI - MODEL TRAINING")
    print("-" * 45)
    print("Loading normal training images...")

    # Extract features from every normal image
    feature_list = []

    for index, image_path in enumerate(image_paths, start=1):
        features = extract_features(image_path)
        feature_list.append(features)

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    X_train = np.array(feature_list)

    print("\nTraining feature matrix:", X_train.shape)

    # Create anomaly detection pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        ))
    ])

    # Train using normal images only
    model.fit(X_train)

    # Save trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nMODEL TRAINING COMPLETED!")
    print("Training images:", len(image_paths))
    print("Features per image:", X_train.shape[1])
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
