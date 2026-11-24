
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import preprocess_image


def extract_features(image_path):
    """Extract basic color and texture features."""

    image = preprocess_image(image_path)

    gray = cv2.cvtColor(
        (image * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(gray, 50, 150)

    color_mean = np.mean(image, axis=(0, 1))
    color_std = np.std(image, axis=(0, 1))
    edge_density = np.mean(edges > 0)

    features = np.concatenate([
        color_mean,
        color_std,
        [edge_density]
    ])

    return features


if __name__ == "__main__":

    sample_image = (
        Path(r"C:\Users\Sucha\Downloads\bottle")
        / "train"
        / "good"
        / "000.png"
    )

    features = extract_features(sample_image)

    print("\nVISIONINSPECT AI - FEATURE EXTRACTION")
    print("-" * 45)
    print("Extracted features:", features)
    print("Feature vector size:", features.shape)
    print("\nFeature extraction successful!")
