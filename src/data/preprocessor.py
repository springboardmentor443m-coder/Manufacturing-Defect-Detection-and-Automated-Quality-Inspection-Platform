
from pathlib import Path
import cv2
import numpy as np


IMAGE_SIZE = (256, 256)


def preprocess_image(image_path):
    """
    Read an image, resize it, and normalize pixel values.
    Returns an RGB image with values between 0 and 1.
    """

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # OpenCV reads images in BGR format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for faster CPU processing.
    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize pixel values from 0-255 to 0-1.
    image = image.astype(np.float32) / 255.0

    return image


if __name__ == "__main__":

    sample_path = (
        Path(r"C:\Users\Sucha\Downloads\bottle")
        / "train"
        / "good"
        / "000.png"
    )

    processed_image = preprocess_image(sample_path)

    print("\nVISIONINSPECT AI - PREPROCESSING REPORT")
    print("-" * 45)
    print("Original image: 900 x 900")
    print("Processed image shape:", processed_image.shape)
    print("Data type:", processed_image.dtype)
    print("Minimum pixel value:", processed_image.min())
    print("Maximum pixel value:", processed_image.max())
    print("\nPreprocessing completed successfully!")
