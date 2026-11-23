"""
Unit Tests for VisionInspect AI - Image Quality Assessment & Preprocessing

Tests verify:
- Focus/blur detection (sharp vs blurred images)
- Exposure analysis (underexposed, normal, overexposed)
- Contrast evaluation (low dynamic range vs high contrast)
- Noise estimation (clean vs noisy image)
- Pass/warn/fail decision logic and actionable reasons
- Pipeline execution & non-destructive image preservation
- Feature extraction sanity
"""

import cv2
import numpy as np
import pytest

from app.processing.features import extract_image_features
from app.processing.pipeline import (
    PipelineValidationError,
    PreprocessingConfig,
    run_preprocessing_pipeline,
)
from app.processing.quality import QualityThresholds, assess_image_quality


@pytest.fixture
def sharp_checkerboard() -> np.ndarray:
    """Creates a high-contrast, perfectly sharp 200x200 checkerboard pattern."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    square_size = 20
    for y in range(0, 200, square_size):
        for x in range(0, 200, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                img[y:y + square_size, x:x + square_size] = [230, 230, 230]
            else:
                img[y:y + square_size, x:x + square_size] = [40, 40, 40]
    return img


@pytest.fixture
def dark_image() -> np.ndarray:
    """Creates an underexposed, nearly black image."""
    return np.full((100, 100, 3), 15, dtype=np.uint8)


@pytest.fixture
def bright_image() -> np.ndarray:
    """Creates an overexposed, blown-out image."""
    return np.full((100, 100, 3), 245, dtype=np.uint8)


@pytest.fixture
def flat_gray_image() -> np.ndarray:
    """Creates a zero-contrast uniform gray image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


def test_blur_detection(sharp_checkerboard: np.ndarray):
    """Verifies that a sharp image gets high blur score, and blurring drastically reduces it."""
    # 1. Sharp image assessment
    res_sharp = assess_image_quality(sharp_checkerboard)
    assert res_sharp["blur_score"] > QualityThresholds.BLUR_WARN
    assert res_sharp["metrics_breakdown"]["blur"]["status"] == "pass"

    # 2. Heavily blurred version
    blurred = cv2.GaussianBlur(sharp_checkerboard, (31, 31), 10.0)
    res_blurred = assess_image_quality(blurred)
    assert res_blurred["blur_score"] < QualityThresholds.BLUR_FAIL
    assert res_blurred["metrics_breakdown"]["blur"]["status"] == "fail"
    assert res_blurred["verdict"] == "fail"
    assert any("blurry" in r.lower() or "focus" in r.lower() for r in res_blurred["reasons"])


def test_exposure_detection(dark_image: np.ndarray, bright_image: np.ndarray):
    """Verifies that underexposure and overexposure trigger fail verdicts with clear explanations."""
    # Dark image
    res_dark = assess_image_quality(dark_image)
    assert res_dark["brightness"] < QualityThresholds.BRIGHTNESS_MIN_FAIL
    assert res_dark["verdict"] == "fail"
    assert any("dark" in r.lower() or "underexposure" in r.lower() for r in res_dark["reasons"])

    # Bright image
    res_bright = assess_image_quality(bright_image)
    assert res_bright["brightness"] > QualityThresholds.BRIGHTNESS_MAX_FAIL
    assert res_bright["verdict"] == "fail"
    assert any("overexposure" in r.lower() or "saturation" in r.lower() for r in res_bright["reasons"])


def test_contrast_detection(flat_gray_image: np.ndarray, sharp_checkerboard: np.ndarray):
    """Verifies that flat uniform images fail contrast checks, while patterned ones pass."""
    # Flat zero contrast
    res_flat = assess_image_quality(flat_gray_image)
    assert res_flat["contrast"] == 0.0
    assert res_flat["verdict"] == "fail"
    assert any("contrast" in r.lower() or "dynamic range" in r.lower() for r in res_flat["reasons"])

    # Normal contrast
    res_pattern = assess_image_quality(sharp_checkerboard)
    assert res_pattern["contrast"] > QualityThresholds.CONTRAST_WARN
    assert res_pattern["metrics_breakdown"]["contrast"]["status"] == "pass"


def test_noise_estimation(sharp_checkerboard: np.ndarray):
    """Verifies that injected sensor noise increases the noise estimate and affects quality score."""
    res_clean = assess_image_quality(sharp_checkerboard)
    clean_noise = res_clean["noise_estimate"]

    # Add strong zero-mean Gaussian noise
    noisy = sharp_checkerboard.copy().astype(np.float32)
    gaussian_noise = np.random.normal(0, 35, noisy.shape)
    noisy = np.clip(noisy + gaussian_noise, 0, 255).astype(np.uint8)

    res_noisy = assess_image_quality(noisy)
    assert res_noisy["noise_estimate"] > clean_noise
    assert res_noisy["noise_estimate"] > QualityThresholds.NOISE_WARN
    assert res_noisy["overall_score"] < res_clean["overall_score"]


def test_empty_corrupted_image_handling():
    """Verifies that empty or invalid image buffers produce clear failure reports without throwing uncaught exceptions."""
    empty = np.array([], dtype=np.uint8)
    res = assess_image_quality(empty)
    assert res["verdict"] == "fail"
    assert len(res["reasons"]) > 0


def test_preprocessing_pipeline_nondestructive(sharp_checkerboard: np.ndarray):
    """Verifies that running the preprocessing pipeline leaves the original array untouched."""
    original_copy = sharp_checkerboard.copy()
    config = PreprocessingConfig(
        do_resize=True,
        target_width=128,
        target_height=128,
        do_denoise=True,
        denoise_method="median",
        do_clahe=True,
        do_edge_detection=True,
    )

    processed, meta = run_preprocessing_pipeline(sharp_checkerboard, config)

    # 1. Original image unchanged
    np.testing.assert_array_equal(sharp_checkerboard, original_copy)

    # 2. Processed image matches target dimensions
    assert processed.shape == (128, 128, 3)
    assert "resize(128x128)" in meta["steps_applied"]
    assert any("clahe" in s for s in meta["steps_applied"])
    assert any("edge_detection" in s for s in meta["steps_applied"])
    assert meta["duration_ms"] >= 0.0


def test_feature_extraction(sharp_checkerboard: np.ndarray):
    """Verifies that feature extraction returns structural, statistical, and shape descriptors."""
    features = extract_image_features(sharp_checkerboard, generate_edge_preview=True)

    # Edges
    assert "edges" in features
    assert features["edges"]["edge_pixel_count"] > 0
    assert features["edges"]["edge_density_pct"] > 0.0
    assert features["edge_preview_data_url"] is not None
    assert features["edge_preview_data_url"].startswith("data:image/png;base64,")

    # Texture
    assert "texture" in features
    assert features["texture"]["entropy"] > 0.0
    assert features["texture"]["variance"] > 0.0

    # Shape
    assert "shape" in features
    assert "contour_count" in features["shape"]
    assert features["shape"]["contour_count"] > 0
    assert "circularity" in features["shape"]
