"""
VisionInspect AI - Image Quality Assessment Module

Analyzes uploaded manufacturing images for optical and acquisition defects:
- Blur detection via Laplacian variance
- Brightness and exposure analysis
- RMS contrast estimation
- High-frequency noise estimation
- Automated pass/warn/fail verdict with actionable engineering reasons
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class QualityThresholds:
    """Configurable industrial imaging quality thresholds."""
    BLUR_FAIL: float = 60.0        # < 60: severe blur
    BLUR_WARN: float = 120.0       # 60 - 120: marginal focus
    
    BRIGHTNESS_MIN_FAIL: float = 40.0   # < 40: underexposed / dark
    BRIGHTNESS_MIN_WARN: float = 55.0   # 40 - 55: dim
    BRIGHTNESS_MAX_WARN: float = 200.0  # 200 - 225: bright
    BRIGHTNESS_MAX_FAIL: float = 225.0  # > 225: overexposed / blown out
    
    CONTRAST_FAIL: float = 22.0    # < 22: severely washed out
    CONTRAST_WARN: float = 32.0    # 22 - 32: low dynamic range
    
    NOISE_WARN: float = 12.0       # 12 - 22: noticeable sensor noise
    NOISE_FAIL: float = 22.0       # > 22: unacceptable optical noise


def assess_image_quality(
    image: np.ndarray,
    thresholds: Optional[QualityThresholds] = None,
) -> Dict[str, Any]:
    """
    Evaluates optical quality metrics on a numpy BGR/grayscale image.

    Returns:
        dict containing:
            - resolution: {width, height, megapixels, aspect_ratio}
            - blur_score: float (Laplacian variance)
            - brightness: float (0 - 255 mean intensity)
            - contrast: float (RMS standard deviation)
            - noise_estimate: float (estimated sigma)
            - overall_score: float (0 - 100)
            - verdict: 'pass' | 'warn' | 'fail'
            - reasons: List[str]
            - metrics_breakdown: detailed status for each parameter
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    if image is None or image.size == 0:
        return {
            "verdict": "fail",
            "overall_score": 0.0,
            "reasons": ["Image buffer is empty or corrupted."],
            "resolution": {"width": 0, "height": 0, "megapixels": 0.0, "aspect_ratio": 0.0},
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "noise_estimate": 0.0,
            "metrics_breakdown": {},
        }

    # Normalize to grayscale for photometric & frequency analysis
    if len(image.shape) == 3:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        h, w = image.shape
        gray = image.copy()

    megapixels = round((w * h) / 1_000_000.0, 3)
    aspect_ratio = round(float(w) / float(h), 3) if h > 0 else 0.0

    # 1. Blur score: variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Brightness: mean intensity
    brightness = float(np.mean(gray))

    # 3. Contrast: standard deviation of pixel intensities (RMS contrast)
    contrast = float(np.std(gray))

    # 4. Noise estimate: difference between image and median-filtered image
    median_filtered = cv2.medianBlur(gray, 3)
    noise_diff = cv2.absdiff(gray, median_filtered)
    noise_estimate = float(np.std(noise_diff))

    # Evaluate verdict and compile reasons
    reasons: List[str] = []
    verdict = "pass"
    breakdown: Dict[str, Dict[str, Any]] = {}

    # Check blur
    if blur_score < thresholds.BLUR_FAIL:
        verdict = "fail"
        reasons.append(
            f"Image is out of focus / blurry (Laplacian variance {blur_score:.1f} < {thresholds.BLUR_FAIL:.1f})."
        )
        blur_status = "fail"
    elif blur_score < thresholds.BLUR_WARN:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"Suboptimal sharpness (Laplacian variance {blur_score:.1f} < {thresholds.BLUR_WARN:.1f})."
        )
        blur_status = "warn"
    else:
        blur_status = "pass"

    breakdown["blur"] = {
        "value": round(blur_score, 2),
        "status": blur_status,
        "label": "Focus & Sharpness",
        "unit": "variance",
    }

    # Check brightness
    if brightness < thresholds.BRIGHTNESS_MIN_FAIL:
        verdict = "fail"
        reasons.append(
            f"Severe underexposure / image too dark (mean brightness {brightness:.1f} < {thresholds.BRIGHTNESS_MIN_FAIL:.1f})."
        )
        bright_status = "fail"
    elif brightness > thresholds.BRIGHTNESS_MAX_FAIL:
        verdict = "fail"
        reasons.append(
            f"Severe overexposure / sensor saturation (mean brightness {brightness:.1f} > {thresholds.BRIGHTNESS_MAX_FAIL:.1f})."
        )
        bright_status = "fail"
    elif brightness < thresholds.BRIGHTNESS_MIN_WARN:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"Lighting is dim (mean brightness {brightness:.1f} < {thresholds.BRIGHTNESS_MIN_WARN:.1f})."
        )
        bright_status = "warn"
    elif brightness > thresholds.BRIGHTNESS_MAX_WARN:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"Lighting is high / near saturation (mean brightness {brightness:.1f} > {thresholds.BRIGHTNESS_MAX_WARN:.1f})."
        )
        bright_status = "warn"
    else:
        bright_status = "pass"

    breakdown["brightness"] = {
        "value": round(brightness, 1),
        "status": bright_status,
        "label": "Exposure & Brightness",
        "unit": "/255",
    }

    # Check contrast
    if contrast < thresholds.CONTRAST_FAIL:
        verdict = "fail"
        reasons.append(
            f"Critically low dynamic range / contrast (std dev {contrast:.1f} < {thresholds.CONTRAST_FAIL:.1f})."
        )
        contrast_status = "fail"
    elif contrast < thresholds.CONTRAST_WARN:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"Low contrast (std dev {contrast:.1f} < {thresholds.CONTRAST_WARN:.1f})."
        )
        contrast_status = "warn"
    else:
        contrast_status = "pass"

    breakdown["contrast"] = {
        "value": round(contrast, 1),
        "status": contrast_status,
        "label": "RMS Contrast",
        "unit": "std",
    }

    # Check noise
    if noise_estimate > thresholds.NOISE_FAIL:
        verdict = "fail"
        reasons.append(
            f"Severe sensor noise detected (noise estimate {noise_estimate:.1f} > {thresholds.NOISE_FAIL:.1f})."
        )
        noise_status = "fail"
    elif noise_estimate > thresholds.NOISE_WARN:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"Elevated high-frequency noise (noise estimate {noise_estimate:.1f} > {thresholds.NOISE_WARN:.1f})."
        )
        noise_status = "warn"
    else:
        noise_status = "pass"

    breakdown["noise"] = {
        "value": round(noise_estimate, 2),
        "status": noise_status,
        "label": "Sensor Noise",
        "unit": "sigma",
    }

    # Compute continuous overall quality score (0 - 100)
    # 1. Blur sub-score (0-100)
    blur_sub = min(100.0, (blur_score / (thresholds.BLUR_WARN * 1.5)) * 100.0)
    # 2. Brightness sub-score (ideal around 128)
    bright_diff = abs(brightness - 128.0)
    bright_sub = max(0.0, 100.0 - (bright_diff / 128.0) * 100.0)
    # 3. Contrast sub-score
    contrast_sub = min(100.0, (contrast / (thresholds.CONTRAST_WARN * 1.8)) * 100.0)
    # 4. Noise sub-score
    noise_sub = max(0.0, 100.0 - (noise_estimate / thresholds.NOISE_FAIL) * 100.0)

    composite_score = (
        blur_sub * 0.35 +
        bright_sub * 0.25 +
        contrast_sub * 0.25 +
        noise_sub * 0.15
    )
    overall_score = round(max(0.0, min(100.0, composite_score)), 1)

    if verdict == "fail":
        overall_score = min(overall_score, 49.0)
    elif verdict == "warn":
        overall_score = min(overall_score, 79.0)
        overall_score = max(overall_score, 50.0)
    else:
        overall_score = max(overall_score, 80.0)

    return {
        "verdict": verdict,
        "overall_score": overall_score,
        "reasons": reasons,
        "resolution": {
            "width": w,
            "height": h,
            "megapixels": megapixels,
            "aspect_ratio": aspect_ratio,
        },
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "noise_estimate": round(noise_estimate, 2),
        "metrics_breakdown": breakdown,
    }
