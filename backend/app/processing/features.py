"""
VisionInspect AI - Feature Extraction Helpers

Extracts structural, statistical, and geometric features from manufactured part images:
- Edges: Canny edge density, gradient energy, base64 edge map preview
- Texture Statistics: mean, variance, entropy, uniformity, skewness
- Shape Descriptors: contour count, dominant area, perimeter, circularity, solidity, bounding box
"""

import base64
import math
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


def extract_image_features(
    image: np.ndarray,
    generate_edge_preview: bool = True,
) -> Dict[str, Any]:
    """
    Computes visual and geometric features on a numpy BGR or grayscale image.

    Returns:
        {
            "edges": {...},
            "texture": {...},
            "shape": {...},
            "edge_preview_data_url": "data:image/png;base64,..." (optional)
        }
    """
    if image is None or image.size == 0:
        return {"edges": {}, "texture": {}, "shape": {}, "edge_preview_data_url": None}

    if len(image.shape) == 3:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        h, w = image.shape
        gray = image.copy()

    total_pixels = h * w if (h * w) > 0 else 1

    # ==========================================
    # 1. Edge Features
    # ==========================================
    canny = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_pixels = int(np.count_nonzero(canny))
    edge_density_pct = round((edge_pixels / total_pixels) * 100.0, 2)

    # Sobel gradient energies
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_gradient_magnitude = round(float(np.mean(grad_magnitude)), 2)

    edge_preview_url = None
    if generate_edge_preview:
        # Create dark background with neon cyan edge lines
        edge_vis = np.zeros((h, w, 3), dtype=np.uint8)
        edge_vis[canny > 0] = [238, 203, 56]  # Cyan/Sky-blue BGR
        success, encoded_buf = cv2.imencode(".png", edge_vis)
        if success:
            b64_str = base64.b64encode(encoded_buf).decode("utf-8")
            edge_preview_url = f"data:image/png;base64,{b64_str}"

    edges_data = {
        "edge_pixel_count": edge_pixels,
        "edge_density_pct": edge_density_pct,
        "mean_gradient_magnitude": mean_gradient_magnitude,
    }

    # ==========================================
    # 2. Texture Statistics (Histogram & Distribution)
    # ==========================================
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob_dist = hist / float(total_pixels)
    non_zero_probs = prob_dist[prob_dist > 0]

    # Shannon Entropy: -sum(p * log2(p))
    entropy = -float(np.sum(non_zero_probs * np.log2(non_zero_probs)))
    # Energy / Uniformity: sum(p^2)
    uniformity = float(np.sum(prob_dist**2))

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    variance_val = float(np.var(gray))

    # Skewness
    if std_val > 0:
        skewness = float(np.mean(((gray - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    texture_data = {
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "variance": round(variance_val, 2),
        "entropy": round(entropy, 3),
        "uniformity_energy": round(uniformity, 5),
        "skewness": round(skewness, 3),
    }

    # ==========================================
    # 3. Shape Descriptors (Foreground Morphology)
    # ==========================================
    # Otsu thresholding + contour analysis
    blurred_for_thresh = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny noise contours (< 20px area)
    min_area = 25.0
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    contour_count = len(valid_contours)
    dominant_area = 0.0
    dominant_perimeter = 0.0
    circularity = 0.0
    solidity = 0.0
    aspect_ratio = 1.0
    bbox = {"x": 0, "y": 0, "width": 0, "height": 0}

    if valid_contours:
        largest = max(valid_contours, key=cv2.contourArea)
        dominant_area = float(cv2.contourArea(largest))
        dominant_perimeter = float(cv2.arcLength(largest, closed=True))

        # Circularity: 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
        if dominant_perimeter > 0:
            circularity = round((4.0 * math.pi * dominant_area) / (dominant_perimeter**2), 3)

        # Solidity: Area / Convex Hull Area
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = round(dominant_area / hull_area, 3)

        bx, by, bw, bh = cv2.boundingRect(largest)
        bbox = {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)}
        aspect_ratio = round(float(bw) / float(bh), 3) if bh > 0 else 1.0

    shape_data = {
        "contour_count": contour_count,
        "dominant_area_px": round(dominant_area, 1),
        "dominant_perimeter_px": round(dominant_perimeter, 1),
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "bounding_box": bbox,
    }

    return {
        "edges": edges_data,
        "texture": texture_data,
        "shape": shape_data,
        "edge_preview_data_url": edge_preview_url,
    }
