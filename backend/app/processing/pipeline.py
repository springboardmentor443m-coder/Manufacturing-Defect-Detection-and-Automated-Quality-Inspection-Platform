"""
VisionInspect AI - Industrial Image Preprocessing Pipeline

Provides a configurable, step-toggleable OpenCV pipeline for manufacturing images:
- Validation (shape, channel integrity)
- Resize and normalize
- Noise reduction (Gaussian, Median, Bilateral)
- Contrast enhancement (CLAHE on luminance channel)
- Optional edge detection overlay / map
- Region of Interest (ROI) extraction
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class PreprocessingConfig:
    # 1. Validation
    do_validate: bool = True

    # 2. Resize
    do_resize: bool = True
    target_width: int = 512
    target_height: int = 512
    keep_aspect_ratio: bool = False

    # 3. Denoising
    do_denoise: bool = True
    denoise_method: str = "gaussian"  # "gaussian", "median", "bilateral"
    denoise_kernel_size: int = 3

    # 4. Contrast Enhancement (CLAHE)
    do_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # 5. Edge Detection
    do_edge_detection: bool = False
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150

    # 6. ROI Extraction
    do_roi_extraction: bool = False
    roi_mode: str = "center"  # "center", "bbox", "auto_foreground"
    roi_margin_percent: float = 0.1  # 10% margin crop if center
    roi_custom_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]


class PipelineValidationError(ValueError):
    """Raised when an input image fails validation."""
    pass


def run_preprocessing_pipeline(
    image: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes configured preprocessing steps on a copy of the input image.
    The original image array is never modified in-place.

    Returns:
        (processed_image_bgr, pipeline_metadata)
    """
    if config is None:
        config = PreprocessingConfig()

    start_time = time.time()
    steps_applied: List[str] = []

    # 1. Validation
    if config.do_validate:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise PipelineValidationError("Image data is empty, null, or not a valid numpy array.")
        if len(image.shape) not in (2, 3):
            raise PipelineValidationError(f"Invalid image dimensions: {image.shape}. Expected 2D or 3D.")
        steps_applied.append("validation")

    # Work on an isolated copy
    processed = image.copy()

    # Convert 1-channel grayscale to 3-channel BGR for standardized pipeline flow
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    orig_h, orig_w = processed.shape[:2]

    # 2. ROI Extraction (if enabled before scaling)
    if config.do_roi_extraction:
        h, w = processed.shape[:2]
        if config.roi_mode == "center":
            margin_y = int(h * config.roi_margin_percent)
            margin_x = int(w * config.roi_margin_percent)
            y1 = max(0, margin_y)
            y2 = min(h, h - margin_y)
            x1 = max(0, margin_x)
            x2 = min(w, w - margin_x)
            processed = processed[y1:y2, x1:x2]
            steps_applied.append(f"roi_extraction(center: [{y1}:{y2}, {x1}:{x2}])")

        elif config.roi_mode == "bbox" and config.roi_custom_box:
            ymin, xmin, ymax, xmax = config.roi_custom_box
            ymin = max(0, min(ymin, h))
            ymax = max(ymin + 1, min(ymax, h))
            xmin = max(0, min(xmin, w))
            xmax = max(xmin + 1, min(xmax, w))
            processed = processed[ymin:ymax, xmin:xmax]
            steps_applied.append(f"roi_extraction(bbox: [{ymin}:{ymax}, {xmin}:{xmax}])")

        elif config.roi_mode == "auto_foreground":
            # Otsu thresholding on grayscale to find dominant object contour
            gray_tmp = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                if bw > 20 and bh > 20:
                    processed = processed[by:by + bh, bx:bx + bw]
                    steps_applied.append(f"roi_extraction(auto_foreground: [{by}:{by+bh}, {bx}:{bx+bw}])")

    # 3. Resize
    if config.do_resize:
        tw, th = config.target_width, config.target_height
        if config.keep_aspect_ratio:
            cur_h, cur_w = processed.shape[:2]
            scale = min(tw / cur_w, th / cur_h)
            nw, nh = int(cur_w * scale), int(cur_h * scale)
            resized = cv2.resize(processed, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            # Pad onto letterbox canvas
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            pad_x = (tw - nw) // 2
            pad_y = (th - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
            processed = canvas
        else:
            processed = cv2.resize(processed, (tw, th), interpolation=cv2.INTER_AREA if (tw < orig_w or th < orig_h) else cv2.INTER_LINEAR)
        steps_applied.append(f"resize({config.target_width}x{config.target_height})")

    # 4. Noise Reduction
    if config.do_denoise:
        k = config.denoise_kernel_size
        if k % 2 == 0:
            k += 1  # Kernel must be odd
        k = max(3, k)

        method = config.denoise_method.lower()
        if method == "median":
            processed = cv2.medianBlur(processed, k)
            steps_applied.append(f"denoise(median, k={k})")
        elif method == "bilateral":
            processed = cv2.bilateralFilter(processed, d=k * 2, sigmaColor=75, sigmaSpace=75)
            steps_applied.append(f"denoise(bilateral, k={k})")
        else:
            # Default Gaussian
            processed = cv2.GaussianBlur(processed, (k, k), 0)
            steps_applied.append(f"denoise(gaussian, k={k})")

    # 5. Contrast Enhancement (CLAHE on L-channel of LAB to preserve true color)
    if config.do_clahe:
        clip = float(config.clahe_clip_limit)
        grid = max(2, int(config.clahe_grid_size))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = clahe.apply(l_channel)
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        processed = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        steps_applied.append(f"clahe(clip={clip}, grid={grid}x{grid})")

    # 6. Optional Edge Detection
    if config.do_edge_detection:
        gray_edges = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray_edges,
            threshold1=config.edge_low_threshold,
            threshold2=config.edge_high_threshold,
        )
        # Convert edges to 3-channel green highlight over dark background or overlay
        edge_bgr = np.zeros_like(processed)
        edge_bgr[edges > 0] = [0, 255, 0]  # Green contours
        # Blend 70% processed + 30% edge highlight
        processed = cv2.addWeighted(processed, 0.75, edge_bgr, 0.25, 0)
        steps_applied.append(f"edge_detection(canny: {config.edge_low_threshold}-{config.edge_high_threshold})")

    duration_ms = round((time.time() - start_time) * 1000.0, 2)
    final_h, final_w = processed.shape[:2]

    pipeline_metadata = {
        "steps_applied": steps_applied,
        "input_dimensions": {"width": orig_w, "height": orig_h},
        "output_dimensions": {"width": final_w, "height": final_h},
        "duration_ms": duration_ms,
        "config": asdict(config),
    }

    return processed, pipeline_metadata
