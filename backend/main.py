
import uuid
import hashlib
import cv2
import joblib
import numpy as np

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "bottle_patch_anomaly_model.joblib"
)

OUTPUT_DIR = BASE_DIR / "artifacts" / "api_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
PATCH_SIZE = 16
ANOMALY_THRESHOLD = 0.0429


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("VisionInspect AI model loaded successfully!")
print("Model path:", MODEL_PATH)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="VisionInspect AI",
    description="AI-powered manufacturing defect inspection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# PATCH FEATURE EXTRACTION
# Must match patch_predict.py
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
# EXTRACT PATCHES
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
# INSPECT IMAGE
# --------------------------------------------------

def inspect_image_data(image_bgr):

    # Resize exactly as in patch_predict.py
    original = cv2.resize(
        image_bgr,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    # Extract patches
    patches = extract_patches(image)

    # Extract features
    patch_features = np.array([
        extract_patch_features(patch)
        for patch in patches
    ])

    # Model predictions
    predictions = model.predict(patch_features)

    anomalous_patches = int(
        np.sum(predictions == -1)
    )

    total_patches = len(patches)

    anomaly_ratio = (
        anomalous_patches / total_patches
    )

    # Image-level decision
    status = (
        "REVIEW"
        if anomaly_ratio >= ANOMALY_THRESHOLD
        else "PASS"
    )

    # Calculate anomaly scores
    anomaly_scores = -model.decision_function(
        patch_features
    )

    # Create 16 x 16 score map
    score_map = anomaly_scores.reshape(
        IMAGE_SIZE // PATCH_SIZE,
        IMAGE_SIZE // PATCH_SIZE
    )

    # Resize score map
    heatmap_resized = cv2.resize(
        score_map.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize heatmap
    normalized = cv2.normalize(
        heatmap_resized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Generate colored heatmap
    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # Generate overlay
    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return {
        "status": status,
        "anomalous_patches": anomalous_patches,
        "total_patches": total_patches,
        "anomaly_ratio": float(anomaly_ratio),
        "threshold": ANOMALY_THRESHOLD,
        "original": original,
        "heatmap": heatmap,
        "overlay": overlay
    }


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "VisionInspect AI API is running",
        "documentation": "/docs"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# IMAGE INSPECTION ENDPOINT
# --------------------------------------------------

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    # Verify exact uploaded image bytes
    image_hash = hashlib.sha256(contents).hexdigest()

    print("\n-----------------------------------")
    print("VISIONINSPECT AI - IMAGE RECEIVED")
    print("-----------------------------------")
    print("Filename:", file.filename)
    print("File size:", len(contents), "bytes")
    print("API IMAGE SHA256:", image_hash)

    # Decode uploaded image
    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read uploaded image."
        )

    print("Original image dimensions:", image_bgr.shape)

    # Run inspection
    result = inspect_image_data(image_bgr)

    # Generate unique inspection ID
    result_id = str(uuid.uuid4())

    # Output paths
    original_path = OUTPUT_DIR / f"{result_id}_original.jpg"
    heatmap_path = OUTPUT_DIR / f"{result_id}_heatmap.jpg"
    overlay_path = OUTPUT_DIR / f"{result_id}_overlay.jpg"

    # Save visualizations
    cv2.imwrite(str(original_path), result["original"])
    cv2.imwrite(str(heatmap_path), result["heatmap"])
    cv2.imwrite(str(overlay_path), result["overlay"])

    print("\nINSPECTION RESULT")
    print("Status:", result["status"])
    print("Anomalous patches:", result["anomalous_patches"])
    print("Total patches:", result["total_patches"])
    print("Anomaly ratio:", result["anomaly_ratio"])
    print("Threshold:", result["threshold"])
    print("-----------------------------------\n")

    # Return inspection report
    return {
        "inspection_id": result_id,
        "filename": file.filename,
        "status": result["status"],
        "anomalous_patches": result["anomalous_patches"],
        "total_patches": result["total_patches"],
        "anomaly_ratio": round(
            result["anomaly_ratio"], 8
        ),
        "decision_threshold": result["threshold"],
        "image_sha256": image_hash,
        "visualizations": {
            "original": str(
                original_path.relative_to(BASE_DIR)
            ),
            "heatmap": str(
                heatmap_path.relative_to(BASE_DIR)
            ),
            "overlay": str(
                overlay_path.relative_to(BASE_DIR)
            )
        },
        "note": (
            "Experimental patch anomaly score. "
            "Not a verified defect segmentation mask."
        )
    }


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
