import requests
import json
import os

# 1. Login
login_res = requests.post(
    "http://127.0.0.1:8000/api/v1/auth/login",
    json={"email": "admin@visioninspect.ai", "password": "Admin@123"},
).json()
token = login_res["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload a real image
image_file = "data/mvtec_ad/bottle/train/good/000.png"
with open(image_file, "rb") as f:
    upload_res = requests.post(
        "http://127.0.0.1:8000/api/v1/inspections/upload",
        headers=headers,
        data={"product_id": 1},
        files={"file": ("000.png", f, "image/png")},
    ).json()

insp_id = upload_res["id"]
orig_path = upload_res["image_path"]
print(f"Uploaded Inspection ID: {insp_id}")
print(f"Original Image Path: {orig_path}")
print(f"Original file exists: {os.path.exists(orig_path)}")

# 3. Call Preprocessing pipeline
proc_config = {
    "do_validate": True,
    "do_resize": True,
    "target_width": 512,
    "target_height": 512,
    "keep_aspect_ratio": True,
    "do_denoise": True,
    "denoise_method": "gaussian",
    "denoise_kernel_size": 5,
    "do_clahe": True,
    "clahe_clip_limit": 2.5,
    "clahe_grid_size": 8,
    "do_edge_detection": False,
    "do_roi_extraction": False,
}
proc_res = requests.post(
    f"http://127.0.0.1:8000/api/v1/inspections/{insp_id}/preprocess",
    headers=headers,
    json=proc_config,
).json()

proc_path = proc_res["processed_image_path"]
print("=== Preprocess Response ===")
print(f"Processed Image Path: {proc_path}")
print(f"Processed file exists: {os.path.exists(proc_path)}")
print(f"Stored Separately: {proc_path != orig_path}")
print(f"Original untouched: {os.path.exists(orig_path)}")
print(f"Quality Verdict: {proc_res['quality_verdict']}")
print(f"Quality Score: {proc_res['quality_score']}")

# 4. Call Quality Report endpoint
qr_res = requests.get(
    f"http://127.0.0.1:8000/api/v1/inspections/{insp_id}/quality-report",
    headers=headers,
).json()
print("=== Quality Report ===")
print(f"Blur Score: {qr_res['blur_score']}")
print(f"Brightness: {qr_res['brightness']}")
print(f"Contrast: {qr_res['contrast']}")
print(f"Noise Estimate: {qr_res['noise_estimate']}")
print(f"Verdict: {qr_res['verdict']}")
print(f"Reasons: {qr_res['reasons']}")
