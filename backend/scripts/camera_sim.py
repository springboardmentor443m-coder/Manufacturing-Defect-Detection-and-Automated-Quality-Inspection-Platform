#!/usr/bin/env python3
"""
VisionInspect AI - Production Line Camera Simulation Script

Simulates an industrial automated inspection camera (e.g. Basler, Cognex, Keyence)
capturing parts on a moving conveyor belt and pushing image frames to the
VisionInspect AI image acquisition pipeline at a configured cycle interval.
"""

import argparse
import io
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from PIL import Image, ImageDraw


def generate_synthetic_manufacturing_images(output_dir: Path, count: int = 5) -> List[Path]:
    """Generates synthetic metal surface defect and clean images for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    defect_types = ["clean_surface", "scratch", "surface_crack", "pit_defect", "discoloration"]

    for i in range(count):
        defect = defect_types[i % len(defect_types)]
        filename = f"sim_camera_frame_{i+1:03d}_{defect}.png"
        file_path = output_dir / filename

        # Create metallic gray textured background (e.g. 800x600 resolution)
        width, height = 800, 600
        base_gray = random.randint(140, 170)
        img = Image.new("RGB", (width, height), color=(base_gray, base_gray, base_gray + 5))
        draw = ImageDraw.Draw(img)

        # Draw subtle machined brush lines
        for y in range(0, height, 4):
            line_tone = base_gray + random.randint(-15, 15)
            draw.line([(0, y), (width, y)], fill=(line_tone, line_tone, line_tone + 3), width=1)

        # Draw simulated defects if not clean
        if defect == "scratch":
            # Linear scratch
            x0 = random.randint(150, 400)
            y0 = random.randint(150, 350)
            x1 = x0 + random.randint(100, 300)
            y1 = y0 + random.randint(-40, 80)
            draw.line([(x0, y0), (x1, y1)], fill=(40, 40, 45), width=2)
            draw.line([(x0, y0 + 1), (x1, y1 + 1)], fill=(220, 220, 230), width=1)

        elif defect == "surface_crack":
            # Zig-zag crack
            points = [(random.randint(200, 300), random.randint(200, 300))]
            for _ in range(6):
                px, py = points[-1]
                points.append((px + random.randint(20, 60), py + random.randint(-25, 35)))
            draw.line(points, fill=(25, 25, 30), width=3)

        elif defect == "pit_defect":
            # Dark pit inclusion
            cx, cy = random.randint(250, 550), random.randint(200, 400)
            r = random.randint(15, 35)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(30, 30, 35), outline=(15, 15, 20))

        elif defect == "discoloration":
            # Heat discoloration patch
            cx, cy = random.randint(300, 500), random.randint(250, 400)
            rx, ry = random.randint(40, 90), random.randint(30, 70)
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=(base_gray + 20, base_gray - 30, base_gray - 40))

        # Add simulated camera timestamp overlay
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        draw.text((20, height - 30), f"CAM-01 [STATION-4] {timestamp_str} SKU-TEST", fill=(255, 255, 255))

        img.save(file_path, "PNG")
        generated_files.append(file_path)

    print(f"✨ Generated {len(generated_files)} synthetic production camera images in '{output_dir}'.")
    return generated_files


def authenticate(api_url: str, email: str, password: str) -> Optional[str]:
    """Logs into VisionInspect AI and retrieves JWT access token."""
    login_url = f"{api_url}/auth/login"
    try:
        resp = httpx.post(login_url, json={"email": email, "password": password}, timeout=10.0)
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            return token
        else:
            print(f"❌ Authentication failed ({resp.status_code}): {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Cannot connect to API at {login_url}: {e}")
        return None


def run_camera_simulation(
    folder_path: str,
    interval: float,
    api_url: str,
    product_id: int,
    email: str,
    password: str,
    count: int = 0,
    batch_size: int = 1,
):
    print("=" * 70)
    print(" Industrial Camera Acquisition Simulator - VisionInspect AI")
    print(f" Station: Line-1 Conveyor | Trigger Interval: {interval:.2f}s | Target Product: #{product_id}")
    print(f" Endpoint: {api_url}/inspections/upload")
    print("=" * 70)

    # 1. Authenticate with backend
    print(f"\n🔐 Authenticating as '{email}'...")
    token = authenticate(api_url, email, password)
    if not token:
        print("❌ Aborting simulation: Authentication failed. Please ensure the backend is running.")
        sys.exit(1)
    print("✅ Authentication successful. JWT session token acquired.\n")

    headers = {"Authorization": f"Bearer {token}"}

    # 2. Locate or generate image files
    folder = Path(folder_path).resolve()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    image_files = []
    if folder.exists() and folder.is_dir():
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"⚠️ No image files found in '{folder}'. Generating synthetic camera frames...")
        image_files = generate_synthetic_manufacturing_images(folder, count=6)

    print(f"📁 Loaded {len(image_files)} source frame(s) for conveyor simulation.\n")

    # 3. Conveyor capture loop
    captured_count = 0
    cycle_index = 0

    try:
        while True:
            cycle_index += 1
            if count > 0 and captured_count >= count:
                print(f"\n🏁 Simulation completed: Captured and uploaded {captured_count} target frames.")
                break

            # Pick image file (sequential cycle or random)
            selected_file = image_files[(cycle_index - 1) % len(image_files)]

            start_time = time.time()
            trigger_ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

            print(f"[{trigger_ts}] 📷 CAMERA TRIGGER #{cycle_index} -> Captured: {selected_file.name} ({selected_file.stat().st_size / 1024:.1f} KB)")

            # Post to upload endpoint
            with open(selected_file, "rb") as f:
                file_bytes = f.read()

            files = {"file": (selected_file.name, file_bytes, "image/png")}
            data = {"product_id": str(product_id)}

            try:
                resp = httpx.post(
                    f"{api_url}/inspections/upload",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=15.0,
                )
                latency_ms = (time.time() - start_time) * 1000

                if resp.status_code == 201:
                    result = resp.json()
                    insp_id = result.get("id")
                    img_w = result.get("image_width")
                    img_h = result.get("image_height")
                    print(f"   ↳ 🟢 Ingested Inspection #{insp_id} | Dimensions: {img_w}x{img_h}px | Upload Latency: {latency_ms:.1f}ms")
                    captured_count += 1
                else:
                    print(f"   ↳ 🔴 Upload Failed [{resp.status_code}]: {resp.text}")

            except Exception as req_err:
                print(f"   ↳ ⚠️ Network or connection error: {req_err}")

            # Wait for next conveyor cycle
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n🛑 Camera simulation stopped by operator. Ingested total: {captured_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="VisionInspect AI Camera Acquisition Simulator")
    parser.add_argument(
        "--folder",
        default="data/sample_images",
        help="Folder containing source inspection images (default: data/sample_images)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Conveyor capture interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/api/v1",
        help="VisionInspect API base URL (default: http://127.0.0.1:8000/api/v1)",
    )
    parser.add_argument(
        "--product-id",
        type=int,
        default=1,
        help="Target manufactured product ID (default: 1)",
    )
    parser.add_argument(
        "--email",
        default="qe@visioninspect.ai",
        help="Operator account email (default: qe@visioninspect.ai)",
    )
    parser.add_argument(
        "--password",
        default="Engineer@123",
        help="Operator account password (default: Engineer@123)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of frames to capture (0 = run continuously)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per trigger",
    )

    args = parser.parse_args()
    run_camera_simulation(
        folder_path=args.folder,
        interval=args.interval,
        api_url=args.api_url,
        product_id=args.product_id,
        email=args.email,
        password=args.password,
        count=args.count,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
