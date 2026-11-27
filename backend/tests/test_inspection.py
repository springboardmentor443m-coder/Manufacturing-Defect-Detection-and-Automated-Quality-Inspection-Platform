import cv2
import numpy as np

from app.models import ValidationStatus
from app.services.image_validation import validate_image


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _textured(seed=0, size=200) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


# --- validation rules -------------------------------------------------------

def test_valid_image():
    r = validate_image(_png(_textured()), "part.png")
    assert r.status == ValidationStatus.valid
    assert (r.width, r.height, r.channels) == (200, 200, 3)


def test_rejects_wrong_extension():
    assert validate_image(_png(_textured()), "notes.txt").status == ValidationStatus.rejected


def test_rejects_corrupt_file():
    assert validate_image(b"definitely not a png", "x.png").status == ValidationStatus.rejected


def test_rejects_blank_and_tiny():
    assert validate_image(_png(np.full((200, 200, 3), 128, np.uint8)), "b.png").status == ValidationStatus.rejected
    assert validate_image(_png(_textured(size=32)), "t.png").status == ValidationStatus.rejected


def test_warns_on_blur():
    # A part-like image with clear structure, then defocused: blurry but not blank.
    part = np.full((200, 200, 3), 60, np.uint8)
    cv2.circle(part, (100, 100), 70, (200, 200, 200), -1)
    cv2.rectangle(part, (80, 80), (120, 120), (30, 30, 30), -1)
    blurry = cv2.GaussianBlur(part, (0, 0), 8)
    r = validate_image(_png(blurry), "blur.png")
    assert r.status == ValidationStatus.warning
    assert any("blurry" in m for m in r.messages)


# --- upload workflow --------------------------------------------------------

def test_upload_batch(client, engineer):
    files = [
        ("files", ("a.png", _png(_textured(1)), "image/png")),
        ("files", ("b.png", _png(_textured(2)), "image/png")),
        ("files", ("bad.png", b"garbage", "image/png")),
    ]
    r = client.post("/api/batches/upload", headers=engineer, files=files,
                    data={"product_category": "Bottle", "batch_name": "Morning shift"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["accepted"] == 2 and body["rejected"] == 1
    assert body["batch"]["product_category"] == "bottle"

    detail = client.get(f"/api/batches/{body['batch']['id']}", headers=engineer).json()
    assert len(detail["images"]) == 3
    img = next(i for i in detail["images"] if i["original_filename"] == "a.png")
    assert img["inspection_status"] == "pending"
    file_resp = client.get(f"/api/images/{img['id']}/file", headers=engineer)
    assert file_resp.status_code == 200 and len(file_resp.content) > 0


def test_duplicate_upload_is_skipped(client, engineer):
    data = _png(_textured(99))
    client.post("/api/batches/upload", headers=engineer, files=[("files", ("d.png", data, "image/png"))],
                data={"product_category": "bottle"})
    r = client.post("/api/batches/upload", headers=engineer, files=[("files", ("d.png", data, "image/png"))],
                    data={"product_category": "bottle"})
    assert r.status_code == 409


def test_supervisor_can_view_but_not_upload(client, supervisor):
    r = client.post("/api/batches/upload", headers=supervisor,
                    files=[("files", ("s.png", _png(_textured(5)), "image/png"))],
                    data={"product_category": "bottle"})
    assert r.status_code == 403
    assert client.get("/api/batches", headers=supervisor).status_code == 200
    assert client.get("/api/dashboard/stats", headers=supervisor).status_code == 200


# --- dataset + camera -------------------------------------------------------

def test_dataset_summary(client, engineer):
    d = client.get("/api/dataset", headers=engineer).json()
    assert d["available"]
    cat = d["categories"][0]
    assert cat["name"] == "metal_disc"
    assert cat["defect_types"] == {"crack": 4, "hole": 4, "scratch": 4}


def test_dataset_import_keeps_labels_and_masks(client, engineer):
    r = client.post("/api/dataset/import", headers=engineer, json={"category": "metal_disc", "limit": 18})
    assert r.status_code == 201, r.text
    detail = client.get(f"/api/batches/{r.json()['id']}", headers=engineer).json()
    labels = {i["ground_truth_label"] for i in detail["images"]}
    assert labels == {"good", "crack", "hole", "scratch"}
    defective = next(i for i in detail["images"] if i["ground_truth_label"] != "good")
    assert client.get(f"/api/images/{defective['id']}/mask", headers=engineer).status_code == 200
    # Importing the same split again finds nothing new.
    again = client.post("/api/dataset/import", headers=engineer, json={"category": "metal_disc", "limit": 18})
    assert again.status_code == 409


def test_thumbnails(client, engineer, supervisor):
    batch = client.get("/api/batches", headers=engineer).json()[0]
    img = client.get(f"/api/batches/{batch['id']}", headers=engineer).json()["images"][0]
    r = client.get(f"/api/images/{img['id']}/thumb", headers=supervisor)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    for kind in ("good", "defect"):
        r = client.get(f"/api/dataset/metal_disc/preview?kind={kind}", headers=supervisor)
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_camera_capture(client, engineer):
    r = client.post("/api/camera/capture", headers=engineer,
                    json={"category": "metal_disc", "count": 8, "defect_ratio": 0.5})
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "camera" and r.json()["image_count"] == 8


def test_unknown_category(client, engineer):
    r = client.post("/api/camera/capture", headers=engineer, json={"category": "nope"})
    assert r.status_code == 404


def test_dashboard_stats(client, engineer):
    s = client.get("/api/dashboard/stats", headers=engineer).json()
    assert s["total_images"] >= 29
    assert s["by_source"]["camera"] == 8
    assert len(s["uploads_per_day"]) == 14
    assert s["recent_batches"]
