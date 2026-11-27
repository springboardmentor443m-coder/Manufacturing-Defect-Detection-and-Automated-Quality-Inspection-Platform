import os
import sys
import tempfile
from pathlib import Path

import pytest

# Point the app at a throwaway DB, storage folder and dataset before it is imported.
_tmp = Path(tempfile.mkdtemp(prefix="visioninspect-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
os.environ["STORAGE_DIR"] = str(_tmp / "storage")
os.environ["MVTEC_ROOT"] = str(_tmp / "mvtec")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from make_sample_dataset import make_sample  # noqa: E402

make_sample(_tmp / "mvtec")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def login(client, email, password) -> dict:
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    s = get_settings()
    return login(client, s.first_admin_email, s.first_admin_password)


@pytest.fixture(scope="session")
def engineer(client, admin):
    client.post("/api/users", headers=admin, json={
        "email": "eng@test.io", "full_name": "Quality Engineer", "role": "quality_engineer", "password": "engineer123",
    })
    return login(client, "eng@test.io", "engineer123")


@pytest.fixture(scope="session")
def supervisor(client, admin):
    client.post("/api/users", headers=admin, json={
        "email": "sup@test.io", "full_name": "Factory Supervisor", "role": "supervisor", "password": "supervisor123",
    })
    return login(client, "sup@test.io", "supervisor123")
