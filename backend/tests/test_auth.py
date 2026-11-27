def test_login_rejects_bad_password(client):
    r = client.post("/api/auth/login", data={"username": "admin@visioninspect.ai", "password": "nope"})
    assert r.status_code == 401


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_role(client, engineer):
    r = client.get("/api/auth/me", headers=engineer)
    assert r.status_code == 200
    assert r.json()["role"] == "quality_engineer"


def test_only_admin_manages_users(client, admin, engineer, supervisor):
    assert client.get("/api/users", headers=admin).status_code == 200
    assert client.get("/api/users", headers=engineer).status_code == 403
    assert client.get("/api/users", headers=supervisor).status_code == 403


def test_duplicate_email_conflict(client, admin, engineer):
    r = client.post("/api/users", headers=admin, json={
        "email": "ENG@test.io", "full_name": "Dup", "role": "supervisor", "password": "whatever123",
    })
    assert r.status_code == 409


def test_disabled_user_cannot_log_in(client, admin):
    r = client.post("/api/users", headers=admin, json={
        "email": "temp@test.io", "full_name": "Temp", "role": "supervisor", "password": "temporary1",
    })
    uid = r.json()["id"]
    client.patch(f"/api/users/{uid}", headers=admin, json={"is_active": False})
    r = client.post("/api/auth/login", data={"username": "temp@test.io", "password": "temporary1"})
    assert r.status_code == 403


def test_admin_cannot_demote_self(client, admin):
    me = client.get("/api/auth/me", headers=admin).json()
    r = client.patch(f"/api/users/{me['id']}", headers=admin, json={"role": "supervisor"})
    assert r.status_code == 400
