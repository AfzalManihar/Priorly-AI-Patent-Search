import hashlib
import time

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings


INVITE = "local-test-invite-code"
PASSWORD = "correct horse battery staple 42"


def _client(tmp_path):
    settings = make_settings(tmp_path, access_token=INVITE)
    app = create_app(settings)
    return TestClient(app)


def _signup(client, username="afzal"):
    return client.post("/api/auth/signup", json={
        "username": username, "password": PASSWORD,
    })


def test_signup_stores_scrypt_hash_and_creates_hashed_session(tmp_path):
    with _client(tmp_path) as client:
        response = _signup(client)
        assert response.status_code == 201
        cookies = response.headers.get_list("set-cookie")
        session_cookie = next(value for value in cookies if value.startswith("priorly_session="))
        csrf_cookie = next(value for value in cookies if value.startswith("priorly_csrf="))
        assert "httponly" in session_cookie.lower()
        assert "max-age=604800" in session_cookie.lower()
        assert "httponly" not in csrf_cookie.lower()
        user = response.json()["user"]
        assert user["username"] == "afzal"
        assert user["is_admin"] is True

        store = client.app.state.s.store
        row = store.db.execute("SELECT password_salt,password_hash FROM users WHERE id=?", (user["id"],)).fetchone()
        assert row is not None
        assert PASSWORD.encode() not in bytes(row["password_hash"])
        assert len(bytes(row["password_salt"])) == 16
        token = client.cookies.get("priorly_session")
        assert token
        assert store.db.execute("SELECT 1 FROM sessions WHERE token_hash=?", (
            hashlib.sha256(token.encode()).hexdigest(),
        )).fetchone()
        assert store.db.execute("SELECT 1 FROM sessions WHERE token_hash=?", (token,)).fetchone() is None
        assert client.get("/api/auth/me").status_code == 200


def test_signup_is_public_and_does_not_require_access_token_or_invite_code(tmp_path):
    settings = make_settings(tmp_path, access_token="")
    with TestClient(create_app(settings)) as client:
        payload = {"username": "person", "password": PASSWORD}
        response = client.post("/api/auth/signup", json=payload)
        assert response.status_code == 201
        assert response.json()["user"]["username"] == "person"
        assert client.get("/api/auth/me").status_code == 200


def test_login_rejects_wrong_password_and_success_sets_session(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        client.post("/api/auth/logout", headers={"x-csrf-token": client.cookies.get("priorly_csrf", "")})
        bad = client.post("/api/auth/login", json={"username": "afzal", "password": "wrong"})
        assert bad.status_code == 401
        good = client.post("/api/auth/login", json={"username": "AFZAL", "password": PASSWORD})
        assert good.status_code == 200
        assert good.json()["user"]["username"] == "afzal"
        assert client.get("/api/me").status_code == 200


def test_expired_session_is_rejected(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        token = client.cookies.get("priorly_session")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        store = client.app.state.s.store
        store.db.execute("UPDATE sessions SET expires_at=? WHERE token_hash=?", (time.time() - 1, token_hash))
        store.db.commit()
        assert client.get("/api/auth/me").status_code == 401


def test_logout_revokes_session_and_csrf_is_required(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        csrf = client.cookies.get("priorly_csrf")
        blocked = client.post("/api/auth/logout")
        assert blocked.status_code == 403
        logged_out = client.post("/api/auth/logout", headers={"x-csrf-token": csrf})
        assert logged_out.status_code == 200
        assert client.get("/api/auth/me").status_code == 401


def test_login_rate_limit_is_applied_by_ip_and_username(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        client.post("/api/auth/logout", headers={"x-csrf-token": client.cookies.get("priorly_csrf", "")})
        payload = {"username": "afzal", "password": "incorrect password"}
        statuses = [client.post("/api/auth/login", json=payload).status_code for _ in range(6)]
        assert statuses[:5] == [401] * 5
        assert statuses[5] == 429


def test_password_and_session_secrets_do_not_appear_in_logs(tmp_path, caplog):
    with _client(tmp_path) as client:
        _signup(client)
        token = client.cookies.get("priorly_session")
        csrf = client.cookies.get("priorly_csrf")
        client.post("/api/auth/logout", headers={"x-csrf-token": csrf})
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert PASSWORD not in joined
        assert token not in joined
        assert csrf not in joined



def test_admin_password_reset_and_delete_revoke_sessions(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        store = client.app.state.s.store
        user = store.user_by_username("afzal")
        now = time.time()
        store.create_session("session-hash-reset", user["id"], "csrf-hash", now, now + 3600)
        new_salt = b"new-admin-salt!!"
        assert store.set_user_password("afzal", new_salt, hashlib.sha256(b"new password").digest())
        assert store.get_session("session-hash-reset") is None

        store.create_session("session-hash-delete", user["id"], "csrf-hash", now, now + 3600)
        assert store.delete_user("afzal") is True
        assert store.user_by_username("afzal") is None
        assert store.get_session("session-hash-delete") is None


def test_plaintext_password_is_absent_from_entire_sqlite_dump(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        store = client.app.state.s.store
        dump = "\n".join(store.db.iterdump())
        assert PASSWORD not in dump
        row = store.db.execute(
            "SELECT password_salt, password_hash FROM users WHERE username=?", ("afzal",)
        ).fetchone()
        assert row is not None
        assert PASSWORD.encode("utf-8") not in bytes(row["password_salt"])
        assert PASSWORD.encode("utf-8") not in bytes(row["password_hash"])


def test_admin_cli_password_reset_revokes_sessions_and_changes_password(tmp_path, monkeypatch):
    from app.admin import main as admin_main

    settings = make_settings(tmp_path, access_token=INVITE)
    with TestClient(create_app(settings)) as client:
        assert _signup(client).status_code == 201
        new_password = "new secure password 987!"

        monkeypatch.setattr(
            "app.admin.Settings.from_env", classmethod(lambda cls: settings)
        )
        monkeypatch.setattr("app.admin.getpass.getpass", lambda _prompt: new_password)

        assert admin_main(["reset-password", "afzal"]) == 0
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={
            "username": "afzal", "password": PASSWORD,
        }).status_code == 401
        assert client.post("/api/auth/login", json={
            "username": "afzal", "password": new_password,
        }).status_code == 200



def test_valid_session_replaces_legacy_access_code_for_remote_browser(tmp_path):
    settings = make_settings(tmp_path, access_token=INVITE)
    with TestClient(create_app(settings), base_url="https://priorly.example") as client:
        # The access code remains required for anonymous remote API requests.
        denied = client.get("/api/benchmark")
        assert denied.status_code == 401

        created = _signup(client, username="remote_user")
        assert created.status_code == 201
        # A valid browser session is sufficient; the access code is not sent to the browser.
        allowed = client.get("/api/benchmark")
        assert allowed.status_code == 200


def test_profile_preferences_are_saved_for_current_user(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        csrf = client.cookies.get("priorly_csrf")
        saved = client.put("/api/auth/profile", headers={"x-csrf-token": csrf}, json={
            "email": "afzal@example.com", "display_name": "Afzal Manihar",
            "learning_topic": "Patent search", "memory_enabled": False,
        })
        assert saved.status_code == 200
        profile = client.get("/api/auth/profile").json()["profile"]
        assert profile["email"] == "afzal@example.com"
        assert profile["display_name"] == "Afzal Manihar"
        assert profile["learning_topic"] == "Patent search"
        assert profile["memory_enabled"] is False


def test_change_password_revokes_existing_sessions(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        csrf = client.cookies.get("priorly_csrf")
        changed = client.post("/api/auth/change-password", headers={"x-csrf-token": csrf}, json={
            "current_password": PASSWORD, "new_password": "another strong password 123!",
        })
        assert changed.status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={"username": "afzal", "password": "another strong password 123!"}).status_code == 200


def test_logout_everywhere_revokes_all_sessions(tmp_path):
    with _client(tmp_path) as client:
        assert _signup(client).status_code == 201
        user = client.app.state.s.store.user_by_username("afzal")
        now = time.time()
        client.app.state.s.store.create_session("another-device-session", user["id"], "csrf-hash", now, now + 3600)
        csrf = client.cookies.get("priorly_csrf")
        response = client.post("/api/auth/logout-everywhere", headers={"x-csrf-token": csrf})
        assert response.status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        assert client.app.state.s.store.get_session("another-device-session") is None
