"""FastAPI application (PRD section 10).

Run:  uvicorn app.main:app --reload     (from the backend/ directory)
The frontend in ../frontend is served at "/", the API lives under "/api".
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import time
import threading
import uuid
from urllib.parse import urlsplit
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .chunkers import CHUNKERS, get_chunker
from .corpus import load_corpus
from .guardrails import BENIGN, REDTEAM, inspect_claim, redact
from .llm import LLMError, make_llm
from .pipeline import Engine, Options
from .report_pdf import build_report_pdf
from .rerank import build_scorer
from .retrievers import BM25Index, DependencyError, build_dense
from .schemas import PlaygroundRequest, ReportRequest, SearchRequest, SettingsModel
from .store import Store

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("priorly.api")

_make_record = logging.getLogRecordFactory()


def _redacting_record(*a, **kw):
    """Every log line passes through redact(), so an API key can never reach the log file."""
    rec = _make_record(*a, **kw)
    # The message template and every string argument are cleaned in place. record.args keeps its shape, because
    # formatters such as uvicorn's AccessFormatter unpack exactly five values from it.
    try:
        if isinstance(rec.msg, str):
            rec.msg = redact(rec.msg)
        if isinstance(rec.args, tuple):
            rec.args = tuple(redact(x) if isinstance(x, str) else x for x in rec.args)
        elif isinstance(rec.args, dict):
            rec.args = {k: redact(v) if isinstance(v, str) else v for k, v in rec.args.items()}
    except Exception:
        pass
    return rec


logging.setLogRecordFactory(_redacting_record)

# Settings that can change while the service runs vs. settings that need a restart.
RUNTIME_SETTINGS = ("rrf_k", "candidates", "rerank_depth")
RESTART_SETTINGS = {"embedding_model": "embed_model", "reranker": "rerank_model", "llm_model": "groq_model"}


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                                "font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data:; "
                                "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"),
}


def _file_stem(title: str) -> str:
    """ASCII-only file name part: HTTP headers cannot carry other characters."""
    return "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in title).strip("-")[:40] or "report"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def error_response(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> int | None:
        """Return seconds to wait if the caller is over the limit, otherwise None."""
        now = time.monotonic()
        window = self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return max(1, int(60 - (now - window[0])) + 1)
        window.append(now)
        return None


class Metrics:
    """In-process counters for the /api/metrics endpoint (PRD NFR-09). Resets when the service restarts."""

    def __init__(self):
        self.started = time.time()
        self.searches = 0
        self.degraded = 0
        self.errors: Counter = Counter()
        self.totals: deque = deque(maxlen=500)
        self.stage_ms: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    def record(self, payload: dict) -> None:
        self.searches += 1
        self.degraded += 1 if payload.get("degraded") else 0
        for name, ms in (payload.get("timings_ms") or {}).items():
            if name == "total":
                self.totals.append(ms)
            else:
                self.stage_ms[name].append(ms)

    @staticmethod
    def _pct(values, q: float):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

    def snapshot(self) -> dict:
        return {
            "uptime_s": int(time.time() - self.started),
            "searches": self.searches,
            "degraded_searches": self.degraded,
            "errors_by_status": {str(k): v for k, v in sorted(self.errors.items())},
            "latency_ms": {"count": len(self.totals), "p50": self._pct(self.totals, 0.5), "p95": self._pct(self.totals, 0.95)},
            "stage_p95_ms": {k: self._pct(v, 0.95) for k, v in sorted(self.stage_ms.items())},
        }


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=12, max_length=1024)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=1024)


class ProfileRequest(BaseModel):
    email: str = Field(default="", max_length=254)
    display_name: str = Field(default="", max_length=80)
    learning_topic: str = Field(default="", max_length=160)
    memory_enabled: bool = True


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


SESSION_COOKIE = "priorly_session"
CSRF_COOKIE = "priorly_csrf"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=64)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthRateLimiter:
    """Per-process login limiter keyed independently by IP and normalized username."""

    def __init__(self, attempts: int = 5, window_s: int = 900):
        self.attempts, self.window_s = attempts, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, keys: tuple[str, ...]) -> int | None:
        now = time.monotonic()
        with self.lock:
            retry = 0
            if len(self.hits) > 5000:
                cutoff = now - self.window_s
                for stale_key in [k for k, q in self.hits.items() if not q or q[-1] <= cutoff]:
                    self.hits.pop(stale_key, None)
            for key in keys:
                q = self.hits[key]
                while q and now - q[0] >= self.window_s:
                    q.popleft()
                if len(q) >= self.attempts:
                    retry = max(retry, int(self.window_s - (now - q[0])) + 1)
            if retry:
                return retry
            for key in keys:
                self.hits[key].append(now)
        return None

    def clear(self, keys: tuple[str, ...]) -> None:
        with self.lock:
            for key in keys:
                self.hits.pop(key, None)


@dataclass
class State:
    settings: Settings
    store: Store
    engine: Engine
    limiter: RateLimiter
    metrics: Metrics
    auth_limiter: AuthRateLimiter


def build_state(settings: Settings) -> State:
    store = Store(settings.db_path)
    saved = store.get_settings()
    for key in RUNTIME_SETTINGS:
        if key in saved:
            setattr(settings, key, int(saved[key]))
    if settings.chunking not in CHUNKERS:
        raise ValueError(f"CHUNKING={settings.chunking!r} is not one of {', '.join(CHUNKERS)}")
    corpus = load_corpus(settings.corpus_path, chunker=get_chunker(settings.chunking))
    if settings.chunking != "claim_aware":  # each chunking keeps its own vector index
        settings.qdrant_collection = f"{settings.qdrant_collection}_{settings.chunking}"
    engine = Engine(
        settings, corpus,
        bm25=BM25Index(corpus),
        dense=build_dense(settings, corpus),
        scorer=build_scorer(settings),
        llm=make_llm(settings, store),
    )
    return State(settings, store, engine, RateLimiter(settings.rate_limit_per_min), Metrics(), AuthRateLimiter())


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = build_state(settings or Settings.from_env())
        yield

    app = FastAPI(title="Priorly API", version="1.0.0", lifespan=lifespan)
    cfg = settings or Settings.from_env()
    if cfg.cors_origins:  # same-origin by default: the API serves the front end itself
        app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def _hardening(request: Request, call_next):
        request_id = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        # Auth and account endpoints are reachable without the legacy API access header; handlers enforce session auth where required.
        is_auth_route = request.url.path in {
            "/api/auth/signup", "/api/auth/login", "/api/auth/logout", "/api/auth/me", "/api/me",
            "/api/auth/profile", "/api/auth/change-password", "/api/auth/logout-everywhere"
        }
        is_localhost = request.url.hostname in {"localhost", "127.0.0.1", "::1"}
        st = getattr(request.app.state, "s", None)
        session_token = request.cookies.get(SESSION_COOKIE, "")
        session = st.store.get_session(_token_hash(session_token)) if st is not None and session_token else None
        access_denied = (
            request.url.path.startswith("/api")
            and request.url.path != "/api/health"
            and not is_auth_route
            and request.method != "OPTIONS"
            and cfg.access_token
            and not is_localhost
            and not session
            and not secrets.compare_digest(request.headers.get("x-priorly-access-token", ""), cfg.access_token)
        )
        if access_denied:
            response = JSONResponse(
                status_code=401,
                content={"error": {"code": "access_denied", "message": "Enter the Priorly access code to continue."}},
            )
        else:
            # If a valid browser session accompanies a mutation, require the double-submit CSRF token
            # and verify it against the hash stored server-side. Legacy non-session API clients remain compatible.
            mutating = request.method in {"POST", "PUT", "PATCH", "DELETE"}
            if mutating and session:
                csrf_cookie = request.cookies.get(CSRF_COOKIE, "")
                csrf_header = request.headers.get("x-csrf-token", "")
                valid_csrf = (
                    bool(csrf_cookie) and bool(csrf_header)
                    and hmac.compare_digest(csrf_cookie, csrf_header)
                    and hmac.compare_digest(_token_hash(csrf_cookie), session["csrf_hash"])
                )
                if not valid_csrf:
                    response = JSONResponse(
                        status_code=403,
                        content={"error": {"code": "csrf_failed", "message": "Refresh the page and retry the request."}},
                    )
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api"):
            st = getattr(request.app.state, "s", None)
            if st is not None and response.status_code >= 400:
                st.metrics.errors[response.status_code] += 1
            log.info("request id=%s method=%s path=%s status=%s ms=%.0f", request_id, request.method,
                     request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
        return response

    # ------------------------------------------------------------ error handling
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = {str(part) for err in exc.errors() for part in err.get("loc", ())}
        if fields & {"options", "k"}:
            return error_response(422, "invalid_options", "One of the search options is not valid. Reset the options and retry.")
        return error_response(422, "invalid_request", "The request is missing a required field or has a wrong type.")

    @app.exception_handler(DependencyError)
    async def _dependency(_: Request, exc: DependencyError):
        log.error("dependency unavailable: %s", exc)
        return error_response(503, "dependency_unavailable",
                              "A search service is unavailable. Try again shortly. No partial results are shown.")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "Something went wrong on our side. Try again.")

    def state(request: Request) -> State:
        return request.app.state.s

    def current_session(request: Request) -> tuple[State, dict] | None:
        st = state(request)
        token = request.cookies.get(SESSION_COOKIE, "")
        if not token:
            return None
        session = st.store.get_session(_token_hash(token))
        return (st, session) if session else None

    def owner_id(request: Request) -> str | None:
        current = current_session(request)
        return current[1]["user_id"] if current else None

    def set_session_cookies(response: Response, request: Request, session_token: str, csrf_token: str) -> None:
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
        common = {"max_age": SESSION_TTL_SECONDS, "path": "/", "secure": secure, "samesite": "lax"}
        response.set_cookie(SESSION_COOKIE, session_token, httponly=True, **common)
        response.set_cookie(CSRF_COOKIE, csrf_token, httponly=False, **common)

    def check_same_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin:
            return
        parsed = urlsplit(origin)
        expected_host = request.headers.get("host", request.url.netloc)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not hmac.compare_digest(
            parsed.netloc.lower(), expected_host.lower()
        ):
            raise ApiError(403, "origin_denied", "This request origin is not allowed.")

    @app.post("/api/auth/signup", status_code=201)
    def signup(body: SignupRequest, request: Request):
        check_same_origin(request)
        st = state(request)
        if not USERNAME_RE.fullmatch(body.username):
            raise ApiError(422, "invalid_username", "Use 3–32 letters, numbers, dots, underscores or hyphens.")
        username = body.username.lower()
        salt = secrets.token_bytes(16)
        created = st.store.create_user(username, salt, _password_hash(body.password, salt))
        if not created:
            raise ApiError(409, "username_taken", "That username is already registered.")
        session_token, csrf_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = time.time()
        st.store.create_session(_token_hash(session_token), created["id"], _token_hash(csrf_token),
                                now, now + SESSION_TTL_SECONDS)
        response = JSONResponse({"user": created}, status_code=201)
        set_session_cookies(response, request, session_token, csrf_token)
        return response

    @app.post("/api/auth/login")
    def login(body: LoginRequest, request: Request):
        check_same_origin(request)
        st = state(request)
        username = body.username.lower()
        ip = request.client.host if request.client else "unknown"
        keys = (f"ip:{ip}", f"user:{username}")
        wait = st.auth_limiter.check(keys)
        if wait is not None:
            raise ApiError(429, "login_rate_limited", "Too many login attempts. Try again later.",
                           {"Retry-After": str(wait)})
        user = st.store.user_by_username(username)
        # Always perform one scrypt operation to reduce username-enumeration timing differences.
        salt = user["password_salt"] if user else b"priorly-auth-dummy-salt"
        try:
            candidate = _password_hash(body.password, salt)
            valid = bool(user) and hmac.compare_digest(candidate, user["password_hash"])
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ApiError(401, "invalid_credentials", "Username or password is incorrect.")
        st.auth_limiter.clear(keys)
        session_token, csrf_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = time.time()
        st.store.create_session(_token_hash(session_token), user["id"], _token_hash(csrf_token),
                                now, now + SESSION_TTL_SECONDS)
        public_user = {"id": user["id"], "username": user["username"],
                       "created_at": user["created_at"], "is_admin": bool(user["is_admin"])}
        response = JSONResponse({"user": public_user})
        set_session_cookies(response, request, session_token, csrf_token)
        return response

    @app.post("/api/auth/logout")
    def logout(request: Request):
        st = state(request)
        token = request.cookies.get(SESSION_COOKIE, "")
        if token:
            st.store.delete_session(_token_hash(token))
        response = JSONResponse({"ok": True})
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax", secure=secure)
        response.delete_cookie(CSRF_COOKIE, path="/", httponly=False, samesite="lax", secure=secure)
        return response

    @app.get("/api/auth/me")
    @app.get("/api/me")
    def me(request: Request):
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        _, session = current
        return {"user": {"id": session["user_id"], "username": session["username"],
                         "created_at": session["user_created_at"], "is_admin": bool(session["is_admin"])}}

    @app.get("/api/auth/profile")
    def get_profile(request: Request):
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        st, session = current
        return {"profile": st.store.profile_by_id(session["user_id"])}

    @app.put("/api/auth/profile")
    def save_profile(body: ProfileRequest, request: Request):
        check_same_origin(request)
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        st, session = current
        email = body.email.strip()
        if email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise ApiError(422, "invalid_email", "Enter a valid email address.")
        profile = st.store.update_profile(session["user_id"], email, body.display_name.strip(),
                                          body.learning_topic.strip(), body.memory_enabled)
        return {"profile": profile}

    @app.post("/api/auth/change-password")
    def change_password(body: PasswordChangeRequest, request: Request):
        check_same_origin(request)
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        st, session = current
        full_user = st.store.user_by_username(session["username"])
        candidate = _password_hash(body.current_password, full_user["password_salt"])
        if not hmac.compare_digest(candidate, full_user["password_hash"]):
            raise ApiError(400, "current_password_incorrect", "Current password is incorrect.")
        salt = secrets.token_bytes(16)
        st.store.update_password(session["user_id"], salt, _password_hash(body.new_password, salt))
        response = JSONResponse({"ok": True, "message": "Password changed. Please sign in again."})
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax", secure=secure)
        response.delete_cookie(CSRF_COOKIE, path="/", httponly=False, samesite="lax", secure=secure)
        return response

    @app.post("/api/auth/logout-everywhere")
    def logout_everywhere(request: Request):
        check_same_origin(request)
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        st, session = current
        st.store.delete_user_sessions(session["user_id"])
        response = JSONResponse({"ok": True})
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax", secure=secure)
        response.delete_cookie(CSRF_COOKIE, path="/", httponly=False, samesite="lax", secure=secure)
        return response

    def guard_search(request: Request, body: SearchRequest) -> State:
        st = state(request)
        wait = st.limiter.check(request.client.host if request.client else "local")
        if wait is not None:
            raise ApiError(429, "rate_limited", f"Too many searches. Retry in {wait} seconds.",
                           {"Retry-After": str(wait)})
        n = len(body.claim.strip())
        if n < st.settings.claim_min_chars:
            raise ApiError(400, "claim_too_short",
                           f"Enter the full claim. It needs at least {st.settings.claim_min_chars} characters.")
        if n > st.settings.claim_max_chars:
            raise ApiError(400, "claim_too_long",
                           f"That claim is too long. Keep it under {st.settings.claim_max_chars:,} characters.")
        body.claim = screen(st, body.claim, "search")
        return st

    def screen(st: State, claim: str, source: str) -> str:
        """Input guardrail. Blocks injection attempts (HTTP 422), cleans hidden characters and links, logs every non-clean verdict."""
        verdict = inspect_claim(claim)
        if verdict.action != "allow":
            st.store.log_guardrail("input", verdict.action, [r["rule"] for r in verdict.reasons], redact(verdict.cleaned), source)
        if verdict.blocked:
            raise ApiError(422, "guardrail_blocked", verdict.message)
        return verdict.cleaned

    def record(st: State, payload: dict) -> None:
        st.metrics.record(payload)
        log.info("search id=%s results=%d degraded=%s timings_ms=%s llm_cache=%s", payload["search_id"],
                 len(payload["results"]), ",".join(payload.get("degraded_stages") or []) or ("yes" if payload.get("degraded") else "no"),
                 json.dumps(payload.get("timings_ms")), json.dumps(st.store.cache_stats()))

    # ------------------------------------------------------------------- search
    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        st = guard_search(request, body)
        user_id = owner_id(request)
        payload = st.engine.search(body.claim.strip(), Options(**body.options.model_dump()), body.k)
        st.store.save_search(payload["search_id"], body.model_dump(), payload, user_id=user_id)
        record(st, payload)
        return payload

    @app.post("/api/search/stream")
    def search_stream(body: SearchRequest, request: Request):
        st = guard_search(request, body)
        user_id = owner_id(request)
        events = st.engine.run_events(body.claim.strip(), Options(**body.options.model_dump()), body.k)

        def sse():
            try:
                for event in events:
                    if event["type"] == "result":
                        st.store.save_search(event["data"]["search_id"], body.model_dump(), event["data"], user_id=user_id)
                        record(st, event["data"])
                    yield f"data: {json.dumps(event)}\n\n"
            except DependencyError:
                yield "data: " + json.dumps({"type": "error", "error": {
                    "code": "dependency_unavailable",
                    "message": "A search service is unavailable. Try again shortly. No partial results are shown."}}) + "\n\n"
            except Exception:
                log.exception("stream failed")
                yield "data: " + json.dumps({"type": "error", "error": {
                    "code": "internal_error", "message": "Something went wrong on our side. Try again."}}) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/search/latest")
    def latest_search(request: Request):
        current = current_session(request)
        if not current:
            raise ApiError(401, "not_authenticated", "Log in to view your saved search.")
        found = current[0].store.latest_search(current[1]["user_id"])
        if not found:
            raise ApiError(404, "not_found", "No saved searches yet.")
        return found

    @app.get("/api/search/{search_id}")
    def get_search(search_id: str, request: Request):
        found = state(request).store.get_search(search_id, owner_id(request))
        if not found:
            raise ApiError(404, "not_found", "That search does not exist.")
        return found

    @app.delete("/api/search/{search_id}")
    def delete_search(search_id: str, request: Request):
        removed = state(request).store.delete_search(search_id, owner_id(request))
        if removed is None:
            raise ApiError(404, "not_found", "That search does not exist.")
        return {"deleted": removed}

    @app.delete("/api/data")
    def delete_all_data(request: Request, confirm: str = Query(default="")):
        """Erase every stored claim (PRD NFR-08). Requires ?confirm=yes so it cannot happen by accident."""
        if confirm != "yes":
            raise ApiError(400, "confirm_required", "Add confirm=yes to erase all stored searches and reports.")
        return {"deleted": state(request).store.delete_all(owner_id(request))}

    # ---------------------------------------------------------------- benchmark
    @app.get("/api/benchmark")
    def benchmark(request: Request):
        path = state(request).settings.benchmark_path
        if not path.exists():
            raise ApiError(404, "benchmark_missing", "No benchmark results yet. Run `python -m eval.run_eval` first.")
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ reports
    @app.get("/api/reports")
    def list_reports(request: Request):
        return {"reports": state(request).store.list_reports(owner_id(request))}

    @app.post("/api/reports", status_code=201)
    def save_report(body: ReportRequest, request: Request):
        st = state(request)
        user_id = owner_id(request)
        found = st.store.get_search(body.search_id, user_id)
        if not found:
            raise ApiError(404, "not_found", "That search does not exist, so it cannot be saved.")
        results = found["response"]["results"]
        best = max((r["score"] for r in results), default=0.0)
        return st.store.save_report("r_" + uuid.uuid4().hex[:10], body.title, body.search_id,
                                    body.shortlist, len(results), best, user_id=user_id)

    @app.get("/api/reports/{report_id}/export.csv")
    def export_report(report_id: str, request: Request):
        st = state(request)
        report = st.store.get_report(report_id, owner_id(request))
        found = st.store.get_search(report["search_id"], owner_id(request)) if report else None
        if not report or not found:
            raise ApiError(404, "not_found", "That report does not exist.")
        def safe(value):
            """Stop spreadsheet apps from running a cell that starts with = + - @ as a formula."""
            text = str(value)
            return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["rank", "patent", "title", "assignee", "year", "cpc", "score", "elements",
                         "location", "shortlisted", "reason"])
        for rank, r in enumerate(found["response"]["results"], start=1):
            writer.writerow([rank] + [safe(v) for v in (
                r["id"], r["title"], r["assignee"], r["year"], r["cpc"], r["score"],
                " ".join(f"E{n}" for n in r["el"]), r["loc"],
                "yes" if r["id"] in report["shortlist"] else "no", r["why"])])
        name = _file_stem(report["title"])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="priorly-{name}.csv"'})

    @app.get("/api/reports/{report_id}/export.pdf")
    def export_report_pdf(report_id: str, request: Request):
        st = state(request)
        report = st.store.get_report(report_id, owner_id(request))
        found = st.store.get_search(report["search_id"], owner_id(request)) if report else None
        if not report or not found:
            raise ApiError(404, "not_found", "That report does not exist.")
        name = _file_stem(report["title"])
        return Response(build_report_pdf(report, found), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="priorly-{name}.pdf"'})

    # ----------------------------------------------------------------- settings
    def current_settings(st: State) -> dict:
        saved, cfg = st.store.get_settings(), st.settings
        return {
            "embedding_model": saved.get("embedding_model", cfg.embed_model),
            "reranker": saved.get("reranker", cfg.rerank_model),
            "llm_model": saved.get("llm_model", cfg.groq_model),
            "rrf_k": cfg.rrf_k, "candidates": cfg.candidates, "rerank_depth": cfg.rerank_depth,
            "backends": {"dense": cfg.dense_backend, "rerank": cfg.rerank_backend, "llm": cfg.llm_backend},
            "requires_restart": ["embedding_model", "reranker", "llm_model"],
        }

    @app.get("/api/settings")
    def get_settings(request: Request):
        return current_settings(state(request))

    @app.put("/api/settings")
    def put_settings(body: SettingsModel, request: Request):
        st = state(request)
        changes = body.model_dump(exclude_none=True)
        for key in RUNTIME_SETTINGS:
            if key in changes:
                setattr(st.settings, key, changes[key])
        st.store.put_settings(changes)
        return current_settings(st)

    # -------------------------------------------------------------- experiments
    def results_dir(st: State):
        return st.settings.corpus_path.parent / "results"

    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    @app.get("/api/experiments")
    def experiments(request: Request):
        data = read_json(results_dir(state(request)) / "grid.json")
        return data if data else {"available": False, "message": "No comparison run yet. Run `python -m eval.grid` first."}

    @app.get("/api/default-config")
    def default_config(request: Request):
        st = state(request)
        best = read_json(results_dir(st) / "best_config.json")
        live = {"chunking": st.settings.chunking, "dense_backend": st.engine.dense.name, "rerank_backend": st.engine.scorer.name,
                "llm_backend": st.engine.llm.name}
        if not best:
            return {"available": False, "live": live, "chunkers": {k: v[0] for k, v in CHUNKERS.items()}}
        opts = best["options"]
        return {"available": True, "config": best, "live": live, "chunkers": {k: v[0] for k, v in CHUNKERS.items()},
                "chunking_matches_live": opts["chunking"] == st.settings.chunking,
                "pipeline_options": {k: opts[k] for k in ("decompose", "hyde", "hybrid", "dense", "prf", "rerank")}}

    @app.get("/api/queries")
    def example_queries(request: Request, limit: int = Query(default=60, ge=1, le=200)):
        st = state(request)
        raw = read_json(st.settings.queries_path) if st.settings.queries_path else None
        if not raw:
            return {"queries": []}
        grid = read_json(results_dir(st) / "grid.json")
        used = {r["id"] for r in grid["combos"][0]["per_query"]} if grid and grid["combos"] else set()
        rows = [q for q in raw["queries"] if not used or q["id"] in used]
        return {"queries": [{"id": q["id"], "type": q["type"], "year": q["year"], "patent_id": q["patent_id"], "claim": q["claim"],
                             "relevant": len(q["relevant"])} for q in rows[:limit]]}

    @app.post("/api/playground")
    def playground(body: PlaygroundRequest, request: Request):
        """Run one claim through up to four pipeline configurations and score each against the examiner citations."""
        from dataclasses import replace

        from eval.metrics import average_precision, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
        from .retrievers import Constraints

        st = state(request)
        wait = st.limiter.check(request.client.host if request.client else "local")
        if wait is not None:
            raise ApiError(429, "rate_limited", f"Too many searches. Retry in {wait} seconds.", {"Retry-After": str(wait)})
        claim, cons, relevant, meta = (body.claim or "").strip(), None, None, None
        if body.query_id:
            raw = read_json(st.settings.queries_path) or {"queries": []}
            q = next((x for x in raw["queries"] if x["id"] == body.query_id), None)
            if not q:
                raise ApiError(404, "not_found", "That benchmark query does not exist.")
            claim = q["claim"]
            cons = Constraints(exclude_ids=frozenset([q["patent_id"]]), max_year=q["year"] - 1)
            relevant = {p for p in q["relevant"] if p in st.engine.corpus.patents and st.engine.corpus.patents[p].year <= cons.max_year}
            meta = {"id": q["id"], "type": q["type"], "patent_id": q["patent_id"], "relevant": sorted(relevant)}
        if len(claim) < st.settings.claim_min_chars or len(claim) > st.settings.claim_max_chars:
            raise ApiError(400, "claim_length", "Provide a claim between the minimum and maximum length.")
        if not body.query_id:
            claim = screen(st, claim, "playground")
        runs = []
        for cfg in body.configs:
            t0 = time.perf_counter()
            payload = st.engine.search(claim, replace(Options(**cfg.options.model_dump()), explain=False), 10, cons)
            st.store.save_search(payload["search_id"], {"claim": claim, "options": cfg.options.model_dump(), "k": 10, "source": "playground"}, payload, user_id=owner_id(request))
            st.metrics.record(payload)
            ranked = [r["id"] for r in payload["results"]]
            row = {"name": cfg.name or "config", "options": cfg.options.model_dump(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                   "timings_ms": payload.get("timings_ms", {}), "work": payload.get("work", {}), "degraded": payload.get("degraded_stages", []),
                   "results": [{"id": r["id"], "title": r["title"], "year": r["year"], "rank": i, "relevant": bool(relevant and r["id"] in relevant),
                                "dense": r.get("dense"), "bm25": r.get("bm25")} for i, r in enumerate(payload["results"], 1)]}
            if relevant:
                row["metrics"] = {"recall@10": recall_at_k(ranked, relevant, 10), "precision@10": precision_at_k(ranked, relevant, 10),
                                  "mrr": reciprocal_rank(ranked, relevant), "ndcg@10": ndcg_at_k(ranked, relevant, 10),
                                  "map": average_precision(ranked, relevant, 10), "hit@10": hit_rate_at_k(ranked, relevant, 10)}
            runs.append(row)
        return {"claim": claim, "query": meta, "runs": runs}

    @app.get("/api/traces")
    def traces(request: Request, limit: int = Query(default=30, ge=1, le=200)):
        return {"traces": state(request).store.recent_searches(limit, owner_id(request))}

    @app.get("/api/experiments/report.pdf")
    def experiments_pdf(request: Request):
        from .report_research import build_research_pdf

        data = read_json(results_dir(state(request)) / "grid.json")
        if not data:
            raise ApiError(404, "no_experiments", "No comparison run yet. Run `python -m eval.grid` first.")
        return Response(build_research_pdf(data), media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="priorly-research-report.pdf"'})

    # ------------------------------------------------------------------- health
    # ------------------------------------------------------------ guardrails and token usage
    @app.post("/api/guardrails/check")
    def guardrails_check(body: dict, request: Request):
        """Run the input guardrail on any text without starting a search. Used by the Safety page tester."""
        st = state(request)
        text = str(body.get("text", ""))[:20000]
        verdict = inspect_claim(text)
        st.store.log_guardrail("input", verdict.action, [r["rule"] for r in verdict.reasons], redact(verdict.cleaned), "tester")
        return verdict.to_dict()

    @app.get("/api/guardrails/events")
    def guardrails_events(request: Request, limit: int = Query(default=30, ge=1, le=200)):
        return state(request).store.guardrail_summary(limit)

    @app.get("/api/guardrails/selftest")
    def guardrails_selftest(request: Request):
        """Measured, not claimed: the red-team list must be caught and the real benchmark claims must pass."""
        st = state(request)
        attacks = [{"name": n, "text": t, "action": inspect_claim(t).action} for n, t in REDTEAM]
        raw = read_json(st.settings.queries_path) if st.settings.queries_path else None
        real = [q["claim"] for q in (raw or {}).get("queries", [])][:200] or [(p.abstract or (p.claims[0] if p.claims else ""))[:600] for p in list(st.engine.corpus.patents.values())[:200]]
        legit = [t for t in BENIGN + real]
        wrong = [t[:80] for t in legit if inspect_claim(t).action != "allow"]
        caught = sum(a["action"] != "allow" for a in attacks)
        return {"attacks": [{"name": a["name"], "text": a["text"], "action": a["action"]} for a in attacks], "attacks_total": len(attacks),
                "attacks_caught": caught, "detection_rate": round(caught / len(attacks), 4),
                "legit_total": len(legit), "false_positives": len(wrong), "false_positive_examples": wrong[:5],
                "legit_source": "benchmark claims" if raw else "corpus passages"}

    @app.get("/api/usage")
    def usage(request: Request):
        st = state(request)
        limit = st.settings.daily_token_limit
        used = st.store.tokens_since(86400)
        out = st.store.usage_summary()
        out.update({"backend": st.settings.llm_backend, "model": st.settings.groq_model if st.settings.llm_backend == "groq" else None,
                    "used_24h": used, "limit_24h": limit, "remaining_24h": max(0, limit - used) if limit else None,
                    "tokens_total": st.store.tokens_total()})
        return out

    @app.get("/api/health")
    def health(request: Request):
        st = state(request)
        eng = st.engine
        return {
            "status": "ok",
            "corpus": {"patents": len(eng.corpus), "chunks": len(eng.corpus.chunks)},
            "dense_backend": getattr(eng.dense, "label", eng.dense.name),
            "rerank_backend": eng.scorer.name,
            "llm_backend": eng.llm.name,
            "benchmark_available": st.settings.benchmark_path.exists(),
            "chunking": st.settings.chunking,
            "budgets_s": {"stage": st.settings.stage_budget_s, "rerank": st.settings.rerank_budget_s,
                          "explain": st.settings.explain_budget_s, "coverage_scope": st.settings.coverage_scope},
            "rerank": {"depth": st.settings.rerank_depth, "chunks_per_patent": st.settings.rerank_chunks,
                       "batch": st.settings.rerank_batch, "element_pairs": st.settings.rerank_elements},
        }

    @app.get("/api/metrics")
    def metrics(request: Request):
        st = state(request)
        return {**st.metrics.snapshot(), "llm_cache": st.store.cache_stats(), "stored": st.store.counts()}

    if cfg.frontend_dir:
        app.mount("/", StaticFiles(directory=str(cfg.frontend_dir), html=True), name="frontend")
    return app


app = create_app()
