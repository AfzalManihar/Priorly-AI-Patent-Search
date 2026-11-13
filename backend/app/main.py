"""FastAPI application (PRD section 10).

Run:  uvicorn app.main:app --reload     (from the backend/ directory)
The frontend in ../frontend is served at "/", the API lives under "/api".
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import secrets
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .corpus import load_corpus
from .llm import LLMError, make_llm
from .pipeline import Engine, Options
from .report_pdf import build_report_pdf
from .rerank import build_scorer
from .retrievers import BM25Index, DependencyError, build_dense
from .schemas import ReportRequest, SearchRequest, SettingsModel
from .store import Store

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("priorly.api")

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


@dataclass
class State:
    settings: Settings
    store: Store
    engine: Engine
    limiter: RateLimiter
    metrics: Metrics


def build_state(settings: Settings) -> State:
    store = Store(settings.db_path)
    saved = store.get_settings()
    for key in RUNTIME_SETTINGS:
        if key in saved:
            setattr(settings, key, int(saved[key]))
    corpus = load_corpus(settings.corpus_path)
    engine = Engine(
        settings, corpus,
        bm25=BM25Index(corpus),
        dense=build_dense(settings, corpus),
        scorer=build_scorer(settings),
        llm=make_llm(settings, store),
    )
    return State(settings, store, engine, RateLimiter(settings.rate_limit_per_min), Metrics())


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
        if (request.url.path.startswith("/api")
                and request.method != "OPTIONS"
                and cfg.access_token
                and not secrets.compare_digest(
                    request.headers.get("x-priorly-access-token", ""),
                    cfg.access_token,
                )):
            response = JSONResponse(
                status_code=401,
                content={"error": {"code": "access_denied", "message": "Enter the Priorly access code to continue."}},
            )
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
        return st

    def record(st: State, payload: dict) -> None:
        st.metrics.record(payload)
        log.info("search id=%s results=%d degraded=%s timings_ms=%s llm_cache=%s", payload["search_id"],
                 len(payload["results"]), ",".join(payload.get("degraded") or []) or "no",
                 json.dumps(payload.get("timings_ms")), json.dumps(st.store.cache_stats()))

    # ------------------------------------------------------------------- search
    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        st = guard_search(request, body)
        payload = st.engine.search(body.claim.strip(), Options(**body.options.model_dump()), body.k)
        st.store.save_search(payload["search_id"], body.model_dump(), payload)
        record(st, payload)
        return payload

    @app.post("/api/search/stream")
    def search_stream(body: SearchRequest, request: Request):
        st = guard_search(request, body)
        events = st.engine.run_events(body.claim.strip(), Options(**body.options.model_dump()), body.k)

        def sse():
            try:
                for event in events:
                    if event["type"] == "result":
                        st.store.save_search(event["data"]["search_id"], body.model_dump(), event["data"])
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

    @app.get("/api/search/{search_id}")
    def get_search(search_id: str, request: Request):
        found = state(request).store.get_search(search_id)
        if not found:
            raise ApiError(404, "not_found", "That search does not exist.")
        return found

    @app.delete("/api/search/{search_id}")
    def delete_search(search_id: str, request: Request):
        removed = state(request).store.delete_search(search_id)
        if removed is None:
            raise ApiError(404, "not_found", "That search does not exist.")
        return {"deleted": removed}

    @app.delete("/api/data")
    def delete_all_data(request: Request, confirm: str = Query(default="")):
        """Erase every stored claim (PRD NFR-08). Requires ?confirm=yes so it cannot happen by accident."""
        if confirm != "yes":
            raise ApiError(400, "confirm_required", "Add confirm=yes to erase all stored searches and reports.")
        return {"deleted": state(request).store.delete_all()}

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
        return {"reports": state(request).store.list_reports()}

    @app.post("/api/reports", status_code=201)
    def save_report(body: ReportRequest, request: Request):
        st = state(request)
        found = st.store.get_search(body.search_id)
        if not found:
            raise ApiError(404, "not_found", "That search does not exist, so it cannot be saved.")
        results = found["response"]["results"]
        best = max((r["score"] for r in results), default=0.0)
        return st.store.save_report("r_" + uuid.uuid4().hex[:10], body.title, body.search_id,
                                    body.shortlist, len(results), best)

    @app.get("/api/reports/{report_id}/export.csv")
    def export_report(report_id: str, request: Request):
        st = state(request)
        report = st.store.get_report(report_id)
        found = st.store.get_search(report["search_id"]) if report else None
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
        report = st.store.get_report(report_id)
        found = st.store.get_search(report["search_id"]) if report else None
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

    # ------------------------------------------------------------------- health
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
        }

    @app.get("/api/metrics")
    def metrics(request: Request):
        st = state(request)
        return {**st.metrics.snapshot(), "llm_cache": st.store.cache_stats(), "stored": st.store.counts()}

    if cfg.frontend_dir:
        app.mount("/", StaticFiles(directory=str(cfg.frontend_dir), html=True), name="frontend")
    return app


app = create_app()
