"""Priorly on Modal: the ORIGINAL backend (BM25 + BGE + local Qdrant/USearch + BGE reranker + Groq) and the same UI, one URL.

Put this file in the project root (next to the `backend` and `frontend` folders). Nothing in backend/ or frontend/ is changed.

  Smoke test (boots the app once, prints timing/RAM, no public URL):  python -m modal run modal_app.py::smoke
  Deploy:                                                             python -m modal deploy modal_app.py

Needs (created before deploying):
  Volume  "priorly-data"    /real/corpus.jsonl, /real/queries.json, /real/benchmark.json, /qdrant_live/  (uploaded already)
  Secret  "priorly-secrets" GROQ_API_KEY=...   (create it in your own terminal; never paste the key in a chat)

Design notes
  * max_containers=1 on purpose: local Qdrant and the SQLite file must only ever be opened by one process.
  * The Qdrant folder is copied from the Volume to local disk at boot (SQLite does not belong on a network volume).
  * Users/history (SQLite) are restored from the Volume at boot and snapshotted back every 60 s and at shutdown.
  * Models are baked into the image, so a cold start never downloads them.
"""
from __future__ import annotations

import faulthandler
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import modal

APP_NAME = "priorly"
VOLUME_NAME = "priorly-data"
SECRET_NAME = "priorly-secrets"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

HERE = Path(__file__).resolve().parent
VOL = "/vol"
LOCAL_QDRANT = "/tmp/qdrant_live"
LOCAL_DB = "/tmp/priorly.db"
DB_ON_VOLUME = f"{VOL}/db/priorly.db"


def _download_models() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer(EMBED_MODEL)
    CrossEncoder(RERANK_MODEL)


# Never ship: virtualenvs, caches, tests, the local data folder (it comes from the Volume), databases, secrets.
_IGNORE = modal.FilePatternMatcher(
    "**/venv", "**/.venv", "**/__pycache__", "**/*.pyc", "**/.pytest_cache", "**/tests",
    "**/.env", "**/*.db", "backend/data", "backend/ingest", "**/*.coverage-backup",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")  # CPU-only wheel, much smaller
    .pip_install(
        "fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "httpx>=0.27", "numpy>=1.26", "scikit-learn>=1.4",
        "rank-bm25>=0.2.2", "qdrant-client>=1.9", "reportlab>=4.0", "sentence-transformers>=3.0", "usearch>=2.26.6",
    )
    .env({"HF_HOME": "/models", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    .run_function(_download_models)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(HERE / "backend", "/srv/backend", ignore=_IGNORE)
    .add_local_dir(HERE / "frontend", "/srv/frontend")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)


def _mem_breakdown() -> dict:
    """RssAnon = real heap, RssFile = file-backed pages (models/index mmaps; reclaimable). cgroup = what the container is charged."""
    out: dict = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            for key in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                if line.startswith(key + ":"):
                    out[key + "_mb"] = round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    for name, path in (("cgroup_current_mb", "/sys/fs/cgroup/memory.current"), ("cgroup_peak_mb", "/sys/fs/cgroup/memory.peak")):
        try:
            out[name] = round(int(Path(path).read_text().strip()) / 1048576, 1)
        except (OSError, ValueError):
            pass
    return out


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return -1.0


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={VOL: volume},
    cpu=(2.0, 4.0),  # torch otherwise sees the host's ~17 cores and thrashes; OMP_NUM_THREADS above matches the limit
    memory=(3072, 8192),  # billed on what is actually used; the high limit only stops an out-of-memory kill while we measure
    max_containers=1,
    min_containers=0,
    scaledown_window=90,
    startup_timeout=1200,
    timeout=900,
)
@modal.concurrent(max_inputs=8)
class Priorly:
    @modal.enter()
    def boot(self) -> None:
        t0 = time.perf_counter()
        self.timings: dict[str, float] = {}
        # Diagnostics: RSS every 15 s and a stack dump every 120 s, so a slow or stuck start shows where and how big.
        faulthandler.dump_traceback_later(120, repeat=True)
        threading.Thread(target=self._rss_logger, args=(t0,), daemon=True).start()
        volume.reload()

        # 1) Qdrant + USearch folder -> local disk (never opened on the network volume)
        t = time.perf_counter()
        if not Path(f"{VOL}/qdrant_live/collection/priorly_chunks").is_dir():
            raise RuntimeError("Volume is missing /qdrant_live/collection/priorly_chunks; upload it first")
        if Path(LOCAL_QDRANT).exists():
            shutil.rmtree(LOCAL_QDRANT)
        shutil.copytree(f"{VOL}/qdrant_live", LOCAL_QDRANT, ignore=shutil.ignore_patterns(".lock"))
        self.timings["copy_qdrant_s"] = round(time.perf_counter() - t, 1)

        # 2) users / history database
        Path(DB_ON_VOLUME).parent.mkdir(parents=True, exist_ok=True)
        if Path(DB_ON_VOLUME).is_file():
            shutil.copyfile(DB_ON_VOLUME, LOCAL_DB)

        # 3) the same settings your local .env uses (paths differ; no secrets here: GROQ_API_KEY comes from the Modal Secret)
        os.environ.update({
            "DENSE_BACKEND": "bge", "RERANK_BACKEND": "bge", "LLM_BACKEND": "groq",
            "QDRANT_URL": LOCAL_QDRANT, "QDRANT_COLLECTION": "priorly_chunks",
            "CORPUS_PATH": f"{VOL}/real/corpus.jsonl", "BENCHMARK_PATH": f"{VOL}/real/benchmark.json",
            "QUERIES_PATH": f"{VOL}/real/queries.json",
            "DB_PATH": LOCAL_DB, "FRONTEND_DIR": "/srv/frontend",
        })
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing from the Modal Secret")

        # 4) build the real engine (corpus, BM25, Qdrant, BGE models) exactly once
        t = time.perf_counter()
        import sys

        print(f"[boot +{time.perf_counter() - t0:.0f}s] data copied, building engine", flush=True)
        sys.path.insert(0, "/srv/backend")
        from contextlib import asynccontextmanager

        from app.config import Settings
        from app.main import build_state, create_app

        settings = Settings.from_env()
        self.web = create_app(settings)

        @asynccontextmanager
        async def _no_lifespan(_app):  # state is built right here, once, not again on ASGI startup
            yield

        self.web.router.lifespan_context = _no_lifespan
        self.web.state.s = build_state(settings)
        self.timings["build_engine_s"] = round(time.perf_counter() - t, 1)
        faulthandler.cancel_dump_traceback_later()
        self.timings["boot_total_s"] = round(time.perf_counter() - t0, 1)
        self.timings["rss_mb_after_boot"] = _rss_mb()
        print("PRIORLY BOOT", self.timings, flush=True)

        self._stop = threading.Event()
        self._saver = threading.Thread(target=self._save_loop, daemon=True)
        self._saver.start()

    @staticmethod
    def _rss_logger(t0: float) -> None:
        peak = 0.0
        while True:
            peak = max(peak, _rss_mb())
            print(f"[boot +{time.perf_counter() - t0:5.0f}s] rss={_rss_mb()} MB peak_seen={peak} MB", flush=True)
            time.sleep(15)

    def _snapshot_db(self) -> None:
        if not Path(LOCAL_DB).is_file():
            return
        tmp = DB_ON_VOLUME + ".tmp"
        src = sqlite3.connect(LOCAL_DB)
        dst = sqlite3.connect(tmp)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        os.replace(tmp, DB_ON_VOLUME)
        volume.commit()

    def _save_loop(self) -> None:
        last = 0.0
        while not self._stop.wait(60):
            try:
                m = Path(LOCAL_DB).stat().st_mtime if Path(LOCAL_DB).exists() else 0.0
                wal = Path(LOCAL_DB + "-wal")
                m = max(m, wal.stat().st_mtime if wal.exists() else 0.0)
                if m > last:
                    self._snapshot_db()
                    last = m
            except Exception as exc:  # never let a save problem take the app down
                print("db snapshot failed:", type(exc).__name__, flush=True)

    @modal.exit()
    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._snapshot_db()
        except Exception as exc:
            print("final db snapshot failed:", type(exc).__name__, flush=True)

    @modal.asgi_app()
    def serve(self):
        return self.web

    @modal.method()
    def search_test(self) -> dict:
        """One real search through the whole pipeline (no login, nothing is written to the users database)."""
        from app.pipeline import Options

        claim = ("A method of wirelessly charging a battery of an electronic device, comprising: transmitting power from a "
                 "transmitter coil to a receiver coil; detecting a foreign object near the coils; and reducing the transmitted "
                 "power in response to the detection.")
        before = _mem_breakdown()
        t = time.perf_counter()
        payload = self.web.state.s.engine.search(claim, Options(), 10)
        took = round(time.perf_counter() - t, 1)
        results = payload.get("results", [])
        first = results[0] if results else {}
        return {
            "search_seconds": took, "results": len(results), "degraded": payload.get("degraded"),
            "degraded_stages": payload.get("degraded_stages"), "result_keys": sorted(first.keys())[:12],
            "top3": [str(r.get("patent_id") or r.get("id") or r.get("doc_id")) for r in results[:3]],
            "retrieval": payload.get("retrieval"), "timings_ms": payload.get("timings_ms"), "work": payload.get("work"), "mem_before": before, "mem_after": _mem_breakdown(),
            "boot": self.timings,
        }

    @modal.method()
    def selftest(self) -> dict:
        s = self.web.state.s
        return {**self.timings, "rss_mb_now": _rss_mb(), "dense": s.settings.dense_backend,
                "rerank": s.settings.rerank_backend, "llm": s.settings.llm_backend,
                "qdrant_collection": s.settings.qdrant_collection,
                "groq_key_present": bool(os.environ.get("GROQ_API_KEY"))}


@app.local_entrypoint()
def smoke() -> None:
    print("--- one real search through the full pipeline ---")
    t = time.perf_counter()
    res = Priorly().search_test.remote()
    print("wall seconds for this call (cold start + real search):", round(time.perf_counter() - t, 1))
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("--- selftest ---")
    smoke_selftest()


def smoke_selftest() -> None:
    t = time.perf_counter()
    result = Priorly().selftest.remote()
    print("wall seconds incl. cold start:", round(time.perf_counter() - t, 1))
    for k, v in result.items():
        print(f"  {k}: {v}")
