"""Runtime configuration. Everything comes from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def load_dotenv(path: Path) -> None:
    """Tiny .env reader. Real environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # Data locations
    corpus_path: Path
    benchmark_path: Path
    db_path: Path
    frontend_dir: Path | None

    # Backends: tfidf | bge   (dense), lexical | bge (rerank), heuristic | groq (llm)
    dense_backend: str = "tfidf"
    rerank_backend: str = "lexical"
    llm_backend: str = "heuristic"

    # Models and services
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "priorly_chunks"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Pipeline parameters (runtime editable through PUT /api/settings)
    rrf_k: int = 60
    candidates: int = 50
    rerank_depth: int = 50
    trace_depth: int = 200
    chunk_pool: int = 5
    chunking: str = "claim_aware"  # CHUNKING: one of app.chunkers.CHUNKERS; changing it needs a re-index
    queries_path: Path | None = None
    stage_budget_s: float = 20.0  # STAGE_BUDGET_S: longest the LLM steps (decompose, HyDE) may take before falling back; 0 = no limit
    rerank_budget_s: float = 45.0  # RERANK_BUDGET_S: reranks as many top candidates as fit; the rest keep fusion order; 0 = no limit
    explain_budget_s: float = 60.0  # EXPLAIN_BUDGET_S: explanations not written in time use the template text; 0 = no limit
    coverage_scope: str = "pool"  # COVERAGE_SCOPE: pool = check claim elements against the passages retrieval surfaced; all = every passage
    rerank_cache: bool = True  # RERANK_CACHE=off makes the reranker score every pair again on every search

    # Service limits
    rate_limit_per_min: int = 60
    claim_min_chars: int = 40
    claim_max_chars: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    access_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT_DIR / ".env")
        load_dotenv(BACKEND_DIR / ".env")
        sample = BACKEND_DIR / "data" / "sample"
        frontend = Path(_env("FRONTEND_DIR", str(ROOT_DIR / "frontend")))
        return cls(
            corpus_path=Path(_env("CORPUS_PATH", str(sample / "corpus.jsonl"))),
            benchmark_path=Path(_env("BENCHMARK_PATH", str(sample / "benchmark.json"))),
            db_path=Path(_env("DB_PATH", str(BACKEND_DIR / "data" / "priorly.db"))),
            frontend_dir=frontend if frontend.is_dir() else None,
            dense_backend=_env("DENSE_BACKEND", "tfidf"),
            rerank_backend=_env("RERANK_BACKEND", "lexical"),
            llm_backend=_env("LLM_BACKEND", "heuristic"),
            embed_model=_env("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            rerank_model=_env("RERANK_MODEL", "BAAI/bge-reranker-base"),
            qdrant_url=_env("QDRANT_URL", ":memory:"),
            qdrant_collection=_env("QDRANT_COLLECTION", "priorly_chunks"),
            groq_api_key=_env("GROQ_API_KEY", ""),
            groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            rrf_k=int(_env("RRF_K", "60")),
            candidates=int(_env("CANDIDATES", "50")),
            rerank_depth=int(_env("RERANK_DEPTH", "50")),
            chunking=_env("CHUNKING", "claim_aware"),
            queries_path=Path(_env("QUERIES_PATH", str(Path(_env("CORPUS_PATH", str(sample / "corpus.jsonl"))).parent / "queries.json"))),
            stage_budget_s=float(_env("STAGE_BUDGET_S", "20")),
            rerank_budget_s=float(_env("RERANK_BUDGET_S", "45")),
            explain_budget_s=float(_env("EXPLAIN_BUDGET_S", "60")),
            coverage_scope=_env("COVERAGE_SCOPE", "pool").strip().lower(),
            rerank_cache=_env("RERANK_CACHE", "on").strip().lower() not in ("off", "0", "false", "no"),
            rate_limit_per_min=int(_env("RATE_LIMIT_PER_MIN", "60")),
            cors_origins=[o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()],
            access_token=_env("PRIORLY_ACCESS_TOKEN", ""),
        )
