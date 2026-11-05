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
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Pipeline parameters (runtime editable through PUT /api/settings)
    rrf_k: int = 60
    candidates: int = 50
    rerank_depth: int = 50
    trace_depth: int = 200
    chunk_pool: int = 5

    # Service limits
    rate_limit_per_min: int = 60
    claim_min_chars: int = 40
    claim_max_chars: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT_DIR / ".env")
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
            groq_model=_env("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            rrf_k=int(_env("RRF_K", "60")),
            candidates=int(_env("CANDIDATES", "50")),
            rerank_depth=int(_env("RERANK_DEPTH", "50")),
            rate_limit_per_min=int(_env("RATE_LIMIT_PER_MIN", "60")),
            cors_origins=[o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()],
        )
