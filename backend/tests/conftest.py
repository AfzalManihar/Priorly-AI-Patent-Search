import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

SAMPLE = BACKEND / "data" / "sample"


def make_settings(tmp_path, **overrides) -> Settings:
    s = Settings.from_env()
    s.dense_backend = "tfidf"
    s.rerank_backend = "lexical"
    s.llm_backend = "heuristic"
    s.access_token = ""
    s.qdrant_url = ":memory:"
    s.db_path = tmp_path / "test.db"
    s.corpus_path = SAMPLE / "corpus.jsonl"
    s.benchmark_path = SAMPLE / "benchmark.json"
    s.rate_limit_per_min = 1000
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@pytest.fixture()
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="session")
def queries():
    return json.loads((SAMPLE / "queries.json").read_text(encoding="utf-8"))["queries"]


@pytest.fixture()
def claim(queries):
    return queries[0]["claim"]
