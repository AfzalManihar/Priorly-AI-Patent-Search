import json
import sys
from pathlib import Path

from eval import run_eval

SAMPLE_QUERIES = Path(__file__).resolve().parents[1] / "data" / "sample" / "queries.json"
SAMPLE_CORPUS = SAMPLE_QUERIES.parent / "corpus.jsonl"


def _run(monkeypatch, tmp_path, *flags):
    out = tmp_path / "bench.json"

    for name, value in {
        "DENSE_BACKEND": "tfidf",
        "RERANK_BACKEND": "lexical",
        "LLM_BACKEND": "heuristic",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setenv("CORPUS_PATH", str(SAMPLE_CORPUS))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("BENCHMARK_PATH", str(out))

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_eval", "--queries", str(SAMPLE_QUERIES), "--out", str(out), *flags],
    )

    run_eval.main()
    return json.loads(out.read_text(encoding="utf-8"))


def test_skip_rerank_runs_only_dense_and_hybrid(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, "--skip-rerank")

    assert [s["key"] for s in report["stages"]] == ["dense", "hybrid"]
    assert report["ablation_extra"] == []
    assert report["rerank_skipped"] is True
    assert report["llm_layers_skipped"] is True
    assert report["by_type"] and all("full" in row for row in report["by_type"])
    assert "failure_cases" in report


def test_skip_llm_still_runs_rerank(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, "--skip-llm")

    assert [s["key"] for s in report["stages"]] == ["dense", "hybrid", "rerank"]
    assert report["rerank_skipped"] is False
    assert report["llm_layers_skipped"] is True


def test_default_runs_all_four_stages(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, "--skip-extra")

    assert [s["key"] for s in report["stages"]] == ["dense", "hybrid", "rerank", "full"]
    assert report["rerank_skipped"] is False
    assert report["llm_layers_skipped"] is False