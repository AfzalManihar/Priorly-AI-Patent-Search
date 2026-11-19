import json
import math
import sys
from pathlib import Path

import pytest

from app.chunkers import CHUNKERS
from app.corpus import Corpus, Patent, load_corpus
from eval import grid
from eval.metrics import average_precision, hit_rate_at_k, ndcg_at_k, paired_bootstrap_ci, percentile

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
PATENT = Patent("US 1", "Coil", "Acme", 2010, "H02J 50/10", "A coil. It charges a car. It is cheap.",
                ["A system comprising: a primary coil; a controller; and a battery. The system is wireless."], [])


@pytest.mark.parametrize("name", list(CHUNKERS))
def test_every_chunker_keeps_all_words_and_the_patent_id(name):
    chunks = CHUNKERS[name][1](PATENT, 5)
    assert chunks and all(c.patent_id == "US 1" for c in chunks)
    assert [c.idx for c in chunks] == list(range(5, 5 + len(chunks)))
    joined = " ".join(c.text for c in chunks)
    for word in ("coil", "controller", "battery", "wireless"):
        assert word in joined


def test_chunkers_differ_in_chunk_count():
    counts = {n: len(Corpus([PATENT], chunker=f[1]).chunks) for n, f in CHUNKERS.items()}
    assert counts["whole_document"] == 1 and counts["sentence"] > counts["whole_document"]


def test_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], {"b", "d"}
    assert hit_rate_at_k(ranked, rel, 1) == 0.0 and hit_rate_at_k(ranked, rel, 2) == 1.0
    assert math.isclose(average_precision(ranked, rel), (1 / 2 + 2 / 4) / 2)
    ideal = 1 + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(ranked, rel, 4), (1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert ndcg_at_k(["b", "d"], rel, 10) == 1.0
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    lo, hi = paired_bootstrap_ci([1, 1, 1, 1], [0, 0, 0, 0])
    assert lo > 0 and hi > 0


def test_grid_runs_on_sample_data_and_picks_a_best(monkeypatch, tmp_path):
    for k, v in {"DENSE_BACKEND": "tfidf", "RERANK_BACKEND": "lexical", "LLM_BACKEND": "heuristic", "CORPUS_PATH": str(SAMPLE / "corpus.jsonl"),
                 "DB_PATH": str(tmp_path / "t.db")}.items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "grid.json"
    monkeypatch.setattr(sys, "argv", ["grid", "--queries", str(SAMPLE / "queries.json"), "--split", "all", "--n", "8", "--pool", "0", "--out", str(out)])
    grid.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    ids = [c["id"] for c in report["combos"]]
    assert len(ids) == len(set(ids)) >= 25 and report["meta"]["complete"] is True
    assert {s["reason"][:5] for s in report["skipped"]} == {"HyDE "}  # HyDE combinations are skipped without Groq, never faked
    for combo in report["combos"]:
        assert set(combo["metrics"]) >= set(grid.METRICS)
        assert len(combo["per_query"]) == report["meta"]["queries"]
        lo, hi = combo["metrics"]["recall@10"]["ci"]
        assert lo <= combo["metrics"]["recall@10"]["mean"] <= hi
    assert report["best"]["combo_id"] in ids
    assert (out.parent / "best_config.json").is_file()
    assert any("vs_baseline" in c for c in report["combos"])


def test_regression_check_flags_a_drop():
    done = {"x": {"id": "x", "metrics": {"recall@10": {"mean": 0.20}, "ndcg@10": {"mean": 0.2}, "mrr": {"mean": 0.2}}}}
    base = {"combos": [{"id": "x", "metrics": {"recall@10": {"mean": 0.30}, "ndcg@10": {"mean": 0.2}, "mrr": {"mean": 0.2}}}]}
    found = grid.regressions(done, base, 0.02)
    assert [f["metric"] for f in found] == ["recall@10"]
    assert grid.regressions(done, base, 0.2) == []


def test_engine_keyword_only_and_prf_options(client, claim):
    base = {"decompose": False, "hyde": False, "hybrid": True, "rerank": False}
    plain = client.post("/api/search", json={"claim": claim, "options": {**base, "dense": False}}).json()
    assert plain["options"]["dense"] is False and plain["results"]
    assert plain["retrieval"]["dense_backend"] != "off"
    with_prf = client.post("/api/search", json={"claim": claim, "options": {**base, "dense": False, "prf": True}}).json()
    assert with_prf["options"]["prf"] is True and with_prf["results"]
    default = client.post("/api/search", json={"claim": claim}).json()  # old clients that send no new fields keep working
    assert default["options"]["dense"] is True and default["options"]["prf"] is False


def test_experiment_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    out = tmp_path / "data"
    (out / "results").mkdir(parents=True)
    for name in ("corpus.jsonl", "queries.json", "benchmark.json"):
        (out / name).write_bytes((SAMPLE / name).read_bytes())
    for k, v in {"DENSE_BACKEND": "tfidf", "RERANK_BACKEND": "lexical", "LLM_BACKEND": "heuristic", "CORPUS_PATH": str(out / "corpus.jsonl"),
                 "DB_PATH": str(tmp_path / "g.db")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["grid", "--queries", str(out / "queries.json"), "--split", "all", "--n", "6", "--pool", "0"])
    grid.main()
    settings = make_settings(tmp_path, corpus_path=out / "corpus.jsonl", queries_path=out / "queries.json")
    with TestClient(create_app(settings)) as c:
        from tests.conftest import login
        login(c, "tester")
        assert c.get("/api/experiments").json()["meta"]["complete"] is True
        d = c.get("/api/default-config").json()
        assert d["available"] and set(d["pipeline_options"]) == {"decompose", "hyde", "hybrid", "dense", "prf", "rerank"}
        qs = c.get("/api/queries").json()["queries"]
        assert qs
        play = c.post("/api/playground", json={"query_id": qs[0]["id"], "configs": [
            {"name": "bm25", "options": {"decompose": False, "hyde": False, "hybrid": True, "dense": False, "rerank": False}},
            {"name": "hybrid", "options": {"decompose": False, "hyde": False, "hybrid": True, "rerank": False}}]}).json()
        assert len(play["runs"]) == 2 and "ndcg@10" in play["runs"][0]["metrics"]
        assert c.get("/api/traces").json()["traces"], "playground runs show up in the trace list"
        assert c.get("/api/experiments/report.pdf").content[:4] == b"%PDF"
        assert c.post("/api/playground", json={"query_id": "nope", "configs": [{"options": {}}]}).status_code == 404


def test_stream_reports_real_progress_for_rerank_and_explain(settings, claim):
    from app.main import build_state
    from app.pipeline import Options

    engine = build_state(settings).engine
    events = list(engine.run_events(claim, Options(decompose=True, hyde=False, hybrid=True, rerank=True, explain=True), 5))
    progress = [e for e in events if e["type"] == "progress"]
    for stage in ("rerank", "explain"):
        mine = [e for e in progress if e["stage"] == stage]
        assert mine, f"{stage} reported no progress"
        assert [e["done"] for e in mine] == sorted(e["done"] for e in mine)
        assert mine[-1]["done"] == mine[-1]["total"] > 0
    names = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "running"]
    assert names.index("rerank") < names.index("explain")
    assert events[-1]["type"] == "result"
