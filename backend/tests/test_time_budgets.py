import time

from app.main import build_state
from app.pipeline import Options
from tests.conftest import make_settings

FULL = Options(decompose=True, hyde=False, hybrid=True, rerank=True, explain=True)


def _engine(tmp_path, **overrides):
    return build_state(make_settings(tmp_path, **overrides)).engine


def test_rerank_stops_at_its_budget_and_keeps_fusion_order_for_the_rest(tmp_path, claim):
    full = _engine(tmp_path, rerank_budget_s=0).search(claim, FULL, 20)
    slow = _engine(tmp_path / "b", rerank_budget_s=0.15)
    real = slow.scorer.score_pairs

    def slower(pairs):
        time.sleep(0.05)
        return real(pairs)

    slow.scorer.score_pairs = slower
    out = slow.search(claim, FULL, 20)
    assert "rerank" in out["degraded_stages"]
    reranked = [r for r in out["results"] if r["score_kind"] == "rerank"]
    rest = [r for r in out["results"] if r["score_kind"] != "rerank"]
    assert 0 < len(reranked) < len(out["results"])
    by_id = {r["id"]: r["score"] for r in full["results"]}
    for r in reranked:  # a candidate the budget reached gets exactly the score an unlimited run gives it
        if r["id"] in by_id:
            assert r["score"] == by_id[r["id"]]
    assert [r["rrf_rank"] for r in rest] == sorted(r["rrf_rank"] for r in rest)


def test_explain_budget_falls_back_to_template_text(tmp_path, claim):
    engine = _engine(tmp_path, explain_budget_s=0.4)
    engine.llm.name = "groq"
    engine.llm.explain = lambda *a, **k: (time.sleep(5), "written")[1]
    started = time.perf_counter()
    out = engine.search(claim, FULL, 10)
    assert time.perf_counter() - started < 3
    assert "explain" in out["degraded_stages"]
    assert out["results"] and all(r["why"] for r in out["results"])


def test_llm_step_budget_falls_back_to_rules(tmp_path, claim):
    engine = _engine(tmp_path, stage_budget_s=0.2)
    engine.llm.decompose = lambda text: (time.sleep(5), ["x"])[1]
    started = time.perf_counter()
    out = engine.search(claim, Options(decompose=True, hyde=False, hybrid=True, rerank=False, explain=False), 5)
    assert time.perf_counter() - started < 3
    assert "decompose" in out["degraded_stages"] and len(out["elements"]) >= 2


def test_rerank_never_overruns_its_budget_by_more_than_one_batch(tmp_path, claim):
    engine = _engine(tmp_path, rerank_budget_s=0.5)
    real = engine.scorer.score_pairs
    engine.scorer.score_pairs = lambda pairs: (time.sleep(0.3), real(pairs))[1]  # each batch of pairs takes 0.3 s
    events = list(engine.run_events(claim, FULL, 10))
    stage = next(e for e in events if e["type"] == "stage" and e["stage"] == "rerank" and e["status"] != "running")
    assert stage["ms"] < 500 + 1000, stage
    assert stage["status"] == "degraded" and "reranked" in stage["detail"]


def test_health_reports_the_active_budgets(client):
    assert client.get("/api/health").json()["budgets_s"]["rerank"] == 45.0
