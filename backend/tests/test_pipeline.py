import pytest

from app.llm import HeuristicLLM, LLMError
from app.main import build_state
from app.pipeline import Options
from app.retrievers import Constraints

FULL = Options()
BASELINE = Options(decompose=False, hyde=False, hybrid=False, rerank=False)


@pytest.fixture()
def engine(settings):
    return build_state(settings).engine


def test_full_run_shape(engine, claim):
    out = engine.search(claim, FULL, 10)
    assert out["results"] and len(out["results"]) <= 10
    assert len(out["elements"]) >= 2 and not out["degraded"]
    first = out["results"][0]
    for field in ("id", "title", "assignee", "year", "cpc", "el", "dense", "bm25", "rrf_rank", "rerank_rank",
                  "score", "score_kind", "loc", "passage", "why"):
        assert field in first
    assert set(out["timings_ms"]) >= {"decompose", "retrieve", "fuse", "rerank", "explain", "total"}


def test_results_are_ordered_by_score(engine, claim):
    scores = [r["score"] for r in engine.search(claim, FULL, 10)["results"]]
    assert scores == sorted(scores, reverse=True)


def test_baseline_has_whole_claim_and_no_coverage(engine, claim):
    out = engine.search(claim, BASELINE, 10)
    assert out["elements"][0].get("whole_claim") and all(r["el"] == [] for r in out["results"])
    assert out["results"][0]["score_kind"] == "rrf"


def test_each_switch_is_honoured(engine, claim):
    skipped = {e["stage"] for e in engine.run_events(claim, BASELINE, 5) if e["type"] == "stage" and e["status"] == "skipped"}
    assert {"decompose", "hyde", "rerank"} <= skipped


def test_k_limits_results(engine, claim):
    assert len(engine.search(claim, FULL, 5)["results"]) == 5


def test_leakage_controls(engine, queries):
    q = queries[0]
    cons = Constraints(exclude_ids=frozenset([q["patent_id"]]), max_year=q["year"] - 1)
    out = engine.search(q["claim"], FULL, 10, cons)
    assert q["patent_id"] not in {r["id"] for r in out["results"]}
    assert all(r["year"] <= q["year"] - 1 for r in out["results"])


def test_keyword_only_flag_is_consistent_with_ranks(engine, queries):
    """`missed_by_vector` must mean: keyword rank within 10, vector rank beyond 10 or absent."""
    for q in queries:
        for r in engine.search(q["claim"], FULL, 10)["results"]:
            expected = (r["dense"] is None or r["dense"] > 10) and (r["bm25"] is not None and r["bm25"] <= 10)
            assert r["missed_by_vector"] == expected


class _BrokenLLM(HeuristicLLM):
    name = "broken"

    def decompose(self, claim):
        raise LLMError("down")

    def hyde(self, element):
        raise LLMError("down")


def test_llm_failure_degrades_instead_of_failing(engine, claim):
    engine.llm = _BrokenLLM()
    out = engine.search(claim, FULL, 10)
    assert out["degraded"] and "decompose" in out["degraded_stages"]
    assert out["results"]  # still answers, and says so


def test_reranker_failure_falls_back_to_fusion_order(engine, claim):
    class Boom:
        name, coverage_threshold = "boom", 0.5

        def score_pairs(self, pairs):
            raise RuntimeError("model crashed")

    engine.scorer = Boom()
    out = engine.search(claim, Options(decompose=False, hyde=False), 10)
    assert "rerank" in out["degraded_stages"] and out["results"][0]["score_kind"] == "rrf"


def test_exact_identifier_is_found_by_keyword_search():
    """The keyword index finds a passage by a patent number or formula even when the rest is generic."""
    from app.corpus import Corpus, Patent
    from app.retrievers import BM25Index

    def patent(pid, text):
        return Patent(pid, pid, "Co", 2010, "X", "", [], [{"no": "0001", "text": text}])

    corpus = Corpus([
        patent("US 1,000,001", "A battery with an electrode and a casing for storing energy."),
        patent("US 1,000,002", "As described in US 5,555,123 B2 the dopant LiFePO4 improves stability."),
        patent("US 1,000,003", "A casing for an electrode in a battery pack."),
    ])
    index = BM25Index(corpus)
    for query in ("see US 5,555,123", "LiFePO4 dopant"):
        hits = index.search(query, 3, Constraints())
        assert corpus.chunks[hits[0][0]].patent_id == "US 1,000,002"
