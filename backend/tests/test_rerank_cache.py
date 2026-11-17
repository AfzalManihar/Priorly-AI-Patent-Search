"""The reranker's score memory: it must never change a score, it must be switchable, and the search must say what
it really computed. Uses a stand-in model, so nothing is downloaded and the real model is not exercised."""
from app.config import Settings
from app.rerank import CrossEncoderScorer


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=0):
        self.calls.append(list(pairs))
        return [float(len(q) - len(p)) / 10.0 for q, p in pairs]  # deterministic, depends only on the pair


def make_scorer(cache: bool) -> CrossEncoderScorer:
    scorer = CrossEncoderScorer.__new__(CrossEncoderScorer)  # skip loading the real model
    scorer.model = FakeModel()
    scorer.cache_enabled = cache
    scorer._cache, scorer._scored, scorer._reused = {}, 0, 0
    return scorer


PAIRS = [("coil", "a primary coil"), ("controller", "adjusts frequency"), ("coil", "a primary coil")]


def test_cache_on_scores_a_pair_once_and_counts_reuse():
    scorer = make_scorer(cache=True)
    first = scorer.score_pairs(PAIRS)
    second = scorer.score_pairs(PAIRS)
    assert second == first
    assert sum(len(c) for c in scorer.model.calls) == 2  # two distinct pairs, scored once in total
    assert scorer.stats() == {"scored": 2, "cached": 4}


def test_cache_off_scores_every_search_again_with_identical_scores():
    on, off = make_scorer(cache=True), make_scorer(cache=False)
    assert off.score_pairs(PAIRS) == on.score_pairs(PAIRS)  # the memory never changes a score
    off.score_pairs(PAIRS)
    off.score_pairs(PAIRS)
    assert len(off.model.calls) == 3 and off.stats()["scored"] == 6
    assert off.stats()["cached"] == 3 and off._cache == {}  # only the repeat inside one call is shared


def test_rerank_cache_setting_defaults_on_and_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("RERANK_CACHE", raising=False)
    assert Settings.from_env().rerank_cache is True
    for value in ("off", "OFF", "0", "false", "no"):
        monkeypatch.setenv("RERANK_CACHE", value)
        assert Settings.from_env().rerank_cache is False
    monkeypatch.setenv("RERANK_CACHE", "on")
    assert Settings.from_env().rerank_cache is True


def test_search_reports_the_work_it_did(client):
    claim = "A wireless charging system comprising a primary coil, a pickup coil and a controller that adjusts frequency."
    body = client.post("/api/search", json={"claim": claim, "k": 5}).json()
    work = body["work"]
    assert set(work) == {"rerank_pairs_scored", "rerank_pairs_from_cache", "llm_calls", "llm_from_cache", "llm_tokens", "rerank_cache"}
    assert work["rerank_pairs_scored"] > 0 and all(isinstance(v, (int, bool)) and v >= 0 for v in work.values())
