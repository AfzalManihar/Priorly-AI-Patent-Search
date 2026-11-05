import json

import httpx
import pytest

from app.fusion import rrf
from app.highlight import highlight
from app.llm import GroqLLM, LLMError, heuristic_decompose
from app.store import Store
from app.tokenizer import tokenize
from eval.metrics import bootstrap_ci, precision_at_k, recall_at_k, reciprocal_rank


# ---------------------------------------------------------------- tokenizer
def test_patent_ids_collapse_to_one_token():
    assert tokenize("see US 9,104,382 B2") == ["see", "us9104382b2", "us9104382"]
    assert tokenize("US9104382B2") == ["us9104382b2", "us9104382"]


def test_patent_id_matches_with_or_without_kind_code():
    assert set(tokenize("US 9,104,382")) <= set(tokenize("cited US 9,104,382 B2 here"))


def test_formulae_and_numbers_survive():
    tokens = tokenize("dissolved in C6H12O6 at 2.5 mol, LiFePO4")
    assert "c6h12o6" in tokens and "2.5" in tokens and "lifepo4" in tokens


def test_stopwords_dropped_and_light_stemming():
    assert tokenize("wherein the controller adjusts") == ["controller", "adjust"]


# --------------------------------------------------------------------- RRF
def test_rrf_matches_formula_and_order():
    fused = dict(rrf([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_is_deterministic_on_ties():
    assert [d for d, _ in rrf([["x", "y"], ["y", "x"]])] == ["x", "y"]


def test_rrf_handles_empty_input():
    assert rrf([]) == [] and rrf([[]]) == []


# ------------------------------------------------------------------ metrics
def test_metrics():
    ranked, relevant = ["a", "b", "c", "d"], {"b", "d", "z"}
    assert recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, relevant, 10) == pytest.approx(0.2)  # divides by k, not by list length
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank(["q"], relevant) == 0.0
    assert recall_at_k(ranked, set()) == 0.0


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    assert lo <= 4 / 6 <= hi
    assert bootstrap_ci([]) == (0.0, 0.0)


# ---------------------------------------------------------------- highlight
def test_highlight_marks_elements_and_escapes_html():
    text = "A <script>x</script> coil embedded in the road surface couples to a pickup under the vehicle."
    out = highlight(text, {1: "coil embedded in a road surface", 2: "pickup mounted under the vehicle"})
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert '<mark data-e="1">' in out and '<mark data-e="2">' in out


def test_highlight_without_match_is_plain_escaped_text():
    assert highlight("a & b", {1: "zebra crossing"}) == "a &amp; b"


# ------------------------------------------------------------ decomposition
def test_heuristic_decompose_splits_on_semicolons():
    parts = heuristic_decompose("A system comprising: a first coil in a road; a second coil on a car; and a controller.")
    assert len(parts) == 3 and parts[0].startswith("A first coil")


def test_heuristic_decompose_falls_back_to_whole_claim():
    claim = "A method of making tea."
    assert heuristic_decompose(claim) == [claim]


# --------------------------------------------------------------------- LLM
def _llm(handler, store=None):
    return GroqLLM("key", "m", "https://example.test/v1", store=store,
                   transport=httpx.MockTransport(handler), sleep=lambda _: None)


def _ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_llm_decompose_parses_json_and_caches():
    calls = []

    def handler(request):
        calls.append(1)
        return _ok(json.dumps({"elements": ["one", "two"]}))

    store = Store(":memory:")
    llm = _llm(handler, store)
    assert llm.decompose("claim text") == ["one", "two"]
    assert llm.decompose("claim text") == ["one", "two"]
    assert len(calls) == 1  # second call came from the cache


def test_llm_rejects_invalid_decomposition():
    with pytest.raises(LLMError):
        _llm(lambda r: _ok("not json")).decompose("claim")
    with pytest.raises(LLMError):
        _llm(lambda r: _ok(json.dumps({"elements": []}))).decompose("claim")


def test_llm_retries_rate_limits_then_gives_up():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(429)

    with pytest.raises(LLMError):
        _llm(handler).hyde("element")
    assert len(attempts) == 3


def test_llm_requires_key():
    with pytest.raises(LLMError):
        GroqLLM("", "m", "https://example.test/v1")
