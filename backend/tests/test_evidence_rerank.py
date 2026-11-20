from dataclasses import replace
from types import SimpleNamespace

from app.config import Settings
from app.evidence import PARTIAL, SUPPORTED, UNSUPPORTED, classify, term_recall
from app.pipeline import Engine, Options
from app.rerank import CrossEncoderScorer, LexicalScorer

S = Settings.from_env()


def test_term_recall_is_zero_for_a_different_topic():
    el = "a transmitter coil embedded in a road surface"
    assert term_recall(el, "The battery pack uses lithium cells and a cooling plate.") == 0.0
    assert term_recall(el, "A transmitter coil is embedded in the road surface of the lane.") >= 0.8


def test_classify_needs_both_signals():
    assert classify(0.9, 0.9, S) == SUPPORTED
    assert classify(0.9, 0.1, S) == UNSUPPORTED  # model likes it, but the words are not there (topical only)
    assert classify(0.1, 0.9, S) == UNSUPPORTED  # words match, model says different meaning
    assert classify(0.65, 0.4, S) == PARTIAL
    assert classify(0.3, 0.4, S) == UNSUPPORTED  # below the model noise floor
    assert classify(None, 1.0, S) == UNSUPPORTED


class _Counting:
    name = "bge"
    batch_size = 16

    def __init__(self, value=0.9):
        self.pairs, self.value = [], value

    def score_pairs(self, pairs):
        self.pairs += pairs
        return [self.value] * len(pairs)


def _engine(passages, scorer, **cfg):
    chunks = [SimpleNamespace(text=t, patent_id="P1", loc=f"[{i}]") for i, t in enumerate(passages)]
    corpus = SimpleNamespace(chunks=chunks, by_patent={"P1": list(range(len(chunks)))}, patents={})
    return SimpleNamespace(corpus=corpus, scorer=scorer, settings=replace(S, **cfg))


def test_coverage_distinguishes_supported_partial_unsupported():
    texts = ["transmitter coil embedded in road surface", "frequency tuning controller", "lithium battery cooling plate"]
    passage = "A transmitter coil is embedded in the road surface. A controller tunes the resonant frequency."
    eng = _engine([passage], _Counting(0.9))
    supported, shown, partial = Engine._coverage(eng, texts, "P1")
    assert shown == 0 and supported == [1, 2] and 3 not in supported and 3 not in partial


def test_unrelated_elements_cost_no_model_call():
    scorer = _Counting()
    eng = _engine(["The shoe has a rubber sole and a lace."], scorer)
    supported, _, partial = Engine._coverage(eng, ["transmitter coil embedded in road", "lithium battery cooling"], "P1")
    assert supported == [] and partial == [] and scorer.pairs == []


def test_rerank_scores_few_pairs_per_patent_not_one_per_element():
    scorer = _Counting()
    pool = {f"P{i}": [i * 10 + j for j in range(5)] for i in range(4)}
    chunks = [SimpleNamespace(text=f"passage {i}") for i in range(60)]
    eng = SimpleNamespace(corpus=SimpleNamespace(chunks=chunks), scorer=scorer, settings=S)
    eng._score_by_deadline = lambda pairs, deadline, step=None: Engine._score_by_deadline(eng, pairs, deadline, step)
    texts = [f"element {i}" for i in range(6)]
    rel, best, n = Engine._rerank(eng, "claim", texts, Options(decompose=True), list(pool), pool)
    assert n == 4 and len(rel) == 4
    assert len(scorer.pairs) == 4 * S.rerank_chunks  # not 4 x 5 chunks x 7


def test_rerank_elements_switch_restores_the_old_work():
    scorer = _Counting()
    pool = {"P0": [0, 1, 2, 3, 4]}
    eng = SimpleNamespace(corpus=SimpleNamespace(chunks=[SimpleNamespace(text=f"p{i}") for i in range(5)]),
                          scorer=scorer, settings=replace(S, rerank_elements=True))
    eng._score_by_deadline = lambda pairs, deadline, step=None: Engine._score_by_deadline(eng, pairs, deadline, step)
    Engine._rerank(eng, "claim", ["a", "b"], Options(decompose=True), ["P0"], pool)
    assert len(scorer.pairs) == 5 * 3


def test_cross_encoder_batches_by_length_and_keeps_scores_per_pair(monkeypatch):
    seen = {}

    class Model:
        def predict(self, pairs, batch_size):
            seen["batch"], seen["order"] = batch_size, list(pairs)
            return [float(len(q) + len(p)) / 1000 for q, p in pairs]

    sc = object.__new__(CrossEncoderScorer)
    sc.model, sc.cache_enabled, sc.batch_size, sc._cache, sc._scored, sc._reused = Model(), False, 16, {}, 0, 0
    pairs = [("q" * 50, "p" * 400), ("q", "p"), ("q" * 5, "p" * 20)]
    out = sc.score_pairs(pairs)
    assert seen["batch"] == 16 and seen["order"][0] == ("q", "p")  # shortest first
    assert out[0] > out[2] > out[1]  # sigmoid of 0.45, 0.025, 0.002: scores come back in the caller's order


def test_defaults_match_the_dev_calibration(monkeypatch):
    """Thresholds are now on a single sigmoid. The DEV controls, converted from the old double sigmoid
    (noise 0.50-0.56 -> p 0.00-0.23, real matches 0.68-0.73 -> p 0.75-1.0), keep the same separation at 0.5."""
    for name in ("COVERAGE_SUPPORT_CE", "COVERAGE_PARTIAL_CE", "COVERAGE_SUPPORT_RECALL", "COVERAGE_PARTIAL_RECALL"):
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env()
    assert (s.support_ce, s.partial_ce, s.support_recall, s.partial_recall) == (0.5, 0.5, 0.6, 0.3)


def test_model_noise_floor_never_counts_even_when_every_word_matches():
    s = Settings.from_env()
    for noise in (0.0, 0.064, 0.233):  # median / p75 of the same-topic controls, as single-sigmoid probabilities
        assert classify(noise, 1.0, s) == UNSUPPORTED
    assert classify(0.50, 1.0, s) == SUPPORTED
    assert classify(0.986, 1.0, s) == SUPPORTED  # median of the positive control


def test_supported_versus_partial_comes_from_term_recall():
    s = Settings.from_env()
    assert classify(0.95, 0.9, s) == SUPPORTED
    assert classify(0.95, 0.4, s) == PARTIAL
    assert classify(0.95, 0.2, s) == UNSUPPORTED


def test_sigmoid_is_applied_once_so_strong_matches_stay_distinguishable(monkeypatch):
    import math
    import sys
    import types

    class Identity:  # stands in for torch.nn.Identity
        pass

    fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Identity=Identity))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model:
        activation_fn = "sigmoid"  # what the library sets for a single-label cross-encoder

        def predict(self, pairs, batch_size):
            assert isinstance(self.activation_fn, Identity)  # the built-in sigmoid is off, so these are logits
            return [{"a": 6.0, "b": 2.0, "c": -4.0}[q] for q, _ in pairs]

    from app.rerank import _use_raw_logits
    sc = object.__new__(CrossEncoderScorer)
    sc.model, sc.cache_enabled, sc.batch_size, sc._cache, sc._scored, sc._reused = Model(), False, 4, {}, 0, 0
    sc._raw_logits = _use_raw_logits(sc.model)
    out = sc.score_pairs([("a", "x"), ("b", "x"), ("c", "x")])
    assert sc._raw_logits and out == [1 / (1 + math.exp(-v)) for v in (6.0, 2.0, -4.0)]
    assert out[0] > 0.99 and out[0] - out[1] > 0.1  # the old double sigmoid gave both ~0.73
    assert sc.stats()["activation"] == "raw_logits"


def test_without_a_switch_probabilities_pass_through_unchanged():
    sc = object.__new__(CrossEncoderScorer)  # _raw_logits stays False: the library already applied its sigmoid

    class Model:
        def predict(self, pairs, batch_size):
            return [0.98 for _ in pairs]

    sc.model, sc.cache_enabled, sc.batch_size, sc._cache, sc._scored, sc._reused = Model(), False, 4, {}, 0, 0
    assert sc.score_pairs([("q", "p")]) == [0.98]
