"""Pair scorers used for reranking and for per-element coverage.

A scorer maps (query, passage) pairs to relevance scores in [0, 1].

  lexical : token containment with a bigram bonus. Offline, no model download.
  bge     : BGE cross-encoder (sentence-transformers). Use for real evaluation.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

from .retrievers import DependencyError
from .tokenizer import tokenize


@lru_cache(maxsize=32768)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(tokenize(text))


def lexical_score(query: str, passage: str) -> float:
    q, p = _tokens(query), _tokens(passage)
    if not q or not p:
        return 0.0
    q_uni, p_uni = set(q), set(p)
    unigram = len(q_uni & p_uni) / len(q_uni)
    q_bi, p_bi = set(zip(q, q[1:])), set(zip(p, p[1:]))
    bigram = (len(q_bi & p_bi) / len(q_bi)) if q_bi else unigram
    return 0.7 * unigram + 0.3 * bigram


class LexicalScorer:
    name = "lexical"
    coverage_threshold = 0.45  # calibrated on the sample data; paraphrase scores low with a lexical scorer

    def __init__(self):
        self._scored = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self._scored += len(pairs)
        return [lexical_score(q, p) for q, p in pairs]

    batch_size = 16

    def stats(self) -> dict:
        return {"scored": self._scored, "cached": 0}


class CrossEncoderScorer:
    name = "bge"
    coverage_threshold = 0.5
    batch_size = 16
    _model_s = 0.0
    _batches = 0
    _raw_logits = False  # True once the model's own sigmoid is switched off, so predict() returns logits

    def __init__(self, model_name: str, cache: bool = True, batch_size: int = 16):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise DependencyError("Install requirements-ml.txt to use the bge reranker") from exc
        self.model = CrossEncoder(model_name)
        self._raw_logits = _use_raw_logits(self.model)
        # Identical (query, passage) pairs get the identical score, so remembering them never changes a ranking.
        # RERANK_CACHE=off turns the memory off, so every search runs the model on every pair again.
        self.cache_enabled = cache
        self.batch_size = max(1, batch_size)
        self._cache: dict[tuple[str, str], float] = {}
        self._scored = 0
        self._reused = 0
        self._model_s = 0.0  # seconds spent inside the model, so a slow search can be told apart from a slow pair count
        self._batches = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        known = self._cache if self.cache_enabled else {}
        missing = list(dict.fromkeys(p for p in pairs if p not in known))
        fresh: dict[tuple[str, str], float] = {}
        if missing:
            # Similar lengths share a batch, so short pairs are not padded to the longest one. Scores do not depend on order.
            missing.sort(key=lambda p: len(p[0]) + len(p[1]))
            t0 = time.perf_counter()
            logits = self.model.predict(missing, batch_size=self.batch_size)
            self._model_s += time.perf_counter() - t0
            self._batches += -(-len(missing) // self.batch_size)
            for pair, x in zip(missing, logits):
                fresh[pair] = self._to_probability(float(x))
            if self.cache_enabled:
                self._cache.update(fresh)
        self._scored += len(missing)
        self._reused += len(pairs) - len(missing)
        return [fresh[p] if p in fresh else known[p] for p in pairs]

    def _to_probability(self, x: float) -> float:
        """One sigmoid, applied exactly once.

        sentence-transformers' CrossEncoder already applies a sigmoid to single-label models. Applying a second one
        squeezed every score into [0.5, 0.731], so strong matches all showed 0.73 and the ranking lost its signal.
        """
        if self._raw_logits:
            return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
        return min(1.0, max(0.0, x))  # the library already returned a probability; do not squash it again

    def stats(self) -> dict:
        return {"scored": self._scored, "cached": self._reused, "activation": "raw_logits" if self._raw_logits else "model_default",
                "model_s": round(getattr(self, "_model_s", 0.0), 3), "batches": getattr(self, "_batches", 0)}


def _use_raw_logits(model) -> bool:
    """Turn the library's built-in sigmoid off so Priorly can apply its own, once. False if this version has no switch."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    for attr in ("activation_fn", "default_activation_function"):  # sentence-transformers 4.x / 3.x
        if hasattr(model, attr):
            setattr(model, attr, torch.nn.Identity())
            return True
    return False


def build_scorer(settings):
    if settings.rerank_backend == "lexical":
        return LexicalScorer()
    if settings.rerank_backend == "bge":
        return CrossEncoderScorer(settings.rerank_model, cache=getattr(settings, "rerank_cache", True),
                                  batch_size=getattr(settings, "rerank_batch", 16))
    raise ValueError(f"Unknown RERANK_BACKEND: {settings.rerank_backend}")
