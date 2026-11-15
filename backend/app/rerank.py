"""Pair scorers used for reranking and for per-element coverage.

A scorer maps (query, passage) pairs to relevance scores in [0, 1].

  lexical : token containment with a bigram bonus. Offline, no model download.
  bge     : BGE cross-encoder (sentence-transformers). Use for real evaluation.
"""
from __future__ import annotations

import math
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

    def stats(self) -> dict:
        return {"scored": self._scored, "cached": 0}


class CrossEncoderScorer:
    name = "bge"
    coverage_threshold = 0.5

    def __init__(self, model_name: str, cache: bool = True):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise DependencyError("Install requirements-ml.txt to use the bge reranker") from exc
        self.model = CrossEncoder(model_name)
        # Identical (query, passage) pairs get the identical score, so remembering them never changes a ranking.
        # RERANK_CACHE=off turns the memory off, so every search runs the model on every pair again.
        self.cache_enabled = cache
        self._cache: dict[tuple[str, str], float] = {}
        self._scored = 0
        self._reused = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        known = self._cache if self.cache_enabled else {}
        missing = list(dict.fromkeys(p for p in pairs if p not in known))
        fresh: dict[tuple[str, str], float] = {}
        if missing:
            logits = self.model.predict(missing, batch_size=4)
            for pair, x in zip(missing, logits):
                fresh[pair] = 1.0 / (1.0 + math.exp(-float(x)))
            if self.cache_enabled:
                self._cache.update(fresh)
        self._scored += len(missing)
        self._reused += len(pairs) - len(missing)
        return [fresh[p] if p in fresh else known[p] for p in pairs]

    def stats(self) -> dict:
        return {"scored": self._scored, "cached": self._reused}


def build_scorer(settings):
    if settings.rerank_backend == "lexical":
        return LexicalScorer()
    if settings.rerank_backend == "bge":
        return CrossEncoderScorer(settings.rerank_model, cache=getattr(settings, "rerank_cache", True))
    raise ValueError(f"Unknown RERANK_BACKEND: {settings.rerank_backend}")
