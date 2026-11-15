"""Pseudo-relevance feedback: add the most characteristic terms of the top BM25 chunks to a query."""
from __future__ import annotations

import math
from collections import Counter

from .tokenizer import tokenize


class Prf:
    def __init__(self, corpus, bm25, terms: int = 12, docs: int = 3):
        self.corpus, self.bm25, self.terms, self.docs = corpus, bm25, terms, docs
        self.df: Counter = Counter()
        for chunk in corpus.chunks:
            self.df.update(set(tokenize(chunk.text)))

    def expand(self, claim: str, cons) -> str:
        own = set(tokenize(claim))
        weights: Counter = Counter()
        for idx, _ in self.bm25.search(claim, self.docs, cons):
            for token, count in Counter(tokenize(self.corpus.chunks[idx].text)).items():
                if token not in own and len(token) >= 4:
                    weights[token] += count * math.log(1 + len(self.corpus.chunks) / self.df[token])
        return claim + " " + " ".join(t for t, _ in weights.most_common(self.terms))
