"""Retrieval metrics and a bootstrap confidence interval (PRD section 12)."""
from __future__ import annotations

import random


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Divides by k even when fewer than k results came back, so short lists are not rewarded."""
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling queries."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(values) for _ in values]) for _ in range(n))
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)
