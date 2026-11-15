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


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """1 when at least one relevant patent is in the top k, else 0."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG: a hit at rank r earns 1/log2(r+1), normalised by the best possible ordering."""
    import math

    if not relevant:
        return 0.0
    gain = sum(1.0 / math.log2(pos + 1) for pos, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def average_precision(ranked: list[str], relevant: set[str], k: int = 50) -> float:
    """AP@k. The mean over queries is MAP."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for pos, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            hits += 1
            total += hits / pos
    return total / min(len(relevant), k)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def paired_bootstrap_ci(a: list[float], b: list[float], n: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float]:
    """CI for mean(a - b) over the same queries. If it excludes 0 the difference is not noise."""
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n=n, seed=seed, alpha=alpha)
