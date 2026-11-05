"""Reciprocal Rank Fusion (PRD section 8)."""
from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked lists of ids.

    score(d) = sum over lists of 1 / (k + rank_in_list(d)), ranks start at 1.
    Ties are broken by the best (lowest) rank the id achieved in any list, then by id,
    so the output is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    ordered = sorted(scores, key=lambda d: (-scores[d], best_rank[d], d))
    return [(d, scores[d]) for d in ordered]
