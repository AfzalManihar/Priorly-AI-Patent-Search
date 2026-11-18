"""Element-level evidence: does a passage support a claim element, support it partly, or not at all?

The cross-encoder says how topically related two texts are. That is not disclosure, and one absolute score cut-off
cannot separate "related" from "discloses" (0.5 marked every element covered, 0.8 marked none). So an element counts
only when two independent signals agree:

  recall : the share of the element's key terms (stemmed, stop words removed) that the passage really contains.
           Cheap, deterministic, and zero when the passage is about something else.
  ce     : the cross-encoder score for (element, passage), which confirms the words are used in the same sense.

A cross-encoder pair is only scored when the recall is high enough to matter, so most unrelated pairs cost nothing.
"""
from __future__ import annotations

from .tokenizer import tokenize

SUPPORTED, PARTIAL, UNSUPPORTED = "supported", "partial", "unsupported"


def term_recall(element: str, passage: str) -> float:
    terms = set(tokenize(element))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(passage))) / len(terms)


def classify(ce: float | None, recall: float, s) -> str:
    """ce is None when the pair was not worth scoring (recall below the partial level)."""
    if ce is None:
        return UNSUPPORTED
    if recall >= s.support_recall and ce >= s.support_ce:
        return SUPPORTED
    if recall >= s.partial_recall and ce >= s.partial_ce:
        return PARTIAL
    return UNSUPPORTED
