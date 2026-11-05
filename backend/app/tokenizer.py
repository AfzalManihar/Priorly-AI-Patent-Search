"""Tokenizer shared by keyword search and the lexical scorer.

Design goals (PRD section 8): keep patent identifiers, numbers and chemical
formulae intact so exact-match queries work, and apply a light stemmer so
"adjusts" and "adjusting" meet.
"""
from __future__ import annotations

import re
from functools import lru_cache

# "US 9,104,382 B2", "US9104382", "us 9,104,382" -> single token "us9104382b2"
_PATENT_ID = re.compile(r"\bUS\s?(\d{1,2}),?(\d{3}),?(\d{3})(?:\s?([AB]\d))?\b", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the
    their then there these this to was were which while with said wherein comprising
    comprises comprise including includes configured such each one more least based
    may can also using used use""".split()
)


def normalize_ids(text: str) -> str:
    """Rewrite patent numbers into one lowercase token."""

    def repl(m: re.Match) -> str:
        number = f"us{m.group(1)}{m.group(2)}{m.group(3)}"
        suffix = (m.group(4) or "").lower()
        # Emit the bare number as well, so "US 5,555,123" finds text that says "US 5,555,123 B2".
        return f" {number}{suffix} {number} " if suffix else f" {number} "

    return _PATENT_ID.sub(repl, text)


def stem(token: str) -> str:
    """Very light suffix stripper. Tokens with digits (identifiers, formulae) are untouched."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


@lru_cache(maxsize=65536)
def _tokenize_cached(text: str, keep_stop: bool) -> tuple[str, ...]:
    lowered = normalize_ids(text).lower()
    out: list[str] = []
    for match in _TOKEN.finditer(lowered):
        tok = match.group(0)
        pieces = [tok]
        if any(sep in tok for sep in "-_./"):
            pieces += [p for p in re.split(r"[-_./]", tok) if len(p) > 1]
        for piece in pieces:
            if not keep_stop and piece in STOPWORDS:
                continue
            out.append(stem(piece))
    return tuple(out)


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    return list(_tokenize_cached(text, keep_stop))


def word_stems(word: str) -> str:
    """Stem a single surface word the same way tokenize does (used by the highlighter)."""
    return stem(word.lower())
