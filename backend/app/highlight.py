"""Mark the text in a passage that matches each claim element.

Output is HTML-safe: all passage text is escaped and only <mark data-e="n"> tags are added
(PRD NFR-07).
"""
from __future__ import annotations

import html
import math
import re

from .tokenizer import tokenize, word_stems

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_./]*[A-Za-z0-9]|[A-Za-z0-9]")
_CLAUSE_BREAK = re.compile(r"[;:]|[.!?](?=\s|$)")


def _best_span(stems: list[str], segments: list[int], needed: set[str], min_hits: int) -> tuple[int, int, int] | None:
    """Return (first_token, last_token, hits) of the densest window inside one clause, or None."""
    window = max(8, 2 * len(needed))
    best: tuple[int, int, int] | None = None
    n = len(stems)
    for i in range(n):
        if stems[i] not in needed:
            continue
        seen: set[str] = set()
        last = i
        for j in range(i, min(n, i + window)):
            if segments[j] != segments[i]:
                break
            if stems[j] in needed:
                seen.add(stems[j])
                last = j
        hits = len(seen)
        if hits < min_hits:
            continue
        candidate = (i, last, hits)
        if best is None or hits > best[2] or (hits == best[2] and (last - i) < (best[1] - best[0])):
            best = candidate
    return best


def highlight(text: str, elements: dict[int, str]) -> str:
    """Return `text` escaped, with <mark data-e="n"> around the best matching span per element."""
    words = [(m.start(), m.end(), word_stems(m.group(0))) for m in _WORD.finditer(text)]
    stems = [w[2] for w in words]
    # A clause ends at ; : or a sentence-final period. Spans never cross clause boundaries.
    segments: list[int] = []
    segment, previous_end = 0, 0
    for start, end, _ in words:
        if _CLAUSE_BREAK.search(text[previous_end:start]):
            segment += 1
        segments.append(segment)
        previous_end = end
    found: list[tuple[int, int, int, int]] = []  # (hits, start_token, end_token, element)
    for n, element_text in elements.items():
        needed = set(tokenize(element_text))
        if not needed:
            continue
        min_hits = 1 if len(needed) == 1 else max(2, math.ceil(len(needed) * 0.5))
        span = _best_span(stems, segments, needed, min_hits)
        if span:
            found.append((span[2], span[0], span[1], n))

    accepted: list[tuple[int, int, int]] = []  # (char_start, char_end, element)
    for hits, first, last, n in sorted(found, key=lambda f: (-f[0], f[1])):
        start, end = words[first][0], words[last][1]
        if any(start < a_end and a_start < end for a_start, a_end, _ in accepted):
            continue
        accepted.append((start, end, n))
    accepted.sort()

    out: list[str] = []
    cursor = 0
    for start, end, n in accepted:
        out.append(html.escape(text[cursor:start], quote=False))
        out.append(f'<mark data-e="{n}">{html.escape(text[start:end], quote=False)}</mark>')
        cursor = end
    out.append(html.escape(text[cursor:], quote=False))
    return "".join(out)
