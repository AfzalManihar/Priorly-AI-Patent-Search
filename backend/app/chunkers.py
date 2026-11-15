"""Chunking strategies. Every strategy turns one patent into a list of Chunk objects.

The evaluation grid (eval/grid.py) builds one corpus per strategy and measures what each one does to
retrieval quality. `claim_aware` is the strategy the search engine used before the grid existed.
"""
from __future__ import annotations

import re
from typing import Callable

from .corpus import Chunk, Patent, split_text

SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")


def _sections(patent: Patent) -> list[tuple[str, str, str]]:
    """(loc, kind, text) for every part of a patent that carries searchable text."""
    parts: list[tuple[str, str, str]] = []
    if patent.abstract:
        parts.append(("Abstract", "abstract", patent.abstract))
    for n, claim in enumerate(patent.claims, start=1):
        parts.append((f"Claim {n}", "claim", claim))
    for para in patent.paragraphs:
        parts.append((f"Paragraph [{para['no']}]", "paragraph", para["text"]))
    return parts


def _emit(patent: Patent, start: int, pieces: list[tuple[str, str, str]]) -> list[Chunk]:
    return [Chunk(start + i, patent.id, loc, kind, text) for i, (loc, kind, text) in enumerate(pieces) if text.strip()]


# 1. claim_aware: sentence-packed chunks up to max_chars, never crossing a section boundary --------------------
def claim_aware(patent: Patent, start_idx: int = 0, max_chars: int = 1200) -> list[Chunk]:
    out = []
    for loc, kind, text in _sections(patent):
        out += [(loc, kind, piece) for piece in split_text(text, max_chars)]
    return _emit(patent, start_idx, out)


# 2. fixed_window: character windows with overlap ------------------------------------------------------------
def fixed_window(patent: Patent, start_idx: int = 0, max_chars: int = 1200, size: int = 400, overlap: int = 80) -> list[Chunk]:
    out = []
    step = max(1, size - overlap)
    for loc, kind, text in _sections(patent):
        text = " ".join(text.split())
        for start in range(0, max(1, len(text)), step):
            window = text[start:start + size]
            if window:
                out.append((loc, kind, window))
            if start + size >= len(text):
                break
    return _emit(patent, start_idx, out)


# 3. sentence: two sentences per chunk -----------------------------------------------------------------------
def sentence(patent: Patent, start_idx: int = 0, max_chars: int = 1200, per_chunk: int = 2) -> list[Chunk]:
    out = []
    for loc, kind, text in _sections(patent):
        sentences = [s for s in SENTENCE_END.split(" ".join(text.split())) if s]
        for i in range(0, len(sentences), per_chunk):
            out.append((loc, kind, " ".join(sentences[i:i + per_chunk])[:max_chars]))
    return _emit(patent, start_idx, out)


# 4. recursive: split on the largest separator that keeps pieces under the size limit -----------------------
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    pieces: list[str] = []
    current = ""
    for part in text.split(sep):
        part = part if not current else part
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) > size:
            pieces += _recursive_split(part, size, rest)
            current = ""
        else:
            current = part
    if current:
        pieces.append(current)
    return pieces


def recursive(patent: Patent, start_idx: int = 0, max_chars: int = 1200, size: int = 500) -> list[Chunk]:
    out = []
    for loc, kind, text in _sections(patent):
        for piece in _recursive_split(text.strip(), size, SEPARATORS):
            piece = " ".join(piece.split())
            if piece:
                out.append((loc, kind, piece))
    return _emit(patent, start_idx, out)


# 5. whole_document: one chunk per patent (title, abstract, then claims), the "no chunking" baseline ----------
def whole_document(patent: Patent, start_idx: int = 0, max_chars: int = 1200, limit: int = 6000) -> list[Chunk]:
    text = " ".join([patent.title, patent.abstract, *patent.claims]).strip()
    return _emit(patent, start_idx, [("Full text", "abstract", " ".join(text.split())[:limit])])


Chunker = Callable[..., list[Chunk]]

CHUNKERS: dict[str, tuple[str, Chunker, str]] = {
    "claim_aware": ("Section-aware (current)", claim_aware,
                    "Packs whole sentences up to 1,200 characters and never crosses an abstract or claim boundary."),
    "fixed_window": ("Fixed window + overlap", fixed_window,
                     "400-character windows that overlap by 80, cutting through sentences."),
    "sentence": ("Sentence pairs", sentence, "Two sentences per chunk, the smallest meaningful unit."),
    "recursive": ("Recursive splitter", recursive,
                  "Splits on paragraph, then sentence, then clause, then word to stay under 500 characters."),
    "whole_document": ("Whole document", whole_document,
                       "One chunk per patent: title, abstract and claims together. The no-chunking baseline."),
}


def get_chunker(name: str) -> Chunker:
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy {name!r}. Choose from: {', '.join(CHUNKERS)}")
    return CHUNKERS[name][1]
