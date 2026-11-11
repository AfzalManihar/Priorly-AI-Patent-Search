"""Corpus loading and chunking.

Corpus file format (JSON Lines), one patent per line:
  {"id": "US 9,104,382 B2", "title": "...", "assignee": "...", "year": 2015,
   "cpc": "B60L 53/12", "abstract": "...", "claims": ["..."],
   "paragraphs": [{"no": "0042", "text": "..."}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Patent:
    id: str
    title: str
    assignee: str
    year: int
    cpc: str
    abstract: str
    claims: list[str]
    paragraphs: list[dict]


@dataclass
class Chunk:
    idx: int
    patent_id: str
    loc: str
    kind: str  # abstract | claim | paragraph
    text: str


def split_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split text into bounded chunks without dropping any content."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    text = " ".join(text.split())
    if not text:
        return []

    sentences = re.split(r"(?<=[.;!?])\s+", text)
    pieces: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            pieces.append(current)
            current = ""

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            # Split exceptionally long tokens as a last resort.
            parts = [word[i:i + max_chars] for i in range(0, len(word), max_chars)]
            for part in parts:
                candidate = f"{current} {part}".strip()
                if current and len(candidate) > max_chars:
                    flush()
                    current = part
                else:
                    current = candidate
        # Preserve sentence boundaries where possible.
    flush()
    return pieces


def chunk_patent(patent: Patent, start_idx: int = 0, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(loc: str, kind: str, text: str) -> None:
        for piece in split_text(text, max_chars):
            chunks.append(Chunk(start_idx + len(chunks), patent.id, loc, kind, piece))

    if patent.abstract:
        add("Abstract", "abstract", patent.abstract)
    for n, claim in enumerate(patent.claims, start=1):
        add(f"Claim {n}", "claim", claim)
    for para in patent.paragraphs:
        add(f"Paragraph [{para['no']}]", "paragraph", para["text"])
    return chunks


class Corpus:
    def __init__(self, patents: list[Patent], max_chars: int = 1200):
        self.patents: dict[str, Patent] = {p.id: p for p in patents}
        self.chunks: list[Chunk] = []
        self.by_patent: dict[str, list[int]] = {}
        for patent in patents:
            new = chunk_patent(patent, start_idx=len(self.chunks), max_chars=max_chars)
            self.chunks.extend(new)
            self.by_patent[patent.id] = [c.idx for c in new]
        self.years = np.array([self.patents[c.patent_id].year for c in self.chunks])
        self.patent_ids = np.array([c.patent_id for c in self.chunks])

    def __len__(self) -> int:
        return len(self.patents)

    def mask(self, exclude_ids=frozenset(), max_year: int | None = None) -> np.ndarray:
        """Boolean array over chunks: True where the chunk may be returned (leakage controls)."""
        allowed = np.ones(len(self.chunks), dtype=bool)
        if max_year is not None:
            allowed &= self.years <= max_year
        if exclude_ids:
            allowed &= ~np.isin(self.patent_ids, list(exclude_ids))
        return allowed


def load_corpus(path: Path, max_chars: int = 1200) -> Corpus:
    patents: list[Patent] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            patents.append(
                Patent(
                    id=raw["id"],
                    title=raw.get("title", ""),
                    assignee=raw.get("assignee", ""),
                    year=int(raw.get("year", 0) or 0),
                    cpc=raw.get("cpc", ""),
                    abstract=raw.get("abstract", ""),
                    claims=list(raw.get("claims", [])),
                    paragraphs=list(raw.get("paragraphs", [])),
                )
            )
    if not patents:
        raise ValueError(f"No patents found in {path}")
    return Corpus(patents, max_chars=max_chars)
