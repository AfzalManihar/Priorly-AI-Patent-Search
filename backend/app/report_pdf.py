"""PDF export for saved reports (PRD FR-14).

Built with ReportLab's standard fonts, so the file opens anywhere and needs no font files.
Characters outside Latin-1 are replaced with "?" because the standard fonts cannot draw them.
"""
from __future__ import annotations

import io
import re
import unicodedata
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

_TAGS = re.compile(r"<[^>]+>")
STAGE_NAMES = {"decompose": "Claim decomposition", "hyde": "HyDE expansion", "hybrid": "Keyword search and fusion",
               "rerank": "Cross-encoder rerank"}


_PUNCT = {"\u2026": "...", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
          "\u2022": "*", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2212": "-", "\u2192": "->", "\u00a0": " "}
_GREEK = {"\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta", "\u03b5": "epsilon", "\u03b7": "eta",
          "\u03b8": "theta", "\u03ba": "kappa", "\u03bb": "lambda", "\u03bc": "mu", "\u03bd": "nu", "\u03c0": "pi",
          "\u03c1": "rho", "\u03c3": "sigma", "\u03c4": "tau", "\u03c6": "phi", "\u03c8": "psi", "\u03c9": "omega",
          "\u0394": "Delta", "\u03a3": "Sigma", "\u03a9": "Omega", "\u03a6": "Phi", "\u03a0": "Pi"}


def _drawable(text: str) -> str:
    """Map characters the standard PDF fonts cannot draw to readable ASCII instead of printing '?'."""
    out = []
    for ch in text:
        if ch in _PUNCT:
            out.append(_PUNCT[ch])
        elif ch in _GREEK:
            out.append(_GREEK[ch])
        else:
            try:
                ch.encode("latin-1")
                out.append(ch)
            except UnicodeEncodeError:
                folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
                out.append(folded or "?")
    return "".join(out)


def _text(value) -> str:
    """Plain, XML-safe text that the standard PDF fonts can draw."""
    plain = _TAGS.sub("", str(value if value is not None else ""))
    return escape(_drawable(plain))


def backend_notes(response: dict) -> list[str]:
    """Which backends produced the results, plus a warning when any of them is an offline stand-in."""
    b = response.get("retrieval") or {}
    if not b:
        return []
    notes = [f"Backends: vector {b.get('dense_backend')}, rerank {b.get('rerank_backend')}, LLM {b.get('llm_backend')}."]
    if any(b.get(k) == v for k, v in (("dense_backend", "tfidf"), ("rerank_backend", "lexical"), ("llm_backend", "heuristic"))):
        notes.append("Offline stand-in backends were used, so element coverage is judged by shared words only and may "
                     "miss paraphrases. Re-run with the real models before relying on it.")
    return notes


def build_report_pdf(report: dict, search: dict) -> bytes:
    request, response = search["request"], search["response"]
    shortlist = set(report["shortlist"])
    results = response["results"]

    base = getSampleStyleSheet()
    ink, quiet = colors.HexColor("#1f2933"), colors.HexColor("#5f6b7a")
    h1 = ParagraphStyle("h1", parent=base["Title"], fontSize=19, leading=23, alignment=0, textColor=ink, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5, leading=16, textColor=ink, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11.5, textColor=quiet)
    passage = ParagraphStyle("passage", parent=body, fontSize=9, leading=12.5, leftIndent=8, borderPadding=(3, 3, 3, 6),
                             backColor=colors.HexColor("#f3f4f6"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(quiet)
        canvas.drawString(18 * mm, 10 * mm, "Priorly. A research aid, not a legal opinion.")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=_text(report["title"]), author="Priorly")
    story = [Paragraph(_text(report["title"]), h1),
             Paragraph(_text(f"Saved {report['created'][:10]}  |  {len(results)} results  |  "
                             f"{len(shortlist)} shortlisted  |  best score {report['best']:.2f}"), small),
             HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d5d9df"), spaceBefore=6, spaceAfter=4),
             Paragraph("Claim searched", h2), Paragraph(_text(request.get("claim", "")), body)]

    elements = response.get("elements") or []
    if elements:
        story.append(Paragraph("Claim elements", h2))
        for el in elements:
            story.append(Paragraph(_text(f"E{el['n']}: {el['text']}"), body))

    options = request.get("options") or {}
    used = [STAGE_NAMES[k] for k in STAGE_NAMES if options.get(k, True)]
    off = [STAGE_NAMES[k] for k in STAGE_NAMES if not options.get(k, True)]
    story.append(Paragraph("Search settings", h2))
    story.append(Paragraph(_text("Stages on: " + (", ".join(used) or "none") +
                                 (". Stages off: " + ", ".join(off) if off else "") + f". Top {request.get('k', len(results))}."), body))
    for note in backend_notes(response):
        story.append(Paragraph(_text(note), small))
    if response.get("degraded"):
        story.append(Paragraph(_text("Degraded run: " + ", ".join(response["degraded"]) +
                                     " fell back to a simpler method."), small))

    story.append(Paragraph("Results", h2))
    for rank, r in enumerate(results, start=1):
        mark = "  [SHORTLISTED]" if r["id"] in shortlist else ""
        covered = ", ".join(f"E{n}" for n in r["el"]) or "none identified"
        block = [
            Paragraph(f"<b>{rank}. {_text(r['title'])}</b>{_text(mark)}", body),
            Paragraph(_text(f"{r['id']}  |  {r['assignee']}  |  {r['year']}  |  {r['cpc']}"), small),
            Paragraph(_text(f"Scorer-matched elements: {covered}  |  Score {r['score']:.2f}  |  {r['loc']}"
                            + ("  |  Found by keyword search only" if r.get("missed_by_vector") else "")), small),
            Spacer(1, 2),
            Paragraph(_text(r["passage"]), passage),
            Spacer(1, 2),
            Paragraph(_text(r["why"]), body),
            Spacer(1, 7),
        ]
        story.append(KeepTogether(block))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
