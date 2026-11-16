"""Research report PDF: built only from eval/grid.py output, so every number in it was measured."""
from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .report_pdf import _drawable

RET = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


def _para(text: str, style) -> Paragraph:
    return Paragraph(_drawable(text).replace("&", "&amp;").replace("<", "&lt;"), style)


def _shade(t: float) -> colors.Color:
    """White to blue for t in 0..1."""
    t = max(0.0, min(1.0, t))
    return colors.Color(0.93 - 0.66 * t, 0.95 - 0.52 * t, 1.0 - 0.28 * t)


def _heatmap(data: dict, metric: str, title: str):
    chunkers = list(data["meta"]["chunkers"])
    cells = {(c["chunking"], c["retrieval"]): c["metrics"][metric]["mean"] for c in data["combos"] if c["stage"] == "A"}
    flat = list(cells.values()) or [0.0]
    lo, hi = min(flat), max(flat)
    small = ParagraphStyle("hm", fontSize=8, leading=10)
    rows = [[_para(title, ParagraphStyle("hmt", parent=small, fontSize=9, leading=11))] + [list(RET.values())[i] for i in range(3)]]
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 1, colors.white), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eefc"))]
    for i, ch in enumerate(chunkers, start=1):
        row = [_para(data["meta"]["chunkers"][ch]["label"], small)]
        for j, r in enumerate(RET, start=1):
            v = cells.get((ch, r))
            row.append("-" if v is None else f"{v:.3f}")
            if v is not None:
                style.append(("BACKGROUND", (j, i), (j, i), _shade(0.15 + 0.85 * ((v - lo) / (hi - lo) if hi > lo else 0.5))))
                if v == hi:
                    style.append(("BOX", (j, i), (j, i), 1.5, colors.HexColor("#0e7a4f")))
        rows.append(row)
    table = Table(rows, colWidths=[62 * mm, 32 * mm, 32 * mm, 32 * mm], rowHeights=[9 * mm] + [10 * mm] * len(chunkers))
    table.setStyle(TableStyle(style))
    return table


def _bars(data: dict):
    from reportlab.graphics.shapes import Drawing, Line, Rect, String

    top = sorted(data["combos"], key=lambda c: -c["metrics"]["ndcg@10"]["mean"])[:14]
    row, left, width = 12, 250, 470
    height = row * len(top) + 18
    d = Drawing(width + 40, height)
    hi = max(c["metrics"]["ndcg@10"]["ci"][1] for c in top) or 1
    x = lambda v: left + (width - left) * v / hi
    for i, c in enumerate(top):
        y = height - 14 - i * row
        m = c["metrics"]["ndcg@10"]
        d.add(String(left - 6, y, _drawable(c["label"])[:62], fontSize=5.5, textAnchor="end"))
        d.add(Rect(left, y - 2, max(1, x(m["mean"]) - left), 7, fillColor=colors.HexColor("#2f5fd0"), strokeColor=None))
        d.add(Line(x(m["ci"][0]), y + 1.5, x(m["ci"][1]), y + 1.5, strokeColor=colors.HexColor("#333333"), strokeWidth=0.8))
        d.add(String(x(m["ci"][1]) + 4, y, f"{m['mean']:.3f}", fontSize=6))
    d.add(String(left, 2, "nDCG@10, top 14 combinations (bars show 95% confidence interval)", fontSize=6))
    return d


def build_research_pdf(data: dict) -> bytes:
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Title"], fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    m = data["meta"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Priorly retrieval research report")
    s = [_para("Priorly: retrieval strategy comparison", h1),
         _para(f"Run {m['created'][:10]}. {len(data['combos'])} pipeline combinations, each measured on the same {m['queries']} real patent claims "
               f"({', '.join(f'{v} {k}' for k, v in m['query_types'].items())}) over a pool of {m['pool_patents']:,} of {m['corpus_patents']:,} patents.", body),
         _para(f"Ground truth: {m['ground_truth']}. Leakage controls: the query patent is excluded and only earlier patents may be returned. "
               f"Backends: dense {m['dense_backend']}{' (' + m['embed_model'] + ')' if m['dense_backend'] == 'bge' else ''}, rerank {m['rerank_backend']}{' (' + m['rerank_model'] + ')' if m['rerank_backend'] == 'bge' else ''}, LLM {m['llm_backend']} "
               f"at temperature {m['llm_temperature']}. Fusion: RRF k={m['rrf_k']}.", small)]
    if m["dense_backend"] != "bge":
        s.append(_para("Note: this run used the offline stand-in for dense retrieval, not BGE embeddings, so it validates the method rather than the final model numbers.", small))
    best = data.get("best")
    if best:
        s += [_para("Selected default", h2), _para(f"{best['label']}. Rule: {best['rule']}. nDCG@10 {best['metrics']['ndcg@10']['mean']:.3f}, "
                                                  f"Recall@10 {best['metrics']['recall@10']['mean']:.3f}, MRR {best['metrics']['mrr']['mean']:.3f}.", body)]
    s.append(_para("Findings", h2))
    for f in data.get("findings", []):
        s.append(_para("- " + f["text"], body))
    s.append(_para("Absolute scores look low because examiner citations are an incomplete record of relevance. Compare strategies with each other, not with 1.0.", small))
    s += [PageBreak(), _para("Chunking strategies", h2)]
    for key, c in m["chunkers"].items():
        s.append(_para(f"{c['label']}: {c['description']}", body))
    s += [Spacer(1, 6), _heatmap(data, "recall@10", "Recall@10"), Spacer(1, 8), _heatmap(data, "ndcg@10", "nDCG@10"),
          PageBreak(), _para("All combinations, ranked by nDCG@10", h2), _bars(data), PageBreak(), _para("Full results", h2)]
    head = ["Combination", "R@10", "R@50", "P@10", "MRR", "nDCG", "MAP", "Hit", "p95 ms"]
    rows = [head]
    for c in sorted(data["combos"], key=lambda c: -c["metrics"]["ndcg@10"]["mean"]):
        mm_ = c["metrics"]
        rows.append([_para(c["label"], ParagraphStyle("t", parent=small, fontSize=6.5, leading=8))] +
                    [f"{mm_[k]['mean']:.3f}" for k in ("recall@10", "recall@50", "precision@10", "mrr", "ndcg@10", "map", "hit@10")] + [f"{c['latency_ms']['p95']:.0f}"])
    table = Table(rows, colWidths=[68 * mm] + [13.5 * mm] * 7 + [14 * mm], repeatRows=1)
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eefc")),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8ccd4")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    s.append(table)
    if data.get("skipped"):
        s += [_para("Not run", h2)] + [_para(f"{x['label']}: {x['reason']}", small) for x in data["skipped"]]
    doc.build(s)
    return buf.getvalue()
