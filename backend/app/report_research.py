"""Research report PDF: built only from eval/grid.py output, so every number in it was measured."""
from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .report_pdf import _drawable

RET = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


def _para(text: str, style) -> Paragraph:
    return Paragraph(_drawable(text).replace("&", "&amp;").replace("<", "&lt;"), style)


def _heatmap(data: dict, metric: str, title: str):
    from reportlab.graphics.shapes import Drawing, Rect, String

    chunkers = list(data["meta"]["chunkers"])
    cells = {
        (c["chunking"], c["retrieval"]): c["metrics"][metric]["mean"]
        for c in data["combos"] if c["stage"] == "A"
    }
    grid = [[cells.get((ch, r)) for r in RET] for ch in chunkers]
    width, height = 468, max(125, 34 * len(chunkers) + 55)
    drawing = Drawing(width, height)

    drawing.add(String(8, height - 16, title, fontName="Helvetica-Bold", fontSize=10))
    left, top, cell_w, cell_h = 125, height - 42, 108, 27

    values = [v for row in grid for v in row if v is not None]
    low, high = (min(values), max(values)) if values else (0, 1)
    spread = max(high - low, 1e-9)

    for j, retrieval in enumerate(RET):
        drawing.add(String(left + j * cell_w + 27, top + 8, RET[retrieval],
                           fontName="Helvetica-Bold", fontSize=8))

    for i, (chunker, row) in enumerate(zip(chunkers, grid)):
        y = top - (i + 1) * cell_h
        label = data["meta"]["chunkers"][chunker]["label"]
        drawing.add(String(4, y + 9, label[:21], fontName="Helvetica", fontSize=7))
        for j, value in enumerate(row):
            x = left + j * cell_w
            intensity = 0 if value is None else (value - low) / spread
            shade = 0.94 - 0.55 * intensity
            drawing.add(Rect(x, y, cell_w - 3, cell_h - 3,
                             fillColor=colors.Color(shade, shade + 0.015, 1),
                             strokeColor=colors.HexColor("#c8ccd4"), strokeWidth=0.5))
            label_text = "-" if value is None else f"{value:.3f}"
            drawing.add(String(x + 35, y + 8, label_text,
                               fontName="Helvetica-Bold", fontSize=8,
                               fillColor=colors.white if intensity > 0.62 else colors.black))
    return drawing


def _bars(data: dict):
    from reportlab.graphics.shapes import Drawing, Rect, String, Line

    top = sorted(data["combos"], key=lambda c: -c["metrics"]["ndcg@10"]["mean"])[:12]
    width, height = 468, 245
    drawing = Drawing(width, height)
    drawing.add(String(8, height - 14, "Top 12 pipeline combinations by nDCG@10",
                       fontName="Helvetica-Bold", fontSize=10))

    label_x, chart_x = 5, 245
    chart_width, row_h = 190, 17
    max_value = max((c["metrics"]["ndcg@10"]["mean"] for c in top), default=1) or 1

    for i, combo in enumerate(top):
        y = height - 38 - (i + 1) * row_h
        metric = combo["metrics"]["ndcg@10"]
        value = metric["mean"]
        low, high = metric["ci"]
        drawing.add(String(label_x, y + 4, combo["label"][:39],
                           fontName="Helvetica", fontSize=6.5))
        drawing.add(Rect(chart_x, y, chart_width * value / max_value, 10,
                         fillColor=colors.HexColor("#2f5fd0"), strokeColor=None))
        drawing.add(Line(chart_x + chart_width * low / max_value, y + 5,
                         chart_x + chart_width * high / max_value, y + 5,
                         strokeColor=colors.HexColor("#333333"), strokeWidth=1))
        drawing.add(String(chart_x + chart_width + 5, y + 2, f"{value:.3f}",
                           fontName="Helvetica", fontSize=7))
    return drawing


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
    s += [Spacer(1, 6), _heatmap(data, "recall@10", "Recall@10: chunking x retrieval"), _heatmap(data, "ndcg@10", "nDCG@10: chunking x retrieval"),
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
