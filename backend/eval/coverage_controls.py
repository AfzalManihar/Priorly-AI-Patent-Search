"""Controls for element coverage, on the DEV split only (the frozen TEST split is never read).

  positive control : each element of a query claim against the passages of the SAME patent. Its own words are there, so
                     these should come out supported.
  negative control : the same elements against passages of random OTHER patents. These should come out unsupported.
  hard negative    : the same elements against passages of patents in the SAME CPC subclass that are not cited for the
                     query. Same topic, usually without the element, so this is the case a topical reranker gets wrong.
                     It is approximate: an uncited patent can still disclose an element.

It prints, for the current thresholds, the share of each control that lands in supported / partial / unsupported, plus
the cross-encoder score quartiles. Use it to choose COVERAGE_* values; it invents no labels and writes no files.

    python -m eval.coverage_controls --queries data/real/queries.json --n 20
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.corpus import load_corpus
from app.evidence import classify, term_recall
from app.llm import heuristic_decompose
from app.rerank import build_scorer
from eval.grid import select_queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="dev", choices=["dev", "val"])
    args = ap.parse_args()

    s = Settings.from_env()
    corpus = load_corpus(s.corpus_path)
    queries = select_queries(args.queries, args.split, args.n, args.seed, corpus.patents)
    scorer = build_scorer(s)
    th = SimpleNamespace(support_recall=s.support_recall, partial_recall=s.partial_recall, support_ce=s.support_ce, partial_ce=s.partial_ce)
    if scorer.name == "lexical":
        th.support_ce, th.partial_ce = scorer.coverage_threshold, 0.6 * scorer.coverage_threshold
    rng = random.Random(args.seed)
    others = sorted(corpus.by_patent)
    out = {"positive": [], "negative": [], "hard_negative": []}
    by_cpc: dict[str, list[str]] = {}
    for pid, pat in corpus.patents.items():
        by_cpc.setdefault((pat.cpc or "")[:4], []).append(pid)

    for q in queries:
        elements = heuristic_decompose(q["claim"])
        own = corpus.by_patent.get(q["patent_id"], [])
        if not own:
            continue
        same = [p for p in by_cpc.get((corpus.patents[q["patent_id"]].cpc or "")[:4], []) if p != q["patent_id"] and p not in q["relevant"]]
        hard = [rng.choice(corpus.by_patent[rng.choice(same)]) for _ in range(4)] if same else []
        for kind, ids in (("positive", own[:4]), ("negative", [rng.choice(corpus.by_patent[rng.choice(others)]) for _ in range(4)]),
                          ("hard_negative", hard)):
            for ci in ids:
                text = corpus.chunks[ci].text
                for el in elements:
                    rec = term_recall(el, text)
                    ce = scorer.score_pairs([(el, text)])[0] if rec >= th.partial_recall else None
                    out[kind].append((classify(ce, rec, th), ce, rec))

    print(f"split={args.split} queries={len(queries)} scorer={scorer.name} thresholds={vars(th)}")
    for kind, rows in out.items():
        if not rows:
            continue
        n = len(rows)
        share = {k: sum(1 for r in rows if r[0] == k) / n for k in ("supported", "partial", "unsupported")}
        ces = sorted(r[1] for r in rows if r[1] is not None)
        recs = sorted(r[2] for r in rows)

        def pct(xs, q):
            return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None

        print(f"{kind:13s} n={n:5d}  " + "  ".join(f"{k}={v:.1%}" for k, v in share.items()) + f"  scored={len(ces)}")
        print(f"{'':13s} CE     min={pct(ces, 0)} p25={pct(ces, .25)} p50={pct(ces, .5)} p75={pct(ces, .75)} p95={pct(ces, .95)} max={pct(ces, .999)}")
        print(f"{'':13s} recall p25={pct(recs, .25)} p50={pct(recs, .5)} p75={pct(recs, .75)} p95={pct(recs, .95)}")
        for cut in (0.55, 0.6, 0.65):
            print(f"{'':13s} share of scored pairs with CE >= {cut}: {sum(1 for c in ces if c >= cut) / max(1, len(ces)):.1%}")


if __name__ == "__main__":
    main()
