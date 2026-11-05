"""Show, for the first N queries of a split, where the examiner-cited patents land with dense-only retrieval.

  python -m eval.peek --queries data/real/queries.json --split dev --n 5
Fast: no reranker, no LLM. Use it to tell a bug (relevant patents missing from the corpus or filtered out) from a
genuinely hard query set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.pipeline import Options
from app.retrievers import Constraints
from eval.run_eval import build_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--depth", type=int, default=100)
    args = parser.parse_args()
    settings = Settings.from_env()
    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [q for q in spec["queries"] if q.get("split") == args.split][: args.n]
    engine = build_engine(settings)
    opts = Options(decompose=False, hyde=False, hybrid=False, rerank=False, explain=False)
    for q in queries:
        cons = Constraints(exclude_ids=frozenset([q["patent_id"]]), max_year=q["year"] - 1)
        in_corpus = [p for p in q["relevant"] if p in engine.corpus.patents]
        eligible = [p for p in in_corpus if engine.corpus.patents[p].year <= cons.max_year]
        ranked = [r["id"] for r in engine.search(q["claim"], opts, args.depth, cons)["results"]]
        print(f"\n{q['id']} {q['patent_id']} ({q['year']}) type={q['type']}")
        print("  claim:", q["claim"][:140].replace("\n", " "), "...")
        print(f"  relevant: {len(q['relevant'])} listed, {len(in_corpus)} in corpus, {len(eligible)} eligible by year")
        for p in eligible:
            print(f"    {p}: dense rank {ranked.index(p) + 1 if p in ranked else f'>{args.depth}'}  | {engine.corpus.patents[p].title[:60]}")
        print("  top 3 returned:", [engine.corpus.patents[p].title[:40] for p in ranked[:3]])


if __name__ == "__main__":
    main()
