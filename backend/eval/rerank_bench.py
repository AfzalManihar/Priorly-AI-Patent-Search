"""Measure the real cross-encoder on DEV queries: pairs, seconds per pair, and the old versus the new rerank work.

No Qdrant, no Groq, no writes. Candidate patents come from the keyword index (BM25) only, so it can run while the
server is up. The frozen TEST split is never read.

  old strategy : every candidate scores CHUNKS passages against the claim AND against every claim element
                 (CHUNKS x (1 + elements) pairs per patent, what the first version did)
  new strategy : every candidate scores RERANK_CHUNKS passages against the claim (RERANK_CHUNKS pairs per patent)

Both run on the SAME candidates and passages. The old strategy is measured on --old-patents patents only (it is slow),
the new one on the same patents and again on all --depth patents. Nothing is extrapolated.

    python -m eval.rerank_bench --queries data/real/queries.json --n 3 --depth 20
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.corpus import load_corpus
from app.llm import heuristic_decompose
from app.rerank import build_scorer
from app.retrievers import BM25Index, Constraints, aggregate_to_patents
from eval.grid import select_queries


def timed(scorer, pairs):
    before = scorer.stats()
    t0 = time.perf_counter()
    scorer.score_pairs(pairs)
    sec = time.perf_counter() - t0
    after = scorer.stats()
    return sec, after["scored"] - before["scored"], after["model_s"] - before.get("model_s", 0.0), after["batches"] - before.get("batches", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--old-patents", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s = Settings.from_env()
    if s.rerank_backend != "bge":
        raise SystemExit("Set RERANK_BACKEND=bge: this measures the real model.")
    corpus = load_corpus(s.corpus_path)
    bm25 = BM25Index(corpus)
    scorer = build_scorer(s)
    scorer.cache_enabled = False  # every pair really runs the model
    scorer.score_pairs([("warm up the model", "warm up the model")])  # first call loads weights lazily on some builds
    queries = select_queries(args.queries, "dev", args.n, args.seed, corpus.patents)
    tok = scorer.model.tokenizer
    print(f"model={s.rerank_model} max_length={getattr(scorer.model, 'max_seq_length', 512)} batch={scorer.batch_size} "
          f"chunks_per_patent={s.rerank_chunks} depth={args.depth}")
    rows = []
    for q in queries:
        cons = Constraints(exclude_ids=frozenset({q["patent_id"]}), max_year=None)
        elements = heuristic_decompose(q["claim"])
        pool = aggregate_to_patents(corpus, bm25.search(q["claim"], 200, cons))[: args.depth]
        chunks = {pid: ids[: s.chunk_pool] for pid, _, ids in pool}
        new_pairs = [(q["claim"], corpus.chunks[ci].text) for pid, ids in chunks.items() for ci in ids[: s.rerank_chunks]]
        old_ids = list(chunks)[: args.old_patents]
        old_pairs = [(t, corpus.chunks[ci].text) for pid in old_ids for ci in chunks[pid] for t in [q["claim"]] + elements]
        new_same = [(q["claim"], corpus.chunks[ci].text) for pid in old_ids for ci in chunks[pid][: s.rerank_chunks]]
        lens = [len(tok(a, b, truncation=True, max_length=512)["input_ids"]) for a, b in new_pairs[:30]]
        o = timed(scorer, old_pairs)
        n_same = timed(scorer, new_same)
        n_all = timed(scorer, new_pairs)
        rows.append((len(elements), o, n_same, n_all))
        print(f"\nquery {q['id']}: {len(elements)} elements, claim {len(q['claim'])} chars, median input {statistics.median(lens):.0f} tokens")
        print(f"  OLD  {len(old_ids)} patents: {o[1]:4d} pairs  {o[0]:6.1f} s  ({o[0] / max(1, o[1]):.3f} s/pair, model {o[2]:.1f} s, {o[3]} batches)")
        print(f"  NEW  {len(old_ids)} patents: {n_same[1]:4d} pairs  {n_same[0]:6.1f} s  ({n_same[0] / max(1, n_same[1]):.3f} s/pair)")
        print(f"  NEW  {len(chunks)} patents: {n_all[1]:4d} pairs  {n_all[0]:6.1f} s  ({n_all[0] / max(1, n_all[1]):.3f} s/pair)  <- the real rerank stage")

    print("\nbatch size sweep on the last query's new pairs (same pairs, same scores):")
    for bs in (1, 4, 16, 32):
        scorer.batch_size = bs
        sec, n, _, _ = timed(scorer, new_pairs)
        print(f"  batch {bs:2d}: {n} pairs in {sec:.1f} s ({sec / max(1, n):.3f} s/pair)")
    try:
        import torch
        print(f"\ntorch threads={torch.get_num_threads()} cpu_count={__import__('os').cpu_count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
