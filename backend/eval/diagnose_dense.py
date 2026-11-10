import json
from pathlib import Path

from app.config import Settings
from app.retrievers import Constraints, aggregate_to_patents
from eval.run_eval import build_engine

settings = Settings.from_env()
spec = json.loads(Path("data/real/queries.json").read_text(encoding="utf-8"))
queries = [q for q in spec["queries"] if q.get("split") == "dev"][:5]

engine = build_engine(settings)

try:
    for q in queries:
        cons = Constraints(
            exclude_ids=frozenset([q["patent_id"]]),
            max_year=q["year"] - 1,
        )

        hits = engine.dense.search(q["claim"], settings.trace_depth, cons)
        ranked = aggregate_to_patents(engine.corpus, hits)

        best_scores = {}
        chunk_counts = {}

        for idx, score in hits:
            pid = engine.corpus.chunks[idx].patent_id
            best_scores[pid] = max(best_scores.get(pid, float("-inf")), score)
            chunk_counts[pid] = chunk_counts.get(pid, 0) + 1

        print(f"\n{'=' * 65}")
        print(f"{q['id']} | {q['patent_id']} | year={q['year']}")
        print(f"Raw chunks: {len(hits)} | Unique patents: {len(ranked)}")

        print("\nTOP 10 PATENTS:")
        for rank, (pid, score, chunks) in enumerate(ranked[:10], 1):
            print(f"{rank:2}. {pid} | score={score:.4f} | chunks={len(chunks)}")

        print("\nRELEVANT PATENTS:")
        for pid in q["relevant"]:
            if pid in best_scores:
                rank = next(i for i, item in enumerate(ranked, 1) if item[0] == pid)
                print(f"{pid} | rank={rank} | score={best_scores[pid]:.4f} | chunks={chunk_counts[pid]}")
            else:
                print(f"{pid} | NOT IN TOP {settings.trace_depth}")

finally:
    client = getattr(engine.dense, "client", None)
    if client is not None:
        client.close()
