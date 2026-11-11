
"""Evaluation harness (PRD section 12).

Runs the pipeline in several configurations over a query set with ground truth
and writes the results file the Benchmark page reads.

Leakage controls: the query patent is excluded from its own results and only
patents from an earlier year than the query patent may be returned.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.corpus import load_corpus
from app.llm import make_llm
from app.pipeline import Engine, Options
from app.rerank import build_scorer
from app.retrievers import BM25Index, Constraints, build_dense
from app.store import Store

from .metrics import (
    bootstrap_ci,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

CONFIGS = [
    ("dense", "Dense only",
     Options(decompose=False, hyde=False, hybrid=False, rerank=False)),
    ("hybrid", "+ BM25 and RRF",
     Options(decompose=False, hyde=False, hybrid=True, rerank=False)),
    ("rerank", "+ Cross-encoder rerank",
     Options(decompose=False, hyde=False, hybrid=True, rerank=True)),
    ("full", "+ Decompose and HyDE",
     Options(decompose=True, hyde=True, hybrid=True, rerank=True)),
]

EXTRA_CONFIGS = [
    ("decompose_only", "Rerank + decomposition only",
     Options(decompose=True, hyde=False, hybrid=True, rerank=True)),
    ("hyde_only", "Rerank + HyDE only",
     Options(decompose=False, hyde=True, hybrid=True, rerank=True)),
]

TYPE_NAMES = {
    "mechanical": "Mechanical claims",
    "chemical": "Chemical and formula queries",
    "software": "Software and protocol claims",
    "exact_id": "Queries containing an exact patent ID",
}


def build_engine(settings: Settings) -> Engine:
    corpus = load_corpus(settings.corpus_path)

    store = Store(
        settings.db_path.with_name("eval_cache.db")
        if settings.llm_backend == "groq"
        else ":memory:"
    )

    return Engine(
        settings,
        corpus,
        BM25Index(corpus),
        build_dense(settings, corpus),
        build_scorer(settings),
        make_llm(settings, store),
    )


def run_config(
    engine: Engine,
    queries: list[dict],
    opts: Options,
    k: int = 10,
) -> list[dict]:
    rows = []

    for q in queries:
        cons = Constraints(
            exclude_ids=frozenset([q["patent_id"]]),
            max_year=q["year"] - 1,
        )

        eligible = {
            pid
            for pid in q["relevant"]
            if pid in engine.corpus.patents
            and engine.corpus.patents[pid].year <= cons.max_year
        }

        if not eligible:
            continue

        result = engine.search(
            q["claim"],
            replace(opts, explain=False),
            k,
            cons,
        )

        ranked = [r["id"] for r in result["results"]]

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "patent_id": q["patent_id"],
            "recall": recall_at_k(ranked, eligible, k),
            "precision": precision_at_k(ranked, eligible, k),
            "rr": reciprocal_rank(ranked, eligible),
            "eligible": sorted(eligible),
            "results": result["results"],
            "degraded": result["degraded"],
        })

    return rows


def summarize(rows: list[dict]) -> dict:
    recalls = [r["recall"] for r in rows]
    lo, hi = bootstrap_ci(recalls)

    return {
        "recall": round(mean(recalls), 4),
        "precision": round(mean([r["precision"] for r in rows]), 4),
        "mrr": round(mean([r["rr"] for r in rows]), 4),
        "recall_ci": [round(lo, 4), round(hi, 4)],
    }


def failure_cases(
    dense_rows: list[dict],
    hybrid_rows: list[dict],
    depth: int = 30,
) -> list[dict]:
    """Relevant patents that dense buried but hybrid retrieved."""

    cases = []

    for row in hybrid_rows:
        for item in row["results"]:
            if (
                item["id"] in row["eligible"]
                and (item["dense"] is None or item["dense"] > depth)
                and item["bm25"]
            ):
                cases.append({
                    "query_id": row["id"],
                    "query_patent": row["patent_id"],
                    "patent": item["id"],
                    "title": item["title"],
                    "dense_rank": item["dense"],
                    "bm25_rank": item["bm25"],
                })

    return cases


def build_query_diagnostics(results: dict) -> list[dict]:
    """Build per-query comparisons without rerunning the pipeline."""

    stage_keys = [
        key for key in ("dense", "hybrid", "rerank")
        if key in results
    ]

    rows_by_stage = {
        key: {row["id"]: row for row in results[key]}
        for key in stage_keys
    }

    query_ids = list(rows_by_stage["dense"].keys())
    diagnostics = []

    for query_id in query_ids:
        base = rows_by_stage["dense"][query_id]

        stage_details = {}

        for key in stage_keys:
            row = rows_by_stage[key].get(query_id)

            if row is None:
                continue

            stage_details[key] = {
                "recall": row["recall"],
                "precision": row["precision"],
                "mrr": row["rr"],
                "retrieved_patents": [
                    {
                        "patent_id": item["id"],
                        "title": item["title"],
                        "dense_rank": item.get("dense"),
                        "bm25_rank": item.get("bm25"),
                    }
                    for item in row["results"]
                ],
                "degraded": row["degraded"],
            }

        diagnostics.append({
            "query_id": query_id,
            "query_type": base["type"],
            "query_patent": base["patent_id"],
            "eligible_patents": base["eligible"],
            "stages": stage_details,
        })

    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["all", "dev", "val", "test"],
        default="all",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="use only the first N queries of the split (0 = all)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="run only dense, hybrid and rerank configurations",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="run only dense and hybrid configurations",
    )
    parser.add_argument(
        "--skip-extra",
        action="store_true",
        help="skip single-layer ablation configurations",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="cutoff for recall, precision and the result list",
    )

    args = parser.parse_args()

    settings = Settings.from_env()

    queries_path = (
        args.queries
        or settings.corpus_path.parent / "queries.json"
    )
    out_path = args.out or settings.benchmark_path

    spec = json.loads(queries_path.read_text(encoding="utf-8"))

    queries = [
        q for q in spec["queries"]
        if args.split == "all" or q.get("split") == args.split
    ]

    if args.limit:
        queries = queries[:args.limit]

    engine = build_engine(settings)

    configs = [
        c for c in CONFIGS
        if not (
            args.skip_llm
            and (c[2].decompose or c[2].hyde)
        )
        and not (
            args.skip_rerank and c[2].rerank
        )
    ]

    extras = (
        []
        if (args.skip_extra or args.skip_llm or args.skip_rerank)
        else EXTRA_CONFIGS
    )

    results = {
        key: run_config(engine, queries, opts, args.k)
        for key, _, opts in configs + extras
    }

    n = len(results["dense"])

    stages = [
        {
            "key": key,
            "name": name,
            **summarize(results[key]),
        }
        for key, name, _ in configs
    ]

    last = configs[-1][0]

    by_type = []

    for qtype in sorted({
        r["type"] for r in results["dense"]
    }):
        pick = lambda key: [
            r["recall"]
            for r in results[key]
            if r["type"] == qtype
        ]

        by_type.append({
            "type": TYPE_NAMES.get(qtype, qtype),
            "n": len(pick("dense")),
            "dense": round(mean(pick("dense")), 4),
            "hybrid": round(mean(pick("hybrid")), 4),
            "full": round(mean(pick(last)), 4),
        })

    report = {
        "dataset_version": spec.get("dataset_version", "unversioned"),
        "sample": bool(spec.get("sample", False)),
        "split": args.split,
        "query_count": n,
        "k": args.k,
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "backends": {
            "dense": settings.dense_backend,
            "rerank": settings.rerank_backend,
            "llm": settings.llm_backend,
        },
        "corpus": {
            "patents": len(engine.corpus),
        },
        "note": (
            "Examiner citations are an incomplete record of relevance, "
            "so these numbers are a lower bound."
            + (
                " SAMPLE DATA: invented patents and a tiny query set. "
                "Do not quote these numbers."
                if spec.get("sample")
                else ""
            )
        ),
        "llm_layers_skipped": bool(
            args.skip_llm or args.skip_rerank
        ),
        "rerank_skipped": bool(args.skip_rerank),
        "stages": stages,
        "ablation_extra": [
            {
                "key": key,
                "name": name,
                **summarize(results[key]),
            }
            for key, name, _ in extras
        ],
        "by_type": by_type,
        "failure_cases": failure_cases(
            results["dense"],
            results["hybrid"],
        ),
        "query_diagnostics": build_query_diagnostics(results),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"{n} queries, split={args.split}, "
        f"backends={report['backends']}"
    )

    for stage in stages:
        print(
            f"  {stage['name']:<28} "
            f"recall@{args.k}={stage['recall']:.2f}  "
            f"precision@{args.k}={stage['precision']:.2f}  "
            f"mrr={stage['mrr']:.2f}"
        )

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
