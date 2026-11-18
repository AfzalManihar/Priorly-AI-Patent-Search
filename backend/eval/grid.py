"""Comparison grid: chunking strategy x retrieval x query processing x rerank, measured on real queries.

  python -m eval.grid --queries data/real/queries.json --n 40 --pool 1500
  python -m eval.grid --queries data/real/queries.json --n 40 --with-llm      # adds the HyDE combinations (Groq)
  python -m eval.grid ... --resume                                            # continue an interrupted run

Stage A  every chunking strategy x {BM25, dense, hybrid}        15 combinations, no LLM
Stage B  on the best stage-A chunking, hybrid + each technique   up to 10 combinations
Stage C  hybrid + rerank on every other chunking                  4 combinations
Stage D  RRF constant k = 10, 30, 100 on hybrid                    3 combinations
Stage E  rerank depth 20 and 100 on hybrid + rerank               2 combinations

Every number comes from running the engine. Nothing is typed in. Leakage controls match eval.run_eval:
the query patent is excluded and only earlier patents may be returned. LLM temperature is 0 and every LLM
answer is cached by input hash, so a rerun repeats the same text.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from app.chunkers import CHUNKERS
from app.config import Settings
from app.corpus import Corpus, load_corpus
from app.llm import make_llm
from app.pipeline import Engine, Options
from app.rerank import build_scorer
from app.retrievers import BM25Index, Constraints, build_dense
from app.store import Store

from .metrics import (
    average_precision,
    bootstrap_ci,
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

K = 50  # depth every run returns; the metrics below cut it down
RETRIEVALS = {"bm25": "BM25 only", "dense": "Dense only", "hybrid": "Hybrid (BM25 + dense, RRF)"}
TRANSFORMS = {
    "none": "no query processing",
    "decompose": "claim decomposition",
    "hyde": "HyDE",
    "decompose+hyde": "decomposition + HyDE",
    "prf": "pseudo-relevance feedback",
}
METRICS = ["recall@10", "recall@50", "precision@10", "mrr", "ndcg@10", "map", "hit@10"]
BASELINE_ID = "dense|claim_aware|none|plain"


@dataclass(frozen=True)
class Combo:
    chunking: str
    retrieval: str
    transform: str = "none"
    rerank: bool = False
    stage: str = "A"
    rrf_k: int = 0  # 0 = the configured default; otherwise the RRF constant for this combination (stage D)
    depth: int = 0  # 0 = the configured default; otherwise how many fused candidates the cross-encoder scores (stage E)

    @property
    def id(self) -> str:
        base = f"{self.retrieval}|{self.chunking}|{self.transform}|{'rerank' if self.rerank else 'plain'}"
        return base + (f"|k{self.rrf_k}" if self.rrf_k else "") + (f"|d{self.depth}" if self.depth else "")

    @property
    def needs_llm(self) -> bool:
        return "hyde" in self.transform

    def label(self) -> str:
        parts = [RETRIEVALS[self.retrieval], CHUNKERS[self.chunking][0]]
        if self.transform != "none":
            parts.append(TRANSFORMS[self.transform])
        if self.rerank:
            parts.append("cross-encoder rerank" + (f" top-{self.depth}" if self.depth else ""))
        if self.rrf_k:
            parts.append(f"RRF k={self.rrf_k}")
        return " / ".join(parts)


def stage_a() -> list[Combo]:
    return [Combo(c, r, stage="A") for c in CHUNKERS for r in RETRIEVALS]


def stage_b(best: str) -> list[Combo]:
    return [
        Combo(best, "hybrid", "none", True, "B"),
        Combo(best, "hybrid", "decompose", False, "B"),
        Combo(best, "hybrid", "hyde", False, "B"),
        Combo(best, "hybrid", "decompose+hyde", False, "B"),
        Combo(best, "hybrid", "decompose+hyde", True, "B"),
        Combo(best, "hybrid", "prf", False, "B"),
        Combo(best, "hybrid", "prf", True, "B"),
        Combo(best, "dense", "none", True, "B"),
        Combo(best, "dense", "hyde", False, "B"),
        Combo(best, "bm25", "prf", False, "B"),
    ]


def stage_d(best: str) -> list[Combo]:
    """RRF constant sensitivity: how much does the fusion setting matter? Same indexes, no LLM."""
    return [Combo(best, "hybrid", "none", False, "D", rrf_k=k) for k in (10, 30, 100)]


def stage_e(best: str) -> list[Combo]:
    """Rerank depth: how many fused candidates must the cross-encoder score before the gain stops?"""
    return [Combo(best, "hybrid", "none", True, "E", depth=d) for d in (20, 100)]


def stage_c(best: str) -> list[Combo]:
    return [Combo(c, "hybrid", "none", True, "C") for c in CHUNKERS if c != best]


# ------------------------------------------------------------------ data

def select_queries(path: Path, split: str, n: int, seed: int, patents: dict) -> list[dict]:
    """A seeded, type-balanced sample of queries that have at least one citable, earlier patent in the corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))["queries"]
    usable = []
    for q in raw:
        if split != "all" and q.get("split") != split:
            continue
        eligible = {p for p in q["relevant"] if p in patents and patents[p].year <= q["year"] - 1}
        if eligible:
            usable.append({**q, "relevant": sorted(eligible)})
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in usable:
        by_type[q["type"]].append(q)
    rng = random.Random(seed)
    for group in by_type.values():
        group.sort(key=lambda q: q["id"])
        rng.shuffle(group)
    chosen: list[dict] = []
    total = len(usable)
    for group in by_type.values():
        chosen += group[: max(1, round(n * len(group) / total))]
    leftovers = [q for g in by_type.values() for q in g if q not in chosen]
    rng.shuffle(leftovers)
    chosen += leftovers[: max(0, n - len(chosen))]
    return sorted(chosen[:n], key=lambda q: q["id"])


def build_pool(patents: dict, queries: list[dict], size: int, seed: int) -> list:
    """Every patent the queries need, plus seeded random distractors up to `size`. size 0 keeps the whole corpus."""
    if size <= 0 or size >= len(patents):
        return list(patents.values())
    must = {q["patent_id"] for q in queries if q["patent_id"] in patents} | {p for q in queries for p in q["relevant"]}
    rest = sorted(set(patents) - must)
    random.Random(seed).shuffle(rest)
    keep = must | set(rest[: max(0, size - len(must))])
    return [patents[p] for p in sorted(keep)]


# ------------------------------------------------------------------ query processing

# ------------------------------------------------------------------ running

def make_engine(settings: Settings, patents: list, chunking: str, llm, scorer, pool_tag: str) -> Engine:
    corpus = Corpus(patents, chunker=CHUNKERS[chunking][1])
    url = settings.qdrant_url
    if url != ":memory:" and not url.startswith("http"):
        # local Qdrant locks its folder: never share the live app's folder, and give each chunking its own
        base = url.rstrip("/").rstrip("\\")
        url = f"{base}_eval/{pool_tag}_{chunking}"
    cfg = replace(settings, qdrant_url=url, qdrant_collection=f"{settings.qdrant_collection}_{pool_tag}_{chunking}",
                  stage_budget_s=0, rerank_budget_s=0, explain_budget_s=0)  # measure the full pipeline, never a time-truncated one
    return Engine(cfg, corpus, BM25Index(corpus), build_dense(cfg, corpus), scorer, llm)


def variant(engine: Engine, combo: Combo) -> Engine:
    """The same indexes with a different RRF constant or rerank depth. The copy shares every index and cache."""
    if not (combo.rrf_k or combo.depth):
        return engine
    clone = copy.copy(engine)
    clone.settings = replace(engine.settings, rrf_k=combo.rrf_k or engine.settings.rrf_k,
                             rerank_depth=combo.depth or engine.settings.rerank_depth)
    return clone


def run_combo(engine: Engine, combo: Combo, queries: list[dict]) -> dict:
    engine = variant(engine, combo)
    rows, stage_ms, work = [], Counter(), Counter()
    for q in queries:
        cons = Constraints(exclude_ids=frozenset([q["patent_id"]]), max_year=q["year"] - 1)
        relevant = set(q["relevant"])
        t0 = time.perf_counter()
        opts = Options(
            decompose="decompose" in combo.transform,
            hyde="hyde" in combo.transform,
            hybrid=combo.retrieval in ("hybrid", "bm25"),
            dense=combo.retrieval != "bm25",
            prf=combo.transform == "prf",
            rerank=combo.rerank,
            explain=False,
        )
        result = engine.search(q["claim"], opts, K, cons)
        ranked = [r["id"] for r in result["results"]]
        timings = result.get("timings_ms", {})
        for key, value in result.get("work", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                work[key] += value
        latency = round((time.perf_counter() - t0) * 1000.0, 1)
        stage_ms.update({k: v for k, v in timings.items() if k != "total"})
        first = next((i for i, p in enumerate(ranked, 1) if p in relevant), None)
        rows.append({
            "id": q["id"], "type": q["type"], "latency_ms": latency, "relevant": len(relevant), "first_hit_rank": first,
            "recall@10": recall_at_k(ranked, relevant, 10), "recall@50": recall_at_k(ranked, relevant, 50),
            "precision@10": precision_at_k(ranked, relevant, 10), "mrr": reciprocal_rank(ranked, relevant),
            "ndcg@10": ndcg_at_k(ranked, relevant, 10), "map": average_precision(ranked, relevant, K),
            "hit@10": hit_rate_at_k(ranked, relevant, 10), "top": ranked[:20],
        })
    n = len(rows)
    return summarise(combo, rows, {k: round(v / n, 1) for k, v in stage_ms.items()}, dict(work))


def summarise(combo: Combo, rows: list[dict], stage_ms: dict, work: dict) -> dict:
    metrics = {}
    for name in METRICS:
        values = [r[name] for r in rows]
        lo, hi = bootstrap_ci(values)
        metrics[name] = {"mean": round(mean(values), 4), "ci": [round(lo, 4), round(hi, 4)]}
    lat = [r["latency_ms"] for r in rows]
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = {"queries": len(sub), **{m: round(mean([r[m] for r in sub]), 4) for m in ("recall@10", "mrr", "ndcg@10")}}
    return {
        "id": combo.id, "stage": combo.stage, "label": combo.label(), "chunking": combo.chunking,
        "retrieval": combo.retrieval, "transform": combo.transform, "rerank": combo.rerank,
        "rrf_k": combo.rrf_k or None, "rerank_depth": combo.depth or None,
        "metrics": metrics, "latency_ms": {"mean": round(mean(lat), 1), "p50": round(percentile(lat, 0.5), 1), "p95": round(percentile(lat, 0.95), 1)},
        "stage_ms_mean": stage_ms, "work": work, "by_type": by_type, "per_query": rows,
    }


def add_deltas(done: dict[str, dict]) -> None:
    """Paired comparison of every combination to the baseline on the same queries."""
    base = done.get(BASELINE_ID)
    for combo in done.values():
        combo.pop("vs_baseline", None)
        if not base or combo["id"] == BASELINE_ID:
            continue
        deltas = {}
        for name in ("recall@10", "ndcg@10", "mrr"):
            a = [r[name] for r in combo["per_query"]]
            b = [r[name] for r in base["per_query"]]
            lo, hi = paired_bootstrap_ci(a, b)
            diff = mean(a) - mean(b)
            deltas[name] = {"diff": round(diff, 4), "ci": [round(lo, 4), round(hi, 4)],
                            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"}
        combo["vs_baseline"] = deltas


def pick_best(done: dict[str, dict]) -> dict | None:
    ranked = sorted(done.values(), key=lambda c: (-c["metrics"]["ndcg@10"]["mean"], -c["metrics"]["recall@10"]["mean"], c["latency_ms"]["p95"]))
    if not ranked:
        return None
    top = ranked[0]
    return {
        "combo_id": top["id"], "label": top["label"],
        "options": {"chunking": top["chunking"], "retrieval": top["retrieval"], "dense": top["retrieval"] != "bm25", "hybrid": top["retrieval"] != "dense", "decompose": "decompose" in top["transform"],
                    "hyde": "hyde" in top["transform"], "prf": top["transform"] == "prf", "rerank": top["rerank"]},
        "rule": "highest nDCG@10; ties broken by Recall@10, then by lower p95 latency",
        "metrics": {k: top["metrics"][k] for k in ("ndcg@10", "recall@10", "mrr", "precision@10")},
    }


def compare(done: dict, a_id: str, b_id: str, metric: str = "ndcg@10") -> dict | None:
    """Paired comparison of combination A against combination B on the same queries."""
    a, b = done.get(a_id), done.get(b_id)
    if not a or not b:
        return None
    xs, ys = [r[metric] for r in a["per_query"]], [r[metric] for r in b["per_query"]]
    lo, hi = paired_bootstrap_ci(xs, ys)
    diff = mean(xs) - mean(ys)
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"
    return {"a": a["label"], "b": b["label"], "metric": metric, "a_mean": round(mean(xs), 4), "b_mean": round(mean(ys), 4),
            "diff": round(diff, 4), "ci": [round(lo, 4), round(hi, 4)], "verdict": verdict}


def make_findings(done: dict[str, dict]) -> list[dict]:
    """Plain-language conclusions, each backed by a paired comparison. They are computed, not written by hand."""
    out = []

    def add(kind: str, title: str, cmp: dict | None, why: str) -> None:
        if cmp:
            direction = {"better": "improved", "worse": "lowered", "no clear difference": "made no clear difference to"}[cmp["verdict"]]
            out.append({"kind": kind, "title": title, "comparison": cmp,
                        "text": f"{title}: {cmp['a_mean']:.3f} vs {cmp['b_mean']:.3f} {cmp['metric']} "
                                f"({cmp['diff']:+.3f}, 95% CI {cmp['ci'][0]:+.3f} to {cmp['ci'][1]:+.3f}), so it {direction} the result. {why}"})

    hybrids = [c for c in done.values() if c["stage"] == "A" and c["retrieval"] == "hybrid"]
    if len(hybrids) >= 2:
        order = sorted(hybrids, key=lambda c: -c["metrics"]["ndcg@10"]["mean"])
        best, worst = order[0], order[-1]
        add("chunking", f"Chunking: {CHUNKERS[best['chunking']][0]} vs {CHUNKERS[worst['chunking']][0]} (best vs worst, hybrid)",
            compare(done, best["id"], worst["id"]),
            "Where the interval spans zero, the data cannot rank the two strategies, and the simpler one is the safer default.")
        best_chunking = best["chunking"]
        for r in ("bm25", "dense"):
            add("retrieval", f"Retrieval: hybrid vs {RETRIEVALS[r]} ({CHUNKERS[best_chunking][0]})",
                compare(done, f"hybrid|{best_chunking}|none|plain", f"{r}|{best_chunking}|none|plain"), "Fusion is only worth its cost if this is positive.")
        plain = f"hybrid|{best_chunking}|none|plain"
        for cid, c in done.items():
            if c["stage"] != "B" or c["retrieval"] != "hybrid" or c["chunking"] != best_chunking:
                continue
            if c["transform"] == "none" and c["rerank"]:
                add("rerank", "Cross-encoder rerank added to hybrid", compare(done, cid, plain),
                    "Reranking costs the most time of any stage, so it must earn its place with a measured gain.")
            elif c["transform"] != "none" and not c["rerank"]:
                add("query", f"Query processing: {TRANSFORMS[c['transform']]} added to hybrid", compare(done, cid, plain),
                    "Judged against the same hybrid retrieval without it.")
        for cid, c in done.items():
            if c["chunking"] != best_chunking or c["retrieval"] != "hybrid" or c["transform"] != "none":
                continue
            if c["stage"] == "D":
                add("fusion", f"RRF constant k={c['rrf_k']} vs the default k", compare(done, cid, plain),
                    "RRF is meant to be insensitive to k; a clear difference would mean the default needs tuning.")
            elif c["stage"] == "E":
                add("depth", f"Rerank depth {c['rerank_depth']} vs the default depth", compare(done, cid, f"hybrid|{best_chunking}|none|rerank"),
                    "A deeper rerank costs more time; it is only worth it if recall rises.")
    return out


def regressions(done: dict[str, dict], baseline: dict, tolerance: float) -> list[dict]:
    old = {c["id"]: c for c in baseline.get("combos", [])}
    found = []
    for cid, combo in done.items():
        if cid not in old:
            continue
        for name in ("recall@10", "ndcg@10", "mrr"):
            before, after = old[cid]["metrics"][name]["mean"], combo["metrics"][name]["mean"]
            if before - after > tolerance:
                found.append({"combo": cid, "metric": name, "before": before, "after": after})
    return found


# ------------------------------------------------------------------ main

def signature(settings: Settings, queries: list[dict], pool: int, seed: int) -> str:
    text = json.dumps([[q["id"] for q in queries], pool, seed, settings.dense_backend, settings.rerank_backend, settings.llm_backend,
                       settings.embed_model, settings.rerank_model, settings.groq_model, settings.rrf_k, settings.candidates, settings.rerank_depth])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save(out: Path, meta: dict, done: dict, skipped: list, final: bool) -> None:
    add_deltas(done)
    payload = {"meta": {**meta, "complete": final}, "combos": list(done.values()), "skipped": skipped, "best": pick_best(done), "findings": make_findings(done)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--split", default="dev", help="dev | test | all (default dev, so the test split stays untouched)")
    ap.add_argument("--n", type=int, default=40, help="queries per combination")
    ap.add_argument("--pool", type=int, default=1500, help="patents in the evaluation pool; 0 = whole corpus")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with-llm", action="store_true", help="also run HyDE combinations (needs LLM_BACKEND=groq)")
    ap.add_argument("--only", default="", help="comma list of chunking names to restrict the grid to (smoke tests)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baseline", type=Path, default=None, help="earlier grid.json; exit 1 if any metric drops by more than --tolerance")
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    settings = Settings.from_env()
    out = args.out or settings.corpus_path.parent / "results" / "grid.json"
    full = load_corpus(settings.corpus_path, chunker=CHUNKERS["whole_document"][1])
    queries = select_queries(args.queries, args.split, args.n, args.seed, full.patents)
    if len(queries) < 5:
        raise SystemExit(f"Only {len(queries)} usable queries in split {args.split!r}; need at least 5.")
    pool = build_pool(full.patents, queries, args.pool, args.seed)
    pool_tag = f"p{len(pool)}s{args.seed}"
    sig = signature(settings, queries, args.pool, args.seed)

    store = Store(settings.db_path.with_name("eval_cache.db") if settings.llm_backend == "groq" else ":memory:")
    llm, scorer = make_llm(settings, store), build_scorer(settings)
    llm_ok = args.with_llm and settings.llm_backend == "groq"
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "signature": sig, "queries": len(queries),
        "query_types": dict(Counter(q["type"] for q in queries)), "split": args.split, "seed": args.seed,
        "pool_patents": len(pool), "corpus_patents": len(full.patents), "k": K, "llm_temperature": 0,
        "dense_backend": settings.dense_backend, "rerank_backend": settings.rerank_backend, "llm_backend": settings.llm_backend,
        "embed_model": settings.embed_model, "rerank_model": settings.rerank_model, "llm_model": settings.groq_model,
        "rrf_k": settings.rrf_k, "rerank_depth": settings.rerank_depth, "ground_truth": "examiner-cited prior art only",
        "chunkers": {k: {"label": v[0], "description": v[2]} for k, v in CHUNKERS.items()},
    }
    done: dict[str, dict] = {}
    if args.resume and out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["meta"].get("signature") == sig:
            done = {c["id"]: c for c in old["combos"]}
            print(f"resuming: {len(done)} combinations already measured")
        else:
            print("existing grid.json was made with different settings; starting fresh")

    engines: dict[str, Engine] = {}
    skipped: list[dict] = []
    allowed = set(filter(None, args.only.split(","))) or set(CHUNKERS)

    def engine_for(chunking: str):
        if chunking not in engines:
            t0 = time.perf_counter()
            eng = make_engine(settings, pool, chunking, llm, scorer, pool_tag)
            engines[chunking] = eng
            print(f"  indexed {chunking}: {len(eng.corpus.chunks):,} chunks in {time.perf_counter() - t0:.0f}s", flush=True)
        return engines[chunking]

    def execute(combos: list[Combo]) -> None:
        for combo in combos:
            if combo.chunking not in allowed:
                continue
            if combo.needs_llm and not llm_ok:
                skipped.append({"id": combo.id, "label": combo.label(), "reason": "HyDE needs Groq; run with LLM_BACKEND=groq and --with-llm"})
                continue
            if combo.id in done:
                continue
            t0 = time.perf_counter()
            done[combo.id] = run_combo(engine_for(combo.chunking), combo, queries)
            m = done[combo.id]["metrics"]
            print(f"[{len(done):2d}] {combo.label()}: recall@10 {m['recall@10']['mean']:.3f}  ndcg@10 {m['ndcg@10']['mean']:.3f}  "
                  f"mrr {m['mrr']['mean']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            save(out, meta, done, skipped, False)

    print(f"{len(queries)} queries, pool {len(pool)} of {len(full.patents)} patents, backends {settings.dense_backend}/{settings.rerank_backend}/{settings.llm_backend}")
    execute(stage_a())
    a_hybrid = [c for c in done.values() if c["stage"] == "A" and c["retrieval"] == "hybrid"]
    if a_hybrid:
        best_chunking = max(a_hybrid, key=lambda c: (c["metrics"]["ndcg@10"]["mean"], c["metrics"]["recall@10"]["mean"]))["chunking"]
        meta["best_chunking_stage_a"] = best_chunking
        print(f"best chunking for stage B and C: {best_chunking}")
        execute(stage_b(best_chunking))
        execute(stage_c(best_chunking))
        execute(stage_d(best_chunking))
        execute(stage_e(best_chunking))
    save(out, meta, done, skipped, True)
    best = pick_best(done)
    print(f"\n{len(done)} combinations measured -> {out}")
    if best:
        print(f"best: {best['label']}  ndcg@10 {best['metrics']['ndcg@10']['mean']}  recall@10 {best['metrics']['recall@10']['mean']}")
        (out.parent / "best_config.json").write_text(json.dumps({**best, "source": str(out.name), "created": meta["created"],
                                                                 "queries": meta["queries"], "pool_patents": meta["pool_patents"]}, indent=1), encoding="utf-8")
    if args.baseline and args.baseline.is_file():
        found = regressions(done, json.loads(args.baseline.read_text(encoding="utf-8")), args.tolerance)
        print(f"regression check vs {args.baseline.name}: {len(found)} metric(s) dropped by more than {args.tolerance}")
        for f in found:
            print("  ", f)
        raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    main()