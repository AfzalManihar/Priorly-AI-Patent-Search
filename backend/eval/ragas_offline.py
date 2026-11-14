"""Offline RAGAS ID-based scoring on a saved Priorly benchmark JSON.

- Does NOT run retrieval, does NOT call any LLM, uses NO Groq tokens.
- Reads per-query records that hold `eligible_patents` and `stages` (each stage has an
  ordered `retrieved_patents` list), re-computes Recall@K / Precision@K / MRR with the
  project's own definition, scores the same lists with RAGAS IDBasedContextRecall and
  IDBasedContextPrecision, and prints both side by side.

Usage (from the backend folder):
    python eval/ragas_offline.py data/real/bench_test40.json [--k 10] [--out data/real/ragas_test40.json]
"""
import argparse
import asyncio
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
except Exception as exc:  # pragma: no cover
    sys.exit(f"RAGAS import failed: {type(exc).__name__}: {exc}\nInstall with: pip install ragas")


def find_records(node, out):
    """Collect every dict that has both `eligible_patents` and `stages`."""
    if isinstance(node, dict):
        if "eligible_patents" in node and "stages" in node:
            out.append(node)
            return
        for value in node.values():
            find_records(value, out)
    elif isinstance(node, list):
        for value in node:
            find_records(value, out)


def pid(item):
    return item["patent_id"] if isinstance(item, dict) else item


def custom_metrics(top, eligible):
    hits = [p for p in top if p in eligible]
    recall = len(set(hits)) / len(eligible)
    precision = len(hits) / len(top) if top else 0.0
    mrr = 0.0
    for rank, p in enumerate(top, start=1):
        if p in eligible:
            mrr = 1.0 / rank
            break
    return recall, precision, mrr


async def main_async(args):
    with open(args.file, encoding="utf-8") as fh:
        data = json.load(fh)

    records = []
    find_records(data, records)
    if not records:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        sys.exit(
            "No per-query records with 'eligible_patents' + 'stages' found.\n"
            f"Top-level keys/type: {keys}\n"
            "RAGAS scoring needs the saved retrieved lists; do not rerun the test split."
        )

    stored_agg = {}
    if isinstance(data, dict) and isinstance(data.get("stages"), list):
        for s in data["stages"]:
            stored_agg[s.get("key") or s.get("name")] = s.get("recall")

    rag_recall = IDBasedContextRecall()
    rag_prec = IDBasedContextPrecision()
    acc = {}
    skipped = 0

    for rec in records:
        eligible = set(rec["eligible_patents"])
        if not eligible:
            skipped += 1
            continue
        for stage_key, stage in rec["stages"].items():
            top = [pid(x) for x in stage.get("retrieved_patents", [])][: args.k]
            c_rec, c_prec, c_mrr = custom_metrics(top, eligible)
            sample = SingleTurnSample(
                retrieved_context_ids=top, reference_context_ids=sorted(eligible)
            )
            r_rec = await rag_recall.single_turn_ascore(sample)
            r_prec = await rag_prec.single_turn_ascore(sample)

            a = acc.setdefault(
                stage_key,
                {"n": 0, "c_rec": 0.0, "r_rec": 0.0, "c_prec": 0.0, "r_prec": 0.0,
                 "c_mrr": 0.0, "max_diff": 0.0, "stored_mismatch": 0, "stored_seen": 0},
            )
            a["n"] += 1
            a["c_rec"] += c_rec
            a["r_rec"] += r_rec
            a["c_prec"] += c_prec
            a["r_prec"] += r_prec
            a["c_mrr"] += c_mrr
            a["max_diff"] = max(a["max_diff"], abs(c_rec - r_rec), abs(c_prec - r_prec))
            if "recall" in stage:
                a["stored_seen"] += 1
                if abs(stage["recall"] - c_rec) > 1e-3:
                    a["stored_mismatch"] += 1

    print(f"file={args.file}  k={args.k}  queries scored={len(records) - skipped}  skipped(no eligible)={skipped}")
    header = (f"{'stage':<10}{'n':>4}{'custom R@k':>12}{'RAGAS R':>10}{'custom P@k':>12}"
              f"{'RAGAS P':>10}{'custom MRR':>12}{'max|diff|':>11}{'stored!=recomp':>16}")
    print(header)
    result = {}
    for key, a in acc.items():
        n = a["n"]
        row = {
            "n": n,
            "custom_recall": a["c_rec"] / n, "ragas_id_recall": a["r_rec"] / n,
            "custom_precision": a["c_prec"] / n, "ragas_id_precision": a["r_prec"] / n,
            "custom_mrr": a["c_mrr"] / n, "max_abs_diff": a["max_diff"],
            "stored_vs_recomputed_mismatches": f"{a['stored_mismatch']}/{a['stored_seen']}",
            "stored_aggregate_recall": stored_agg.get(key),
        }
        result[key] = row
        print(f"{key:<10}{n:>4}{row['custom_recall']:>12.4f}{row['ragas_id_recall']:>10.4f}"
              f"{row['custom_precision']:>12.4f}{row['ragas_id_precision']:>10.4f}"
              f"{row['custom_mrr']:>12.4f}{row['max_abs_diff']:>11.2e}{row['stored_vs_recomputed_mismatches']:>16}")
    print("stored aggregate recall (from file):", {k: v for k, v in stored_agg.items()})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"k": args.k, "source": args.file, "stages": result}, fh, indent=2)
        print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
