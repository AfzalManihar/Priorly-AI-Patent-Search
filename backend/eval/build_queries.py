"""Build the evaluation query set from a corpus and its examiner citations (PRD sections 7 and 12).

  python -m eval.build_queries --corpus data/real/corpus.jsonl --citations data/real/citations.json \
        --out data/real/queries.json

Each query is claim 1 of a patent that has examiner citations inside the corpus. Splits are assigned
by a stable hash of the patent id (40% dev, 20% val, 40% test) so they never change between runs.
The query type is a rough heuristic; edit it by hand if you want cleaner categories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ID_IN_TEXT = re.compile(r"\bUS\s?\d{1,2},?\d{3},?\d{3}\b", re.I)
FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
SOFTWARE = re.compile(r"\b(processor|software|network|packet|protocol|memory|server|data|computer|instructions)\b", re.I)


def split_of(patent_id: str) -> str:
    bucket = int(hashlib.sha256(patent_id.encode()).hexdigest(), 16) % 100
    return "dev" if bucket < 40 else "val" if bucket < 60 else "test"


def query_type(claim: str) -> str:
    if ID_IN_TEXT.search(claim):
        return "exact_id"
    if len(FORMULA.findall(claim)) >= 2 or re.search(r"\b(compound|polymer|alloy|solution|electrolyte)\b", claim, re.I):
        return "chemical"
    if len(SOFTWARE.findall(claim)) >= 2:
        return "software"
    return "mechanical"


def build(corpus_path: Path, citations_path: Path, dataset_version: str) -> dict:
    patents = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            patents[raw["id"]] = raw
    cites = json.loads(citations_path.read_text(encoding="utf-8"))
    queries = []
    for pid, cited in sorted(cites.items()):
        patent = patents.get(pid)
        relevant = [c for c in cited if c in patents and c != pid]
        if not patent or not patent["claims"] or not relevant:
            continue
        claim = patent["claims"][0]
        queries.append({"id": f"q{len(queries) + 1}", "patent_id": pid, "year": patent["year"],
                        "type": query_type(claim), "split": split_of(pid), "relevant": relevant, "claim": claim})
    return {"dataset_version": dataset_version, "sample": False, "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--citations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    spec = build(args.corpus, args.citations, args.version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    counts = {s: sum(q["split"] == s for q in spec["queries"]) for s in ("dev", "val", "test")}
    print(f"{len(spec['queries'])} queries {counts} -> {args.out}")


if __name__ == "__main__":
    main()
