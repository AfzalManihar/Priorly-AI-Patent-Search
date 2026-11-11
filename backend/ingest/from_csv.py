"""Build corpus.jsonl + citations.json from four CSV files exported from BigQuery (PatentsView tables).

  python -m ingest.from_csv --raw data/real_raw --out data/real

Expected files in --raw:
  patents.csv        patent_id, patent_number, kind, date, title, abstract, cpc, assignee   (the query patents' area)
  claims.csv         patent_id, claim_sequence, exemplary, claim_text
  citations.csv      patent_id, citation_id                                                 (examiner citations only)
  cited_patents.csv  patent_id, patent_number, kind, date, title, abstract                  (the cited prior art)

Rules:
  - A patent that has claims in claims.csv can be a query patent (claim 1 is the query text).
  - Cited patents have no claims in these exports, so they are indexed by title and abstract only.
  - Only citations whose cited patent exists in the corpus are kept, so recall can be measured.
  - There are no description paragraphs in this data, so passages come from the abstract and the claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LEADING_NUMBER = re.compile(r"^\s*\d+\s*\.\s*")


def display_id(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return f"US {int(raw):,}"
    return f"US {raw}"  # reissue and design patents, for example RE43699, keep their prefix


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_cpc(value: str) -> str:
    first = (value or "").split(",")[0].strip()
    match = re.match(r"^([A-HY]\d{2}[A-Z])\s*(\d+)\s*/\s*(\d+)$", first)
    return f"{match.group(1)} {match.group(2)}/{match.group(3)}" if match else first


def year_of(date: str) -> int:
    return int((date or "0")[:4] or 0)


def build(raw: Path, out: Path) -> dict:
    patents = {r["patent_id"]: r for r in read(raw / "patents.csv")}
    cited_meta = {r["patent_id"]: r for r in read(raw / "cited_patents.csv")}
    claim_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in read(raw / "claims.csv"):
        text = LEADING_NUMBER.sub("", r["claim_text"]).strip()
        if text:
            claim_rows[r["patent_id"]].append((int(r["claim_sequence"] or 0), text))
    # Additional real claims exported separately; metadata is sourced from cited_patents.csv.
    missing_claims_path = raw / "missing_claims_all.csv"
    if missing_claims_path.exists():
        for r in read(missing_claims_path):
            text = LEADING_NUMBER.sub("", r["claim_text"]).strip()
            if text:
                claim_rows[r["patent_id"]].append((int(r["claim_sequence"] or 0), text))
    cites: dict[str, set[str]] = defaultdict(set)
    for r in read(raw / "citations.csv"):
        if r["citation_id"]:
            cites[r["patent_id"]].add(r["citation_id"])

    query_ids = sorted(p for p in claim_rows if p in patents and cites.get(p))
    keep: dict[str, dict] = {}
    for pid in claim_rows:
        if pid in patents:
            keep[pid] = patents[pid]
        elif pid in cited_meta:
            keep[pid] = cited_meta[pid]
    wanted = set().union(*(cites[p] for p in query_ids)) if query_ids else set()
    for cid in wanted:
        if cid in keep:
            continue
        if cid in cited_meta:
            keep[cid] = cited_meta[cid]
        elif cid in patents:
            keep[cid] = patents[cid]

    out.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for pid in sorted(keep):
            info = keep[pid]
            abstract = (info.get("abstract") or "").strip()
            claims = [t for _, t in sorted(claim_rows.get(pid, []))]
            if not abstract and not claims:
                continue
            handle.write(json.dumps({
                "id": display_id(pid), "title": (info.get("title") or "").strip(),
                "assignee": (info.get("assignee") or "").strip(), "year": year_of(info.get("date", "")),
                "cpc": clean_cpc(info.get("cpc", "")), "abstract": abstract, "claims": claims, "paragraphs": [],
            }) + "\n")
            written.add(pid)

    citations = {}
    for pid in query_ids:
        if pid not in written:
            continue
        kept = sorted(display_id(c) for c in cites[pid] if c in written and c != pid)
        if kept:
            citations[display_id(pid)] = kept
    (out / "citations.json").write_text(json.dumps(citations, indent=1), encoding="utf-8")
    return {"patents": len(written), "with_claims": sum(1 for p in written if p in claim_rows),
            "abstract_only": sum(1 for p in written if p not in claim_rows), "query_patents": len(citations),
            "citation_pairs": sum(len(v) for v in citations.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/real_raw"))
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    print(build(args.raw, args.out))


if __name__ == "__main__":
    main()
