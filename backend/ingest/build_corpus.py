"""Build corpus.jsonl and citations.json from PatentsView tables (PRD section 7).

  python -m ingest.build_corpus --raw data/raw --cpc "H02J 50" --out data/real --max-query-patents 300
  python -m ingest.build_corpus --raw data/raw --check      # print each table's header and exit

Steps
  1. Select patents whose CPC group starts with --cpc (the technology area).
  2. Collect the references cited by the EXAMINER on those patents (applicant citations are ignored).
  3. Corpus = selected patents + every cited patent that has text in the tables. Cited patents must be in
     the corpus or recall cannot be measured.
  4. Write corpus.jsonl (one patent per line) and citations.json (examiner citations per selected patent).

IMPORTANT: column names below follow the PatentsView data dictionary as remembered when this project
was generated. Verify them with --check against your download and edit COLUMNS if they differ.
Description text in these tables has no [0042]-style paragraph numbers, so paragraphs are numbered in
order of appearance and shown as "Paragraph [n]".
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

COLUMNS = {
    "patent": {"file": "g_patent.tsv", "id": "patent_id", "title": "patent_title", "abstract": "patent_abstract",
               "date": "patent_date"},
    "cpc": {"file": "g_cpc_current.tsv", "id": "patent_id", "group": "cpc_group"},
    "citation": {"file": "g_us_patent_citation.tsv", "id": "patent_id", "cited": "citation_patent_id",
                 "category": "citation_category"},
    "assignee": {"file": "g_assignee_disambiguated.tsv", "id": "patent_id", "name": "disambig_assignee_organization"},
    "claims": {"glob": "g_claims_*.tsv", "id": "patent_id", "text": "claim_text", "number": "claim_number",
               "sequence": "claim_sequence"},
    "desc": {"glob": "g_detail_desc_text_*.tsv", "id": "patent_id", "text": "description_text",
             "sequence": "description_sequence"},
}
EXAMINER_MARK = "examiner"


def display_id(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    return f"US {int(digits):,}" if digits else raw


def rows(path: Path):
    with open(path, encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def files(raw: Path, spec: dict) -> list[Path]:
    return sorted(raw.glob(spec["glob"])) if "glob" in spec else [raw / spec["file"]]


def paragraphs_from(text: str, limit: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n+|\\n", text) if len(p.strip()) > 40]
    return parts[:limit]


def build(raw: Path, cpc_prefix: str, out: Path, max_query_patents: int, max_paragraphs: int = 60) -> dict:
    c = COLUMNS
    in_area = set()
    cpc_of: dict[str, str] = {}
    for r in rows(raw / c["cpc"]["file"]):
        group = r[c["cpc"]["group"]]
        if group.startswith(cpc_prefix.replace(" ", "")) or group.startswith(cpc_prefix):
            in_area.add(r[c["cpc"]["id"]])
            cpc_of.setdefault(r[c["cpc"]["id"]], group)
    patent_info = {}
    for r in rows(raw / c["patent"]["file"]):
        pid = r[c["patent"]["id"]]
        patent_info[pid] = r
    in_area &= set(patent_info)

    citations: dict[str, list[str]] = defaultdict(list)
    for r in rows(raw / c["citation"]["file"]):
        pid = r[c["citation"]["id"]]
        if pid in in_area and EXAMINER_MARK in (r.get(c["citation"]["category"]) or "").lower():
            cited = r[c["citation"]["cited"]]
            if cited:
                citations[pid].append(cited)

    queries = sorted(p for p in in_area if citations.get(p))[:max_query_patents]
    keep = set(in_area)
    for pid in queries:
        keep.update(x for x in citations[pid] if x in patent_info)

    assignee = {r[c["assignee"]["id"]]: r[c["assignee"]["name"]] for r in rows(raw / c["assignee"]["file"])
                if r[c["assignee"]["id"]] in keep} if (raw / c["assignee"]["file"]).exists() else {}
    claims: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path in files(raw, c["claims"]):
        for r in rows(path):
            if r[c["claims"]["id"]] in keep:
                claims[r[c["claims"]["id"]]].append((int(r[c["claims"]["sequence"]] or 0), r[c["claims"]["text"]]))
    desc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path in files(raw, c["desc"]):
        for r in rows(path):
            if r[c["desc"]["id"]] in keep:
                desc[r[c["desc"]["id"]]].append((int(r[c["desc"]["sequence"]] or 0), r[c["desc"]["text"]]))

    out.mkdir(parents=True, exist_ok=True)
    written, dropped = set(), 0
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for pid in sorted(keep):
            info = patent_info[pid]
            claim_list = [t for _, t in sorted(claims.get(pid, []))]
            if not claim_list:
                dropped += 1
                continue
            text = "\n".join(t for _, t in sorted(desc.get(pid, [])))
            handle.write(json.dumps({
                "id": display_id(pid), "title": info[c["patent"]["title"]], "assignee": assignee.get(pid, ""),
                "year": int((info.get(c["patent"]["date"]) or "0")[:4] or 0), "cpc": cpc_of.get(pid, ""),
                "abstract": info.get(c["patent"]["abstract"], ""), "claims": claim_list,
                "paragraphs": [{"no": f"{i:04d}", "text": p}
                               for i, p in enumerate(paragraphs_from(text, max_paragraphs), start=1)],
            }) + "\n")
            written.add(pid)
    cites = {display_id(p): [display_id(x) for x in citations[p] if x in written] for p in queries if p in written}
    (out / "citations.json").write_text(json.dumps(cites, indent=1), encoding="utf-8")
    return {"patents": len(written), "dropped_without_claims": dropped, "query_patents": len(cites)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--cpc", default="H02J 50")
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    parser.add_argument("--max-query-patents", type=int, default=300)
    parser.add_argument("--max-paragraphs", type=int, default=60)
    parser.add_argument("--check", action="store_true", help="print table headers and exit")
    args = parser.parse_args()
    if args.check:
        for name, spec in COLUMNS.items():
            for path in files(args.raw, spec)[:1]:
                header = next(rows(path), None)
                print(f"{name:9} {path.name}: {list(header) if header else 'EMPTY'}")
        return
    print(build(args.raw, args.cpc, args.out, args.max_query_patents, args.max_paragraphs))


if __name__ == "__main__":
    main()
