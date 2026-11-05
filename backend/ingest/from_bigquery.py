import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(10 * 1024 * 1024)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "real"
SOURCE_FILE = DATA_DIR / "patents.csv"
CITED_FILE = DATA_DIR / "cited_patents.csv"
CLAIMS_FILE = DATA_DIR / "claims.csv"
CITATIONS_FILE = DATA_DIR / "citations.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def patent_display_id(row):
    number = "".join(c for c in row.get("patent_number", "") if c.isdigit())
    return f"US {int(number):,}" if number else None


def main():
    sources = read_csv(SOURCE_FILE)
    cited = read_csv(CITED_FILE)
    claim_rows = read_csv(CLAIMS_FILE)
    citation_rows = read_csv(CITATIONS_FILE)

    patents = {}

    # Source patents
    for row in sources:
        patents[row["patent_id"]] = {
            "row": row,
            "is_source": True
        }

    # Add cited patents without overwriting source metadata
    for row in cited:
        patents.setdefault(row["patent_id"], {
            "row": row,
            "is_source": False
        })

    claims_by_patent = defaultdict(list)
    for row in claim_rows:
        text = (row.get("claim_text") or "").strip()
        if text:
            claims_by_patent[row["patent_id"]].append(
                (int(row.get("claim_sequence") or 0), text)
            )

    for patent_id in claims_by_patent:
        claims_by_patent[patent_id].sort(key=lambda x: x[0])

    corpus = []
    display_ids = {}

    for patent_id, item in patents.items():
        row = item["row"]
        display_id = patent_display_id(row)

        if not display_id:
            continue

        display_ids[patent_id] = display_id
        date = row.get("date", "")

        record = {
            "id": display_id,
            "title": (row.get("title") or "").strip(),
            "assignee": row.get("assignee") or "",
            "year": int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None,
            "cpc": row.get("cpc") or "",
            "abstract": (row.get("abstract") or "").strip(),
            "claims": [
                text for _, text in claims_by_patent.get(patent_id, [])
            ],
            "paragraphs": []
        }
        corpus.append(record)

    # Build citation graph, keeping only targets present in the corpus
    citation_map = defaultdict(set)

    for row in citation_rows:
        source_id = row["patent_id"]
        target_id = row["citation_id"]

        if source_id in display_ids and target_id in display_ids:
            citation_map[display_ids[source_id]].add(display_ids[target_id])

    citations = {
        source: sorted(targets)
        for source, targets in citation_map.items()
    }

    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for record in corpus:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "citations.json", "w", encoding="utf-8") as f:
        json.dump(citations, f, ensure_ascii=False, indent=2)

    source_without_claims = sum(
        1 for row in sources if not claims_by_patent.get(row["patent_id"])
    )

    print("CORPUS BUILD COMPLETE")
    print("Source patents:", len(sources))
    print("Cited patent records:", len(cited))
    print("Total unique patents in corpus:", len(corpus))
    print("Patents with claims:", len(claims_by_patent))
    print("Source patents without claims:", source_without_claims)
    print("Citation source patents:", len(citations))
    print("Citation edges retained:", sum(map(len, citations.values())))
    print("Output: data/real/corpus.jsonl")
    print("Output: data/real/citations.json")


if __name__ == "__main__":
    main()



