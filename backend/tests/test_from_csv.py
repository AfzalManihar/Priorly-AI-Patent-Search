import csv
import json

from ingest.from_csv import build, display_id


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        w.writerows(rows)


def test_display_id_keeps_reissue_prefix():
    assert display_id("8928182") == "US 8,928,182"
    assert display_id("RE43699") == "US RE43699"


def test_build_from_csv(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    _write(raw / "patents.csv", ["patent_id", "patent_number", "kind", "date", "title", "abstract", "cpc", "assignee"], [
        ["9000001", "9000001", "B2", "2015-04-07", "Coil system", "A coil.", "H02J50/20, H02J50/90", "Acme"],
        ["9000002", "9000002", "B2", "2017-05-01", "No claims here", "Abstract two.", "H02J50/10", ""],
    ])
    _write(raw / "claims.csv", ["patent_id", "claim_sequence", "exemplary", "claim_text"], [
        ["9000001", "10", "FALSE", "10. The system of claim 1, wherein x."],
        ["9000001", "1", "TRUE", "1. A coil system comprising a primary coil and a controller."],
    ])
    _write(raw / "citations.csv", ["patent_id", "citation_id"], [["9000001", "5000001"], ["9000001", "RE43699"], ["9000001", "999"]])
    _write(raw / "cited_patents.csv", ["patent_id", "patent_number", "kind", "date", "title", "abstract"], [
        ["5000001", "5000001", "A", "1991-03-26", "Old coil", "An old coil."],
        ["RE43699", "RE43699", "E", "2012-10-02", "Reissue", "A reissue."],
    ])
    stats = build(raw, out)
    assert stats["patents"] == 3 and stats["with_claims"] == 1 and stats["query_patents"] == 1
    lines = [json.loads(x) for x in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    q = next(x for x in lines if x["id"] == "US 9,000,001")
    assert q["claims"][0].startswith("A coil system")  # claim number stripped, sorted by sequence
    assert q["cpc"] == "H02J 50/20" and q["year"] == 2015
    cites = json.loads((out / "citations.json").read_text(encoding="utf-8"))
    assert cites == {"US 9,000,001": ["US 5,000,001", "US RE43699"]}  # id 999 has no metadata, dropped



def test_build_includes_missing_claims_with_cited_metadata(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()

    _write(raw / "patents.csv",
           ["patent_id", "patent_number", "kind", "date", "title", "abstract", "cpc", "assignee"], [])
    _write(raw / "claims.csv",
           ["patent_id", "claim_sequence", "exemplary", "claim_text"], [])
    _write(raw / "missing_claims_all.csv",
           ["patent_id", "claim_sequence", "exemplary", "claim_text", "publication_number"],
           [["5000001", "1", "TRUE", "1. A real additional patent claim.", "US5000001A"]])
    _write(raw / "citations.csv", ["patent_id", "citation_id"], [])
    _write(raw / "cited_patents.csv",
           ["patent_id", "patent_number", "kind", "date", "title", "abstract"],
           [["5000001", "5000001", "A1", "1991-03-26", "Old coil", "A real abstract."]])

    stats = build(raw, out)
    lines = [json.loads(x) for x in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]

    assert stats["patents"] == 1
    assert stats["with_claims"] == 1
    assert lines[0]["id"] == "US 5,000,001"
    assert lines[0]["claims"] == ["A real additional patent claim."]
