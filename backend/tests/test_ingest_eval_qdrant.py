import csv
import json

from app.config import Settings
from app.corpus import load_corpus
from app.llm import HeuristicLLM
from app.pipeline import Engine, Options
from app.rerank import LexicalScorer
from app.retrievers import BM25Index, QdrantDense, make_embedder
from app.store import Store
from eval.build_queries import build as build_queries, query_type, split_of
from eval.run_eval import CONFIGS, run_config, summarize
from ingest.build_corpus import build as build_corpus, display_id
from tests.conftest import SAMPLE


# ------------------------------------------------------------------ ingest
def _tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def make_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    long = "A wireless power system couples a primary coil with a secondary coil across a gap in the road."
    _tsv(raw / "g_patent.tsv", ["patent_id", "patent_title", "patent_abstract", "patent_date"], [
        ["9000001", "Query patent", "abs", "2016-03-01"], ["8000001", "Cited by examiner", "abs", "2012-01-01"],
        ["8000002", "Cited by applicant", "abs", "2011-01-01"], ["7000001", "Other area", "abs", "2010-01-01"]])
    _tsv(raw / "g_cpc_current.tsv", ["patent_id", "cpc_sequence", "cpc_group"], [
        ["9000001", "0", "H02J50/10"], ["7000001", "0", "A61K9/00"]])
    _tsv(raw / "g_us_patent_citation.tsv", ["patent_id", "citation_sequence", "citation_patent_id", "citation_category"], [
        ["9000001", "1", "8000001", "cited by examiner"], ["9000001", "2", "8000002", "cited by applicant"]])
    _tsv(raw / "g_assignee_disambiguated.tsv", ["patent_id", "disambig_assignee_organization"], [["9000001", "Acme"]])
    _tsv(raw / "g_claims_2016.tsv", ["patent_id", "claim_sequence", "claim_text", "dependent", "claim_number"], [
        ["9000001", "0", "A system comprising: a coil; and a controller.", "", "1"],
        ["8000001", "0", "A coil assembly in a road.", "", "1"], ["8000002", "0", "Applicant claim.", "", "1"]])
    _tsv(raw / "g_detail_desc_text_2016.tsv", ["patent_id", "description_sequence", "description_text"], [
        ["9000001", "0", long + "\n" + long + " Second paragraph here."], ["8000001", "0", long]])
    return raw


def test_build_corpus_uses_only_examiner_citations(tmp_path):
    raw, out = make_raw(tmp_path), tmp_path / "out"
    summary = build_corpus(raw, "H02J 50", out, max_query_patents=10)
    assert summary["query_patents"] == 1
    ids = {json.loads(l)["id"] for l in (out / "corpus.jsonl").read_text().splitlines()}
    assert display_id("8000001") in ids and display_id("8000002") not in ids and display_id("7000001") not in ids
    cites = json.loads((out / "citations.json").read_text())
    assert cites == {display_id("9000001"): [display_id("8000001")]}


def test_build_queries_splits_and_ground_truth(tmp_path):
    raw, out = make_raw(tmp_path), tmp_path / "out"
    build_corpus(raw, "H02J 50", out, 10)
    spec = build_queries(out / "corpus.jsonl", out / "citations.json", "t1")
    q = spec["queries"][0]
    assert q["relevant"] == ["US 8,000,001"] and q["split"] in {"dev", "val", "test"} and q["year"] == 2016


def test_splits_are_stable_and_types_are_sane():
    assert split_of("US 1,000,000") == split_of("US 1,000,000")
    assert query_type("compare with US 9,104,382 B2 in view of") == "exact_id"
    assert query_type("a processor and memory storing instructions") == "software"


# -------------------------------------------------------------------- eval
def test_run_config_enforces_leakage_and_summarizes(settings, queries):
    corpus = load_corpus(settings.corpus_path)
    engine = Engine(settings, corpus, BM25Index(corpus), __import__("app.retrievers", fromlist=["x"]).build_dense(settings, corpus),
                    LexicalScorer(), HeuristicLLM())
    rows = run_config(engine, queries, CONFIGS[0][2])
    assert rows and all(r["patent_id"] not in {x["id"] for x in r["results"]} for r in rows)
    summary = summarize(rows)
    assert 0 <= summary["recall"] <= 1 and summary["recall_ci"][0] <= summary["recall_ci"][1]


def test_benchmark_file_is_flagged_as_sample():
    report = json.loads((SAMPLE / "benchmark.json").read_text())
    assert report["sample"] is True and "SAMPLE DATA" in report["note"]


# ------------------------------------------------------------------ qdrant
def test_qdrant_path_works_in_memory():
    s = Settings.from_env()
    corpus = load_corpus(SAMPLE / "corpus.jsonl")
    dense = QdrantDense(corpus, make_embedder("hash"), ":memory:", "test_collection")
    from app.retrievers import Constraints

    hits = dense.search("transmitter coil embedded in a road surface", 5, Constraints())
    assert hits and all(0 <= idx < len(corpus.chunks) for idx, _ in hits)
    excluded = corpus.chunks[hits[0][0]].patent_id
    again = dense.search("transmitter coil embedded in a road surface", 5, Constraints(exclude_ids=frozenset([excluded])))
    assert all(corpus.chunks[i].patent_id != excluded for i, _ in again)
