"""Production behaviour: PDF export, erasing stored data, metrics, security headers (PRD NFR-06 to NFR-09, FR-14)."""
import io
import re


def _report(client, claim, title="Roadway charging", shortlist=True):
    run = client.post("/api/search", json={"claim": claim}).json()
    chosen = [run["results"][0]["id"]] if shortlist else []
    rid = client.post("/api/reports", json={"search_id": run["search_id"], "title": title, "shortlist": chosen}).json()["report_id"]
    return run, rid


def test_pdf_export_is_a_real_pdf_with_the_report_content(client, claim):
    run, rid = _report(client, claim)
    resp = client.get(f"/api/reports/{rid}/export.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert 'filename="priorly-Roadway-charging.pdf"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF") and resp.content.rstrip().endswith(b"%%EOF")
    assert len(resp.content) > 1500


def test_pdf_export_survives_markup_and_non_latin_text(client, claim):
    run, rid = _report(client, claim, title="<b>Report</b> 日本語 & more")
    resp = client.get(f"/api/reports/{rid}/export.pdf")
    assert resp.status_code == 200 and resp.content.startswith(b"%PDF")


def test_pdf_export_unknown_report_is_404(client):
    resp = client.get("/api/reports/r_nope/export.pdf")
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "not_found"


def test_deleting_a_search_removes_its_reports(client, claim):
    run, rid = _report(client, claim)
    resp = client.delete(f"/api/search/{run['search_id']}")
    assert resp.status_code == 200 and resp.json()["deleted"] == {"searches": 1, "reports": 1}
    assert client.get(f"/api/search/{run['search_id']}").status_code == 404
    assert client.get("/api/reports").json()["reports"] == []
    assert client.get(f"/api/reports/{rid}/export.csv").status_code == 404
    assert client.delete(f"/api/search/{run['search_id']}").status_code == 404


def test_erase_all_needs_explicit_confirmation(client, claim):
    _report(client, claim)
    refused = client.delete("/api/data")
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "confirm_required"
    assert client.get("/api/reports").json()["reports"] != []
    done = client.delete("/api/data?confirm=yes").json()["deleted"]
    assert done["searches"] >= 1 and done["reports"] == 1
    assert client.get("/api/reports").json()["reports"] == []
    assert client.get("/api/metrics").json()["stored"] == {"searches": 0, "reports": 0}


def test_erase_all_keeps_settings(client):
    client.put("/api/settings", json={"rrf_k": 42})
    client.delete("/api/data?confirm=yes")
    assert client.get("/api/settings").json()["rrf_k"] == 42


def test_metrics_report_searches_latency_and_cache_hit_rate(client, claim):
    assert client.get("/api/metrics").json()["searches"] == 0
    client.post("/api/search", json={"claim": claim})
    client.post("/api/search", json={"claim": "x"})  # rejected: too short
    m = client.get("/api/metrics").json()
    assert m["searches"] == 1
    assert m["errors_by_status"].get("400") == 1
    assert m["latency_ms"]["count"] == 1 and m["latency_ms"]["p95"] is not None
    assert set(m["llm_cache"]) == {"hits", "misses", "hit_rate"}
    assert m["stored"]["searches"] == 1


def test_security_headers_and_request_id_on_every_response(client):
    for path in ("/api/health", "/api/nope", "/"):
        resp = client.get(path)
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert re.fullmatch(r"[0-9a-f]{12}", resp.headers["x-request-id"])
    assert client.get("/api/health", headers={"x-request-id": "trace-me-1"}).headers["x-request-id"] == "trace-me-1"


def test_api_key_is_never_returned_by_settings_or_health(client, settings):
    settings.groq_api_key = "super-secret-key-123"
    for path in ("/api/settings", "/api/health", "/api/metrics"):
        assert "super-secret-key-123" not in client.get(path).text


def test_pdf_text_maps_typographic_and_greek_characters_instead_of_question_marks():
    from app.report_pdf import _drawable
    assert _drawable("a tran… α-phase “quoted” ≥ 5") == 'a tran... alpha-phase "quoted" >= 5'
    assert _drawable("café µm") == "café µm"


def test_pdf_names_the_backends_and_warns_when_they_are_offline(client, claim):
    from app.report_pdf import backend_notes
    run = client.post("/api/search", json={"claim": claim}).json()
    notes = backend_notes(run)
    assert "vector tfidf" in notes[0] and "rerank lexical" in notes[0]
    assert "Offline stand-in" in notes[1]
    assert len(backend_notes({**run, "retrieval": {"dense_backend": "bge", "rerank_backend": "bge", "llm_backend": "groq"}})) == 1


def test_offline_reasons_do_not_claim_an_element_is_absent(client, claim):
    reasons = " ".join(r["why"] for r in client.post("/api/search", json={"claim": claim}).json()["results"])
    assert "Does not clearly cover" not in reasons and "No claim element is clearly covered" not in reasons


def test_explain_off_gives_same_ranking_and_no_reasons(client):
    from dataclasses import replace
    engine = client.app.state.s.engine
    from app.pipeline import Options
    claim = "A wireless charging system comprising a primary coil and a pickup coil and a controller that shifts frequency."
    on = engine.search(claim, Options(), 5)["results"]
    off = engine.search(claim, replace(Options(), explain=False), 5)["results"]
    assert [r["id"] for r in on] == [r["id"] for r in off]
    assert all(r["why"] == "" for r in off)
