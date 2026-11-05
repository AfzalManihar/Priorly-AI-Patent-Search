import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings


def test_search_contract(client, claim):
    r = client.post("/api/search", json={"claim": claim})
    assert r.status_code == 200
    body = r.json()
    assert body["search_id"].startswith("s_") and body["elements"] and body["results"]
    item = body["results"][0]
    assert item["passage"] and item["why"] and item["loc"]


def test_search_is_stored_and_replayable(client, claim):
    sid = client.post("/api/search", json={"claim": claim}).json()["search_id"]
    stored = client.get(f"/api/search/{sid}").json()
    assert stored["response"]["search_id"] == sid and stored["request"]["claim"] == claim
    assert client.get("/api/search/s_missing").status_code == 404


@pytest.mark.parametrize("claim,code", [("too short", "claim_too_short"), ("x" * 8001, "claim_too_long")])
def test_claim_length_errors(client, claim, code):
    r = client.post("/api/search", json={"claim": claim})
    assert r.status_code == 400 and r.json()["error"]["code"] == code


def test_invalid_options_are_422(client, claim):
    r = client.post("/api/search", json={"claim": claim, "options": {"bogus": True}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_options"
    r = client.post("/api/search", json={"claim": claim, "k": 7})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_options"


def test_errors_never_leak_exception_text(client, claim, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(client.app.state.s.engine, "search", boom)
    quiet = TestClient(client.app, raise_server_exceptions=False)  # lifespan state is already set up
    r = quiet.post("/api/search", json={"claim": claim})
    assert r.status_code == 500 and "secret" not in r.text


def test_rate_limit(tmp_path, claim):
    with TestClient(create_app(make_settings(tmp_path, rate_limit_per_min=2))) as c:
        assert c.post("/api/search", json={"claim": claim}).status_code == 200
        assert c.post("/api/search", json={"claim": claim}).status_code == 200
        r = c.post("/api/search", json={"claim": claim})
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited" and "Retry-After" in r.headers


def test_stream_emits_stages_then_result(client, claim):
    with client.stream("POST", "/api/search/stream", json={"claim": claim}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages[0] == "decompose" and "rerank" in stages and events[-1]["type"] == "result"
    assert events[-1]["data"]["results"]


def test_benchmark_endpoint(client):
    body = client.get("/api/benchmark").json()
    assert [s["key"] for s in body["stages"]] == ["dense", "hybrid", "rerank", "full"]
    assert {"recall", "precision", "mrr"} <= set(body["stages"][0]) and body["by_type"]


def test_benchmark_missing_is_404(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, benchmark_path=tmp_path / "nope.json"))) as c:
        assert c.get("/api/benchmark").json()["error"]["code"] == "benchmark_missing"


def test_reports_flow_and_csv_export(client, claim):
    sid = client.post("/api/search", json={"claim": claim}).json()
    first = sid["results"][0]["id"]
    saved = client.post("/api/reports", json={"search_id": sid["search_id"], "title": "Roadway charging", "shortlist": [first]})
    assert saved.status_code == 201
    listing = client.get("/api/reports").json()["reports"]
    assert listing[0]["title"] == "Roadway charging" and listing[0]["short"] == 1
    csv_text = client.get(f"/api/reports/{saved.json()['report_id']}/export.csv").text
    assert csv_text.splitlines()[0].startswith("rank,patent,title") and first in csv_text
    assert client.post("/api/reports", json={"search_id": "s_nope", "title": "x"}).status_code == 404


def test_settings_roundtrip_and_validation(client):
    assert client.get("/api/settings").json()["rrf_k"] == 60
    updated = client.put("/api/settings", json={"rrf_k": 30, "candidates": 40}).json()
    assert updated["rrf_k"] == 30 and updated["candidates"] == 40
    assert client.put("/api/settings", json={"rrf_k": 0}).status_code == 422
    assert client.get("/api/settings").json()["rrf_k"] == 30


def test_health_reports_backends(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["corpus"]["patents"] > 0 and body["benchmark_available"]


def test_frontend_is_served(client):
    assert client.get("/").status_code == 200
    assert "Priorly" in client.get("/app.html").text


def test_csv_export_neutralises_formula_cells(client):
    resp = client.post("/api/search", json={"claim": "A system comprising a transmitter coil in a road surface; a receiver coil on a vehicle; and a controller adjusting power.", "k": 5})
    body = resp.json()
    body["results"][0]["title"] = "=HYPERLINK(\"http://x\")"
    store = client.app.state.s.store
    store.save_search(body["search_id"], {}, body)
    rid = client.post("/api/reports", json={"search_id": body["search_id"], "title": "t", "shortlist": []}).json()["report_id"]
    text = client.get(f"/api/reports/{rid}/export.csv").text
    assert "'=HYPERLINK" in text and ",=HYPERLINK" not in text
