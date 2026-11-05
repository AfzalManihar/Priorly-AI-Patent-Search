# Priorly: patent prior-art search engine

Paste a patent claim. Priorly splits it into elements, searches by keyword and by meaning, merges the
results, reranks them, and shows each hit with its matching passage, the elements it covers, and a
trace of how it was ranked. Every retrieval layer is measured against prior art that patent examiners
actually cited.

Author: Afzal Manihar. The requirements live in the PRD (`docs/PRD.md`, text only). The full PRD with
its diagrams and charts is the published document; its section 17 maps every requirement to code and tests,
and section 20 lists what is built and what is not. This repository implements it.

## Run it in two minutes

```bash
make setup          # pip install -r backend/requirements-dev.txt
make serve          # API + UI at http://localhost:8000
```

Open http://localhost:8000. The landing page is `index.html`, the app is `app.html`.
No API keys, no model downloads: the default backends are offline stand-ins (see "Backends").

```bash
make test           # 60 tests
make eval           # re-run the benchmark on the configured corpus
docker compose up --build    # same thing in containers, plus a Qdrant service
```

## What is here

```
priorly/
  frontend/                 landing page + workspace (plain HTML, CSS, JS, no build step)
    js/api.js               API client (search stream, reports, benchmark, settings)
    js/app.js               UI logic; falls back to built-in demo data if the API is down
  backend/
    app/                    FastAPI service and pipeline
      pipeline.py           decompose, HyDE, retrieve, fuse, rerank, explain (PRD section 8)
      retrievers.py         BM25, TF-IDF stand-in, Qdrant + BGE dense retriever
      fusion.py             Reciprocal Rank Fusion
      rerank.py             lexical scorer, BGE cross-encoder
      llm.py                rule-based LLM stand-in, Groq client with SQLite cache
      highlight.py          HTML-safe passage highlighting per claim element
      main.py               endpoints, error model, rate limit, SSE stream, CSV export
      store.py              SQLite: search log, reports, settings, LLM cache
    ingest/
      make_sample_data.py   bundled fictional sample corpus and queries
      download.py           PatentsView bulk download
      build_corpus.py       PatentsView tables to corpus.jsonl + examiner citations
    eval/
      build_queries.py      query set with dev/val/test splits and ground truth
      run_eval.py           benchmark harness (writes the file the Benchmark page reads)
      metrics.py            Recall@k, Precision@k, MRR, bootstrap CI
    tests/                  60 tests (units, pipeline, API contract, ingest, eval, Qdrant path, production behaviour)
  docker-compose.yml  Makefile  .env.example  .github/workflows/ci.yml
```

## Backends

| Setting | Default (offline) | Real configuration |
| --- | --- | --- |
| `DENSE_BACKEND` | `tfidf`: lexical stand-in, no download | `bge`: BGE embeddings stored in Qdrant |
| `RERANK_BACKEND` | `lexical`: token overlap scorer | `bge`: BGE cross-encoder |
| `LLM_BACKEND` | `heuristic`: rule-based decomposition, templated reasons | `groq`: decomposition, HyDE, grounded explanations |

The defaults let everything run anywhere, but they are not semantic. The paraphrase advantage of
vector search, and the value of HyDE and a cross-encoder, only show up with the real backends:

```bash
make setup-ml                              # installs sentence-transformers
cp .env.example .env                       # then edit
#   DENSE_BACKEND=bge  RERANK_BACKEND=bge  LLM_BACKEND=groq  GROQ_API_KEY=...
#   QDRANT_URL=http://localhost:6333       (docker run -p 6333:6333 qdrant/qdrant)
```

## Using real data (PRD section 7)

```bash
cd backend
python -m ingest.download --years 2015 2016 2017 --out data/raw
python -m ingest.build_corpus --raw data/raw --check          # print table headers, verify column names
python -m ingest.build_corpus --raw data/raw --cpc "H02J 50" --out data/real --max-query-patents 300
python -m eval.build_queries --corpus data/real/corpus.jsonl --citations data/real/citations.json \
        --out data/real/queries.json --version v1
export CORPUS_PATH=data/real/corpus.jsonl BENCHMARK_PATH=data/real/benchmark.json
python -m eval.run_eval --queries data/real/queries.json --split test
```

Leakage controls are built into the harness: a query patent never appears in its own results, and only
patents from an earlier year can be returned. Tune on the `dev` split, choose on `val`, and report `test`
once per release. Examiner citations are an incomplete record of relevance, so every number is a lower bound.

## What is verified, and what is not

Verified here (by the test suite and a real-browser run): the full pipeline on the sample corpus, every
API endpoint and error code, stage switches, leakage controls, degraded-run behaviour, rate limiting, SSE
streaming, reports and CSV export, settings, the Qdrant code path (in memory, with a hash embedder),
the ingest and query-building scripts on a small synthetic PatentsView-style fixture, and the UI in
Chromium at desktop and phone widths in light and dark themes.

Not verified, so check these before relying on them:

- **Sample numbers mean nothing.** The bundled corpus is 31 invented patents and 6 queries. Even the
  benchmark file says "SAMPLE DATA". The Benchmark page shows real lift only after you run the harness on real data.
- **Landing page numbers are placeholders.** The bars on `index.html` come from `frontend/js/data.js`.
  Replace them with your measured results before showing the page to anyone.
- **`ingest/download.py` and the PatentsView column names** follow the published layout as remembered
  when this project was generated. They were not run against the live site. Use `--check` and edit `COLUMNS`
  in `build_corpus.py` if the headers differ.
- **BGE embeddings, the BGE cross-encoder, and Groq** were not run (no model downloads or API key here).
  The Groq client is tested with a mocked transport; the BGE classes are thin wrappers over sentence-transformers.
- **Latency and accessibility targets** from the PRD (p95 under 8 s, audit score 95) are not measured against real data.
- **The container image** was not built here (no Docker daemon was available). The Dockerfile and compose file follow the same steps as `make setup` and `make serve`, which were run.

## PRD coverage

Every requirement in the PRD (FR-01 to FR-23, NFR-01 to NFR-14) has an implementation. Section 17 of the PRD
maps each one to its module and test. The items that need a measurement before they can be called done:

| PRD item | State |
| --- | --- |
| FR-14 report export | CSV and PDF, both built and tested |
| FR-19 benchmark CSV export | Built in the UI (downloads the stage table) |
| NFR-08 delete stored claims | `DELETE /api/search/{id}` and `DELETE /api/data?confirm=yes`, plus an Erase button in Settings |
| NFR-09 observability | Per-search timings and LLM cache hit rate in the log line; `GET /api/metrics` for latency percentiles, errors and cache |
| NFR-01 and NFR-02 latency | Measured per stage by `/api/metrics`; the target itself is unmeasured until the real corpus is loaded |
| NFR-13 accessibility | Built to the standard; the automated audit score is not measured |
| Passage location | Paragraph references (`Paragraph [0042]`), as the PRD specifies |
| Model and key settings | Editable in the UI; model changes apply after an API restart |

## Operating it

| Concern | How |
| --- | --- |
| Health | `GET /api/health` (container health check uses it) |
| Metrics | `GET /api/metrics`: searches, degraded runs, p50 and p95 latency, per-stage p95, errors by status, LLM cache hit rate, stored counts |
| Logs | One line per request (id, path, status, ms) and one per search (timings, degraded stages, cache). Set `LOG_LEVEL`. Send `X-Request-ID` to trace a call end to end |
| Security headers | CSP, frame denial, no sniffing, no referrer on every response |
| Privacy | Claims are stored for replay and reports only. Erase one search, or everything, at any time. The provider key is never returned by any endpoint |
| Containers | Unprivileged user, health check, restart policy, named volumes for the database and vector index |
| CI | `.github/workflows/ci.yml` runs the tests on Python 3.11 and 3.12 and syntax-checks the front end. `make check` does the same locally |
| Behind a proxy | The container starts uvicorn with `--proxy-headers`; put TLS in front of it and set `CORS_ORIGINS` only if the front end is hosted elsewhere |

## API summary (PRD section 10)

`POST /api/search`, `POST /api/search/stream` (server-sent events), `GET|DELETE /api/search/{id}`,
`DELETE /api/data?confirm=yes`, `GET /api/benchmark`, `GET|POST /api/reports`,
`GET /api/reports/{id}/export.csv`, `GET /api/reports/{id}/export.pdf`, `GET|PUT /api/settings`,
`GET /api/health`, `GET /api/metrics`. Interactive docs at `/docs` while the server runs.
Result fields: `id, title, assignee, year, cpc, el, dense, bm25, rrf_rank, rerank_rank, score,
score_kind, loc, passage, why, missed_by_vector`.
