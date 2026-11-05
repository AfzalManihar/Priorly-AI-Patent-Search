# Priorly: Patent Prior-Art Search Engine, Product Requirements Document

Author: Afzal Manihar · Version 1.0 · Nov 2, 2026

## 1. Product overview

Priorly takes a patent claim and returns a ranked list of prior-art documents, each with the matching passage, the claim elements it covers, and a trace showing how it was ranked.

| Field | Value |
| --- | --- |
| Product | Priorly, a patent prior-art search engine |
| Author | Afzal Manihar |
| Version | 1.0 |
| Status | Implemented; build status in section 20 |
| Document type | Product requirements document |
| Audience | Engineering, design, evaluation, reviewers |

**Vision.** A patent professional pastes a claim and quickly gets a defensible, evidence-backed shortlist of the references most likely to affect its validity, with every ranking decision open to inspection.

**Product summary.** Priorly is a multi-stage retrieval system. An LLM decomposes a claim into atomic elements. Each element is expanded with a hypothetical passage (HyDE). Keyword search (BM25) and vector search run in parallel. Results are merged with Reciprocal Rank Fusion (RRF), and a cross-encoder reranks the shortlist. Each layer is measured against prior art that patent examiners actually cited, so the value of every stage is shown with numbers.

**Deliverables.**

- Web application: marketing landing page and an authenticated-style search workspace.
- Backend API that runs the retrieval pipeline.
- Ingestion pipeline that builds the corpus and both indexes.
- Evaluation harness and benchmark report that feed the Benchmark page.

## 2. Problem and opportunity

A single missed reference can invalidate a patent opinion, yet the tools that find references force a trade-off between exact wording and meaning.

**The problem.** Patent claims are long, legalistic sentences made of several independent ideas. A prior-art document rarely discloses the whole claim; it usually discloses one or two elements. Searching with the full claim text returns noisy results, and no single retrieval method handles all the ways a relevant document can look.

**Where current approaches fail**

- Keyword search misses paraphrase. A document that says "magnetic coupling element" is not found by a query for "inductive coil".
- Vector search misses exact matches. Patent numbers, chemical formulae, and rare technical terms are blurred by embeddings, and a document that uses the claim's own phrase can sit far down the list.
- Claim language differs from description language. Words like "comprising" and "wherein" do not appear in the passages that actually disclose the idea.
- Results are opaque. A ranked list without the matching passage, the elements covered, or a ranking trace cannot be defended in front of a client or a partner.
- Most pipelines add retrieval layers by intuition and never measure what each layer contributes.

**The opportunity.** A pipeline that treats a claim as a set of elements, searches by both keyword and meaning, and reports measured lift at every stage can give reviewers a shortlist they can trust and explain.

## 3. Goals, non-goals, and success metrics

The product succeeds if the full pipeline retrieves examiner-cited prior art far more reliably than a vector-only baseline, and shows its work.

**Goals**

- G1. Retrieve relevant prior art for a claim with Recall@10 of at least 0.75 on the benchmark set.
- G2. Prove the contribution of each retrieval layer with a measured lift table.
- G3. Explain every result with a matching passage, element coverage, and a stage-by-stage rank trace.
- G4. Never present an explanation that is not grounded in a retrieved passage.
- G5. Deliver a search experience that feels like a production SaaS product: fast, clear, accessible, and consistent in light and dark themes.

**Non-goals**

- Legal opinions on validity, infringement, or patentability. Priorly is a research aid.
- Full-text coverage of every patent office. The first corpus is limited to one technology area.
- Multi-language search and machine translation.
- Image or drawing search.
- Real billing, payments, or multi-tenant enterprise administration. Pricing and team features are interface placeholders.

**Success metrics**

| Metric | Definition | Target |
| --- | --- | --- |
| Recall@10 | Share of examiner-cited references found in the top 10 | 0.75 or higher for the full pipeline |
| Recall lift over baseline | Full pipeline minus vector-only baseline, in points | 25 points or more |
| Precision@10 | Share of the top 10 that are examiner-cited | 0.35 or higher |
| MRR | Mean reciprocal rank of the first relevant result | 0.60 or higher |
| Exact-identifier recall | Recall@10 on queries that contain a patent ID or formula | 0.80 or higher |
| Grounded explanations | Result reasons whose claims are supported by the cited passage | 95% or higher on a reviewed sample |
| Search latency, p95 | Time from request to complete result set | 8 seconds or less with the LLM stages on |
| Interface accessibility | Automated audit score on core screens | 95 or higher |

## 4. Target users and personas

Priorly serves people who must find and defend references for a claim, ranging from specialists who search daily to inventors who search once.

| Persona | Role | Primary job to be done | What they need from Priorly |
| --- | --- | --- | --- |
| Maya, patent examiner or search analyst | Searches prior art for many claims each week | Build a thorough reference set quickly | Broad recall, element coverage, exact-term matching, saved reports |
| Daniel, patent attorney | Prepares validity or freedom-to-operate opinions | Identify references that could invalidate a claim and cite them correctly | Matching passages with locations, traceable rankings, exportable reports |
| Priya, in-house IP counsel | Screens inventions before filing | Decide whether an invention is worth filing | Fast shortlist, plain-language reasons, clear indication of weak coverage |
| Arjun, inventor or R&D engineer | Checks novelty before a disclosure | Understand what already exists in a technical area | Simple input, readable results, no legal jargon required |

**Primary persona.** Daniel, the patent attorney, drives the design. His output is audited, so every result must be explainable and reproducible.

**Secondary needs.** All personas value speed, and none should need training to run a first search.

## 5. User stories and acceptance criteria

Each story names the user, the outcome, and the condition that proves it works.

| ID | User story | Acceptance criteria |
| --- | --- | --- |
| US-01 | As an attorney, I paste a claim and run a search so that I get ranked prior art. | A claim of 40 or more characters can be submitted. Results appear in rank order with title, number, assignee, year, and classification. |
| US-02 | As an examiner, I see the claim split into elements so that I know what was searched. | Elements are shown as labelled, color-coded items. Each result shows which elements it covers. |
| US-03 | As an attorney, I read the matching passage so that I can verify relevance myself. | Every result shows the passage with matched text highlighted by element and a location reference. |
| US-04 | As an attorney, I open a result trace so that I can explain its ranking. | The trace shows the rank in vector search, keyword search, fusion, and rerank, plus the reason text. |
| US-05 | As an examiner, I turn pipeline stages on and off so that I can see what each one changes. | Toggles for decomposition, HyDE, hybrid search, and rerank re-run the search and visibly change the ordering. |
| US-06 | As counsel, I shortlist results and save a report so that I can return to my work. | Shortlisted results persist in the saved report with date, result count, and best score. |
| US-07 | As a reviewer, I open the benchmark so that I can see the evidence behind the pipeline. | Recall, precision, and MRR are shown per stage, by query type, with the contribution of each layer. |
| US-08 | As an engineer, I adjust pipeline settings so that I can tune retrieval. | Embedding model, reranker, RRF constant, candidate counts, and LLM provider are editable and saved. |
| US-09 | As any user, I see a clear message when something fails so that I know what to do next. | Errors state what happened and the next action. Partial results are never shown as complete. |
| US-10 | As any user, I switch between light and dark themes so that I can work comfortably. | The choice persists across sessions and respects the system setting by default. |

**User journey**

&#91;embedded content: user journey · 5 steps, mapped to user stories and requirements\]

Steps 2 and 4 are what separate Priorly from a plain search box: the claim is shown as elements before it is searched, and every ranking can be opened and re-run.

## 6. Functional requirements

Requirements are grouped by module. Priority uses P0 (must have), P1 (should have), and P2 (nice to have).

**Search and results**

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-01 | Accept a claim of at least 40 and at most 8,000 characters and reject input outside that range with a clear message. | P0 |
| FR-02 | Decompose the claim into 2 to 8 atomic, independently searchable elements using an LLM with a structured output schema. | P0 |
| FR-03 | Run keyword and vector retrieval for each element and merge all ranked lists with RRF. | P0 |
| FR-04 | Rerank the top 50 fused candidates with a cross-encoder and return the top K, where K is 5 or 10. | P0 |
| FR-05 | Return, per result: patent number, title, assignee, year, classification, matched passage with location, covered elements, scores, and a reason. | P0 |
| FR-06 | Generate each reason only from the retrieved passage and refuse to state anything the passage does not support. | P0 |
| FR-07 | Highlight matched text in the passage by claim element. | P1 |
| FR-08 | Flag results found by keyword search but missed by vector search. | P1 |
| FR-09 | Show element coverage across the result set. | P1 |
| FR-10 | Allow each pipeline stage (decompose, HyDE, hybrid, rerank) to be switched on or off per request. | P0 |

**Trace and reporting**

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-11 | Provide a per-result trace with the rank at vector, keyword, fusion, and rerank stages. | P0 |
| FR-12 | Let users shortlist results and save a search report with its settings and results. | P1 |
| FR-13 | List saved reports with title, date, result count, shortlist count, and best score. | P1 |
| FR-14 | Export a report as PDF and as CSV. | P2 |
| FR-15 | Record every search with its input, options, stage timings, and results for reproducibility. | P0 |

**Benchmark**

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-16 | Display Recall@10, Precision@10, and MRR for each pipeline configuration. | P0 |
| FR-17 | Display recall by query type, including queries that contain exact patent identifiers or formulae. | P0 |
| FR-18 | Display the contribution of each layer as the difference between consecutive configurations. | P0 |
| FR-19 | Export the benchmark table as CSV. | P2 |

**Settings and platform**

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-20 | Expose embedding model, reranker, RRF constant, candidates per retriever, rerank depth, and LLM provider as settings. | P1 |
| FR-21 | Cache LLM outputs for decomposition and HyDE by input hash. | P0 |
| FR-22 | Provide a light and a dark theme that follows the system preference by default. | P1 |
| FR-23 | Provide a marketing landing page with product explanation, comparison, benchmark summary, pricing placeholder, and FAQ. | P1 |

**Requirement mix**

&#91;embedded content: requirement tables in section 6 · 23 functional requirements\]

Benchmark export and report PDF export are the only P2 items, so everything a reviewer needs to judge the product is P0 or P1.

## 7. Data requirements

The corpus and the ground truth come from the same public source, so retrieval quality can be measured against real examiner decisions.

&#91;embedded content: data model · 4 files, 2 indexes, 4 SQLite tables\]

The four files are produced offline; the service itself stores only the SQLite tables.

**Source.** USPTO bulk data distributed through PatentsView: patent text (title, abstract, claims, description) and the citation records that identify who added each reference. Exact field names must be confirmed against the current PatentsView data dictionary before ingestion.

**Corpus definition**

- One technology area, for example wireless power transfer, sized between 5,000 and 20,000 patents.
- The corpus must include every patent that is cited by a query patent, not only the query patents themselves. Without them, recall cannot be measured.
- Documents are indexed at chunk level (abstract, claims, and description paragraphs) with patent identifier and paragraph number as metadata.
- Passage locations are reported as paragraph numbers because the source provides paragraphs rather than column and line numbers.

**Ground truth**

- Query set: claim 1 of about 200 patents sampled across the corpus.
- Relevant documents: references cited by the examiner on each query patent. References added by the applicant are excluded.
- Examiner citations are an incomplete record of relevance, so reported metrics are treated as a lower bound and documented as such.

**Leakage controls**

- The query patent is removed from the searchable set for its own query.
- Only documents published before the query patent's filing date are eligible results.
- The query set is fixed once and versioned. Tuning is done on a development split; reported numbers come from a held-out test split.

**Data splits**

| Split | Share of queries | Use |
| --- | --- | --- |
| Development | 40% | Tune chunking, candidate counts, RRF constant |
| Validation | 20% | Choose between configurations |
| Test | 40% | Final reported benchmark, run once per release |

**Derived data**

- Cached LLM outputs: decomposition and HyDE passages keyed by input hash.
- Search log: request, options, timings, results, and trace for every search.
- Benchmark results file consumed by the Benchmark page.

## 8. Retrieval pipeline specification

Each stage has one job, a defined input and output, and a switch, so its contribution can be measured on its own.

&#91;embedded content: retrieval pipeline · 6 stages, each with its own switch\]

Stages 3 and 4 are the hybrid core; the other stages can each be switched off to measure what they add.

| Stage | Input | Output | Method | Switch |
| --- | --- | --- | --- | --- |
| 1. Decompose | Full claim text | 2 to 8 elements | LLM with a structured output schema (list of strings), temperature 0, cached by input hash | decompose |
| 2. Expand (HyDE) | One element | A short hypothetical passage written as a matching patent would phrase it | LLM, temperature 0, cached by input hash | hyde |
| 3a. Keyword retrieval | Element text | Top 50 chunks | BM25 with a tokenizer that keeps patent identifiers, numbers, and chemical formulae intact | hybrid |
| 3b. Vector retrieval | HyDE passage, or the element when HyDE is off | Top 50 chunks | BGE embeddings searched in a vector database | always on |
| 4. Fuse | All ranked lists | One ranked list of up to 50 patents | Reciprocal Rank Fusion, k = 60 | hybrid |
| 5. Rerank | Claim and top 50 fused candidates | Top K patents with relevance scores | BGE cross-encoder scoring claim and passage together | rerank |
| 6. Explain | Top K results and their best passages | Reason text and element coverage | LLM constrained to the retrieved passage; coverage from per-element passage scores | always on |

**Fusion formula**

```latex
\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}, \quad k = 60
```

**Aggregation rules**

- Chunk scores are aggregated to patent level by taking the maximum chunk score.
- The best-scoring chunk per patent becomes the displayed passage.
- An element counts as covered when its best passage score in that patent passes a threshold set on the development split.
- When decomposition is off, the whole claim is one element and coverage is not reported.

**Grounding rules for explanations**

- The LLM receives only the matched passage, the claim element, and the result title.
- Each reason must quote or closely paraphrase text present in the passage.
- If the passage does not support a relationship to the claim, the reason states the gap, such as "Covers the receiver coil but does not mention frequency tuning".

**Failure handling**

- If the LLM is unavailable, the pipeline falls back to the whole claim as a single element, skips HyDE, and labels the run as degraded.
- If the reranker is unavailable, results are ordered by fusion score and the run is labelled accordingly.

**Search sequence**

&#91;embedded content: search sequence · 6 components, 12 messages\]

The browser receives each stage as an event while the search runs, so progress is visible long before the result set is complete. Nothing is stored until the final result exists.

**Latency budget**

&#91;embedded content: illustrative timings from the section 10 response example · limit from NFR-01\]

The two LLM-backed stages, decomposition and explanation, account for most of the time, which is why LLM outputs are cached by input hash and why decomposition and expansion can each be switched off.

## 9. System architecture

The web app talks to a single API service that runs the pipeline; indexing and benchmarking happen offline against the same stores.

&#91;embedded content: system architecture · API, 6 stages, 4 services, offline pipeline\]

The API service calls the LLM provider for decomposition, HyDE, and explanations, queries the vector and keyword indexes during retrieval, and uses SQLite for saved reports, the search log, and the LLM cache. The offline pipeline ingests and chunks patents, builds both indexes, and runs the benchmark that feeds the Benchmark page.

**Data flow for one search**

1. The web app sends the claim and stage options to the API.
2. The API decomposes the claim and expands each element with a hypothetical passage, reading from the cache when the input was seen before.
3. Keyword and vector retrieval run for each element and return ranked chunks.
4. Fusion merges the ranked lists, and the reranker scores the top candidates against the claim.
5. The explainer writes a reason for each result from its matched passage.
6. The API stores the search, with timings and trace, and returns the result set.

**Deployment topology**

- Local development: one container composition command starts the API and the vector store; the front end is served as static files.
- Public demo: the API on a free container host, the front end on static hosting, and a small demonstration corpus to stay within free-tier limits.
- Configuration: model names, provider keys, and index locations come from environment variables.

## 10. API specification

The API is JSON over HTTPS. Result field names match the interface's data contract exactly, so the front end swaps its mock data for live calls without changes to rendering code.

| Method and path | Purpose | Key inputs | Key outputs |
| --- | --- | --- | --- |
| POST /api/search | Run the pipeline for one claim | claim, options (decompose, hyde, hybrid, rerank), k | search\_id, elements, results, timings, degraded flag |
| GET /api/search/{search\_id} | Fetch a past search for reproducibility | search\_id | Full stored request and response |
| POST /api/search/stream | Same as search, streaming stage progress over server-sent events | Same as POST /api/search | Stage events, then the final payload |
| GET /api/benchmark | Return the latest benchmark results | none | stages, by\_type, query\_count, dataset\_version |
| GET /api/reports | List saved reports | none | Report summaries |
| POST /api/reports | Save a report | search\_id, title, shortlist | report\_id |
| GET /api/settings | Read pipeline settings | none | Settings object |
| PUT /api/settings | Update pipeline settings | Settings object | Updated settings |
| GET /api/health | Liveness and dependency status | none | Status of index, vector store, LLM, reranker |
| DELETE /api/search/{search\_id} | Erase one stored search and the reports built from it | search\_id | Counts of searches and reports removed |
| DELETE /api/data?confirm=yes | Erase every stored search, report and cached LLM output; settings are kept | confirm=yes, required | Counts removed for each kind |
| GET /api/reports/{report\_id}/export.csv | Download a saved report as a spreadsheet file | report\_id | CSV with formula-safe cells |
| GET /api/reports/{report\_id}/export.pdf | Download a saved report as a document | report\_id | PDF with the claim, shortlist, passages and reasons |
| GET /api/metrics | Operational counters for monitoring | none | searches, degraded runs, latency p50 and p95, stage p95, errors by status, LLM cache hit rate, stored counts |

**Search request**

```json
{
  "claim": "A system for wireless charging of an electric vehicle, comprising ...",
  "options": { "decompose": true, "hyde": true, "hybrid": true, "rerank": true },
  "k": 10
}
```

**Search response (abridged)**

```json
{
  "search_id": "s_01HX...",
  "degraded": false,
  "elements": [ { "n": 1, "text": "Transmitter coil embedded in a road surface" } ],
  "results": [
    {
      "id": "US 9,104,382 B2",
      "title": "Inductive power transfer system for roadway vehicles",
      "assignee": "Voltaic Mobility Inc.",
      "year": 2015,
      "cpc": "B60L 53/12",
      "el": [1, 2, 3],
      "dense": 1,
      "bm25": 4,
      "rrf_rank": 1,
      "score": 0.91,
      "loc": "Paragraph [0042]",
      "passage": "... <mark data-e=\"1\">coil embedded beneath the road surface</mark> ...",
      "why": "Discloses all three elements in one embodiment."
    }
  ],
  "timings_ms": { "decompose": 900, "retrieve": 420, "fuse": 5, "rerank": 1300, "explain": 1800 }
}
```

**Error model**

| HTTP status | Code | Meaning | User-facing action |
| --- | --- | --- | --- |
| 400 | claim\_too\_short or claim\_too\_long | Claim length is outside 40 to 8,000 characters | Edit the claim and retry |
| 422 | invalid\_options | Unknown option or k value | Reset options |
| 429 | rate\_limited | Too many searches in the window | Wait and retry after the stated delay |
| 503 | dependency\_unavailable | Vector store or index unreachable | Retry shortly; no partial results are shown |
| 200 with degraded true | degraded\_run | LLM or reranker unavailable, fallback used | Results shown with a visible degraded label |

The `loc` field carries a paragraph reference such as "Paragraph \[0042\]". Error bodies never include raw exception text.

## 11. UI and UX requirements

The interface is a two-part web product: a marketing landing page and a workspace for running and reviewing searches. A working front-end prototype with mock data already exists and defines the visual and interaction standard.

**Screens**

| Screen | Purpose | Key elements |
| --- | --- | --- |
| Landing page | Explain the product and the evidence behind it | Hero with animated claim highlighting and a sample result, five-step pipeline explanation, vector-only versus hybrid comparison, benchmark bars, feature list, pricing placeholder, FAQ |
| New search | Run a search and review results | Claim input with character count, four stage toggles, top-K selector, stage progress, ranked result cards, claim elements panel, coverage panel, run summary |
| Result trace drawer | Explain one result | Rank at vector, keyword, fusion, and rerank; matched passage; reason |
| Benchmark | Show the evidence | Metric tiles, grouped bar chart by stage, table by query type, per-layer contribution list, CSV export |
| Saved reports | Return to earlier work | List with title, date, result count, shortlist count, best score |
| Pipeline settings | Tune the pipeline | Embedding model, reranker, RRF constant, candidate counts, rerank depth, LLM provider |

&#91;embedded content: screen map · landing page, 4 workspace screens, 1 drawer, 1 mode bar\]

The workspace needs no sign-in screen: it is a single-user product and the first release has no accounts.

**Result card content.** Rank, title, patent number, assignee, year, classification, element chips, a flag when keyword search found a result that vector search missed, the matched passage with location, a plain-language reason, the score, a shortlist control, and a trace control.

&#91;embedded content: result card wireframe · 7 callouts · illustrative content\]

The text in the wireframe is illustrative. Marked phrases carry the colour and label of the claim element they match, so colour is never the only signal.

**Interaction requirements**

- Running a search shows five named stages with running, done, and skipped states.
- Turning a stage off and re-running visibly changes the result order or membership.
- The trace drawer opens and closes with the keyboard and traps no focus after closing.
- Shortlist state and saved reports survive a page reload.
- Theme choice persists and defaults to the system preference.

**State requirements**

| State | Behaviour |
| --- | --- |
| Empty | Invites the first search and explains what appears |
| Loading | Stage progress is shown; results are never partially rendered as final |
| Error | States what happened and the next action, without raw exception text |
| Degraded | Results are shown with a visible label naming the stage that fell back |
| No results | Explains the likely cause and suggests editing the claim or enabling stages |

**Design and accessibility standards**

- Responsive from 360 px phones to wide desktops; the sidebar collapses to a top bar on narrow screens.
- Light and dark themes with sufficient contrast for body text and interactive controls.
- Every control is reachable and operable by keyboard with a visible focus indicator.
- Color is never the only carrier of meaning: element chips carry labels (E1, E2, E3).
- Motion respects the reduced-motion preference.
- Sentence-case copy, verb-first button labels, no raw error strings.

## 12. Evaluation and benchmark methodology

Every retrieval layer must justify itself with a measured gain on held-out queries, and layers that do not help are reported as such.

**Metrics**

- Recall@10: the share of a query's examiner-cited references that appear in the top 10, averaged over queries.
- Precision@10: the share of the top 10 that are examiner-cited, averaged over queries.
- MRR: the mean of the reciprocal rank of the first relevant result.
- Exact-identifier recall: Recall@10 restricted to queries that contain a patent number or a chemical formula.

**Configurations compared**

| ID | Configuration | What it isolates |
| --- | --- | --- |
| C0 | Vector search only, whole claim | Baseline |
| C1 | C0 plus BM25 and RRF | Value of keyword retrieval and fusion |
| C2 | C1 plus cross-encoder rerank | Value of reranking |
| C3 | C2 plus claim decomposition and HyDE | Value of query transformation |

Decomposition and HyDE are also run separately from each other on top of C2, so the contribution of each is reported on its own.

&#91;embedded content: ablation ladder · 4 configurations, 2 single-layer branches\]

**Protocol**

- Tune on the development split, choose configurations on the validation split, and report the test split once per release.
- All LLM outputs are cached by input hash so repeated runs are deterministic and cost nothing.
- Report each metric with a bootstrap 95% confidence interval across queries.
- Report results by query type: mechanical claims, chemical and formula queries, software and protocol claims, and queries containing exact identifiers.
- Keep the evaluation script, query set version, and corpus version with every results file.

**Failure analysis**

- Curate at least five queries where vector search ranked an examiner-cited reference below position 30 and the hybrid pipeline recovered it. These serve as the documented failure demonstration.
- Review a sample of 50 explanations and record the share fully supported by the cited passage.

**Reporting**

- Results are written to a single results file that the Benchmark page reads directly.
- The README states that examiner citations are an incomplete record of relevance, so the numbers are a lower bound.
- Sample numbers in the prototype are placeholders and are replaced by measured values before any public claim is made.

## 13. Non-functional requirements

| ID | Area | Requirement |
| --- | --- | --- |
| NFR-01 | Performance | p95 search latency of 8 seconds or less with all stages on; 2 seconds or less with LLM stages off |
| NFR-02 | Performance | Keyword and vector retrieval complete within 1 second for a corpus of up to 20,000 patents |
| NFR-03 | Throughput | At least 5 concurrent searches on a single small instance without errors |
| NFR-04 | Reliability | Graceful degradation when the LLM or reranker is unavailable; no silent partial results |
| NFR-05 | Reproducibility | A stored search can be replayed from its request, options, and cached LLM outputs |
| NFR-06 | Security | API keys held in environment variables, never in the repository or the client |
| NFR-07 | Security | Input length limits, request rate limiting, and output escaping for highlighted passages |
| NFR-08 | Privacy | Claims entered by users are stored only to support saved reports and replay, can be deleted on request, and are never used for model training |
| NFR-09 | Observability | Per-stage timings, cache hit rate, and error counts logged for every search |
| NFR-10 | Testing | Unit tests for tokenizer, RRF, aggregation, and API contract; an end-to-end test that runs the full pipeline on a small fixed corpus |
| NFR-11 | Portability | Entire stack starts with a single container composition command |
| NFR-12 | Cost | Run on free tiers and local models; LLM calls minimized through caching |
| NFR-13 | Accessibility | Core screens meet WCAG 2.1 AA for contrast, keyboard operation, and focus visibility |
| NFR-14 | Maintainability | Pipeline stages are separate modules behind stable interfaces so any stage can be replaced or disabled |

**Threat model**

Priorly stores confidential claim text and calls external services, so each exposure has a named control. The status column says what is already in the code.

| Exposure | Threat | Control | Status |
| --- | --- | --- | --- |
| Claim text at rest | Unpublished claims are readable by anyone with access to the database file | Claims are stored only to support replay and saved reports; the database path is configurable and excluded from version control | Built and tested: erase one search or all stored data on request |
| Claim text in transit to the LLM | Unpublished claims leave the machine when a hosted LLM is enabled | The LLM backend is off by default (rule-based stand-in); the hosted provider is an explicit opt-in setting | Built |
| Provider API key | Key leaks through the client, logs, or the repository | Key lives only in environment variables or an untracked `.env`; the settings API never returns it; `.env` is ignored by version control | Built |
| Passage rendering | Markup inside patent text runs as script in the browser | Passage text is escaped before highlight marks are added; only `mark` tags are emitted | Built and tested |
| Prompt injection | Text inside a retrieved patent tells the LLM to ignore its instructions | The LLM receives only the matched passage, one element, and the title, and its output is shown as plain text and never executed | Mitigated by design; not red-teamed |
| Abuse and cost | A client floods the API and exhausts the LLM quota | Per-client rate limit with a clear 429 and retry delay; LLM outputs cached by input hash | Built and tested |
| Error disclosure | Stack traces or internal paths reach the user | Errors use one stable shape with a code and a message; raw exception text is never returned | Built and tested |
| Spreadsheet formulas | A title beginning with `=` or `+` runs as a formula when a CSV export is opened | Export cells that begin with `=`, `+`, `-`, `@`, tab or carriage return are prefixed with a quote | Built and tested |

## 14. Technology stack

The stack favors free, local, and widely used components so the whole system can be reproduced on a laptop.

| Layer | Choice | Rationale |
| --- | --- | --- |
| Front end | HTML, CSS, JavaScript (existing prototype) | Complete, dependency-free, ready to connect to the API |
| API | FastAPI with Pydantic | Typed request and response contracts, async support, automatic API documentation |
| Orchestration | LangChain components or plain Python modules | Stages stay replaceable and testable |
| LLM | Groq-hosted open model | Fast, free tier; used for decomposition, HyDE, and explanations |
| Embeddings | BGE base model via sentence-transformers | Strong open retrieval model that runs on CPU for a small corpus |
| Vector store | Qdrant in a container | Filtering by date and metadata, simple local setup |
| Keyword search | rank-bm25 with a custom tokenizer | Exact-term matching that keeps identifiers and formulae intact |
| Reranker | BGE cross-encoder reranker | Free, local, and strong for passage-level relevance |
| Data source | PatentsView bulk data | Public patent text plus examiner-labelled citations |
| Storage | SQLite for reports, search log, and settings | No server needed; adequate for a single-user product |
| Packaging | Docker Compose | One command starts the API and the vector store |
| Testing | pytest | Unit and end-to-end tests |
| Deployment | A free container host for the API and static hosting for the front end | Public demo at no cost |

## 15. Risks, assumptions, and dependencies

**Risks and mitigations**

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Examiner citations miss relevant documents, so measured recall understates quality | High | Medium | Report numbers as a lower bound; add a small manually judged sample |
| Cited references are absent from the corpus, making recall unmeasurable | Medium | High | Always add every cited patent of every query patent to the corpus |
| Data leakage inflates results | Medium | High | Remove the query patent, restrict to earlier publications, and use a held-out test split |
| LLM hallucination in explanations | Medium | High | Constrain the LLM to the retrieved passage; review a sample; label gaps explicitly |
| LLM rate limits slow evaluation | High | Medium | Cache all LLM outputs by input hash; batch and retry with backoff |
| Full pipeline latency is too high | Medium | Medium | Limit rerank depth, cache expansions, offer stage toggles |
| Layers show little or no gain | Medium | Medium | Report the finding honestly; it is still a measured result |
| Free-tier hosting limits slow the public demo | Medium | Low | Use a small demonstration corpus and warm-start the service |
| Passage locations differ from patent column and line citations | High | Low | Use paragraph references and state this in the interface |

**Assumptions**

- PatentsView data fields for examiner versus applicant citations are available and can be confirmed against its data dictionary.
- A single technology area of 5,000 to 20,000 patents is sufficient to show meaningful differences between configurations.
- Free-tier LLM access remains available for development.
- A CPU-only environment is adequate for embedding and reranking at this corpus size.

**Dependencies**

- Public patent data from USPTO via PatentsView.
- Hosted LLM provider for decomposition, HyDE, and explanations.
- Open embedding and reranker models from the sentence-transformers ecosystem.
- Container runtime for the vector store.

## 16. Open questions, glossary, and appendix

**Open questions**

- Which technology area and CPC range define the first corpus?
- Should the product index full descriptions or only abstracts and claims for the first release?
- Is a small manually judged relevance sample needed alongside examiner citations?
- Which LLM provider and model are used for the final benchmark run?
- Is a hosted public demo required, and on which free tier?

**Glossary**

| Term | Meaning |
| --- | --- |
| Prior art | Any earlier publication that may affect whether a claim is new or non-obvious |
| Claim element | One independently searchable limitation within a claim |
| BM25 | A keyword ranking function based on term frequency and rarity |
| Dense retrieval | Search by similarity between text embeddings |
| HyDE | Hypothetical Document Embeddings: embedding a generated passage instead of the raw query |
| RRF | Reciprocal Rank Fusion: merging ranked lists by summing 1 divided by (k + rank) |
| Cross-encoder | A model that reads a query and a passage together to score relevance |
| Recall@10 | Share of relevant documents found in the top 10 |
| Precision@10 | Share of the top 10 that are relevant |
| MRR | Mean reciprocal rank of the first relevant result |
| Examiner citation | A reference cited by the patent examiner during examination |
| CPC | Cooperative Patent Classification |
| Leakage | Information in the test setup that lets a system see the answer |

**Appendix: result data contract**

| Field | Type | Description |
| --- | --- | --- |
| id | string | Patent number |
| title | string | Patent title |
| assignee | string | Assignee name |
| year | integer | Publication year |
| cpc | string | Primary classification |
| el | integer array | Claim element numbers the result covers |
| dense | integer | Rank in vector search |
| bm25 | integer | Rank in keyword search |
| rrf\_rank | integer | Rank after fusion |
| score | number | Reranker relevance score from 0 to 1 |
| loc | string | Paragraph reference of the matched passage |
| passage | string | Passage text with mark tags carrying element numbers |
| why | string | Reason grounded in the passage |

## 17. Implementation map and requirement traceability

This section ties every requirement in section 6 to the module that implements it and the check that proves it, so a reviewer can start from any requirement and land on code and evidence. The product is delivered as one repository that runs from a single command.

**Repository layout**

| Path | Responsibility | PRD sections |
| --- | --- | --- |
| `frontend/` | Landing page, search workspace, benchmark, saved reports, settings; plain HTML, CSS and JavaScript with no build step; falls back to built-in demo data when the API is unreachable | 11 |
| `frontend/js/api.js` | API client: health, streaming search, benchmark, reports, CSV export link, settings | 10, 11 |
| `backend/app/main.py` | FastAPI application: endpoints, error model, rate limiter, server-sent events, CSV export, static front end | 6, 10, 13 |
| `backend/app/pipeline.py` | Decompose, expand, retrieve, fuse, rerank, explain; per-stage switches; leakage controls; degraded-run handling | 8 |
| `backend/app/retrievers.py` | Keyword index (BM25), dense retriever (TF-IDF stand-in or BGE with Qdrant), chunk-to-patent aggregation | 8, 14 |
| `backend/app/tokenizer.py` | Tokenizer that keeps patent numbers, numbers and chemical formulae intact | 8 |
| `backend/app/fusion.py` | Reciprocal Rank Fusion | 8 |
| `backend/app/rerank.py` | Lexical scorer and BGE cross-encoder | 8 |
| `backend/app/llm.py` | Rule-based stand-in and Groq client with retries and a SQLite cache | 8, 13 |
| `backend/app/highlight.py` | Escaped passage highlighting by claim element | 6 |
| `backend/app/store.py` | SQLite: search log, saved reports, settings, LLM cache | 7, 10 |
| `backend/ingest/` | Sample corpus generator, PatentsView download and corpus builder | 7 |
| `backend/eval/` | Query builder with splits, benchmark harness, metrics with bootstrap intervals | 12 |
| `backend/tests/` | Unit, pipeline, API contract, ingest and evaluation tests | 18 |
| `docker-compose.yml`, `Makefile`, `.env.example` | One-command run, container stack, configuration template | 9, 19 |

**Requirement traceability**

| Requirement | Implemented in | Verified by |
| --- | --- | --- |
| FR-01 Claim length limits | `main.py` length check, codes `claim_too_short` and `claim_too_long` | `test_claim_length_errors` |
| FR-02 Decompose into elements | `llm.py` decomposition, `pipeline.py` | `test_heuristic_decompose_*`, `test_llm_decompose_parses_json_and_caches`, `test_llm_rejects_invalid_decomposition` |
| FR-03 Keyword and vector retrieval with RRF | `retrievers.py`, `fusion.py`, `pipeline.py` | `test_rrf_*`, `test_full_run_shape`, `test_exact_identifier_is_found_by_keyword_search` |
| FR-04 Rerank top candidates | `rerank.py`, `pipeline.py` | `test_results_are_ordered_by_score`, `test_k_limits_results` |
| FR-05 Result fields | `pipeline.py`, `schemas.py` | `test_search_contract` |
| FR-06 Grounded reasons | Reason built from the matched passage only; LLM prompt limited to that passage | Mock transport tests only; the reviewed-sample measure in section 3 is not yet run |
| FR-07 Passage highlighting | `highlight.py` | `test_highlight_marks_elements_and_escapes_html`, `test_highlight_without_match_is_plain_escaped_text` |
| FR-08 Keyword-only flag | `pipeline.py` | `test_keyword_only_flag_is_consistent_with_ranks` |
| FR-09 Element coverage | `pipeline.py`, coverage panel in the UI | `test_baseline_has_whole_claim_and_no_coverage` |
| FR-10 Stage switches | `pipeline.py` options | `test_each_switch_is_honoured`, `test_invalid_options_are_422` |
| FR-11 Result trace | Rank fields in the result, trace drawer in the UI | `test_full_run_shape` |
| FR-12, FR-13 Save and list reports | `store.py`, `/api/reports` | `test_reports_flow_and_csv_export` |
| FR-14 Report export | CSV and PDF built | `test_reports_flow_and_csv_export`, `test_csv_export_neutralises_formula_cells` |
| FR-15 Replayable search log | `store.py`, `/api/search/{id}` | `test_search_is_stored_and_replayable` |
| FR-16 Benchmark metrics | `eval/run_eval.py`, `eval/metrics.py`, `/api/benchmark` | `test_run_config_enforces_leakage_and_summarizes`, `test_metrics`, `test_benchmark_endpoint` |
| FR-17 Recall by query type | `eval/build_queries.py` | `test_splits_are_stable_and_types_are_sane` |
| FR-18 Layer contribution | `eval/run_eval.py`, Benchmark page | `test_run_config_enforces_leakage_and_summarizes` |
| FR-19 Benchmark CSV export | Benchmark page | Checked in a real browser run |
| FR-20 Settings | `/api/settings`, Settings page | `test_settings_roundtrip_and_validation` |
| FR-21 LLM output cache | `llm.py`, `store.py` | `test_llm_decompose_parses_json_and_caches` |
| FR-22 Light and dark themes | `frontend/css/styles.css` | Checked in a real browser run, both themes |
| FR-23 Landing page | `frontend/index.html` | Checked in a real browser run |
| NFR-06 Secrets | Keys read from the environment only; settings, health and metrics never return them | test\_api\_key\_is\_never\_returned\_by\_settings\_or\_health |
| NFR-07 Input limits, rate limit, escaping | Length checks, rate limiter and security-header middleware in main.py; escaped highlighting in highlight.py | test\_claim\_length\_errors, test\_rate\_limit, test\_highlight\_marks\_elements\_and\_escapes\_html, test\_security\_headers\_and\_request\_id\_on\_every\_response |
| NFR-08 Delete stored claims | store.py, DELETE /api/search/{id}, DELETE /api/data, Erase button in Settings | test\_deleting\_a\_search\_removes\_its\_reports, test\_erase\_all\_needs\_explicit\_confirmation, test\_erase\_all\_keeps\_settings |
| NFR-09 Observability | One log line per request and per search with timings, degraded stages and cache hit rate; GET /api/metrics | test\_metrics\_report\_searches\_latency\_and\_cache\_hit\_rate |

## 18. Test and verification plan

Verification has six levels, and each one answers a different question. The first five run automatically on every change; the browser level is a scripted run, and a separate set of gates must pass before any public claim about quality.

| Level | Question it answers | What runs |
| --- | --- | --- |
| Unit | Does each building block behave exactly as specified? | Tokenizer, RRF, metrics, bootstrap interval, highlighting, decomposition, LLM client retries and cache |
| Pipeline | Do the stages work together and fail safely? | Full run shape, ordering, every switch, leakage controls, degraded runs, exact patent number retrieval |
| API contract | Does the service keep the promises in section 10? | Search, stream, replay, errors, rate limit, reports, CSV export, settings, health, static front end |
| Data and evaluation | Do ingestion and the benchmark enforce the rules in sections 7 and 12? | Examiner-only citations, stable splits, leakage enforcement, sample-data flag, Qdrant code path |
| Browser | Does the interface work end to end in both modes? | Live and demo search, trace drawer, save and export, benchmark, settings, phone width, no console errors |
| Production behaviour | Do exports, erasure, metrics and security headers work as promised? | PDF and CSV export, erasing one search or everything, metrics, headers and request ids, key never exposed |

**Tests by area**

&#91;embedded content: backend test suite · 60 tests across 5 areas\]

The browser level is a scripted run and is not counted here.

**Quality gates before any public claim**

| Gate | Pass condition | State |
| --- | --- | --- |
| Real corpus | A corpus of 5,000 or more patents with every cited reference present | Not run; the bundled sample has 31 fictional patents |
| Measured targets | Recall@10, lift, Precision@10, MRR and exact-identifier recall reach the section 3 targets on the test split | Not run |
| Real model backends | Benchmark run with BGE embeddings, the BGE cross-encoder and the hosted LLM | Not run; stand-ins only |
| Grounded explanations | At least 95% of 50 reviewed reasons are fully supported by their passage | Not run |
| Latency | p95 within 8 seconds with all stages on, measured on the real corpus | Not measured |
| Accessibility | Automated audit score of 95 or higher on the core screens | Not measured |

## 19. Configuration reference

Every runtime choice is an environment variable, read from the process or from an untracked `.env` file; real environment variables win. The defaults run with no keys and no downloads.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DENSE_BACKEND` | `tfidf` | Vector retriever: `tfidf` (offline stand-in) or `bge` (BGE embeddings in Qdrant) |
| `RERANK_BACKEND` | `lexical` | Reranker: `lexical` (token overlap) or `bge` (cross-encoder) |
| `LLM_BACKEND` | `heuristic` | Decomposition, expansion and reasons: `heuristic` (rules) or `groq` (hosted model) |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model when the dense backend is `bge` |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder when the rerank backend is `bge` |
| `QDRANT_URL` | `:memory:` | Vector store location; a server address such as `http://localhost:6333` for a persistent index |
| `QDRANT_COLLECTION` | `priorly_chunks` | Collection name for chunk vectors |
| `GROQ_API_KEY` | empty | Provider key; required only when the LLM backend is `groq` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Hosted model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `RRF_K` | `60` | Fusion constant, allowed range 1 to 200 |
| `CANDIDATES` | `50` | Chunks kept per retriever, allowed range 10 to 200 |
| `RERANK_DEPTH` | `50` | Fused candidates passed to the reranker, allowed range 10 to 100 |
| `RATE_LIMIT_PER_MIN` | `60` | Searches allowed per client per minute |
| `CORS_ORIGINS` | `*` | Allowed browser origins, comma separated |
| `CORPUS_PATH` | bundled sample | Corpus file the index is built from |
| `BENCHMARK_PATH` | bundled sample | Benchmark results file the Benchmark page reads |
| `DB_PATH` | `backend/data/priorly.db` | SQLite file for searches, reports, settings and the LLM cache |
| `FRONTEND_DIR` | `frontend/` | Static files served with the API |
| LOG\_LEVEL | INFO | Log verbosity; each request and each search writes one line |

**Run modes**

| Mode | Settings | Needs | Shows |
| --- | --- | --- | --- |
| Offline default | All three backends at their defaults | Python only | Full interface and every switch; retrieval is lexical, so no paraphrase benefit |
| Full local | `bge`, `bge`, `groq`, Qdrant server | Model download, provider key, Qdrant container | Real semantic retrieval, expansion and grounded reasons |
| Containers | `docker compose up --build` | Docker | The API and a Qdrant service together |

## 20. Delivered status and known limitations

This section states plainly what exists, what has been exercised, and what has not, so the document can be read as a record of the build and not only as a plan.

| Area | State | Notes |
| --- | --- | --- |
| Retrieval pipeline, six stages with switches | Built and tested | Runs on the sample corpus with offline backends |
| API, streaming, errors, rate limit, reports, settings | Built and tested | Contract matches section 10 |
| Search workspace, trace drawer, benchmark, reports, settings | Built and checked in a browser | Live mode against the API and demo mode without it; light and dark; phone width |
| Corpus and query builders | Built; tested on a small synthetic fixture | Column names for PatentsView tables are unconfirmed; a header check mode is provided |
| Benchmark harness | Built and tested | Numbers on the bundled sample are flagged as sample data and show no layer lift |
| BGE embeddings, BGE cross-encoder, hosted LLM | Code written; not run | No model download or key was available; the hosted client is tested against a mock transport |
| Qdrant server mode | Code written; not run against a server | The in-memory path is tested |
| Landing page benchmark bars | Placeholder values | Replace with measured results before the page is shown |
| Report export | CSV and PDF | Both formats are built and tested |
| Delete stored claims on request | Built and tested | Erase one search or all stored data (NFR-08); Settings has an Erase button |
| Cache hit rate in logs | Built and tested | Cache hit rate in the search log and at /api/metrics (NFR-09), with latency percentiles and error counts |
| Passage location format | Paragraph sequence number | Not the patent's own bracketed paragraph label |
| Container image | Written; not built here | No container daemon was available; the image follows the steps that were run locally and has an unprivileged user, a health check and a restart policy |
| Automated checks | 60 backend tests passing | A continuous integration workflow runs them on two Python versions and syntax-checks the front end |

The next milestone is a measured result: ingest the real corpus, enable the real backends, run the test split once, and replace every placeholder number with the output.
