"""The search pipeline (PRD section 8).

decompose -> expand (HyDE) -> retrieve (BM25 + dense) -> fuse (RRF) -> rerank -> explain

Every stage has a switch and a measured timing. `run_events` yields progress events and finally the
result payload; `search` returns only the payload.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from typing import Iterator

from .corpus import Corpus
from .fusion import rrf
from .highlight import highlight
from .llm import LLMError, heuristic_decompose
from .retrievers import Constraints, aggregate_to_patents

log = logging.getLogger("priorly.pipeline")
MAX_COVERAGE_CHUNKS = 40


@dataclass(frozen=True)
class Options:
    decompose: bool = True
    hyde: bool = True
    hybrid: bool = True
    rerank: bool = True
    explain: bool = True
    dense: bool = True  # False = keyword (BM25) retrieval only
    prf: bool = False  # pseudo-relevance feedback: expand the query with terms from the top keyword hits


class BudgetExceeded(Exception):
    """A step used up its time budget."""


class _NoDense:
    name = "off"

    def search(self, query, n, cons):
        return []


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


def _best_rank(
    lists: list[list[tuple[str, float, list[int]]]],
) -> dict[str, int]:
    ranks: dict[str, int] = {}

    for lst in lists:
        for pos, (pid, _, _) in enumerate(lst, start=1):
            if pid not in ranks or pos < ranks[pid]:
                ranks[pid] = pos

    return ranks


def _build_chunk_pool(
    dense_lists: list[list[tuple[str, float, list[int]]]],
    bm25_lists: list[list[tuple[str, float, list[int]]]],
    limit: int,
    include_bm25: bool = True,
) -> dict[str, list[int]]:
    """
    Build a per-patent chunk pool by alternating dense and BM25 sources.

    This prevents dense retrieval from filling the entire reranking pool
    before keyword-retrieved chunks get a chance to be included.
    """
    limit = max(0, limit)

    def merge_chunks(
        result_lists: list[list[tuple[str, float, list[int]]]],
    ) -> dict[str, list[int]]:
        merged: dict[str, list[int]] = {}

        for result_list in result_lists:
            for pid, _, chunk_idxs in result_list:
                bucket = merged.setdefault(pid, [])

                for idx in chunk_idxs:
                    if idx not in bucket:
                        bucket.append(idx)

        return merged

    sources = [merge_chunks(dense_lists)]

    if include_bm25:
        sources.append(merge_chunks(bm25_lists))

    patent_ids = dict.fromkeys(
        pid
        for source in sources
        for pid in source
    )

    pool: dict[str, list[int]] = {}

    for pid in patent_ids:
        selected: list[int] = []
        seen: set[int] = set()

        source_chunks = [
            source.get(pid, [])
            for source in sources
        ]

        positions = [0] * len(source_chunks)

        while len(selected) < limit:
            added_this_round = False

            for source_idx, chunks_for_source in enumerate(source_chunks):
                while positions[source_idx] < len(chunks_for_source):
                    idx = chunks_for_source[positions[source_idx]]
                    positions[source_idx] += 1

                    if idx in seen:
                        continue

                    selected.append(idx)
                    seen.add(idx)
                    added_this_round = True
                    break

                if len(selected) >= limit:
                    break

            if not added_this_round:
                break

        pool[pid] = selected

    return pool


class Engine:
    def __init__(self, settings, corpus: Corpus, bm25, dense, scorer, llm):
        self.settings = settings
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.scorer = scorer
        self.llm = llm
        self._prf = None

    # ------------------------------------------------------------------ public

    def search(
        self,
        claim: str,
        opts: Options,
        k: int = 10,
        cons: Constraints | None = None,
    ) -> dict:
        result = None

        for event in self.run_events(claim, opts, k, cons):
            if event["type"] == "result":
                result = event["data"]

        assert result is not None
        return result

    def run_events(
        self,
        claim: str,
        opts: Options,
        k: int = 10,
        cons: Constraints | None = None,
    ) -> Iterator[dict]:
        cons = cons or Constraints()
        s = self.settings
        corpus = self.corpus

        timings: dict[str, float] = {}
        degraded: list[str] = []
        started = time.perf_counter()
        work_before = self._work_counters()

        def stage(name: str, status: str, **extra) -> dict:
            return {
                "type": "stage",
                "stage": name,
                "status": status,
                **extra,
            }

        # 1. decompose ---------------------------------------------------------

        query_text = claim
        if opts.prf:
            if self._prf is None:
                from .prf import Prf

                self._prf = Prf(corpus, self.bm25)
            query_text = self._prf.expand(claim, cons)

        if opts.decompose:
            yield stage("decompose", "running")
            t0 = time.perf_counter()

            try:
                texts = self._within(s.stage_budget_s, self.llm.decompose, query_text)

            except (LLMError, BudgetExceeded) as exc:
                log.warning(
                    "decomposition failed, falling back to rules: %s", str(exc)[:200]
                )
                texts = heuristic_decompose(query_text)
                degraded.append("decompose")

            timings["decompose"] = _ms(t0)

            yield stage(
                "decompose",
                "done",
                ms=timings["decompose"],
                detail=f"{len(texts)} claim element{'s' if len(texts) != 1 else ''}",
            )

        else:
            texts = [query_text]
            yield stage("decompose", "skipped")

        # 2. expand (HyDE) -----------------------------------------------------

        if opts.hyde:
            yield stage("hyde", "running")
            t0 = time.perf_counter()

            dense_queries = []

            for n, text in enumerate(texts, start=1):
                yield {"type": "progress", "stage": "hyde", "done": n - 1, "total": len(texts), "label": "passages drafted by the LLM"}
                try:
                    left = s.stage_budget_s - (time.perf_counter() - t0) if s.stage_budget_s > 0 else 0
                    if s.stage_budget_s > 0 and left <= 0:
                        raise BudgetExceeded("HyDE stage budget used up")
                    dense_queries.append(self._within(left, self.llm.hyde, text))

                except (LLMError, BudgetExceeded) as exc:
                    log.warning(
                        "HyDE failed for one element, using the element text: %s", str(exc)[:200]
                    )
                    dense_queries.append(text)

                    if "hyde" not in degraded:
                        degraded.append("hyde")

            yield {"type": "progress", "stage": "hyde", "done": len(texts), "total": len(texts), "label": "passages drafted by the LLM"}
            timings["hyde"] = _ms(t0)

            yield stage(
                "hyde",
                "done",
                ms=timings["hyde"],
                detail=f"{len(dense_queries) - sum(1 for a, b in zip(dense_queries, texts) if a == b)} of {len(texts)} passages drafted by the LLM",
            )

        else:
            dense_queries = list(texts)
            yield stage("hyde", "skipped")

        # 3. retrieve ----------------------------------------------------------

        # Keyword and vector ranks are always computed so the trace can show both.
        # The hybrid switch decides whether keyword results take part in fusion.

        yield stage("retrieve", "running")
        t0 = time.perf_counter()

        dense_lists = []
        bm25_lists = []

        for text, dq in zip(texts, dense_queries):
            dense_lists.append(
                aggregate_to_patents(
                    corpus,
                    (self.dense if opts.dense else _NoDense()).search(dq, s.trace_depth, cons),
                )
            )

            bm25_lists.append(
                aggregate_to_patents(
                    corpus,
                    self.bm25.search(text, s.trace_depth, cons),
                )
            )

        dense_rank = _best_rank(dense_lists)
        bm25_rank = _best_rank(bm25_lists)

        pool = _build_chunk_pool(
            dense_lists,
            bm25_lists,
            s.chunk_pool,
            include_bm25=opts.hybrid,
        )

        timings["retrieve"] = _ms(t0)

        found = len({pid for lst in dense_lists + bm25_lists for pid, _, _ in lst})

        yield stage(
            "retrieve",
            "done",
            ms=timings["retrieve"],
            detail=f"{found} candidate patents from {len(self.corpus.chunks):,} passages",
        )

        # 4. fuse --------------------------------------------------------------

        rank_lists = [
            [pid for pid, _, _ in lst[:s.candidates]]
            for lst in dense_lists
        ]

        if opts.hybrid:
            rank_lists += [
                [pid for pid, _, _ in lst[:s.candidates]]
                for lst in bm25_lists
            ]

        t0 = time.perf_counter()

        fused = rrf(
            rank_lists,
            k=s.rrf_k,
        )

        timings["fuse"] = _ms(t0)

        yield stage(
            "fuse",
            "skipped" if len(rank_lists) == 1 else "done",
            ms=timings["fuse"],
            detail=f"{len(fused)} patents ranked by reciprocal rank fusion",
        )

        fused_pos = {
            pid: i
            for i, (pid, _) in enumerate(fused, start=1)
        }

        fused_score = dict(fused)

        candidates = [
            pid
            for pid, _ in fused[:s.rerank_depth]
        ]

        # 5. rerank ------------------------------------------------------------

        relevance: dict[str, float] = {}
        best_chunk: dict[str, int] = {}

        if opts.rerank and candidates:
            yield stage("rerank", "running")
            t0 = time.perf_counter()

            reranked_n = 0
            try:
                deadline = t0 + s.rerank_budget_s if s.rerank_budget_s > 0 else None
                relevance, best_chunk, reranked_n = yield from self._with_progress(
                    self._rerank,
                    claim,
                    texts,
                    opts,
                    candidates,
                    pool,
                    deadline,
                )
                if reranked_n < len(candidates):
                    degraded.append("rerank")

            except Exception:
                log.exception(
                    "reranker failed, using fusion order"
                )

                relevance = {}
                best_chunk = {}
                degraded.append("rerank")

            timings["rerank"] = _ms(t0)
            log.info("rerank: %d of %d patents in %.1f s (budget %.0f s)", reranked_n, len(candidates), timings["rerank"] / 1000, s.rerank_budget_s)

            yield stage(
                "rerank",
                "done" if relevance and reranked_n == len(candidates) else "degraded",
                ms=timings["rerank"],
                detail=(
                    f"{reranked_n} of {len(candidates)} patents reranked"
                    + ("" if reranked_n == len(candidates) else f" within the {s.rerank_budget_s:.0f} s budget; the rest keep fusion order")
                ),
            )

        else:
            yield stage("rerank", "skipped")

        if relevance:
            scored_ids = [p for p in candidates if p in relevance]
            unscored_ids = [p for p in candidates if p not in relevance]
            ordered = sorted(
                scored_ids,
                key=lambda p: (
                    -relevance[p],
                    fused_pos[p],
                ),
            ) + unscored_ids

        else:
            ordered = list(candidates)

        order_pos = {
            pid: i
            for i, pid in enumerate(ordered, start=1)
        }

        top = ordered[:k]

        # 6. explain -----------------------------------------------------------

        yield stage("explain", "running")
        t0 = time.perf_counter()

        results = yield from self._with_progress(
            self._explain,
            claim,
            texts,
            opts,
            top,
            pool,
            best_chunk,
            dense_rank,
            bm25_rank,
            degraded,
            t0 + s.explain_budget_s if s.explain_budget_s > 0 else None,
        )

        max_rrf = (
            max(fused_score.values())
            if fused_score
            else 1.0
        )

        for item in results:
            pid = item["id"]

            if relevance and pid in relevance:
                item["score"] = round(relevance[pid], 4)
                item["score_kind"] = "rerank"

            else:
                item["score"] = round(
                    fused_score[pid] / max_rrf,
                    4,
                )
                item["score_kind"] = "rrf"

            item["rrf_rank"] = fused_pos[pid]
            item["rerank_rank"] = order_pos[pid]

        timings["explain"] = _ms(t0)

        yield stage(
            "explain",
            "degraded" if "explain" in degraded else "done",
            ms=timings["explain"],
            detail=f"{len(results)} results explained"
            + (" (some use template text because the time budget ran out)" if "explain" in degraded else ""),
        )

        timings["total"] = _ms(started)

        elements = (
            [
                {"n": i, "text": t}
                for i, t in enumerate(texts, start=1)
            ]
            if opts.decompose
            else [
                {
                    "n": 1,
                    "text": claim,
                    "whole_claim": True,
                }
            ]
        )

        yield {
            "type": "result",
            "data": {
                "search_id": "s_" + uuid.uuid4().hex[:12],
                "options": asdict(opts),
                "k": k,
                "degraded": bool(degraded),
                "degraded_stages": degraded,
                "elements": elements,
                "results": results,
                "candidates_considered": len(fused),
                "corpus": {
                    "patents": len(corpus),
                    "chunks": len(corpus.chunks),
                },
                "retrieval": {
                    "dense_backend": getattr(
                        self.dense,
                        "name",
                        "?",
                    ),
                    "rerank_backend": getattr(
                        self.scorer,
                        "name",
                        "?",
                    ),
                    "llm_backend": getattr(
                        self.llm,
                        "name",
                        "?",
                    ),
                    "rrf_k": s.rrf_k,
                },
                "timings_ms": timings,
                "work": self._work_since(work_before),
            },
        }

    # ----------------------------------------------------------------- helpers

    def _work_counters(self) -> dict:
        stats = getattr(self.scorer, "stats", None)
        scorer = stats() if callable(stats) else {"scored": 0, "cached": 0}
        store = getattr(self.llm, "store", None)
        llm = store.cache_stats() if store is not None else {"hits": 0, "misses": 0}
        return {"scored": scorer["scored"], "reused": scorer["cached"], "llm_hits": llm["hits"], "llm_misses": llm["misses"]}

    def _work_since(self, before: dict) -> dict:
        """What this search really computed, and what came from a cache. Counters are shared, so a search that
        overlaps another one on the same server can include some of its work."""
        now = self._work_counters()
        return {
            "rerank_pairs_scored": now["scored"] - before["scored"],
            "rerank_pairs_from_cache": now["reused"] - before["reused"],
            "llm_calls": now["llm_misses"] - before["llm_misses"],
            "llm_from_cache": now["llm_hits"] - before["llm_hits"],
            "rerank_cache": bool(getattr(self.scorer, "cache_enabled", False)),
        }

    def _within(self, budget: float, fn, *args):
        """Run fn but stop waiting after `budget` seconds. The worker keeps going, so an LLM answer that arrives late
        is still cached for the next search. budget <= 0 means no limit."""
        if budget <= 0:
            return fn(*args)
        box: dict = {}
        done = threading.Event()

        def work() -> None:
            try:
                box["value"] = fn(*args)
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        if not done.wait(timeout=max(0.05, budget)):
            raise BudgetExceeded(f"took longer than {budget:.0f}s")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _with_progress(self, fn, *args):
        """Run a slow stage in a worker thread and yield its real progress events as they happen.

        `fn` receives a `progress(stage, done, total, label)` callback. The generator's return value is fn's result.
        """
        events: queue.Queue = queue.Queue()
        box: dict = {}

        def progress(stage: str, done: int, total: int, label: str = "") -> None:
            events.put({"type": "progress", "stage": stage, "done": done, "total": total, "label": label})

        def work() -> None:
            try:
                box["value"] = fn(*args, progress=progress)
            except BaseException as exc:  # re-raised in the caller's thread
                box["error"] = exc
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _score_by_deadline(self, pairs, deadline, step: int = 6):
        """Score pairs a few at a time. Returns None if the deadline passes first, so one slow candidate cannot overrun the budget."""
        out: list[float] = []
        for i in range(0, len(pairs), step):
            if deadline is not None and time.perf_counter() > deadline:
                return None
            out.extend(self.scorer.score_pairs(pairs[i:i + step]))
        return out

    def _rerank(
        self,
        claim,
        texts,
        opts,
        candidates,
        pool,
        deadline=None,
        progress=None,
    ):
        """Cross-encoder rerank, top candidates first. Stops at the deadline and reports how many it finished.

        Scores for the candidates it did reach are exactly what an unlimited run would give them.
        """
        chunks = self.corpus.chunks
        use_elements = bool(opts.decompose and len(texts) > 1)
        relevance: dict[str, float] = {}
        best_chunk: dict[str, int] = {}
        label = "patents reranked by the cross-encoder"

        for n, pid in enumerate(candidates, start=1):
            if deadline is not None and time.perf_counter() > deadline:
                return relevance, best_chunk, n - 1

            chunk_ids = pool[pid]
            claim_scores = self._score_by_deadline([(claim, chunks[ci].text) for ci in chunk_ids], deadline)
            if claim_scores is None:
                return relevance, best_chunk, n - 1
            top_i = max(range(len(chunk_ids)), key=lambda i: claim_scores[i])
            best_chunk[pid] = chunk_ids[top_i]
            score = claim_scores[top_i]

            if use_elements:
                pairs = [(text, chunks[ci].text) for ci in chunk_ids for text in texts]
                flat = self._score_by_deadline(pairs, deadline)
                if flat is None:  # out of time half way through this candidate: leave it unscored, never half-scored
                    best_chunk.pop(pid, None)
                    return relevance, best_chunk, n - 1
                per_el = [
                    max(flat[i * len(texts) + e] for i in range(len(chunk_ids)))
                    for e in range(len(texts))
                ]
                score = 0.5 * score + 0.5 * (sum(per_el) / len(texts))

            relevance[pid] = score
            if progress:
                progress("rerank", n, len(candidates), label)

        return relevance, best_chunk, len(candidates)

    def _coverage(self, texts, pid, pool=None):
        """
        Per-element best score over all chunks of one patent.
        Returns (covered, display_chunk_idx).
        """
        if getattr(getattr(self, "settings", None), "coverage_scope", "pool") != "all" and pool and pool.get(pid):
            # The passages retrieval already surfaced. The reranker scored these pairs, so they come from its cache.
            chunk_ids = list(pool[pid])
        else:
            chunk_ids = self.corpus.by_patent[pid][
                :MAX_COVERAGE_CHUNKS
            ]

        pairs = [
            (t, self.corpus.chunks[ci].text)
            for ci in chunk_ids
            for t in texts
        ]

        scores = self.scorer.score_pairs(pairs)

        n_el = len(texts)

        matrix = [
            scores[i * n_el:(i + 1) * n_el]
            for i in range(len(chunk_ids))
        ]

        threshold = self.scorer.coverage_threshold

        display = max(
            range(len(chunk_ids)),
            key=lambda i: sum(matrix[i]),
        )

        # Coverage must match the exact passage shown as evidence.
        covered = [
            n
            for n in range(1, n_el + 1)
            if matrix[display][n - 1] >= threshold
        ]

        return covered, chunk_ids[display]

    def _explain(
        self,
        claim,
        texts,
        opts,
        top,
        pool,
        best_chunk,
        dense_rank,
        bm25_rank,
        degraded,
        deadline=None,
        progress=None,
    ):
        corpus = self.corpus
        prepared = []
        steps_total = len(top) * (2 if opts.explain else 1)
        steps_done = 0

        for pid in top:
            patent = corpus.patents[pid]

            covered: list[int] = []

            chunk_idx = best_chunk.get(
                pid,
                pool[pid][0]
                if pool.get(pid)
                else corpus.by_patent[pid][0],
            )

            if opts.decompose and opts.explain:
                try:
                    if deadline is not None and time.perf_counter() > deadline:
                        raise BudgetExceeded("explain budget used up")
                    covered, chunk_idx = self._coverage(
                        texts,
                        pid,
                        pool,
                    )

                except Exception:
                    log.exception(
                        "coverage scoring failed"
                    )

                    if "coverage" not in degraded:
                        degraded.append("coverage")

            chunk = corpus.chunks[chunk_idx]

            steps_done += 1
            if progress and opts.explain:
                progress("explain", steps_done, steps_total, "patents checked against the claim elements")

            d_rank = dense_rank.get(pid)
            b_rank = bm25_rank.get(pid)

            missed = (
                (d_rank is None or d_rank > 10)
                and b_rank is not None
                and b_rank <= 10
            )

            marks = (
                {
                    n: texts[n - 1]
                    for n in covered
                }
                if opts.decompose
                else {}
            )

            prepared.append(
                (
                    pid,
                    patent,
                    chunk,
                    covered,
                    d_rank,
                    b_rank,
                    missed,
                    marks,
                )
            )

        def build_why(item, use_llm: bool = True) -> str:
            if not opts.explain:
                return ""

            (
                pid,
                patent,
                chunk,
                covered,
                d_rank,
                b_rank,
                missed,
                _,
            ) = item

            if opts.decompose:
                names = ", ".join(
                    f"E{n}"
                    for n in covered
                )

                if covered:
                    parts = [
                        f"Model match for {names} ({chunk.loc}); this is a relevance signal, not verified disclosure."
                    ]

                    gap = [
                        n
                        for n in range(1, len(texts) + 1)
                        if n not in covered
                    ]

                    if gap:
                        gap_names = ", ".join(
                            f"E{n}"
                            for n in gap
                        )

                        # Offline scoring only sees shared words,
                        # so it must not claim an element is absent.
                        parts.append(
                            f"No clear text match for {gap_names}."
                            if self.scorer.name == "lexical"
                            else f"No strong model match for {gap_names}."
                        )

                else:
                    parts = [
                        (
                            f"No clear text match for any claim element. "
                            f"Closest passage is in {chunk.loc}."
                            if self.scorer.name == "lexical"
                            else
                            f"No strong model match detected for any claim element. "
                            f"Closest passage is in {chunk.loc}."
                        )
                    ]

            else:
                parts = [
                    f"Closest passage to the claim text is in {chunk.loc}."
                ]

            if missed:
                where = (
                    f"rank {d_rank}"
                    if d_rank
                    else "beyond the retrieval depth"
                )

                parts.append(
                    f"Found by keyword search (rank {b_rank}); "
                    f"vector search placed it at {where}."
                )

            template = " ".join(parts)

            if not use_llm:
                return template

            try:
                elements = [
                    (n, t)
                    for n, t in enumerate(texts, start=1)
                ]

                written = self.llm.explain(
                    elements,
                    covered,
                    chunk.text,
                    patent.title,
                )

            except LLMError:
                if "explain" not in degraded:
                    degraded.append("explain")

                written = None

            return (
                f"{written} {parts[-1]}"
                if written and missed
                else (written or template)
            )

        if (
            getattr(self.llm, "name", "") == "heuristic"
            or len(prepared) < 2
        ):
            whys = []
            for item in prepared:
                whys.append(build_why(item))
                steps_done += 1
                if progress and opts.explain:
                    progress("explain", steps_done, steps_total, "explanations written")

        else:
            pool_exec = ThreadPoolExecutor(max_workers=4)
            futures = [pool_exec.submit(build_why, item) for item in prepared]
            whys = []
            for item, future in zip(prepared, futures):
                try:
                    wait = None if deadline is None else max(0.05, deadline - time.perf_counter())
                    whys.append(future.result(timeout=wait))
                except FutureTimeout:
                    whys.append(build_why(item, use_llm=False))  # template text, no LLM
                    if "explain" not in degraded:
                        degraded.append("explain")
                steps_done += 1
                if progress:
                    progress("explain", steps_done, steps_total, "explanations written")
            pool_exec.shutdown(wait=False, cancel_futures=True)

        results = []

        for item, why in zip(prepared, whys):
            (
                pid,
                patent,
                chunk,
                covered,
                d_rank,
                b_rank,
                missed,
                marks,
            ) = item

            results.append(
                {
                    "id": pid,
                    "title": patent.title,
                    "assignee": patent.assignee,
                    "year": patent.year,
                    "cpc": patent.cpc,
                    "el": covered,
                    "dense": d_rank,
                    "bm25": b_rank,
                    "loc": chunk.loc,
                    "passage": highlight(chunk.text, marks),
                    "why": why,
                    "missed_by_vector": bool(missed),
                }
            )

        return results