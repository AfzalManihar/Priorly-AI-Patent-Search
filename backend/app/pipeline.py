"""The search pipeline (PRD section 8).

decompose -> expand (HyDE) -> retrieve (BM25 + dense) -> fuse (RRF) -> rerank -> explain

Every stage has a switch and a measured timing. `run_events` yields progress events and finally the
result payload; `search` returns only the payload.
"""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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

        def stage(name: str, status: str, **extra) -> dict:
            return {
                "type": "stage",
                "stage": name,
                "status": status,
                **extra,
            }

        # 1. decompose ---------------------------------------------------------

        if opts.decompose:
            yield stage("decompose", "running")
            t0 = time.perf_counter()

            try:
                texts = self.llm.decompose(claim)

            except LLMError as exc:
                log.warning(
                    "decomposition failed, falling back to rules: %s", str(exc)[:200]
                )
                texts = heuristic_decompose(claim)
                degraded.append("decompose")

            timings["decompose"] = _ms(t0)

            yield stage(
                "decompose",
                "done",
                ms=timings["decompose"],
            )

        else:
            texts = [claim]
            yield stage("decompose", "skipped")

        # 2. expand (HyDE) -----------------------------------------------------

        if opts.hyde:
            yield stage("hyde", "running")
            t0 = time.perf_counter()

            dense_queries = []

            for text in texts:
                try:
                    dense_queries.append(self.llm.hyde(text))

                except LLMError as exc:
                    log.warning(
                        "HyDE failed for one element, using the element text: %s", str(exc)[:200]
                    )
                    dense_queries.append(text)

                    if "hyde" not in degraded:
                        degraded.append("hyde")

            timings["hyde"] = _ms(t0)

            yield stage(
                "hyde",
                "done",
                ms=timings["hyde"],
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
                    self.dense.search(dq, s.trace_depth, cons),
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

        yield stage(
            "retrieve",
            "done",
            ms=timings["retrieve"],
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

            try:
                relevance, best_chunk = self._rerank(
                    claim,
                    texts,
                    opts,
                    candidates,
                    pool,
                )

            except Exception:
                log.exception(
                    "reranker failed, using fusion order"
                )

                relevance = {}
                best_chunk = {}
                degraded.append("rerank")

            timings["rerank"] = _ms(t0)

            yield stage(
                "rerank",
                "done" if relevance else "degraded",
                ms=timings["rerank"],
            )

        else:
            yield stage("rerank", "skipped")

        if relevance:
            ordered = sorted(
                candidates,
                key=lambda p: (
                    -relevance[p],
                    fused_pos[p],
                ),
            )

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

        results = self._explain(
            claim,
            texts,
            opts,
            top,
            pool,
            best_chunk,
            dense_rank,
            bm25_rank,
            degraded,
        )

        max_rrf = (
            max(fused_score.values())
            if fused_score
            else 1.0
        )

        for item in results:
            pid = item["id"]

            if relevance:
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
            "done",
            ms=timings["explain"],
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
            },
        }

    # ----------------------------------------------------------------- helpers

    def _rerank(
        self,
        claim,
        texts,
        opts,
        candidates,
        pool,
    ):
        chunks = self.corpus.chunks

        claim_pairs = []
        claim_index = []

        for pid in candidates:
            for ci in pool[pid]:
                claim_pairs.append(
                    (claim, chunks[ci].text)
                )
                claim_index.append((pid, ci))

        claim_best: dict[str, float] = {}
        best_chunk: dict[str, int] = {}

        for (pid, ci), score in zip(
            claim_index,
            self.scorer.score_pairs(claim_pairs),
        ):
            if pid not in claim_best or score > claim_best[pid]:
                claim_best[pid] = score
                best_chunk[pid] = ci

        relevance = dict(claim_best)

        if opts.decompose and len(texts) > 1:
            el_pairs = []
            el_index = []

            for pid in candidates:
                for ci in pool[pid]:
                    for n, text in enumerate(texts, start=1):
                        el_pairs.append(
                            (text, chunks[ci].text)
                        )
                        el_index.append((pid, n))

            per_element: dict[tuple[str, int], float] = {}

            for key, score in zip(
                el_index,
                self.scorer.score_pairs(el_pairs),
            ):
                if score > per_element.get(key, -1.0):
                    per_element[key] = score

            for pid in candidates:
                mean = (
                    sum(
                        per_element.get((pid, n), 0.0)
                        for n in range(1, len(texts) + 1)
                    )
                    / len(texts)
                )

                relevance[pid] = (
                    0.5 * claim_best[pid]
                    + 0.5 * mean
                )

        return relevance, best_chunk

    def _coverage(self, texts, pid):
        """
        Per-element best score over all chunks of one patent.
        Returns (covered, display_chunk_idx).
        """
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
    ):
        corpus = self.corpus
        prepared = []

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
                    covered, chunk_idx = self._coverage(
                        texts,
                        pid,
                    )

                except Exception:
                    log.exception(
                        "coverage scoring failed"
                    )

                    if "coverage" not in degraded:
                        degraded.append("coverage")

            chunk = corpus.chunks[chunk_idx]

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

        def build_why(item) -> str:
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
                        f"Covers {names} ({chunk.loc})."
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
                            else f"Does not clearly cover {gap_names}."
                        )

                else:
                    parts = [
                        (
                            f"No clear text match for any claim element. "
                            f"Closest passage is in {chunk.loc}."
                            if self.scorer.name == "lexical"
                            else
                            f"No claim element is clearly covered. "
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
            whys = [
                build_why(item)
                for item in prepared
            ]

        else:
            with ThreadPoolExecutor(max_workers=4) as pool_exec:
                whys = list(
                    pool_exec.map(
                        build_why,
                        prepared,
                    )
                )

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