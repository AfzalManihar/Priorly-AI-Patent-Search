
"""Keyword (BM25) and dense retrievers.

Both return chunk-level hits: list of (chunk_idx, score), best first.
Patent-level aggregation (max chunk score per patent) happens in the pipeline.

Dense backends
  tfidf : offline stand-in built on scikit-learn.
  bge   : BGE embeddings stored and searched in Qdrant.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Corpus
from .tokenizer import tokenize


class DependencyError(RuntimeError):
    """A backing service (vector store, model) is unavailable."""


@dataclass(frozen=True)
class Constraints:
    """Leakage controls: which patents may appear in results."""

    exclude_ids: frozenset = frozenset()
    max_year: int | None = None


def _top(
    scores: np.ndarray,
    mask: np.ndarray,
    n: int,
    positive_only: bool,
) -> list[tuple[int, float]]:
    masked = np.where(mask, scores, -np.inf)
    n = min(n, int(mask.sum()))

    if n <= 0:
        return []

    idx = np.argpartition(-masked, n - 1)[:n]
    idx = idx[np.argsort(-masked[idx], kind="stable")]

    hits = []
    for i in idx:
        value = float(masked[i])
        if not np.isfinite(value) or (positive_only and value <= 0):
            continue
        hits.append((int(i), value))

    return hits


class BM25Index:
    name = "bm25"

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi(
            [tokenize(c.text) or ["_empty_"] for c in corpus.chunks]
        )

        # Build an inverted index once; queries only visit matching documents.
        postings = {}
        for doc_idx, frequencies in enumerate(self.bm25.doc_freqs):
            for token, frequency in frequencies.items():
                postings.setdefault(token, ([], []))
                postings[token][0].append(doc_idx)
                postings[token][1].append(frequency)

        self.postings = {
            token: (
                np.asarray(indices, dtype=np.int32),
                np.asarray(freqs, dtype=float),
            )
            for token, (indices, freqs) in postings.items()
        }
        self.doc_len = np.asarray(self.bm25.doc_len, dtype=float)

    def search(
        self,
        query: str,
        n: int,
        cons: Constraints,
    ) -> list[tuple[int, float]]:
        tokens = tokenize(query)

        if not tokens:
            return []

        scores = np.zeros(self.bm25.corpus_size, dtype=float)

        for token in tokens:
            posting = self.postings.get(token)
            if posting is None:
                continue

            indices, frequencies = posting
            idf = self.bm25.idf.get(token) or 0
            if idf == 0:
                continue

            denominator = frequencies + self.bm25.k1 * (
                1 - self.bm25.b
                + self.bm25.b * self.doc_len[indices] / self.bm25.avgdl
            )
            scores[indices] += idf * (
                frequencies * (self.bm25.k1 + 1) / denominator
            )

        mask = self.corpus.mask(cons.exclude_ids, cons.max_year)

        return _top(scores, mask, n, positive_only=True)


class TfidfDense:
    name = "tfidf"
    label = "tfidf (offline stand-in for dense retrieval)"

    def __init__(self, corpus: Corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [c.text for c in corpus.chunks]
        )

    def search(
        self,
        query: str,
        n: int,
        cons: Constraints,
    ) -> list[tuple[int, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        mask = self.corpus.mask(cons.exclude_ids, cons.max_year)

        return _top(scores, mask, n, positive_only=True)


class HashEmbedder:
    """Deterministic hashing embedder used by tests."""

    dim = 256

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = (
                    int(hashlib.md5(token.encode()).hexdigest(), 16)
                    % self.dim
                )
                out[row, bucket] += 1.0

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        return out / norms


class SentenceTransformerEmbedder:
    QUERY_PREFIX = (
        "Represent this sentence for searching relevant passages: "
    )

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DependencyError(
                "Install requirements-ml.txt to use the bge backend"
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.use_prefix = (
            "bge" in model_name.lower()
            and "en" in model_name.lower()
        )

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> np.ndarray:
        if is_query and self.use_prefix:
            texts = [
                self.QUERY_PREFIX + text
                for text in texts
            ]

        return np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
            ),
            dtype=np.float32,
        )


def make_embedder(model_name: str):
    return (
        HashEmbedder()
        if model_name == "hash"
        else SentenceTransformerEmbedder(model_name)
    )


class QdrantDense:
    name = "bge"
    label = "dense embeddings in Qdrant"

    def __init__(
        self,
        corpus: Corpus,
        embedder,
        url: str,
        collection: str,
    ):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise DependencyError(
                "qdrant-client is not installed"
            ) from exc

        self.models = models
        self.corpus = corpus
        self.embedder = embedder
        self.collection = collection

        try:
            if url == ":memory:":
                self.client = QdrantClient(location=":memory:")
            elif url.startswith(("http://", "https://")):
                self.client = QdrantClient(url=url)
            else:
                self.client = QdrantClient(path=url)

            self.ensure_index()

        except DependencyError:
            raise
        except Exception as exc:
            raise DependencyError(
                f"Vector store unavailable: {type(exc).__name__}"
            ) from exc

    def ensure_index(self, batch: int = 64) -> None:
        m = self.models
        total = len(self.corpus.chunks)

        if self.client.collection_exists(self.collection):
            existing = self.client.count(
                self.collection,
                exact=True,
            ).count

            if existing == total:
                return

            self.client.delete_collection(self.collection)

        self.client.create_collection(
            self.collection,
            vectors_config=m.VectorParams(
                size=int(self.embedder.dim),
                distance=m.Distance.COSINE,
            ),
        )

        for batch_no, start in enumerate(range(0, total, batch), start=1):
            part = self.corpus.chunks[start:start + batch]

            vectors = self.embedder.encode(
                [c.text for c in part]
            )

            self.client.upsert(
                self.collection,
                points=[
                    m.PointStruct(
                        id=c.idx,
                        vector=vectors[i].tolist(),
                        payload={
                            "patent_id": c.patent_id,
                            "year": self.corpus.patents[
                                c.patent_id
                            ].year,
                        },
                    )
                    for i, c in enumerate(part)
                ],
            )
            if batch_no == 1 or batch_no % 25 == 0 or start + len(part) >= total:
                print(f"Qdrant indexing: {min(start + len(part), total)}/{total} chunks", flush=True)

    def search(
        self,
        query: str,
        n: int,
        cons: Constraints,
    ) -> list[tuple[int, float]]:
        m = self.models
        must = []
        must_not = []

        if cons.max_year is not None:
            must.append(
                m.FieldCondition(
                    key="year",
                    range=m.Range(lte=cons.max_year),
                )
            )

        if cons.exclude_ids:
            must_not.append(
                m.FieldCondition(
                    key="patent_id",
                    match=m.MatchAny(
                        any=sorted(cons.exclude_ids)
                    ),
                )
            )

        flt = (
            m.Filter(
                must=must or None,
                must_not=must_not or None,
            )
            if (must or must_not)
            else None
        )

        try:
            vector = self.embedder.encode(
                [query],
                is_query=True,
            )[0].tolist()

            result = self.client.query_points(
                self.collection,
                query=vector,
                limit=n,
                query_filter=flt,
            )

        except Exception as exc:
            raise DependencyError(
                f"Vector store unavailable: {type(exc).__name__}"
            ) from exc

        return [
            (int(point.id), float(point.score))
            for point in result.points
        ]


def build_dense(settings, corpus: Corpus):
    if settings.dense_backend == "tfidf":
        return TfidfDense(corpus)

    if settings.dense_backend == "bge":
        return QdrantDense(
            corpus,
            make_embedder(settings.embed_model),
            settings.qdrant_url,
            settings.qdrant_collection,
        )

    raise ValueError(
        f"Unknown DENSE_BACKEND: {settings.dense_backend}"
    )


def aggregate_to_patents(
    corpus: Corpus,
    hits: list[tuple[int, float]],
) -> list[tuple[str, float, list[int]]]:
    """Convert chunk hits into patent rankings."""

    best: dict[str, float] = {}
    chunks: dict[str, list[int]] = {}

    for idx, score in hits:
        pid = corpus.chunks[idx].patent_id

        if pid not in best:
            best[pid] = score
            chunks[pid] = []

        chunks[pid].append(idx)

    ordered = sorted(
        best,
        key=lambda patent: (-best[patent], patent),
    )

    return [
        (patent, best[patent], chunks[patent])
        for patent in ordered
    ]

