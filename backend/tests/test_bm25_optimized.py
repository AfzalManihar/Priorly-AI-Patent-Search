from app.corpus import Corpus, Patent
from app.retrievers import BM25Index, Constraints, _top
from app.tokenizer import tokenize
import numpy as np


def make_index():
    texts = [
        "A battery with an electrode and casing for storing energy.",
        "A solar cell with an electrode and semiconductor layer.",
        "A battery pack with improved energy storage and casing.",
        "A semiconductor device for electronic circuits.",
    ]

    patents = [
        Patent(
            f"US {i + 1},000,001",
            f"US {i + 1},000,001",
            "Test Co",
            2010 + i,
            "X",
            "",
            [],
            [{"no": "0001", "text": text}],
        )
        for i, text in enumerate(texts)
    ]

    corpus = Corpus(patents)
    return corpus, BM25Index(corpus)


def test_optimized_bm25_matches_original_scores_and_ranking():
    corpus, index = make_index()

    for query in (
        "battery electrode",
        "battery battery energy",
        "unknownword battery",
        "US 1,000,001",
    ):
        expected = _top(
            index.bm25.get_scores(tokenize(query)),
            corpus.mask(),
            10,
            positive_only=True,
        )
        actual = index.search(query, 10, Constraints())

        assert [i for i, _ in actual] == [i for i, _ in expected]
        assert np.allclose(
            [score for _, score in actual],
            [score for _, score in expected],
            rtol=1e-12,
            atol=1e-12,
        )


def test_optimized_bm25_preserves_leakage_constraints():
    corpus, index = make_index()
    constraints = Constraints(
        exclude_ids=frozenset({"US 1,000,001"}),
        max_year=2011,
    )
    query = "battery electrode"

    expected = _top(
        index.bm25.get_scores(tokenize(query)),
        corpus.mask(
            constraints.exclude_ids,
            constraints.max_year,
        ),
        10,
        positive_only=True,
    )
    actual = index.search(query, 10, constraints)

    assert [i for i, _ in actual] == [i for i, _ in expected]
    assert all(
        corpus.chunks[i].patent_id != "US 1,000,001"
        for i, _ in actual
    )
    assert all(corpus.chunks[i].year <= 2011 for i, _ in actual)
