"""
Unit tests for retrieval module (EmbeddingIndex) and utility metrics.

Tests use synthetic embeddings — no CIFAR-10 data required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from nanovision.retrieval.index import EmbeddingIndex, RetrievalResult
from nanovision.utils.metrics import accuracy, retrieval_recall, mean_average_precision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 32
N         = 20


def _make_index(n: int = N, d: int = EMBED_DIM) -> EmbeddingIndex:
    """Build an index populated with random (but normalised) embeddings."""
    img   = F.normalize(torch.randn(n, d), dim=-1)
    text  = F.normalize(torch.randn(n, d), dim=-1)
    labels = torch.arange(n) % 10

    idx = EmbeddingIndex(embed_dim=d)
    idx.image_feats = img
    idx.text_feats  = text
    idx.labels      = labels
    idx.texts       = [f"text {i}" for i in range(n)]
    idx.n_items     = n
    return idx


def _make_identity_index(n: int = N, d: int = EMBED_DIM) -> EmbeddingIndex:
    """
    Index where image[i] == text[i] exactly — recall@1 must be 100%.
    Uses random but fixed orthonormal-ish vectors so diagonal is the maximum.
    """
    feats = F.normalize(torch.randn(n, d), dim=-1)
    labels = torch.arange(n) % 10

    idx = EmbeddingIndex(embed_dim=d)
    idx.image_feats = feats
    idx.text_feats  = feats.clone()
    idx.labels      = labels
    idx.texts       = [f"text {i}" for i in range(n)]
    idx.n_items     = n
    return idx


# ---------------------------------------------------------------------------
# EmbeddingIndex construction
# ---------------------------------------------------------------------------

class TestEmbeddingIndexInit:
    def test_init_stores_embed_dim(self):
        idx = EmbeddingIndex(embed_dim=64)
        assert idx.embed_dim == 64

    def test_empty_index_len(self):
        idx = EmbeddingIndex(embed_dim=32)
        assert len(idx) == 0

    def test_populated_index_len(self):
        idx = _make_index()
        assert len(idx) == N

    def test_repr(self):
        idx = _make_index()
        r = repr(idx)
        assert str(N) in r


# ---------------------------------------------------------------------------
# query_text (text feat -> images)
# ---------------------------------------------------------------------------

class TestQueryText:
    def test_returns_retrieval_result(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=5)
        assert isinstance(res, RetrievalResult)

    def test_top_k_indices_count(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=5)
        assert len(res.indices) == 5

    def test_scores_descending(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=10)
        scores = res.scores.tolist()
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_cosine_range(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=5)
        assert (res.scores >= -1.0001).all()
        assert (res.scores <=  1.0001).all()

    def test_top_k_capped_at_n_items(self):
        idx  = _make_index(n=5)
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=100)
        assert len(res.indices) == 5

    def test_labels_returned(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_text(feat, top_k=3)
        assert res.labels is not None
        assert len(res.labels) == 3


# ---------------------------------------------------------------------------
# query_image (image feat -> texts)
# ---------------------------------------------------------------------------

class TestQueryImage:
    def test_returns_retrieval_result(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_image(feat, top_k=5)
        assert isinstance(res, RetrievalResult)

    def test_scores_descending(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_image(feat, top_k=10)
        scores = res.scores.tolist()
        assert scores == sorted(scores, reverse=True)

    def test_texts_returned(self):
        idx  = _make_index()
        feat = F.normalize(torch.randn(EMBED_DIM), dim=-1)
        res  = idx.query_image(feat, top_k=3)
        assert res.texts is not None
        assert len(res.texts) == 3


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_identity_index_recall_at_1_is_100(self):
        idx = _make_identity_index()
        r   = idx.recall_at_k(k=1)
        assert r == pytest.approx(100.0, abs=0.1)

    def test_identity_index_recall_at_5_is_100(self):
        idx = _make_identity_index()
        r   = idx.recall_at_k(k=5)
        assert r == pytest.approx(100.0, abs=0.1)

    def test_recall_between_0_and_100(self):
        idx = _make_index()
        for k in (1, 5, 10):
            r = idx.recall_at_k(k)
            assert 0.0 <= r <= 100.0

    def test_recall_increases_with_k(self):
        idx = _make_index()
        r1  = idx.recall_at_k(1)
        r5  = idx.recall_at_k(5)
        r10 = idx.recall_at_k(10)
        assert r1 <= r5 <= r10

    def test_t2i_direction(self):
        idx = _make_identity_index()
        r   = idx.recall_at_k(k=1, direction="t2i")
        assert r == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# add_image
# ---------------------------------------------------------------------------

class TestAddImage:
    def test_add_single_image_increases_count(self):
        idx  = EmbeddingIndex(embed_dim=EMBED_DIM)
        feat = torch.randn(EMBED_DIM)
        idx.add_image(feat, label=3, text="a cat")
        assert idx.n_items == 1
        assert len(idx) == 1

    def test_added_image_is_normalised(self):
        idx  = EmbeddingIndex(embed_dim=EMBED_DIM)
        feat = torch.randn(EMBED_DIM) * 100.0   # Deliberately not unit norm
        idx.add_image(feat, label=0)
        norm = idx.image_feats[0].norm().item()
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        idx  = _make_index()
        path = tmp_path / "test_index.pt"
        idx.save(path)

        loaded = EmbeddingIndex.load(path)
        assert loaded.n_items     == idx.n_items
        assert loaded.embed_dim   == idx.embed_dim
        assert len(loaded.texts)  == len(idx.texts)
        assert torch.allclose(loaded.image_feats, idx.image_feats)
        assert torch.allclose(loaded.text_feats,  idx.text_feats)

    def test_loaded_recall_matches(self, tmp_path):
        idx  = _make_identity_index()
        path = tmp_path / "identity.pt"
        idx.save(path)

        loaded = EmbeddingIndex.load(path)
        r_orig   = idx.recall_at_k(1)
        r_loaded = loaded.recall_at_k(1)
        assert abs(r_orig - r_loaded) < 1e-3


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

class TestAccuracy:
    def _logits(self, labels: list[int], n_classes: int = 10) -> torch.Tensor:
        """Produce logits where the argmax equals `labels`."""
        B = len(labels)
        logits = torch.zeros(B, n_classes)
        for i, l in enumerate(labels):
            logits[i, l] = 10.0
        return logits

    def test_top1_all_correct(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits = self._logits([0, 1, 2, 3])
        assert accuracy(logits, labels, topk=1) == 4

    def test_top1_none_correct(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits = self._logits([1, 2, 3, 4])    # All wrong
        assert accuracy(logits, labels, topk=1) == 0

    def test_top5_includes_top1(self):
        labels = torch.tensor([0])
        logits = self._logits([0])
        assert accuracy(logits, labels, topk=5) >= accuracy(logits, labels, topk=1)

    def test_top5_returns_int(self):
        labels = torch.zeros(4, dtype=torch.long)
        logits = torch.randn(4, 10)
        result = accuracy(logits, labels, topk=5)
        assert isinstance(result, int)


class TestRetrievalRecall:
    def test_perfect_recall(self):
        n = 8
        sim = torch.eye(n)   # diagonal = 1, off-diagonal = 0
        # Recall @1 should be 100%
        r = retrieval_recall(sim, k=1)
        assert r == pytest.approx(100.0, abs=0.1)

    def test_zero_recall(self):
        n = 4
        sim = 1.0 - torch.eye(n)   # positive pairs have score 0, negatives have 1
        r = retrieval_recall(sim, k=1)
        assert r == pytest.approx(0.0, abs=0.1)

    def test_recall_increases_with_k(self):
        n = 10
        sim = torch.rand(n, n)
        r1 = retrieval_recall(sim, k=1)
        r5 = retrieval_recall(sim, k=5)
        assert r1 <= r5


class TestMAP:
    def test_perfect_map(self):
        n = 6
        sim = torch.eye(n)
        mAP = mean_average_precision(sim)
        assert mAP == pytest.approx(100.0, abs=0.1)

    def test_map_between_0_and_100(self):
        n   = 8
        sim = torch.rand(n, n)
        mAP = mean_average_precision(sim)
        assert 0.0 <= mAP <= 100.0
