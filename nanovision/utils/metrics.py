"""
Evaluation metrics for ViT classification and CLIP retrieval.

Functions:
  accuracy()         — Top-K classification accuracy (count, not percent)
  retrieval_recall() — Recall@K for a precomputed similarity matrix
"""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: int = 1) -> int:
    """
    Number of correct predictions in a batch (Top-K).

    Args:
        logits:  (B, n_classes) unnormalised logits.
        targets: (B,) integer class labels.
        topk:    K for top-K accuracy.

    Returns:
        Number of correct predictions (integer, NOT a percentage).
        Divide by len(targets) to get accuracy.
    """
    with torch.no_grad():
        k = min(topk, logits.shape[-1])
        _, top_indices = logits.topk(k, dim=-1)   # (B, k)
        correct = top_indices.eq(targets.unsqueeze(1).expand_as(top_indices))
        return correct.any(dim=-1).sum().item()


def retrieval_recall(
    sim:  torch.Tensor,
    k:    int = 1,
) -> float:
    """
    Recall@K for a ground-truth-diagonal similarity matrix.

    Given an (N, N) cosine similarity matrix where entry [i, j] = similarity
    between query i and database item j, and the ground-truth match for
    query i is item i (diagonal), compute the fraction of queries for
    which the correct item appears in the top-K results.

    Args:
        sim: (N, N) similarity matrix (higher = more similar).
        k:   Recall threshold.

    Returns:
        Recall@K as a percentage (0–100).
    """
    N = sim.shape[0]
    k = min(k, N)
    _, top_indices = sim.topk(k, dim=-1)    # (N, k)
    gt   = torch.arange(N, device=sim.device).unsqueeze(1)   # (N, 1)
    hits = (top_indices == gt).any(dim=-1).float()
    return hits.mean().item() * 100.0


def retrieval_recall_class_aware(
    sim:          torch.Tensor,
    query_labels: torch.Tensor,
    db_labels:    torch.Tensor,
    k:            int = 1,
) -> float:
    """
    Recall@K where a hit is ANY top-K database item sharing the query's
    class label — not strictly the index-diagonal item.

    `retrieval_recall()` assumes a 1:1 diagonal ground truth: query i's only
    correct answer is database item i. That assumption breaks whenever the
    database has duplicate/near-duplicate items for the same query (e.g.
    NanoVision's CLIP captions are a single template per CIFAR-10 class, so
    every image of the same class shares an identical caption embedding).
    In that setting the model can retrieve a perfectly correct same-class
    item and still be scored as "wrong" by the diagonal metric, silently
    deflating recall. This function scores by class label instead, which is
    the metric that is actually meaningful when captions/labels repeat.

    Args:
        sim:          (N_q, N_db) similarity matrix (higher = more similar).
        query_labels: (N_q,) int class label per query row.
        db_labels:    (N_db,) int class label per database column.
        k:            Recall threshold.

    Returns:
        Recall@K as a percentage (0-100).
    """
    k = min(k, sim.shape[1])
    _, top_indices = sim.topk(k, dim=-1)                    # (N_q, k)
    retrieved_labels = db_labels[top_indices]                # (N_q, k)
    hits = (retrieved_labels == query_labels.unsqueeze(1)).any(dim=-1).float()
    return hits.mean().item() * 100.0


def mean_average_precision(
    sim:    torch.Tensor,
    top_k:  int | None = None,
) -> float:
    """
    Mean Average Precision (mAP) for retrieval.

    Assumes ground-truth diagonal structure (same as retrieval_recall).

    Args:
        sim:   (N, N) similarity matrix.
        top_k: Evaluate only the top-K results (None = all).

    Returns:
        mAP as a percentage.
    """
    N = sim.shape[0]
    k = top_k or N

    _, sorted_indices = sim.sort(dim=-1, descending=True)   # (N, N)
    sorted_indices = sorted_indices[:, :k]                  # (N, k)

    aps = []
    for i in range(N):
        # Find rank of ground-truth item i in the sorted list
        hits = (sorted_indices[i] == i).float()   # 1 where correct item appears
        rank = hits.argmax().item() + 1            # 1-indexed rank
        if hits.any():
            aps.append(1.0 / rank)
        else:
            aps.append(0.0)

    return float(sum(aps) / len(aps)) * 100.0


def confusion_matrix(
    preds:     torch.Tensor,
    targets:   torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        preds:     (N,) predicted class indices.
        targets:   (N,) ground-truth class indices.
        n_classes: Number of classes.

    Returns:
        (n_classes, n_classes) int tensor.
    """
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for p, t in zip(preds.tolist(), targets.tolist()):
        cm[t][p] += 1
    return cm
