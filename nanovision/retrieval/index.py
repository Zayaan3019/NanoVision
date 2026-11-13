"""
Cross-modal embedding store and retrieval.

EmbeddingIndex provides:
  - add()        : encode and store image or text embeddings
  - query_text() : retrieve top-K images given a text query
  - query_image(): retrieve top-K texts given an image query
  - save() / load(): persist the index to disk

All similarity is cosine (dot product of L2-normalised vectors),
computed with exact brute-force search (O(N x D)) which is fast
enough for CIFAR-10's 10,000 test images.  For larger corpora an
approximate method (e.g. HNSW) would replace the matmul.

The index stores:
  image_feats  : (N, embed_dim) float32 tensor, L2-normalised
  text_feats   : (N, embed_dim) float32 tensor, L2-normalised
  labels       : (N,) int tensor  — class indices
  texts        : list[str]        — raw text strings
  image_paths  : list[str|None]   — source paths (None if in-memory)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

class RetrievalResult:
    """Container for a single retrieval result."""

    def __init__(
        self,
        indices:    torch.Tensor,   # (K,) indices into the stored embeddings
        scores:     torch.Tensor,   # (K,) cosine similarity scores
        labels:     Optional[torch.Tensor] = None,   # (K,) class labels
        texts:      Optional[list[str]]    = None,   # corresponding text strings
    ) -> None:
        self.indices = indices
        self.scores  = scores
        self.labels  = labels
        self.texts   = texts

    def __repr__(self) -> str:
        items = []
        for i, (idx, score) in enumerate(zip(self.indices.tolist(), self.scores.tolist())):
            label = self.labels[i].item() if self.labels is not None else "?"
            text  = self.texts[i] if self.texts is not None else ""
            items.append(f"  [{i}] idx={idx:5d} score={score:.4f} label={label} text={text!r}")
        return "RetrievalResult(\n" + "\n".join(items) + "\n)"


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """
    Brute-force cross-modal retrieval index.

    Stores L2-normalised image and text embeddings and supports
    top-K retrieval by cosine similarity in both directions.

    Usage:
        index = EmbeddingIndex(embed_dim=128)

        # Build index from a CLIP model and dataset
        index.build_from_clip(model, test_loader, device)

        # Query: text -> images
        result = index.query_text("a photo of a cat", model, device, top_k=5)
        print(result)

        # Query: image -> texts
        result = index.query_image(image_tensor, top_k=5)
        print(result)
    """

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim   = embed_dim
        self.image_feats: Optional[torch.Tensor] = None   # (N, D)
        self.text_feats:  Optional[torch.Tensor] = None   # (N, D)
        self.labels:      Optional[torch.Tensor] = None   # (N,)
        self.texts:       list[str]              = []
        self.n_items:     int                    = 0

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_from_clip(
        self,
        model:      "nanovision.model.clip.CLIP",
        dataloader: "torch.utils.data.DataLoader",
        device:     torch.device,
        verbose:    bool = True,
    ) -> None:
        """
        Encode all image-text pairs from `dataloader` using a trained CLIP model
        and store the resulting embeddings.

        Expects `dataloader` to yield (images, tokens, labels) tuples
        (as returned by CIFAR10CLIPDataset).
        """
        model.eval()
        model.to(device)

        all_img:   list[torch.Tensor] = []
        all_text:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        raw_texts: list[str]           = []

        for batch in dataloader:
            images, tokens, labels = batch
            images = images.to(device)
            tokens = tokens.to(device)

            img_feats  = model.encode_image(images)   # (B, D)
            text_feats = model.encode_text(tokens)    # (B, D)

            all_img.append(img_feats.cpu())
            all_text.append(text_feats.cpu())
            all_labels.append(labels)

            # Decode raw text for display
            for tok_ids in tokens.cpu().tolist():
                raw_texts.append(model.tokenizer.decode(tok_ids))

        self.image_feats = torch.cat(all_img,   dim=0)   # (N, D)
        self.text_feats  = torch.cat(all_text,  dim=0)   # (N, D)
        self.labels      = torch.cat(all_labels, dim=0)  # (N,)
        self.texts       = raw_texts
        self.n_items     = self.image_feats.shape[0]

        if verbose:
            print(f"EmbeddingIndex built: {self.n_items:,} items, embed_dim={self.embed_dim}")

    def add_image(self, feat: torch.Tensor, label: int = -1, text: str = "") -> None:
        """Add a single (already L2-normalised) image embedding."""
        feat = F.normalize(feat.cpu().unsqueeze(0), dim=-1)  # (1, D)
        self.image_feats = feat if self.image_feats is None else torch.cat(
            [self.image_feats, feat], dim=0
        )
        self.labels = torch.tensor([label]) if self.labels is None else torch.cat(
            [self.labels, torch.tensor([label])], dim=0
        )
        self.texts.append(text)
        self.n_items += 1

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_image(
        self,
        img_feat: torch.Tensor,
        top_k:    int = 10,
    ) -> RetrievalResult:
        """
        Retrieve the top-K stored text embeddings most similar to `img_feat`.

        Args:
            img_feat: (D,) or (1, D) L2-normalised image embedding.
            top_k:    Number of results to return.

        Returns:
            RetrievalResult with matched text indices, scores, labels, and texts.
        """
        assert self.text_feats is not None, "Index is empty — call build_from_clip first."

        q = F.normalize(img_feat.cpu().view(1, -1), dim=-1)   # (1, D)
        sim = (q @ self.text_feats.t()).squeeze(0)             # (N,)

        k       = min(top_k, self.n_items)
        scores, indices = sim.topk(k)

        return RetrievalResult(
            indices=indices,
            scores=scores,
            labels=self.labels[indices] if self.labels is not None else None,
            texts=[self.texts[i] for i in indices.tolist()],
        )

    def query_text(
        self,
        text_feat: torch.Tensor,
        top_k:     int = 10,
    ) -> RetrievalResult:
        """
        Retrieve the top-K stored image embeddings most similar to `text_feat`.

        Args:
            text_feat: (D,) or (1, D) L2-normalised text embedding.
            top_k:     Number of results to return.

        Returns:
            RetrievalResult with matched image indices and scores.
        """
        assert self.image_feats is not None, "Index is empty — call build_from_clip first."

        q = F.normalize(text_feat.cpu().view(1, -1), dim=-1)  # (1, D)
        sim = (q @ self.image_feats.t()).squeeze(0)             # (N,)

        k       = min(top_k, self.n_items)
        scores, indices = sim.topk(k)

        return RetrievalResult(
            indices=indices,
            scores=scores,
            labels=self.labels[indices] if self.labels is not None else None,
            texts=[self.texts[i] for i in indices.tolist()],
        )

    def query_text_string(
        self,
        text:   str,
        model:  "nanovision.model.clip.CLIP",
        device: torch.device,
        top_k:  int = 10,
    ) -> RetrievalResult:
        """
        Encode a raw text string and retrieve top-K matching images.

        Convenience wrapper around encode_texts_from_strings + query_text.
        """
        model.eval()
        with torch.no_grad():
            text_feat = model.encode_texts_from_strings([text], device)  # (1, D)
        return self.query_text(text_feat.squeeze(0), top_k=top_k)

    # ------------------------------------------------------------------
    # Batch retrieval statistics
    # ------------------------------------------------------------------

    def recall_at_k(
        self,
        k:           int = 1,
        direction:   str = "i2t",
        class_aware: bool = False,
    ) -> float:
        """
        Compute Recall@K for the full index.

        Args:
            k:           Recall threshold.
            direction:   "i2t" (image->text) or "t2i" (text->image).
            class_aware: If False (default), ground truth is strictly the
                index-diagonal item (query i's only correct answer is item
                i). If True, a hit is any top-K item sharing the query's
                class `label` — the metric to use whenever captions/labels
                repeat across items (e.g. one caption template per class),
                which makes the diagonal assumption invalid. See
                `nanovision.utils.metrics.retrieval_recall_class_aware`.

        Returns:
            Recall@K as a percentage.
        """
        assert self.image_feats is not None and self.text_feats is not None

        if direction == "i2t":
            q_feats = self.image_feats    # (N, D)
            db_feats = self.text_feats   # (N, D)
        else:
            q_feats = self.text_feats
            db_feats = self.image_feats

        sim = q_feats @ db_feats.t()    # (N, N) — ground truth on diagonal

        if class_aware:
            assert self.labels is not None, "class_aware recall requires labels"
            from nanovision.utils.metrics import retrieval_recall_class_aware
            return retrieval_recall_class_aware(sim, self.labels, self.labels, k)

        # Tie-aware: identical captions give identical scores, and which tied
        # item `topk` returns differs between PyTorch versions.
        from nanovision.utils.metrics import strict_recall_tie_aware
        return strict_recall_tie_aware(sim, k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the index to a .pt file."""
        torch.save(
            {
                "embed_dim":   self.embed_dim,
                "image_feats": self.image_feats,
                "text_feats":  self.text_feats,
                "labels":      self.labels,
                "texts":       self.texts,
                "n_items":     self.n_items,
            },
            path,
        )
        print(f"Index saved to {path} ({self.n_items:,} items)")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        """Load an index previously saved with .save()."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        idx = cls(embed_dim=d["embed_dim"])
        idx.image_feats = d["image_feats"]
        idx.text_feats  = d["text_feats"]
        idx.labels      = d["labels"]
        idx.texts       = d["texts"]
        idx.n_items     = d["n_items"]
        print(f"Index loaded from {path} ({idx.n_items:,} items)")
        return idx

    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        return f"EmbeddingIndex(n_items={self.n_items}, embed_dim={self.embed_dim})"
