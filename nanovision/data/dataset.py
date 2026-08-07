"""
CIFAR-10 dataset loading and image-text pair generation.

Two dataset classes:
  CIFAR10Dataset     — standard image classification (for ViT training)
  CIFAR10CLIPDataset — image + text pairs for contrastive CLIP training

CIFAR-10 binary format (from https://www.cs.toronto.edu/~kriz/cifar.html):
  Each record = 1 label byte + 3072 pixel bytes (1024 R, 1024 G, 1024 B)
  10,000 records per file, 5 training files + 1 test file.

Downloads are handled by `prepare_data.py`; this module just reads the
already-extracted files from `data/cifar-10-batches-py/`.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nanovision.data.augmentation import (
    Compose,
    cifar10_train_transform,
    cifar10_val_transform,
    clip_train_transform,
)
from nanovision.model.text_encoder import ByteTokenizer
from nanovision.utils.seed import seed_worker


# ---------------------------------------------------------------------------
# CIFAR-10 class metadata
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Richer prompt templates for CLIP (multiple prompts per class → averaged embedding)
CLIP_PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "a photograph of a {}.",
    "an image of a {}.",
    "a {} in a photo.",
    "this is a {}.",
    "a close-up photo of a {}.",
    "a rendering of a {}.",
]

# ---------------------------------------------------------------------------
# Raw CIFAR-10 binary reader
# ---------------------------------------------------------------------------

def _load_cifar10_batch(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single CIFAR-10 batch file.

    Returns:
        images: (N, 32, 32, 3) uint8 array.
        labels: (N,) int array with class indices 0–9.
    """
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    imgs   = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,H,W,C)
    labels = np.array(d[b"labels"], dtype=np.int64)
    return imgs, labels


def load_cifar10(
    data_dir: str | Path,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all CIFAR-10 images for a given split.

    Args:
        data_dir: Root data directory (contains cifar-10-batches-py/).
        split:    "train" or "test".

    Returns:
        images: (N, 32, 32, 3) uint8 array.
        labels: (N,) int64 array.
    """
    batch_dir = Path(data_dir) / "cifar-10-batches-py"

    if split == "train":
        batch_files = [batch_dir / f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        batch_files = [batch_dir / "test_batch"]
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    imgs_list, labels_list = [], []
    for bf in batch_files:
        imgs, labels = _load_cifar10_batch(bf)
        imgs_list.append(imgs)
        labels_list.append(labels)

    return np.concatenate(imgs_list, axis=0), np.concatenate(labels_list, axis=0)


# ---------------------------------------------------------------------------
# CIFAR-10 classification dataset
# ---------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 image classification dataset.

    Returns (image_tensor, label) pairs where image_tensor is a float32
    tensor (C, H, W) in the range appropriate for the specified transform.

    Args:
        data_dir:  Root data directory.
        split:     "train" or "test".
        transform: Image transform pipeline (defaults to standard CIFAR-10 aug).
    """

    def __init__(
        self,
        data_dir:  str | Path,
        split:     str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.split    = split
        self.images, self.labels = load_cifar10(data_dir, split)
        self.transform = transform or (
            cifar10_train_transform() if split == "train" else cifar10_val_transform()
        )
        self.classes = CIFAR10_CLASSES

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_np  = self.images[idx]                  # (H, W, C) uint8
        pil_img = Image.fromarray(img_np, mode="RGB")
        tensor  = self.transform(pil_img)            # (C, H, W) float32
        label   = int(self.labels[idx])
        return tensor, label

    def __repr__(self) -> str:
        return (
            f"CIFAR10Dataset(split={self.split!r}, "
            f"n={len(self)}, classes={len(self.classes)})"
        )


# ---------------------------------------------------------------------------
# CIFAR-10 CLIP dataset (image + text pairs)
# ---------------------------------------------------------------------------

class CIFAR10CLIPDataset(Dataset):
    """
    CIFAR-10 as image-text pairs for CLIP contrastive training.

    Each sample returns:
        image_tensor: (C, H, W) float32
        tokens:       (max_seq_len,) int64 — tokenised text description
        label:        int — class index (for zero-shot eval)

    Text descriptions use the template "a photo of a {class_name}"
    (or a randomly chosen template from CLIP_PROMPT_TEMPLATES if
    `multi_template=True`).
    """

    def __init__(
        self,
        data_dir:       str | Path | None = None,
        split:          str = "train",
        transform:      Optional[Callable] = None,
        max_seq_len:    int = 77,
        multi_template: bool = False,
        images:         Optional[np.ndarray] = None,
        labels:         Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            data_dir: Root data directory (contains cifar-10-batches-py/).
                      Ignored if `images`/`labels` are both given.
            images:   Pre-loaded (N, 32, 32, 3) uint8 array. Lets callers
                      build a dataset from an in-memory subset (e.g. the
                      first `n` test images for a retrieval eval) without
                      re-reading the full split from disk.
            labels:   Pre-loaded (N,) int array, paired with `images`.
        """
        self.split = split
        if images is not None and labels is not None:
            self.images, self.labels = images, labels
        elif data_dir is not None:
            self.images, self.labels = load_cifar10(data_dir, split)
        else:
            raise ValueError(
                "CIFAR10CLIPDataset requires either data_dir or both images and labels"
            )
        self.transform      = transform or (
            clip_train_transform() if split == "train" else cifar10_val_transform()
        )
        self.tokenizer      = ByteTokenizer()
        self.max_seq_len    = max_seq_len
        self.multi_template = multi_template
        self.classes        = CIFAR10_CLASSES

        # Pre-tokenise all class descriptions (speeds up __getitem__)
        self._class_tokens: dict[str, list[list[int]]] = {}
        for cls in self.classes:
            if multi_template:
                self._class_tokens[cls] = [
                    self.tokenizer.encode(t.format(cls), max_len=max_seq_len)
                    for t in CLIP_PROMPT_TEMPLATES
                ]
            else:
                self._class_tokens[cls] = [
                    self.tokenizer.encode(f"a photo of a {cls}", max_len=max_seq_len)
                ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_np  = self.images[idx]
        pil_img = Image.fromarray(img_np, mode="RGB")
        tensor  = self.transform(pil_img)

        label    = int(self.labels[idx])
        cls_name = self.classes[label]

        # Pick a random template if multi_template, else use the single one
        tokens_list = self._class_tokens[cls_name]
        tok_ids     = tokens_list[torch.randint(len(tokens_list), (1,)).item()]
        tokens      = torch.tensor(tok_ids, dtype=torch.long)

        return tensor, tokens, label

    def __repr__(self) -> str:
        return (
            f"CIFAR10CLIPDataset(split={self.split!r}, "
            f"n={len(self)}, multi_template={self.multi_template})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_classification_dataloaders(
    data_dir:    str | Path,
    batch_size:  int = 128,
    num_workers: int = 0,
    augment:     bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CIFAR-10 classification.

    Returns:
        (train_loader, test_loader)
    """
    train_ds = CIFAR10Dataset(data_dir, split="train",
                              transform=cifar10_train_transform() if augment else cifar10_val_transform())
    test_ds  = CIFAR10Dataset(data_dir, split="test",
                              transform=cifar10_val_transform())

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    return train_loader, test_loader


def create_clip_dataloaders(
    data_dir:       str | Path,
    batch_size:     int = 64,
    num_workers:    int = 0,
    multi_template: bool = False,
    max_seq_len:    int = 77,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CLIP contrastive training.

    Returns:
        (train_loader, test_loader)
    """
    train_ds = CIFAR10CLIPDataset(
        data_dir, split="train",
        transform=clip_train_transform(),
        max_seq_len=max_seq_len,
        multi_template=multi_template,
    )
    test_ds = CIFAR10CLIPDataset(
        data_dir, split="test",
        transform=cifar10_val_transform(),
        max_seq_len=max_seq_len,
        multi_template=False,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,    # InfoNCE requires consistent batch size
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False,
                              batch_size=batch_size, num_workers=num_workers,
                              worker_init_fn=seed_worker if num_workers > 0 else None)

    return train_loader, test_loader
