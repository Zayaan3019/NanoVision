"""
Comprehensive evaluation of trained ViT and CLIP models.

Metrics computed:
  ViT  : Top-1 / Top-5 accuracy, per-class accuracy
  CLIP : Zero-shot Top-1 / Top-5 accuracy, Retrieval R@1 / R@5 / R@10

Usage:
    python evaluate.py --vit_ckpt  checkpoints/vit_best.pt
    python evaluate.py --clip_ckpt checkpoints/clip_best.pt
    python evaluate.py --vit_ckpt  checkpoints/vit_best.pt \
                       --clip_ckpt checkpoints/clip_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from nanovision.data.dataset import (
    load_cifar10, CIFAR10CLIPDataset,
    create_classification_dataloaders, create_clip_dataloaders,
)
from nanovision.data.augmentation import cifar10_val_transform
from nanovision.retrieval.index import EmbeddingIndex
from nanovision.utils.checkpoint import load_vit_from_checkpoint, load_clip_from_checkpoint
from nanovision.utils.metrics import accuracy
from nanovision.utils.seed import set_seed

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

LINE = "-" * 52


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained NanoVision models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vit_ckpt",  default=None, help="Path to ViT checkpoint")
    p.add_argument("--clip_ckpt", default=None, help="Path to CLIP checkpoint")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--device",    default=None, help="cuda / cpu / mps (default: auto)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_retrieval", type=int, default=1000,
                   help="Number of test samples for retrieval evaluation")
    p.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility)")
    return p.parse_args()


def _auto_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ViT evaluation
# ---------------------------------------------------------------------------

def evaluate_vit(ckpt_path: str, data_dir: str, device: torch.device,
                 batch_size: int) -> None:
    print(f"\n{'=' * 52}")
    print(" ViT Evaluation")
    print(f"{'=' * 52}")

    model, cfg, _ = load_vit_from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    print(f"Loaded  : {ckpt_path}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")

    _, test_loader = create_classification_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        augment=False,
    )

    correct_top1 = 0
    correct_top5 = 0
    all_preds:  list[int] = []
    all_labels: list[int] = []
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            correct_top1 += accuracy(logits, labels, topk=1)
            correct_top5 += accuracy(logits, labels, topk=5)
            total += labels.size(0)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc1 = 100.0 * correct_top1 / total
    acc5 = 100.0 * correct_top5 / total

    print(f"\n{LINE}")
    print(f"  Top-1 Accuracy : {acc1:.2f}%")
    print(f"  Top-5 Accuracy : {acc5:.2f}%")
    print(f"  Test samples   : {total:,}")

    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':<14} {'Correct':>7} {'Total':>7} {'Acc':>8}")
    print(f"  {'-' * 42}")
    for cls_id, cls_name in enumerate(CIFAR10_CLASSES):
        mask    = labels_np == cls_id
        n_total = mask.sum()
        n_corr  = (preds_np[mask] == cls_id).sum()
        pct     = 100.0 * n_corr / n_total if n_total > 0 else 0.0
        print(f"  {cls_name:<14} {n_corr:>7} {n_total:>7} {pct:>7.1f}%")
    print(LINE)


# ---------------------------------------------------------------------------
# CLIP evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(ckpt_path: str, data_dir: str, device: torch.device,
                  batch_size: int, n_retrieval: int) -> None:
    print(f"\n{'=' * 52}")
    print(" CLIP Evaluation")
    print(f"{'=' * 52}")

    model, cfg, _ = load_clip_from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    print(f"Loaded  : {ckpt_path}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")

    _, test_loader = create_clip_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        multi_template=False,
        max_seq_len=cfg.text.max_seq_len,
    )

    # Zero-shot classification
    print(f"\n{LINE}")
    print("  Zero-Shot Classification")
    print(LINE)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, tokens, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model.zero_shot_classify(images)

            correct_top1 += accuracy(logits, labels, topk=1)
            correct_top5 += accuracy(logits, labels, topk=5)
            total += labels.size(0)

    acc1  = 100.0 * correct_top1 / total
    acc5  = 100.0 * correct_top5 / total
    temp  = model.log_temp.exp().item()

    print(f"  Top-1 Accuracy  : {acc1:.2f}%")
    print(f"  Top-5 Accuracy  : {acc5:.2f}%")
    print(f"  Temperature     : {temp:.4f}")
    print(f"  Test samples    : {total:,}")

    # Retrieval evaluation
    print(f"\n{LINE}")
    print(f"  Cross-Modal Retrieval (n={n_retrieval})")
    print(LINE)

    images_np, labels_np = load_cifar10(Path(data_dir), split="test")
    images_np = images_np[:n_retrieval]
    labels_np = labels_np[:n_retrieval]

    transform = cifar10_val_transform()
    dataset   = CIFAR10CLIPDataset(
        images=images_np, labels=labels_np,
        transform=transform,
        max_seq_len=cfg.text.max_seq_len,
        multi_template=False,
    )
    retrieval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    index = EmbeddingIndex(embed_dim=model.embed_dim)
    index.build_from_clip(model, retrieval_loader, device=device)

    print(f"  {'':>4} {'strict (index)':>16} {'class-aware':>14}")
    for k in (1, 5, 10):
        r_strict = index.recall_at_k(k, direction="i2t", class_aware=False)
        r_class  = index.recall_at_k(k, direction="i2t", class_aware=True)
        print(f"  R@{k:<2d} {r_strict:>15.1f}% {r_class:>13.1f}%")
    print(
        "\n  NOTE: captions are one template per CIFAR-10 class, so many "
        "images/texts\n  in this gallery legitimately share an identical "
        "caption. 'strict' recall\n  (query i's only correct answer is item "
        "i) under-counts correct retrievals\n  in that regime; 'class-aware' "
        "(any top-K item of the same class counts)\n  is the metric that "
        "actually reflects retrieval quality here."
    )
    print(LINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = _auto_device(args.device)

    if args.vit_ckpt is None and args.clip_ckpt is None:
        print("No checkpoint specified. Use --vit_ckpt and/or --clip_ckpt.")
        return

    if args.vit_ckpt:
        evaluate_vit(
            ckpt_path=args.vit_ckpt,
            data_dir=args.data_dir,
            device=device,
            batch_size=args.batch_size,
        )

    if args.clip_ckpt:
        evaluate_clip(
            ckpt_path=args.clip_ckpt,
            data_dir=args.data_dir,
            device=device,
            batch_size=args.batch_size,
            n_retrieval=args.n_retrieval,
        )


if __name__ == "__main__":
    main()
