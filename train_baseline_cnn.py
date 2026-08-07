"""
Train the small ResNet9-style CNN baseline on CIFAR-10.

This exists ONLY to give the ViT-from-scratch result an honest point of
comparison (AUDIT.md §5, item 14). It reuses the exact same
data pipeline (nanovision.data) and checkpointing conventions as the ViT/CLIP
trainers, but its own optimizer (SGD+Nesterov, the standard CNN recipe)
and a lightweight training loop, since BaselineCNN is not part of the core
from-scratch transformer architecture this repo is about.

Usage:
    python train_baseline_cnn.py
    python train_baseline_cnn.py --max_epochs 30 --batch_size 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovision.model.cnn import BaselineCNN, CNNConfig
from nanovision.data.dataset import create_classification_dataloaders
from nanovision.training.scheduler import get_epoch_lr, set_lr
from nanovision.utils.metrics import accuracy
from nanovision.utils.seed import set_seed


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CNN baseline on CIFAR-10 (for ViT comparison).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       default="data")
    p.add_argument("--device",         default=None)
    p.add_argument("--max_epochs",     type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--max_lr",         type=float, default=0.1)
    p.add_argument("--min_lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=5e-4)
    p.add_argument("--warmup_epochs",  type=int, default=2)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--deterministic",  action="store_true")
    p.add_argument("--widths", type=int, nargs=4, default=[64, 128, 256, 512],
                   metavar=("W0", "W1", "W2", "W3"),
                   help="Channel widths per stage. Default is the full "
                        "ResNet9 recipe (~6.6M params); on CPU-only hardware "
                        "consider e.g. --widths 32 64 128 256 (~1.6M params, "
                        "roughly ViT-Tiny's parameter count) to train faster.")
    return p.parse_args()


def evaluate(model: BaselineCNN, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            top1 += accuracy(logits, labels, topk=1)
            top5 += accuracy(logits, labels, topk=5)
            n += labels.shape[0]
    return total_loss / n, top1 / n * 100, top5 / n * 100


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")

    device = torch.device(args.device) if args.device else _get_device()

    train_loader, test_loader = create_classification_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0, augment=True,
    )
    print(f"Train: {len(train_loader.dataset):,} images | Test: {len(test_loader.dataset):,} images")

    cfg = CNNConfig(widths=tuple(args.widths))
    model = BaselineCNN(cfg).to(device)
    print(model)
    print(f"Params: {model.count_params() / 1e6:.2f}M | Device: {device} | Epochs: {args.max_epochs}")

    optimizer = model.configure_optimizer(lr=args.max_lr, weight_decay=args.weight_decay)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.max_epochs):
        lr = get_epoch_lr(epoch, args.warmup_epochs, args.max_epochs, args.max_lr, args.min_lr)
        set_lr(optimizer, lr)

        model.train()
        t0 = time.perf_counter()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            B = labels.shape[0]
            total_loss += loss.item() * B
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += B
        dt = time.perf_counter() - t0

        val_loss, top1, top5 = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch + 1:>3}/{args.max_epochs} | lr {lr:.2e} | "
            f"train_loss {total_loss / total_n:.4f} | train_acc {100 * total_correct / total_n:.2f}% | "
            f"val_loss {val_loss:.4f} | top1 {top1:.2f}% | top5 {top5:.2f}% | {dt:.1f}s"
        )

        if top1 > best_acc:
            best_acc = top1
            torch.save(
                {
                    "model_type": "baseline_cnn",
                    "epoch": epoch + 1,
                    "metric": top1,
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg.to_dict(),
                },
                Path(args.checkpoint_dir) / "cnn_baseline_best.pt",
            )
            print(f"  >> [ckpt] Best CNN baseline saved (top1={top1:.2f}%)")

    print("=" * 70)
    print(f"CNN baseline training complete. Best top-1 accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
