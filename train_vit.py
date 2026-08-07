"""
ViT classification training entry point.

Quick start — CPU (Intel Core Ultra 7 / any machine):
    python prepare_data.py
    python train_vit.py

GPU:
    python train_vit.py --device cuda --batch_size 256

Override any hyperparameter:
    python train_vit.py --max_epochs 50 --max_lr 3e-4 --dropout 0.1

Resume from checkpoint:
    python train_vit.py --resume checkpoints/vit_best.pt
"""

from __future__ import annotations

import argparse
import torch

from nanovision.model.config import ViTConfig, TrainConfig
from nanovision.model.vit import ViT
from nanovision.data.dataset import create_classification_dataloaders
from nanovision.training.trainer import ViTTrainer
from nanovision.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Vision Transformer on CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset",      choices=["tiny", "small", "base"], default="tiny")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--device",      default=None, help="cuda / cpu / mps (default: auto)")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for torch/numpy/random (reproducibility)")
    p.add_argument("--deterministic", action="store_true",
                   help="Request bit-for-bit deterministic algorithms (slower)")

    # TrainConfig overrides
    p.add_argument("--max_epochs",   type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--max_lr",       type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--mixup_alpha",  type=float, default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs/vit")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")

    # Model config
    preset_fn = {"tiny": ViTConfig.tiny, "small": ViTConfig.small, "base": ViTConfig.base}
    model_cfg = preset_fn[args.preset]()

    # Apply optional dropout override
    if args.dropout is not None:
        model_cfg.dropout = args.dropout

    # Train config
    train_cfg = TrainConfig.vit_cifar10()
    overrides = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "max_lr":     args.max_lr,
        "weight_decay": args.weight_decay,
        "mixup_alpha":  args.mixup_alpha,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir":        args.log_dir,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(train_cfg, k, v)

    print(model_cfg)
    print(train_cfg)

    # Device
    device = torch.device(args.device) if args.device else None

    # Data
    train_loader, test_loader = create_classification_dataloaders(
        data_dir=args.data_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        augment=train_cfg.use_augmentation,
    )
    print(f"Train: {len(train_loader.dataset):,} images | Test: {len(test_loader.dataset):,} images")

    # Model
    model = ViT(model_cfg)

    # Trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=train_cfg,
        device=device,
    )
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
