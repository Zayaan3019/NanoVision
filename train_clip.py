"""
CLIP contrastive training entry point.

Quick start:
    python prepare_data.py
    python train_clip.py

With multi-template prompts (stronger CLIP):
    python train_clip.py --multi_template

Override hyperparameters:
    python train_clip.py --max_epochs 30 --batch_size 128 --max_lr 1e-3

Resume:
    python train_clip.py --resume checkpoints/clip_best.pt
"""

from __future__ import annotations

import argparse
import torch

from nanovision.model.config import CLIPConfig, TrainConfig
from nanovision.model.clip import CLIP
from nanovision.data.dataset import create_clip_dataloaders
from nanovision.training.trainer import CLIPTrainer
from nanovision.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CLIP on CIFAR-10 image-text pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset",         choices=["tiny", "micro"], default="tiny",
                   help="'tiny' is the reference architecture (GPU recommended); "
                        "'micro' (~0.8M params) trains to completion on CPU-only "
                        "hardware in well under 2 hours.")
    p.add_argument("--data_dir",       default="data")
    p.add_argument("--device",         default=None, help="cuda / cpu / mps (default: auto)")
    p.add_argument("--resume",         default=None, help="Path to checkpoint to resume from")
    p.add_argument("--multi_template", action="store_true",
                   help="Use multiple text prompt templates per class")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for torch/numpy/random (reproducibility)")
    p.add_argument("--deterministic",  action="store_true",
                   help="Request bit-for-bit deterministic algorithms (slower)")

    # TrainConfig overrides
    p.add_argument("--max_epochs",   type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--max_lr",       type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs/clip")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")

    # Configs
    clip_cfg  = CLIPConfig.tiny() if args.preset == "tiny" else CLIPConfig.micro()
    train_cfg = TrainConfig.clip_cifar10()

    overrides = {
        "max_epochs":   args.max_epochs,
        "batch_size":   args.batch_size,
        "max_lr":       args.max_lr,
        "weight_decay": args.weight_decay,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir":        args.log_dir,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(train_cfg, k, v)

    print(clip_cfg)
    print(train_cfg)

    # Device
    device = torch.device(args.device) if args.device else None

    # Data
    train_loader, test_loader = create_clip_dataloaders(
        data_dir=args.data_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        multi_template=args.multi_template,
        max_seq_len=clip_cfg.text.max_seq_len,
    )
    print(f"Train: {len(train_loader.dataset):,} pairs | Test: {len(test_loader.dataset):,} pairs")

    # Model + trainer
    model   = CLIP(clip_cfg)
    trainer = CLIPTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=train_cfg,
        device=device,
    )
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
