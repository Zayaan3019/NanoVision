"""
Learning rate schedules for ViT and CLIP training.

Two schedules:
  cosine_with_warmup  — linear warmup then cosine decay (standard ViT/CLIP)
  get_epoch_lr        — epoch-level LR lookup (wraps cosine_with_warmup)

Both use the GPT-3 / ViT convention:
  - Linear warmup from 0 to max_lr over `warmup_epochs`
  - Cosine decay from max_lr to min_lr over the remaining epochs
  - Constant min_lr after max_epochs
"""

from __future__ import annotations

import math


def cosine_with_warmup(
    step:          int,
    warmup_steps:  int,
    max_steps:     int,
    max_lr:        float,
    min_lr:        float = 0.0,
) -> float:
    """
    Cosine LR schedule with linear warmup (step-level).

    Args:
        step:         Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps:    Total training steps.
        max_lr:       Peak learning rate.
        min_lr:       Minimum (floor) learning rate.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / max(warmup_steps, 1)

    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress   = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cos_factor * (max_lr - min_lr)


def get_epoch_lr(
    epoch:          int,
    warmup_epochs:  int,
    max_epochs:     int,
    max_lr:         float,
    min_lr:         float = 1e-6,
) -> float:
    """
    Cosine LR schedule with linear warmup (epoch-level).

    Args:
        epoch:         Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        max_epochs:    Total training epochs.
        max_lr:        Peak learning rate.
        min_lr:        Minimum learning rate.

    Returns:
        Learning rate for the current epoch.
    """
    return cosine_with_warmup(
        step=epoch,
        warmup_steps=warmup_epochs,
        max_steps=max_epochs,
        max_lr=max_lr,
        min_lr=min_lr,
    )


def set_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    """Update all parameter groups in an optimizer to the given lr."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr
