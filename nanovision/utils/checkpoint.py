"""
Checkpoint save / load utilities for ViT and CLIP models.

Every checkpoint is self-describing: it contains the model state dict,
optimizer state, epoch, metric, model config (as dict), and model type.
This enables exact model reconstruction without external config files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    path:         str | Path,
    model:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    epoch:        int,
    metric:       float,
    config:       object,          # ViTConfig or CLIPConfig (has .to_dict())
    train_config: object,
    model_type:   str = "vit",     # "vit" or "clip"
) -> None:
    """
    Save a full training checkpoint.

    Args:
        path:         Destination .pt file.
        model:        The model (ViT or CLIP).
        optimizer:    Optimizer with current state.
        epoch:        Current epoch (1-indexed).
        metric:       Best metric so far (accuracy or zero-shot acc).
        config:       Model configuration object.
        train_config: Training configuration object.
        model_type:   "vit" or "clip".
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type":          model_type,
            "epoch":               epoch,
            "metric":              metric,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config":        config.to_dict(),
            "train_config":        train_config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path:      str | Path,
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """
    Load a checkpoint into `model` (and optionally `optimizer`).

    Returns:
        The epoch at which the checkpoint was saved.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)


def load_vit_from_checkpoint(path: str | Path) -> tuple["ViT", "ViTConfig", int]:
    """Reconstruct a ViT model entirely from a checkpoint."""
    from nanovision.model.config import ViTConfig
    from nanovision.model.vit import ViT

    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    config = ViTConfig.from_dict(ckpt["model_config"])
    model  = ViT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt.get("epoch", 0)


def load_clip_from_checkpoint(path: str | Path) -> tuple["CLIP", "CLIPConfig", int]:
    """Reconstruct a CLIP model entirely from a checkpoint."""
    from nanovision.model.config import CLIPConfig
    from nanovision.model.clip import CLIP

    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    config = CLIPConfig.from_dict(ckpt["model_config"])
    model  = CLIP(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt.get("epoch", 0)


def load_cnn_from_checkpoint(path: str | Path) -> tuple["BaselineCNN", "CNNConfig", int]:
    """Reconstruct the baseline CNN entirely from a checkpoint."""
    from nanovision.model.cnn import BaselineCNN, CNNConfig

    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    config = CNNConfig.from_dict(ckpt["model_config"])
    model  = BaselineCNN(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt.get("epoch", 0)
