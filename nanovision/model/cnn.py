"""
A small convolutional baseline for CIFAR-10, used ONLY as an honest point of
comparison against ViT-Tiny trained from scratch (see AUDIT.md §5,
item 14 — an honest comparison against a simple baseline).

Architecture: a compact ResNet-style CNN in the spirit of the well-known
"ResNet9" CIFAR-10 recipe (Page, "How to Train Your ResNet" / the
DAWNBench cifar10-fast entries) — conv-BN-ReLU stem, three downsampling
stages with two residual blocks, global max-pool, linear head. It is not
architecturally novel; it exists purely as a strong, standard, well-known
baseline so the ViT-vs-CNN comparison is meaningful rather than a straw man.

No torchvision dependency, consistent with the rest of NanoVision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    in_channels: int = 3
    n_classes:   int = 10
    # Channel width at each stage (stem, stage1, stage2, stage3)
    widths: tuple = (64, 128, 256, 512)
    dropout: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["widths"] = list(d["widths"])
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "CNNConfig":
        d = dict(d)
        if "widths" in d:
            d["widths"] = tuple(d["widths"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "CNNConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _conv_bn_relu(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers with an identity skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _conv_bn_relu(channels, channels)
        self.conv2 = _conv_bn_relu(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class BaselineCNN(nn.Module):
    """
    ResNet9-style CNN for CIFAR-10 (32x32x3 -> 10 classes).

    Pipeline:
      stem   : conv 3->w0                              (B, w0, 32, 32)
      stage1 : conv w0->w1 + maxpool2                   (B, w1, 16, 16)
               + residual block (w1)
      stage2 : conv w1->w2 + maxpool2                   (B, w2, 8, 8)
      stage3 : conv w2->w3 + maxpool2                   (B, w3, 4, 4)
               + residual block (w3)
      head   : global max-pool -> dropout -> linear     (B, n_classes)
    """

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        w0, w1, w2, w3 = config.widths

        self.stem   = _conv_bn_relu(config.in_channels, w0)
        self.stage1 = _conv_bn_relu(w0, w1, pool=True)
        self.res1   = ResidualBlock(w1)
        self.stage2 = _conv_bn_relu(w1, w2, pool=True)
        self.stage3 = _conv_bn_relu(w2, w3, pool=True)
        self.res2   = ResidualBlock(w3)

        self.pool    = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.head    = nn.Linear(w3, config.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.res1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)   # (B, w3)
        x = self.dropout(x)
        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(
        self, lr: float, weight_decay: float, momentum: float = 0.9,
    ) -> torch.optim.SGD:
        """
        SGD with Nesterov momentum — the standard, well-tuned recipe for
        CNNs on CIFAR-10 (unlike the AdamW used for the transformer models
        in this repo; using each architecture's own standard optimizer is
        what makes the comparison a fair "best each architecture can do",
        not an apples-to-apples optimizer ablation).
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "bn" in name.lower() or "bias" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)

    def __repr__(self) -> str:
        n = self.count_params()
        return f"BaselineCNN(widths={self.config.widths}, {n / 1e6:.2f}M params)"
