"""
Model configuration dataclasses for NanoVision.

Three model families:
  ViTConfig         — Vision Transformer image encoder
  TextEncoderConfig — Bidirectional transformer text encoder
  CLIPConfig        — Joint vision-language model (combines both)

All configs support named presets that match the ViT paper's
scale nomenclature (Ti/16, S/16, B/16) adapted to CIFAR-10's
32x32 resolution (patch size 4 instead of 16).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Vision Transformer config
# ---------------------------------------------------------------------------

@dataclass
class ViTConfig:
    """
    Configuration for a Vision Transformer (ViT).

    Architecture follows "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    with patch size adapted to 32x32 CIFAR-10 images.
    """

    # Image & patch geometry
    img_size:   int = 32    # Input image height = width (CIFAR-10 is 32x32)
    patch_size: int = 4     # Patch height = width; yields (32/4)^2 = 64 patches
    in_channels: int = 3    # RGB

    # Architecture
    d_model:   int = 192    # Embedding dimension
    n_layers:  int = 6      # Transformer encoder layers
    n_heads:   int = 6      # Attention heads (d_model must be divisible by n_heads)
    mlp_ratio: float = 4.0  # MLP hidden dim = mlp_ratio * d_model

    # Regularisation
    dropout:    float = 0.0   # Attention + residual dropout
    drop_path:  float = 0.0   # Stochastic depth (per-layer drop probability)

    # Classification head
    n_classes:  int = 10      # CIFAR-10

    # Representation
    representation_size: int | None = None  # Pre-logits MLP bottleneck (None = disable)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by patch_size ({self.patch_size})"
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        """Number of image patches (sequence length before prepending [CLS])."""
        return (self.img_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        """Total sequence length = patches + 1 [CLS] token."""
        return self.n_patches + 1

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.d_model * self.mlp_ratio)

    @property
    def n_params(self) -> int:
        """Approximate non-embedding parameter count."""
        per_layer = (
            3 * self.d_model * self.d_model      # QKV
            + self.d_model * self.d_model         # out proj
            + self.d_model * self.mlp_hidden      # MLP fc
            + self.mlp_hidden * self.d_model      # MLP proj
            + 2 * self.d_model                    # two LayerNorms
        )
        return self.n_layers * per_layer

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> ViTConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> ViTConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Named presets  (patch_size=4 for 32x32 input)
    # ------------------------------------------------------------------

    @classmethod
    def tiny(cls) -> ViTConfig:
        """ViT-Tiny/4: 2.69 M params — fast to train on CPU."""
        return cls(d_model=192, n_layers=6, n_heads=6)

    @classmethod
    def small(cls) -> ViTConfig:
        """ViT-Small/4: 14.2 M params — single GPU or patient CPU."""
        return cls(d_model=384, n_layers=8, n_heads=6)

    @classmethod
    def base(cls) -> ViTConfig:
        """ViT-Base/4: 85.2 M params — GPU recommended."""
        return cls(d_model=768, n_layers=12, n_heads=12)

    def __repr__(self) -> str:
        return (
            f"ViTConfig(img={self.img_size}x{self.img_size}, "
            f"patch={self.patch_size}, n_patches={self.n_patches}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"d_model={self.d_model}, ~{self.n_params / 1e6:.1f}M params)"
        )


# ---------------------------------------------------------------------------
# Text Encoder config
# ---------------------------------------------------------------------------

@dataclass
class TextEncoderConfig:
    """
    Configuration for the bidirectional transformer text encoder used in CLIP.

    Uses a byte-level character tokenizer (vocab_size=131) so it handles
    any UTF-8 text without unknown tokens.
    """

    # Tokenizer
    vocab_size:  int = 131   # 128 printable ASCII + [PAD]=128, [SOT]=129, [EOT]=130
    max_seq_len: int = 77    # Matches original CLIP text encoder context length

    # Architecture
    d_model:  int = 192
    n_layers: int = 4
    n_heads:  int = 6

    # Regularisation
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TextEncoderConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"TextEncoderConfig(vocab={self.vocab_size}, seq={self.max_seq_len}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )


# ---------------------------------------------------------------------------
# CLIP config
# ---------------------------------------------------------------------------

@dataclass
class CLIPConfig:
    """
    Configuration for the full CLIP model.

    CLIP jointly trains a vision encoder and a text encoder with a symmetric
    InfoNCE contrastive loss (Radford et al., 2021).
    """

    vision: ViTConfig           = field(default_factory=ViTConfig.tiny)
    text:   TextEncoderConfig   = field(default_factory=TextEncoderConfig)

    # Shared embedding dimension (projection heads map both encoders here)
    embed_dim: int = 128

    # Temperature (log scale — actual temperature = exp(-log_temp) or 1/exp(log_temp))
    # Initialised to log(1/0.07) ≈ 2.659 as in the original CLIP paper
    init_log_temp: float = 2.659

    # Label smoothing for contrastive loss (0 = no smoothing)
    label_smoothing: float = 0.0

    def to_dict(self) -> dict:
        return {
            "vision":           self.vision.to_dict(),
            "text":             self.text.to_dict(),
            "embed_dim":        self.embed_dim,
            "init_log_temp":    self.init_log_temp,
            "label_smoothing":  self.label_smoothing,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> CLIPConfig:
        return cls(
            vision=ViTConfig.from_dict(d["vision"]),
            text=TextEncoderConfig.from_dict(d["text"]),
            embed_dim=d.get("embed_dim", 128),
            init_log_temp=d.get("init_log_temp", 2.659),
            label_smoothing=d.get("label_smoothing", 0.0),
        )

    @classmethod
    def load(cls, path: str | Path) -> CLIPConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def tiny(cls) -> CLIPConfig:
        """
        Tiny CLIP — the architecture documented in the README (ViT-Tiny
        vision encoder + 4-layer text encoder). On the CPU used to develop
        this repo this measures ~2.2s/batch(64), i.e. ~28 min/epoch — a GPU
        is recommended for training this preset to completion; see
        `CLIPConfig.micro()` for a preset sized for CPU-only iteration.
        """
        return cls(vision=ViTConfig.tiny(), text=TextEncoderConfig(), embed_dim=128)

    @classmethod
    def micro(cls) -> CLIPConfig:
        """
        CLIP-Micro (~0.8M params) — a deliberately smaller preset so a full
        training run completes in well under two hours on CPU-only
        hardware. Same architecture family (Pre-LN ViT + bidirectional
        text transformer, symmetric InfoNCE), just narrower/shallower.
        Not a substitute for `tiny()` as the reference architecture — use
        it when you need a real, fully-trained, reproducible CLIP checkpoint
        without GPU access.
        """
        vision = ViTConfig(d_model=96, n_layers=4, n_heads=4, patch_size=4, img_size=32)
        text   = TextEncoderConfig(d_model=96, n_layers=3, n_heads=4)
        return cls(vision=vision, text=text, embed_dim=64)

    def __repr__(self) -> str:
        return (
            f"CLIPConfig(\n"
            f"  vision = {self.vision}\n"
            f"  text   = {self.text}\n"
            f"  embed_dim={self.embed_dim}, init_log_temp={self.init_log_temp:.3f}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters for ViT classification or CLIP contrastive training."""

    # Optimiser
    max_lr:       float = 1e-3
    min_lr:       float = 1e-5
    weight_decay: float = 0.05   # ViT paper uses 0.05
    beta1:        float = 0.9
    beta2:        float = 0.999
    grad_clip:    float = 1.0

    # LR schedule
    warmup_epochs: int = 5
    max_epochs:    int = 100

    # Batch
    batch_size:  int = 128
    num_workers: int = 0

    # Precision
    use_amp:  bool = False
    dtype:    Literal["float32", "bfloat16"] = "float32"

    # Checkpointing & logging
    checkpoint_dir: str = "checkpoints"
    log_dir:        str = "logs"
    log_interval:   int = 10     # batches
    eval_interval:  int = 1      # epochs
    use_tensorboard: bool = True

    # Data augmentation
    use_augmentation: bool = True
    mixup_alpha:      float = 0.0  # 0 = disabled; 0.2 = typical ViT setting

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def vit_cifar10(cls) -> TrainConfig:
        """Standard ViT-Tiny training on CIFAR-10 (~2 h on CPU)."""
        return cls(
            max_lr=1e-3, min_lr=1e-5,
            weight_decay=0.05,
            warmup_epochs=5, max_epochs=100,
            batch_size=128,
            use_amp=False, dtype="float32",
            use_augmentation=True,
            mixup_alpha=0.2,
        )

    @classmethod
    def clip_cifar10(cls) -> TrainConfig:
        """CLIP contrastive training on CIFAR-10 image-text pairs."""
        return cls(
            max_lr=5e-4, min_lr=1e-5,
            weight_decay=0.2,    # CLIP paper uses 0.2
            warmup_epochs=5, max_epochs=50,
            batch_size=64,       # Larger batch = more negatives = better CLIP
            use_amp=False, dtype="float32",
            use_augmentation=True,
            mixup_alpha=0.0,     # Mixup not used for contrastive
        )

    def __repr__(self) -> str:
        return (
            f"TrainConfig(epochs={self.max_epochs}, lr={self.max_lr}, "
            f"batch={self.batch_size}, wd={self.weight_decay})"
        )
