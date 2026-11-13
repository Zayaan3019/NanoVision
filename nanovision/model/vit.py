"""
Vision Transformer (ViT) implemented from scratch.

Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale" (Dosovitskiy et al., 2020), https://arxiv.org/abs/2010.11929

Key design decisions:
- Patch embedding via Conv2d (efficient equivalent of linear projection over patches)
- Learnable [CLS] token prepended to the patch sequence
- Learnable 1-D position embeddings (absolute, one per sequence position)
- Pre-LayerNorm transformer encoder (more stable than post-LN at depth)
- Full (non-causal) self-attention using Flash Attention via torch SDPA
- Stochastic depth (DropPath) for regularisation at scale
- Weight-tying is NOT used here (vision and language modalities differ)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovision.model.config import ViTConfig


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Per-sample stochastic depth (Huang et al., 2016).

    Randomly drops entire residual branches during training, scaling the
    surviving branches up so the expected sum is preserved.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Shape (B, 1, 1) so it broadcasts over (B, T, C)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x * random_tensor / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project each to d_model.

    Implemented as a single Conv2d with kernel_size = stride = patch_size,
    which is mathematically equivalent to flattening each patch and passing
    it through a shared linear layer — but runs in a single GPU kernel.

    Input : (B, C, H, W)
    Output: (B, N, d_model)  where N = (H/P) * (W/P) = n_patches
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.n_patches  = config.n_patches
        self.proj = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)          # (B, d_model, H/P, W/P)
        x = x.flatten(2)          # (B, d_model, N)
        x = x.transpose(1, 2)     # (B, N, d_model)
        return x


# ---------------------------------------------------------------------------
# Self-attention (full, non-causal)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Multi-head self-attention without causal masking.

    For vision tasks every patch attends to every other patch freely.
    Uses PyTorch 2.0 scaled_dot_product_attention (Flash Attention kernel)
    with is_causal=False.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.head_dim
        self.dropout  = config.dropout

        # Fused QKV projection
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.c_proj = nn.Linear(config.d_model, config.d_model,     bias=True)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split(q), _split(k), _split(v)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn  = (q @ k.transpose(-2, -1)) * scale
            attn  = F.softmax(attn, dim=-1)
            attn  = self.attn_drop(attn)
            y     = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Uses GELU activation and a 4x expansion ratio (same as original ViT).
    Note: ViT uses bias=True in linear layers (unlike GPT-style models).
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        hidden = config.mlp_hidden
        self.fc1  = nn.Linear(config.d_model, hidden, bias=True)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, config.d_model, bias=True)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Pre-norm Transformer encoder block.

    Pre-norm (LN before each sub-layer), as in the original ViT
    (Dosovitskiy et al., 2021, eqs. 2-3). It is more stable at depth than
    the post-norm ordering of the original Transformer.
    """

    def __init__(self, config: ViTConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.ln_1  = nn.LayerNorm(config.d_model)
        self.attn  = SelfAttention(config)
        self.ln_2  = nn.LayerNorm(config.d_model)
        self.mlp   = MLP(config)
        self.dp    = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(self.attn(self.ln_1(x)))
        x = x + self.dp(self.mlp(self.ln_2(x)))
        return x


# ---------------------------------------------------------------------------
# Full Vision Transformer
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Processing pipeline:
      image  (B, C, H, W)
        -> PatchEmbedding     (B, N, d_model)
        -> prepend [CLS]      (B, N+1, d_model)
        -> add pos embedding  (B, N+1, d_model)
        -> n_layers x Block   (B, N+1, d_model)
        -> LayerNorm (final)  (B, N+1, d_model)
        -> [CLS] token        (B, d_model)
        -> classification head (B, n_classes)
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.seq_len, config.d_model)
        )  # seq_len = n_patches + 1

        self.pos_drop = nn.Dropout(config.dropout)

        # Stochastic depth: linearly increase drop rate across layers
        dp_rates = [
            config.drop_path * i / max(config.n_layers - 1, 1)
            for i in range(config.n_layers)
        ]
        self.blocks = nn.ModuleList([
            Block(config, drop_path_rate=dp_rates[i])
            for i in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        # Classification head
        if config.representation_size is not None:
            # Pre-logits MLP (used in some ViT variants)
            self.pre_logits = nn.Sequential(
                nn.Linear(config.d_model, config.representation_size),
                nn.Tanh(),
            )
            head_in = config.representation_size
        else:
            self.pre_logits = nn.Identity()
            head_in = config.d_model

        self.head = nn.Linear(head_in, config.n_classes)

        # Initialise weights
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Patch embedding (similar to linear layer init)
        nn.init.normal_(self.cls_token, std=0.02)

        # Position embeddings — initialised with sinusoidal values then
        # fine-tuned as learned parameters (common in ViT practice)
        nn.init.normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Classification head initialised to small values
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the transformer encoder and return the [CLS] token representation.

        Args:
            x: (B, C, H, W) normalised image tensor.

        Returns:
            (B, d_model) — [CLS] token after final LayerNorm.
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                            # (B, N, d_model)

        # Prepend [CLS] token (broadcast across batch)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                   # (B, N+1, d_model)

        # Add position embeddings
        x = self.pos_drop(x + self.pos_embed)              # (B, N+1, d_model)

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x[:, 0]                                     # [CLS] token: (B, d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               (B, C, H, W) normalised image tensor.
            return_features: If True, also return the [CLS] features before head.

        Returns:
            logits:   (B, n_classes) class logits.
            features: (B, d_model) — only if return_features=True.
        """
        features = self.forward_features(x)       # (B, d_model)
        pre      = self.pre_logits(features)       # (B, d_model) or (B, repr_size)
        logits   = self.head(pre)                  # (B, n_classes)

        if return_features:
            return logits, features
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_embed.numel()
            n -= self.cls_token.numel()
        return n

    def configure_optimizer(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
    ) -> torch.optim.AdamW:
        """
        AdamW with layer-wise weight decay.

        Position embeddings, [CLS] token, LayerNorm, and biases are
        excluded from weight decay following the ViT paper.
        """
        decay:    set[str] = set()
        no_decay: set[str] = set()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # 1-D tensors (biases, LN scale/bias) and pos/cls embeddings: no decay
            if (param.dim() < 2
                    or "pos_embed" in name
                    or "cls_token" in name
                    or "norm" in name.lower()
                    or "ln" in name.lower()):
                no_decay.add(name)
            else:
                decay.add(name)

        param_dict = dict(self.named_parameters())
        param_groups = [
            {"params": [param_dict[n] for n in sorted(decay)],
             "weight_decay": weight_decay},
            {"params": [param_dict[n] for n in sorted(no_decay)],
             "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    def __repr__(self) -> str:
        n = self.count_params(non_embedding=True)
        return (
            f"ViT | {self.config.n_layers}L x {self.config.n_heads}H x "
            f"{self.config.d_model}D | patch={self.config.patch_size} | "
            f"{n / 1e6:.2f}M params"
        )
