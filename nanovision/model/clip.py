"""
CLIP — Contrastive Language-Image Pre-training, implemented from scratch.

Reference: "Learning Transferable Visual Models from Natural Language
Supervision" (Radford et al., 2021), https://arxiv.org/abs/2103.00020

Architecture:
  vision encoder (ViT) ---> linear projection ---> L2-normalised embed
  text encoder (Transformer) -> linear projection -> L2-normalised embed

  InfoNCE loss on the N x N cosine similarity matrix:
    rows    = image-to-text (which text matches each image?)
    columns = text-to-image (which image matches each text?)
    Diagonal entries are the positive pairs; all off-diagonal are negatives.

  Loss = 0.5 * (CE(rows, diag) + CE(cols, diag))   -- symmetric

Learnable temperature:
  log_temp is a learnable scalar; actual temperature = exp(log_temp).
  Clamped to [0, log(100)] to avoid numerical instability.

Zero-shot classification:
  For each CIFAR-10 class c, compute text embedding of "a photo of a {c}".
  Assign each image to the class with highest cosine similarity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovision.model.config import CLIPConfig
from nanovision.model.vit import ViT
from nanovision.model.text_encoder import TextEncoder, ByteTokenizer


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Single linear projection from encoder dimension to shared embed_dim.

    The original CLIP paper uses a single linear layer (no non-linearity).
    Weights are initialised with truncated normal; no bias.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=in_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    """
    CLIP: joint vision-language model trained with symmetric InfoNCE loss.

    Usage — training:
        model = CLIP(CLIPConfig.tiny())
        loss, logits_i2t, logits_t2i = model(images, tokens)

    Usage — zero-shot classification:
        zs_logits = model.zero_shot_classify(images, class_names)
        predicted  = zs_logits.argmax(dim=-1)

    Usage — feature extraction:
        img_feats  = model.encode_image(images)   # (B, embed_dim), L2-normalised
        text_feats = model.encode_text(tokens)    # (B, embed_dim), L2-normalised
    """

    # CIFAR-10 class names for zero-shot classification
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # Prompt template used for zero-shot (following original CLIP paper)
    PROMPT_TEMPLATE = "a photo of a {}"

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config    = config
        self.embed_dim = config.embed_dim

        # Encoders
        self.vision_encoder = ViT(config.vision)
        self.text_encoder   = TextEncoder(config.text)

        # Projection heads
        self.vision_proj = ProjectionHead(config.vision.d_model, config.embed_dim)
        self.text_proj   = ProjectionHead(config.text.d_model,   config.embed_dim)

        # Learnable temperature: logit_scale = log(1/temperature)
        # Clamped to [0, log(100)] as in the CLIP paper
        self.log_temp = nn.Parameter(torch.tensor(config.init_log_temp))

        # Tokenizer (convenience access)
        self.tokenizer = ByteTokenizer()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to L2-normalised embedding vectors.

        Args:
            images: (B, C, H, W) normalised image tensor.

        Returns:
            (B, embed_dim) — unit-norm image embeddings.
        """
        feats = self.vision_encoder.forward_features(images)   # (B, d_model)
        proj  = self.vision_proj(feats)                        # (B, embed_dim)
        return F.normalize(proj, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenised text to L2-normalised embedding vectors.

        Args:
            tokens: (B, T) integer token ids.

        Returns:
            (B, embed_dim) — unit-norm text embeddings.
        """
        feats = self.text_encoder(tokens)    # (B, d_model)
        proj  = self.text_proj(feats)        # (B, embed_dim)
        return F.normalize(proj, dim=-1)

    def encode_texts_from_strings(
        self, texts: list[str], device: torch.device
    ) -> torch.Tensor:
        """Tokenise and encode a list of raw strings."""
        tokens = self.tokenizer.encode_batch(
            texts, max_len=self.config.text.max_seq_len
        ).to(device)
        return self.encode_text(tokens)

    # ------------------------------------------------------------------
    # Forward pass + loss
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute symmetric InfoNCE loss over a batch of image-text pairs.

        Args:
            images: (B, C, H, W) normalised image tensor.
            tokens: (B, T) integer token ids.

        Returns:
            loss:        Scalar contrastive loss.
            logits_i2t:  (B, B) image-to-text similarity logits.
            logits_t2i:  (B, B) text-to-image similarity logits.
        """
        img_feats  = self.encode_image(images)     # (B, embed_dim)
        text_feats = self.encode_text(tokens)      # (B, embed_dim)

        return self._contrastive_loss(img_feats, text_feats)

    def _contrastive_loss(
        self,
        img_feats:  torch.Tensor,
        text_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symmetric InfoNCE loss.

        The N x N similarity matrix S[i,j] = img_feats[i] . text_feats[j].
        Positive pairs are on the diagonal; negatives are everything else.

        Temperature is applied as:  logits = S * exp(log_temp)
        where log_temp is a learnable scalar clamped to [0, log(100)].
        """
        B = img_feats.shape[0]

        # Clamp log_temp to avoid extreme temperatures
        log_temp = self.log_temp.clamp(0, 4.605)   # 4.605 = log(100)
        temp     = log_temp.exp()

        # Similarity matrix
        logits_i2t = temp * (img_feats  @ text_feats.t())   # (B, B)
        logits_t2i = logits_i2t.t()                          # (B, B)

        # Labels: diagonal = positive pair index
        labels = torch.arange(B, device=img_feats.device)

        loss_i2t = F.cross_entropy(
            logits_i2t, labels,
            label_smoothing=self.config.label_smoothing,
        )
        loss_t2i = F.cross_entropy(
            logits_t2i, labels,
            label_smoothing=self.config.label_smoothing,
        )
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss, logits_i2t, logits_t2i

    # ------------------------------------------------------------------
    # Zero-shot classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images:      torch.Tensor,
        class_names: Optional[list[str]] = None,
        template:    str = PROMPT_TEMPLATE,
    ) -> torch.Tensor:
        """
        Zero-shot image classification via text similarity.

        Encodes one text prompt per class, then assigns each image to the
        class whose text embedding is most similar.

        Args:
            images:      (B, C, H, W) normalised image tensor.
            class_names: List of class name strings.  Defaults to CIFAR-10.
            template:    Format string with one {} for class name insertion.

        Returns:
            (B, n_classes) cosine-similarity logits.
        """
        if class_names is None:
            class_names = self.CIFAR10_CLASSES

        # Build text prompts
        prompts = [template.format(c) for c in class_names]   # list[str]

        # Encode class text embeddings (cached pattern: no grad needed)
        text_feats = self.encode_texts_from_strings(prompts, images.device)
        # text_feats: (n_classes, embed_dim), already L2-normalised

        # Encode images
        img_feats = self.encode_image(images)   # (B, embed_dim)

        # Similarity logits (temperature scaled by CLIP convention)
        temp = self.log_temp.clamp(0, 4.605).exp()
        logits = temp * (img_feats @ text_feats.t())   # (B, n_classes)

        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_params(self) -> dict[str, int]:
        """Return parameter counts per module."""
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "vision_encoder": _n(self.vision_encoder),
            "text_encoder":   _n(self.text_encoder),
            "vision_proj":    _n(self.vision_proj),
            "text_proj":      _n(self.text_proj),
            "total":          _n(self),
        }

    def configure_optimizer(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
    ) -> torch.optim.AdamW:
        """
        AdamW with weight decay applied only to 2-D weight matrices.

        Biases, LayerNorm parameters, embeddings, and the temperature
        scalar are excluded from weight decay.
        """
        decay:    list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []

        skip_keywords = {"pos_embed", "cls_token", "tok_embed", "log_temp",
                         "norm", "ln", "bias"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or any(kw in name.lower() for kw in skip_keywords):
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    def __repr__(self) -> str:
        counts = self.count_params()
        return (
            f"CLIP(\n"
            f"  {self.vision_encoder}\n"
            f"  {self.text_encoder}\n"
            f"  embed_dim={self.embed_dim}\n"
            f"  total={counts['total'] / 1e6:.2f}M params\n"
            f")"
        )
