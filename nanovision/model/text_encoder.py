"""
Bidirectional transformer text encoder for CLIP.

Design:
  - Byte-level tokenizer: every printable ASCII character is one token,
    plus [PAD]=128, [SOT]=129 (start-of-text), [EOT]=130 (end-of-text).
    This handles any text without unknown tokens and keeps the vocabulary
    minimal (131 tokens).
  - Bidirectional (non-causal) self-attention: text is encoded holistically,
    not autoregressively.  [EOT] token representation is used as the
    sequence-level embedding (following the original CLIP paper).
  - Pre-LayerNorm transformer architecture (same as ViT).

Reference: "Learning Transferable Visual Models from Natural Language
Supervision" (Radford et al., 2021), https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovision.model.config import TextEncoderConfig


# ---------------------------------------------------------------------------
# Byte-level tokenizer
# ---------------------------------------------------------------------------

class ByteTokenizer:
    """
    Minimal byte-level text tokenizer.

    Vocabulary layout:
      0 – 127  : printable ASCII characters (direct mapping)
      128      : [PAD]  — padding token
      129      : [SOT]  — start of text
      130      : [EOT]  — end of text

    Total vocabulary: 131 tokens.

    Any non-ASCII character is encoded as its UTF-8 byte sequence split
    into individual bytes (each byte 0-127 maps to itself; bytes 128-255
    are clamped to the [PAD] token since we stay ASCII for CIFAR labels).

    For CIFAR-10 class descriptions ("a photo of a cat") only ASCII is needed.
    """

    PAD_ID = 128
    SOT_ID = 129
    EOT_ID = 130

    VOCAB_SIZE = 131

    def encode(
        self,
        text: str,
        max_len: int = 77,
        add_special: bool = True,
    ) -> list[int]:
        """
        Encode text to a list of token ids.

        Args:
            text:        Input text (ASCII expected for CIFAR-10 labels).
            max_len:     Maximum sequence length (truncate if longer).
            add_special: Prepend [SOT] and append [EOT].

        Returns:
            List of token ids, padded to max_len with [PAD].
        """
        # Convert to ASCII byte values; clamp non-ASCII to PAD
        byte_ids = [b if b < 128 else self.PAD_ID for b in text.encode("utf-8")]

        if add_special:
            ids = [self.SOT_ID] + byte_ids + [self.EOT_ID]
        else:
            ids = byte_ids

        # Truncate to max_len (keep [EOT] at the end if possible)
        if len(ids) > max_len:
            if add_special:
                ids = ids[: max_len - 1] + [self.EOT_ID]
            else:
                ids = ids[:max_len]

        # Pad to max_len
        ids = ids + [self.PAD_ID] * (max_len - len(ids))
        return ids

    def encode_batch(
        self,
        texts: list[str],
        max_len: int = 77,
    ) -> torch.Tensor:
        """Encode a list of strings to a (B, max_len) long tensor."""
        ids = [self.encode(t, max_len=max_len) for t in texts]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string (special tokens are stripped)."""
        skip = {self.PAD_ID, self.SOT_ID, self.EOT_ID}
        return bytes(i for i in ids if i not in skip and i < 128).decode("ascii", errors="replace")

    def __len__(self) -> int:
        return self.VOCAB_SIZE

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={self.VOCAB_SIZE})"


# ---------------------------------------------------------------------------
# Self-attention (bidirectional, with padding mask support)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Full (non-causal) multi-head self-attention with optional key padding mask."""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.head_dim
        self.dropout  = config.dropout

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.c_proj = nn.Linear(config.d_model, config.d_model,     bias=True)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x:           torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, T, C)
            key_padding_mask: (B, T) bool tensor; True = position is padding
                              (should be masked out of attention).
        """
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split(q), _split(k), _split(v)

        if self.flash:
            attn_mask = None
            if key_padding_mask is not None:
                # Convert (B, T) bool mask to (B, 1, 1, T) float mask
                # Masked positions get -inf so softmax produces 0 weight
                attn_mask = (~key_padding_mask).float().unsqueeze(1).unsqueeze(2)
                attn_mask = (1.0 - attn_mask) * -1e9
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn  = (q @ k.transpose(-2, -1)) * scale        # (B, nh, T, T)
            if key_padding_mask is not None:
                attn = attn.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            attn = F.softmax(attn, dim=-1)
            y    = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        hidden = config.d_model * 4
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
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Bidirectional transformer text encoder.

    Encodes variable-length text sequences to fixed-size vectors.
    The representation of the [EOT] token (which the encoder sees last
    in the sequence) is used as the global text embedding.

    Following CLIP: a projection head maps the d_model representation
    to the shared embedding space (embed_dim) via a linear layer
    (added externally in the CLIP model to keep this class reusable).
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config  = config
        self.tokenizer = ByteTokenizer()

        # Embeddings
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop      = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f   = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.tok_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) integer token ids.  Padding positions hold PAD_ID.

        Returns:
            (B, d_model) — [EOT] token representation after final LayerNorm.
        """
        B, T = tokens.shape
        assert T <= self.config.max_seq_len, (
            f"Token sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Build padding mask: True where token == PAD_ID
        pad_mask = (tokens == ByteTokenizer.PAD_ID)   # (B, T)

        # Embeddings
        pos = torch.arange(T, device=tokens.device)
        x   = self.drop(self.tok_embed(tokens) + self.pos_embed(pos))

        # Transformer encoder
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        x = self.ln_f(x)   # (B, T, d_model)

        # Extract [EOT] token: it's the last non-padding token in each sequence.
        # Since all sequences are padded to the same length, we find [EOT] position
        # as the index of the EOT_ID in each row.
        eot_mask = (tokens == ByteTokenizer.EOT_ID)        # (B, T)
        # Take the last EOT position per batch element (in case of truncation edge cases)
        eot_pos  = eot_mask.long().cumsum(dim=1).clamp(max=1).bool()
        eot_idx  = eot_mask.float().argmax(dim=1)          # (B,) first EOT per row

        # Gather the [EOT] representation
        eot_idx_expanded = eot_idx.view(B, 1, 1).expand(B, 1, self.config.d_model)
        text_feats = x.gather(1, eot_idx_expanded).squeeze(1)  # (B, d_model)

        return text_feats

    def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        Convenience method: tokenise and encode a list of strings.

        Returns:
            (N, d_model) text features.
        """
        tokens = self.tokenizer.encode_batch(texts, max_len=self.config.max_seq_len)
        tokens = tokens.to(device)
        with torch.no_grad():
            return self.forward(tokens)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (
            f"TextEncoder({self.config.n_layers}L x {self.config.n_heads}H x "
            f"{self.config.d_model}D | {n / 1e6:.2f}M params)"
        )
