"""
Unit tests for Vision Transformer (ViT) model.

Tests verify shape contracts, attention properties, weight initialisation,
optimizer configuration, and numerical stability — without requiring data.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovision.model.config import ViTConfig
from nanovision.model.vit import ViT, PatchEmbedding, SelfAttention, Block, drop_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg() -> ViTConfig:
    """Micro ViT that runs in milliseconds on CPU."""
    return ViTConfig(
        img_size=32, patch_size=4,
        d_model=48, n_layers=2, n_heads=6,
        n_classes=10, dropout=0.0, drop_path=0.0,
    )


@pytest.fixture
def tiny_vit(tiny_cfg: ViTConfig) -> ViT:
    return ViT(tiny_cfg)


def _rand_images(B: int = 2, C: int = 3, H: int = 32) -> torch.Tensor:
    return torch.randn(B, C, H, H)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestViTConfig:
    def test_n_patches(self):
        cfg = ViTConfig(img_size=32, patch_size=4)
        assert cfg.n_patches == 64

    def test_seq_len(self):
        cfg = ViTConfig(img_size=32, patch_size=4)
        assert cfg.seq_len == 65   # 64 patches + 1 CLS

    def test_head_dim(self):
        cfg = ViTConfig(d_model=192, n_heads=6)
        assert cfg.head_dim == 32

    def test_mlp_hidden(self):
        cfg = ViTConfig(d_model=192, mlp_ratio=4.0)
        assert cfg.mlp_hidden == 768

    def test_invalid_head_count_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            ViTConfig(d_model=192, n_heads=7)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            ViTConfig(img_size=32, patch_size=6)

    def test_presets_valid(self):
        for preset in (ViTConfig.tiny, ViTConfig.small, ViTConfig.base):
            cfg = preset()
            assert cfg.d_model % cfg.n_heads == 0

    def test_serialisation_roundtrip(self, tiny_cfg: ViTConfig, tmp_path):
        p = tmp_path / "cfg.json"
        tiny_cfg.save(p)
        loaded = ViTConfig.load(p)
        assert loaded.d_model == tiny_cfg.d_model
        assert loaded.n_layers == tiny_cfg.n_layers


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------

class TestPatchEmbedding:
    def test_output_shape(self, tiny_cfg: ViTConfig):
        pe  = PatchEmbedding(tiny_cfg)
        x   = _rand_images(B=3)
        out = pe(x)
        assert out.shape == (3, tiny_cfg.n_patches, tiny_cfg.d_model)

    def test_single_sample(self, tiny_cfg: ViTConfig):
        pe  = PatchEmbedding(tiny_cfg)
        x   = _rand_images(B=1)
        out = pe(x)
        assert out.shape == (1, tiny_cfg.n_patches, tiny_cfg.d_model)


# ---------------------------------------------------------------------------
# ViT forward shapes
# ---------------------------------------------------------------------------

class TestViTForward:
    def test_logits_shape(self, tiny_vit: ViT):
        x      = _rand_images(B=4)
        logits = tiny_vit(x)
        assert logits.shape == (4, 10)

    def test_forward_features_shape(self, tiny_vit: ViT, tiny_cfg: ViTConfig):
        x     = _rand_images(B=4)
        feats = tiny_vit.forward_features(x)
        assert feats.shape == (4, tiny_cfg.d_model)

    def test_return_features_flag(self, tiny_vit: ViT, tiny_cfg: ViTConfig):
        x               = _rand_images(B=2)
        logits, features = tiny_vit(x, return_features=True)
        assert logits.shape   == (2, 10)
        assert features.shape == (2, tiny_cfg.d_model)

    def test_batch_size_one(self, tiny_vit: ViT):
        x = _rand_images(B=1)
        out = tiny_vit(x)
        assert out.shape == (1, 10)

    def test_batch_size_16(self, tiny_vit: ViT):
        x = _rand_images(B=16)
        out = tiny_vit(x)
        assert out.shape == (16, 10)

    def test_output_is_finite(self, tiny_vit: ViT):
        x   = _rand_images(B=4)
        out = tiny_vit(x)
        assert torch.isfinite(out).all()

    def test_eval_mode_deterministic(self, tiny_vit: ViT):
        tiny_vit.eval()
        x  = _rand_images(B=2)
        y1 = tiny_vit(x)
        y2 = tiny_vit(x)
        assert torch.allclose(y1, y2)

    def test_train_eval_differ_with_dropout(self, tiny_cfg: ViTConfig):
        cfg = ViTConfig(
            img_size=32, patch_size=4,
            d_model=48, n_layers=2, n_heads=6,
            n_classes=10, dropout=0.5,
        )
        model = ViT(cfg)
        x  = _rand_images(B=2)
        model.train()
        y1 = model(x)
        y2 = model(x)
        # With 50% dropout, two forward passes on the SAME input must
        # actually differ (this is what the test name promises — the
        # original version only re-checked eval determinism here).
        assert not torch.allclose(y1, y2), \
            "train-mode outputs with dropout=0.5 should differ between calls"

        model.eval()
        y3 = model(x)
        y4 = model(x)
        assert torch.allclose(y3, y4), "eval outputs must be deterministic"


# ---------------------------------------------------------------------------
# Non-causal attention (bidirectional)
# ---------------------------------------------------------------------------

class TestNonCausalAttention:
    def test_attention_is_not_causal(self, tiny_cfg: ViTConfig):
        """
        If attention were causal, zeroing out token i would affect tokens > i.
        With non-causal attention, zeroing token i only affects token i itself
        when we look at symmetry: in particular, the FIRST patch should still
        see the LAST patch.  We verify this indirectly by confirming that all
        output tokens change when we change ANY single input token.
        """
        tiny_cfg_nodrop = ViTConfig(
            img_size=32, patch_size=4,
            d_model=48, n_layers=1, n_heads=6,
            n_classes=10, dropout=0.0,
        )
        model = ViT(tiny_cfg_nodrop).eval()
        x1 = _rand_images(B=1)
        x2 = x1.clone()

        # Perturb only a small patch in the LAST position of the image
        x2[:, :, 28:32, 28:32] += 10.0

        with torch.no_grad():
            f1 = model.forward_features(x1)
            f2 = model.forward_features(x2)

        # All feature dimensions should change (non-local attention)
        assert not torch.allclose(f1, f2, atol=1e-4)


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

class TestDropPath:
    def test_identity_at_eval(self):
        x  = torch.randn(4, 65, 48)
        dp = torch.nn.Module()
        out = drop_path(x, drop_prob=0.5, training=False)
        assert torch.allclose(out, x)

    def test_identity_at_zero_prob(self):
        x   = torch.randn(4, 65, 48)
        out = drop_path(x, drop_prob=0.0, training=True)
        assert torch.allclose(out, x)

    def test_stochastic_during_training(self):
        torch.manual_seed(0)
        x    = torch.ones(64, 65, 48)
        out  = drop_path(x, drop_prob=0.5, training=True)
        # With p=0.5 and 64 samples, some rows should be zeroed
        row_norms = out.view(64, -1).norm(dim=1)
        assert (row_norms == 0).any(), "Expected some samples to be dropped"
        assert (row_norms > 0).any(), "Expected some samples to survive"


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

class TestWeightInit:
    def test_head_bias_zero(self, tiny_vit: ViT):
        assert torch.all(tiny_vit.head.bias == 0)

    def test_ln_f_weight_one(self, tiny_vit: ViT):
        assert torch.all(tiny_vit.ln_f.weight == 1)

    def test_ln_f_bias_zero(self, tiny_vit: ViT):
        assert torch.all(tiny_vit.ln_f.bias == 0)

    def test_cls_token_shape(self, tiny_vit: ViT, tiny_cfg: ViTConfig):
        assert tiny_vit.cls_token.shape == (1, 1, tiny_cfg.d_model)

    def test_pos_embed_shape(self, tiny_vit: ViT, tiny_cfg: ViTConfig):
        assert tiny_vit.pos_embed.shape == (1, tiny_cfg.seq_len, tiny_cfg.d_model)


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------

class TestOptimizerConfig:
    def test_optimizer_type(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.05, lr=1e-3, betas=(0.9, 0.999))
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.05, lr=1e-3, betas=(0.9, 0.999))
        assert len(opt.param_groups) == 2

    def test_weight_decay_applied_correctly(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.05, lr=1e-3, betas=(0.9, 0.999))
        decay_group    = opt.param_groups[0]
        no_decay_group = opt.param_groups[1]
        assert decay_group["weight_decay"]    == 0.05
        assert no_decay_group["weight_decay"] == 0.0

    def test_pos_embed_in_no_decay(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.05, lr=1e-3, betas=(0.9, 0.999))
        no_decay_params = opt.param_groups[1]["params"]
        pos_embed = tiny_vit.pos_embed
        assert any(p is pos_embed for p in no_decay_params)

    def test_all_params_covered(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.05, lr=1e-3, betas=(0.9, 0.999))
        all_opt_params = set()
        for g in opt.param_groups:
            for p in g["params"]:
                all_opt_params.add(id(p))
        all_model_params = {id(p) for p in tiny_vit.parameters() if p.requires_grad}
        assert all_opt_params == all_model_params

    def test_count_params(self, tiny_vit: ViT):
        n = tiny_vit.count_params(non_embedding=False)
        assert n == sum(p.numel() for p in tiny_vit.parameters())

    def test_count_params_non_embedding(self, tiny_vit: ViT):
        n_all  = tiny_vit.count_params(non_embedding=False)
        n_no_e = tiny_vit.count_params(non_embedding=True)
        assert n_no_e < n_all


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

class TestBackward:
    def test_gradients_flow(self, tiny_vit: ViT):
        x       = _rand_images(B=2)
        labels  = torch.zeros(2, dtype=torch.long)
        logits  = tiny_vit(x)
        loss    = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        for name, p in tiny_vit.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_loss_decreases(self, tiny_vit: ViT):
        opt = tiny_vit.configure_optimizer(weight_decay=0.01, lr=1e-2, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()
        x      = _rand_images(B=8)
        labels = torch.zeros(8, dtype=torch.long)

        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = criterion(tiny_vit(x), labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease over 5 steps"


# ---------------------------------------------------------------------------
# Gradient correctness (finite-difference gradcheck, not just "grad exists")
# ---------------------------------------------------------------------------
#
# `test_gradients_flow` above only proves the backward graph is connected
# (every .grad is non-None). It says nothing about whether those gradients
# are numerically CORRECT. `torch.autograd.gradcheck` compares the
# autograd-computed Jacobian against a central finite-difference
# approximation in float64 — this checks gradient VALUES, not just that
# gradients exist.

class TestGradcheck:
    def test_self_attention_gradcheck(self):
        cfg = ViTConfig(
            img_size=8, patch_size=4,
            d_model=8, n_layers=1, n_heads=2,
            n_classes=2, dropout=0.0,
        )
        attn = SelfAttention(cfg).double().eval()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(attn, (x,), eps=1e-6, atol=1e-4)

    def test_patch_embedding_gradcheck(self):
        cfg = ViTConfig(img_size=8, patch_size=4, d_model=6, n_heads=2, n_classes=2)
        pe = PatchEmbedding(cfg).double()
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(pe, (x,), eps=1e-6, atol=1e-4)

    def test_full_vit_block_gradcheck(self):
        cfg = ViTConfig(
            img_size=8, patch_size=4,
            d_model=8, n_layers=1, n_heads=2,
            n_classes=2, dropout=0.0, drop_path=0.0,
        )
        block = Block(cfg, drop_path_rate=0.0).double().eval()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(block, (x,), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Patch embedding: Conv2d must equal an explicit unfold + linear projection
# ---------------------------------------------------------------------------

class TestPatchEmbeddingEquivalence:
    def test_conv2d_equals_manual_unfold_linear(self, tiny_cfg: ViTConfig):
        """
        The docstring claims Conv2d(kernel=stride=patch_size) is "mathematically
        equivalent to flattening each patch and passing it through a shared
        linear layer." Verify that claim numerically rather than assuming it.
        """
        pe = PatchEmbedding(tiny_cfg)
        x  = _rand_images(B=2)

        out = pe(x)   # (B, n_patches, d_model) via Conv2d

        P = tiny_cfg.patch_size
        B, C, H, W = x.shape
        # Unfold into non-overlapping P x P patches: (B, C*P*P, n_patches)
        patches = F.unfold(x, kernel_size=P, stride=P)
        patches = patches.transpose(1, 2)   # (B, n_patches, C*P*P)

        weight = pe.proj.weight.view(tiny_cfg.d_model, -1)   # (d_model, C*P*P)
        bias   = pe.proj.bias                                  # (d_model,)
        manual = patches @ weight.t() + bias                   # (B, n_patches, d_model)

        assert torch.allclose(out, manual, atol=1e-5), \
            "Conv2d patch embedding must equal explicit unfold + linear projection"

    def test_stride_equals_patch_size(self, tiny_cfg: ViTConfig):
        pe = PatchEmbedding(tiny_cfg)
        assert pe.proj.stride == (tiny_cfg.patch_size, tiny_cfg.patch_size)
        assert pe.proj.kernel_size == (tiny_cfg.patch_size, tiny_cfg.patch_size)

    def test_sequence_length_includes_cls(self, tiny_cfg: ViTConfig):
        assert tiny_cfg.seq_len == tiny_cfg.n_patches + 1
        vit = ViT(tiny_cfg)
        assert vit.pos_embed.shape[1] == (tiny_cfg.img_size // tiny_cfg.patch_size) ** 2 + 1


# ---------------------------------------------------------------------------
# DropPath: expectation-preserving rescaling
# ---------------------------------------------------------------------------

class TestDropPathExpectation:
    def test_expectation_preserved_by_inverse_scaling(self):
        """
        DropPath zeroes a residual branch with prob `drop_prob` and rescales
        survivors by 1/keep_prob, so E[output] == input exactly. Verify this
        empirically over many samples (a wrong rescaling factor — e.g.
        forgetting to divide by keep_prob, or using drop_prob instead of
        keep_prob — would shift the sample mean measurably).
        """
        torch.manual_seed(42)
        drop_prob = 0.3
        n_samples = 40_000
        x = torch.ones(n_samples, 4)
        out = drop_path(x, drop_prob=drop_prob, training=True)
        empirical_mean = out.mean().item()
        assert abs(empirical_mean - 1.0) < 0.02, (
            f"E[drop_path(x)] should be ~= x (1.0), got {empirical_mean:.4f} "
            f"— rescaling factor is likely wrong"
        )

    def test_survival_rate_matches_keep_prob(self):
        torch.manual_seed(7)
        drop_prob = 0.4
        n_samples = 20_000
        x = torch.ones(n_samples, 3)
        out = drop_path(x, drop_prob=drop_prob, training=True)
        survived = (out.view(n_samples, -1).norm(dim=1) > 0).float().mean().item()
        assert abs(survived - (1.0 - drop_prob)) < 0.02
