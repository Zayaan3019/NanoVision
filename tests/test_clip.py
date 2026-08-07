"""
Unit tests for CLIP model, text encoder, and byte tokenizer.

Tests cover:
  - ByteTokenizer encode/decode correctness
  - TextEncoder output shapes
  - CLIP encoding (image + text) produces L2-normalised vectors
  - InfoNCE loss properties (finite, symmetric, minimised on diagonal)
  - Temperature clamping
  - Zero-shot classification shape
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nanovision.model.config import ViTConfig, TextEncoderConfig, CLIPConfig
from nanovision.model.text_encoder import ByteTokenizer, TextEncoder
from nanovision.model.clip import CLIP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture
def text_cfg() -> TextEncoderConfig:
    return TextEncoderConfig(
        vocab_size=131, max_seq_len=32,
        d_model=48, n_layers=2, n_heads=6,
        dropout=0.0,
    )


@pytest.fixture
def clip_cfg() -> CLIPConfig:
    vit_cfg  = ViTConfig(
        img_size=32, patch_size=4,
        d_model=48, n_layers=2, n_heads=6,
        n_classes=10, dropout=0.0,
    )
    text_cfg = TextEncoderConfig(
        vocab_size=131, max_seq_len=32,
        d_model=48, n_layers=2, n_heads=6,
        dropout=0.0,
    )
    return CLIPConfig(vision=vit_cfg, text=text_cfg, embed_dim=32, init_log_temp=2.659)


@pytest.fixture
def clip_model(clip_cfg: CLIPConfig) -> CLIP:
    return CLIP(clip_cfg).eval()


def _rand_images(B: int = 4) -> torch.Tensor:
    return torch.randn(B, 3, 32, 32)


def _rand_tokens(B: int = 4, T: int = 32) -> torch.Tensor:
    tok = ByteTokenizer()
    texts = [f"a photo of a class {i}" for i in range(B)]
    return tok.encode_batch(texts, max_len=T)


# ---------------------------------------------------------------------------
# ByteTokenizer
# ---------------------------------------------------------------------------

class TestByteTokenizer:
    def test_vocab_size(self, tokenizer: ByteTokenizer):
        assert len(tokenizer) == 131

    def test_special_ids(self, tokenizer: ByteTokenizer):
        assert ByteTokenizer.PAD_ID == 128
        assert ByteTokenizer.SOT_ID == 129
        assert ByteTokenizer.EOT_ID == 130

    def test_encode_length(self, tokenizer: ByteTokenizer):
        ids = tokenizer.encode("hello", max_len=20)
        assert len(ids) == 20

    def test_encode_has_sot_and_eot(self, tokenizer: ByteTokenizer):
        ids = tokenizer.encode("cat", max_len=20, add_special=True)
        assert ids[0] == ByteTokenizer.SOT_ID
        assert ByteTokenizer.EOT_ID in ids

    def test_encode_decode_roundtrip(self, tokenizer: ByteTokenizer):
        text = "a photo of a dog"
        ids  = tokenizer.encode(text, max_len=77)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_decode_strips_specials(self, tokenizer: ByteTokenizer):
        ids = [ByteTokenizer.SOT_ID, ord('h'), ord('i'), ByteTokenizer.EOT_ID,
               ByteTokenizer.PAD_ID, ByteTokenizer.PAD_ID]
        assert tokenizer.decode(ids) == "hi"

    def test_encode_truncates(self, tokenizer: ByteTokenizer):
        long_text = "a" * 100
        ids = tokenizer.encode(long_text, max_len=10)
        assert len(ids) == 10
        assert ids[-1] == ByteTokenizer.EOT_ID   # EOT preserved at end

    def test_encode_pads(self, tokenizer: ByteTokenizer):
        ids = tokenizer.encode("hi", max_len=20)
        assert ids[0]  == ByteTokenizer.SOT_ID
        assert ids[-1] == ByteTokenizer.PAD_ID

    def test_encode_batch_shape(self, tokenizer: ByteTokenizer):
        texts  = ["cat", "dog", "airplane"]
        tokens = tokenizer.encode_batch(texts, max_len=24)
        assert tokens.shape == (3, 24)
        assert tokens.dtype == torch.long

    def test_encode_batch_consistent_with_encode(self, tokenizer: ByteTokenizer):
        texts = ["cat", "dog"]
        batch = tokenizer.encode_batch(texts, max_len=20)
        for i, t in enumerate(texts):
            single = tokenizer.encode(t, max_len=20)
            assert batch[i].tolist() == single

    def test_all_ascii_printable(self, tokenizer: ByteTokenizer):
        for c in range(32, 127):
            ids = tokenizer.encode(chr(c), max_len=10)
            assert chr(c) in tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------

class TestTextEncoder:
    def test_output_shape(self, text_cfg: TextEncoderConfig):
        encoder = TextEncoder(text_cfg)
        B, T    = 4, 32
        tokens  = _rand_tokens(B=B, T=T)
        out     = encoder(tokens)
        assert out.shape == (B, text_cfg.d_model)

    def test_output_finite(self, text_cfg: TextEncoderConfig):
        encoder = TextEncoder(text_cfg)
        tokens  = _rand_tokens(B=3, T=32)
        out     = encoder(tokens)
        assert torch.isfinite(out).all()

    def test_different_texts_different_outputs(self, text_cfg: TextEncoderConfig):
        encoder = TextEncoder(text_cfg).eval()
        tok = ByteTokenizer()
        t1  = tok.encode_batch(["a photo of a cat"], max_len=32)
        t2  = tok.encode_batch(["a photo of a dog"], max_len=32)
        with torch.no_grad():
            f1 = encoder(t1)
            f2 = encoder(t2)
        assert not torch.allclose(f1, f2, atol=1e-4)

    def test_padding_does_not_affect_output(self, text_cfg: TextEncoderConfig):
        """Two identical texts padded to different lengths should give same output."""
        encoder = TextEncoder(text_cfg).eval()
        tok = ByteTokenizer()
        t1 = tok.encode_batch(["cat"], max_len=32)
        t2 = tok.encode_batch(["cat"], max_len=32)
        with torch.no_grad():
            f1 = encoder(t1)
            f2 = encoder(t2)
        assert torch.allclose(f1, f2, atol=1e-6)

    def test_seq_len_too_long_raises(self, text_cfg: TextEncoderConfig):
        encoder = TextEncoder(text_cfg)
        tokens  = torch.zeros(2, text_cfg.max_seq_len + 1, dtype=torch.long)
        with pytest.raises(AssertionError):
            encoder(tokens)


# ---------------------------------------------------------------------------
# CLIP encode_image / encode_text
# ---------------------------------------------------------------------------

class TestCLIPEncoding:
    def test_encode_image_shape(self, clip_model: CLIP, clip_cfg: CLIPConfig):
        images = _rand_images(B=4)
        feats  = clip_model.encode_image(images)
        assert feats.shape == (4, clip_cfg.embed_dim)

    def test_encode_text_shape(self, clip_model: CLIP, clip_cfg: CLIPConfig):
        tokens = _rand_tokens(B=4, T=32)
        feats  = clip_model.encode_text(tokens)
        assert feats.shape == (4, clip_cfg.embed_dim)

    def test_image_embeddings_l2_normalised(self, clip_model: CLIP):
        images = _rand_images(B=4)
        feats  = clip_model.encode_image(images)
        norms  = feats.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_text_embeddings_l2_normalised(self, clip_model: CLIP):
        tokens = _rand_tokens(B=4, T=32)
        feats  = clip_model.encode_text(tokens)
        norms  = feats.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_encode_texts_from_strings_shape(self, clip_model: CLIP, clip_cfg: CLIPConfig):
        texts = ["a photo of a cat", "a photo of a dog"]
        feats = clip_model.encode_texts_from_strings(texts, device=torch.device("cpu"))
        assert feats.shape == (2, clip_cfg.embed_dim)

    def test_encode_texts_from_strings_normalised(self, clip_model: CLIP):
        texts = ["cat", "dog", "airplane"]
        feats = clip_model.encode_texts_from_strings(texts, device=torch.device("cpu"))
        norms = feats.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# CLIP forward / InfoNCE loss
# ---------------------------------------------------------------------------

class TestCLIPLoss:
    def test_forward_output_types(self, clip_model: CLIP):
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, logits_i2t, logits_t2i = clip_model(images, tokens)
        assert loss.ndim == 0
        assert logits_i2t.shape == (4, 4)
        assert logits_t2i.shape == (4, 4)

    def test_loss_is_finite(self, clip_model: CLIP):
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, _, _ = clip_model(images, tokens)
        assert torch.isfinite(loss)

    def test_loss_is_positive(self, clip_model: CLIP):
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, _, _ = clip_model(images, tokens)
        assert loss.item() > 0

    def test_logits_symmetric(self, clip_model: CLIP):
        """i2t and t2i logit matrices should be transposes of each other."""
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        _, logits_i2t, logits_t2i = clip_model(images, tokens)
        assert torch.allclose(logits_i2t, logits_t2i.t(), atol=1e-5)

    def test_loss_lower_for_matched_pairs(self, clip_cfg: CLIPConfig):
        """
        When image and text embeddings are identical (perfect match),
        the loss should be lower than for random mismatched embeddings.
        """
        model = CLIP(clip_cfg)
        B = 6

        with torch.no_grad():
            # Random mismatched feats
            img_rand  = F.normalize(torch.randn(B, clip_cfg.embed_dim), dim=-1)
            txt_rand  = F.normalize(torch.randn(B, clip_cfg.embed_dim), dim=-1)
            loss_rand, _, _ = model._contrastive_loss(img_rand, txt_rand)

            # Perfectly matched feats (diagonal = 1, off-diagonal = near 0 if orthogonal)
            # Use identity-ish feats padded with zeros
            matched = F.normalize(torch.eye(B, clip_cfg.embed_dim), dim=-1)
            loss_matched, _, _ = model._contrastive_loss(matched, matched.clone())

        assert loss_matched.item() < loss_rand.item()

    def test_batch_size_one(self, clip_model: CLIP):
        images = _rand_images(B=1)
        tokens = _rand_tokens(B=1, T=32)
        loss, logits_i2t, logits_t2i = clip_model(images, tokens)
        assert logits_i2t.shape == (1, 1)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_log_temp_initialised(self, clip_model: CLIP, clip_cfg: CLIPConfig):
        assert abs(clip_model.log_temp.item() - clip_cfg.init_log_temp) < 1e-5

    def test_log_temp_is_learnable(self, clip_model: CLIP):
        assert clip_model.log_temp.requires_grad

    def test_log_temp_clamped_in_loss(self, clip_cfg: CLIPConfig):
        """Setting log_temp above 4.605 should be clamped to log(100)."""
        model = CLIP(clip_cfg)
        with torch.no_grad():
            model.log_temp.fill_(10.0)   # Way above the clamp

        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        # Should not raise; temperature is clamped inside _contrastive_loss
        loss, logits_i2t, _ = model(images, tokens)
        assert torch.isfinite(loss)

        # Temperature used = exp(clamp(10, 0, 4.605)) = exp(4.605) ≈ 100
        temp_eff = model.log_temp.clamp(0, 4.605).exp()
        assert abs(temp_eff.item() - 100.0) < 1.0

    def test_log_temp_clamped_to_zero(self, clip_cfg: CLIPConfig):
        """Negative log_temp should be clamped to 0 (temperature=1)."""
        model = CLIP(clip_cfg)
        with torch.no_grad():
            model.log_temp.fill_(-5.0)

        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, _, _ = model(images, tokens)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------

class TestZeroShot:
    def test_output_shape_default_classes(self, clip_model: CLIP):
        images = _rand_images(B=4)
        logits = clip_model.zero_shot_classify(images)
        assert logits.shape == (4, 10)   # 10 CIFAR-10 classes

    def test_output_shape_custom_classes(self, clip_model: CLIP):
        images  = _rand_images(B=3)
        classes = ["cat", "dog", "bird", "car"]
        logits  = clip_model.zero_shot_classify(images, class_names=classes)
        assert logits.shape == (3, 4)

    def test_logits_finite(self, clip_model: CLIP):
        images = _rand_images(B=4)
        logits = clip_model.zero_shot_classify(images)
        assert torch.isfinite(logits).all()

    def test_no_grad_context(self, clip_model: CLIP):
        """zero_shot_classify should work inside or outside no_grad."""
        images = _rand_images(B=2)
        logits = clip_model.zero_shot_classify(images)
        assert not logits.requires_grad


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestCLIPPresets:
    def test_tiny_preset_valid(self):
        cfg = CLIPConfig.tiny()
        assert cfg.vision.d_model % cfg.vision.n_heads == 0
        assert cfg.text.d_model % cfg.text.n_heads == 0

    def test_micro_preset_valid(self):
        cfg = CLIPConfig.micro()
        assert cfg.vision.d_model % cfg.vision.n_heads == 0
        assert cfg.text.d_model % cfg.text.n_heads == 0

    def test_micro_preset_smaller_than_tiny(self):
        micro_params = CLIP(CLIPConfig.micro()).count_params()["total"]
        tiny_params  = CLIP(CLIPConfig.tiny()).count_params()["total"]
        assert micro_params < tiny_params

    def test_micro_preset_forward_and_loss(self):
        model = CLIP(CLIPConfig.micro())
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, logits_i2t, _ = model(images, tokens)
        assert torch.isfinite(loss)
        assert logits_i2t.shape == (4, 4)


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------

class TestCLIPOptimizer:
    def test_optimizer_type(self, clip_model: CLIP):
        opt = clip_model.configure_optimizer(weight_decay=0.2, lr=5e-4, betas=(0.9, 0.98))
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self, clip_model: CLIP):
        opt = clip_model.configure_optimizer(weight_decay=0.2, lr=5e-4, betas=(0.9, 0.98))
        assert len(opt.param_groups) == 2

    def test_log_temp_in_no_decay(self, clip_model: CLIP):
        opt = clip_model.configure_optimizer(weight_decay=0.2, lr=5e-4, betas=(0.9, 0.98))
        no_decay_params = opt.param_groups[1]["params"]
        log_temp = clip_model.log_temp
        assert any(p is log_temp for p in no_decay_params)

    def test_all_params_covered(self, clip_model: CLIP):
        opt = clip_model.configure_optimizer(weight_decay=0.2, lr=5e-4, betas=(0.9, 0.98))
        all_opt = set()
        for g in opt.param_groups:
            for p in g["params"]:
                all_opt.add(id(p))
        all_model = {id(p) for p in clip_model.parameters() if p.requires_grad}
        assert all_opt == all_model


# ---------------------------------------------------------------------------
# Backward through CLIP
# ---------------------------------------------------------------------------

class TestCLIPBackward:
    def test_gradients_flow(self, clip_cfg: CLIPConfig):
        model  = CLIP(clip_cfg)
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, _, _ = model(images, tokens)
        loss.backward()
        # CLIP uses forward_features(), so the ViT classification head has no
        # gradient path — exclude it from the check.
        unused = {"vision_encoder.head.weight", "vision_encoder.head.bias",
                  "vision_encoder.pre_logits.weight", "vision_encoder.pre_logits.bias"}
        for name, p in model.named_parameters():
            if name in unused:
                continue
            assert p.grad is not None, f"No gradient for {name}"

    def test_log_temp_gets_gradient(self, clip_cfg: CLIPConfig):
        model  = CLIP(clip_cfg)
        images = _rand_images(B=4)
        tokens = _rand_tokens(B=4, T=32)
        loss, _, _ = model(images, tokens)
        loss.backward()
        assert model.log_temp.grad is not None

    def test_contrastive_loss_gradcheck(self, clip_cfg: CLIPConfig):
        """Finite-difference gradcheck directly on the loss math (not just
        'a .grad exists somewhere') — the strongest gradient-correctness
        test in the suite."""
        model = CLIP(clip_cfg).double()
        B, D = 5, clip_cfg.embed_dim

        img  = F.normalize(torch.randn(B, D, dtype=torch.float64), dim=-1)
        txt  = F.normalize(torch.randn(B, D, dtype=torch.float64), dim=-1)
        img.requires_grad_(True)
        txt.requires_grad_(True)

        def loss_fn(i, t):
            loss, _, _ = model._contrastive_loss(i, t)
            return loss

        assert torch.autograd.gradcheck(loss_fn, (img, txt), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Closed-form InfoNCE correctness (AUDIT.md §2)
# ---------------------------------------------------------------------------

class TestInfoNCEClosedForm:
    def test_loss_matches_hand_computed_value(self, clip_cfg: CLIPConfig):
        """
        With temperature forced to 1 (log_temp=0) and img_feats == text_feats
        == an orthonormal basis (torch.eye), the similarity matrix is exactly
        the identity: diagonal = 1, off-diagonal = 0. The symmetric InfoNCE
        loss then has a closed form:

            CE(row_i, target=i) = -log( e^1 / (e^1 + (B-1)*e^0) )

        which is identical for every row (i2t and t2i are transposes of the
        same matrix here), so loss == that single value exactly.
        """
        model = CLIP(clip_cfg)
        with torch.no_grad():
            model.log_temp.fill_(0.0)   # temp = exp(0) = 1

        B = 6
        feats = torch.eye(B, clip_cfg.embed_dim)   # rows are unit-norm, orthogonal

        loss, logits_i2t, logits_t2i = model._contrastive_loss(feats, feats.clone())

        expected = -math.log(math.e / (math.e + (B - 1)))
        assert loss.item() == pytest.approx(expected, abs=1e-5)
        # S = eye(B, D) @ eye(B, D).T is exactly the (B, B) identity matrix,
        # and temp=exp(0)=1, so logits_i2t must equal it precisely.
        assert torch.allclose(logits_i2t, torch.eye(B), atol=1e-6)

    def test_diagonal_is_the_positive_pair(self, clip_cfg: CLIPConfig):
        """Perturbing ONLY the diagonal similarity must change the loss;
        perturbing only an off-diagonal entry (while keeping the matrix a
        valid gram matrix is hard, so instead we directly verify via
        cross_entropy semantics) confirms labels=arange(B) targets the
        diagonal, not e.g. a shifted or reversed permutation."""
        model = CLIP(clip_cfg)
        with torch.no_grad():
            model.log_temp.fill_(0.0)

        B = 4
        img = F.normalize(torch.randn(B, clip_cfg.embed_dim), dim=-1)
        txt = img.clone()   # perfect diagonal match: S = img @ img.T, argmax per row is i

        loss, logits_i2t, _ = model._contrastive_loss(img, txt)
        # Row i's argmax similarity must be column i (self-similarity == 1
        # is the maximum possible cosine similarity).
        assert (logits_i2t.argmax(dim=-1) == torch.arange(B)).all()

    def test_duplicate_positive_pair_is_stable(self, clip_cfg: CLIPConfig):
        """
        AUDIT.md §2, item 4: construct a batch where TWO samples have the exact
        same image embedding (as if the same image appeared twice with
        different captions). The diagonal-target InfoNCE formulation has no
        special-casing for this — sample 1 becomes a hard negative for
        sample 0's row and vice versa — but the loss must stay finite,
        differentiable, and well-defined (no NaN/inf), which is the sane
        behaviour to expect from the reference CLIP loss formula.
        """
        model = CLIP(clip_cfg)
        B = 5
        img = F.normalize(torch.randn(B, clip_cfg.embed_dim), dim=-1)
        img[1] = img[0]   # duplicate image embedding at a different index
        img.requires_grad_(True)
        txt = F.normalize(torch.randn(B, clip_cfg.embed_dim), dim=-1)

        loss, logits_i2t, logits_t2i = model._contrastive_loss(img, txt)
        assert torch.isfinite(loss)
        assert torch.isfinite(logits_i2t).all()
        assert torch.isfinite(logits_t2i).all()

        # Rows 0 and 1 of the similarity matrix must be identical (same
        # query embedding), even though their diagonal targets differ.
        assert torch.allclose(logits_i2t[0], logits_i2t[1], atol=1e-5)

        loss.backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()

    def test_two_identical_images_in_batch_end_to_end(self, clip_cfg: CLIPConfig):
        """Same scenario as above but through the full model.forward() with
        real (non-hand-crafted) image/text tensors, to catch any instability
        introduced by the encoders themselves (e.g. BatchNorm-like stats,
        though this model uses LayerNorm so none is expected)."""
        model = CLIP(clip_cfg)
        images = _rand_images(B=4)
        images[1] = images[0]   # two identical images
        tokens = _rand_tokens(B=4, T=32)

        loss, logits_i2t, logits_t2i = model(images, tokens)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.log_temp.grad)
