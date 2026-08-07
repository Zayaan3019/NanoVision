# NanoVision: ViT + CLIP from Scratch

A complete, production-grade implementation of two foundational vision models built entirely from scratch using only PyTorch, NumPy, and Pillow — no torchvision dependency.

| Model | Task | Architecture |
|-------|------|-------------|
| **ViT** | Image classification | Vision Transformer (Dosovitskiy et al., 2021) |
| **CLIP** | Contrastive vision-language | Radford et al., 2021 |

Both models are trained on CIFAR-10 and evaluated with a full metrics suite including per-class accuracy, zero-shot classification, and cross-modal retrieval (R@1/R@5/R@10).

> **Audit disclosure.** This repository underwent a correctness and
> evaluation-integrity audit (see [`AUDIT.md`](AUDIT.md)) that found and
> fixed a real retrieval-evaluation bug, a real crash in two entry-point
> scripts, upgraded Mixup to per-sample lambda, added reproducibility
> (seeding), and corrected two unverified accuracy claims. Every number
> below is reproducible by a single command — see [`RESULTS.md`](RESULTS.md).
> Architecture/dataset/epoch details are in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Architecture

### Vision Transformer (ViT)

```
Image (3, 32, 32)
    -> PatchEmbedding  Conv2d(3, d, P, P)     (B, 64, d)
    -> prepend [CLS]                           (B, 65, d)
    -> + pos_embed (learnable)                 (B, 65, d)
    -> N x Block [ Pre-LN -> FlashAttn -> Pre-LN -> MLP -> DropPath ]
    -> LayerNorm
    -> [CLS] token                             (B, d)
    -> Linear head                             (B, 10)
```

Key design choices:
- **Patch size 4** for 32x32 input → 64 patches (vs. patch 16 for ImageNet 224x224)
- **Pre-LayerNorm** throughout (more stable than post-LN at depth)
- **Flash Attention** via `F.scaled_dot_product_attention` with `is_causal=False`
- **Stochastic depth** (DropPath) linearly increasing from 0 to `drop_path` across layers
- **No causal masking** — all patches attend to all other patches
- **Mixup** data augmentation with KL divergence loss for soft-label training

Model presets (patch=4, img=32):

| Preset | d_model | Layers | Heads | Params |
|--------|---------|--------|-------|--------|
| tiny   | 192     | 6      | 6     | 2.69M  |
| small  | 384     | 8      | 6     | 14.2M  |
| base   | 768     | 12     | 12    | 85.2M  |

### CLIP (Contrastive Language-Image Pre-training)

```
Image  (B, 3, 32, 32)  ->  ViT.forward_features()  ->  Linear  ->  L2-norm  -> (B, embed_dim)
                                                                                      |
                                                         symmetric InfoNCE loss       |
                                                         (N x N similarity matrix)    |
                                                                                      |
Text   (B, T)          ->  TextEncoder (bidirectional) ->  Linear  ->  L2-norm  -> (B, embed_dim)
```

**Text encoder**: Bidirectional transformer with byte-level tokenizer.
- Vocabulary: 131 tokens — all 128 ASCII bytes + `[PAD]=128`, `[SOT]=129`, `[EOT]=130`
- Representation: `[EOT]` token embedding (following the original CLIP paper)
- No `<unk>` token needed — all characters map to their byte value

**InfoNCE loss** (symmetric):
```
S[i,j] = img_feats[i] · text_feats[j]    (NxN cosine similarity)
logits  = S * exp(log_temp)              (learnable temperature, clamped to [0, log(100)])
loss    = 0.5 * ( CE(logits_rows, diag) + CE(logits_cols, diag) )
```

**Zero-shot classification**: Encode `"a photo of a {class}"` for each CIFAR-10 class, then assign each image to the class with highest cosine similarity. This exact fixed template is used (not tuned against test accuracy — see `AUDIT.md` §3, item 6).

**Cross-modal retrieval**: Build a brute-force embedding index. Given a text query, return the top-K most similar images (exact cosine similarity, O(N·d)). Recall@K is reported in TWO conventions: strict (query `i`'s only correct answer is gallery item `i`) and **class-aware** (any top-K item sharing the query's class counts as a hit). Strict recall is the standard metric when every caption is unique; NanoVision's captions are one template per CIFAR-10 class, so many gallery items legitimately share an identical caption, and class-aware recall is the metric that actually reflects retrieval quality here — see `AUDIT.md` §3, item 7 for the measured gap between the two.

---

## Project Structure

```
vision-from-scratch/
├── nanovision/
│   ├── model/
│   │   ├── config.py          # ViTConfig, TextEncoderConfig, CLIPConfig (tiny/micro), TrainConfig
│   │   ├── vit.py             # PatchEmbedding, SelfAttention, Block, ViT
│   │   ├── text_encoder.py    # ByteTokenizer, TextEncoder
│   │   ├── clip.py            # ProjectionHead, CLIP (encode_image, encode_text, InfoNCE)
│   │   └── cnn.py             # BaselineCNN — ResNet9-style CNN, comparison baseline only
│   ├── data/
│   │   ├── augmentation.py    # PIL/NumPy transforms: RandomCrop, ColorJitter, Mixup, Normalize
│   │   └── dataset.py         # load_cifar10, CIFAR10Dataset, CIFAR10CLIPDataset
│   ├── training/
│   │   ├── scheduler.py       # cosine LR with linear warmup
│   │   └── trainer.py         # ViTTrainer (Mixup + label smoothing), CLIPTrainer (InfoNCE + retrieval eval)
│   ├── retrieval/
│   │   └── index.py           # EmbeddingIndex (build, query_text, query_image, recall@K, save/load)
│   └── utils/
│       ├── checkpoint.py      # save/load checkpoints (self-describing .pt files)
│       ├── metrics.py         # accuracy (top-K), retrieval_recall (+class-aware), mAP, confusion_matrix
│       └── seed.py            # set_seed() — reproducibility (random/numpy/torch, optional determinism)
├── scripts/
│   └── classify_tests.py      # Reproducibly classifies every test as shape-only/numeric/structural
├── tests/
│   ├── test_vit.py            # ViT: shapes, attention, DropPath, gradcheck, init, optimizer
│   ├── test_clip.py           # CLIP: tokenizer, encoder, InfoNCE closed-form, gradcheck, presets
│   ├── test_retrieval.py      # index build/query, recall@K, save/load, metrics
│   ├── test_augmentation.py   # RandomCrop reflect-padding correctness, Mixup per-sample lambda, hue shift
│   └── test_cnn.py            # baseline CNN shapes/gradients/optimizer
├── prepare_data.py            # Download + verify CIFAR-10
├── train_vit.py               # ViT training entry point
├── train_clip.py              # CLIP training entry point (--preset tiny|micro)
├── train_baseline_cnn.py      # CNN baseline training entry point (comparison only)
├── evaluate.py                # Full evaluation (accuracy + retrieval metrics, strict + class-aware)
├── retrieve.py                # Interactive retrieval demo
├── AUDIT.md                   # Correctness / evaluation-integrity audit findings
├── MODEL_CARD.md               # Architecture, dataset, augmentation, epochs, baseline comparison
├── RESULTS.md                  # Every reported number + the exact command that reproduces it
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CIFAR-10

```bash
python prepare_data.py
```

### 3. Train ViT

```bash
# Tiny preset — ~2h on CPU, ~15 min on GPU
python train_vit.py

# GPU with larger batch
python train_vit.py --device cuda --batch_size 256

# Override hyperparameters
python train_vit.py --max_epochs 50 --max_lr 3e-4 --dropout 0.1

# Resume from checkpoint
python train_vit.py --resume checkpoints/vit_best.pt
```

### 4. Train CLIP

```bash
# Tiny preset (reference architecture) — GPU recommended; ~28 min/epoch on CPU
python train_clip.py --preset tiny --device cuda

# Micro preset — trains to completion in well under 2h on CPU-only hardware
python train_clip.py --preset micro --max_epochs 15

# Multi-template prompts for stronger embeddings (training-time caption augmentation only)
python train_clip.py --multi_template

# Override epochs / batch size
python train_clip.py --max_epochs 30 --batch_size 128
```

### 4b. Train the CNN baseline (comparison only, not part of the core architecture)

```bash
python train_baseline_cnn.py --widths 32 64 128 256 --max_epochs 12
```

### 5. Evaluate

```bash
# ViT: top-1/top-5 accuracy + per-class breakdown
python evaluate.py --vit_ckpt checkpoints/vit_best.pt

# CLIP: zero-shot accuracy + R@1/R@5/R@10
python evaluate.py --clip_ckpt checkpoints/clip_best.pt

# Both at once
python evaluate.py --vit_ckpt checkpoints/vit_best.pt --clip_ckpt checkpoints/clip_best.pt
```

### 6. Cross-modal retrieval demo

```bash
# Build embedding index
python retrieve.py --build --clip_ckpt checkpoints/clip_best.pt

# Query with text
python retrieve.py --query "a photo of a dog" --clip_ckpt checkpoints/clip_best.pt

# Evaluate recall@K on test set
python retrieve.py --eval --clip_ckpt checkpoints/clip_best.pt
```

### 7. Run tests

```bash
python -m pytest tests/ -v
# 146 tests, all passing — see AUDIT.md §1 for what they actually
# assert (numerics/gradients vs. shape-only), reproducible via:
python scripts/classify_tests.py
```

---

## Actually Achieved Results

These are **measured, reproducible numbers**, not projections — see
[`RESULTS.md`](RESULTS.md) for the exact command behind each one and
[`MODEL_CARD.md`](MODEL_CARD.md) for full training details. The
previous version of this README quoted unverified target ranges for
both models; those have been replaced.

### ViT-Tiny on CIFAR-10 (100 epochs)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **83.68%** |
| Top-5 Accuracy | **98.86%** |

### CNN baseline vs. ViT-Tiny (12 epochs, 1.65M params, smaller than ViT-Tiny's 2.69M)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **87.96%** (beats ViT-Tiny's 83.68%, in 8x fewer epochs) |
| Top-5 Accuracy | **99.37%** |

A ViT trained from scratch on CIFAR-10 losing to a smaller CNN is a
well-known, legitimate finding about data/compute efficiency, not an
implementation bug — see [`MODEL_CARD.md`](MODEL_CARD.md) §4 for the full
discussion and correctness checks that rule out the alternative
explanation (a broken ViT).

### CLIP-Micro on CIFAR-10 (15 epochs, 0.83M params)

| Metric | Value |
|--------|-------|
| Zero-Shot Top-1 | **64.22%** |
| Zero-Shot Top-5 | **96.87%** |
| R@1 (i2t, strict index) | 0.3% |
| R@1 (i2t, class-aware) | **62.3%** |

Strict and class-aware recall disagree by >60 points here — see
[`MODEL_CARD.md`](MODEL_CARD.md) §3 for why (the dataset has only 10
unique captions total) and why class-aware is the metric that actually
tracks model quality. `CLIPConfig.tiny()` remains the documented reference
architecture; `micro` was used so a real, fully-trained checkpoint (rather
than an unverified number) could be reported within this environment's
CPU-only compute budget.

> Results vary with random seed and hardware; all commands now accept
> `--seed` (default 42) for reproducibility — see `AUDIT.md` §5.

---

## Design Philosophy

**No torchvision.** Every transform (random crop with reflect padding, ColorJitter via PIL ImageEnhance, Mixup, Normalize) is implemented from scratch using Pillow + NumPy. This makes the augmentation pipeline fully transparent and dependency-minimal — transparent enough that this audit caught and fixed a real bug in it: the hand-rolled reflect-padding never painted the four corner blocks of the padded canvas (they silently stayed black), and hue jitter's pure-Python per-pixel `colorsys` loop was the dominant cost in CLIP data loading — now vectorised with numpy (verified pixel-exact against `colorsys`, ~4.5x faster). See `tests/test_augmentation.py`.

**Self-describing checkpoints.** Every `.pt` file stores the model config alongside the weights, so `load_vit_from_checkpoint()` and `load_clip_from_checkpoint()` reconstruct the exact model without any external config file.

**Flash Attention via SDPA.** `F.scaled_dot_product_attention` dispatches to Flash Attention on CUDA and falls back to a manual implementation on CPU. `is_causal=False` for both ViT (full patch attention) and the text encoder (bidirectional).

**Learnable temperature.** CLIP's temperature is stored as `log_temp` (a scalar parameter) and clamped to `[0, log(100)]` during the forward pass. This prevents numerical instability while allowing the model to learn the optimal temperature.

**Exact retrieval.** The `EmbeddingIndex` uses brute-force cosine similarity (`q @ db.T`). This is exact and fast enough for CIFAR-10's 10K test images. For larger corpora, replace with an approximate method (e.g. HNSW via `faiss` or `hnswlib`).

---

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021) — [arxiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (2021) — [arxiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Huang et al., "Deep Networks with Stochastic Depth" (2016) — [arxiv:1603.09382](https://arxiv.org/abs/1603.09382)
- Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018) — [arxiv:1710.09412](https://arxiv.org/abs/1710.09412)
