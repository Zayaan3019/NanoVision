# NanoVision Model Card

This card documents exactly what was trained, on what data, for how long,
and how it compares to a simple baseline (`AUDIT.md` §5, item 14).
Every number below is reproducible with the single command listed in
`RESULTS.md`.

## 1. ViT-Tiny (image classification)

| | |
|---|---|
| Architecture preset | `ViTConfig.tiny()` — d_model=192, 6 layers, 6 heads, patch=4, **2,693,578 params** (2.67M in the six transformer blocks, ~24K in the patch/position embeddings, CLS token and head) |
| Dataset | CIFAR-10, canonical 50,000 train / 10,000 test split |
| Augmentation | RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip(0.5) + ColorJitter(0.4,0.4,0.4) + Mixup(alpha=0.2). This checkpoint was trained with **one lambda per batch**; the code now samples lambda per sample (`AUDIT.md` §5, item 11), which applies to new runs only. |
| Optimizer | AdamW, lr 1e-3 (cosine + 5-epoch warmup), wd 0.05, grad clip 1.0 |
| Epochs | 100 |
| Loss | KL against the Mixup soft labels (label smoothing 0.1 when Mixup inactive) |
| Seed | **not recorded** — this checkpoint predates the reproducibility fix in this audit (see `AUDIT.md` §5, item 13). All commands below use a fixed, documented seed going forward. |

**Result: 83.68% top-1 / 98.86% top-5** on the CIFAR-10 test set.

Per-class accuracy (from `python evaluate.py --vit_ckpt checkpoints/vit_best.pt`):

| Class | Accuracy |
|---|---|
| airplane | 86.2% |
| automobile | 92.1% |
| bird | 78.8% |
| **cat** | **66.1%** |
| deer | 78.6% |
| dog | 78.1% |
| frog | 89.3% |
| horse | 88.1% |
| ship | 90.2% |
| truck | 89.3% |

Cat is by far the hardest class (confused with dog — a standard, well-known
CIFAR-10 failure mode), consistent with the literature.

The README previously claimed "~88-90%" top-1 for this exact preset/epoch
count. **That number was not backed by a reproducible run** and has been
replaced with the measured 83.68%.

## 2. CNN baseline (for comparison, not part of the core architecture)

| | |
|---|---|
| Architecture | `BaselineCNN`, ResNet9-style (prep conv + 3 downsampling stages + 2 residual blocks + global max-pool), widths `(32, 64, 128, 256)` — **1,646,122 params (1.65M), about 40% fewer than ViT-Tiny's 2.69M**, so the comparison gives the CNN no capacity advantage |
| Dataset | Same CIFAR-10 split, same `CIFAR10Dataset` loader |
| Augmentation | Identical pipeline to ViT: RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip + ColorJitter(0.4,0.4,0.4). No Mixup (kept simple/standard for the baseline). |
| Optimizer | SGD + Nesterov momentum (0.9), lr 0.05->0.1 (2-epoch warmup) cosine to 1e-4, wd 5e-4, grad clip 1.0 — the standard, well-tuned CNN-on-CIFAR recipe, deliberately NOT AdamW, so this is each architecture's own best-practice optimizer rather than an optimizer ablation |
| Epochs | 12 (vs. ViT's 100 — see "epoch budget" note below) |
| Seed | 42 (`nanovision.utils.seed.set_seed`) |

Command: `python train_baseline_cnn.py --widths 32 64 128 256 --max_epochs 12 --batch_size 256 --seed 42`

**Result: 87.96% top-1 / 99.37% top-5** on the CIFAR-10 test set, reached at
epoch 12 (still improving — val loss was still dropping at the final
epoch; this is not a fully-converged CNN, just a well-trained one, given
the CPU compute budget available).

## 3. CLIP (contrastive vision-language)

| | |
|---|---|
| Architecture preset | `CLIPConfig.micro()` — vision d_model=96/4 layers/4 heads (~0.45M), text d_model=96/3 layers/4 heads (~0.36M), embed_dim=64. **0.83M params total.** Not the README's originally-documented `tiny` preset — see "why micro" below. |
| Dataset | CIFAR-10 image-text pairs, caption = `"a photo of a {class}"` (single template; `--multi_template` not used for this run) |
| Augmentation | RandomCrop + HorizontalFlip + ColorJitter(0.8,0.8,0.8,hue=0.2) + RandomGrayscale(0.2) (`clip_train_transform`) |
| Optimizer | AdamW, lr 5e-4 (cosine + 5-epoch warmup), wd 0.2, grad clip 1.0 |
| Epochs | 15 |
| Batch size | 128 |
| Seed | 42 |

Command: `python train_clip.py --preset micro --max_epochs 15 --batch_size 128 --seed 42`

**Why `micro`, not `tiny`**: `CLIPConfig.tiny()` (the architecture documented
in the README) measured ~2.2s/batch(64) on this machine's CPU — a full
15-50 epoch run would take many hours. No GPU was available in this
environment. Rather than report unverified numbers for `tiny` (every reported number must be regenerable; `AUDIT.md` §5, item 13), a smaller
preset in the same architecture family was trained to completion instead,
so every number below is real. `tiny` remains the documented reference
architecture; reproduce it on a GPU with the command in `RESULTS.md`.

**Result** (`python evaluate.py --clip_ckpt checkpoints/clip_best.pt --n_retrieval 1000`):

| Metric | Value |
|---|---|
| Zero-shot Top-1 | **64.22%** |
| Zero-shot Top-5 | **96.87%** |
| R@1 (i2t, strict index) | 0.3% |
| R@1 (i2t, class-aware) | **62.3%** |
| R@5 (i2t, strict index) | 2.3% |
| R@5 (i2t, class-aware) | 62.3% |
| R@10 (i2t, strict index) | 5.6% |
| R@10 (i2t, class-aware) | 62.3% |

**Two things jump out, and both are explained, not swept under the rug:**

1. **Strict recall stays near-random (0.3-5.6%) for all 15 epochs of
   training**, even as the loss drops from 4.63 to 3.75 and zero-shot
   accuracy climbs from 31.8% to 64.2% — i.e. strict recall is essentially
   *uninformative* about whether the model is learning anything, on this
   dataset. This is the evaluation-integrity bug from `AUDIT.md` §3,
   item 7, made concrete: it isn't a marginal effect, it's the difference
   between "the model looks broken" and "the model works."
2. **Class-aware R@1 == R@5 == R@10 exactly (62.3%).** This is not a bug in
   the metric — it's a direct, verified consequence of the caption
   duplication: there are only **10 unique caption strings** in the entire
   dataset (one per class), so the "text gallery" for image-to-text
   retrieval has only 10 distinct embedding vectors, each repeated ~100
   times verbatim (confirmed: `torch.unique(text_feats, dim=0).shape[0] ==
   10` for the 10 class prompts). Once the top-ranked class's ~100
   duplicate columns are ahead of every other class in the similarity
   sort, expanding K from 1 to 10 only pulls in more copies of that same
   (already-counted) class — it can't reach a different class until K
   exceeds the ~100 duplicates per class. So R@1/R@5/R@10 collapsing to
   the same number is the mathematically correct output of a
   well-defined metric, given this dataset's caption redundancy — not
   a bug, but a real limitation of the *dataset* (template captions) that
   the metric is faithfully reporting, and the reason a template-based
   CIFAR-10 caption dataset makes K>1 retrieval metrics uninformative.

## 4. Honest comparison: does ViT-from-scratch beat a good CNN?

**No — the CNN baseline wins, decisively, and with far less training.**

| Model | Params | Epochs | Top-1 | Top-5 |
|---|---|---|---|---|
| ViT-Tiny (this repo) | 2.69M | 100 | 83.68% | 98.86% |
| CNN baseline (this repo) | 1.65M | **12** | **87.96%** | **99.37%** |

The smaller CNN baseline beats ViT-Tiny by **+4.28 points top-1**
while training for **8x fewer epochs**. This is the well-known,
expected result: Vision Transformers lack
convolution's built-in inductive biases (locality, translation
equivariance), so on a small dataset like CIFAR-10 trained fully from
scratch (no ImageNet pretraining), a ViT needs substantially more data or
training signal to reach parity with a CNN of similar or smaller size — and here,
given the same epoch budget the CNN would have an even larger lead, since
its 12-epoch curve was still improving (val loss still dropping epoch to
epoch; see `reports/cnn_baseline_train_log.txt`, reproduced via the command
in `RESULTS.md`) and had far from saturated. This is a legitimate finding about data/compute
efficiency, not a flaw in the ViT implementation — see `AUDIT.md` §4
for the correctness checks (gradcheck, patch-embedding equivalence,
DropPath expectation, position embedding placement) that confirm the ViT
itself is implemented correctly; it's simply the less
sample-efficient architecture on this dataset, which is the standard,
published finding for ViT trained from scratch on small datasets
(Dosovitskiy et al. 2021 make the same point about ImageNet-scale data
being necessary for ViT to close the gap with CNNs).

## 5. Known limitations

- **CLIP captions are template-based, not natural**: every image of a
  class shares the identical caption `"a photo of a {class}"`. This makes
  strict index-diagonal retrieval recall an unreliable metric here (see
  `AUDIT.md` §3, item 7) — use class-aware recall. It also means the
  InfoNCE contrastive loss treats same-class images in a batch as hard
  negatives of each other, which is a known, inherent property of
  diagonal-target InfoNCE with non-unique captions, not something this
  codebase can silently fix without deviating from the reference loss
  formulation.
- **`vit_best.pt`'s exact training seed is unknown** (trained before the
  reproducibility fix in this audit). Its reported number is real (produced
  by an actual 100-epoch run) but re-running the same command today will
  land close to, not exactly at, 83.68%.
- **CPU-only training environment**: this repository was developed and
  evaluated without GPU access. Where that materially changed what could be
  trained end-to-end in this audit (CLIP), it is disclosed explicitly
  rather than papered over with unverified numbers.
