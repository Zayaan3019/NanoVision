# Reported numbers and how to regenerate them

Every number quoted in `README.md`, `MODEL_CARD.md`, and `AUDIT.md` is
reproducible by exactly one of the commands below. Nothing in this file was
computed by hand or copy-pasted from an untracked run.

```bash
python prepare_data.py --seed 42        # not seed-sensitive, listed for completeness
```

## ViT-Tiny classification

Training (already run; checkpoint at `checkpoints/vit_best.pt`, trained
before the seeding fix in this audit — see `MODEL_CARD.md` §5):

```bash
python train_vit.py --seed 42
```

Evaluation (regenerate the reported 83.68% top-1 / 98.86% top-5 / per-class table):

```bash
python evaluate.py --vit_ckpt checkpoints/vit_best.pt --batch_size 256
```

## CNN baseline

Training:

```bash
python train_baseline_cnn.py --widths 32 64 128 256 --max_epochs 12 --batch_size 256 --seed 42
```

Result (final epoch, `checkpoints/cnn_baseline_best.pt`):

```
epoch  12/12 | lr 2.54e-03 | train_loss 0.7988 | train_acc 89.47% | val_loss 0.4261 | top1 87.96% | top5 99.37%
CNN baseline training complete. Best top-1 accuracy: 87.96%
```

## CLIP

`CLIPConfig.tiny()` (the reference architecture) measures ~28 min/epoch on
the CPU this repo was developed on — training it to completion end-to-end
inside this audit's time budget was not feasible. `CLIPConfig.micro()`
(~0.8M params, same architecture family: Pre-LN ViT vision encoder +
bidirectional text transformer + symmetric InfoNCE) was trained instead, to
completion, so every CLIP number reported is real:

```bash
python train_clip.py --preset micro --max_epochs 15 --batch_size 128 --seed 42
```

Result (final epoch, `checkpoints/clip_best.pt`):

```
epoch   15/15 | lr 2.20e-05 | loss 3.7521 | temp 13.291 | zs_acc 64.22% | i2t_R@1(strict) 0.30% | i2t_R@1(class) 62.30%
CLIP training complete. Best zero-shot accuracy: 64.22%
```

Evaluation (zero-shot + retrieval, both strict and class-aware recall):

```bash
python evaluate.py --clip_ckpt checkpoints/clip_best.pt --n_retrieval 1000
```

```
Zero-Shot Top-1 Accuracy  : 64.22%
Zero-Shot Top-5 Accuracy  : 96.87%

         strict (index)    class-aware
  R@1              0.8%          62.3%
  R@5              3.6%          62.3%
  R@10             6.8%          62.3%
```

Strict recall is scored **tie-aware**: identical captions give identical scores,
and which of the tied items `topk` returns is an implementation detail that
changed between PyTorch versions (the same checkpoint gave strict R@1 0.3% on
the original run and 0.7% on PyTorch 2.13). A query whose correct item is tied
now counts as the probability it would make the top K under uniformly random
tie-breaking, so the number is deterministic; with no ties it equals plain
`topk` recall. Class-aware recall and zero-shot accuracy were unaffected.

(R@1==R@5==R@10 under class-aware scoring is explained in `MODEL_CARD.md` §3
— it's a verified consequence of only 10 unique captions existing in the
whole dataset, not a metric bug.)

To reproduce the full `tiny` preset on a GPU:

```bash
python train_clip.py --preset tiny --device cuda --max_epochs 50 --batch_size 128 --seed 42
python evaluate.py --clip_ckpt checkpoints/clip_best.pt
```

## Test suite

```bash
python -m pytest tests/ -v
python scripts/classify_tests.py --verbose
```
