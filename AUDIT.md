# NanoVision Audit — CLIP correctness and evaluation integrity

A line-by-line audit of the test suite, the CLIP loss, evaluation integrity,
ViT correctness and reproducibility. It reflects the state of the repository
AFTER the fixes described below were applied (all fixes are in this same
change set).

---

## §1 — What the tests actually assert

**Question:** do the tests check numerical values and gradients, or only
tensor shapes?

**Finding: before this audit, very few tests touched gradients, and none
checked gradient values. The suite has been strengthened so that it does.**

Before this audit, the suite had 109 tests. A manual read of every test
function found:

| Category (manual read, pre-audit) | Count |
|---|---|
| Shape-only (`.shape`/`len` and nothing else) | ~19 |
| Numeric/value (allclose, closed-form, statistical) | ~62 |
| Gradient-existence only (`p.grad is not None`) | 3 |
| Structural/behavioral (type checks, `pytest.raises`, membership) | ~25 |

Only **3 of 109 tests touched gradients at all**, and none of those checked
gradient *values* — they only asserted `.grad is not None` (the backward
graph is connected), never that the gradient is numerically correct. No
test used `torch.autograd.gradcheck` or any finite-difference / closed-form
reference. So gradient coverage existed only in the weakest sense (existence, not
correctness) and for under 3% of the suite.

**Fix applied**: the suite was strengthened so that gradients are checked
for correctness, not just existence:

- Added `torch.autograd.gradcheck` tests (finite-difference vs. autograd,
  float64) for `SelfAttention`, `PatchEmbedding`, a full `Block`, and the
  CLIP contrastive loss (`_contrastive_loss`) — see `TestGradcheck` in
  `tests/test_vit.py` and `test_contrastive_loss_gradcheck` in
  `tests/test_clip.py`.
- Added closed-form numeric tests: exact InfoNCE loss value on a
  hand-constructed identity similarity matrix, `Conv2d` patch embedding
  proven algebraically equivalent to unfold+linear, DropPath's rescaling
  proven expectation-preserving, Mixup's per-sample lambda proven identical
  between pixel-space and label-space mixing, reflect-padding proven
  pixel-exact against a numpy reference.
- Added whole new, previously nonexistent test files for the augmentation
  pipeline (`tests/test_augmentation.py`, 14 tests) and for the CNN
  baseline (`tests/test_cnn.py`, 8 tests), plus preset-validity tests for
  the new `CLIPConfig.micro()` architecture.
- Fixed one existing test whose name overpromised: `test_vit.py::
  TestViTForward::test_train_eval_differ_with_dropout` was named to prove
  dropout makes two training-mode forward passes differ, but its body only
  re-checked eval-mode determinism. It now actually asserts
  `not torch.allclose(y1, y2)` under training-mode dropout.

**Reproducible classification**: `scripts/classify_tests.py` walks the AST
of every test function and buckets it by what its `assert` statements
actually check (regex evidence over each assertion, not vibes). Run it
yourself:

```bash
python scripts/classify_tests.py
```

Current output (146 tests total, after the fixes above):

```
Category        Count       %
------------------------------
shape-only         16   11.0%
gradient           12    8.2%
numeric           104   71.2%
structural         12    8.2%
other               2    1.4%
no-assertions       0    0.0%

130/146 tests (89.0%) assert something beyond tensor shape.
Of those, 12 directly touch gradients (.grad / gradcheck / requires_grad).
```

The remaining "other" (2 tests) are tuple-equality checks on non-tensor
shape-like attributes (`PIL.Image.size`, `Conv2d.stride`) that the `.shape`
regex doesn't match — inspect them with `--verbose`; they are not weak
tests, just outside the regex's pattern list.

**Summary**: the large majority of tests assert numeric values or
behaviour, not just shapes, and a dedicated subset uses finite-difference
gradient checking — verifiable by running the classifier above.

---

## §2 — CLIP contrastive loss

**2. Symmetric InfoNCE?** Yes, verified in `nanovision/model/clip.py::CLIP._contrastive_loss`:
`loss = 0.5 * (CE(logits_i2t, diag) + CE(logits_t2i, diag))`, and
`logits_t2i = logits_i2t.t()` exactly (tested: `test_logits_symmetric`).
Confirmed with a closed-form example (`TestInfoNCEClosedForm::
test_loss_matches_hand_computed_value`): with `img_feats == text_feats ==
eye(B)` and `temp=1`, the loss equals `-log(e / (e + B - 1))` to 1e-5,
exactly what symmetric InfoNCE predicts.

**3. Learnable log-temperature, clamped?** Yes — `self.log_temp =
nn.Parameter(...)`, clamped every forward pass via `log_temp.clamp(0,
4.605)` (`4.605 = log(100)`, the reference CLIP clamp). Tested at both
clamp boundaries (`test_log_temp_clamped_in_loss`,
`test_log_temp_clamped_to_zero`) — verified the *effective* temperature
saturates at 100 / 1 respectively, not just that the loss stays finite.

**4. Duplicate-positive-pair handling.** Added
`TestInfoNCEClosedForm::test_duplicate_positive_pair_is_stable` and
`::test_two_identical_images_in_batch_end_to_end`: construct a batch where
two samples share the exact same image embedding. Result: loss stays
finite, gradients stay finite, and (as expected from the reference
diagonal-target InfoNCE formulation — this is not NanoVision-specific,
it's how CLIP's loss is defined in the paper) the duplicate becomes a hard
negative for its twin's row. This is a known, inherent limitation of
diagonal-target InfoNCE, not a bug — see the §3 finding below for
where it actually bites in this repo (retrieval evaluation).

**5. L2-normalised before the similarity matrix?** Yes —
`encode_image`/`encode_text` both end in `F.normalize(proj, dim=-1)` before
any similarity is computed. Verified numerically (not just by reading the
code): `test_image_embeddings_l2_normalised` / `test_text_embeddings_l2_normalised`
check `norm(dim=-1) == 1` to 1e-5.

---

## §3 — Evaluation integrity

**6. Zero-shot prompt construction — tuned on test set?**

No evidence found. The zero-shot template is a single hardcoded constant,
`CLIP.PROMPT_TEMPLATE = "a photo of a {}"`, matching the original CLIP
paper's convention (cited in the docstring) — not derived from any sweep.
Grepped the codebase for any prompt-selection/grid-search logic
(`sweep|grid_search|best_prompt|template_scores` etc.) — none exists. The
*training-time* multi-template list (`CLIP_PROMPT_TEMPLATES`, 8 templates
in `nanovision/data/dataset.py`) is caption-diversity data augmentation
applied at random per training sample; it is never used for zero-shot
evaluation (`multi_template=False` is hardcoded for all eval/test
dataloaders), so it cannot leak into the reported zero-shot number.
**Conclusion: no prompt-engineering leakage found; nothing to retune.**

**7. Retrieval R@1/R@5/R@10 — computation and gallery contamination.**

*Contamination*: clean. `retrieve.py`, `evaluate.py`, and
`CLIPTrainer._evaluate_retrieval` all build the retrieval gallery
exclusively from `split="test"` (CIFAR-10's canonical 10,000-image test
split, physically separate files from the 5 training batches — see
`nanovision/data/dataset.py::load_cifar10`). No code path mixes train and
test images into the same gallery.

*Computation — a real bug found and fixed*: **`retrieve.py` and
`evaluate.py` were calling `CIFAR10CLIPDataset(images_np, labels_np,
transform=...)` positionally**, but the class's actual constructor
signature was `(data_dir, split, transform, ...)` — the first two
positional arguments are `data_dir` and `split`, not `images`/`labels`.
This meant `python evaluate.py --clip_ckpt ...` and `python retrieve.py
--eval` **crashed** the moment they hit the retrieval section (confirmed by
reproducing the crash before the fix). This bears directly on §5,
item 13 (every reported number must be regenerable) — the
retrieval numbers in the original README could not have been produced by
running the documented commands as written. Fixed by giving
`CIFAR10CLIPDataset` a proper `images=`/`labels=` in-memory constructor
path and updating both call sites to use it explicitly by keyword.

*Computation — a second, more serious integrity bug found and fixed*:
`EmbeddingIndex.recall_at_k()` (and the trainer's inline copy of the same
logic) assumed the similarity matrix's **diagonal is the only correct
answer** for each query — standard practice for retrieval benchmarks where
every caption is unique (e.g. Flickr30k, COCO). **NanoVision's captions are
not unique**: a CIFAR-10 image's caption is `"a photo of a {class}"`,
templated per class, so in any evaluation subset dozens of images
legitimately share the exact same caption (and therefore the exact same
text embedding). Under strict diagonal scoring, a model that retrieves a
*different, equally-correct, same-class* image is scored as **wrong**.
Measured impact on the CLIP-Micro checkpoint trained as part of this audit
(15 epochs, real training run — see `MODEL_CARD.md` §3 / `RESULTS.md`):

```
         strict (index)    class-aware
  R@1              0.3%          62.3%
  R@5              2.3%          62.3%
  R@10             5.6%          62.3%
```

Strict recall stayed pinned near-random (0.2-0.7%) for **all 15 epochs**,
even as training loss dropped 4.63->3.75 and zero-shot accuracy climbed
31.8%->64.2% — i.e. strict recall would have told a reader the model wasn't
learning anything, when class-aware recall (which tracks zero-shot accuracy
almost exactly, as it should — both measure the same representation
quality) shows it clearly was. This is a >60-point gap purely from the
ground-truth convention, not model quality, and it is the difference
between "this model looks broken" and "this model works."

A second, subtler thing this real run surfaced: class-aware **R@1 == R@5 ==
R@10 exactly (62.3%)**. This is not a metric bug — verified directly:
`torch.unique(text_feats, dim=0).shape[0] == 10` for the 10 class prompts,
i.e. the entire dataset has only 10 distinct caption embeddings, each
repeated ~100 times bit-for-bit identically. Once the correct class's ~100
duplicate columns occupy the top of the similarity ranking, expanding K
from 1 to 10 only pulls in more copies of that SAME already-counted class —
it structurally cannot reach a different class until K exceeds ~100. So
the metric is behaving exactly as it mathematically should; the dataset's
caption redundancy is what makes K>1 retrieval metrics uninformative here.
Documented in `MODEL_CARD.md` §3 rather than presented as a clean number
without explanation.

**Fix**: added `retrieval_recall_class_aware()` (a hit is any top-K item
sharing the query's class, not strictly the diagonal index) to
`nanovision/utils/metrics.py`, wired it through `EmbeddingIndex.recall_at_k
(class_aware=True)`, `CLIPTrainer._evaluate_retrieval`, `evaluate.py`, and
`retrieve.py --eval`, which now print both conventions side by side with an
explanation. Strict recall is kept (not removed) because it is still a
valid, standard, conservative lower bound — but class-aware recall is the
number that actually reflects retrieval quality in this dataset and is
labeled as such everywhere it's printed.

**8. Actual achieved numbers.** See `RESULTS.md` (reproducible via one
command each, per §5). Summary:

- **ViT-Tiny**, 100 epochs, CIFAR-10: **83.68% top-1 / 98.86% top-5** test
  accuracy (`checkpoints/vit_best.pt`, reproduced via `python evaluate.py
  --vit_ckpt checkpoints/vit_best.pt`). This is below the README's original
  claim of "~88-90%" — the original number was aspirational/unverified;
  it has been corrected.
- **CLIP**: no fully-trained checkpoint existed anywhere in the repository
  before this audit (`checkpoints/` had no `clip_best.pt` at all — only a
  1-epoch smoke-test artifact under `checkpoints_smoke/`, `zs_acc=33.93%`).
  The README's CLIP numbers ("zero-shot ~75-80%, R@1 ~70-75%") were
  **never produced by an actual training run**, so they were removed
  (§5, item 13: every reported number must be regenerable). Trained a real `CLIPConfig.micro()` (0.83M params) for 15
  epochs as part of this audit (CPU compute budget — the `tiny` preset
  measured ~28 min/epoch on this machine, infeasible to train to
  completion here; see §5 below). **Result: 64.22% zero-shot
  top-1 / 96.87% top-5; R@1 0.3% strict / 62.3% class-aware.** Full
  breakdown and the R@1==R@5==R@10 explanation in `MODEL_CARD.md` §3.
- **CNN baseline**: 1.65M params (matched to ViT-Tiny), 12 epochs:
  **87.96% top-1 / 99.37% top-5** — beats ViT-Tiny's 83.68%/98.86% despite
  8x fewer epochs. See §5, item 14 / `MODEL_CARD.md` §4.

---

## §4 — ViT correctness

**9. Patch embedding.** `PatchEmbedding` uses
`nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)`
— stride exactly equals patch size (verified:
`TestPatchEmbeddingEquivalence::test_stride_equals_patch_size`, reads
`.stride`/`.kernel_size` directly off the layer, not just re-deriving them
from config). Sequence length: `n_patches + 1 = (img_size/patch_size)^2 + 1`
= 64 + 1 = 65 for the default 32x32/patch-4 config, and the CLS token is
prepended before position embeddings are added (verified by shape AND by
the closed-form equivalence test below).

New: `test_conv2d_equals_manual_unfold_linear` proves, numerically (not by
assumption), the docstring's claim that "Conv2d(kernel=stride=P) is
mathematically equivalent to flattening each patch and passing it through a
shared linear layer" — computed via `F.unfold` + explicit matmul against
the conv weight reshaped to `(d_model, C*P*P)`, `atol=1e-5`.

**10. DropPath.** Confirmed identity in eval mode
(`drop_path(x, p, training=False) == x` exactly, any `p`) and identity at
`p=0` regardless of mode. New: `TestDropPathExpectation` proves the
rescaling is *expectation-preserving* — `E[drop_path(x)] ≈ x` empirically
over 40,000 samples (within 2%), which is what "randomly zero + rescale by
1/keep_prob" is supposed to guarantee and is the property that actually
matters for training stability; a wrong rescaling constant (e.g. dividing
by `drop_prob` instead of `keep_prob`) would fail this test but pass the
existing identity-only tests.

**11. Mixup + KL loss — per-sample lambda.**

**Bug found and fixed**: `mixup_batch()` sampled **one lambda for the
entire batch** (`lam = np.random.beta(alpha, alpha)`, a scalar), not one
lambda per sample. This is what the original Zhang et al. mixup reference
implementation also does, so it isn't "wrong" relative to the paper, but it
is weaker regularization than the per-sample variant used by modern
training recipes (timm/DeiT/ConvNeXt). **Fixed**: `mixup_batch()` now samples
`lam[i] ~ Beta(alpha, alpha)` independently for every sample `i`, and uses
`lam[i]` for BOTH that sample's image mix and its label mix — proven
numerically identical per-sample in
`TestMixupPerSample::test_lambda_identical_between_image_and_label_per_sample`
(reconstructs the implied lambda from the label distribution and from the
pixel value independently for every sample in the batch and checks
self-consistency), and proven to actually vary across the batch (not
secretly still one shared value) in `test_lambda_varies_across_batch`.

**KL target is a proper distribution**: `test_soft_labels_are_proper_distributions`
verifies every row of the mixed one-hot label sums to 1.0 (±1e-6) and has
no negative or >1 entries — a valid probability distribution, so
`-(soft_labels * log_softmax(logits)).sum(-1).mean()` in `ViTTrainer._train_epoch`
is a well-defined KL-style (cross-entropy against a soft target) loss.

**12. Positional embedding.** Learnable (`nn.Parameter`), added to the
sequence immediately after the CLS token is prepended and before the first
transformer block (`x = self.pos_drop(x + self.pos_embed)` in
`ViT.forward_features`, executed before the `for block in self.blocks`
loop). Correctly sized: `pos_embed.shape == (1, seq_len, d_model)` with
`seq_len = n_patches + 1`, verified directly against the derived formula
`(img_size // patch_size) ** 2 + 1` in
`test_sequence_length_includes_cls`, not just against the config's own
(possibly-also-wrong) `seq_len` property.

---

## §5 — Reproducibility

**13. Seeding, deterministic flags, one-command reproduction.**

**Before this audit: none of `train_vit.py`, `train_clip.py`,
`evaluate.py`, or `retrieve.py` seeded anything.** No `torch.manual_seed`,
no `--seed` flag, nothing — every run was silently non-reproducible, and
the existing `checkpoints/vit_best.pt` (83.68% top-1) has no recorded seed.

**Fixed**: added `nanovision/utils/seed.py::set_seed(seed, deterministic)`
(seeds `random`, `numpy`, `torch` CPU+CUDA, optional
`torch.use_deterministic_algorithms`) and a `--seed` (default 42) /
`--deterministic` flag to every entry point (`train_vit.py`,
`train_clip.py`, `evaluate.py`, `retrieve.py`, `train_baseline_cnn.py`).
DataLoader workers (when `num_workers > 0`) are seeded via
`worker_init_fn=seed_worker` so multi-process augmentation is also
reproducible. The pre-existing `vit_best.pt`'s exact seed is unrecoverable
and that is disclosed rather than papered over — every number produced
*after* this fix has a documented, reproducible seed.

**One command per reported number** — see `RESULTS.md` for the exact
commands. None of the numbers in that file were computed by hand or copied
from an untracked run; each line names the exact command that produces it.

**14. Model card / baseline comparison.** Full details in `MODEL_CARD.md`.
Headline: a smaller (1.65M vs. 2.69M params) CNN baseline beats
ViT-Tiny **87.96% vs. 83.68% top-1**, training for **12 epochs vs. 100**.
This is the expected, legitimate finding — ViT lacks convolution's
built-in locality/translation-equivariance bias, so trained from scratch
on a small dataset it needs more signal to match a CNN of similar or smaller size;
this is consistent with Dosovitskiy et al. (2021)'s own finding that ViT
needs large-scale pretraining data to close the gap with CNNs. Reported
honestly rather than omitted or reframed.

---

## Definition of done — status

- [x] Test split (numerics vs shapes) reported, reproducible via `scripts/classify_tests.py`
- [x] Symmetric InfoNCE with clamped temperature verified (closed-form + gradcheck)
- [x] Zero-shot prompts proven not to have been tuned on test (no sweep logic found; documented)
- [x] Retrieval gallery proven free of train contamination; evaluation-integrity bug (diagonal-only ground truth vs. duplicate captions) found and fixed with a class-aware metric
- [x] Every reported number reproducible by one command — see `RESULTS.md`
