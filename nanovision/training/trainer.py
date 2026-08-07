"""
Training loops for ViT classification and CLIP contrastive learning.

Two trainers:
  ViTTrainer  — trains ViT for image classification on CIFAR-10.
  CLIPTrainer — trains CLIP with symmetric InfoNCE loss.

Both trainers share the same design:
  - Epoch-based training (standard for classification / CLIP on CIFAR-10)
  - Cosine LR schedule with linear warmup
  - Gradient norm clipping
  - Optional mixup (ViT only)
  - Optional AMP (auto-disabled on CPU)
  - TensorBoard logging (graceful fallback if unavailable)
  - Best-model + periodic checkpointing
  - Structured console output (ASCII-safe for Windows cp1252)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nanovision.model.config import TrainConfig
from nanovision.model.vit import ViT
from nanovision.model.clip import CLIP
from nanovision.training.scheduler import get_epoch_lr, set_lr
from nanovision.data.augmentation import mixup_batch
from nanovision.utils.checkpoint import save_checkpoint, load_checkpoint
from nanovision.utils.metrics import accuracy, retrieval_recall, retrieval_recall_class_aware


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore  # noqa
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _try_tensorboard(log_dir: str):
    """Return a SummaryWriter or None if TensorBoard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception as e:
        print(f"[warn] TensorBoard unavailable ({type(e).__name__}) - skipping TB logging")
        return None


# ---------------------------------------------------------------------------
# ViT Trainer
# ---------------------------------------------------------------------------

class ViTTrainer:
    """
    Training engine for ViT image classification.

    Trains for `cfg.max_epochs` epochs using:
      - AdamW with weight decay (via model.configure_optimizer)
      - Cosine LR + linear warmup
      - Optional mixup data augmentation
      - Label smoothing cross-entropy
      - Gradient clipping
    """

    def __init__(
        self,
        model:       ViT,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        cfg:         TrainConfig,
        device:      Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.device       = device or _get_device()

        self.model.to(self.device)

        self.optimizer = model.configure_optimizer(
            weight_decay=cfg.weight_decay,
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )

        self.use_amp = cfg.use_amp and self.device.type in ("cuda", "xpu")
        self.scaler  = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.use_amp,
        )

        self.writer       = _try_tensorboard(cfg.log_dir) if cfg.use_tensorboard else None
        self.epoch        = 0
        self.best_acc     = 0.0

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch. Returns (mean_loss, top1_accuracy)."""
        self.model.train()
        n_classes = self.model.config.n_classes

        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Mixup
            if self.cfg.mixup_alpha > 0.0:
                images, soft_labels = mixup_batch(
                    images, labels, n_classes, alpha=self.cfg.mixup_alpha
                )
                soft_labels = soft_labels.to(self.device)
            else:
                soft_labels = None

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32,
                enabled=self.use_amp,
            ):
                logits = self.model(images)   # (B, n_classes)

                if soft_labels is not None:
                    # Mixup: use KL loss against soft targets
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(soft_labels * log_probs).sum(dim=-1).mean()
                else:
                    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            B            = labels.shape[0]
            total_loss  += loss.item() * B
            total_samples += B

            if soft_labels is None:
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()

        mean_loss = total_loss / total_samples
        acc = total_correct / total_samples if soft_labels is None else float("nan")
        return mean_loss, acc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        """Run evaluation. Returns (loss, top1_acc, top5_acc)."""
        self.model.eval()
        total_loss, top1, top5, n = 0.0, 0, 0, 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            B      = labels.shape[0]

            logits = self.model(images)
            loss   = F.cross_entropy(logits, labels)

            total_loss += loss.item() * B
            top1       += accuracy(logits, labels, topk=1)
            top5       += accuracy(logits, labels, topk=5)
            n          += B

        return total_loss / n, top1 / n * 100, top5 / n * 100

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from:
            self.epoch = load_checkpoint(resume_from, self.model, self.optimizer)
            print(f"Resumed from epoch {self.epoch} ({resume_from})")

        n_params = self.model.count_params(non_embedding=True)
        print("=" * 70)
        print(self.model)
        print(f"  Params (non-emb) : {n_params / 1e6:.2f} M")
        print(f"  Epochs           : {self.cfg.max_epochs}")
        print(f"  Batch size       : {self.cfg.batch_size}")
        print(f"  Device           : {self.device}")
        print("=" * 70)

        for epoch in range(self.epoch, self.cfg.max_epochs):
            # Update LR
            lr = get_epoch_lr(
                epoch, self.cfg.warmup_epochs, self.cfg.max_epochs,
                self.cfg.max_lr, self.cfg.min_lr,
            )
            set_lr(self.optimizer, lr)

            t0 = time.perf_counter()
            train_loss, train_acc = self._train_epoch()
            dt = time.perf_counter() - t0

            # Evaluate every eval_interval epochs (always on last epoch)
            if (epoch + 1) % self.cfg.eval_interval == 0 or epoch == self.cfg.max_epochs - 1:
                val_loss, top1, top5 = self._evaluate()

                print(
                    f"epoch {epoch + 1:>4}/{self.cfg.max_epochs} | "
                    f"lr {lr:.2e} | "
                    f"train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"top1 {top1:.2f}% | top5 {top5:.2f}% | "
                    f"{dt:.1f}s"
                )

                if self.writer:
                    step = epoch + 1
                    self.writer.add_scalar("train/loss",   train_loss, step)
                    self.writer.add_scalar("val/loss",     val_loss,   step)
                    self.writer.add_scalar("val/top1_acc", top1,       step)
                    self.writer.add_scalar("val/top5_acc", top5,       step)
                    self.writer.add_scalar("train/lr",     lr,         step)

                if top1 > self.best_acc:
                    self.best_acc = top1
                    save_checkpoint(
                        path=Path(self.cfg.checkpoint_dir) / "vit_best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        metric=top1,
                        config=self.model.config,
                        train_config=self.cfg,
                        model_type="vit",
                    )
                    print(f"  >> [ckpt] Best model saved (top1={top1:.2f}%)")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    path=Path(self.cfg.checkpoint_dir) / f"vit_epoch_{epoch + 1:04d}.pt",
                    model=self.model, optimizer=self.optimizer,
                    epoch=epoch + 1, metric=self.best_acc,
                    config=self.model.config, train_config=self.cfg,
                    model_type="vit",
                )

        print("=" * 70)
        print(f"ViT training complete. Best top-1 accuracy: {self.best_acc:.2f}%")
        if self.writer:
            self.writer.close()


# ---------------------------------------------------------------------------
# CLIP Trainer
# ---------------------------------------------------------------------------

class CLIPTrainer:
    """
    Training engine for CLIP contrastive learning on CIFAR-10.

    Trains with symmetric InfoNCE loss.  Evaluates with zero-shot top-1
    accuracy on the test set and image-text retrieval recall.
    """

    def __init__(
        self,
        model:       CLIP,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        cfg:         TrainConfig,
        device:      Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.device       = device or _get_device()

        self.model.to(self.device)

        self.optimizer = model.configure_optimizer(
            weight_decay=cfg.weight_decay,
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )

        self.use_amp = cfg.use_amp and self.device.type in ("cuda", "xpu")
        self.scaler  = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.use_amp,
        )

        self.writer       = _try_tensorboard(cfg.log_dir) if cfg.use_tensorboard else None
        self.epoch        = 0
        self.best_zs_acc  = 0.0

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        """Returns (mean_loss, mean_temperature)."""
        self.model.train()
        total_loss, total_temp, n_batches = 0.0, 0.0, 0

        for images, tokens, _ in self.train_loader:
            images = images.to(self.device)
            tokens = tokens.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32,
                enabled=self.use_amp,
            ):
                loss, _, _ = self.model(images, tokens)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_temp += self.model.log_temp.exp().item()
            n_batches  += 1

        return total_loss / n_batches, total_temp / n_batches

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_zero_shot(self) -> float:
        """Zero-shot classification accuracy on test set."""
        self.model.eval()
        correct, total = 0, 0

        for images, _, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model.zero_shot_classify(images)   # (B, 10)
            preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]

        return correct / total * 100.0

    @torch.no_grad()
    def _evaluate_retrieval(self, n_samples: int = 1000) -> dict[str, float]:
        """
        Image-to-text and text-to-image retrieval R@1, R@5, R@10
        on a subset of the test set.

        Reports TWO ground-truth conventions:
          - strict index-diagonal recall (query i's only correct answer is
            item i) — this is what most CLIP papers report, valid when every
            caption is unique.
          - class-aware recall (any top-K item sharing the query's class
            counts as a hit) — the metric that is actually meaningful here,
            since NanoVision's captions are one template per CIFAR-10 class,
            so many images/texts in any eval subset legitimately share an
            identical caption. Strict recall silently under-counts correct
            retrievals in that regime; see
            `nanovision.utils.metrics.retrieval_recall_class_aware`.
        """
        self.model.eval()
        all_img_feats  = []
        all_text_feats = []
        all_labels     = []
        count = 0

        for images, tokens, labels in self.test_loader:
            if count >= n_samples:
                break
            images = images.to(self.device)
            tokens = tokens.to(self.device)

            all_img_feats.append(self.model.encode_image(images).cpu())
            all_text_feats.append(self.model.encode_text(tokens).cpu())
            all_labels.append(labels)
            count += images.shape[0]

        img_feats  = torch.cat(all_img_feats,  dim=0)[:n_samples]
        text_feats = torch.cat(all_text_feats, dim=0)[:n_samples]
        labels     = torch.cat(all_labels,     dim=0)[:n_samples]

        # Similarity matrix
        sim = img_feats @ text_feats.t()   # (N, N)

        results = {}
        for k in (1, 5, 10):
            results[f"i2t_R@{k}"]            = retrieval_recall(sim, k=k)
            results[f"t2i_R@{k}"]            = retrieval_recall(sim.t(), k=k)
            results[f"i2t_R@{k}_class"] = retrieval_recall_class_aware(sim, labels, labels, k=k)
            results[f"t2i_R@{k}_class"] = retrieval_recall_class_aware(sim.t(), labels, labels, k=k)

        return results

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from:
            self.epoch = load_checkpoint(resume_from, self.model, self.optimizer)
            print(f"Resumed from epoch {self.epoch} ({resume_from})")

        counts = self.model.count_params()
        print("=" * 70)
        print(self.model)
        print(f"  Total params     : {counts['total'] / 1e6:.2f} M")
        print(f"  Epochs           : {self.cfg.max_epochs}")
        print(f"  Batch size       : {self.cfg.batch_size}")
        print(f"  Device           : {self.device}")
        print("=" * 70)

        for epoch in range(self.epoch, self.cfg.max_epochs):
            lr = get_epoch_lr(
                epoch, self.cfg.warmup_epochs, self.cfg.max_epochs,
                self.cfg.max_lr, self.cfg.min_lr,
            )
            set_lr(self.optimizer, lr)

            t0 = time.perf_counter()
            train_loss, mean_temp = self._train_epoch()
            dt = time.perf_counter() - t0

            if (epoch + 1) % self.cfg.eval_interval == 0 or epoch == self.cfg.max_epochs - 1:
                zs_acc   = self._evaluate_zero_shot()
                retrieval = self._evaluate_retrieval()

                print(
                    f"epoch {epoch + 1:>4}/{self.cfg.max_epochs} | "
                    f"lr {lr:.2e} | loss {train_loss:.4f} | "
                    f"temp {mean_temp:.3f} | zs_acc {zs_acc:.2f}% | "
                    f"i2t_R@1(strict) {retrieval['i2t_R@1']:.2f}% | "
                    f"i2t_R@1(class) {retrieval['i2t_R@1_class']:.2f}% | "
                    f"{dt:.1f}s"
                )

                if self.writer:
                    step = epoch + 1
                    self.writer.add_scalar("train/loss",        train_loss,     step)
                    self.writer.add_scalar("train/temperature", mean_temp,      step)
                    self.writer.add_scalar("train/lr",          lr,             step)
                    self.writer.add_scalar("eval/zs_acc",       zs_acc,         step)
                    for k, v in retrieval.items():
                        self.writer.add_scalar(f"eval/{k}", v, step)

                if zs_acc > self.best_zs_acc:
                    self.best_zs_acc = zs_acc
                    save_checkpoint(
                        path=Path(self.cfg.checkpoint_dir) / "clip_best.pt",
                        model=self.model, optimizer=self.optimizer,
                        epoch=epoch + 1, metric=zs_acc,
                        config=self.model.config, train_config=self.cfg,
                        model_type="clip",
                    )
                    print(f"  >> [ckpt] Best CLIP saved (zs_acc={zs_acc:.2f}%)")

        print("=" * 70)
        print(f"CLIP training complete. Best zero-shot accuracy: {self.best_zs_acc:.2f}%")
        if self.writer:
            self.writer.close()
