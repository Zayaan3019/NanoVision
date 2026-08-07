"""
Image augmentation pipeline implemented from scratch using PIL only.

No torchvision dependency — every transform is built on top of PIL.Image
primitives so the codebase remains self-contained.

Transforms are composable in the same way as torchvision.transforms:

    aug = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    tensor = aug(pil_image)

CIFAR-10 standard augmentation:
    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + ColorJitter + Normalize
    Val:   ToTensor + Normalize
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Transform:
    """Abstract base for all transforms."""

    def __call__(self, img: Image.Image) -> Image.Image | torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------

class Resize:
    """Resize to (height, width) using bilinear interpolation."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR)

    def __repr__(self) -> str:
        return f"Resize(size={self.size})"


class CenterCrop:
    """Crop the center of the image to `size x size`."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        left  = (w - self.size) // 2
        top   = (h - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class RandomCrop:
    """
    Randomly crop `size x size` pixels from the image.

    If `padding > 0`, the image is first padded then randomly cropped —
    standard CIFAR-10 augmentation. `padding_mode="reflect"` mirrors each
    border strip INCLUDING the edge pixel (OpenCV's BORDER_REFLECT /
    numpy's `mode="symmetric"`), not numpy's edge-excluding `mode="reflect"`
    (a.k.a. BORDER_REFLECT_101) — both are standard conventions, this one
    just avoids ever duplicating a padded pixel value at the seam.
    """

    def __init__(self, size: int, padding: int = 0, padding_mode: str = "reflect") -> None:
        self.size         = size
        self.padding      = padding
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.padding > 0:
            p = self.padding
            w, h   = img.size
            # PIL padding: left, top, right, bottom
            padded = Image.new(img.mode, (w + 2 * p, h + 2 * p))
            padded.paste(img, (p, p))

            if self.padding_mode == "reflect":
                # Left / right borders
                left_strip = img.crop((0, 0, p, h)).transpose(Image.FLIP_LEFT_RIGHT)
                padded.paste(left_strip, (0, p))
                right_strip = img.crop((w - p, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
                padded.paste(right_strip, (w + p, p))
                # Top / bottom borders
                top_strip = img.crop((0, 0, w, p)).transpose(Image.FLIP_TOP_BOTTOM)
                padded.paste(top_strip, (p, 0))
                bot_strip = img.crop((0, h - p, w, h)).transpose(Image.FLIP_TOP_BOTTOM)
                padded.paste(bot_strip, (p, h + p))
                # Corners — mirrored on BOTH axes. Pasting only the four
                # edge strips above leaves the corner p x p blocks
                # unpainted (black), since none of the edge strips cover
                # them; each corner must be filled explicitly.
                def _mirror_both(box: tuple[int, int, int, int]) -> Image.Image:
                    return (
                        img.crop(box)
                        .transpose(Image.FLIP_LEFT_RIGHT)
                        .transpose(Image.FLIP_TOP_BOTTOM)
                    )
                padded.paste(_mirror_both((0, 0, p, p)), (0, 0))                  # top-left
                padded.paste(_mirror_both((w - p, 0, w, p)), (w + p, 0))          # top-right
                padded.paste(_mirror_both((0, h - p, p, h)), (0, h + p))          # bottom-left
                padded.paste(_mirror_both((w - p, h - p, w, h)), (w + p, h + p))  # bottom-right

            img = padded

        w, h = img.size
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        return img.crop((x, y, x + self.size, y + self.size))

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class RandomHorizontalFlip:
    """Flip image horizontally with probability `p`."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Flip image vertically with probability `p`."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomRotation:
    """Rotate image by a random angle in [-degrees, +degrees]."""

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR)

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees})"


# ---------------------------------------------------------------------------
# Color transforms
# ---------------------------------------------------------------------------

class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue.

    Each factor is sampled uniformly from [max(0, 1-factor), 1+factor].
    Applied in a random order (following torchvision convention).
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast:   float = 0.0,
        saturation: float = 0.0,
        hue:        float = 0.0,
    ) -> None:
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    @staticmethod
    def _jitter_factor(factor: float) -> float:
        return random.uniform(max(0.0, 1.0 - factor), 1.0 + factor)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Build list of (fn, factor) pairs and shuffle order
        ops = []
        if self.brightness > 0:
            ops.append(("brightness", self._jitter_factor(self.brightness)))
        if self.contrast > 0:
            ops.append(("contrast",   self._jitter_factor(self.contrast)))
        if self.saturation > 0:
            ops.append(("saturation", self._jitter_factor(self.saturation)))
        if self.hue > 0:
            ops.append(("hue",        random.uniform(-self.hue, self.hue)))

        random.shuffle(ops)

        for op, factor in ops:
            if op == "brightness":
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif op == "contrast":
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif op == "saturation":
                img = ImageEnhance.Color(img).enhance(factor)
            elif op == "hue":
                # PIL doesn't have direct hue control; convert to HSV, shift, convert back
                img = _shift_hue(img, factor)

        return img

    def __repr__(self) -> str:
        return (
            f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of `colorsys.rgb_to_hsv`, applied elementwise
    over the last axis. `rgb` is float in [0, 1], shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))
        safe_delta = np.where(delta == 0, 1.0, delta)
        rc = (maxc - r) / safe_delta
        gc = (maxc - g) / safe_delta
        bc = (maxc - b) / safe_delta

    h = np.select(
        [r == maxc, g == maxc],
        [bc - gc, 2.0 + rc - bc],
        default=4.0 + gc - rc,
    )
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of `colorsys.hsv_to_rgb`, shape (..., 3), float in [0, 1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = (i.astype(np.int64)) % 6

    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [t, v, v, q, p, p], default=v)
    b = np.select(conditions, [p, p, t, v, v, q], default=v)

    # s == 0 => grayscale, colorsys returns (v, v, v) regardless of h
    r = np.where(s == 0, v, r)
    g = np.where(s == 0, v, g)
    b = np.where(s == 0, v, b)
    return np.stack([r, g, b], axis=-1)


def _shift_hue(img: Image.Image, shift: float) -> Image.Image:
    """
    Shift hue by `shift` (-0.5 to 0.5).

    Vectorised with numpy (matches `colorsys.rgb_to_hsv`/`hsv_to_rgb` exactly
    — see `tests/test_augmentation.py::TestShiftHue` for a per-pixel
    cross-check) instead of a pure-Python per-pixel loop, which was ~2ms per
    32x32 image and dominated CLIP training's data-loading time (every
    training image gets hue-jittered every epoch).
    """
    arr = np.asarray(img, dtype=np.float64) / 255.0   # (H, W, 3)
    hsv = _rgb_to_hsv_np(arr)
    hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
    rgb = _hsv_to_rgb_np(hsv)
    out = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


class Grayscale:
    """Convert image to grayscale (optionally keeping num_output_channels channels)."""

    def __init__(self, num_output_channels: int = 1) -> None:
        self.num_output_channels = num_output_channels

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        if self.num_output_channels == 3:
            gray = Image.merge("RGB", (gray, gray, gray))
        return gray

    def __repr__(self) -> str:
        return f"Grayscale(num_output_channels={self.num_output_channels})"


class RandomGrayscale:
    """Convert to grayscale with probability `p`."""

    def __init__(self, p: float = 0.1) -> None:
        self.p     = p
        self._gray = Grayscale(num_output_channels=3)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self._gray(img)
        return img


# ---------------------------------------------------------------------------
# Tensor conversion & normalisation
# ---------------------------------------------------------------------------

class ToTensor:
    """
    Convert a PIL Image (H, W, C) to a float32 torch.Tensor (C, H, W)
    in the range [0, 1].

    Does NOT require torchvision.
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0    # (H, W, C), float32
        tensor = torch.from_numpy(arr).permute(2, 0, 1)   # (C, H, W)
        return tensor

    def __repr__(self) -> str:
        return "ToTensor()"


class Normalize:
    """
    Normalise a tensor by channel-wise mean and std:
        output = (input - mean) / std

    Mean and std should be provided as sequences of length C (num channels).
    """

    def __init__(
        self,
        mean: Sequence[float],
        std:  Sequence[float],
    ) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()})"


class Denormalize:
    """Inverse of Normalize — useful for visualisation."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * self.std + self.mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(
    images:  torch.Tensor,
    labels:  torch.Tensor,
    n_classes: int,
    alpha:   float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup data augmentation (Zhang et al., 2018), with a PER-SAMPLE mixing
    coefficient (as used by e.g. timm/DeiT recipes) rather than a single
    lambda shared across the whole batch.

    Each sample i in the batch gets its own lambda_i ~ Beta(alpha, alpha),
    and lambda_i mixes BOTH that sample's image and its one-hot label with
    its randomly paired partner — so the soft label always matches the
    actual pixel-space mixing ratio for that sample.

    Args:
        images:    (B, C, H, W)
        labels:    (B,) integer class labels
        n_classes: Total number of classes
        alpha:     Beta distribution parameter (higher = stronger mix)

    Returns:
        mixed_images: (B, C, H, W)
        mixed_labels: (B, n_classes) soft labels; each row is a proper
                      probability distribution (non-negative, sums to 1).
    """
    B = images.shape[0]

    # One lambda per sample, not one lambda for the whole batch.
    lam_np = np.random.beta(alpha, alpha, size=B).astype(np.float32)
    lam    = torch.from_numpy(lam_np).to(device=images.device, dtype=images.dtype)

    perm = torch.randperm(B, device=images.device)

    lam_img = lam.view(B, *([1] * (images.dim() - 1)))
    mixed   = lam_img * images + (1.0 - lam_img) * images[perm]

    # One-hot encode
    one_hot = torch.zeros(B, n_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    one_hot_perm = one_hot[perm]

    lam_lbl = lam.to(labels.device).view(B, 1)
    mixed_labels = lam_lbl * one_hot + (1.0 - lam_lbl) * one_hot_perm

    return mixed, mixed_labels


# ---------------------------------------------------------------------------
# Preset pipelines
# ---------------------------------------------------------------------------

# CIFAR-10 per-channel stats (computed over training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def cifar10_train_transform() -> Compose:
    """Standard CIFAR-10 training augmentation pipeline."""
    return Compose([
        RandomCrop(32, padding=4, padding_mode="reflect"),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensor(),
        Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def cifar10_val_transform() -> Compose:
    """Standard CIFAR-10 validation/test pipeline (no augmentation)."""
    return Compose([
        ToTensor(),
        Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def clip_train_transform() -> Compose:
    """
    CLIP-style image augmentation for contrastive training.

    Stronger augmentation than classification: colour jitter is applied
    more aggressively since the model must learn view-invariant features.
    """
    return Compose([
        RandomCrop(32, padding=4, padding_mode="reflect"),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        RandomGrayscale(p=0.2),
        ToTensor(),
        Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
