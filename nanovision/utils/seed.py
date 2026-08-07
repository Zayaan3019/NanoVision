"""
Reproducibility helpers.

`set_seed()` is the single entry point every training/eval script calls
before touching random state, so that a given (seed, config, hardware)
triple reproduces the same run. Without it, `random`, `numpy`, and
`torch`'s global RNGs are left at whatever state the interpreter started
in, and augmentation order, weight init, mixup pairing, and dataloader
shuffling all differ silently between runs.

Note on determinism: `torch.use_deterministic_algorithms(True)` makes CUDA
convolutions/attention kernels select deterministic (sometimes slower)
implementations, and raises if an operator has no deterministic
implementation. It is off by default here because this project trains on
CPU/MPS as much as CUDA and SDPA's deterministic coverage varies by
backend; pass `deterministic=True` explicitly when exact bit-for-bit
reproducibility matters more than speed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed python's `random`, `numpy`, and `torch` (CPU + all CUDA devices).

    Args:
        seed:          Integer seed.
        deterministic: If True, also request deterministic algorithms
                        (`torch.use_deterministic_algorithms`) and disable
                        cuDNN autotuning. Slower, but bit-for-bit
                        reproducible on a given device/driver/torch version.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only=True: some ops (e.g. certain SDPA backward kernels) have
        # no deterministic implementation on every backend; fail loudly in
        # research code but don't hard-crash a CPU run over it.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for `DataLoader` — seeds each worker process's RNGs
    so multi-worker dataloading (`num_workers > 0`) is also reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
