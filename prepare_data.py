"""
CIFAR-10 data download and extraction.

Downloads the CIFAR-10 Python version from the official source and
extracts it to data/cifar-10-batches-py/.  All subsequent scripts
read from that directory.

Usage:
    python prepare_data.py
    python prepare_data.py --data_dir my_data/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


class _TqdmProgress(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cifar10(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-10-python.tar.gz"

    if tar_path.exists() and _md5(tar_path) == CIFAR10_MD5:
        print(f"Archive already present and verified: {tar_path}")
    else:
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        with _TqdmProgress(unit="B", unit_scale=True, miniters=1, desc="cifar-10") as t:
            urllib.request.urlretrieve(CIFAR10_URL, tar_path, reporthook=t.update_to)

        actual_md5 = _md5(tar_path)
        if actual_md5 != CIFAR10_MD5:
            raise RuntimeError(
                f"MD5 mismatch: expected {CIFAR10_MD5}, got {actual_md5}. "
                "Try deleting the file and re-running."
            )
        print("Download verified.")

    # Extract
    batch_dir = data_dir / "cifar-10-batches-py"
    if batch_dir.exists():
        print(f"Already extracted: {batch_dir}")
    else:
        print("Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"Extracted to {batch_dir}")

    # Quick sanity check
    expected = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    missing  = [e for e in expected if not (batch_dir / e).exists()]
    if missing:
        raise RuntimeError(f"Extraction incomplete. Missing files: {missing}")

    print("\nCIFAR-10 ready.")
    print(f"  Location : {batch_dir}")
    print(f"  Train    : 50,000 images (5 batches x 10,000)")
    print(f"  Test     :  10,000 images")
    print(f"  Classes  : 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
    print("\nNext steps:")
    print("  python train_vit.py   --data_dir data")
    print("  python train_clip.py  --data_dir data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare CIFAR-10.")
    p.add_argument("--data_dir", default="data", help="Root data directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_cifar10(Path(args.data_dir))
