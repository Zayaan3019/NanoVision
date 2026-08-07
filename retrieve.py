"""
Cross-modal retrieval demo using a trained CLIP model.

Retrieves images given a text query, or shows recall metrics.

Usage:
    # Build an index from the test set
    python retrieve.py --build --clip_ckpt checkpoints/clip_best.pt

    # Text-to-image retrieval
    python retrieve.py --query "a photo of a cat" --clip_ckpt checkpoints/clip_best.pt

    # Evaluate recall metrics
    python retrieve.py --eval --clip_ckpt checkpoints/clip_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanovision.model.clip import CLIP
from nanovision.data.dataset import load_cifar10, CIFAR10CLIPDataset
from nanovision.data.augmentation import cifar10_val_transform
from nanovision.retrieval.index import EmbeddingIndex
from nanovision.utils.checkpoint import load_clip_from_checkpoint
from nanovision.utils.seed import set_seed

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-modal retrieval with CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clip_ckpt", default="checkpoints/clip_best.pt",
                   help="Path to trained CLIP checkpoint")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--index_path", default="checkpoints/retrieval_index.pt",
                   help="Path to save/load the embedding index")
    p.add_argument("--device",    default=None, help="cuda / cpu / mps (default: auto)")
    p.add_argument("--batch_size", type=int, default=256)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--build",  action="store_true",
                      help="Build embedding index from the test set")
    mode.add_argument("--eval",   action="store_true",
                      help="Evaluate recall@K metrics on the test set")
    mode.add_argument("--query",  default=None, metavar="TEXT",
                      help="Text query for image retrieval")

    p.add_argument("--top_k",     type=int, default=5,
                   help="Number of results to return for --query")
    p.add_argument("--n_eval",    type=int, default=1000,
                   help="Number of test samples to use for --eval and --build")
    p.add_argument("--seed",      type=int, default=42, help="Random seed (reproducibility)")

    return p.parse_args()


def _auto_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(ckpt_path: str, device: torch.device) -> tuple[CLIP, object]:
    model, cfg, _ = load_clip_from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    print(f"Loaded CLIP from {ckpt_path} -> {device}")
    return model, cfg


def _make_dataloader(model: CLIP, data_dir: str, n_eval: int,
                     batch_size: int) -> DataLoader:
    images, labels = load_cifar10(Path(data_dir), split="test")
    if n_eval > 0:
        images, labels = images[:n_eval], labels[:n_eval]

    transform = cifar10_val_transform()
    dataset   = CIFAR10CLIPDataset(
        images=images, labels=labels,
        transform=transform,
        max_seq_len=model.config.text.max_seq_len,
        multi_template=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _build_index(model: CLIP, data_dir: str, device: torch.device,
                 index_path: str, n_eval: int, batch_size: int) -> EmbeddingIndex:
    loader = _make_dataloader(model, data_dir, n_eval, batch_size)
    index  = EmbeddingIndex(embed_dim=model.embed_dim)
    index.build_from_clip(model, loader, device=device)
    index.save(Path(index_path))
    return index


def _load_or_build_index(model: CLIP, data_dir: str, device: torch.device,
                         index_path: str, n_eval: int, batch_size: int) -> EmbeddingIndex:
    p = Path(index_path)
    if p.exists():
        index = EmbeddingIndex.load(p)
        print(f"Loaded existing index from {index_path} ({len(index)} items)")
    else:
        print(f"Index not found at {index_path}. Building...")
        index = _build_index(model, data_dir, device, index_path, n_eval, batch_size)
    return index


def cmd_build(args: argparse.Namespace) -> None:
    device = _auto_device(args.device)
    model, _ = _load_model(args.clip_ckpt, device)
    _build_index(model, args.data_dir, device, args.index_path,
                 args.n_eval, args.batch_size)


def cmd_eval(args: argparse.Namespace) -> None:
    device = _auto_device(args.device)
    model, _ = _load_model(args.clip_ckpt, device)
    index = _load_or_build_index(model, args.data_dir, device,
                                  args.index_path, args.n_eval, args.batch_size)

    print(f"\nEvaluating retrieval on {len(index)} samples...")
    print(f"  {'':>4} {'strict (index)':>16} {'class-aware':>14}")
    for k in (1, 5, 10):
        r_strict = index.recall_at_k(k, direction="i2t", class_aware=False)
        r_class  = index.recall_at_k(k, direction="i2t", class_aware=True)
        print(f"  R@{k:<2d} {r_strict:>15.1f}% {r_class:>13.1f}%")
    print(
        "\n  'strict' = query i's only correct answer is item i (index diagonal).\n"
        "  'class-aware' = any top-K item sharing the query's class counts as a hit.\n"
        "  Captions are one template per class, so many items share an identical\n"
        "  caption — class-aware recall is the metric that reflects retrieval quality."
    )


def cmd_query(args: argparse.Namespace) -> None:
    device = _auto_device(args.device)
    model, _ = _load_model(args.clip_ckpt, device)
    index = _load_or_build_index(model, args.data_dir, device,
                                  args.index_path, args.n_eval, args.batch_size)

    query = args.query
    print(f'\nQuery: "{query}"')
    result = index.query_text_string(query, model, device=device, top_k=args.top_k)

    print(f"\nTop-{args.top_k} results:")
    print(f"  {'Rank':<6} {'Score':>8}  {'Class':<12}  Text")
    print("  " + "-" * 56)
    for rank, (idx, score, label) in enumerate(
        zip(result.indices.tolist(), result.scores.tolist(), result.labels.tolist()), 1
    ):
        cls_name = CIFAR10_CLASSES[label] if label < len(CIFAR10_CLASSES) else str(label)
        text_str = result.texts[rank - 1] if result.texts else ""
        print(f"  {rank:<6} {score:>8.4f}  {cls_name:<12}  {text_str}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.build:
        cmd_build(args)
    elif args.eval:
        cmd_eval(args)
    elif args.query:
        cmd_query(args)
    else:
        print("Specify one of: --build, --eval, --query TEXT")
        print("Example: python retrieve.py --query 'a photo of a dog'")


if __name__ == "__main__":
    main()
