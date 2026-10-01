"""
Step 2: DistilBERT Multilingual Frozen Embedding Extraction.

Extracts contextual representations using frozen 'distilbert-base-multilingual-cased'
and saves embeddings and label arrays to disk (.npy) per GEMINI.md.

Design constraints:
- Encoder: distilbert-base-multilingual-cased (frozen, no fine-tuning)
- Default pooling: Mean pooling over token vectors excluding padding
- Optional pooling: CLS token pooling (reported alongside mean pooling)
- Sequence length: 128 tokens (reviews are short)
- Saves .npy arrays to data/processed/ (gitignored)
- Compatible with local execution (CPU/CUDA) and Google Colab (T4 GPU).
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.embedding import DistilBERTEmbeddingExtractor
from backend.app.constants import InternalClass


def load_split_data(csv_path: str, text_col: str = "cleaned_text", label_col: Optional[str] = "label", limit: Optional[int] = None) -> Tuple[List[str], Optional[np.ndarray]]:
    """Load text and optional binary label array (0 for Genuine, 1 for Deceptive) from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    texts = []
    labels = []

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get(text_col, "").strip()
            if not t:
                continue
            texts.append(t)

            if label_col and label_col in row:
                lbl = row[label_col].strip()
                # Binary label encoding: Genuine -> 0, Deceptive -> 1
                if lbl == InternalClass.GENUINE.value:
                    labels.append(0)
                elif lbl == InternalClass.DECEPTIVE.value:
                    labels.append(1)
                else:
                    labels.append(-1)

            if limit and len(texts) >= limit:
                break

    labels_array = np.array(labels, dtype=np.int64) if labels else None
    return texts, labels_array


def extract_and_save(
    extractor: DistilBERTEmbeddingExtractor,
    split_name: str,
    input_csv: str,
    output_dir: Path,
    batch_size: int = 64,
    pooling: str = "mean",
    limit: Optional[int] = None,
    has_labels: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract embeddings for a split and save .npy arrays to disk."""
    print(f"\n--- Processing [{split_name.upper()}] (pooling={pooling}) ---")
    start_time = time.time()

    label_col = "label" if has_labels else None
    texts, labels = load_split_data(input_csv, label_col=label_col, limit=limit)
    print(f"Loaded {len(texts)} reviews from {input_csv}")

    embeddings = extractor.extract(texts, batch_size=batch_size, pooling_strategy=pooling)
    elapsed = time.time() - start_time

    # Save embeddings
    emb_filename = f"{split_name}_embeddings_{pooling}.npy"
    emb_path = output_dir / emb_filename
    np.save(emb_path, embeddings)

    file_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"Saved: {emb_path.name} -> Shape: {embeddings.shape}, Size: {file_size_mb:.2f} MB, Time: {elapsed:.2f}s ({len(texts)/max(elapsed, 0.001):.1f} reviews/sec)")

    # Save labels if present
    if labels is not None and len(labels) == len(embeddings):
        lbl_filename = f"{split_name}_labels.npy"
        lbl_path = output_dir / lbl_filename
        np.save(lbl_path, labels)
        print(f"Saved: {lbl_path.name} -> Count: {len(labels)} (Genuine: {np.sum(labels == 0)}, Deceptive: {np.sum(labels == 1)})")

    return embeddings, labels


def run_embedding_pipeline(
    output_dir: str = "data/processed",
    pooling: str = "mean",
    max_length: int = 128,
    batch_size: int = 64,
    splits: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Execute embedding extraction pipeline for all requested splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "val", "test", "firecs"]

    print("=" * 65)
    print(f"RevuLens DistilBERT Embedding Extraction (Step 2)")
    print(f"Model: distilbert-base-multilingual-cased (Frozen)")
    print(f"Pooling Strategy: {pooling}")
    print(f"Max Sequence Length: {max_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Target Splits: {splits}")
    print("=" * 65)

    extractor = DistilBERTEmbeddingExtractor(
        max_length=max_length,
        pooling_strategy=pooling
    )
    print(f"Model successfully loaded on device: '{extractor.device.upper()}'")

    results = {}
    split_configs = {
        "train": ("train.csv", True),
        "val": ("val.csv", True),
        "test": ("test.csv", True),
        "firecs": ("firecs_exploratory.csv", False),
    }

    for sp in splits:
        if sp not in split_configs:
            continue
        csv_name, has_labels = split_configs[sp]
        csv_path = out_path / csv_name

        if not csv_path.exists():
            print(f"Skipping '{sp}' - file not found: {csv_path}")
            continue

        emb, lbl = extract_and_save(
            extractor=extractor,
            split_name=sp,
            input_csv=str(csv_path),
            output_dir=out_path,
            batch_size=batch_size,
            pooling=pooling,
            limit=limit,
            has_labels=has_labels
        )
        results[sp] = {
            "shape": list(emb.shape),
            "file": f"{sp}_embeddings_{pooling}.npy"
        }

    print("\n" + "=" * 65)
    print("Embedding Extraction Step 2 Complete! Saved artifacts:")
    for sp, info in results.items():
        print(f"  [{sp}] {info['file']} -> Shape: {info['shape']}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuLens DistilBERT Embedding Extraction (Step 2)")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy (default: mean)")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length (default: 128)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "firecs"], help="Splits to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split (for testing)")
    args = parser.parse_args()

    run_embedding_pipeline(
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        splits=args.splits,
        limit=args.limit
    )
