"""
RevuLens Dataset Preprocessing & Splitting Pipeline (Step 1).

Adheres strictly to GEMINI.md:
1. One shared preprocessing function from backend.app.preprocessing (lowercase, remove punctuation/URLs/HTML, collapse whitespace, NO stop-word removal).
2. Deduplication before splitting:
   - Salminen: drops duplicate texts (~20 duplicates).
   - FiReCS: drops duplicate texts and train/test overlaps.
3. Salminen Stratified Split: 80/10/10 train/validation/test with fixed seed 42.
4. FiReCS is NEVER used for training and NEVER labeled Genuine or Deceptive. Saved as an external exploratory check only.
5. All outputs saved to data/processed/ (gitignored).
"""

import os
import sys
import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.preprocessing import normalize_text
from backend.app.constants import InternalClass

RANDOM_SEED = 42


def load_and_deduplicate_salminen(file_path: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load Salminen et al. (2022) Fake Reviews Dataset and deduplicate by normalized text.
    Mapping: 'OR' -> Genuine, 'CG' -> Deceptive.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Salminen dataset file not found: {file_path}")

    raw_records = []
    seen_texts = set()
    duplicate_count = 0

    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row.get("label", "").strip()
            if raw_label in ("OR", "Genuine", "1", "real", "genuine"):
                internal_label = InternalClass.GENUINE.value
            elif raw_label in ("CG", "Deceptive", "0", "fake", "deceptive"):
                internal_label = InternalClass.DECEPTIVE.value
            else:
                continue

            raw_text = row.get("text_", row.get("text", "")).strip()
            if not raw_text:
                continue

            cleaned_text = normalize_text(raw_text)
            if not cleaned_text:
                continue

            # Deduplication check
            if cleaned_text in seen_texts:
                duplicate_count += 1
                continue

            seen_texts.add(cleaned_text)
            raw_records.append({
                "text": raw_text,
                "cleaned_text": cleaned_text,
                "label": internal_label,
                "category": row.get("category", "General"),
                "rating": row.get("rating", "")
            })

    return raw_records, duplicate_count


def load_and_deduplicate_firecs(train_path: str, test_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load FiReCS (Cosme & De Leon, 2024) dataset, normalize, and deduplicate across train/test files.
    IMPORTANT (GEMINI.md): FiReCS is NEVER labeled Genuine or Deceptive.
    It contains sentiment labels only and serves strictly as an external exploratory check.
    """
    paths = [p for p in (train_path, test_path) if p and os.path.exists(p)]
    seen_texts = set()
    records = []
    duplicate_count = 0

    for p in paths:
        with open(p, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_text = row.get("review", row.get("text", "")).strip()
                if not raw_text:
                    continue

                cleaned_text = normalize_text(raw_text)
                if not cleaned_text:
                    continue

                if cleaned_text in seen_texts:
                    duplicate_count += 1
                    continue

                seen_texts.add(cleaned_text)
                records.append({
                    "text": raw_text,
                    "cleaned_text": cleaned_text,
                    "sentiment_label": row.get("label", "").strip(),
                    "source": "FiReCS_Taglish_Exploratory"
                })

    return records, duplicate_count


def stratified_split_80_10_10(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = RANDOM_SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stratified 80/10/10 train/validation/test split on Salminen with fixed seed 42.
    Guarantees equal class distribution (Genuine / Deceptive) across all splits.
    """
    random.seed(random_state)
    genuine = [r for r in records if r["label"] == InternalClass.GENUINE.value]
    deceptive = [r for r in records if r["label"] == InternalClass.DECEPTIVE.value]

    random.shuffle(genuine)
    random.shuffle(deceptive)

    def split_group(group):
        n = len(group)
        n_test = max(1, int(n * test_ratio)) if n >= 10 else int(n * test_ratio)
        n_val = max(1, int(n * val_ratio)) if n >= 10 else int(n * val_ratio)
        test = group[:n_test]
        val = group[n_test:n_test + n_val]
        train = group[n_test + n_val:]
        return train, val, test

    train_g, val_g, test_g = split_group(genuine)
    train_d, val_d, test_d = split_group(deceptive)

    train = train_g + train_d
    val = val_g + val_d
    test = test_g + test_d

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_csv(records: List[Dict[str, Any]], file_path: str, fieldnames: List[str]) -> None:
    """Save records to CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run_pipeline(
    raw_salminen_path: Optional[str] = None,
    firecs_train_path: Optional[str] = None,
    firecs_test_path: Optional[str] = None,
    output_dir: str = "data/processed",
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Execute full data preparation pipeline per GEMINI.md.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Locate raw paths
    default_salminen = PROJECT_ROOT / "data" / "raw" / "fake reviews dataset.csv"
    default_firecs_train = PROJECT_ROOT / "data" / "raw" / "FiReCS_train_set.csv"
    default_firecs_test = PROJECT_ROOT / "data" / "raw" / "FiReCS_test_set.csv"

    if not raw_salminen_path and default_salminen.exists():
        raw_salminen_path = str(default_salminen)
    if not firecs_train_path and default_firecs_train.exists():
        firecs_train_path = str(default_firecs_train)
    if not firecs_test_path and default_firecs_test.exists():
        firecs_test_path = str(default_firecs_test)

    # 1. Process Salminen (Training & Evaluation Dataset)
    if raw_salminen_path and os.path.exists(raw_salminen_path):
        print(f"Loading and deduplicating Salminen dataset: {raw_salminen_path}")
        salminen_records, salminen_dupes = load_and_deduplicate_salminen(raw_salminen_path)
        print(f"Salminen deduplication: dropped {salminen_dupes} duplicate rows.")
        print(f"Remaining unique Salminen records: {len(salminen_records)}")
    else:
        raise FileNotFoundError(f"Salminen dataset not found at: {raw_salminen_path}")

    # Stratified 80/10/10 split
    train, val, test = stratified_split_80_10_10(salminen_records, val_ratio=0.10, test_ratio=0.10, random_state=random_state)

    salminen_fields = ["text", "cleaned_text", "label", "category", "rating"]
    save_csv(train, str(output_path / "train.csv"), salminen_fields)
    save_csv(val, str(output_path / "val.csv"), salminen_fields)
    save_csv(test, str(output_path / "test.csv"), salminen_fields)

    # 2. Process FiReCS (External Exploratory Dataset ONLY)
    firecs_records = []
    firecs_dupes = 0
    if firecs_train_path and os.path.exists(firecs_train_path):
        print(f"\nLoading and deduplicating FiReCS Taglish dataset: {firecs_train_path}")
        firecs_records, firecs_dupes = load_and_deduplicate_firecs(firecs_train_path, firecs_test_path)
        print(f"FiReCS deduplication: dropped {firecs_dupes} duplicate / overlapping rows.")
        print(f"Remaining unique FiReCS records: {len(firecs_records)}")

        firecs_fields = ["text", "cleaned_text", "sentiment_label", "source"]
        save_csv(firecs_records, str(output_path / "firecs_exploratory.csv"), firecs_fields)

    # Summary statistics
    def get_split_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        g = sum(1 for r in subset if r["label"] == InternalClass.GENUINE.value)
        d = sum(1 for r in subset if r["label"] == InternalClass.DECEPTIVE.value)
        avg_len = sum(len(r["cleaned_text"].split()) for r in subset) / max(len(subset), 1)
        return {
            "total": len(subset),
            "genuine": g,
            "deceptive": d,
            "genuine_pct": round(g / max(len(subset), 1) * 100, 2),
            "deceptive_pct": round(d / max(len(subset), 1) * 100, 2),
            "avg_word_count": round(avg_len, 2)
        }

    metadata = {
        "dataset_rules_version": "GEMINI.md (80/10/10 split, fixed seed 42)",
        "salminen_total_unique": len(salminen_records),
        "salminen_duplicates_dropped": salminen_dupes,
        "firecs_total_unique": len(firecs_records),
        "firecs_duplicates_dropped": firecs_dupes,
        "firecs_role": "External exploratory check only (never used in training, no authenticity label)",
        "splits": {
            "train": get_split_stats(train),
            "val": get_split_stats(val),
            "test": get_split_stats(test)
        }
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPipeline successfully completed! Splits saved to {output_dir}:")
    print(f"  Train: {len(train)} records ({metadata['splits']['train']['genuine']} Genuine, {metadata['splits']['train']['deceptive']} Deceptive)")
    print(f"  Val:   {len(val)} records ({metadata['splits']['val']['genuine']} Genuine, {metadata['splits']['val']['deceptive']} Deceptive)")
    print(f"  Test:  {len(test)} records ({metadata['splits']['test']['genuine']} Genuine, {metadata['splits']['test']['deceptive']} Deceptive)")
    if firecs_records:
        print(f"  FiReCS Exploratory: {len(firecs_records)} unique Taglish records (unlabeled for authenticity)")

    return metadata


if __name__ == "__main__":
    run_pipeline()
