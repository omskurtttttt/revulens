"""
RevuLens Dataset Preprocessing & Ingestion Pipeline.

Processes raw e-commerce review datasets:
- Fake Reviews Dataset (Salminen et al., 2022): maps 'OR' -> 'Genuine', 'CG' -> 'Deceptive'
- FiReCS / SentiTaglish (Cosme & De Leon, 2024): ingestion for localized Taglish vocabulary and robustness testing
  (sentiment labels kept separate from authenticity labels per GEMINI.md)

Outputs stratified train/val/test splits to data/processed/ (gitignored).
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

from backend.app.services.preprocessor import ReviewPreprocessor
from backend.app.constants import InternalClass


def load_raw_salminen_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load and parse the Salminen et al. (2022) Fake Reviews Dataset.
    Expected columns: category, rating, label ('OR' or 'CG'), text_
    """
    records = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Salminen dataset file not found at: {file_path}")

    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row.get("label", "").strip()
            # Map labels: 'OR' (Original) -> Genuine, 'CG' (Computer-Generated) -> Deceptive
            if raw_label in ("OR", "Genuine", "1", "real", "genuine"):
                internal_label = InternalClass.GENUINE.value
            elif raw_label in ("CG", "Deceptive", "0", "fake", "deceptive"):
                internal_label = InternalClass.DECEPTIVE.value
            else:
                continue

            text = row.get("text_", row.get("text", "")).strip()
            if text:
                records.append({
                    "text": text,
                    "label": internal_label,
                    "category": row.get("category", "General"),
                    "rating": row.get("rating", "")
                })

    return records


def load_raw_firecs_dataset(train_path: str, test_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load the FiReCS / SentiTaglish dataset (Cosme & De Leon, 2024).
    Contains authentic Filipino-English code-switched (Taglish) e-commerce reviews.
    Per GEMINI.md, sentiment polarity is preserved as sentiment metadata, NOT authenticity labels.
    """
    records = []
    paths = [p for p in (train_path, test_path) if p and os.path.exists(p)]

    for p in paths:
        with open(p, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("review", row.get("text", "")).strip()
                sentiment = row.get("label", "").strip()
                if text:
                    records.append({
                        "text": text,
                        "sentiment_label": sentiment,
                        "source": "FiReCS_Taglish",
                        "category": "E-Commerce-Taglish"
                    })

    return records


def generate_sample_dataset() -> List[Dict[str, str]]:
    """
    Generate a balanced representative sample dataset (English, Filipino, Taglish)
    for development, testing, and pipeline validation when the raw datasets are absent.
    """
    samples = [
        # Genuine English
        {"text": "I bought this mouse two weeks ago. The click latency is low and battery lasts long. Highly recommend!", "label": InternalClass.GENUINE.value, "category": "Electronics"},
        {"text": "The fabric is a bit thinner than expected, but for the price it's definitely reasonable. Fast delivery too.", "label": InternalClass.GENUINE.value, "category": "Clothing"},
        {"text": "Arrived within 3 days in solid packaging. Tested and working as described.", "label": InternalClass.GENUINE.value, "category": "Home"},
        # Deceptive English (template-like / repetitive / GPT-style)
        {"text": "Best product ever bought in my entire life! Amazing! Buy it now everyone! Super super good!", "label": InternalClass.DECEPTIVE.value, "category": "Electronics"},
        {"text": "This amazing item completely transformed my expectations. The incredible quality makes it superior to all alternatives.", "label": InternalClass.DECEPTIVE.value, "category": "Beauty"},
        {"text": "Five stars five stars best seller best product fast shipment highly recommend to all consumers forever.", "label": InternalClass.DECEPTIVE.value, "category": "Home"},
        # Genuine Taglish / Filipino
        {"text": "Sobrang ganda ng packaging, may bubble wrap pa. Legit yung item at maayos kausap si seller. Salamat po!", "label": InternalClass.GENUINE.value, "category": "General"},
        {"text": "Dumating kahapon yung order ko. Medyo matagal lang shipping pero worth it naman kasi gumagana lahat.", "label": InternalClass.GENUINE.value, "category": "Electronics"},
        {"text": "Ayos yung tela, sakto ang fit sa akin. Order ulit ako sa susunod ibang kulay naman.", "label": InternalClass.GENUINE.value, "category": "Clothing"},
        # Deceptive Taglish / Filipino
        {"text": "Napakaganda napakaganda sobra ganda ganda bili na kayo legit na legit 1000 stars para kay seller!", "label": InternalClass.DECEPTIVE.value, "category": "General"},
        {"text": "Maganda maganda maganda maganda maganda salamat salamat salamat seller ganda ganda ganda.", "label": InternalClass.DECEPTIVE.value, "category": "General"},
        {"text": "The greatest item in the universe sobrang ganda talaga buy now best quality ever.", "label": InternalClass.DECEPTIVE.value, "category": "General"},
    ]
    return samples


def stratified_split(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split records into train, validation, and test sets with stratified class balance.
    """
    random.seed(random_state)
    genuine = [r for r in records if r["label"] == InternalClass.GENUINE.value]
    deceptive = [r for r in records if r["label"] == InternalClass.DECEPTIVE.value]

    random.shuffle(genuine)
    random.shuffle(deceptive)

    def split_group(group):
        n = len(group)
        if n >= 3:
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
        else:
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
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


def save_split_csv(records: List[Dict[str, Any]], file_path: str, extra_fields: Optional[List[str]] = None) -> None:
    """Save records to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fieldnames = ["text", "cleaned_text", "label", "language_hint", "char_count", "category"]
    if extra_fields:
        fieldnames.extend(extra_fields)

    with open(file_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run_pipeline(
    raw_csv_path: Optional[str] = None,
    firecs_train_path: Optional[str] = None,
    firecs_test_path: Optional[str] = None,
    output_dir: str = "data/processed",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run data ingestion, preprocessing, and stratified splitting for Salminen and FiReCS datasets.
    """
    preprocessor = ReviewPreprocessor()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect default raw dataset paths if not provided
    default_salminen = PROJECT_ROOT / "data" / "raw" / "fake reviews dataset.csv"
    default_firecs_train = PROJECT_ROOT / "data" / "raw" / "FiReCS_train_set.csv"
    default_firecs_test = PROJECT_ROOT / "data" / "raw" / "FiReCS_test_set.csv"

    if not raw_csv_path and default_salminen.exists():
        raw_csv_path = str(default_salminen)
    if not firecs_train_path and default_firecs_train.exists():
        firecs_train_path = str(default_firecs_train)
    if not firecs_test_path and default_firecs_test.exists():
        firecs_test_path = str(default_firecs_test)

    # 1. Process Authenticity Labeled Dataset (Salminen et al.)
    if raw_csv_path and os.path.exists(raw_csv_path):
        print(f"Loading Salminen et al. dataset from: {raw_csv_path}")
        raw_records = load_raw_salminen_dataset(raw_csv_path)
    else:
        print("No raw authenticity dataset found. Using representative sample dataset.")
        raw_records = generate_sample_dataset()

    print(f"Total raw authenticity records: {len(raw_records)}")

    # Clean and filter
    processed_records = []
    skipped_count = 0

    for r in raw_records:
        meta = preprocessor.process(r["text"])
        if meta["is_valid"]:
            processed_records.append({
                "text": r["text"],
                "cleaned_text": meta["cleaned_text"],
                "label": r["label"],
                "language_hint": meta["language_hint"],
                "char_count": meta["char_count"],
                "category": r.get("category", "General"),
            })
        else:
            skipped_count += 1

    print(f"Valid authenticity records: {len(processed_records)} (skipped: {skipped_count})")

    # Stratified split
    train, val, test = stratified_split(
        processed_records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    save_split_csv(train, str(output_path / "train.csv"))
    save_split_csv(val, str(output_path / "val.csv"))
    save_split_csv(test, str(output_path / "test.csv"))

    # 2. Process FiReCS / SentiTaglish (Localized Taglish Reference Dataset)
    firecs_count = 0
    if firecs_train_path and os.path.exists(firecs_train_path):
        print(f"Loading FiReCS Taglish dataset from: {firecs_train_path}")
        raw_firecs = load_raw_firecs_dataset(firecs_train_path, firecs_test_path)
        processed_firecs = []
        for r in raw_firecs:
            meta = preprocessor.process(r["text"])
            if meta["is_valid"]:
                processed_firecs.append({
                    "text": r["text"],
                    "cleaned_text": meta["cleaned_text"],
                    "label": "Taglish_Reference",
                    "sentiment_label": r["sentiment_label"],
                    "language_hint": meta["language_hint"],
                    "char_count": meta["char_count"],
                    "category": r["category"]
                })
        firecs_count = len(processed_firecs)
        save_split_csv(processed_firecs, str(output_path / "firecs_taglish_reference.csv"), extra_fields=["sentiment_label"])
        print(f"FiReCS Taglish reference records saved: {firecs_count}")

    # Compute summary statistics
    def get_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        genuine_count = sum(1 for r in subset if r["label"] == InternalClass.GENUINE.value)
        deceptive_count = sum(1 for r in subset if r["label"] == InternalClass.DECEPTIVE.value)
        avg_len = sum(r["char_count"] for r in subset) / max(len(subset), 1)
        lang_counts = {}
        for r in subset:
            l = r["language_hint"]
            lang_counts[l] = lang_counts.get(l, 0) + 1
        return {
            "total": len(subset),
            "genuine": genuine_count,
            "deceptive": deceptive_count,
            "avg_char_length": round(avg_len, 2),
            "language_distribution": lang_counts
        }

    metadata = {
        "total_authenticity_processed": len(processed_records),
        "total_authenticity_skipped": skipped_count,
        "firecs_taglish_reference_count": firecs_count,
        "splits": {
            "train": get_stats(train),
            "val": get_stats(val),
            "test": get_stats(test)
        }
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPipeline successfully completed! Splits saved to {output_dir}:")
    print(f"  Train: {len(train)} records")
    print(f"  Val:   {len(val)} records")
    print(f"  Test:  {len(test)} records")
    if firecs_count:
        print(f"  FiReCS Taglish reference: {firecs_count} records")

    return metadata


if __name__ == "__main__":
    salminen = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(raw_csv_path=salminen)
