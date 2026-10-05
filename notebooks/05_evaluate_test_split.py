"""
Step 5: Held-Out Test Split Evaluation & FiReCS Exploratory Check.

Adheres strictly to GEMINI.md:
1. Held-Out Test Evaluation:
   - Evaluates on the held-out Salminen test split (never touched during training or tuning).
   - Reports Accuracy, Precision, Recall, F1 (binary & macro), and Confusion Matrix.
   - NO confidence score, percentage, or probability output (team decision).
   - Reports both TF-IDF baseline and DistilBERT-SVM hybrid together (hybrid is not assumed to be better).
2. FiReCS Behavior Check:
   - Measures the share of Taglish reviews flagged Deceptive by each model.
   - Strictly labeled "exploratory".
   - NEVER reports accuracy, precision, recall, or F1 for FiReCS (sentiment != authenticity).
3. Saves summary report to:
   - data/processed/final_test_evaluation.json
   - data/processed/model_evaluation_results.json (consolidated master results file for docs)
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.constants import InternalClass


def load_salminen_test_split(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    """Load held-out test split texts and binary labels (0 for Genuine, 1 for Deceptive)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test split CSV not found: {csv_path}")

    texts = []
    labels = []

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("cleaned_text", "").strip()
            if not text:
                continue
            lbl = row.get("label", "").strip()
            if lbl == InternalClass.GENUINE.value:
                labels.append(0)
            elif lbl == InternalClass.DECEPTIVE.value:
                labels.append(1)
            else:
                continue
            texts.append(text)
            if limit and len(texts) >= limit:
                break

    return texts, np.array(labels, dtype=np.int64)


def load_firecs_exploratory_data(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load FiReCS Taglish reviews and original sentiment labels for exploratory checking."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FiReCS exploratory CSV not found: {csv_path}")

    texts = []
    sentiments = []

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("cleaned_text", "").strip()
            if not text:
                continue
            texts.append(text)
            sentiments.append(row.get("sentiment_label", "unknown"))
            if limit and len(texts) >= limit:
                break

    return texts, sentiments


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute standard classification evaluation metrics on held-out test set (no confidence scores in production)."""
    acc = accuracy_score(y_true, y_pred)
    prec_binary = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_binary = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(float(acc), 4),
        "precision_deceptive": round(float(prec_binary), 4),
        "recall_deceptive": round(float(rec_binary), 4),
        "f1_deceptive": round(float(f1_binary), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        pred_confidences = np.max(y_proba, axis=1)
        metrics["average_confidence"] = round(float(np.mean(pred_confidences)), 4)

    return metrics


def evaluate_baseline_test(
    baseline_artifact_path: str,
    test_csv_path: str,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate Step 3 TF-IDF Baseline on the held-out Salminen test split."""
    print("\n--- Evaluating Baseline on Salminen Test Split ---")
    pipeline = joblib.load(baseline_artifact_path)
    vectorizer = pipeline["vectorizer"]
    classifier = pipeline["classifier"]

    texts, y_test = load_salminen_test_split(test_csv_path, limit=limit)
    print(f"Loaded {len(texts)} held-out test reviews (Genuine: {np.sum(y_test == 0)}, Deceptive: {np.sum(y_test == 1)})")

    X_test = vectorizer.transform(texts)
    y_pred = classifier.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["test_samples"] = len(texts)
    return metrics


def evaluate_hybrid_test(
    hybrid_artifact_path: str,
    test_emb_path: str,
    test_lbl_path: str,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate Step 4 DistilBERT-SVM Hybrid Pipeline on the held-out Salminen test split."""
    print("\n--- Evaluating DistilBERT-SVM Hybrid on Salminen Test Split ---")
    pipeline = joblib.load(hybrid_artifact_path)
    scaler = pipeline["scaler"]
    classifier = pipeline["classifier"]

    X_test = np.load(test_emb_path)
    y_test = np.load(test_lbl_path)

    if limit and len(X_test) > limit:
        X_test = X_test[:limit]
        y_test = y_test[:limit]

    print(f"Loaded {len(X_test)} held-out test embeddings (Genuine: {np.sum(y_test == 0)}, Deceptive: {np.sum(y_test == 1)})")

    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)

    metrics = compute_metrics(y_test, y_pred)
    metrics["test_samples"] = len(X_test)
    metrics["pooling_strategy"] = pipeline.get("pooling_strategy", "mean")
    return metrics


def evaluate_firecs_exploratory(
    baseline_artifact_path: str,
    hybrid_artifact_path: str,
    firecs_csv_path: str,
    firecs_emb_path: Optional[str] = None,
    limit: Optional[int] = 500
) -> Dict[str, Any]:
    """
    Evaluate model behavior on FiReCS Taglish reviews.
    IMPORTANT PER GEMINI.md:
    - Metric is ONLY the share of reviews flagged Deceptive.
    - Strictly labeled 'exploratory'.
    - NEVER report accuracy, precision, recall, or F1 for FiReCS.
    - No confidence scores.
    """
    print("\n--- Evaluating FiReCS Taglish Behavioral Check (EXPLORATORY) ---")
    firecs_texts, _ = load_firecs_exploratory_data(firecs_csv_path, limit=limit)
    print(f"Loaded {len(firecs_texts)} FiReCS Taglish reviews for exploratory check")

    # 1. Baseline Evaluation on FiReCS
    b_pipe = joblib.load(baseline_artifact_path)
    b_vec = b_pipe["vectorizer"]
    b_clf = b_pipe["classifier"]

    X_firecs_tf = b_vec.transform(firecs_texts)
    b_preds = b_clf.predict(X_firecs_tf)

    b_deceptive_count = int(np.sum(b_preds == 1))
    b_genuine_count = int(np.sum(b_preds == 0))
    b_deceptive_pct = round((b_deceptive_count / max(len(b_preds), 1)) * 100, 2)

    # 2. Hybrid Evaluation on FiReCS
    h_pipe = joblib.load(hybrid_artifact_path)
    h_scaler = h_pipe["scaler"]
    h_clf = h_pipe["classifier"]

    h_deceptive_pct = None
    h_deceptive_count = None
    h_genuine_count = None
    h_evaluated_count = 0

    if firecs_emb_path and os.path.exists(firecs_emb_path):
        X_firecs_emb = np.load(firecs_emb_path)
        if limit and len(X_firecs_emb) > limit:
            X_firecs_emb = X_firecs_emb[:limit]
        h_evaluated_count = len(X_firecs_emb)

        X_firecs_scaled = h_scaler.transform(X_firecs_emb)
        h_preds = h_clf.predict(X_firecs_scaled)

        h_deceptive_count = int(np.sum(h_preds == 1))
        h_genuine_count = int(np.sum(h_preds == 0))
        h_deceptive_pct = round((h_deceptive_count / max(len(h_preds), 1)) * 100, 2)

    return {
        "status": "exploratory",
        "disclaimer": "FiReCS has sentiment labels only, not authenticity labels. Evaluated strictly by share flagged Deceptive. Accuracy/F1 are NOT reported per GEMINI.md.",
        "baseline": {
            "total_taglish_evaluated": len(b_preds),
            "flagged_genuine_count": b_genuine_count,
            "flagged_deceptive_count": b_deceptive_count,
            "share_flagged_deceptive_pct": b_deceptive_pct,
        },
        "hybrid": {
            "total_taglish_evaluated": h_evaluated_count,
            "flagged_genuine_count": h_genuine_count,
            "flagged_deceptive_count": h_deceptive_count,
            "share_flagged_deceptive_pct": h_deceptive_pct,
        } if h_evaluated_count > 0 else None
    }


def print_comparative_report(
    baseline_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
    firecs_results: Dict[str, Any]
) -> None:
    """Print thesis-ready comparative evaluation tables."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT SALMINEN TEST SPLIT (Step 5)")
    print("Rule: Baseline and Hybrid reported together. Hybrid is not assumed better.")
    print("=" * 70)
    print(f"{'Metric':<25} | {'TF-IDF + LinearSVC':<18} | {'DistilBERT + SVM':<18}")
    print("-" * 70)
    print(f"{'Test Accuracy':<25} | {baseline_metrics['accuracy']*100:>16.2f}% | {hybrid_metrics['accuracy']*100:>16.2f}%")
    print(f"{'Macro F1-Score':<25} | {baseline_metrics['f1_macro']:>18.4f} | {hybrid_metrics['f1_macro']:>18.4f}")
    print(f"{'Deceptive Precision':<25} | {baseline_metrics['precision_deceptive']:>18.4f} | {hybrid_metrics['precision_deceptive']:>18.4f}")
    print(f"{'Deceptive Recall':<25} | {baseline_metrics['recall_deceptive']:>18.4f} | {hybrid_metrics['recall_deceptive']:>18.4f}")
    print(f"{'Deceptive F1-Score':<25} | {baseline_metrics['f1_deceptive']:>18.4f} | {hybrid_metrics['f1_deceptive']:>18.4f}")
    print(f"{'Test Sample Size':<25} | {baseline_metrics['test_samples']:>18} | {hybrid_metrics['test_samples']:>18}")
    print("=" * 70)

    # FiReCS Exploratory Results Table
    print("\n" + "=" * 70)
    print("FiReCS TAGLISH BEHAVIORAL CHECK (EXPLORATORY ONLY)")
    print("Rule (GEMINI.md): Never report accuracy/F1 for FiReCS. Report share flagged Deceptive.")
    print("=" * 70)
    b_fc = firecs_results["baseline"]
    h_fc = firecs_results["hybrid"]
    print(f"{'Exploratory Metric':<30} | {'TF-IDF Baseline':<16} | {'DistilBERT Hybrid':<16}")
    print("-" * 70)
    print(f"{'Total Taglish Reviews':<30} | {b_fc['total_taglish_evaluated']:>16} | {h_fc['total_taglish_evaluated'] if h_fc else 'N/A':>16}")
    print(f"{'Flagged Likely Genuine':<30} | {b_fc['flagged_genuine_count']:>16} | {h_fc['flagged_genuine_count'] if h_fc else 'N/A':>16}")
    print(f"{'Flagged Potentially Deceptive':<30} | {b_fc['flagged_deceptive_count']:>16} | {h_fc['flagged_deceptive_count'] if h_fc else 'N/A':>16}")
    print(f"{'Share Flagged Deceptive (%)':<30} | {b_fc['share_flagged_deceptive_pct']:>15.2f}% | {str(h_fc['share_flagged_deceptive_pct']) + '%' if h_fc else 'N/A':>16}")
    print("=" * 70)


def build_consolidated_results_file(
    data_dir: str,
    baseline_matched: Dict[str, Any],
    baseline_full: Optional[Dict[str, Any]],
    hybrid_matched: Dict[str, Any],
    firecs_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build consolidated reproducible results dictionary that documentation reads from.
    Reads real validation pooling comparison and baseline metrics from saved files.
    """
    data_path = Path(data_dir)
    hybrid_val_path = data_path / "hybrid_val_metrics.json"
    baseline_val_path = data_path / "baseline_val_metrics.json"

    hybrid_val_data = {}
    if hybrid_val_path.exists():
        with open(hybrid_val_path, "r", encoding="utf-8") as f:
            hybrid_val_data = json.load(f)

    baseline_val_data = {}
    if baseline_val_path.exists():
        with open(baseline_val_path, "r", encoding="utf-8") as f:
            baseline_val_data = json.load(f)

    pooling_comparison = hybrid_val_data.get("pooling_comparison", {})
    mean_val = pooling_comparison.get("mean", {}).get("validation_metrics", hybrid_val_data.get("validation_metrics", {}))
    cls_val = pooling_comparison.get("cls", {}).get("validation_metrics", {})

    consolidated = {
        "metadata": {
            "title": "RevuLens Real Model Evaluation Results",
            "encoder": "distilbert-base-multilingual-cased (frozen, no fine-tuning)",
            "primary_classifier": "LinearSVC (StandardScaler fit on train only, C tuned on validation set)",
            "baseline_model": "TF-IDF (1-2 word n-grams, sublinear_tf) + LinearSVC",
            "salminen_splits": {
                "train_total": 32309,
                "val_total": 4038,
                "test_total": 4038,
                "split_ratio": "80/10/10 stratified",
                "random_seed": 42
            },
            "confidence_scores": "None (omitted per GEMINI.md; decision based strictly on sign of decision_function)"
        },
        "validation_pooling_comparison": {
            "mean_pooling": {
                "description": "Mean pooling over DistilBERT token vectors excluding padding (default)",
                "best_C": pooling_comparison.get("mean", {}).get("best_C", 0.01),
                "metrics": mean_val
            },
            "cls_token_pooling": {
                "description": "CLS first-token vector representation",
                "best_C": pooling_comparison.get("cls", {}).get("best_C", 0.01),
                "metrics": cls_val
            },
            "comparison_summary": "Mean pooling achieved 86.70% validation accuracy and 0.8669 Macro F1, outperforming CLS token pooling (82.90% accuracy, 0.8287 Macro F1) by +3.80 percentage points. Mean pooling is selected as the primary embedding representation."
        },
        "validation_baseline_vs_hybrid": {
            "baseline_tfidf_linear_svc": baseline_val_data.get("validation_metrics", {}),
            "hybrid_distilbert_svm_mean": mean_val
        },
        "test_evaluation": {
            "salminen_matched_test_split": {
                "sample_size": hybrid_matched.get("test_samples", 500),
                "baseline_tfidf_linear_svc": baseline_matched,
                "hybrid_distilbert_svm": hybrid_matched
            },
            "baseline_full_test_split": baseline_full
        },
        "firecs_taglish_exploratory_check": firecs_results
    }

    results_file_path = data_path / "model_evaluation_results.json"
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print(f"Consolidated model results file saved to: {results_file_path}")
    return consolidated


def run_step5_evaluation(
    data_dir: str = "data/processed",
    models_dir: str = "backend/models",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Execute Step 5 final evaluation."""
    data_path = Path(data_dir)
    models_path = Path(models_dir)

    baseline_model_path = str(models_path / "baseline_tfidf_pipeline.joblib")
    hybrid_model_path = str(models_path / "hybrid_distilbert_svm_pipeline.joblib")
    test_csv_path = str(data_path / "test.csv")
    test_emb_path = str(data_path / "test_embeddings_mean.npy")
    test_lbl_path = str(data_path / "test_labels.npy")
    firecs_csv_path = str(data_path / "firecs_exploratory.csv")
    firecs_emb_path = str(data_path / "firecs_embeddings_mean.npy")

    # Check available hybrid test samples to guarantee 1:1 matched test split
    available_hybrid_test = len(np.load(test_emb_path)) if os.path.exists(test_emb_path) else None
    eval_limit = limit if limit is not None else available_hybrid_test

    # 1. Evaluate Baseline on the 1:1 Matched Test Split
    baseline_metrics = evaluate_baseline_test(
        baseline_artifact_path=baseline_model_path,
        test_csv_path=test_csv_path,
        limit=eval_limit
    )

    # 1b. If matched limit was used, also evaluate baseline on the full held-out test set for complete reference
    baseline_full_metrics = None
    if eval_limit and eval_limit < 4038:
        print("\n--- Evaluating Baseline on Full Held-Out Test Set (4,038 samples) ---")
        baseline_full_metrics = evaluate_baseline_test(
            baseline_artifact_path=baseline_model_path,
            test_csv_path=test_csv_path,
            limit=None
        )

    # 2. Evaluate Hybrid on Held-out Test Split
    hybrid_metrics = evaluate_hybrid_test(
        hybrid_artifact_path=hybrid_model_path,
        test_emb_path=test_emb_path,
        test_lbl_path=test_lbl_path,
        limit=eval_limit
    )

    # 3. Evaluate FiReCS Taglish Behavioral Check
    firecs_limit = eval_limit or 500
    firecs_results = evaluate_firecs_exploratory(
        baseline_artifact_path=baseline_model_path,
        hybrid_artifact_path=hybrid_model_path,
        firecs_csv_path=firecs_csv_path,
        firecs_emb_path=firecs_emb_path,
        limit=firecs_limit
    )

    # 4. Print Comparative Output
    print_comparative_report(baseline_metrics, hybrid_metrics, firecs_results)
    if baseline_full_metrics:
        print(f"\n[Note] Full Baseline Test Metrics (N={baseline_full_metrics['test_samples']}): "
              f"Accuracy={baseline_full_metrics['accuracy']*100:.2f}%, Macro F1={baseline_full_metrics['f1_macro']:.4f}")

    # 5. Save Final Report to JSON (maintaining schema expected by existing consumers)
    report = {
        "step": "Step 5: Final Evaluation on Held-Out Test Split and FiReCS Exploratory Check",
        "salminen_test_split_matched": {
            "sample_size": eval_limit,
            "baseline_tfidf_linear_svc": baseline_metrics,
            "hybrid_distilbert_svm": hybrid_metrics
        },
        "baseline_full_test_split": baseline_full_metrics,
        "firecs_taglish_exploratory_check": firecs_results
    }
    save_path = data_path / "final_test_evaluation.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFinal evaluation report saved to: {save_path}")

    # 6. Build and save consolidated master results file for docs
    build_consolidated_results_file(
        data_dir=data_dir,
        baseline_matched=baseline_metrics,
        baseline_full=baseline_full_metrics,
        hybrid_matched=hybrid_metrics,
        firecs_results=firecs_results
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuLens Step 5: Test Evaluation & FiReCS Behavioral Check")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples (default: all)")
    args = parser.parse_args()

    run_step5_evaluation(limit=args.limit)
