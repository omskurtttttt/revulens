"""
Step 3: TF-IDF + LinearSVC Baseline Pipeline.

Adheres strictly to GEMINI.md:
- Required baseline model: TF-IDF (1-2 word n-grams) + LinearSVC
- Evaluated on the exact same splits (Salminen 80/10/10, fixed seed 42)
- Test split is NOT touched until final evaluation (evaluation on validation split only)
- Vectorizer, LinearSVC, and calibrator fit on TRAIN ONLY
- Tuning of regularization parameter 'C' evaluated on the validation set
- Probability calibration via CalibratedClassifierCV for confidence scores
- Model artifacts saved to backend/models/baseline_tfidf_pipeline.joblib (gitignored)
- Baseline metrics saved to data/processed/baseline_val_metrics.json for comparative reporting
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.constants import InternalClass

RANDOM_SEED = 42


def load_dataset_split(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Load texts and binary labels (0 for Genuine, 1 for Deceptive) from split CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

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

    return texts, np.array(labels, dtype=np.int64)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
    """Calculate standard classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_binary = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_binary = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()

    avg_conf = None
    if y_proba is not None:
        # Confidence is the predicted class probability
        pred_confidences = np.max(y_proba, axis=1)
        avg_conf = float(np.mean(pred_confidences))

    return {
        "accuracy": round(float(acc), 4),
        "precision_deceptive": round(float(prec_binary), 4),
        "recall_deceptive": round(float(rec_binary), 4),
        "f1_deceptive": round(float(f1_binary), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm,
        "average_confidence": round(avg_conf, 4) if avg_conf is not None else None,
    }


def tune_linear_svc(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Tuple[float, Dict[float, Dict[str, Any]]]:
    """
    Tune LinearSVC regularization parameter 'C' on the validation set.
    Selects the candidate maximizing validation F1-score (macro).
    """
    print("\n--- Tuning LinearSVC 'C' on Validation Set ---")
    tuning_results = {}
    best_c = c_candidates[0]
    best_f1 = -1.0

    for c in c_candidates:
        svc = LinearSVC(C=c, random_state=RANDOM_SEED, max_iter=3000, dual="auto")
        svc.fit(X_train, y_train)

        val_preds = svc.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_preds)

        tuning_results[c] = {
            "val_accuracy": round(float(val_acc), 4),
            "val_f1_macro": round(float(val_f1), 4)
        }
        print(f"  C={c:<6} -> Val Accuracy: {val_acc:.4f} | Val F1 (Macro): {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c

    print(f"Optimal C selected: {best_c} (Val F1: {best_f1:.4f})")
    return best_c, tuning_results


def run_baseline_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "backend/models",
    max_features: int = 50000,
    c_candidates: List[float] = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Dict[str, Any]:
    """
    Execute Step 3 TF-IDF Baseline Training and Validation Pipeline.
    """
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    train_csv = data_path / "train.csv"
    val_csv = data_path / "val.csv"

    print("=" * 65)
    print("RevuLens Step 3: TF-IDF (1-2 n-grams) + LinearSVC Baseline")
    print(f"Train path: {train_csv}")
    print(f"Val path:   {val_csv}")
    print(f"NOTE: Test split is strictly held out until final evaluation.")
    print("=" * 65)

    # 1. Load data
    train_texts, y_train = load_dataset_split(str(train_csv))
    val_texts, y_val = load_dataset_split(str(val_csv))

    print(f"\nLoaded Train split: {len(train_texts)} reviews (Genuine: {np.sum(y_train == 0)}, Deceptive: {np.sum(y_train == 1)})")
    print(f"Loaded Val split:   {len(val_texts)} reviews (Genuine: {np.sum(y_val == 0)}, Deceptive: {np.sum(y_val == 1)})")

    # 2. Fit TF-IDF Vectorizer on TRAIN ONLY
    print("\nFitting TfidfVectorizer (ngram_range=(1, 2)) on TRAIN only...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2
    )
    X_train = vectorizer.fit_transform(train_texts)
    fit_time = time.time() - start_time
    print(f"TF-IDF fit complete in {fit_time:.2f}s. Vocabulary size: {len(vectorizer.vocabulary_)} features.")

    # Transform validation set using fitted vectorizer
    X_val = vectorizer.transform(val_texts)

    # 3. Tune LinearSVC 'C' on validation set
    best_c, tuning_history = tune_linear_svc(X_train, y_train, X_val, y_val, c_candidates=c_candidates)

    # 4. Train final CalibratedClassifierCV on train only
    print(f"\nTraining CalibratedClassifierCV with best LinearSVC (C={best_c}, 5-fold CV on train)...")
    calib_start = time.time()
    base_svc = LinearSVC(C=best_c, random_state=RANDOM_SEED, max_iter=3000, dual="auto")
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv=5)
    calibrated_clf.fit(X_train, y_train)
    calib_time = time.time() - calib_start
    print(f"Calibrated classifier trained in {calib_time:.2f}s.")

    # 5. Evaluate on Validation Set
    print("\nEvaluating Calibrated Baseline on Validation Set...")
    val_preds = calibrated_clf.predict(X_val)
    val_proba = calibrated_clf.predict_proba(X_val)
    val_metrics = evaluate_predictions(y_val, val_preds, val_proba)

    print("\n" + "=" * 50)
    print("BASELINE VALIDATION RESULTS (TF-IDF + Calibrated LinearSVC):")
    print(f"  Accuracy:             {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score:       {val_metrics['f1_macro']:.4f}")
    print(f"  Deceptive Precision:  {val_metrics['precision_deceptive']:.4f}")
    print(f"  Deceptive Recall:     {val_metrics['recall_deceptive']:.4f}")
    print(f"  Deceptive F1-Score:   {val_metrics['f1_deceptive']:.4f}")
    print(f"  Average Confidence:   {val_metrics['average_confidence']:.4f}")
    print(f"  Confusion Matrix:     TN={val_metrics['confusion_matrix'][0][0]}, FP={val_metrics['confusion_matrix'][0][1]}")
    print(f"                        FN={val_metrics['confusion_matrix'][1][0]}, TP={val_metrics['confusion_matrix'][1][1]}")
    print("=" * 50)

    # 6. Save Pipeline Artifacts
    pipeline_artifact = {
        "model_type": "TF-IDF + LinearSVC (Baseline)",
        "vectorizer": vectorizer,
        "classifier": calibrated_clf,
        "best_C": best_c,
        "ngram_range": [1, 2],
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": joblib.__version__,
        "train_samples": len(train_texts),
        "val_metrics": val_metrics
    }
    model_save_path = models_path / "baseline_tfidf_pipeline.joblib"
    joblib.dump(pipeline_artifact, model_save_path)
    file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"\nSaved baseline model artifact to: {model_save_path} ({file_size_mb:.2f} MB)")

    # Save validation metrics JSON
    metrics_save_path = data_path / "baseline_val_metrics.json"
    summary_report = {
        "model": "TF-IDF + LinearSVC (Baseline)",
        "best_C": best_c,
        "tuning_history": {str(k): v for k, v in tuning_history.items()},
        "validation_metrics": val_metrics,
        "training_metadata": {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "fit_time_seconds": round(fit_time, 2),
            "calibration_time_seconds": round(calib_time, 2)
        }
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved validation metrics JSON to: {metrics_save_path}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuLens TF-IDF Baseline (Step 3)")
    parser.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF features (default: 50000)")
    args = parser.parse_args()

    run_baseline_pipeline(max_features=args.max_features)
