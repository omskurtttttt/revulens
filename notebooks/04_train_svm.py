"""
Step 4: SVM Classifier Training, Hyperparameter Tuning & Calibration.

Adheres strictly to GEMINI.md:
- Classifier: SVM on DistilBERT frozen embeddings
- Default model: LinearSVC
- Feature standardization: StandardScaler fit on TRAIN ONLY
- Hyperparameter tuning: C parameter tuned on validation set
- Probability calibration: CalibratedClassifierCV for confidence scores
- Pooling comparison: Supports both mean pooling (default) and CLS token pooling
- Saves pipeline artifact: backend/models/hybrid_distilbert_svm_pipeline.joblib (gitignored)
- Saves validation metrics: data/processed/hybrid_val_metrics.json (gitignored)
- Reports hybrid vs baseline comparison (hybrid is NOT assumed to be better)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

RANDOM_SEED = 42


def load_split_embeddings(
    split_name: str,
    pooling: str = "mean",
    data_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted embeddings and label arrays for a split.
    """
    data_path = Path(data_dir)
    emb_file = data_path / f"{split_name}_embeddings_{pooling}.npy"
    lbl_file = data_path / f"{split_name}_labels.npy"

    if not emb_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_file}")
    if not lbl_file.exists():
        raise FileNotFoundError(f"Labels file not found: {lbl_file}")

    embeddings = np.load(emb_file)
    labels = np.load(lbl_file)

    if len(embeddings) != len(labels):
        raise ValueError(f"Mismatch: {emb_file.name} has {len(embeddings)} rows but {lbl_file.name} has {len(labels)} labels")

    return embeddings, labels


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


def train_and_tune_hybrid_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    pooling_name: str = "mean"
) -> Dict[str, Any]:
    """
    Standardize features (fit on train only), tune LinearSVC C on validation set,
    and calibrate with CalibratedClassifierCV.
    """
    print(f"\n=======================================================")
    print(f"Training DistilBERT-SVM Pipeline (pooling={pooling_name})")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    print(f"Embedding dimension: {X_train.shape[1]}")
    print(f"=======================================================")

    # 1. Feature Standardization (FIT ON TRAIN ONLY per GEMINI.md)
    print("\n[1/4] Standardizing feature embeddings (StandardScaler fit on TRAIN only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 2. Hyperparameter Tuning on Validation Split
    print("\n[2/4] Tuning LinearSVC regularization parameter 'C' on Validation Set...")
    tuning_history = {}
    best_c = c_candidates[0]
    best_f1 = -1.0

    for c in c_candidates:
        svc = LinearSVC(C=c, random_state=RANDOM_SEED, max_iter=3000, dual="auto")
        svc.fit(X_train_scaled, y_train)

        val_preds = svc.predict(X_val_scaled)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_preds)

        tuning_history[c] = {
            "val_accuracy": round(float(val_acc), 4),
            "val_f1_macro": round(float(val_f1), 4)
        }
        print(f"  C={c:<6} -> Val Accuracy: {val_acc:.4f} | Val F1 (Macro): {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c

    print(f"Optimal C selected: {best_c} (Validation F1 Macro: {best_f1:.4f})")

    # 3. Fit Raw Best SVC
    best_svc = LinearSVC(C=best_c, random_state=RANDOM_SEED, max_iter=3000, dual="auto")
    best_svc.fit(X_train_scaled, y_train)

    # 4. Wrap with CalibratedClassifierCV (fit on train only per GEMINI.md)
    print(f"\n[3/4] Wrapping with CalibratedClassifierCV on train...")
    calib_start = time.time()
    min_class_count = min(int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))
    cv_folds = min(5, max(2, min_class_count))

    calibrated_clf = CalibratedClassifierCV(
        estimator=LinearSVC(C=best_c, random_state=RANDOM_SEED, max_iter=3000, dual="auto"),
        cv=cv_folds
    )
    calibrated_clf.fit(X_train_scaled, y_train)
    calib_time = time.time() - calib_start
    print(f"Calibrated classifier trained in {calib_time:.2f}s ({cv_folds}-fold CV).")

    # 5. Evaluate on Validation Set
    print("\n[4/4] Evaluating Calibrated Hybrid Pipeline on Validation Set...")
    val_preds = calibrated_clf.predict(X_val_scaled)
    val_proba = calibrated_clf.predict_proba(X_val_scaled)
    val_metrics = evaluate_predictions(y_val, val_preds, val_proba)

    print("\n" + "=" * 55)
    print(f"HYBRID VALIDATION RESULTS ({pooling_name.upper()} Pooling + SVM):")
    print(f"  Accuracy:             {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score:       {val_metrics['f1_macro']:.4f}")
    print(f"  Deceptive Precision:  {val_metrics['precision_deceptive']:.4f}")
    print(f"  Deceptive Recall:     {val_metrics['recall_deceptive']:.4f}")
    print(f"  Deceptive F1-Score:   {val_metrics['f1_deceptive']:.4f}")
    print(f"  Average Confidence:   {val_metrics['average_confidence']:.4f}")
    print(f"  Confusion Matrix:     TN={val_metrics['confusion_matrix'][0][0]}, FP={val_metrics['confusion_matrix'][0][1]}")
    print(f"                        FN={val_metrics['confusion_matrix'][1][0]}, TP={val_metrics['confusion_matrix'][1][1]}")
    print("=" * 55)

    return {
        "pooling": pooling_name,
        "scaler": scaler,
        "best_svc": best_svc,
        "calibrated_clf": calibrated_clf,
        "best_c": best_c,
        "tuning_history": tuning_history,
        "val_metrics": val_metrics,
        "calibration_time_seconds": round(calib_time, 2)
    }


def compare_with_baseline(hybrid_metrics: Dict[str, Any], data_dir: str = "data/processed") -> None:
    """Print comparative table between Step 3 TF-IDF Baseline and Step 4 DistilBERT-SVM Hybrid."""
    baseline_path = Path(data_dir) / "baseline_val_metrics.json"
    if not baseline_path.exists():
        print("No baseline metrics found to compare.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    b_val = baseline.get("validation_metrics", {})
    h_val = hybrid_metrics

    print("\n" + "=" * 65)
    print("COMPARATIVE EVALUATION: BASELINE vs HYBRID (Validation Set)")
    print("Per GEMINI.md: The hybrid is NOT assumed to be better; report both.")
    print("=" * 65)
    print(f"{'Metric':<25} | {'TF-IDF + LinearSVC':<18} | {'DistilBERT + SVM':<18}")
    print("-" * 65)
    print(f"{'Validation Accuracy':<25} | {b_val.get('accuracy', 0)*100:>16.2f}% | {h_val.get('accuracy', 0)*100:>16.2f}%")
    print(f"{'Macro F1-Score':<25} | {b_val.get('f1_macro', 0):>18.4f} | {h_val.get('f1_macro', 0):>18.4f}")
    print(f"{'Deceptive Precision':<25} | {b_val.get('precision_deceptive', 0):>18.4f} | {h_val.get('precision_deceptive', 0):>18.4f}")
    print(f"{'Deceptive Recall':<25} | {b_val.get('recall_deceptive', 0):>18.4f} | {h_val.get('recall_deceptive', 0):>18.4f}")
    print(f"{'Deceptive F1-Score':<25} | {b_val.get('f1_deceptive', 0):>18.4f} | {h_val.get('f1_deceptive', 0):>18.4f}")
    print(f"{'Average Confidence':<25} | {b_val.get('average_confidence', 0):>18.4f} | {h_val.get('average_confidence', 0):>18.4f}")
    print("=" * 65)


def run_hybrid_training_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "backend/models",
    pooling: str = "mean",
    c_candidates: List[float] = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Dict[str, Any]:
    """Execute Step 4: SVM classifier training, tuning, and calibration."""
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("RevuLens Step 4: DistilBERT-SVM Hybrid Pipeline (Training & Tuning)")
    print(f"Data directory:   {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Primary Pooling:  {pooling}")
    print(f"NOTE: Test split is strictly held out until Step 5 final evaluation.")
    print("=" * 65)

    # 1. Load pre-extracted embeddings for train and val
    X_train, y_train = load_split_embeddings("train", pooling=pooling, data_dir=data_dir)
    X_val, y_val = load_split_embeddings("val", pooling=pooling, data_dir=data_dir)

    # 2. Train, tune C, and calibrate on validation split
    results = train_and_tune_hybrid_svm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        c_candidates=c_candidates,
        pooling_name=pooling
    )

    # 3. Save full hybrid model artifact
    pipeline_artifact = {
        "model_type": f"DistilBERT-SVM Hybrid ({pooling.upper()} Pooling)",
        "encoder_checkpoint": "distilbert-base-multilingual-cased",
        "pooling_strategy": pooling,
        "feature_dim": X_train.shape[1],
        "scaler": results["scaler"],
        "classifier": results["calibrated_clf"],
        "raw_svc": results["best_svc"],
        "best_C": results["best_c"],
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X_train),
        "val_metrics": results["val_metrics"]
    }
    model_save_path = models_path / "hybrid_distilbert_svm_pipeline.joblib"
    joblib.dump(pipeline_artifact, model_save_path)
    file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"\nSaved hybrid pipeline artifact to: {model_save_path} ({file_size_mb:.2f} MB)")

    # 4. Save validation metrics JSON
    metrics_save_path = data_path / "hybrid_val_metrics.json"
    summary_report = {
        "model": f"DistilBERT-SVM Hybrid ({pooling.upper()} Pooling)",
        "encoder": "distilbert-base-multilingual-cased",
        "pooling": pooling,
        "best_C": results["best_c"],
        "tuning_history": {str(k): v for k, v in results["tuning_history"].items()},
        "validation_metrics": results["val_metrics"],
        "training_metadata": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "feature_dim": X_train.shape[1],
            "calibration_time_seconds": results["calibration_time_seconds"]
        }
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved validation metrics JSON to: {metrics_save_path}")

    # 5. Compare with TF-IDF baseline
    compare_with_baseline(results["val_metrics"], data_dir=data_dir)

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuLens SVM Training & Tuning (Step 4)")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy (default: mean)")
    args = parser.parse_args()

    run_hybrid_training_pipeline(pooling=args.pooling)
