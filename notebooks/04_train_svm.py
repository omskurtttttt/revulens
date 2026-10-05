"""
Step 4: DistilBERT-SVM Hybrid Pipeline Training, Tuning & Validation Pooling Comparison.

Adheres strictly to GEMINI.md:
- Encoder: distilbert-base-multilingual-cased (frozen, never fine-tuned).
- Feature standardization: StandardScaler fit on TRAIN ONLY.
- Classifier: LinearSVC with C parameter tuned on validation set.
- Decision rule: The sign of decision_function decides the class.
- NO confidence score anywhere (no CalibratedClassifierCV, no predict_proba).
- Pooling comparison: Evaluates both Mean pooling (default) and CLS token pooling on the validation set.
- Saves pipeline artifact: backend/models/hybrid_distilbert_svm_pipeline.joblib (gitignored).
- Saves validation metrics: data/processed/hybrid_val_metrics.json (gitignored).
- Reports hybrid vs baseline comparison (hybrid is NOT assumed to be better).
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
    """Load pre-extracted embeddings and label arrays for a split."""
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate standard classification evaluation metrics (no confidence scores in production)."""
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


def train_and_tune_linear_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    pooling_name: str = "mean"
) -> Dict[str, Any]:
    """
    Standardize features (fit on train only per GEMINI.md) and tune LinearSVC C on validation set.
    Predictions are determined by the sign of decision_function (no calibration).
    """
    print(f"\n=======================================================")
    print(f"Training DistilBERT-SVM Pipeline (pooling={pooling_name})")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    print(f"Embedding dimension: {X_train.shape[1]}")
    print(f"=======================================================")

    # 1. Feature Standardization (FIT ON TRAIN ONLY per GEMINI.md)
    print("\n[1/3] Standardizing feature embeddings (StandardScaler fit on TRAIN only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 2. Hyperparameter Tuning on Validation Split
    print("\n[2/3] Tuning LinearSVC regularization parameter 'C' on Validation Set...")
    tuning_history: Dict[float, Dict[str, float]] = {}
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

    print(f"Optimal C selected for {pooling_name.upper()} pooling: {best_c} (Validation Macro F1: {best_f1:.4f})")

    # 3. Fit Best SVC (Uncalibrated LinearSVC per GEMINI.md)
    print("\n[3/3] Fitting final LinearSVC model on standardized train embeddings...")
    best_svc = LinearSVC(C=best_c, random_state=RANDOM_SEED, max_iter=3000, dual="auto")
    best_svc.fit(X_train_scaled, y_train)

    val_preds = best_svc.predict(X_val_scaled)
    val_decision = best_svc.decision_function(X_val_scaled)
    val_metrics = evaluate_predictions(y_val, val_preds)

    print("\n" + "=" * 55)
    print(f"HYBRID VALIDATION RESULTS ({pooling_name.upper()} Pooling + LinearSVC, C={best_c}):")
    print(f"  Accuracy:             {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score:       {val_metrics['f1_macro']:.4f}")
    print(f"  Deceptive Precision:  {val_metrics['precision_deceptive']:.4f}")
    print(f"  Deceptive Recall:     {val_metrics['recall_deceptive']:.4f}")
    print(f"  Deceptive F1-Score:   {val_metrics['f1_deceptive']:.4f}")
    print(f"  Decision Score Range: [{float(np.min(val_decision)):.4f}, {float(np.max(val_decision)):.4f}]")
    print(f"  Confusion Matrix:     TN={val_metrics['confusion_matrix'][0][0]}, FP={val_metrics['confusion_matrix'][0][1]}")
    print(f"                        FN={val_metrics['confusion_matrix'][1][0]}, TP={val_metrics['confusion_matrix'][1][1]}")
    print("=" * 55)

    return {
        "pooling": pooling_name,
        "scaler": scaler,
        "best_svc": best_svc,
        "best_c": best_c,
        "tuning_history": tuning_history,
        "val_metrics": val_metrics,
        "decision_scores_summary": {
            "min": round(float(np.min(val_decision)), 4),
            "max": round(float(np.max(val_decision)), 4),
            "mean": round(float(np.mean(val_decision)), 4),
            "std": round(float(np.std(val_decision)), 4),
        }
    }


def compare_validation_pooling(
    data_dir: str = "data/processed",
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare Mean Pooling vs. CLS Token Pooling on the validation set per GEMINI.md.
    Uses frozen embeddings and tunes C for each pooling strategy independently.
    """
    print("\n" + "=" * 65)
    print("VALIDATION POOLING COMPARISON (Mean vs. CLS Token Pooling)")
    print("GEMINI.md: Mean pooling is default; CLS pooling is compared & reported.")
    print("=" * 65)

    # 1. Mean Pooling Evaluation
    X_train_mean, y_train_mean = load_split_embeddings("train", pooling="mean", data_dir=data_dir)
    X_val_mean, y_val_mean = load_split_embeddings("val", pooling="mean", data_dir=data_dir)
    mean_results = train_and_tune_linear_svc(
        X_train=X_train_mean,
        y_train=y_train_mean,
        X_val=X_val_mean,
        y_val=y_val_mean,
        c_candidates=c_candidates,
        pooling_name="mean"
    )

    # 2. CLS Token Pooling Evaluation
    X_train_cls, y_train_cls = load_split_embeddings("train", pooling="cls", data_dir=data_dir)
    X_val_cls, y_val_cls = load_split_embeddings("val", pooling="cls", data_dir=data_dir)
    cls_results = train_and_tune_linear_svc(
        X_train=X_train_cls,
        y_train=y_train_cls,
        X_val=X_val_cls,
        y_val=y_val_cls,
        c_candidates=c_candidates,
        pooling_name="cls"
    )

    # 3. Print Side-by-Side Comparison Table
    m_vm = mean_results["val_metrics"]
    c_vm = cls_results["val_metrics"]
    acc_diff = (m_vm["accuracy"] - c_vm["accuracy"]) * 100
    f1_diff = m_vm["f1_macro"] - c_vm["f1_macro"]

    print("\n" + "=" * 68)
    print("POOLED REPRESENTATION VALIDATION COMPARISON (Salminen Validation Split)")
    print("=" * 68)
    print(f"{'Metric':<25} | {'Mean Pooling (Default)':<22} | {'CLS Token Pooling':<18}")
    print("-" * 68)
    print(f"{'Optimal C Parameter':<25} | {mean_results['best_c']:<22} | {cls_results['best_c']:<18}")
    print(f"{'Validation Accuracy':<25} | {m_vm['accuracy']*100:>20.2f}% | {c_vm['accuracy']*100:>16.2f}%")
    print(f"{'Macro F1-Score':<25} | {m_vm['f1_macro']:>22.4f} | {c_vm['f1_macro']:>18.4f}")
    print(f"{'Deceptive Precision':<25} | {m_vm['precision_deceptive']:>22.4f} | {c_vm['precision_deceptive']:>18.4f}")
    print(f"{'Deceptive Recall':<25} | {m_vm['recall_deceptive']:>22.4f} | {c_vm['recall_deceptive']:>18.4f}")
    print(f"{'Deceptive F1-Score':<25} | {m_vm['f1_deceptive']:>22.4f} | {c_vm['f1_deceptive']:>18.4f}")
    print(f"{'Confusion Matrix':<25} | TN={m_vm['confusion_matrix'][0][0]}, FP={m_vm['confusion_matrix'][0][1]:<12} | TN={c_vm['confusion_matrix'][0][0]}, FP={c_vm['confusion_matrix'][0][1]}")
    print(f"{'':<25} | FN={m_vm['confusion_matrix'][1][0]}, TP={m_vm['confusion_matrix'][1][1]:<12} | FN={c_vm['confusion_matrix'][1][0]}, TP={c_vm['confusion_matrix'][1][1]}")
    print("-" * 68)
    print(f"Summary: Mean pooling delta = {acc_diff:+.2f}% Accuracy, {f1_diff:+.4f} Macro F1 vs CLS.")
    print("=" * 68)

    return mean_results, cls_results


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
    print("=" * 65)


def run_hybrid_training_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "backend/models",
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Dict[str, Any]:
    """Execute Step 4: SVM classifier training, tuning, validation pooling comparison, and artifact export."""
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("RevuLens Step 4: DistilBERT-SVM Hybrid Pipeline (Training & Pooling Comparison)")
    print(f"Data directory:   {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Primary Pooling:  mean (excluding padding)")
    print(f"NOTE: Test split is strictly held out until Step 5 final evaluation.")
    print("=" * 65)

    # 1. Run validation pooling comparison (mean vs cls)
    mean_results, cls_results = compare_validation_pooling(
        data_dir=data_dir,
        c_candidates=c_candidates
    )

    # 2. Save full hybrid model artifact (primary: mean pooling)
    X_train_mean, _ = load_split_embeddings("train", pooling="mean", data_dir=data_dir)
    pipeline_artifact = {
        "model_type": "DistilBERT-SVM Hybrid (MEAN Pooling)",
        "encoder_checkpoint": "distilbert-base-multilingual-cased",
        "pooling_strategy": "mean",
        "feature_dim": X_train_mean.shape[1],
        "scaler": mean_results["scaler"],
        "classifier": mean_results["best_svc"],
        "raw_svc": mean_results["best_svc"],
        "best_C": mean_results["best_c"],
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X_train_mean),
        "val_metrics": mean_results["val_metrics"],
        "pooling_comparison": {
            "mean": {
                "best_C": mean_results["best_c"],
                "validation_metrics": mean_results["val_metrics"],
                "decision_scores_summary": mean_results["decision_scores_summary"]
            },
            "cls": {
                "best_C": cls_results["best_c"],
                "validation_metrics": cls_results["val_metrics"],
                "decision_scores_summary": cls_results["decision_scores_summary"]
            }
        }
    }
    model_save_path = models_path / "hybrid_distilbert_svm_pipeline.joblib"
    joblib.dump(pipeline_artifact, model_save_path)
    file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"\nSaved hybrid pipeline artifact to: {model_save_path} ({file_size_mb:.2f} MB)")

    # 3. Save validation metrics JSON
    metrics_save_path = data_path / "hybrid_val_metrics.json"
    summary_report = {
        "model": "DistilBERT-SVM Hybrid (MEAN Pooling)",
        "encoder": "distilbert-base-multilingual-cased",
        "pooling": "mean",
        "best_C": mean_results["best_c"],
        "tuning_history": {str(k): v for k, v in mean_results["tuning_history"].items()},
        "validation_metrics": mean_results["val_metrics"],
        "decision_scores_summary": mean_results["decision_scores_summary"],
        "pooling_comparison": {
            "mean": {
                "best_C": mean_results["best_c"],
                "validation_metrics": mean_results["val_metrics"]
            },
            "cls": {
                "best_C": cls_results["best_c"],
                "validation_metrics": cls_results["val_metrics"]
            },
            "delta_mean_minus_cls": {
                "accuracy": round(mean_results["val_metrics"]["accuracy"] - cls_results["val_metrics"]["accuracy"], 4),
                "f1_macro": round(mean_results["val_metrics"]["f1_macro"] - cls_results["val_metrics"]["f1_macro"], 4),
            }
        },
        "training_metadata": {
            "train_samples": len(X_train_mean),
            "val_samples": len(mean_results["scaler"].mean_),
            "feature_dim": X_train_mean.shape[1],
            "calibrated": False,
            "decision_rule": "sign of LinearSVC decision_function"
        }
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved validation metrics JSON to: {metrics_save_path}")

    # 4. Compare with TF-IDF baseline
    compare_with_baseline(mean_results["val_metrics"], data_dir=data_dir)

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RevuLens SVM Training, Tuning & Pooling Comparison (Step 4)")
    args = parser.parse_args()

    run_hybrid_training_pipeline()
