#!/usr/bin/env python3
"""
src/step2_metrics.py — Metric Computation for Arm A, TF-IDF LR, and Linear SVM
Computes: Accuracy, Precision, Recall, F1, False Positive Rate (FPR), False Negative Rate (FNR),
Continuous AUROC (from probabilities/scores), Hard-Label AUROC (= balanced accuracy),
and Confusion Matrix (TN, FP, FN, TP).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message

def compute_all_metrics(y_true, y_pred, scores, model_name: str) -> dict:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Continuous AUROC from real-valued decision scores / probabilities
    auroc_cont = None
    if scores is not None and len(scores) == len(y_t):
        try:
            auroc_cont = float(roc_auc_score(y_t, scores))
        except Exception:
            auroc_cont = None

    # Hard-label AUROC (= balanced accuracy = 0.5 * (recall + (1 - FPR)))
    # This was the definition used in the earlier report (0.9916)
    hard_label_auroc = float(balanced_accuracy_score(y_t, y_p))

    return {
        "model": model_name,
        "n_samples": len(y_t),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "auroc_continuous": round(auroc_cont, 4) if auroc_cont is not None else np.nan,
        "hard_label_auroc_balanced_acc": round(hard_label_auroc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def main():
    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    scores_path = os.path.join(output_dir, "armA_val_scores.csv")
    table_out_path = os.path.join(output_dir, "metrics_table.csv")
    json_out_path = os.path.join(output_dir, "metrics_summary.json")

    log_message("=" * 60)
    log_message("STEP 2: VALIDATION METRICS & COMPARATIVE ANALYSIS")
    log_message("=" * 60)

    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing Arm A scores file: {scores_path}. Run step1_score.py first.")

    arma_df = pd.read_csv(scores_path)
    evaluated_ids = set(arma_df["email_id"].tolist())
    log_message(f"Arm A evaluated emails: {len(arma_df)}")

    # 1. Arm A Metrics
    arma_metrics = compute_all_metrics(
        arma_df["y_true"], arma_df["y_pred"], arma_df["score"],
        model_name="Arm A (Qwen2.5-1.5B Naive QLoRA)"
    )

    all_metrics = [arma_metrics]

    # 2. Baseline Models on the EXACT same evaluated email_ids
    lr_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_logisticregression_val.csv")
    svm_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_linearsvm_val.csv")

    if os.path.exists(lr_csv):
        lr_df = pd.read_csv(lr_csv)
        lr_matched = lr_df[lr_df["email_id"].isin(evaluated_ids)].sort_values("email_id")
        lr_metrics = compute_all_metrics(
            lr_matched["y_true"], lr_matched["y_pred"], lr_matched["score"],
            model_name="TF-IDF + Logistic Regression"
        )
        all_metrics.append(lr_metrics)
        log_message(f"Computed TF-IDF Logistic Regression metrics on {len(lr_matched)} identical emails.")

    if os.path.exists(svm_csv):
        svm_df = pd.read_csv(svm_csv)
        svm_matched = svm_df[svm_df["email_id"].isin(evaluated_ids)].sort_values("email_id")
        svm_metrics = compute_all_metrics(
            svm_matched["y_true"], svm_matched["y_pred"], svm_matched["score"],
            model_name="TF-IDF + Linear SVM"
        )
        all_metrics.append(svm_metrics)
        log_message(f"Computed TF-IDF Linear SVM metrics on {len(svm_matched)} identical emails.")

    # 3. Save metrics table CSV & JSON
    summary_rows = []
    for m in all_metrics:
        cm = m["confusion_matrix"]
        row = {
            "model": m["model"],
            "n_samples": m["n_samples"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"],
            "auroc_continuous": m["auroc_continuous"],
            "hard_label_auroc_balanced_acc": m["hard_label_auroc_balanced_acc"],
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"]
        }
        summary_rows.append(row)

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(table_out_path, index=False)
    log_message(f"Saved metrics table to: {table_out_path}")

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    log_message(f"Saved metrics summary JSON to: {json_out_path}")

    # Display comparison
    log_message("\n--- Summary Performance Table ---")
    log_message(metrics_df.to_string(index=False))
    log_message("\n")

if __name__ == "__main__":
    main()
