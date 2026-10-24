#!/usr/bin/env python3
"""
src/step3_persource.py — Per-Source Subpopulation Evaluation
Evaluates n, F1, FPR, FNR across sub-corpora:
Assassin, CEAS-08, Enron, TREC-05, TREC-06, TREC-07 for Arm A, LR, and SVM.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message

def compute_source_metrics(df, model_col, true_col="y_true"):
    y_t = df[true_col].values
    y_p = df[model_col].values

    f1 = float(f1_score(y_t, y_p, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "n": len(df),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def main():
    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    scores_path = os.path.join(output_dir, "armA_val_scores.csv")
    out_csv = os.path.join(output_dir, "per_source_metrics.csv")
    out_json = os.path.join(output_dir, "per_source_summary.json")

    log_message("=" * 60)
    log_message("STEP 3: PER-SOURCE GENERALIZATION BREAKDOWN")
    log_message("=" * 60)

    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing Arm A scores: {scores_path}")

    arma_df = pd.read_csv(scores_path)
    val_csv = os.path.join(kit_root, "data", "phishing_val.csv")
    val_df = pd.read_csv(val_csv)

    # Ensure dataset_name is present
    if "dataset_name" not in arma_df.columns:
        arma_df = pd.merge(arma_df, val_df[["email_id", "dataset_name"]], on="email_id", how="left")

    # Merge baselines
    lr_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_logisticregression_val.csv")
    svm_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_linearsvm_val.csv")

    combined = arma_df.copy().rename(columns={"y_pred": "pred_arma"})

    if os.path.exists(lr_csv):
        lr_df = pd.read_csv(lr_csv)
        combined = pd.merge(combined, lr_df[["email_id", "y_pred"]].rename(columns={"y_pred": "pred_lr"}), on="email_id", how="left")
    else:
        combined["pred_lr"] = np.nan

    if os.path.exists(svm_csv):
        svm_df = pd.read_csv(svm_csv)
        combined = pd.merge(combined, svm_df[["email_id", "y_pred"]].rename(columns={"y_pred": "pred_svm"}), on="email_id", how="left")
    else:
        combined["pred_svm"] = np.nan

    sources = sorted(combined["dataset_name"].dropna().unique().tolist())
    log_message(f"Detected sub-corpora: {sources}")

    records = []
    summary_dict = {}

    for src in sources:
        sub = combined[combined["dataset_name"] == src]
        res_arma = compute_source_metrics(sub, "pred_arma")
        res_lr = compute_source_metrics(sub, "pred_lr") if "pred_lr" in sub else None
        res_svm = compute_source_metrics(sub, "pred_svm") if "pred_svm" in sub else None

        records.append({
            "dataset_name": src,
            "n": len(sub),
            "arma_f1": res_arma["f1"],
            "arma_fpr": res_arma["fpr"],
            "arma_fnr": res_arma["fnr"],
            "lr_f1": res_lr["f1"] if res_lr else np.nan,
            "lr_fpr": res_lr["fpr"] if res_lr else np.nan,
            "lr_fnr": res_lr["fnr"] if res_lr else np.nan,
            "svm_f1": res_svm["f1"] if res_svm else np.nan,
            "svm_fpr": res_svm["fpr"] if res_svm else np.nan,
            "svm_fnr": res_svm["fnr"] if res_svm else np.nan
        })

        summary_dict[src] = {
            "n": len(sub),
            "Arm_A": res_arma,
            "Logistic_Regression": res_lr,
            "Linear_SVM": res_svm
        }

    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    log_message(f"Saved per-source metrics CSV: {out_csv}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    log_message(f"Saved per-source summary JSON: {out_json}")

    log_message("\n--- Per-Source Performance Matrix ---")
    log_message(res_df.to_string(index=False))
    log_message("\n")

if __name__ == "__main__":
    main()
