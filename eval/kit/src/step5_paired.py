#!/usr/bin/env python3
"""
src/step5_paired.py — Paired Significance Tests & Bootstrap Confidence Intervals
Performs exact McNemar tests (Arm A vs LR, Arm A vs SVM) on identical validation rows.
Computes 95% Bootstrap Confidence Intervals (1,000 resamples, seed 42) for F1, FPR, AUROC,
and for the performance deltas (Arm A - LR).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message

def exact_mcnemar(y_true, y_pred_a, y_pred_b):
    """
    Computes exact two-sided McNemar test using binomial distribution.
    Contingency table on discordant predictions:
      b: Model A correct, Model B incorrect
      c: Model A incorrect, Model B correct
    """
    corr_a = (y_pred_a == y_true)
    corr_b = (y_pred_b == y_true)

    a = int((corr_a & corr_b).sum())      # both correct
    b = int((corr_a & (~corr_b)).sum())    # A correct, B wrong
    c = int(((~corr_a) & corr_b).sum())    # A wrong, B correct
    d = int(((~corr_a) & (~corr_b)).sum()) # both wrong

    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        try:
            from scipy.stats import binomtest
            res = binomtest(b, n_discordant, 0.5, alternative="two-sided")
            p_val = float(res.pvalue)
        except ImportError:
            from scipy.stats import binom_test
            p_val = float(binom_test(b, n_discordant, 0.5, alternative="two-sided"))

    return {
        "contingency": {"both_correct": a, "a_correct_b_wrong": b, "a_wrong_b_correct": c, "both_wrong": d},
        "discordant_total": n_discordant,
        "p_value": p_val,
        "statistically_significant_05": p_val < 0.05
    }

def compute_single_metrics(y_true, y_pred, scores):
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = np.nan
    return f1, fpr, auroc

def main():
    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    scores_path = os.path.join(output_dir, "armA_val_scores.csv")
    out_json = os.path.join(output_dir, "paired_tests.json")
    out_csv = os.path.join(output_dir, "bootstrap_cis.csv")

    log_message("=" * 60)
    log_message("STEP 5: PAIRED SIGNIFICANCE TESTS & 95% BOOTSTRAP CIs")
    log_message("=" * 60)

    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing Arm A scores: {scores_path}")

    arma_df = pd.read_csv(scores_path)
    eval_ids = set(arma_df["email_id"].tolist())

    lr_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_logisticregression_val.csv")
    svm_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_linearsvm_val.csv")

    if not os.path.exists(lr_csv) or not os.path.exists(svm_csv):
        raise FileNotFoundError("Missing baseline predictions for paired test!")

    lr_df = pd.read_csv(lr_csv)
    svm_df = pd.read_csv(svm_csv)

    # Filter to identical evaluated emails
    df = arma_df[["email_id", "y_true", "y_pred", "score"]].rename(
        columns={"y_pred": "pred_arma", "score": "score_arma"}
    )
    df = pd.merge(df, lr_df[["email_id", "y_pred", "score"]].rename(
        columns={"y_pred": "pred_lr", "score": "score_lr"}
    ), on="email_id")
    df = pd.merge(df, svm_df[["email_id", "y_pred", "score"]].rename(
        columns={"y_pred": "pred_svm", "score": "score_svm"}
    ), on="email_id")

    log_message(f"Running paired analysis on {len(df)} identical validation emails...")

    # 1. Exact McNemar Tests
    mcnemar_lr = exact_mcnemar(df["y_true"].values, df["pred_arma"].values, df["pred_lr"].values)
    mcnemar_svm = exact_mcnemar(df["y_true"].values, df["pred_arma"].values, df["pred_svm"].values)

    log_message("\n--- Exact McNemar Tests ---")
    log_message(f"Arm A vs Logistic Regression:")
    log_message(f"  Arm A correct & LR wrong : {mcnemar_lr['contingency']['a_correct_b_wrong']}")
    log_message(f"  Arm A wrong & LR correct : {mcnemar_lr['contingency']['a_wrong_b_correct']}")
    log_message(f"  Two-sided p-value        : {mcnemar_lr['p_value']:.4e} (Significant: {mcnemar_lr['statistically_significant_05']})")

    log_message(f"\nArm A vs Linear SVM:")
    log_message(f"  Arm A correct & SVM wrong: {mcnemar_svm['contingency']['a_correct_b_wrong']}")
    log_message(f"  Arm A wrong & SVM correct: {mcnemar_svm['contingency']['a_wrong_b_correct']}")
    log_message(f"  Two-sided p-value        : {mcnemar_svm['p_value']:.4e} (Significant: {mcnemar_svm['statistically_significant_05']})")

    # 2. 95% Bootstrap Confidence Intervals (1,000 resamples, seed 42)
    n_boot = 1000
    rng = np.random.RandomState(42)
    n_rows = len(df)

    log_message(f"\nComputing 95% Bootstrap Confidence Intervals ({n_boot} iterations, seed 42)...")

    boot_arma_f1, boot_arma_fpr, boot_arma_auc = [], [], []
    boot_lr_f1, boot_lr_fpr, boot_lr_auc = [], [], []
    boot_svm_f1, boot_svm_fpr, boot_svm_auc = [], [], []

    delta_f1, delta_fpr, delta_auc = [], [], []

    y_t = df["y_true"].values
    p_a = df["pred_arma"].values
    s_a = df["score_arma"].values
    p_lr = df["pred_lr"].values
    s_lr = df["score_lr"].values
    p_svm = df["pred_svm"].values
    s_svm = df["score_svm"].values

    for _ in range(n_boot):
        idx = rng.randint(0, n_rows, size=n_rows)
        # Check that both classes are present
        if len(np.unique(y_t[idx])) < 2:
            continue

        f1_a, fpr_a, auc_a = compute_single_metrics(y_t[idx], p_a[idx], s_a[idx])
        f1_l, fpr_l, auc_l = compute_single_metrics(y_t[idx], p_lr[idx], s_lr[idx])
        f1_s, fpr_s, auc_s = compute_single_metrics(y_t[idx], p_svm[idx], s_svm[idx])

        boot_arma_f1.append(f1_a)
        boot_arma_fpr.append(fpr_a)
        boot_arma_auc.append(auc_a)

        boot_lr_f1.append(f1_l)
        boot_lr_fpr.append(fpr_l)
        boot_lr_auc.append(auc_l)

        boot_svm_f1.append(f1_s)
        boot_svm_fpr.append(fpr_s)
        boot_svm_auc.append(auc_s)

        delta_f1.append(f1_a - f1_l)
        delta_fpr.append(fpr_a - fpr_l)
        delta_auc.append(auc_a - auc_l)

    def ci_stats(arr, name):
        clean = [x for x in arr if not np.isnan(x)]
        return {
            "metric": name,
            "mean": round(float(np.mean(clean)), 4),
            "median": round(float(np.median(clean)), 4),
            "ci_lower_2.5": round(float(np.percentile(clean, 2.5)), 4),
            "ci_upper_97.5": round(float(np.percentile(clean, 97.5)), 4)
        }

    ci_records = [
        ci_stats(boot_arma_f1, "Arm_A_F1"),
        ci_stats(boot_arma_fpr, "Arm_A_FPR"),
        ci_stats(boot_arma_auc, "Arm_A_AUROC"),
        ci_stats(boot_lr_f1, "LR_F1"),
        ci_stats(boot_lr_fpr, "LR_FPR"),
        ci_stats(boot_lr_auc, "LR_AUROC"),
        ci_stats(boot_svm_f1, "SVM_F1"),
        ci_stats(boot_svm_fpr, "SVM_FPR"),
        ci_stats(boot_svm_auc, "SVM_AUROC"),
        ci_stats(delta_f1, "Delta_F1_(Arm_A_minus_LR)"),
        ci_stats(delta_fpr, "Delta_FPR_(Arm_A_minus_LR)"),
        ci_stats(delta_auc, "Delta_AUROC_(Arm_A_minus_LR)"),
    ]

    ci_df = pd.DataFrame(ci_records)
    ci_df.to_csv(out_csv, index=False)
    log_message(f"Saved Bootstrap CIs to: {out_csv}")

    paired_summary = {
        "n_evaluated": len(df),
        "mcnemar_armA_vs_lr": mcnemar_lr,
        "mcnemar_armA_vs_svm": mcnemar_svm,
        "bootstrap_cis": {rec["metric"]: rec for rec in ci_records}
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(paired_summary, f, indent=2)
    log_message(f"Saved Paired Tests JSON to: {out_json}")

    log_message("\n--- Bootstrap 95% Confidence Intervals ---")
    log_message(ci_df.to_string(index=False))
    log_message("\n")

if __name__ == "__main__":
    main()
