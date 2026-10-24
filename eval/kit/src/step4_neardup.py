#!/usr/bin/env python3
"""
src/step4_neardup.py — Train-to-Validation Near-Duplicate Leakage Analysis
Fits TF-IDF vectorizer on training emails ONLY.
Computes maximum cosine similarity of each validation email to any training email.
Evaluates performance of Arm A, LR, and SVM below vs at-or-above thresholds (0.8 and 0.9).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message

def combine_text(df):
    return (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

def compute_subset_metrics(y_true, y_pred):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    if len(y_t) == 0:
        return {"n": 0, "f1": np.nan, "fpr": np.nan, "fnr": np.nan}
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return {"n": len(y_t), "f1": round(f1, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4)}

def main():
    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    scores_path = os.path.join(output_dir, "armA_val_scores.csv")
    out_csv = os.path.join(output_dir, "near_duplicate_table.csv")
    out_json = os.path.join(output_dir, "near_duplicate_analysis.json")

    log_message("=" * 60)
    log_message("STEP 4: NEAR-DUPLICATE LEAKAGE & ROBUSTNESS ANALYSIS")
    log_message("=" * 60)

    train_path = os.path.join(kit_root, "data", "phishing_train.csv")
    val_path = os.path.join(kit_root, "data", "phishing_val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Missing train or val dataset in data/")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing Arm A scores: {scores_path}")

    log_message("Loading train and validation datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    arma_df = pd.read_csv(scores_path)

    # Align with evaluated emails
    eval_ids = set(arma_df["email_id"].tolist())
    val_sub = val_df[val_df["email_id"].isin(eval_ids)].copy().reset_index(drop=True)
    val_sub = pd.merge(val_sub, arma_df[["email_id", "y_pred"]].rename(columns={"y_pred": "pred_arma"}), on="email_id")

    # Load baseline predictions if available
    lr_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_logisticregression_val.csv")
    svm_csv = os.path.join(kit_root, "data", "preds_tf-idf_plus_linearsvm_val.csv")
    if os.path.exists(lr_csv):
        lr_df = pd.read_csv(lr_csv)
        val_sub = pd.merge(val_sub, lr_df[["email_id", "y_pred"]].rename(columns={"y_pred": "pred_lr"}), on="email_id", how="left")
    else:
        val_sub["pred_lr"] = np.nan

    if os.path.exists(svm_csv):
        svm_df = pd.read_csv(svm_csv)
        val_sub = pd.merge(val_sub, svm_df[["email_id", "y_pred"]].rename(columns={"y_pred": "pred_svm"}), on="email_id", how="left")
    else:
        val_sub["pred_svm"] = np.nan

    # 1. Fit TF-IDF on train text ONLY
    log_message(f"Fitting TF-IDF vectorizer on {len(train_df)} training emails...")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(combine_text(train_df))
    X_val = vec.transform(combine_text(val_sub))
    log_message(f"TF-IDF fitted and transformed in {time.time() - t0:.2f} s")

    # 2. Compute Max Cosine Similarity for each val email to ANY train email
    log_message("Computing maximum cosine similarity to train emails for each validation email...")
    max_sims = []
    # Block processing for memory efficiency
    block_size = 500
    for i in range(0, X_val.shape[0], block_size):
        end = min(i + block_size, X_val.shape[0])
        sim_block = X_val[i:end].dot(X_train.T)
        block_max = sim_block.max(axis=1).toarray().flatten()
        max_sims.extend(block_max.tolist())

    val_sub["max_train_sim"] = max_sims

    # 3. Analyze at Thresholds 0.8 and 0.9
    thresholds = [0.8, 0.9]
    analysis_records = []
    summary_dict = {
        "n_evaluated": len(val_sub),
        "mean_max_train_sim": round(float(np.mean(max_sims)), 4),
        "median_max_train_sim": round(float(np.median(max_sims)), 4),
        "thresholds": {}
    }

    log_message("\n--- Near-Duplicate Analysis Summary ---")
    log_message(f"Overall Median Max Similarity: {summary_dict['median_max_train_sim']:.4f}")

    for th in thresholds:
        near_twins = val_sub["max_train_sim"] >= th
        count_near = int(near_twins.sum())
        count_novel = len(val_sub) - count_near
        pct_near = (count_near / len(val_sub)) * 100.0

        sub_near = val_sub[near_twins]
        sub_novel = val_sub[~near_twins]

        # Performance on novel vs near-duplicate
        m_arma_novel = compute_subset_metrics(sub_novel["label"], sub_novel["pred_arma"])
        m_arma_near = compute_subset_metrics(sub_near["label"], sub_near["pred_arma"])

        m_lr_novel = compute_subset_metrics(sub_novel["label"], sub_novel["pred_lr"])
        m_lr_near = compute_subset_metrics(sub_near["label"], sub_near["pred_lr"])

        m_svm_novel = compute_subset_metrics(sub_novel["label"], sub_novel["pred_svm"])
        m_svm_near = compute_subset_metrics(sub_near["label"], sub_near["pred_svm"])

        log_message(f"\nThreshold >= {th}: {count_near} near-twins ({pct_near:.2f}%), {count_novel} novel emails ({100-pct_near:.2f}%)")
        log_message(f"  Novel subset (< {th}): Arm A F1={m_arma_novel['f1']}, LR F1={m_lr_novel['f1']}, SVM F1={m_svm_novel['f1']}")
        log_message(f"  Near-twin subset (>= {th}): Arm A F1={m_arma_near['f1']}, LR F1={m_lr_near['f1']}, SVM F1={m_svm_near['f1']}")

        analysis_records.append({
            "threshold": th,
            "subset": f"Novel (< {th})",
            "n": count_novel,
            "pct": round(100.0 - pct_near, 2),
            "arma_f1": m_arma_novel["f1"],
            "arma_fpr": m_arma_novel["fpr"],
            "arma_fnr": m_arma_novel["fnr"],
            "lr_f1": m_lr_novel["f1"],
            "lr_fpr": m_lr_novel["fpr"],
            "lr_fnr": m_lr_novel["fnr"],
            "svm_f1": m_svm_novel["f1"],
            "svm_fpr": m_svm_novel["fpr"],
            "svm_fnr": m_svm_novel["fnr"]
        })

        analysis_records.append({
            "threshold": th,
            "subset": f"Near-Twins (>= {th})",
            "n": count_near,
            "pct": round(pct_near, 2),
            "arma_f1": m_arma_near["f1"],
            "arma_fpr": m_arma_near["fpr"],
            "arma_fnr": m_arma_near["fnr"],
            "lr_f1": m_lr_near["f1"],
            "lr_fpr": m_lr_near["fpr"],
            "lr_fnr": m_lr_near["fnr"],
            "svm_f1": m_svm_near["f1"],
            "svm_fpr": m_svm_near["fpr"],
            "svm_fnr": m_svm_near["fnr"]
        })

        summary_dict["thresholds"][str(th)] = {
            "near_twin_count": count_near,
            "near_twin_pct": round(pct_near, 2),
            "novel_count": count_novel,
            "performance": {
                "novel": {"Arm_A": m_arma_novel, "LR": m_lr_novel, "SVM": m_svm_novel},
                "near_twins": {"Arm_A": m_arma_near, "LR": m_lr_near, "SVM": m_svm_near}
            }
        }

    res_df = pd.DataFrame(analysis_records)
    res_df.to_csv(out_csv, index=False)
    log_message(f"\nSaved near-duplicate table to: {out_csv}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    log_message(f"Saved near-duplicate analysis JSON to: {out_json}\n")

if __name__ == "__main__":
    main()
