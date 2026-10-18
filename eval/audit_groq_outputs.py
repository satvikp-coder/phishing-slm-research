import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def audit_groq_run(val_path: str, preds_path: str = None, cache_dir: str = None, raw_output_col: str = "raw_output"):
    print("============================================================")
    print("=== GROQ ZERO-SHOT AUDIT & METADATA FORENSICS ===")
    print("============================================================\n")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_path}")

    val_df = pd.read_csv(val_path)
    val_map = dict(zip(val_df['email_id'], val_df['label']))

    records = []
    exact_model_id = None

    # Case 1: Cache directory with per-email JSONs
    if cache_dir and os.path.exists(cache_dir):
        print(f"Reading cached API responses from: {cache_dir}")
        for fname in os.listdir(cache_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(cache_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    eid = int(data.get("email_id", fname.replace(".json", "")))
                    raw_text = data.get("output", data.get("raw_text", ""))
                    if exact_model_id is None and "model" in data:
                        exact_model_id = data["model"]
                    records.append({"email_id": eid, "raw_output": raw_text})
                except Exception:
                    pass
        df_preds = pd.DataFrame(records)
    elif preds_path and os.path.exists(preds_path):
        print(f"Reading predictions file from: {preds_path}")
        df_preds = pd.read_csv(preds_path)
    else:
        print(f"CRITICAL: Neither preds_path ({preds_path}) nor cache_dir ({cache_dir}) was found!")
        print("Please provide the path to the Colab Drive predictions CSV or cache directory.")
        return

    print(f"Loaded {len(df_preds)} predictions. Aligning with ground truth from {val_path}...")
    df_preds['y_true'] = df_preds['email_id'].map(val_map)
    df_preds = df_preds.dropna(subset=['y_true']).copy()
    df_preds['y_true'] = df_preds['y_true'].astype(int)

    # Parsing logic
    def parse_text(text):
        if not isinstance(text, str):
            return -1
        t = text.strip().lower()
        if t == "phishing":
            return 1
        elif t == "legitimate":
            return 0
        has_p = "phishing" in t
        has_l = "legitimate" in t
        if has_p and not has_l:
            return 1
        if has_l and not has_p:
            return 0
        return -1

    if raw_output_col in df_preds.columns:
        df_preds['y_parsed'] = df_preds[raw_output_col].apply(parse_text)
    elif 'y_pred' in df_preds.columns:
        df_preds['y_parsed'] = df_preds['y_pred'].astype(int)
    else:
        raise ValueError("Predictions file must contain either raw_output or y_pred column!")

    unparseable_mask = (df_preds['y_parsed'] == -1)
    unparseable_count = int(unparseable_mask.sum())
    total_eval = len(df_preds)

    print(f"\n--- Model & Execution Metadata ---")
    print(f"Exact Model ID (from API metadata): {exact_model_id if exact_model_id else 'Not recorded in metadata'}")
    print(f"Evaluated Emails:                   {total_eval}")
    print(f"Unparseable Count:                  {unparseable_count} ({unparseable_count/total_eval*100:.2f}%)")

    # (a) Unparseable excluded and counted separately
    df_clean = df_preds[~unparseable_mask].copy()
    y_true_a = df_clean['y_true'].values
    y_pred_a = df_clean['y_parsed'].values

    if len(df_clean) > 0:
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1]).ravel()
        print("\n============================================================")
        print("=== EVALUATION (A): UNPARSEABLE EXCLUDED & COUNTED SEPARATELY ===")
        print("============================================================")
        print(f"Evaluated subset (n={len(df_clean)}):")
        print(f"  TN: {tn_a} | FP: {fp_a} | FN: {fn_a} | TP: {tp_a}")
        print(f"  Accuracy:  {accuracy_score(y_true_a, y_pred_a):.4f}")
        print(f"  Precision: {precision_score(y_true_a, y_pred_a, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_true_a, y_pred_a, zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(y_true_a, y_pred_a, zero_division=0):.4f}")
        print(f"  FPR:       {fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0:.4f}")

    # (b) Unparseable counted as wrong
    y_true_b = df_preds['y_true'].values
    y_pred_b = df_preds['y_parsed'].values.copy()
    y_pred_b[unparseable_mask] = 1 - y_true_b[unparseable_mask]

    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1]).ravel()
    print("\n============================================================")
    print("=== EVALUATION (B): UNPARSEABLE COUNTED AS WRONG ===")
    print("============================================================")
    print(f"Total evaluated (n={total_eval}):")
    print(f"  TN: {tn_b} | FP: {fp_b} | FN: {fn_b} | TP: {tp_b}")
    print(f"  Accuracy:  {accuracy_score(y_true_b, y_pred_b):.4f}")
    print(f"  Precision: {precision_score(y_true_b, y_pred_b, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true_b, y_pred_b, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_true_b, y_pred_b, zero_division=0):.4f}")
    print(f"  FPR:       {fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0:.4f}")

    # Show 20 raw model outputs (10 FP, 10 FN)
    if raw_output_col in df_preds.columns:
        fps = df_preds[(df_preds['y_true'] == 0) & (df_preds['y_parsed'] == 1)]
        fns = df_preds[(df_preds['y_true'] == 1) & (df_preds['y_parsed'] == 0)]

        print("\n============================================================")
        print("=== 10 FALSE POSITIVES (Ground Truth: Legitimate -> Model: Phishing) ===")
        print("============================================================")
        for idx, (_, r) in enumerate(fps.head(10).iterrows(), 1):
            print(f"FP {idx:02d} | email_id: {r['email_id']} | Raw Model Output: {repr(r[raw_output_col])}")

        print("\n============================================================")
        print("=== 10 FALSE NEGATIVES (Ground Truth: Phishing -> Model: Legitimate) ===")
        print("============================================================")
        for idx, (_, r) in enumerate(fns.head(10).iterrows(), 1):
            print(f"FN {idx:02d} | email_id: {r['email_id']} | Raw Model Output: {repr(r[raw_output_col])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/processed/phishing_val.csv")
    parser.add_argument("--preds_path", default="eval/results/preds_groq_qwen_qwen3_8-27b_val.csv")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()
    audit_groq_run(args.val_path, args.preds_path, args.cache_dir)
