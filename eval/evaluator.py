# eval/evaluator.py — Shared Evaluation Framework for Phishing-Email Detection

import os
import re
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------
# 1. Prompt Builder (Single Source of Truth)
# ---------------------------------------------------------

ZERO_SHOT_SYSTEM_PROMPT = (
    "You are a cybersecurity expert analyzing emails for phishing indicators. "
    "Classify the following email as either 'Phishing' or 'Legitimate'. "
    "Provide only the label as your final answer."
)

def build_prompt(
    subject: str,
    body: str,
    mode: str = "zero_shot",
    reasoning: Optional[str] = None,
    label: Optional[str] = None
) -> str:
    """
    Constructs standardized prompt strings for zero-shot, naive LoRA, and CoT LoRA.
    """
    subject_str = str(subject) if pd.notna(subject) else ""
    body_str = str(body) if pd.notna(body) else ""

    if mode == "zero_shot":
        return (
            f"{ZERO_SHOT_SYSTEM_PROMPT}\n\n"
            f"Subject: {subject_str}\n"
            f"Body:\n{body_str}\n\n"
            f"Classification (Phishing or Legitimate):"
        )
    elif mode == "naive_sft":
        prompt = (
            f"### Instruction:\n"
            f"Classify the following email as 'Phishing' or 'Legitimate'.\n\n"
            f"### Email:\n"
            f"Subject: {subject_str}\n\n"
            f"Body:\n{body_str}\n\n"
            f"### Response:\n"
        )
        if label:
            prompt += f"{label}"
        return prompt
    elif mode == "cot_sft":
        prompt = (
            f"### Instruction:\n"
            f"Classify the following email as 'Phishing' or 'Legitimate'.\n\n"
            f"### Email:\n"
            f"Subject: {subject_str}\n\n"
            f"Body:\n{body_str}\n\n"
            f"### Response:\n"
        )
        if reasoning and label:
            prompt += f"{reasoning}\n\nFinal answer: {label}"
        elif label:
            prompt += f"Final answer: {label}"
        return prompt
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

# ---------------------------------------------------------
# 2. Deterministic Output Parser
# ---------------------------------------------------------

def parse_model_output(raw_output: str, mode: str = "standard") -> Tuple[int, bool]:
    """
    Deterministic extraction of phishing vs legitimate label.
    Policy for unparseable outputs:
      - Returns (-1, False) if neither or ambiguous label found.
      - During evaluation metrics, unparseable outputs count as incorrect classification (penalty)
        and are explicitly tracked in parse_failure_rate.
    Returns:
      (pred_int, parsed_successfully)
      where 1 = Phishing/Spam, 0 = Legitimate/Ham, -1 = Unparseable.
    """
    if not raw_output or not isinstance(raw_output, str):
        return -1, False

    text = raw_output.strip()

    # If CoT mode, prioritize looking for "Final answer: <label>"
    if "final answer:" in text.lower():
        after_fa = text.lower().split("final answer:")[-1].strip()
        match = re.search(r"\b(phishing|legitimate|spam|ham)\b", after_fa, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            return (1 if val in ("phishing", "spam") else 0), True

    # First-word or beginning-of-response heuristic (standard for naive/zero-shot)
    match = re.search(r"\b(phishing|legitimate|spam|ham)\b", text, re.IGNORECASE)
    if match:
        val = match.group(1).lower()
        return (1 if val in ("phishing", "spam") else 0), True

    # Unparseable output policy
    return -1, False

# ---------------------------------------------------------
# 3. Metric Computation & Schema Enforcement
# ---------------------------------------------------------

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    scores: Optional[List[float]] = None,
    fit_time_sec: float = 0.0,
    latency_ms_per_email: float = 0.0,
    cost_usd_per_1000: float = 0.0,
    model_name: str = "Model",
    split: str = "val"
) -> Tuple[Dict[str, Any], float]:
    """
    Computes standard classification metrics with unparseable policy:
    unparseable predictions (y_pred == -1) count as wrong predictions for accuracy/F1.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    unparseable_mask = (y_p == -1)
    parse_failure_rate = float(unparseable_mask.mean())

    # Map unparseable to wrong prediction
    # If ground truth was 1, make pred 0; if ground truth was 0, make pred 1
    sanitized_preds = y_p.copy()
    sanitized_preds[unparseable_mask] = 1 - y_t[unparseable_mask]

    acc = float(accuracy_score(y_t, sanitized_preds))
    prec = float(precision_score(y_t, sanitized_preds, zero_division=0))
    rec = float(recall_score(y_t, sanitized_preds, zero_division=0))
    f1 = float(f1_score(y_t, sanitized_preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_t, sanitized_preds, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    auroc = None
    if scores is not None and len(scores) == len(y_t):
        try:
            auroc = float(roc_auc_score(y_t, scores))
        except Exception:
            auroc = None

    metrics = {
        "model": model_name,
        "split": split,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if auroc is not None else np.nan,
        "false_positive_rate": round(fpr, 4),
        "fit_time_sec": round(fit_time_sec, 2),
        "latency_ms_per_email": round(latency_ms_per_email, 2),
        "cost_usd_per_1000_emails": round(cost_usd_per_1000, 4)
    }
    return metrics, parse_failure_rate

# ---------------------------------------------------------
# 4. Results Writer & Checkpoint Handler
# ---------------------------------------------------------

def save_evaluation_results(
    results_path: str,
    metrics: Dict[str, Any],
    predictions_df: Optional[pd.DataFrame] = None,
    preds_output_path: Optional[str] = None
):
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    new_df = pd.DataFrame([metrics])

    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        key = existing["model"] + "|" + existing.get("split", "val")
        new_key = new_df["model"] + "|" + new_df["split"]
        existing = existing[~key.isin(new_key)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(results_path, index=False)
    print(f"Results appended to {results_path}")

    if predictions_df is not None and preds_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(preds_output_path)), exist_ok=True)
        predictions_df.to_csv(preds_output_path, index=False)
        print(f"Predictions saved to {preds_output_path}")
