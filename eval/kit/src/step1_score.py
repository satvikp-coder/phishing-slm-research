#!/usr/bin/env python3
"""
src/step1_score.py — Arm A Single Forward Pass Logit Scoring on Validation Set
Replicates exact tokenization, truncation, and model configuration.
Calculates logit difference: score = logit("Phishing") - logit("Legitimate").
Saves OUTPUT/armA_val_scores.csv and computes agreement against original predictions.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message, get_system_hardware_info
from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    MAX_SEQ_LENGTH,
    truncate_body_text,
    truncate_subject_text,
    build_naive_prompt
)

def locate_model_path(model_dir: str) -> str:
    for root, dirs, files in os.walk(model_dir):
        if "config.json" in files:
            return root
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Step 1: Arm A Validation Scoring")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation rows (e.g. 50 for quick test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)
    scores_out_path = os.path.join(output_dir, "armA_val_scores.csv")
    summary_out_path = os.path.join(output_dir, "step1_summary.json")

    log_message("=" * 60)
    log_message("STEP 1: ARM A LOGIT SCORING (SINGLE FORWARD PASS)")
    log_message(f"Limit rows : {args.limit if args.limit is not None else 'Full validation set (2,954)'}")
    log_message("=" * 60)

    # 1. Load validation data
    val_csv = os.path.join(kit_root, "data", "phishing_val.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing validation data: {val_csv}")

    val_df = pd.read_csv(val_csv)
    total_val_len = len(val_df)
    log_message(f"Loaded validation set: {total_val_len} emails from {val_csv}")

    if args.limit is not None and args.limit > 0:
        val_df = val_df.head(args.limit).copy()
        log_message(f"Evaluating subset: first {len(val_df)} emails (--limit {args.limit})")

    # 2. Setup offline environment
    model_cache = os.path.join(kit_root, "model_cache")
    adapter_path = os.path.join(kit_root, "adapter")
    base_model_path = locate_model_path(model_cache)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = model_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = model_cache

    # 3. Model Loading with Fallback Hierarchy
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    hw = get_system_hardware_info()
    cuda_available = torch.cuda.is_available() and not hw["blackwell_unsupported"]

    load_mode = "unknown"
    model = None
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target token IDs for scoring
    # Strictly retrieve first token ID for Phishing and Legitimate
    tok_p = tokenizer.encode("Phishing", add_special_tokens=False)[0]
    tok_l = tokenizer.encode("Legitimate", add_special_tokens=False)[0]
    log_message(f"Scoring tokens: 'Phishing'={tok_p} ({tokenizer.decode([tok_p])}), 'Legitimate'={tok_l} ({tokenizer.decode([tok_l])})")

    # Attempt 1: 4-bit NF4 with BitsAndBytes (matches exact original validation setup)
    if cuda_available:
        try:
            log_message("Attempting 4-bit NF4 BitsAndBytes GPU load...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
            model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
            load_mode = "4bit_bnb_gpu"
            log_message("[SUCCESS] Loaded model in 4-bit NF4 QLoRA on GPU.")
        except Exception as e:
            log_message(f"4-bit BitsAndBytes loading failed: {e}. Falling back to float16 unquantized on GPU...")

    # Attempt 2: FP16 unquantized base on GPU
    if model is None and cuda_available:
        try:
            log_message("Attempting FP16 base + LoRA adapter on GPU...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
            model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
            load_mode = "fp16_gpu"
            log_message("[SUCCESS] Loaded model in FP16 on GPU.")
        except Exception as e:
            log_message(f"FP16 GPU loading failed: {e}. Falling back to CPU float32...")

    # Attempt 3: CPU Float32
    if model is None:
        log_message("Loading base + LoRA adapter in Float32 on CPU...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=None,
            torch_dtype=torch.float32,
            local_files_only=True
        ).to("cpu")
        model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
        load_mode = "cpu_float32"
        log_message("[SUCCESS] Loaded model in Float32 on CPU.")

    model.eval()

    # 4. Scoring Loop
    log_message(f"\nStarting scoring for {len(val_df)} emails (load_mode: {load_mode})...")
    scores = []
    preds = []
    elapsed_times = []
    t_start = time.time()

    # Determine device
    device = next(model.parameters()).device

    for idx, (_, row) in enumerate(val_df.iterrows()):
        t0 = time.time()
        # Exact truncation and prompt formatting
        s_trunc, _ = truncate_subject_text(row["subject"], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
        b_trunc, _ = truncate_body_text(row["body"], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
        prompt = build_naive_prompt(s_trunc, b_trunc)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Logits at the last token position
            logits = outputs.logits[0, -1, :]
            score = float(logits[tok_p] - logits[tok_l])
            y_pred = 1 if score > 0.0 else 0

        dt = (time.time() - t0) * 1000.0
        scores.append(score)
        preds.append(y_pred)
        elapsed_times.append(dt)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_df):
            log_message(f"  Processed {idx + 1}/{len(val_df)} emails... (last latency: {dt:.1f} ms)")

    total_time = time.time() - t_start
    avg_latency = float(np.mean(elapsed_times))
    log_message(f"Scoring complete in {total_time:.2f} s (Average latency: {avg_latency:.2f} ms/email)")

    # 5. Save armA_val_scores.csv
    val_df["score"] = scores
    val_df["y_pred"] = preds
    output_cols = ["email_id", "dataset_name", "label", "score", "y_pred"]
    # If label column is named 'label', rename to y_true for standard output
    out_df = val_df[output_cols].rename(columns={"label": "y_true"})
    out_df.to_csv(scores_out_path, index=False)
    log_message(f"Saved scores to: {scores_out_path}")

    # 6. Agreement Analysis vs Original Predictions
    orig_preds_path = os.path.join(kit_root, "data", "preds_qwen2_5_1_5b_naive_val.csv")
    agreement_pct = None
    disagreeing_ids = []

    if os.path.exists(orig_preds_path):
        orig_df = pd.read_csv(orig_preds_path)
        # Match on email_id
        merged = pd.merge(out_df, orig_df[["email_id", "y_pred"]], on="email_id", suffixes=("", "_orig"))
        if len(merged) > 0:
            matches = (merged["y_pred"] == merged["y_pred_orig"])
            agreement_pct = float(matches.mean() * 100.0)
            disagreements = merged[~matches]
            disagreeing_ids = disagreements["email_id"].tolist()
            log_message("\n--- Agreement vs Original Arm A Predictions ---")
            log_message(f"Evaluated Emails       : {len(merged)}")
            log_message(f"Identical Predictions  : {matches.sum()} / {len(merged)}")
            log_message(f"Agreement Rate         : {agreement_pct:.2f}%")
            if disagreeing_ids:
                log_message(f"Disagreeing Email IDs  : {disagreeing_ids}")
            else:
                log_message("All predictions match original Arm A predictions 100%!")

    # 7. Write Step 1 Summary JSON
    summary = {
        "step": "step1_score",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "load_mode": load_mode,
        "n_evaluated": len(out_df),
        "total_val_set": total_val_len,
        "wall_time_sec": round(total_time, 2),
        "mean_latency_ms": round(avg_latency, 2),
        "agreement_with_original_pct": agreement_pct,
        "disagreeing_ids_count": len(disagreeing_ids),
        "disagreeing_ids": disagreeing_ids
    }
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Step 1 summary saved to: {summary_out_path}\n")

if __name__ == "__main__":
    main()
