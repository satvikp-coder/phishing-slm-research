import os
import sys
import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = r"c:\Users\satvi\Desktop\Amit Sir\phishing-slm-research"
VAL_PATH = os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv")
ADAPTER_DIR = os.path.join(REPO_ROOT, "finetuning", "checkpoints", "Qwen2.5-1.5B-Instruct-naive")

sys.path.insert(0, REPO_ROOT)
from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    MAX_SEQ_LENGTH,
    build_naive_prompt,
    truncate_body_text,
    truncate_subject_text
)

print("Loading validation dataset...", flush=True)
df_val = pd.read_csv(VAL_PATH)
print(f"Total validation rows: {len(df_val)}", flush=True)

print("Loading offline tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, local_files_only=True)

print("Calculating total prompt tokens for all validation emails...", flush=True)
t0 = time.time()
prompt_token_counts = []
for idx, row in df_val.iterrows():
    subj, _ = truncate_subject_text(row.get("subject", ""), max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
    body, _ = truncate_body_text(row.get("body", ""), max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
    prompt = build_naive_prompt(subj, body)
    toks = len(tokenizer.encode(prompt, add_special_tokens=False))
    prompt_token_counts.append(toks)
    if (idx + 1) % 500 == 0 or idx == len(df_val) - 1:
        print(f"  Processed {idx + 1}/{len(df_val)} emails ({time.time() - t0:.1f}s)...", flush=True)

df_val["prompt_tokens"] = prompt_token_counts
print("Prompt token summary:", flush=True)
print(df_val["prompt_tokens"].describe(), flush=True)

# Stratify into 5 length quintiles
df_val["bucket"] = pd.qcut(
    df_val["prompt_tokens"], 
    q=5, 
    labels=["B1_very_short", "B2_short", "B3_medium", "B4_long", "B5_very_long"]
)

print("\nBucket intervals and counts:", flush=True)
for b in ["B1_very_short", "B2_short", "B3_medium", "B4_long", "B5_very_long"]:
    sub = df_val[df_val["bucket"] == b]
    print(f"  {b}: range [{sub['prompt_tokens'].min()} - {sub['prompt_tokens'].max()}], count={len(sub)}", flush=True)

# Sample 100 per bucket with seed 42
sample_dfs = []
for b in ["B1_very_short", "B2_short", "B3_medium", "B4_long", "B5_very_long"]:
    sub = df_val[df_val["bucket"] == b]
    s = sub.sample(n=100, random_state=42)
    sample_dfs.append(s)

df_500 = pd.concat(sample_dfs).sort_values("email_id").reset_index(drop=True)
print(f"\nFinal sampled set: {len(df_500)} emails (100 per bucket)", flush=True)
print("Bucket counts in sample:", flush=True)
print(df_500["bucket"].value_counts(), flush=True)
print("Label distribution in sample:", flush=True)
print(df_500["label"].value_counts(), flush=True)

# Save sample ids
out_csv = os.path.join(REPO_ROOT, "eval", "results", "latency_sample_ids.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_500[["email_id", "bucket", "prompt_tokens", "label", "dataset_name"]].to_csv(out_csv, index=False)
print(f"\n[SUCCESS] Saved 500 stratified sample IDs to: {out_csv}", flush=True)
