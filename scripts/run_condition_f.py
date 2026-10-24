import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REPO_ROOT = r"c:\Users\satvi\Desktop\Amit Sir\phishing-slm-research"
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "phishing_train.csv")
VAL_PATH = os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv")
SAMPLE_PATH = os.path.join(REPO_ROOT, "eval", "results", "latency_sample_ids.csv")

sys.path.insert(0, REPO_ROOT)
from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    build_naive_prompt,
    truncate_body_text,
    truncate_subject_text
)

print("--- Condition F: Baseline TF-IDF + Logistic Regression Benchmark ---", flush=True)

# 1. Fit TF-IDF + Logistic Regression on train set
print(f"Loading training data: {TRAIN_PATH}...", flush=True)
train_df = pd.read_csv(TRAIN_PATH)
X_train_text = (train_df["subject"].fillna("") + " " + train_df["body"].fillna("")).str.strip()
y_train = train_df["label"].values

print("Fitting Pipeline(TF-IDF, LogisticRegression)...", flush=True)
t_fit_start = time.time()
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
])
pipeline.fit(X_train_text, y_train)
t_fit = time.time() - t_fit_start
print(f"Fitted in {t_fit:.2f} seconds.", flush=True)

# 2. Load the 500 sample emails
sample_meta = pd.read_csv(SAMPLE_PATH)
val_df = pd.read_csv(VAL_PATH)
val_sample = sample_meta.merge(val_df[["email_id", "subject", "body"]], on="email_id", how="left")
print(f"Loaded {len(val_sample)} sample emails across 5 buckets.", flush=True)

# 3. Setup Hygiene: 20 warm-up emails
print("Running 20 warm-up emails (discarded)...", flush=True)
for i in range(20):
    row = val_sample.iloc[i % len(val_sample)]
    text = (str(row["subject"]) + " " + str(row["body"])).strip()
    _ = pipeline.predict([text])
    _ = pipeline.predict_proba([text])

# 4. Measure 3 Full Repeats for Batch Size 1 (End-to-End per email)
print("\nMeasuring Batch Size 1 (End-to-End: text concatenation + vectorizer + model predict_proba)...", flush=True)
repeats_data = [] # list of dicts per repeat: {repeat_idx, email_id, bucket, latency_ms}

for rep in range(3):
    t_rep_start = time.time()
    rep_records = []
    for idx, row in val_sample.iterrows():
        t0 = time.perf_counter()
        # End-to-end text preparation + inference
        text = (str(row["subject"]) + " " + str(row["body"])).strip()
        prob = pipeline.predict_proba([text])[0, 1]
        pred = 1 if prob > 0.5 else 0
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        rep_records.append({
            "repeat": rep + 1,
            "email_id": row["email_id"],
            "bucket": row["bucket"],
            "latency_ms": lat_ms,
            "pred": pred
        })
    rep_df = pd.DataFrame(rep_records)
    repeats_data.append(rep_df)
    p50 = rep_df["latency_ms"].median()
    p95 = np.percentile(rep_df["latency_ms"], 95)
    print(f"  Repeat {rep+1}: p50={p50:.3f} ms | p95={p95:.3f} ms | wall={time.time()-t_rep_start:.2f}s", flush=True)

all_repeats_df = pd.concat(repeats_data)

# Summary statistics across 3 repeats
print("\n--- Summary Statistics (Overall and Per Bucket) for Condition F (Batch Size 1) ---")
buckets = ["Overall"] + sorted(val_sample["bucket"].unique().tolist())
f_summary = []

for b in buckets:
    if b == "Overall":
        sub = all_repeats_df
        n = 500
    else:
        sub = all_repeats_df[all_repeats_df["bucket"] == b]
        n = 100
    
    # Per-repeat medians
    rep_p50s = [sub[sub["repeat"] == r]["latency_ms"].median() for r in [1, 2, 3]]
    rep_p95s = [np.percentile(sub[sub["repeat"] == r]["latency_ms"], 95) for r in [1, 2, 3]]
    rep_p99s = [np.percentile(sub[sub["repeat"] == r]["latency_ms"], 99) for r in [1, 2, 3]]
    rep_means = [sub[sub["repeat"] == r]["latency_ms"].mean() for r in [1, 2, 3]]
    
    med_p50 = np.median(rep_p50s)
    iqr_p50 = np.percentile(rep_p50s, 75) - np.percentile(rep_p50s, 25)
    
    med_p95 = np.median(rep_p95s)
    med_p99 = np.median(rep_p99s)
    mean_val = np.mean(rep_means)
    std_val = np.std(rep_means)
    throughput = 1000.0 / med_p50
    
    print(f"{b:<15} | n={n} | p50={med_p50:.3f} ms (IQR: {iqr_p50:.3f}) | p95={med_p95:.3f} ms | p99={med_p99:.3f} ms | Mean={mean_val:.3f} ms | Throughput={throughput:.1f} emails/s")
    f_summary.append({
        "condition": "F_Baseline_TFIDF_LogReg_batch1",
        "bucket": b,
        "n": n,
        "repeats": 3,
        "p50_ms": round(med_p50, 4),
        "iqr_ms": round(iqr_p50, 4),
        "p95_ms": round(med_p95, 4),
        "p99_ms": round(med_p99, 4),
        "mean_ms": round(mean_val, 4),
        "std_ms": round(std_val, 4),
        "throughput_emails_per_sec": round(throughput, 2),
        "throughput_emails_per_hour": round(throughput * 3600, 2),
        "hardware": "CPU (Intel)",
        "precision": "fp32"
    })

# 5. Measure Batched Throughput (Batch Sizes 8, 16, 32, 500)
print("\n--- Batched Inference Throughput for Condition F (CPU) ---")
texts = [(str(row["subject"]) + " " + str(row["body"])).strip() for _, row in val_sample.iterrows()]

for bs in [8, 16, 32, 500]:
    b_times = []
    # 3 repeats
    for rep in range(3):
        t0 = time.perf_counter()
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            _ = pipeline.predict_proba(batch)
        t1 = time.perf_counter()
        b_times.append(t1 - t0)
    
    med_time = np.median(b_times)
    batch_lat_ms = (med_time / (len(texts) / bs)) * 1000.0
    email_lat_ms = (med_time / len(texts)) * 1000.0
    th_sec = len(texts) / med_time
    th_hour = th_sec * 3600
    print(f"Batch Size {bs:>3}: Total={med_time:.3f}s | Per-Batch Latency={batch_lat_ms:.2f} ms | Per-Email={email_lat_ms:.3f} ms | Throughput={th_sec:.1f} emails/s ({th_hour:,.0f} emails/hr)")

print("\nCondition F complete successfully.")
