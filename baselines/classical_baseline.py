import os
import sys
import time
import json
import hashlib
import subprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths - Notice strictly train and val only! NO test file references.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "phishing_train.csv")
VAL_PATH = os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "eval", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
RESULTS_PATH = os.path.join(RESULTS_DIR, "baseline_results.csv")

os.makedirs(RUNS_DIR, exist_ok=True)

def get_git_hash():
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return res.decode().strip()
    except Exception:
        return "unknown"

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def combine_text(df):
    return (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

def evaluate(model, X, y, val_df, name, fit_time):
    t0 = time.time()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    latency = (time.time() - t0) / len(X) * 1000

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auroc = roc_auc_score(y, probs)
    fpr = fp / (fp + tn)

    # Save per-example predictions
    model_slug = name.lower().replace(" ", "_").replace("+", "plus")
    preds_df = pd.DataFrame({
        "email_id": val_df["email_id"].values,
        "y_true": y,
        "y_pred": preds,
        "score": probs
    })
    preds_path = os.path.join(RESULTS_DIR, f"preds_{model_slug}_val.csv")
    preds_df.to_csv(preds_path, index=False)

    # Compute per-source F1
    per_source_f1 = {}
    for src in sorted(val_df["dataset_name"].unique()):
        mask = (val_df["dataset_name"] == src).values
        if mask.sum() > 0:
            per_source_f1[src] = float(f1_score(y[mask], preds[mask], zero_division=0))

    metrics = {
        "model": name,
        "split": "val",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
        "false_positive_rate": float(fpr),
        "fit_time_sec": float(fit_time),
        "latency_ms_per_email": float(latency),
        "cost_usd_per_1000_emails": 0.0
    }
    return metrics, per_source_f1

def main():
    print(f"Loading data: {TRAIN_PATH} and {VAL_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print(f"Train: {train_df.shape} | Val: {val_df.shape}")
    X_train_text = combine_text(train_df)
    y_train = train_df["label"].values
    X_val_text = combine_text(val_df)
    y_val = val_df["label"].values

    train_sha = get_file_sha256(TRAIN_PATH)
    val_sha = get_file_sha256(VAL_PATH)
    git_hash = get_git_hash()

    param_grid_lr = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0]
    }
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])

    print("\n--- Running GridSearch for LogisticRegression (3-fold CV on Train) ---")
    t0 = time.time()
    grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, scoring="f1", n_jobs=-1)
    grid_lr.fit(X_train_text, y_train)
    lr_fit_time = time.time() - t0
    print(f"Best LR params: {grid_lr.best_params_} | Best CV F1: {grid_lr.best_score_:.4f} in {lr_fit_time:.1f}s")

    param_grid_svm = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__estimator__C": [0.1, 1.0, 10.0]
    }
    pipeline_svm = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, max_iter=5000), cv=3))
    ])

    print("\n--- Running GridSearch for Calibrated LinearSVC (3-fold CV on Train) ---")
    t0 = time.time()
    grid_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=3, scoring="f1", n_jobs=-1)
    grid_svm.fit(X_train_text, y_train)
    svm_fit_time = time.time() - t0
    print(f"Best SVM params: {grid_svm.best_params_} | Best CV F1: {grid_svm.best_score_:.4f} in {svm_fit_time:.1f}s")

    print("\n--- Evaluating on Validation Set ---")
    lr_metrics, lr_sources = evaluate(grid_lr.best_estimator_, X_val_text, y_val, val_df, "TF-IDF + LogisticRegression", lr_fit_time)
    svm_metrics, svm_sources = evaluate(grid_svm.best_estimator_, X_val_text, y_val, val_df, "TF-IDF + LinearSVM", svm_fit_time)

    print("\n[Validation Metrics]")
    results_df = pd.DataFrame([lr_metrics, svm_metrics])
    print(results_df.round(4).to_string(index=False))

    print("\n[Per-Source Validation F1]")
    sources_df = pd.DataFrame([
        {"model": "TF-IDF + LogisticRegression", **lr_sources},
        {"model": "TF-IDF + LinearSVM", **svm_sources}
    ])
    print(sources_df.round(4).to_string(index=False))

    # Append to baseline_results.csv (or create)
    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH)
        key = existing["model"] + "|" + existing.get("split", "val")
        new_key = results_df["model"] + "|" + results_df["split"]
        existing = existing[~key.isin(new_key)]
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df
    combined.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to: {RESULTS_PATH}")

    # Log run metadata
    run_id = f"week3_classical_{int(time.time())}"
    run_metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": git_hash,
        "library_versions": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "data_sha256": {
            "phishing_train.csv": train_sha,
            "phishing_val.csv": val_sha
        },
        "models": {
            "logistic_regression": {
                "best_params": grid_lr.best_params_,
                "best_cv_f1": float(grid_lr.best_score_),
                "fit_time_sec": lr_fit_time,
                "val_metrics": lr_metrics,
                "val_per_source_f1": lr_sources
            },
            "linear_svm": {
                "best_params": grid_svm.best_params_,
                "best_cv_f1": float(grid_svm.best_score_),
                "fit_time_sec": svm_fit_time,
                "val_metrics": svm_metrics,
                "val_per_source_f1": svm_sources
            }
        }
    }
    run_meta_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Logged run metadata to: {run_meta_path}")

if __name__ == "__main__":
    main()
