# Evaluation Directory (`eval/`)

This directory contains the unified evaluation harness, metric computation functions, and experimental result logs.

---

## Directory Organization

```
eval/
├── evaluator.py               # Standardized evaluation module (scores, metrics, latency, audit logging)
├── Week4_ZeroShot_Baselines.ipynb # Interactive Colab runner for zero-shot SLM & Groq LLM baselines
├── results/                   # Verified experimental outputs & predictions
│   ├── baseline_results.csv   # Master tabular comparisons across all evaluated models
│   ├── preds_*.csv            # Per-example predictions (email_id, true label, prediction, probability)
│   └── runs/                  # JSON provenance logs recording environment, seeds, and execution time
└── README.md                  # This guide
```

---

## Standardized Evaluation Protocol

All models (classical, SLM, and LLM) are evaluated using the unified functions in [`evaluator.py`](file:///eval/evaluator.py):

* **Macro Metrics:** Accuracy, Precision, Recall, Macro $F_1$, and AUROC.
* **Operational Metrics:**
  * **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$. Crucial for email security where false alarms disrupt business.
  * **Inference Latency:** Milliseconds per email.
  * **Financial Cost:** USD per 1,000 processed emails.
* **Conservative Parsing Rule:** Any unparseable, malformed, or ambiguous model output is penalized as an error ($y_{\text{pred}} = 1 - y_{\text{true}}$).

---

## Current Results Summary (`eval/results/baseline_results.csv`)

| Model | Evaluation Split | Accuracy | Precision | Recall | F1 Score | FPR | Latency / Email |
|---|---|---|---|---|---|---|---|
| **TF-IDF + LogisticRegression** | val ($n=2954$) | **97.36%** | **0.9703** | **0.9671** | **0.9687** | **2.17%** | 0.43 ms |
| **TF-IDF + LinearSVM** | val ($n=2954$) | **97.02%** | **0.9677** | **0.9615** | **0.9646** | **2.34%** | 0.51 ms |
| **Qwen2.5-1.5B-Instruct** | val ($n=2954$) | **63.64%** | **0.5661** | **0.5970** | **0.5811** | **33.47%** | 432.53 ms |
| **Qwen-27B (Groq Cloud)** | val ($n=1850$) | **32.54%** | **0.2864** | **0.3926** | **0.3312** | **72.44%** | 38.12 ms |
