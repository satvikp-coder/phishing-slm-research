# Baselines Directory (`baselines/`)

This directory houses the supervised classical machine learning baseline models for phishing email classification.

---

## Contents

* **[`classical_baseline.py`](file:///baselines/classical_baseline.py):**  
  Standalone production-grade Python script that executes feature extraction, 3-fold cross-validated hyperparameter grid search, model fitting, and evaluation on the validation set.
* **[`Week3_Classical_Baseline.ipynb`](file:///baselines/Week3_Classical_Baseline.ipynb):**  
  Interactive Jupyter notebook version for running and inspecting classical baselines in Google Colab or local environments.

---

## Models Evaluated

1. **TF-IDF + Logistic Regression:**
   - Feature Extraction: Word and Character n-grams (`ngram_range=(1,2)`, `max_features=20,000`, sublinear TF scaling).
   - Optimization: L2 Regularization, `C=10.0`, `solver='lbfgs'`, `max_iter=1000`.
2. **TF-IDF + Linear Support Vector Machine (LinearSVC):**
   - Optimization: Squared hinge loss, `C=1.0`, calibrated probabilities via Platt scaling (`CalibratedClassifierCV`).

---

## Experimental Protocol & Integrity

* **Training Split:** Evaluated strictly on `data/processed/phishing_train.csv` (20,681 emails).
* **Validation Split:** Evaluated strictly on `data/processed/phishing_val.csv` (2,954 emails).
* **Strict Test Set Isolation:** Under no circumstances does any baseline code access `phishing_test.csv` (locked until Week 8).
* **Random Seed:** Enforced global seed `42` across all cross-validation folds and estimators.

---

## Results Summary (Validation Set, $n = 2,954$)

| Model | Accuracy | Precision | Recall | F1 Score | AUROC | False Positive Rate | Fit Time | Latency / Email |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **97.36%** | **0.9703** | **0.9671** | **0.9687** | **0.9963** | **2.17%** | 220.5 s | **0.43 ms** |
| **Linear SVM** | **97.02%** | **0.9677** | **0.9615** | **0.9646** | **0.9960** | **2.34%** | 217.2 s | **0.51 ms** |

*Raw prediction logs and run metadata JSON files are stored in [`eval/results/`](file:///eval/results).*
