# Research Progress & Experimental Evidence Report

**Project Title:** Small but Sufficient? — A Cost–Performance–Robustness Comparison of Fine-Tuned Small Language Models and Proprietary LLM APIs for Phishing-Email Detection Under Adversarial Shift  
**Principal Investigator:** Satvik (Student)  
**Supervisor:** Prof. Amit Sir  
**Date:** October 16, 2026  
**Repository:** [https://github.com/satvikp-coder/phishing-slm-research](https://github.com/satvikp-coder/phishing-slm-research)  
**Git Branch:** `main` | **Audited Git Commits:** `5118672`, `96cbb4d`, `e6bb3f0`

---

## 1. Executive Summary & Core Research Question

### The Research Question (Finalized)
> *Does reasoning-trace-distilled (Chain-of-Thought) LoRA fine-tuning of small ($\le$ 3B) language models improve phishing-email classification robustness against AI-generated / rephrased phishing more than naive label-only LoRA fine-tuning, and how do both compare to zero-shot SLM prompting, commercial APIs, and classical ML baselines?*

### Key Milestone Achieved This Week:
1. **Dataset Integrity Verified:** 29,545 emails unified, cleaned, deduplicated, and split across 6 major corpora. The test set (`5,910` emails) is **strictly locked** to ensure scientific validity.
2. **Classical ML Baselines Established:** Fit 3-fold cross-validated grid search on 20,681 training emails. Logistic Regression achieved **$F_1 = 0.9687$**, and Linear SVM achieved **$F_1 = 0.9646$**.
3. **Zero-Shot SLM Floor Established:** Evaluated off-the-shelf, untrained `Qwen2.5-1.5B-Instruct` across all 2,954 validation emails. It achieved only **$F_1 = 0.5811$** with a high false-positive rate (**$33.47\%$**).
4. **Core Scientific Takeaway:** Off-the-shelf small language models fail on zero-shot phishing classification, providing direct empirical proof that **fine-tuning (Weeks 5 & 6) is necessary**.

---

## 2. Experimental Results Table (The Hard Evidence)

All metrics are evaluated strictly on the untouched Validation Split (`phishing_val.csv`, $n = 2,954$ emails).

| Model | Architecture | Accuracy | Precision | Recall | **F1 Score** | AUROC | False Positive Rate | Latency / Email | Cost / 1k Emails |
|---|---|---|---|---|---|---|---|---|
| **TF-IDF + LogisticRegression** | Classical ML (Trained) | **97.36%** | 0.9703 | 0.9671 | **0.9687** | 0.9963 | **2.17%** | **0.43 ms** | $0.00 |
| **TF-IDF + Linear SVM** | Classical ML (Trained) | **97.02%** | 0.9677 | 0.9615 | **0.9646** | 0.9960 | **2.34%** | **0.51 ms** | $0.00 |
| **Qwen2.5-1.5B-Instruct** | Zero-Shot SLM (Untrained) | **63.64%** | **0.5661** | **0.5970** | **0.5811** | — | **33.47%** | **432.53 ms** | $0.00 |

*Data source: `eval/results/baseline_results.csv`*

---

## 3. In-Distribution Source Breakdown (Generalization Audit)

To prove models did not simply exploit dataset-specific shortcuts or source-identity tokens, we measured validation $F_1$ across each contributing sub-corpus:

| Sub-Corpus | Source Description | TF-IDF + Logistic Regression ($F_1$) | TF-IDF + Linear SVM ($F_1$) |
|---|---|---|---|
| **SpamAssassin** | Open-source security benchmark | **0.9594** | **0.9594** |
| **CEAS-2008** | Real-world enterprise/phishing contest | **0.9909** | **0.9910** |
| **Enron Corpus** | Industry standard legitimate email | **0.9758** | **0.9712** |
| **TREC-2005** | NIST spam/phishing track | **0.9615** | **0.9519** |
| **TREC-2006** | NIST spam/phishing track | **0.9440** | **0.9286** |
| **TREC-2007** | NIST spam/phishing track | **0.9617** | **0.9615** |

*Observation:* Both classical baselines generalized across all 6 corpora ($F_1 > 0.92$ on every individual source), confirming that the in-distribution baseline ceiling is solidly established at **~$0.968$**.

---

## 4. Dataset Provenance & Scientific Hygiene

* **Primary Corpus:** `puyang2025/seven-phishing-email-datasets` (Hugging Face)
* **Sampling & Deduplication:** Per-source cap of 5,000 emails, exact MD5 body deduplication (38 exact duplicates removed).
* **Cross-Split Disjointness:** Zero duplicate email IDs, zero exact body-hash leakage across splits.
* **Locked Test Set:** `phishing_test.csv` ($n = 5,910$ emails, 42.25% positive rate) was verified by shape and SHA-256 hash only. Zero test emails were loaded, printed, or inspected.

### Immutable File Checksums (SHA-256):
* `phishing_train.csv` (20,681 rows): `81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825`
* `phishing_val.csv` (2,954 rows): `3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f`
* `phishing_test.csv` (5,910 rows): `6ca5d30bd4a1f4252cf2639d714d51b87f2b022763666fb78e112d8352de69d5` *(LOCKED until Week 8)*

---

## 5. Verifiable Artifacts & Proof Files (In Repository)

The professor or review committee can directly open and verify each of the following files:

1. **Results Master Table:**  
   [`eval/results/baseline_results.csv`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/eval/results/baseline_results.csv)  
   Contains the verified tabular accuracy, precision, recall, F1, AUROC, FPR, and latency metrics.
2. **Per-Example Raw Predictions (2,954 rows each):**  
   * [`eval/results/preds_tf-idf_plus_logisticregression_val.csv`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/eval/results/preds_tf-idf_plus_logisticregression_val.csv) (Logistic Regression)  
   * [`eval/results/preds_tf-idf_plus_linearsvm_val.csv`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/eval/results/preds_tf-idf_plus_linearsvm_val.csv) (Linear SVM)  
   Each file contains `email_id, y_true, y_pred, score` for every single email in the validation set.
3. **Execution Run Metadata & Hyperparameter Log:**  
   [`eval/results/runs/week3_classical_1792088697.json`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/eval/results/runs/week3_classical_1792088697.json)  
   Records exact wall-time (220.5s), library versions (`scikit-learn 1.8.0`, `pandas 2.3.3`), git commit hash, and best hyperparameters selected via grid search (`max_features=20000`, `ngram_range=(1,2)`).
4. **Formal Data Audit Checklist:**  
   [`docs/handover/VERIFICATION_REPORT.md`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/docs/handover/VERIFICATION_REPORT.md)  
   Documents hardware checks, row counts, and character/token length percentiles ($p50 = 166$ tokens, $p95 = 1,030$ tokens).
5. **Standalone Execution Code:**  
   * [`baselines/classical_baseline.py`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/baselines/classical_baseline.py) (Classical Baseline script)  
   * [`eval/evaluator.py`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/eval/evaluator.py) (Standardized evaluation framework)  
   * [`requirements.txt`](file:///c:/Users/satvi/Desktop/Amit%20Sir/phishing-slm-research/requirements.txt) (Pinned reproducible library versions)

---

## 6. Next Experimental Steps

* **Weeks 5 & 6 (The Core Comparison):**  
  Fine-tune `Qwen2.5-1.5B-Instruct` using QLoRA:
  * **Arm A (Naive):** Mapping `Email -> Label`
  * **Arm B (CoT-Distilled):** Mapping `Email -> Reasoning Diagnostic Chain -> Label`
* **Week 7:** Build the Adversarial (LLM-generated / rephrased) Phishing Test Set.
* **Week 8:** Main Evaluation on Locked Test & Adversarial Sets to prove or disprove Hypothesis H1.
