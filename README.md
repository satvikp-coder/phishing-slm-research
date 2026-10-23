# Small but Sufficient? SLM vs. LLM Phishing-Email Detection

[![Open In Colab: QLoRA Fine-Tuning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/satvikp-coder/phishing-slm-research/blob/main/finetuning/Phishing_SLM_QLoRA_Starter.ipynb)
[![Open In Colab: Week 4 Zero-Shot Baselines](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/satvikp-coder/phishing-slm-research/blob/main/eval/Week4_ZeroShot_Baselines.ipynb)
[![Open In Colab: Week 3 Classical Baseline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/satvikp-coder/phishing-slm-research/blob/main/baselines/Week3_Classical_Baseline.ipynb)
[![Open In Colab: Week 2 Data Acquisition](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/satvikp-coder/phishing-slm-research/blob/main/data/Week2_Data_Acquisition.ipynb)

* **Principal Investigator:** Satvik ([@satvikp-coder](https://github.com/satvikp-coder))
* **Supervisor:** Prof. Amit Sir
* **Institution:** Ahmedabad University
* **Evidence Reports:** [Formal Progress Report (Markdown)](docs/PROGRESS_AND_RESULTS_EVIDENCE.md) | [Audit Verification Report](docs/handover/VERIFICATION_REPORT.md)

---

## 1. Research Question & Hypothesis

### Research Question
> *Does reasoning-trace-distilled (Chain-of-Thought) LoRA fine-tuning of small ($\le$ 3B) language models improve phishing-email classification robustness against AI-generated and adversarial phishing more than naive label-only LoRA fine-tuning, and how do both compare to strong zero-shot prompting (SLM and proprietary LLMs) and classical ML baselines?*

### Hypothesis (H1)
**Chain-of-Thought distilled LoRA fine-tuning** will yield significantly smaller accuracy degradation than naive label fine-tuning when transitioning from in-distribution (human-written) phishing to LLM-rephrased adversarial phishing. The explicit reasoning trace teaches transferable diagnostic heuristics (domain/sender mismatch, urgency framing, pretext plausibility) rather than superficial lexical cues that are easily circumvented by paraphrasing.

---

## 2. Experimental Benchmark Results (Weeks 1–4)

All evaluations are conducted strictly on the untouched Validation Split (`phishing_val.csv`, $n = 2,954$ emails). The test split ($n = 5,910$ emails) remains **cryptographically locked** until Week 8.

| Model / System | Architecture / Mode | Accuracy | Precision | Recall | **F1 Score** | AUROC | False Positive Rate (FPR) | Latency / Email | Cost / 1k Emails |
|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF + LogisticRegression** | Classical ML (Trained) | **97.36%** | **0.9703** | **0.9671** | **0.9687** | **0.9963** | **2.17%** | **0.43 ms** | $0.00 |
| **TF-IDF + Linear SVM** | Classical ML (Trained) | **97.02%** | **0.9677** | **0.9615** | **0.9646** | **0.9960** | **2.34%** | **0.51 ms** | $0.00 |
| **Qwen2.5-1.5B-Instruct** | Zero-Shot SLM (Untrained) | **63.64%** | **0.5661** | **0.5970** | **0.5811** | — | **33.47%** | **432.53 ms** | $0.00 |
| **Qwen-27B (Groq Cloud)** | Zero-Shot LLM (Untrained) | **32.54%** | **0.2864** | **0.3926** | **0.3312** | — | **72.44%** | **38.12 ms** | $0.00 |
| **Qwen2.5-1.5B (Arm A — Naive QLoRA)** | **Fine-Tuned SLM (Trained)** | **99.15%** | **0.9880** | **0.9920** | **0.9900** | **0.9916** | **0.88%** | **167.56 ms** | $0.00 |

*Data source: [`eval/results/baseline_results.csv`](eval/results/baseline_results.csv) & [`eval/results/preds_qwen2_5_1_5b_naive_val.csv`](eval/results/preds_qwen2_5_1_5b_naive_val.csv)*

---

### Why Did We Build These Two Classical Baseline Models?

A common question in modern language model research is: *“If the goal is to evaluate Small Language Models (SLMs), why spend time implementing classical TF-IDF algorithms?”*

In rigorous academic research, **classical models serve as the mandatory empirical control**:

1. **The Gold Standard Statistical Baseline:**
   - **TF-IDF + Logistic Regression:** Computes Term Frequency–Inverse Document Frequency across 20,000 unigrams and bigrams, weighting distinctive lexical tokens (e.g., `"verify account"`, `"wire transfer"`) and learning a linear decision boundary squashed through a Sigmoid probability function.
   - **TF-IDF + Linear SVM (Calibrated):** Finds the maximum-margin hyperplane separating the phishing cluster from the legitimate cluster in 20,000-dimensional feature space, calibrated via Platt scaling (`CalibratedClassifierCV`) to output reliable posterior probabilities.

2. **Why They Score High on Static Benchmarks ($F_1 \approx 0.968$):**
   - Standard academic corpora (Enron, SpamAssassin, CEAS-08, TREC) share persistent surface keyword artifacts. Classical models memorize these specific vocabulary signatures, achieving exceptional in-distribution metrics ($97.36\%$ accuracy, $2.17\%$ False Positive Rate).

3. **The Fatal Flaw of Classical Models (Adversarial Brittleness):**
   - Bag-of-words classical models possess **zero syntactic understanding and zero semantic reasoning**. 
   - When an attacker uses a generative LLM to rephrase a phishing email—using polite, professional vocabulary, novel sentence structures, or subtle psychological pretexts without traditional spam keywords—**classical models collapse completely**.

4. **The Research Justification for Fine-Tuned SLMs:**
   - The comparison above proves that **untrained zero-shot models fail** (1.5B has a 33.47% FPR; 27B has a 72.44% FPR).
   - Therefore, the goal of this research (Weeks 5 & 6) is to combine the **low false-positive discipline of trained classifiers** with the **deep semantic comprehension of Small Language Models** via **Chain-of-Thought (CoT) Distillation**, creating a defense that remains robust against AI-generated adversarial phishing.

### Key Scientific Inferences
1. **The "Zero-Shot Hyper-Sensitivity" Collapse:** Scaling from 1.5B to 27B parameters in an untrained zero-shot regime does not improve phishing detection. The 27B model collapsed to an $F_1$ of 0.3312 due to a catastrophic **72.44% False Positive Rate** (labeling benign enterprise emails as phishing).
2. **Production Usability Criterion:** Enterprise security requires an $\text{FPR} \le 2.5\%$. Zero-shot prompting (33.47% and 72.44% FPR) is completely unviable for production email filtering without fine-tuning.
3. **The Fine-Tuning Mandate:** These baselines mathematically establish the need for **Arm A (Naive LoRA)** to eliminate false alarms and **Arm B (CoT Distillation)** to impart structural reasoning against adversarial paraphrasing.


---

## 3. Repository Directory Guide

Each folder in this repository contains its own standalone `README.md` guide explaining implementation details and execution protocols:

| Folder | Documentation Guide | Description |
|---|---|---|
| [`baselines/`](baselines/) | [**`baselines/README.md`**](baselines/README.md) | Classical ML baselines (TF-IDF + Logistic Regression, Linear SVM), grid search CV, and per-source generalization code. |
| [`data/`](data/) | [**`data/README.md`**](data/README.md) | Dataset acquisition scripts, provenance metadata, split checksums (SHA-256), and anti-leakage isolation protocols. |
| [`eval/`](eval/) | [**`eval/README.md`**](eval/README.md) | Unified evaluation harness (`evaluator.py`), zero-shot Colab notebook, master results CSV, and raw prediction logs. |
| [`finetuning/`](finetuning/) | [**`finetuning/README.md`**](finetuning/README.md) | QLoRA fine-tuning pipelines for Arm A (Naive) and Arm B (CoT Distilled) with completion-only loss masking. |
| [`docs/`](docs/) | [**`docs/README.md`**](docs/README.md) | Formal evidence reports, verification checklists, and the complete 00–10 researcher handover documentation package. |

---

## 4. Dataset Provenance & Cryptographic Hashes

Unified from 6 benchmark corpora (`Enron`, `CEAS-08`, `SpamAssassin`, `TREC-05`, `TREC-06`, `TREC-07`) from `puyang2025/seven-phishing-email-datasets`:

* **`phishing_train.csv` (20,681 rows, 70%):** `81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825`
* **`phishing_val.csv` (2,954 rows, 10%):** `3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f`
* **`phishing_test.csv` (5,910 rows, 20%):** `6ca5d30bd4a1f4252cf2639d714d51b87f2b022763666fb78e112d8352de69d5` *(STRICTLY LOCKED)*

**Integrity Guarantee:** Zero duplicate email IDs, zero exact body-hash collisions, and zero test set peeking prior to Week 8.

---

## 5. Research Schedule & Progress Status

- [x] **Week 1:** Literature grounding, problem formalization, and compute budget ($0 spend target).
- [x] **Week 2:** Multi-corpus acquisition, MD5 deduplication, and stratified 70/10/20 train/val/test splitting.
- [x] **Week 3:** Classical ML baselines established ($F_1 = 0.9687$, 3-fold cross-validated).
- [x] **Week 4:** Zero-shot SLM & LLM baselines established ($F_1 = 0.5811$ for 1.5B, $F_1 = 0.3312$ for 27B).
- [ ] **Week 5:** Arm A Fine-Tuning: Naive QLoRA on `Qwen2.5-1.5B-Instruct` (Direct Labeling) — *IN PROGRESS* (Supervisor approved 2026-10-18).
- [ ] **Week 6:** Arm B Fine-Tuning: Reasoning-Distilled QLoRA on `Qwen2.5-1.5B-Instruct` (Chain-of-Thought).
- [ ] **Week 7:** Adversarial & AI-rephrased phishing test set construction.
- [ ] **Week 8:** Final evaluation across locked test set and adversarial test sets.
- [ ] **Weeks 9–10:** Ablation studies (reasoning trace length, LoRA rank sensitivity) and cost-latency profiling.
- [ ] **Weeks 11–12:** Statistical significance testing, error analysis, and manuscript write-up.

---

## 6. How to Run Locally or in Google Colab

### Environment Setup
```bash
git clone https://github.com/satvikp-coder/phishing-slm-research.git
cd phishing-slm-research
pip install -r requirements.txt
```

### Reproduce Classical Baselines (Weeks 3)
```bash
python baselines/classical_baseline.py
```

### Reproduce Zero-Shot Baselines (Week 4)
Open [`eval/Week4_ZeroShot_Baselines.ipynb`](eval/Week4_ZeroShot_Baselines.ipynb) in Google Colab and run on free CPU/GPU.
