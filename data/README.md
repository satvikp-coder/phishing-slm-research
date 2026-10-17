# Data Directory (`data/`)

This directory contains scripts, metadata, and pointers for the unified phishing email benchmark datasets.

---

## Directory Organization

```
data/
├── raw/                      # Downloaded source datasets (.gitkeep, raw files ignored)
├── processed/                # Unified, cleaned, and split CSV files
│   ├── phishing_train.csv    # 20,681 rows (70% stratified split) - Unlocked
│   ├── phishing_val.csv      # 2,954 rows (10% stratified split) - Unlocked
│   └── phishing_test.csv     # 5,910 rows (20% stratified split) - STRICTLY LOCKED
├── data_summary.md           # Detailed statistical breakdown, distributions, and length percentiles
├── Week2_Data_Acquisition.ipynb # Colab reproducible data curation pipeline
└── README.md                 # This guide
```

---

## Benchmark Corpora Unified

The dataset unifies 6 foundational email benchmarks from `puyang2025/seven-phishing-email-datasets` (Hugging Face):
1. **SpamAssassin:** Open-source security benchmark corpus (spam & ham).
2. **CEAS-2008:** Real-world enterprise phishing contest data.
3. **Enron Corpus:** Industry-standard benign enterprise emails.
4. **TREC-2005:** NIST spam track benchmark.
5. **TREC-2006:** NIST spam track benchmark.
6. **TREC-2007:** NIST spam track benchmark.

---

## Preprocessing & Deduplication Protocol

* **Per-Source Balance Cap:** Capped at 5,000 samples per corpus to prevent Enron/TREC from dominating representations.
* **Exact Deduplication:** MD5 hash computed on normalized email bodies (`text.strip().lower()`). 38 exact duplicate email bodies removed.
* **Missing Value Filtering:** Rows with null/empty subjects or bodies dropped.
* **Stratification:** Stratified 70/10/20 train/val/test split across combined key `label_dataset_name` with random seed `42`.
* **Zero Data Leakage:** Confirmed zero duplicate `email_id` and zero exact body-hash collisions between train, val, and test splits.

---

## Cryptographic File Checksums (SHA-256)

For verifiable audit and reproducibility, the exact splits match the following cryptographic signatures:

* **`phishing_train.csv`:** `81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825`
* **`phishing_val.csv`:** `3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f`
* **`phishing_test.csv`:** `6ca5d30bd4a1f4252cf2639d714d51b87f2b022763666fb78e112d8352de69d5`

> **IMPORTANT POLICY:**  
> `phishing_test.csv` is strictly **LOCKED** until Week 8 of the research schedule. No model tuning, prompt exploration, or hyperparameter selection is permitted on test data.
