# Verification Report (Phase 1 — docs/handover/09 Checklist)

**Date:** 2026-10-15  
**Auditor:** Antigravity / Pair Programming Agent  
**Context:** Verification preceding Week 3 Classical Baseline execution per `docs/handover/08` operating instructions.

---

## 1. Environment & Hardware Verification

| Component | Status / Value | Verification Method |
|---|---|---|
| **OS** | Windows 11 | Local system query [Verified] |
| **Python** | `Python 3.13.14` | `python --version` [Verified] |
| **Local GPU** | **No local GPU** (`nvidia-smi` not found, CPU-only execution) | Terminal check [Verified] |
| **PyTorch** | `torch 2.11.0+cpu` | `pip list` [Verified] |
| **ML Libraries** | `scikit-learn 1.8.0`, `pandas 2.3.3`, `numpy 2.4.0` | `pip list` [Verified] |
| **HF Datasets** | `datasets 5.1.0` | Installed and verified [Verified] |

---

## 2. Repository File Map & Discrepancies vs Handover 04/07

### Tracked Files Audit:
- `README.md` [Verified, updated with Colab quick-launch badges]
- `literature-notes.md` [Verified]
- `.gitignore` [Verified, ignores `*.csv` except `eval/results/*.csv`]
- `data/Week2_Data_Acquisition.ipynb` [Verified]
- `data/data_summary.md` [Verified]
- `data/processed/.gitkeep` [Verified]
- `data/raw/.gitkeep` [Verified]
- `baselines/Week3_Classical_Baseline.ipynb` [Verified, unrun]
- `finetuning/Phishing_SLM_QLoRA_Starter.ipynb` [Verified]
- `finetuning/checkpoints/.gitkeep` [Verified]
- `eval/results/.gitkeep` [Verified]
- `docs/handover/00_START_HERE.md` through `10_CODEX_BOOTSTRAP_PROMPT.md` [Verified]

### Discrepancies / Additional Files Not in Original 04 Spec:
1. `AGENTS.md` and `RESEARCH_HANDOVER_AUDIT.md`: Added in commit `da7ae11` and `470018c` to record agent rules and audit findings.
2. `README.md`: Modified to include 1-click Google Colab launch badges for `Phishing_SLM_QLoRA_Starter.ipynb`, `Week3_Classical_Baseline.ipynb`, and `Week2_Data_Acquisition.ipynb`.
3. Root workspace artifacts: Sibling duplicates exist in `c:\Users\satvi\Desktop\Amit Sir\` (`handover/`, `Phishing_SLM_QLoRA_Starter.ipynb`, `Week2_Data_Acquisition.ipynb`). The repo copy is confirmed as the authoritative one.

---

## 3. Dataset Splits Audit (`phishing_train.csv`, `phishing_val.csv`, `phishing_test.csv`)

The authoritative splits have been generated locally using the verified Week 2 methodology from `puyang2025/seven-phishing-email-datasets` with `RANDOM_SEED=42`.

### Files and Checksums:
| File | Size (Bytes) | SHA256 Checksum |
|---|---|---|
| `phishing_train.csv` | 40,565,194 | `81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825` |
| `phishing_val.csv` | 5,397,485 | `3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f` |
| `phishing_test.csv` | 10,398,381 | `6ca5d30bd4a1f4252cf2639d714d51b87f2b022763666fb78e112d8352de69d5` |

### Checklist Verification Results:

1. **Schema Check:**
   - All 3 CSVs contain exact columns: `['email_id', 'subject', 'body', 'label', 'dataset_name']` [VERIFIED]
2. **Row Counts:**
   - `phishing_train.csv`: **20,681** rows (Expected: 20,681) [MATCH: TRUE]
   - `phishing_val.csv`: **2,954** rows (Expected: 2,954) [MATCH: TRUE]
   - `phishing_test.csv`: **5,910** rows (Expected: 5,910) [MATCH: TRUE]
   - Total rows: **29,545** (Capped to 5,000 per source, 38 exact body duplicates removed).
3. **Label=1 (Phishing/Spam) Rates:**
   - `phishing_train.csv`: **42.25%** (Expected: ~42.3%) [MATCH: TRUE]
   - `phishing_val.csv`: **42.25%** (Expected: ~42.2%) [MATCH: TRUE]
   - `phishing_test.csv`: **42.25%** (Expected: ~42.3%) [MATCH: TRUE]
4. **Cross-Split Disjointness:**
   - Duplicate `email_id` across Train/Val/Test: **0** [VERIFIED]
   - Exact body MD5 hash overlap Train/Val: **0** [VERIFIED]
   - Exact body MD5 hash overlap Train/Test: **0** [VERIFIED]
   - Exact body MD5 hash overlap Val/Test: **0** [VERIFIED]
5. **Locked Test Set Guard:**
   - In accordance with `08_CODEX_OPERATING_INSTRUCTIONS.md`, **only** row count (`5,910`), label rate (`42.25%`), and metadata hashes were computed.
   - Zero test email subjects, bodies, or examples were printed, sampled, or inspected.
6. **Train-Only Length Distribution:**
   - **Character Length:**
     - $p50$: 806 chars
     - $p90$: 3,565 chars
     - $p95$: 5,193 chars
     - $p99$: 14,743 chars
   - **Approximate Token Length** (~$1.3 \times$ word count):
     - $p50$: 166 tokens
     - $p90$: 682 tokens
     - $p95$: 1,030 tokens
     - $p99$: 2,796 tokens

---

## 4. Status

✅ **Step 1 (Verification) is 100% COMPLETE.**  
All data counts, hashes, schemas, and guards are fully verified and aligned with the project specification.

Awaiting user's **"go"** to begin **Step 2 (Week 3 Classical Baseline)**!
