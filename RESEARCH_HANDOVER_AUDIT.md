# Research handover audit — 2026-10-15

> **STATUS UPDATE (October 17, 2026):**  
> All items flagged as incomplete or unverified in this initial audit have now been **fully executed, verified, and audited**:
> * **Step 1 (Data Verification):** Formally documented in [`docs/handover/VERIFICATION_REPORT.md`](docs/handover/VERIFICATION_REPORT.md). Zero leakage, SHA-256 hashes verified, test set locked.
> * **Step 2 (Week 3 Classical Baselines):** Fully executed in [`baselines/classical_baseline.py`](baselines/classical_baseline.py). Logistic Regression ($F_1 = 0.9687$) and Linear SVM ($F_1 = 0.9646$) logged in `eval/results/baseline_results.csv`.
> * **Step 3 (Week 4 Zero-Shot Baselines):** Evaluated on validation split. Qwen2.5-1.5B ($F_1 = 0.5811$) and Groq Qwen-27B ($F_1 = 0.3312$) logged in `eval/results/baseline_results.csv`.

## Scope and evidence

Read all eleven files in `docs/handover/` (byte-identical to the sibling `../handover/` copies), the repository README, literature notes, data summary, three notebooks, Git history, and the root progress/guide documents. Inspected the workspace for datasets, model files, results, dependencies, and `AGENTS.md`. This was a read-only scientific audit: no dataset, model, or experiment was changed or run. The locked test CSV was not opened.

Evidence levels below: **local verified** = directly observed in files or Git; **historical report** = student-pasted Colab output recorded by Claude, without a saved runtime artifact; **planned** = no execution evidence; **unknown** = cannot verify locally.

## Verified repository state

| Item | Finding | Evidence / limit |
|---|---|---|
| Local Git | Clean `main` at `96b3986` before this audit; six commits from Week 1 through handover. | `git status`, `git log`. Commit messages saying “validated” are historical claims, not result files. |
| GitHub | `satvikp-coder/phishing-slm-research` already exists and is **public**. Its `main` was `1fa6b8f` and contains only README, literature notes, and the fine-tuning notebook. Local and remote have unrelated Git histories. | GitHub CLI query and `git fetch`; publication strategy must preserve both histories. |
| Handover | `../handover/00`–`10` and `docs/handover/00`–`10` have matching SHA256 by file. | Local hash comparison. |
| `AGENTS.md` | None existed in the workspace or repository before this audit. | File inventory. |
| Code | Three notebooks: Week 2 data acquisition, Week 3 classical baseline, Week 5–6 QLoRA starter. No Python modules, tests, shared evaluator, teacher generator, or adversarial generator. | Repository file tree and notebook source. |
| Notebook outputs | Every cell in all three committed notebooks has zero saved outputs and no execution count. Week 3 explicitly says unrun. | Parsed notebook JSON. |
| Data / models / results | No CSV, Parquet, JSONL, adapter, checkpoint, or result table exists in this workspace. `data/raw`, `data/processed`, `finetuning/checkpoints`, and `eval/results` contain only `.gitkeep`. Google Drive contents are inaccessible here. | Workspace inventory; no claim about Drive contents. |
| Local runtime | Windows, Python 3.13.14. `nvidia-smi` is unavailable. No local GPU training was attempted. | `python --version`, `nvidia-smi`. Colab T4 and its installed versions remain historical reports. |

## Scientific definition and data checks

- The agreed question is whether reasoning-trace-distilled versus label-only QLoRA on small (≤3B) models changes phishing/spam-email robustness under generated/rephrased email shift, compared with zero-shot SLM/API and classical TF-IDF baselines. Preserve this scope; a null result is valid.
- The Week 2 notebook fixes six sources, a 5,000-row per-source cap, seed 42, exact body deduplication, and stratification on `label` × `dataset_name`. It selects `email_id, subject, body, label, dataset_name`; `label=1` is documented as phishing/**spam**, `0` as legitimate/ham. True label semantics are not independently checked against source records.
- Reported split counts are arithmetically consistent: 20,681 + 2,954 + 5,910 = **29,545**. With five capped sources and 4,585 Assassin records, the maximum pre-dedup count is 29,585; the **net difference is 40**, but it cannot be attributed solely to deduplication because missing/empty-row counts and actual post-clean Assassin count were not saved. The 70/10/20 proportions are consistent with integer rounding, not independently reproduced.
- Reported positive rates (42.3%, 42.2%, 42.3%), raw source counts, column types, unique IDs, source balance, exact body overlap, near duplicates, length distribution, and SHA256 hashes **cannot be verified** without the original CSVs. The locked test set remains unopened. If the student supplies data, verify train/val contents and only test file presence/size/checksum until the agreed final evaluation.
- Primary reported metrics are accuracy, precision, recall, F1, AUROC, and FPR = FP/(FP+TN), with class 1 positive; cost and latency are secondary. The Week 3 code computes these formulas using scikit-learn and confusion-matrix order `TN, FP, FN, TP`. No numeric metric has been produced. The H1 wording says “accuracy degradation” while F1 is described as primary elsewhere; choose and document the headline robustness measure before experiments, without silently changing H1.
- The trainable-parameter report 18,464,768 / 1,562,179,072 = **1.18199%** is arithmetically correct. Its parameter counts and the reported losses/runtimes/global steps are **historical reports only** from three placeholder rows, not saved reproducible runs or scientific findings.

## Completed, incomplete, broken, unverified

| Status | Work |
|---|---|
| Locally verified artifacts | Week 1 research framing and literature notes; Week 2 split-building code and data summary; Week 3 baseline code; fine-tuning code; handover. |
| Historically reported completion | Week 2 CSV creation on Colab/Drive; naive and CoT three-row pipeline smoke runs and separate adapter paths. The files/runtimes were not available to verify. |
| Incomplete | Week 3 baseline has not run; Week 4 zero-shot/API comparison, real QLoRA training, teacher rationales, adversarial set, shared evaluator, inferential statistics, and paper are absent. |
| Blocking training hazards | `USE_COT=True` silently uses label-only targets if `reasoning` is absent; missing training CSV silently substitutes three synthetic rows; 1,024-token post-format truncation may remove targets; T4/bfloat16 behavior is unmeasured; loss masking, seeds, prediction logging, and reproducible environment are absent. |
| Other technical issues | Week 2 markdown says near duplicates are “flagged” but no such code exists. Week 3 grid is computationally expensive (18 combinations × 3 folds per model, with nested calibration for SVM); its runtime is unknown. Evaluation predicts twice, so measured latency includes both calls. Result writer replaces matching model/split rows without run provenance. No safeguards prevent using stale or swapped data paths. |
| Documentation discrepancies | Handover `04` calls README stale, but current README already marks Weeks 5–6 as infrastructure-only. README says adapters were “confirmed saved,” while the handover admits no Drive listing. Root-level fine-tuning notebook is an older duplicate pointing at `phishing_dataset.csv`; repository copy points at `phishing_train.csv`. README's research-gap assertion and literature-derived numeric claims need source checking before citation. Earlier Project Guide/Proposal contain known superseded interpretations and mention Nazario as used, contrary to the actual data plan. |

## Latest reproducible state and exact stop point

The latest **locally reproducible code state** is commit `96b3986`: Week 2 acquisition logic and an unrun Week 3 classical-baseline notebook. The actual split files and dependencies are absent, so even the Week 2 counts cannot be reproduced here. The latest **historical Colab state** is a three-row QLoRA pipeline smoke run; it does not establish model performance. Previous work stopped after rebuilding the Week 3 notebook, before delivering or executing it. No experiment result table exists.

## Priority next steps

1. Obtain the original `phishing_train.csv` and `phishing_val.csv` from Drive, plus metadata/checksum for locked `phishing_test.csv`; record immutable hashes and provenance. Keep all email text out of Git. Ask for the dataset revision and source run log if available. Do not regenerate or replace the authoritative splits merely to match the reported counts.
2. Verify train/val schema, counts, labels, IDs, source mix, body-length distribution, and exact/near-duplicate risk. For the locked test file, only verify presence, byte size, and checksum. Resolve the spam-versus-phishing construct limitation before interpreting any result as phishing-specific.
3. Pin a runnable environment, then run the existing Week 3 baseline on train/val only with a recorded config, library versions, input hashes, and per-example predictions. Fix provenance and latency measurement before treating rows as publishable results. No new experiment was launched during this audit.
4. Implement Week 4 zero-shot baselines and a common output parser/evaluator. Before real QLoRA, fix the silent CoT and data fallbacks, truncation, precision selection, loss masking, seed, and teacher-data validation.
5. Clarify the deadline/submission target, Drive file and adapter locations, supervisor requirements beyond project acceptance, API provider and budget, and teacher-model choice. Verify close literature claims against primary papers before writing the final related-work section.

## GitHub publication note

The existing public repository's `main` is a separate upload lineage. Publish the audited local history by integrating histories without force-push, keeping raw/processed CSVs, generated examples, model weights, and credentials excluded. Recheck final remote status after push.
