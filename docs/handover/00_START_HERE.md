# 00 — START HERE (Codex onboarding)

## Evidence tags used in every handover file
| Tag | Meaning |
|---|---|
| **[R]** | Verified by Claude inspecting the repo files on handover date (2026-10-09). Repo = restored copy from last saved zip (HEAD `b560753`), NOT the student's live local/GitHub copy. |
| **[U]** | Numbers/output the student pasted into chat from their own Colab runs. Claude never saw the runtime itself. |
| **[S]** | Derived from web searches done during the conversation. Raw results were partly not retained; re-verify before citing. |
| **[P]** | Planned/discussed only. Not executed. |
| **[?]** | `UNKNOWN — REQUIRES VERIFICATION` |

## Project identity
- **Working title:** "Small but Sufficient? A Cost–Performance–Robustness Comparison of Fine-Tuned Small Language Models and Proprietary LLM APIs for Phishing-Email Detection Under Adversarial (LLM-Generated) Phishing" [R: README, docs]
- **Repo:** `phishing-slm-research` (git, branch `main`). GitHub URL / whether pushed: [?]
- **Institution / department / supervisor name / start date / hard deadline:** [?] — never provided. Student calls supervisor "sir".
- **Student constraints (stated at project start):** ~12-week plan, 1 student, LOW compute (inference, small open models, QLoRA, APIs; no pretraining), goal = research paper with quantitative evaluation. Compute actually used so far: Google Colab free-tier Tesla T4 [U].
- **Supervisor input received:** only that the supervisor **accepted "Project 3" (= Project C, this project)** out of 3 candidates presented [U, one sentence]. No other feedback recorded.

## One-paragraph science summary
Compare ≤3B-parameter open SLMs (fine-tuned with QLoRA two ways: **naive label-only** vs **CoT/reasoning-trace-distilled**) against zero-shot SLMs, a proprietary LLM API, and a TF-IDF classical baseline on phishing-email classification, on (a) an in-distribution locked test set and (b) a self-built set of LLM-generated phishing emails. Core contribution (narrowed after novelty check): **the controlled naive-vs-CoT fine-tuning comparison and its robustness under LLM-generated phishing.** "LLM-rephrased phishing degrades detectors" is NOT novel (arXiv:2507.07406, 2411.13874, 2510.11915). Hypothesis H1 may be false; a null result is an acceptable, planned outcome.

## Current status (honest)
- Done: Week 1 (plan/lit/RQ), Week 2 (data built + test locked). [R][U]
- Infra-validated only (placeholder 3-row data, Qwen2.5-1.5B, T4): Weeks 5–6 fine-tuning notebook, both arms. **No real result exists yet.** [U]
- Week 3 classical baseline: notebook written, **never executed**; the original was lost in a sandbox reset and was rebuilt for this handover (`baselines/Week3_Classical_Baseline.ipynb`, UNRUN). [R]
- Not started: Week 4, 7–12. Teacher script for CoT `reasoning` column does not exist. [R]
- Rough completion estimate ≈ 25–30% of plan-weeks (Claude's judgment, not a measurement). Calendar time elapsed since proposal (2026-08-21) ≈ 7 weeks → **check the real deadline immediately** [?].

## Authoritative artifacts
| Artifact | Role |
|---|---|
| `README.md`, `literature-notes.md` | Authoritative RQ/hypothesis/literature (corrected versions) [R] |
| `data/data_summary.md`, `data/Week2_Data_Acquisition.ipynb` | Authoritative dataset definition [R] |
| `finetuning/Phishing_SLM_QLoRA_Starter.ipynb` | Fine-tuning pipeline (hazards in `04`, `06`) [R] |
| `baselines/Week3_Classical_Baseline.ipynb` | UNRUN [R] |
| Google Drive `MyDrive/phishing-slm-research/` | Holds the actual CSV splits + adapters (NOT in git) [U, unverified listing] |

**SUPERSEDED / PARTLY WRONG (do not trust their Project C passages):** `LLM_Cybersecurity_Research_Proposal.md`, `Project_Guide_SLM_vs_LLM_Phishing_Detection.{md,docx,pdf}` — they call the SOC paper "DartLab", claim CoT-distillation "fixes" naive fine-tuning (false per the paper's own table), cite the IEEE Access DOI as unverified, and plan Nazario (not used). `Progress_Report`/`Detailed_Progress_Evidence` are consistent with the corrected state. These files are not in the repo.

## Immediate next task
1. Run the verification plan in `09` (do not modify anything first).
2. Obtain the 3 CSV splits (see `03` §Regeneration) and run the Week 3 baseline on train/val only.
3. Then Week 4 zero-shot baselines (builds the label parser reused in Week 8).
4. Do NOT run real fine-tuning until the hazards in `06` (silent CoT fallback, truncation, bf16-on-T4, teacher script) are fixed.

## Top warnings
1. `phishing_test.csv` is LOCKED until Week 8 — never load, inspect, tune on, or print it.
2. No numbers from placeholder runs may appear in any result table.
3. Never fabricate results, citations, paths. Use `[?]`.
4. Naive vs CoT arms must differ ONLY in target text. Same model, hyperparameters, seed, data rows.
5. `USE_COT=True` silently degrades to label-only if the `reasoning` column is missing/NaN (see `06`).

## Read next
`01` science → `03` data → `04` code → `06` bugs/hazards → `07` next steps → `08` operating rules → `09` verification → `10` bootstrap prompt.
