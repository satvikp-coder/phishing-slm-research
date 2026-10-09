# 10 — Bootstrap prompt (paste into Codex at repo root)

```
You are continuing an active, in-progress academic research project. This is a handover, not a restart. Do not redesign the project, change its research question, or repeat completed work.

1. Read, in order, every file in docs/handover/ (00 → 09). Treat evidence tags ([R] repo-verified, [U] student-pasted output, [S] search-derived, [P] planned, [?] unknown) literally. Never convert [?] or [P] items into facts.
2. Read README.md, literature-notes.md, data/data_summary.md, and the three notebooks (data/Week2_Data_Acquisition.ipynb, finetuning/Phishing_SLM_QLoRA_Starter.ipynb, baselines/Week3_Classical_Baseline.ipynb). Inspect `git log` and `git status`.
3. Obey docs/handover/08_CODEX_OPERATING_INSTRUCTIONS.md. In particular: NEVER open or use phishing_test.csv or any adversarial test set before the final Week 8 evaluation; no result from the 3-row placeholder runs is valid; naive vs CoT arms must differ only in target text; do not fabricate numbers, citations or paths.
4. Execute the verification checklist in docs/handover/09_CONTEXT_GAPS_AND_VERIFICATION.md. Make NO modifications to code, data, or git history during this step except creating docs/handover/VERIFICATION_REPORT.md.
5. In that report list: confirmed facts, discrepancies vs the handover, newly discovered risks, and the questions in 07 that need the student's answers (deadline, GitHub remote, Drive contents, hardware, API provider/budget, teacher model).
6. Stop and present the report. After the student approves, resume at the exact resumption point in docs/handover/07_CURRENT_STATE_AND_NEXT_STEPS.md: run the Week 3 classical baseline on train/val only, then build the Week 4 zero-shot baselines. Do NOT launch real fine-tuning until the unresolved hazards listed in 06 (silent CoT fallback, truncation, bf16-on-T4, loss masking, missing teacher data, seeds) are fixed and reviewed.
7. After every task report: files changed, commands run, exact outputs, what remains unverified, and the next step. Commit validated steps with messages of the form "Week N: ...".

Project in one line: compare naive vs reasoning-trace-distilled QLoRA fine-tuning of ≤3B SLMs against zero-shot SLMs, a proprietary LLM API and TF-IDF for phishing-email detection, in-distribution and on LLM-generated phishing (low-compute, single student, ~12-week plan).
```
