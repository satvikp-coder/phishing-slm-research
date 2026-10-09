# 09 — Context Gaps and Verification Plan

## Context-coverage statement
Claude had access to: **one conversation thread** (this one), the files it created in a sandbox that resets between turns, and outputs the student pasted. Claude did **not** have: Claude Project knowledge/instructions (none observed), earlier conversations, the student's local machine, GitHub, Google Drive, any supervisor communication, or the Colab runtime. Some earlier web-search/tool results were no longer in Claude's working context when writing this; literature details therefore come from files Claude wrote earlier (`README.md`, `literature-notes.md`) and are tagged [S]. The repo inspected is a **restored zip copy** (HEAD `b560753`) plus this handover's additions.

## Unknowns
Institution, department, supervisor, start date, deadline, submission/venue target; GitHub remote & pushed state; Drive contents; local hardware/OS/Python; library versions; HF dataset revision; API provider/model/budget; teacher model; Springer paper authors; arXiv:2411.13874 title; Greco et al. link; real runtimes; email-length distribution; label-semantics purity; whether the `eval`/`test` splits of the HF dataset overlap `train`; exact per-step cleaning counts (Week 2 cells 7–8).

## Contradictions / stale items
README Week 5/6 `[x]` vs "infra-validated only"; Project Guide/Proposal vs corrected literature (see `02`); "few hundred dedup" vs ≈40; "<1 h per run" vs 3,900-step estimate; Progress_Report says Week 3 "code built" but file had been lost.

## Verification checklist for Codex (report before changing anything)
1. `git log`, `git status`, remote; confirm HEAD includes handover commit; list tracked files vs `04` map.
2. Read README, literature-notes, data_summary, three notebooks; list discrepancies.
3. Locate the 3 CSVs (Drive/local). Check columns `email_id,subject,body,label,dataset_name`; row counts 20,681/2,954/5,910; label rates; no duplicate `email_id` across splits; **no body-hash overlap between splits (exact)**; compute near-dup rate (MinHash) as info only. Do not open test content beyond shape/hash — row counts and hashes only.
4. Compute email length distribution on **train only** (chars/tokens) → decide truncation (hazard 2).
5. Verify adapters exist on Drive (`…-naive`, `…-cot`) — treat as disposable placeholder artifacts; do not use for results.
6. Environment: Python/CUDA/GPU, installable versions; run a tiny smoke test of the SFT path with a **fixed seed** on a 200-row train subset to measure it/s and memory; record.
7. Run Week 3 notebook/script on train/val; compare against expectation.
8. Confirm no code path touches the test file.
9. Write `docs/handover/VERIFICATION_REPORT.md` (findings, deviations, decisions needed) and wait for the student's go-ahead on Phase 2.

## Final QC audit (Claude's self-check)
| Item | Result |
|---|---|
| Objectives/method preserved | Yes |
| Dataset lineage/transforms | Yes, from notebook code [R]; per-step counts missing |
| Experiments/results | Yes; all placeholder (stated) |
| Implemented vs proposed separated | Yes |
| Failed approaches / bugs | Yes |
| Professor instructions | Only one item exists; stated |
| Current state & next task | Yes |
| Missing sources disclosed | Yes (above) |
| Codex can continue without chat | Yes, after verification; student answers to `07` questions needed |
