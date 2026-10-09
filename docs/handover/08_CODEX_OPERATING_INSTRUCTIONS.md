# 08 — Codex Operating Instructions

## Source-of-truth hierarchy
Local files/executed outputs > this handover > Claude's memory/superseded docs. Where handover and repo disagree, report the discrepancy; don't paper over it. Scientific evidence outranks both.

## Hard rules
1. **Never read, print, load, sample, or tune on `phishing_test.csv`** (or the future adversarial test set) before the designated Week 8 run. Add a guard in code (e.g., test path only reachable via an explicit `--final-eval` flag) and a repo check (`grep -r phishing_test`) before each commit.
2. Never edit raw data. Never regenerate splits unless verification fails — and then stop and ask.
3. No fabricated numbers, citations, links, paths. Use `UNKNOWN — REQUIRES VERIFICATION`.
4. Placeholder-run numbers are not results.
5. Naive vs CoT: change only the target text. Same model, seed, hyperparameters, rows, template, truncation, loss masking.
6. Don't change D2–D7, D12, metric definitions without written justification in `docs/DECISIONS.md` (create it) and a note to the student.
7. Ask before destructive actions (deleting checkpoints, force-push, rewriting history, overwriting results CSVs). Prefer new files/branches.
8. Respect the dual-use boundary: generate phishing emails only locally for the defensive adversarial test set; don't publish, post, or send them; keep them out of public commits unless the student decides (default: gitignore the generated set, commit only the methodology + a handful of redacted examples).
9. API spend: cache every response to disk; estimate cost before batch runs; stop at the student's cap. Never commit keys (`.env` is ignored).

## Engineering conventions
- One logical change per commit; message format `Week N: <what/why>` (matches history). Commit only validated steps. Keep `README.md` status honest (use "infra-validated", "run", "valid result").
- Notebooks remain runnable on Colab (Drive paths); add scripts for local use rather than breaking notebooks. Put shared logic in one importable module so notebooks and scripts don't diverge.
- Every run writes: model, adapter path, seed, hyperparameters, data checksums, git hash, wall time, GPU, library versions → `eval/results/runs/<id>.json`; per-example predictions → CSV keyed by `email_id`.
- Set seeds (python/numpy/torch/Trainer). Pin versions in `requirements.txt` after the first green run.
- Small results CSVs are tracked; emails/checkpoints are not.
- Report back after each task: what changed, what ran, exact outputs, what's unverified, next step. Prefer evidence (command + output) to assertions.

## Scientific hygiene
Fit/choose on train (CV) or val only. Report CIs. Report per-source and parse-failure rates. State limitations (era drift, label semantics, near-dups). Don't claim novelty beyond D2. If H1 fails, report it as planned fallback.
