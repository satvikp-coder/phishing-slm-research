# Project operating instructions

Read `RESEARCH_HANDOVER_AUDIT.md`, `docs/handover/00_START_HERE.md`, and the relevant research files before changing methods. Treat local code and saved outputs as direct evidence, the handover's `[U]` items as historical reports, `[P]` as plans, and `[?]` as unknown. Preserve the established research question and professor-approved Project C scope; document any proposed scientific change before implementation.

## Data and evaluation boundaries

- Keep `phishing_test.csv` and any future adversarial test set locked until the planned final evaluation. Do not read, sample, print, tune on, or use test content during development. Presence, file size, and cryptographic checksum checks are allowed without opening the content for analysis.
- Never overwrite original raw data or the existing authoritative splits. Record source revision, input SHA256, transformation versions, seed, row counts, schema, and class/source distributions when data becomes available. If counts differ from the reported 20,681/2,954/5,910, stop and investigate rather than silently regenerating.
- Train and select models using train/CV; use validation for development scoring. Keep final tests reserved. Check exact and near duplicates across splits without using locked test contents before final evaluation; report any limitation that cannot be measured under the lock.
- Label 1 is the positive phishing/spam class; label 0 is legitimate/ham. FPR means FP/(FP+TN). Use the documented metric definitions and make the primary robustness measure explicit before experiments. Do not describe mixed spam labels as a pure phishing corpus.
- Naive and CoT arms must use the same model, rows, seed, prompt, truncation, loss policy, optimizer settings, and evaluation protocol. Their supervised target text is the only planned difference. Fail if CoT rationales are missing or inconsistent with labels.

## Experiments and provenance

- Three-row QLoRA runs are infrastructure checks only. Never report their loss, inference, or adapters as research performance.
- Before real training, remove silent placeholder-data and CoT fallbacks; handle truncation so labels remain in the supervised target; choose a precision supported by the actual GPU; decide loss masking; set seeds; and log versions and hardware.
- For every result, save the exact run configuration, Git commit, seed, data/model identifiers and hashes, library versions, hardware, wall time, predictions keyed by `email_id`, parse failures, and metric calculation. Keep results reproducible and distinguish measured results from estimates and predictions.
- Fit TF-IDF and tune hyperparameters on train only. API use requires response caching, an explicit spend cap, and no committed credentials. Do not launch new experiments during an audit-only task.
- Keep notebooks usable in Colab while sharing evaluation logic with scripts when created. Commit small metadata/result tables only after validation; never commit email CSVs, generated phishing samples, model weights, checkpoints, secrets, or personal data.

## Research record

- Preserve raw evidence and prior versions. Do not silently change source selection, splitting, label semantics, model arms, or scientific objectives. Record justified decisions and limitations in a versioned note.
- Verify citations and paper-specific numbers from primary sources before final use. The older Project Guide and Proposal contain superseded claims; the handover audit documents them.
- Report what ran, exact outputs, unverified items, and the next task after each research step. Use `Week N: <what/why>` commit messages for validated research changes. Keep GitHub history intact; avoid force-pushing.
