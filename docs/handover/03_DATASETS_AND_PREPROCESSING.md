# 03 — Datasets and Preprocessing

## Inventory
| Dataset | Source | Status / authority |
|---|---|---|
| HF `puyang2025/seven-phishing-email-datasets` | Hugging Face. Sub-corpora: Enron, CEAS-08, Assassin, TREC-05/06/07, Ling. Total 203,017 per card [S]; `train` split loaded = **162,413** rows [U] (a `test` split of 40,604 rows also generated [U]; an `eval` parquet downloaded [U]; only `train` used) | RAW. Revision/commit hash NOT pinned [?] |
| Raw per-source counts (train split) [U] | TREC-05 44,461; TREC-07 42,964; CEAS-08 31,150; Enron 23,849; TREC-06 13,113; Assassin 4,585; Ling 2,291 (sum = 162,413 ✔) | |
| `phishing_train.csv` | Drive `.../data/processed/` [U] | **AUTHORITATIVE training data**, 20,681 rows, 42.3% label 1 |
| `phishing_val.csv` | same | **AUTHORITATIVE val**, 2,954 rows, 42.2% |
| `phishing_test.csv` | same | **LOCKED**, 5,910 rows, 42.3%. Never opened since creation [U, by student discipline] |
| `data/data_summary.md` | repo [R] | Documentation of above |
| Nazario corpus | — | NOT USED |
| Adversarial LLM-generated set | — | not created; spec in `01` |
| CoT `reasoning` column | — | **not created**; no teacher script |
Total after filter/cap/dedup = 29,545 (20,681+2,954+5,910) [U]. Actual file existence/row counts on Drive: [?] (listing never viewed).

## Schema (final CSVs) [R: Week2 notebook cell 9]
`email_id` (int, 0..N-1 *before* splitting, so unique across splits), `subject` (str, NaN→""), `body` (str; renamed from HF `text`), `label` (int; **1 = phishing/spam, 0 = legitimate/ham** per notebook mapping; semantics from dataset card [S]), `dataset_name` (str source). HF columns used: `text, subject, label, dataset_name`. Other HF columns (card mentions sender/receiver/date/urls) [?] — dropped.
Fine-tuning notebook expects `subject, body, label` and optionally `reasoning`.

## Pipeline (Input → Transformation → Output → Reason → Code)
| Step | Detail | Code |
|---|---|---|
| 1 Load | `load_dataset(repo)["train"].to_pandas()` | Week2 cell 5 |
| 2 Filter | keep `dataset_name ∈ {Enron, CEAS-08, Assassin, TREC-05, TREC-06, TREC-07}` | cell 6 |
| 3 Drop | rows with NaN `text`/`label`, or empty/whitespace `text` | cell 7 |
| 4 Cap | per-source `sample(n=min(len,5000), random_state=42)` via `groupby.apply` | cell 7 |
| 5 Dedup | MD5 of `text.strip().lower()`; `drop_duplicates` (keeps first) — **exact match only** | cell 8 |
| 6 Rename/select | `text→body`; `email_id=range(len)`; cols `[email_id, subject, body, label, dataset_name]` | cell 9 |
| 7 Split | key = `label_str + "_" + dataset_name`; `train_test_split(test_size=0.30, stratify=key, random_state=42)` then split temp with `test_size=2/3` → val 10% / test 20% | cell 10 |
| 8 Save | 3 CSVs to Drive | cell 11 |
Scientifically sensitive: cap-then-dedup order (changes which rows survive); seed 42; pandas/sklearn version-dependent sampling; stratification key.

## Regeneration (needed because CSVs live only on Drive)
Preferred: copy the 3 CSVs from Drive. Fallback: rerun Week2 logic locally with identical seed/cap/sources, then **assert** row counts 20,681 / 2,954 / 5,910 and label rates 42.3/42.2/42.3%. If counts differ (HF revision or library version drift), STOP and report — do not silently proceed. After locating files, record SHA256 of all three in `data/MANIFEST.md` [P].

## Known data risks (state in paper)
1. Historical (~1999–2008) email only; modern style absent → motivates adversarial set.
2. `label=1` mixes spam and phishing for some corpora [S]; spot-check ~50–100 random positives per source before trusting "phishing" wording [P].
3. Exact-dup only; near-duplicate/quoted-thread leakage across splits uncontrolled [S card warning]. Optional MinHash pass [P].
4. Sources have very different label ratios/styles → models may learn **source identity**. Recommend per-source metric breakdown (the `dataset_name` column is retained for this) [P].
5. 42% positive rate ≠ deployment prevalence.
6. HF dataset's own `test`/`eval` splits ignored; any overlap with `train` unchecked [?].
7. Body length distribution unknown [?] — matters for truncation (see `06`).

## CoT training data spec [P]
For each **train** row only (never val/test): teacher outputs short rationale citing concrete cues, ending with the true label; store as `reasoning`; generate once, cache to disk (JSONL, keyed by `email_id`), record teacher model/version/prompt/temperature. Validate: rationale's stated label == true label (filter or flag mismatches — rationale hallucination risk), length bounds, no empty rows.
