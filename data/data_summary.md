# Data Summary — Week 2

**Source:** `puyang2025/seven-phishing-email-datasets` (Hugging Face), `train` split only.

**Sub-corpora used:** Enron, CEAS-08, Assassin (SpamAssassin), TREC-05, TREC-06, TREC-07
**Excluded:** Ling-Spam (smallest, least commonly cited in recent phishing-LLM literature)
**Per-source cap:** 5,000 rows (Assassin had fewer available — 4,585 — and was used in full)
**Random seed:** 42

## Raw per-source counts (before capping/cleaning)
| Source | Rows |
|---|---|
| TREC-05 | 44,461 |
| TREC-07 | 42,964 |
| CEAS-08 | 31,150 |
| Enron | 23,849 |
| TREC-06 | 13,113 |
| Assassin | 4,585 |
| Ling (excluded) | 2,291 |

## Final split sizes (after filtering, capping, dedup, stratified split)
| Split | Rows | % phishing/spam |
|---|---|---|
| Train | 20,681 | 42.3% |
| Validation | 2,954 | 42.2% |
| **Test (LOCKED as of this run — do not reopen until Week 8)** | 5,910 | 42.3% |

Stratified by `label` × `dataset_name` jointly, so class balance and source composition are both consistent across all three splits (confirmed: all three land within 0.1pp of each other on phishing/spam rate).

## Known limitations (state explicitly in the paper's Data/Limitations section)
- Does not include the Nazario corpus specifically; TREC-05/06/07 and SpamAssassin substitute as comparably standard, widely-cited benchmarks in the phishing-detection literature.
- The source dataset's `label=1` conflates general spam with phishing specifically for some sub-corpora (per the dataset's own documentation) — not a pure phishing-only signal. Worth a manual spot-check of a sample during Week 3 baseline work to gauge how much this matters in practice.
- Exact-duplicate emails were removed; near-duplicate quoted-thread/reply-chain text was **not** specifically deduplicated beyond exact match — a residual train/test leakage risk the dataset's own card explicitly warns about. Flag as a limitation; consider a stronger near-dedup pass (e.g. MinHash/SimHash) as a stretch goal if time allows.
- Source emails are historical (corpora span roughly 1999–2008). Modern phishing style may differ substantially — this is precisely why the Week 7 adversarial (LLM-generated) test set matters as a check on concept drift.
- Class balance sits at ~42% phishing/spam across all splits, which is a reasonable, not-too-imbalanced split for standard classification metrics (unlike, e.g., real-world email traffic where phishing is a small minority — worth noting this is a research-benchmark balance, not a deployment-realistic one).
