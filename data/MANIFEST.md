# Dataset Cryptographic Manifest

This document records the immutable checksums, row counts, and class balance of the processed dataset splits.
In accordance with **docs/handover/08**, `phishing_test.csv` remains strictly locked and must never be loaded, evaluated, or accessed prior to Week 8.

## Split Summary

| Split | File | Row Count | Phishing Rate (`label=1`) | SHA-256 Checksum |
|---|---|---|---|---|
| **Train** | `phishing_train.csv` | 20,681 | 42.25% | `81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825` |
| **Validation** | `phishing_val.csv` | 2,954 | 42.25% | `3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f` |
| **Test (LOCKED)** | `phishing_test.csv` | 5,910 | 42.25% | `6ca5d30bd4a1f4252cf2639d714d51b87f2b022763666fb78e112d8352de69d5` |
| **Total** | | 29,545 | | |

## Clarification on the 5,909 vs 5,910 Row Count Discrepancy

Earlier draft notes occasionally cited **5,909** test rows, whereas the actual count is **5,910** rows.
This was an off-by-one error caused by confusing the zero-based index of the final row (`index = 5,909`) with the cardinal row count (`len(df) = 5,910`):
- `range(5910)` spans indices `0` through `5,909`.
- The dataset partition: `20,681` (train) + `2,954` (validation) + `5,910` (test) = `29,545` total emails.
- Exactly 5,910 distinct emails are preserved in the locked test set.
