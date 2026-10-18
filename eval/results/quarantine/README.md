# Quarantined Evaluation Results: Groq Zero-Shot Baseline (n=1,850)

## Overview
This directory stores quarantined baseline artifacts that have not met full audit verification standards.

The zero-shot run on Groq (`qwen/qwen3.8-27b`) on an incomplete validation slice ($n=1,850 / 2,954$) recorded anomalous metrics:
- Accuracy: 0.3254
- Precision: 0.2864
- Recall: 0.3926
- F1-Score: 0.3312
- False Positive Rate: 0.7244 (770 false positives out of 1,063 legitimate emails)

## Quarantine Rationale & Audit Directives
1. **Unverified Model Metadata**: The run was requested with `model="qwen/qwen3.8-27b"`. The exact `resp.model` string returned by Groq API response headers was not preserved in version control.
2. **Missing Raw Outputs**: The saved predictions file (`preds_groq_qwen_qwen3_8-27b_val.csv`) recorded only `email_id` and parsed `y_pred`, omitting the raw model output text (`resp.choices[0].message.content`).
3. **No Unsubstantiated "Label Inversion" Claims**: Without auditing the raw output strings, it is unscientific to assert "label inversion". High FPR could be caused by excessive refusal strings, verbose explanations mentioning both words, or prompt non-compliance.
4. **Dual Reporting Requirement**: For any LLM API evaluation, results must report confusion matrices BOTH ways:
   - **(a) Unparseable excluded and counted separately**
   - **(b) Unparseable counted as wrong (sanitized)**

## Audit Script & Retrospective Limitation
- `eval/audit_groq_outputs.py` was written to parse per-email raw cache JSONs if available.
- **Important Audit Finding:** During the original Colab run, per-email raw completion JSONs were **never saved or preserved** (only the parsed `email_id` and `y_pred` in `preds_groq_qwen_qwen3_8-27b_val.csv` were saved to Drive).
- Consequently, this specific $n=1,850$ row **cannot be retrospectively audited without a complete re-run**, which is currently **low priority** compared to the primary Arm A fine-tuning research objective. It remains strictly quarantined.

