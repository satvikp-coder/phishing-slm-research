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

## Audit Script
Run `python eval/audit_groq_outputs.py` against the cached responses to generate the full 20-example raw output audit (10 FP, 10 FN) and dual confusion matrices.
