# 01 — Research Master Context

## Three-level description
- **A (executive):** Benchmark small open LLMs (≤3B, QLoRA) vs a proprietary API and TF-IDF for phishing-email detection; test whether reasoning-trace-distilled fine-tuning is more robust than label-only fine-tuning on LLM-generated phishing.
- **B (scientific):** See RQ/H1 below. Contribution = fine-tuning-recipe comparison under distribution shift (human-written corpora → LLM-generated emails), plus cost/latency trade-offs.
- **C (engineering):** HF dataset → stratified CSV splits on Drive → sklearn TF-IDF baselines → zero-shot prompting (SLM + API) → TRL SFTTrainer QLoRA (naive & CoT arms) → unified evaluator appending to `eval/results/*.csv` → bootstrap/McNemar stats → paper.

## Research question (final, Week 1) [R: README]
Does reasoning-trace-distilled (CoT) LoRA fine-tuning of small (≤3B) language models improve phishing-email classification robustness to LLM-rephrased/generated phishing more than naive label-only LoRA fine-tuning, and how do both compare to strong zero-shot prompting (SLM and proprietary)?

## Hypothesis H1 [R]
CoT-distilled LoRA yields smaller in-distribution→adversarial degradation than naive LoRA, because reasoning traces teach transferable diagnostic cues (sender/link mismatch, urgency, pretext plausibility) rather than surface lexical patterns easily erased by paraphrase.
**Caveat (keep):** the only closest recipe evidence (arXiv:2512.06660, different task) shows naive ≈ CoT, both below strong prompting. H1 is open. Null result is reportable (fallback: "phishing classification resists the reasoning-distillation recipe").

## Planned comparison arms [P unless noted]
| Arm | Description | Status |
|---|---|---|
| B1 | TF-IDF + LogisticRegression; TF-IDF + calibrated LinearSVC (CV grid on train only) | notebook written, UNRUN |
| B2 | Proprietary LLM zero-/few-shot (provider/model [?], spend cap planned $30–50, set? [?]) | not started |
| B3 | Zero-shot SLMs (no fine-tune) | not started |
| F1 | Naive QLoRA fine-tune (`email → label`) | pipeline validated on 3 rows only |
| F2 | CoT QLoRA fine-tune (`email → reasoning + "Final answer: label"`) | pipeline validated on 3 rows only; teacher data absent |
| opt | Header-feature ablation (cf. Springer header paper) | optional |

**Candidate SLMs [P]:** Qwen2.5-1.5B-Instruct (only one ever run [U]), Qwen2.5-3B-Instruct, Llama-3.2-1B/3B-Instruct (HF-gated: license/token needed [?]), Phi-3.5-mini-instruct.
**Eval sets:** (i) locked in-distribution test (5,910); (ii) adversarial LLM-generated set: 200–500 phishing + matched AI-written legitimate emails, generated locally with an open model, 15–20% manually spot-checked, test-only, never trained on, documented in `data/generation_methodology.md` (not yet created) [P]. Note: related work used LLM-*rephrased* emails; this plan uses LLM-*generated* — provisional.
**Ablations [P]:** LoRA rank {8,16,32} on 1–2 models; train fraction {25,50,100}%; optional header features.
**Stats [P]:** bootstrap 95% CI on F1 (~1000 resamples) per method per test set; McNemar (or paired bootstrap) naive vs CoT on identical items; qualitative error analysis 10–15 misclassifications.
**Tables/Figures [P]:** T1 in-dist comparison (acc/P/R/F1/AUROC/FPR/cost/latency); T2 adversarial; T3 naive vs CoT ablation; F1 cost-vs-F1 scatter; F2 in-dist→adversarial drop per model.

## Metric definitions (do not change silently)
Positive class = label 1 (phishing/spam). Accuracy, Precision, Recall, F1 (primary, `sklearn`), AUROC (needs scores/probs; for LLM arms define score source explicitly [?]), **FPR = FP/(FP+TN)**, cost = USD/1000 emails, latency = ms/email. LLM-arm AUROC definition is UNDECIDED.

## Technical concepts (project-relevant only)
- **LoRA:** W' = W + (α/r)·BA, r=16, α=32 → scale 2; targets `all-linear`; dropout 0.05. **QLoRA:** base in 4-bit NF4 + double-quant; compute dtype currently bfloat16 [R] (T4 lacks native bf16 → see `06`). Trainable params for Qwen2.5-1.5B: 18,464,768 / 1,562,179,072 = 1.18% [U].
- **CoT distillation:** teacher LLM writes short rationale per training email; student SFT'd on rationale + final label. Teacher choice (API vs open 7–8B) undecided [?].
- **Why prediction parsing matters:** fine-tuned-on-3-rows models kept generating text after the label [U]; evaluator must parse the first label (naive) / `Final answer:` (CoT) deterministically and define behavior for unparseable output.

## Literature
**A. Reviewed** (details/links in `literature-notes.md` [R]; characterizations [S]):
| Ref | Role |
|---|---|
| arXiv:2511.15434 Goldenits et al., SLMs for phishing *website* detection (15 open SLMs, HTML, no fine-tune; cost/perf/privacy) | Table-1 template; closest sibling |
| IEEE Access 13:28335–28352 (2025), DOI 10.1109/ACCESS.2025.3540075, Zhang et al., 12 LLMs, SME phishing email | Zero-shot LLM baseline method |
| arXiv:2512.06660 Muzammil et al., SLMs for KQL security query generation (Table 3 as read [S]: prompting NL2KQL 0.988/0.960 syntax/semantic; supervised LoRA 0.911/0.726; CoT LoRA 0.909/0.716; two-stage multi-agent 0.987/0.906) | Fine-tuning recipe + the *near-tie* finding |
| Springer chapter DOI 10.1007/978-981-96-6941-7_12, header features + LLM, Enron+Nazario, ~99.8% acc; authors [?] | Header ablation |
| Array 30:100775 (2026), DOI 10.1016/j.array.2026.100775, Sivaneswaran et al., PRISMA SLR (36 studies) | "Dual-use" framing |
| arXiv:2507.07406 Thapa et al., Quantized LLMs vs classical models (rephrased phishing degrades both) | **Closest prior work** |
| arXiv:2411.13874 (title/authors [?]), commercial filters vs LLM-rephrased phishing | supports non-novelty of degradation |
| arXiv:2510.11915 Kulal et al., robust classical ML + preprocessing vs adversarial phishing | comparator |
| arXiv:2508.21457 Chen et al., SoK LLM-generated phishing campaigns | intro citation |
| Greco et al., "David versus Goliath…", ITASEC 2024 (link [?]) | adversarial-set design ideas |
**B. Identified, NOT reviewed (check overlap):** arXiv:2506.13746 (LLM phishing detection, self-consistency/faithfulness); arXiv:2602.02641 (zero/few-shot phishing URL LLM benchmark); arXiv:2503.22038 (debate multi-agent LLM phishing email); arXiv:2406.06663 (SecureNet DeBERTa vs LLMs); arXiv:2601.21261 (RAG+LLM phishing); arXiv:2602.05484 (prompt injection vs LLM phishing detectors; seen only in a result list); arXiv:2507.17978 (MeAJOR phishing-email corpus); Springer "Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct" (10.1007/s13369-026-11412-w) — **potentially overlapping LoRA-on-Qwen spam work; read first**.
**C. Future:** none agreed.
**D. Gap (as recorded):** no found paper combines multi-model QLoRA ≤3B + naive-vs-CoT head-to-head + in-dist vs LLM-generated phishing *email* evaluation. Absence-of-evidence caveat: re-search with alternate terms (rationale tuning, distillation, spam, smishing).

## Alternatives considered (Aug 2026 proposal, [R: proposal md in outputs, not repo])
Claude's weighted scores: A "source-code-grounded detection of AI-fabricated vulnerability reports" 8.9; B "quantization × security-refusal erosion" 7.8; **C (this project) 7.4**. Student/supervisor chose C. Reason for not following the higher-scored A: [?]. Do not revisit unless asked.
