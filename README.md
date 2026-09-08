# Small but Sufficient? SLM vs. LLM Phishing-Email Detection

## Research Question (finalized, Week 1)
Does reasoning-trace-distilled (CoT) LoRA fine-tuning of small (≤3B) language models improve phishing-email classification robustness to LLM-rephrased/generated phishing more than naive label-only LoRA fine-tuning, and how do both compare to strong zero-shot prompting (SLM and proprietary)?

This extends the fine-tuning-recipe question tested for a structured-generation task (SOC KQL-query generation) in arXiv:2512.06660 into a natural-language classification task — in a space where the broader "SLM-vs-LLM performance degrades under AI-rephrased phishing" pattern is already established (arXiv:2507.07406, arXiv:2411.13874, arXiv:2510.11915), but the specific fine-tuning-recipe question (naive vs. CoT-distilled) has not been tested for this task.

## Hypothesis
H1: CoT-distilled LoRA fine-tuning will produce smaller accuracy degradation than naive LoRA fine-tuning when moving from in-distribution (human-written) phishing to LLM-rephrased/generated adversarial phishing, because the reasoning trace teaches the model transferable diagnostic cues (sender/link mismatch, urgency framing, pretext plausibility) rather than surface-level lexical patterns that are more easily erased by paraphrasing.

**Honest caveat:** the closest related paper (arXiv:2512.06660) found CoT-distilled and naive fine-tuning performed similarly on their task (both underperformed strong prompting). H1 may not hold — see the Failure Strategy in the project guide for why a null result here is still a reportable finding.

## Status
- [x] Week 1: literature grounding, environment setup, RQ finalized
- [x] Week 2: dataset acquisition and cleaning — 20,681 train / 2,954 val / 5,910 test (locked), ~42.3% phishing/spam, stratified by label × source. See `data/data_summary.md`.
- [ ] Week 3: classical baseline
- [ ] Week 4: zero-shot LLM baselines
- [x] Week 5: naive LoRA fine-tuning — pipeline validated end-to-end on Qwen2.5-1.5B-Instruct (placeholder data; real dataset pending Week 2)
- [x] Week 6: CoT-distilled LoRA fine-tuning — pipeline validated end-to-end on Qwen2.5-1.5B-Instruct (placeholder data; real dataset pending Week 2)
- [ ] Week 7: adversarial test set construction
- [ ] Week 8: main experiment
- [ ] Week 9: sanity checks, lock results
- [ ] Week 10: ablations
- [ ] Week 11: statistics, error analysis
- [ ] Week 12: write-up

## Dataset
Built in `data/Week2_Data_Acquisition.ipynb` from `puyang2025/seven-phishing-email-datasets` (Hugging Face) — combines Enron, CEAS-08, SpamAssassin, and TREC-05/06/07 into one unified, cleaned, stratified train/val/test split. Full details and known limitations in `data/data_summary.md`. **Test set is locked — do not open again until Week 8.**

## Fine-tuning notebook
`finetuning/Phishing_SLM_QLoRA_Starter.ipynb` — Colab-ready QLoRA fine-tuning pipeline. Mounts Google Drive so checkpoints survive disconnects; single `USE_COT` toggle switches between the naive (Week 5) and CoT-distilled (Week 6) arms so the two are directly comparable. Validated Sep 2026 on Qwen2.5-1.5B-Instruct, Tesla T4, naive arm — trains, saves adapter to Drive, and produces coherent inference output. **Known issue to handle in the Week 8 eval harness:** the model doesn't reliably stop generating right after the label even in the naive arm; label-parsing must extract from the start of the response rather than assume a bare single-word output.

Both `Qwen2.5-1.5B-Instruct-naive` and `Qwen2.5-1.5B-Instruct-cot` adapters confirmed saved to separate Drive folders, and produce genuinely different inference outputs on the same test email (naive: "Phishing", CoT: "legitimate" — expected instability from training on 3 placeholder rows, not a meaningful result; both arms need re-validation once real data is loaded).

Model checkpoints themselves are NOT tracked in this repo (see `.gitignore`) — they live in Google Drive under `phishing-slm-research/finetuning/checkpoints/`.

## Repo structure
```
data/{raw,processed}/
baselines/
finetuning/checkpoints/
eval/results/
analysis/
paper/
literature-notes.md   <- Week 1 deliverable
README.md             <- this file
```

## Key related work (see literature-notes.md for full paraphrased notes)
1. Goldenits, König, Raubitzek, Ekelhart. "Small Language Models for Phishing Website Detection: Cost, Performance, and Privacy Trade-Offs." arXiv:2511.15434 (Nov 2025).
2. Zhang, Wu, London, Tenney. "Benchmarking and Evaluating Large Language Models in Phishing Detection for Small and Midsize Enterprises." IEEE Access 13, 28335–28352 (2025). DOI 10.1109/ACCESS.2025.3540075.
3. Muzammil et al. "Towards Small Language Models for Security Query Generation in SOC Workflows." arXiv:2512.06660 (Dec 2025).
4. [Springer authors]. "Phishing Email Detection using Header Features Leveraging Large Language Models (LLMs)." Springer, DOI 10.1007/978-981-96-6941-7_12.
5. Sivaneswaran, Hewage, Herath, Rathore, Singh, Jiang. "A systematic literature review of large language models in phishing attack generation and detection." Array 30, 100775 (2026). DOI 10.1016/j.array.2026.100775.

**Close prior work found during novelty check (must cite, must differentiate from):**
6. Thapa, Chahal, Voinea Gabreanu, Otoum. "Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models." arXiv:2507.07406 (Jul 2025).
7. (2024) Study evaluating Gmail Spam Filter/SpamAssassin/Proofpoint/classic ML against LLM-rephrased phishing. arXiv:2411.13874.
8. Kulal, Arannonu, Anwar, Rastogi, Niyaz. "Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing." arXiv:2510.11915.
9. Chen, Wu, Nguyen, Rudolph. "SoK: LLM-Generated Textual Phishing Campaigns — End-to-End Analysis of Generation, Characteristics, and Detection." arXiv:2508.21457.
10. Greco, Desolda, Esposito, Carelli. "David versus Goliath: can machine learning detect LLM-generated text? A case study in the detection of phishing emails." Italian Conference on CyberSecurity (2024).

## Compute budget
- Single consumer GPU (8–16GB) or free-tier Colab, QLoRA only, ~5–15 GPU-hours total.
- API spend capped at $30–50.

## Ethical note
Any LLM-generated phishing emails produced for the adversarial test set (Week 7) are generated and stored strictly for defensive benchmarking, are not distributed, and generation methodology is documented in `data/generation_methodology.md`.
