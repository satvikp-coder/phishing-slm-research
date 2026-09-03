# Small but Sufficient? SLM vs. LLM Phishing-Email Detection

## Research Question (finalized, Week 1)
Does reasoning-trace-distilled (CoT) LoRA fine-tuning of small (≤3B) language models improve phishing-email classification robustness to LLM-rephrased/generated phishing more than naive label-only LoRA fine-tuning, and how do both compare to strong zero-shot prompting (SLM and proprietary)?

This extends the fine-tuning-recipe question tested for a structured-generation task (SOC KQL-query generation) in arXiv:2512.06660 into a natural-language classification task — in a space where the broader "SLM-vs-LLM performance degrades under AI-rephrased phishing" pattern is already established (arXiv:2507.07406, arXiv:2411.13874, arXiv:2510.11915), but the specific fine-tuning-recipe question (naive vs. CoT-distilled) has not been tested for this task.

## Hypothesis
H1: CoT-distilled LoRA fine-tuning will produce smaller accuracy degradation than naive LoRA fine-tuning when moving from in-distribution (human-written) phishing to LLM-rephrased/generated adversarial phishing, because the reasoning trace teaches the model transferable diagnostic cues (sender/link mismatch, urgency framing, pretext plausibility) rather than surface-level lexical patterns that are more easily erased by paraphrasing.

**Honest caveat:** the closest related paper (arXiv:2512.06660) found CoT-distilled and naive fine-tuning performed similarly on their task (both underperformed strong prompting). H1 may not hold — see the Failure Strategy in the project guide for why a null result here is still a reportable finding.

## Status
- [x] Week 1: literature grounding, environment setup, RQ finalized
- [ ] Week 2: dataset acquisition and cleaning
- [ ] Week 3: classical baseline
- [ ] Week 4: zero-shot LLM baselines
- [ ] Week 5: naive LoRA fine-tuning
- [ ] Week 6: CoT-distilled LoRA fine-tuning
- [ ] Week 7: adversarial test set construction
- [ ] Week 8: main experiment
- [ ] Week 9: sanity checks, lock results
- [ ] Week 10: ablations
- [ ] Week 11: statistics, error analysis
- [ ] Week 12: write-up

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
