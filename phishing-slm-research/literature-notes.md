# Literature Notes — Week 1

*One paragraph per paper, in my own words. This becomes the seed for the Related Work section in Week 12. All papers below were verified to exist via search on the date these notes were written; re-check before final submission in case of updates/revisions.*

---

## Core five (from the project guide)

### 1. Goldenits, König, Raubitzek, Ekelhart — "Small Language Models for Phishing Website Detection: Cost, Performance, and Privacy Trade-Offs" (arXiv:2511.15434, Nov 2025)
Benchmarks 15 open small language models (roughly 1B–70B parameters) on classifying phishing **websites** from raw HTML, with no fine-tuning — pure zero-shot/prompted evaluation. About 9 of the 15 models were judged "practical" (reliable structured output, reasonable latency), most clearing 80% accuracy, but even the best SLMs still trailed top proprietary LLMs on raw accuracy. Some models had strong precision but were disqualified for unreliable JSON-formatted output or weak recall. This is my direct template paper — I'm doing its email-text-and-adversarial-robustness sibling. I'll reuse its evaluation-table structure (model × accuracy × cost × latency) for my own Table 1.

### 2. Zhang, Wu, London, Tenney — "Benchmarking and Evaluating Large Language Models in Phishing Detection for Small and Midsize Enterprises: A Comprehensive Analysis" (IEEE Access 13, 28335–28352, 2025, DOI 10.1109/ACCESS.2025.3540075)
Benchmarks 12 LLMs (not specifically small models, no fine-tuning) on phishing-email detection aimed at an SME deployment context. Main finding: open-source models such as Llama-3-8B-Instruct are competitive with proprietary models and notably more cost-effective, using base/default parameters rather than heavy tuning. This gives me the zero-shot LLM baseline methodology for my Week 4 baseline arm, and a real precedent for the "open models are viable for SMEs" framing in my discussion section.

### 3. Muzammil et al. — "Towards Small Language Models for Security Query Generation in SOC Workflows" (arXiv:2512.06660, Dec 2025)
Tests small language models generating KQL security queries from natural-language requests, comparing strong prompting, supervised (naive) LoRA fine-tuning, reasoning-trace-distilled (CoT) LoRA fine-tuning, and a two-stage multi-agent (SLM-generates / LLM-judge-refines) architecture. **Correction to my earlier notes:** their actual numbers show both fine-tuning approaches (naive: 0.911 syntax/0.726 semantic; CoT: 0.909 syntax/0.716 semantic) underperforming the strong prompting baseline (0.988/0.960), and CoT-distillation did not clearly beat naive fine-tuning — the two are roughly tied, if anything naive was marginally better on their semantic metric. What won overall was the multi-agent architecture, not fine-tuning at all. This is the paper my fine-tuning recipe is borrowed from, but I need to frame my own RQ as testing whether this same near-tie between naive and CoT-distilled fine-tuning holds for a *classification* task (phishing) rather than assuming CoT-distillation will clearly win.

### 4. "Phishing Email Detection using Header Features Leveraging Large Language Models (LLMs)" (Springer, DOI 10.1007/978-981-96-6941-7_12)
Fine-tunes an LLM to classify phishing emails specifically using email **header** features (sender, routing, authentication-adjacent metadata) rather than body text, using the Enron Email Dataset and Nazario Phishing Corpus — the same two datasets I'm using. Reports very high performance (99.81% accuracy, 99.66% precision, 99.77% recall, 99.71% F1). This is my source for the optional header-feature ablation (Week 10) — if I have time, I can test whether adding header features on top of body-text classification helps my fine-tuned SLMs the way it apparently helped their model.

### 5. Sivaneswaran, Hewage, Herath, Rathore, Singh, Jiang — "A systematic literature review of large language models in phishing attack generation and detection" (Array 30, 100775, 2026, DOI 10.1016/j.array.2026.100775)
A PRISMA-methodology systematic review screening 142 records (Jan 2023–Apr 2025) and analyzing 36 eligible studies on the "dual role" of LLMs in both generating and detecting phishing content. Key observed trend: most existing studies use manually generated (not publicly benchmarked) datasets, and GPT-family models are studied far more than other architectures — a gap my project partially addresses by using public benchmark corpora (Enron/Nazario/CEAS) rather than only manually curated data. This is my source for the "dual-use" framing language in my introduction, and a good citation for "most prior work skips public benchmark datasets" as a small secondary contribution of my own dataset choices.

---

## Close prior work found during the novelty check (not in the original reading list — found while verifying #3, and important to cite)

### 6. Thapa, Chahal, Voinea Gabreanu, Otoum — "Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models" (arXiv:2507.07406, Jul 2025)
Compares classical ML/DL against quantized small LLMs on phishing-email detection, tests zero-shot/few-shot prompting, and — importantly — already shows that **LLM-rephrased phishing emails significantly degrade both ML-based and LLM-based detectors.** Also benchmarks cost/VRAM trade-offs (DeepSeek-R1-Distill-Qwen-14B at Q8_0: 80%+ accuracy, 17GB VRAM). This is the single closest paper to my adversarial-robustness angle. My project must clearly state what I add beyond this: a controlled naive-vs-CoT-distilled LoRA fine-tuning comparison, which this paper does not do (it doesn't fine-tune at all).

### 7. Gmail Spam Filter / SpamAssassin / Proofpoint vs. LLM-rephrased phishing study (arXiv:2411.13874)
Tests real commercial detectors and classic ML models against LLM-rephrased phishing emails, finds accuracy declines across the board for all detectors tested. Reinforces that the "adversarial rephrasing hurts detection" finding is already well-established — not my novel contribution, but useful supporting citation.

### 8. Kulal, Arannonu, Anwar, Rastogi, Niyaz — "Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing" (arXiv:2510.11915)
Proposes a text-preprocessing pipeline (spelling correction, word-splitting) specifically to counter LLM-generated/adversarially-perturbed phishing, evaluated with classical ML, achieving 94.26% accuracy / 84.39% F1 with degradation testing against adversarial phishing. No SLM or fine-tuning involved. Good comparator for how much a *classical, non-LLM* defense can close the adversarial gap, versus my fine-tuning-based approach.

### 9. Chen, Wu, Nguyen, Rudolph — "SoK: LLM-Generated Textual Phishing Campaigns — End-to-End Analysis of Generation, Characteristics, and Detection" (arXiv:2508.21457)
A systematization-of-knowledge paper mapping the generation methods, content characteristics, and countermeasures specific to LLM-generated phishing campaigns. Good high-level "state of the field" citation for my introduction and related work, to show I've surveyed the SoK-level view, not just individual papers.

### 10. Greco, Desolda, Esposito, Carelli — "David versus Goliath: can machine learning detect LLM-generated text? A case study in the detection of phishing emails" (Italian Conference on CyberSecurity, 2024)
Tests whether ML-based detectors can distinguish LLM-generated phishing text from human-written phishing text (a slightly different question from "can they detect phishing at all"). Relevant background for my adversarial test-set design in Week 7 — worth checking their generation methodology for ideas.

---

## What's still genuinely open (my actual contribution)
No paper found combines: (a) controlled QLoRA fine-tuning of multiple small (≤3B) models, (b) a head-to-head naive-vs-CoT-distilled fine-tuning comparison specifically, and (c) evaluation on both in-distribution and LLM-rephrased/generated adversarial phishing **email** text. Papers #1 and #6–9 establish the broader landscape (SLM cost/performance trade-offs for websites; adversarial degradation for emails) but none test the fine-tuning-recipe question. Paper #3 tests the fine-tuning-recipe question but for a different (structured-generation) task with an inconclusive result. My project sits precisely in the gap between these.
