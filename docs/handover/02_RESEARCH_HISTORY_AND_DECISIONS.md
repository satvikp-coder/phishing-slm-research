# 02 — Research History, Decisions, Corrections

Dates are approximate (file timestamps / chat order). Calendar dates of student actions are [?].

## Timeline
| Period | Task | Method | Result | Evidence | Decision / consequence |
|---|---|---|---|---|---|
| ~2026-08-21 | Topic selection | Claude searched 2025–26 LLM×security literature; 13 candidate RQs scored; 3 finalists | Proposal md; Claude ranked A>B>C | proposal md [R-outputs] | Student/supervisor picked **C**; supervisor "accepted project 3" [U] |
| ~08-22 | Project guide | Claude wrote 12-week guide (md/docx/pdf) | Guide contains later-corrected errors (see Corrections) | outputs | Guide superseded in part |
| ~09-03 | Week 1 | Verified 5 core papers + found 5 more; finalized RQ; repo skeleton | README, literature-notes, 2 commits | repo [R] | Novelty narrowed to naive-vs-CoT comparison |
| ~09-03→08 | Weeks 5–6 infra test | Colab T4, Qwen2.5-1.5B, 3 placeholder rows, both arms | Both ran, adapters saved to Drive (`-naive`, `-cot`) | [U] outputs | Pipeline mechanically OK; results meaningless |
| ~09-08 | Week 2 data | HF dataset `puyang2025/seven-phishing-email-datasets`; Colab notebook | 20,681/2,954/5,910; test locked | [U] outputs, [R] notebook+summary | Substituted HF corpus for Nazario/CEAS/Enron downloads |
| after 09-08 | Ordering decision | Claude advised baselines (3–4) before real fine-tuning (5–6); student: "do which is best" | Week 3 notebook written | chat | Baselines first |
| ~09-29 | Reporting | Progress report + evidence docs (docx/pdf) for supervisor | ~25–30% complete estimate | outputs | Judgment estimate |
| 2026-10-09 | Handover to Codex | this package | — | — | — |

## Decision register
| # | Decision | Rationale | Status |
|---|---|---|---|
| D1 | Project C chosen over A, B | Supervisor acceptance [U]; reasons beyond that [?] | Final |
| D2 | RQ narrowed to naive-vs-CoT fine-tuning robustness | Degradation under rephrasing already shown by 3 papers | **DO NOT CHANGE WITHOUT JUSTIFICATION** |
| D3 | Test split locked until Week 8 | Prevent selection bias | **DO NOT CHANGE WITHOUT JUSTIFICATION** |
| D4 | Split 70/10/20 stratified on label×dataset_name, seed 42 | Preserve class & source mix | **DO NOT CHANGE WITHOUT JUSTIFICATION** |
| D5 | Baselines before real fine-tuning | Avoid anchoring; reuse parser/evaluator | Final (can be reordered only with reason) |
| D6 | Hyperparameter selection for baselines uses CV on train only; val for first scoring | No val leakage into selection | **DO NOT CHANGE** |
| D7 | Naive & CoT arms identical except target text | Controlled comparison | **DO NOT CHANGE** |
| D8 | Sources: Enron, CEAS-08, Assassin, TREC-05/06/07; excluded Ling | Standard, larger; Ling small/less cited | Provisional |
| D9 | Cap 5,000 rows/source (pre-dedup, post-dropna) | Prevent TREC/Enron dominance | Provisional |
| D10 | Checkpoints to Google Drive; adapters not in git | Colab disconnects; size | Final |
| D11 | Models list & LoRA r16/α32/3 epochs/lr 2e-4/eff. batch 16 | Defaults; untested on real data | Provisional |
| D12 | Adversarial set generated locally with open model; defensive use only; not distributed | Dual-use ethics | **DO NOT CHANGE WITHOUT JUSTIFICATION** |
| D13 | Metrics set in `01` | Standard | Final |
| D14 | Teacher for CoT traces (API vs open 7–8B) | — | **OPEN** |
| D15 | Proprietary comparator model/provider | — | **OPEN** |

## Corrections of earlier Claude statements (important — superseded text may still circulate)
1. Source paper arXiv:2512.06660 was mislabeled "DartLab"; real title *Towards Small Language Models for Security Query Generation in SOC Workflows*.
2. Claimed "naive fine-tuning underperformed prompting and CoT-distillation fixes it." Per Table 3 [S]: both fine-tunes < prompting; CoT ≈ naive (slightly lower semantic). Winner was a two-stage SLM+LLM-judge pipeline. H1 reframed accordingly.
3. IEEE Access DOI first cited without verification; existence later confirmed via search [S]. Springer header paper confirmed [S] (authors not captured).
4. Novelty originally overstated; arXiv:2507.07406 already shows adversarial degradation + quantized-LLM cost trade-offs.
5. Week 2: Claude said dedup removed "a few hundred" rows. Arithmetic: 5×5,000 + 4,585 = 29,585 vs final 29,545 → net **≈40** rows removed. Per-step counts (Cells 7–8 outputs) were not shared [?].
6. Claude earlier estimated "well under an hour per QLoRA run" — **unverified** and likely wrong for 20k rows × 3 epochs on T4 (~3,900 optimizer steps at eff. batch 16 for naive; longer for CoT). Measure it.
7. Week 3 notebook was announced but never delivered to the student; original lost; rebuilt here.

## Rejected / abandoned
Project A, B (not pursued). Separate Nazario/Enron/CEAS raw downloads (replaced by HF corpus). Nazario corpus: NOT used (must be stated in paper). Header-feature ablation: optional, undecided. Modern 2023–26 phishing sample (to reduce era drift): suggested, not done.

## Supervisor information
- Supervisor: Prof. Amit Das (AMSOM).
- Institution / Department: Inferred as Ahmedabad University, Amrut Mody School of Management (AMSOM, from the student's email domain `@ahduni.edu.in` in git metadata and supervisor signature "Amit Das, AMSOM"); marked as [inferred, confirm].
- Direct instruction/feedback:
  - 2026-08-21: Acceptance of Project C [U].
  - 2026-10-18: Prof. Amit Das (AMSOM) reviewed the Week 3 & 4 baseline report, noted that classical algorithms (TF-IDF+LogReg, TF-IDF+SVM) performed very well on F1 and AUROC, and explicitly instructed: *"Please proceed to your next task: Arm A Fine-Tuning: Naive QLoRA on Qwen2.5-1.5B (Direct Classification)"*.
- Claude's own suggestions (NOT supervisor's): 12-week plan, metrics, ablations, baselines-first ordering.
- Student requirements: low compute; paper-oriented; GitHub kept in sync with each validated step; Colab for GPU work; wants simple-language status text for supervisor.

