# 05 — Experiments and Results

## Bottom line
**No scientifically valid experimental result exists yet.** Everything below is either infrastructure validation on 3 placeholder rows or dataset statistics. Do not cite any of it as a finding.

## Experiment registry
| ID | What | Config | Outcome | Evidence | Valid result? |
|---|---|---|---|---|---|
| E0-naive | Pipeline test, naive arm | Qwen2.5-1.5B-Instruct, 3 hard-coded rows, 3 epochs, T4 | Trainable 18,464,768/1,562,179,072 (1.1820%); global_step=3; train_loss 2.2201; runtime 15.27 s; mean_token_accuracy 0.5657; num_tokens 504; entropy 1.9389. Inference on 1 test email: began "Phishing" then kept generating an unrequested explanation | [U] | **NO** (n=3) |
| E0-cot | Pipeline test, CoT arm (fresh reloaded model) | same, rationale targets | global_step=3; train_loss 2.4737; runtime 10.58 s; mean_token_accuracy 0.5058; num_tokens 786. Formatted example verified to contain rationale + "Final answer: Phishing". Inference on same email: "legitimate" (a plausible-phishing email) with rationale | [U] | **NO** |
| E0-note | A first CoT attempt saved to `-naive` path because `USE_COT` cell not re-run after restart; student then re-ran and `-cot` path printed | — | Adapter folders: `Qwen2.5-1.5B-Instruct-naive` and `-cot` on Drive — **existence never verified by listing**; differing inference outputs are indirect evidence | [U] | — |
| D1 | Dataset build | see `03` | 162,413 loaded → 29,545 final; splits and balance as above | [U] | Yes (data stats) |
| B1 | TF-IDF LR / SVM on val | — | **NOT RUN** | — | — |
Training-loss comparison naive(2.22) vs CoT(2.47) is uninterpretable: different target lengths, 3 examples.

## Findings that ARE supportable
| Finding | Support | Confidence | Limits |
|---|---|---|---|
| Fine-tuning pipeline (load 4-bit → LoRA → SFT → save → generate) runs end-to-end on T4 for both target formats | E0 runs | High (mechanics only) | Not run on real data/size; bf16-on-T4 behavior unmeasured |
| Placeholder-trained models do not stop after the label | E0-naive | High | Expected at n=3; eval parser must be robust regardless |
| Splits are stratified (rates 42.3/42.2/42.3) | [U] | High | Per-source breakdown not shown |
| Prior literature: LLM-rephrased phishing degrades detectors; SLM-vs-LLM tradeoffs exist for websites; naive≈CoT for KQL | papers [S] | Medium (re-verify) | Different tasks/data |

## Planned results products
`eval/results/baseline_results.csv` (Table 1/2 rows appended), `ablation_results.csv`, per-example prediction files (needed for McNemar — **currently nothing exports predictions**), figures. Define `results/` schema once and reuse.

## Reproducibility status
Low: no seeds in fine-tuning notebook, unpinned libs, data only on Drive, no manifests, no run logs. Fix before real runs.
