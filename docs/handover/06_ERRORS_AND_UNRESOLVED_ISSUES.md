# 06 — Errors, Debugging History, Unresolved Issues

## Resolved
| Problem | Cause | Fix | Status |
|---|---|---|---|
| `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'` [U] | Newer TRL renamed it | Use `max_length=1024` (committed) | Resolved |
| `NameError: name 'SFTConfig' is not defined` [U] | Root cause **unconfirmed** — likely lost/failed imports after session state change (the `trl` import was failing, see next row) | Re-run cells from the top | Resolved (cause [?]) |
| `AttributeError: partially initialized module 'wandb' … no attribute 'errors'` → `RuntimeError: Failed to import trl.trainer.sft_trainer` [U] | Colab's preinstalled `wandb` circular import triggered by TRL | `pip uninstall -y wandb` (now in Cell 1) **plus Runtime→Restart session** (module stays poisoned in memory otherwise) | Resolved; CoT arm trained afterwards |
| First CoT save landed in `-naive` path [U] | Student ran later cells without re-running config cell after restart | Re-run Cell 4 then 5→11; `-cot` path then printed | Resolved (student process error) |
| Week 3 notebook never delivered; original lost in sandbox reset | Claude built it but did not export; sandbox filesystem resets between turns | Rebuilt for handover | Resolved, but UNRUN |
| Wrong citations/claims about 2512.06660 | Claude error | Corrected in README/lit-notes | Resolved in repo; **persists in superseded docs** |

## UNRESOLVED — fix before real training (priority order)
1. **Silent CoT→naive fallback (critical).** Cell 6 uses `if USE_COT and "reasoning" in row and pd.notna(row.get("reasoning"))`. On the real CSV there is no `reasoning` column ⇒ `USE_COT=True` trains label-only targets but saves to `…-cot`. Mislabelled arm ⇒ invalid core comparison. Fix: `assert` column present and no NaN when `USE_COT`; fail loudly.
2. **Truncation can delete the answer (critical).** `max_length=1024` with prompt-first template; long emails push the response (label / "Final answer") past 1024 tokens ⇒ model trains on prompts with truncated/missing targets. Email length distribution unknown [?]. Fix: truncate the email body to a token budget *before* formatting (same budget at eval), or measure and filter; log % truncated.
3. **bf16 on T4.** T4 (Turing) has no native bf16; notebook sets `bnb_4bit_compute_dtype=torch.bfloat16` and `bf16=True`. It ran, but speed/numerics unmeasured. Consider fp16 compute on T4, or an Ampere+ GPU. Decide and keep identical across arms/models.
4. **Loss includes prompt tokens.** Plain `text` field ⇒ loss over the whole sequence, not completion-only. For classification SFT this dilutes signal and favours learning email text. Decide (completion-only/assistant-only loss) deliberately; apply to both arms; document.
5. **No teacher script / no `reasoning` data** (see `03`). CoT arm cannot run.
6. **No seeds, no eval during training, no run metadata.**
7. **Training-time estimate unverified.** 20,681 rows × 3 epochs ÷ eff. batch 16 ≈ 3,900 steps/run; CoT sequences longer. Measure on 200 steps first; Colab free-tier sessions/idle limits may force `resume_from_checkpoint` support or fewer epochs/subsampling (changing this is a D11 decision — log it).
8. **Output parsing undefined.** Needs deterministic first-label extraction; policy for unparseable outputs (count as wrong? report rate); for CoT use `Final answer:`; prefer constrained decoding/logit comparison of the two label tokens for naive arm (also gives AUROC scores) — proposal, not agreed.
9. **Gated models.** Llama-3.2 needs HF license acceptance/token [?]; Phi-3.5 chat/template quirks unchecked.
10. **Data risks** in `03` (label semantics, source-identity shortcut, near-dup leakage, era drift).
11. **Possible overlap paper** (Springer GRPO+LoRA Qwen2.5-3B spam) not read.
12. **README stale**: marks Week 5/6 `[x]` and says "dataset pending"; should read "infra-validated only".
13. **Colab vs local split**: data/checkpoints on Drive; Codex's local machine may lack GPU/data.
14. **Deadline unknown**; ~7 calendar weeks elapsed vs ~2 completed plan-weeks.
15. **API budget cap** set in provider dashboard? [?]; provider/model [?].

## Failed/abandoned scientific approaches
None experimentally failed yet. Abandoned planning choices listed in `02`.
