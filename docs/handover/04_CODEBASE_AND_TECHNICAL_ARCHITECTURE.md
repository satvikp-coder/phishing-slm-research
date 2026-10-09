# 04 — Codebase and Technical Architecture

## Environment
- Runtime used: Google Colab free tier, **Tesla T4**, Python 3.13 (from traceback path) [U]. Local OS/GPU/Python/venv: [?]. Library versions: [?] (installed unpinned; TRL is recent — it renamed `max_seq_length→max_length` and prints "Adding EOS to train dataset" [U]). `requirements.txt` does not exist. → Codex: pin versions after first successful run.
- Planned packages: transformers, peft, bitsandbytes, trl, datasets, accelerate, scikit-learn, pandas, numpy, matplotlib.
- `wandb` is **uninstalled** in notebook cell 1 (broken circular import on Colab) — don't re-add; `report_to="none"`.
- Git: branch `main`, 5 commits to `b560753` (+ this handover's commit). Remote/GitHub: [?]. Student workflow: Claude gave notebook → student ran in Colab → pasted output → Claude committed/zipped. Codex replaces that loop.

## Repo map [R]
```
README.md                       status checklist (STALE: marks Wk5/6 [x], says "dataset pending")
literature-notes.md             10 papers, own-words notes
.gitignore                      ignores *.csv except eval/results/*.csv (fixed in handover commit)
data/Week2_Data_Acquisition.ipynb   builds splits (executed by student on Colab [U])
data/data_summary.md
data/{raw,processed}/.gitkeep   (CSVs intentionally untracked)
finetuning/Phishing_SLM_QLoRA_Starter.ipynb   executed on placeholder data only [U]
finetuning/checkpoints/.gitkeep (adapters on Drive, untracked)
baselines/Week3_Classical_Baseline.ipynb      UNRUN (rebuilt)
eval/results/.gitkeep           baseline_results.csv will be written here
docs/handover/                  this package
```
Empty planned dirs (analysis/, paper/) are untracked. No `.py` modules, no tests, no evaluator, no zero-shot script exist.

## Notebook: `data/Week2_Data_Acquisition.ipynb` (17 cells) [R]
Purpose: build splits (see `03`). Config cell: `SOURCES_TO_USE`, `MAX_ROWS_PER_SOURCE=5000`, `RANDOM_SEED=42`, fracs .70/.10/.20, output paths under `DRIVE_ROOT`. Final cell writes `data_summary.md` text; the committed `data_summary.md` was hand-written from the student's pasted numbers, not generated.
Drive pattern: `IN_COLAB = "google.colab" in str(get_ipython())`; else `DRIVE_ROOT="."`.

## Notebook: `finetuning/Phishing_SLM_QLoRA_Starter.ipynb` (15 cells) [R]
| Cell | Content |
|---|---|
| 1 | pip install transformers peft bitsandbytes trl datasets accelerate scikit-learn pandas; `pip uninstall -y wandb` |
| 2 | Drive mount |
| 3 | imports (`SFTTrainer, SFTConfig`, `LoraConfig, get_peft_model, TaskType`, `BitsAndBytesConfig`) + GPU check |
| 4 | Config: `MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"`, `USE_COT=False`, `DATA_PATH=.../phishing_train.csv`, `OUTPUT_DIR=.../checkpoints/{model}-{cot|naive}`, `LORA_R=16, ALPHA=32, DROPOUT=0.05, NUM_EPOCHS=3` |
| 5 | Load CSV; **falls back to 3 synthetic rows if file missing** (this is what produced all runs to date) |
| 6 | `format_example`: `### Instruction:\nClassify the following email as 'Phishing' or 'Legitimate'.\n\n### Email:\nSubject: {s}\n\nBody: {b}\n\n### Response:\n{target}`; target = `Phishing`/`Legitimate`, or CoT: `{reasoning}\n\nFinal answer: {label}` |
| 7 | 4-bit NF4, double-quant, compute dtype **bfloat16**; `device_map="auto"`; pad=eos if missing; `use_cache=False` |
| 8 | LoRA r16/α32/drop .05, `target_modules="all-linear"`, CAUSAL_LM |
| 9 | `SFTConfig`: epochs 3, per-device batch 4, grad-accum 4 (eff. 16), lr 2e-4, logging 10, save per epoch, `bf16=cuda`, `report_to="none"`, `dataset_text_field="text"`, `max_length=1024`; `SFTTrainer(model, args, train_dataset)` |
| 10 | `trainer.save_model(OUTPUT_DIR)`; tokenizer save |
| 11 | greedy `generate(max_new_tokens=120)` on one hard-coded email |
Not present: seed setting, eval split/metrics during training, prediction export, completion-only loss, prompt/length truncation logic, resume-from-checkpoint, run-metadata logging.
Template is a custom "### Instruction" format, not the tokenizer chat template.

## Notebook: `baselines/Week3_Classical_Baseline.ipynb` (11 cells, UNRUN) [R]
TF-IDF(stop_words="english") + LogisticRegression(max_iter=1000) and + `CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)`; GridSearchCV cv=3, scoring f1, grid max_features {5k,10k,20k} × ngram {(1,1),(1,2)} × C {.1,1,10}; fit on train; evaluate on **val**; metrics acc/P/R/F1/AUROC/FPR, fit time, ms/email, cost 0; writes `eval/results/baseline_results.csv` with columns `model, split, …`, replacing rows with same model+split. Risks: `clf__estimator__C` naming depends on sklearn ≥1.2; whole-grid runtime on 20k emails unknown; stop-word list may hurt phishing cues (untested design choice).

## Needed but absent (build next, in this order)
1. `src/` package or scripts replacing notebooks for local runs (optional proposal; keep notebooks working for Colab).
2. Shared `eval/` module: prompt builder (single source of truth for the template), output parser, metric computation, results writer (same schema as baseline CSV), bootstrap CI + McNemar utilities.
3. Zero-shot runner (SLM local + API with disk cache keyed by (model, prompt-hash, email_id)).
4. CoT teacher-generation script (see `03`).
5. Adversarial-set generator + `generation_methodology.md`.
6. `requirements.txt`, `data/MANIFEST.md`, run-metadata logging (model, seed, git hash, hparams, timing).
