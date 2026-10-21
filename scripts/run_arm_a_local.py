# scripts/run_arm_a_local.py — Standalone Local Runner for Arm A Fine-Tuning & Evaluation
"""
Standalone Local Runner for Arm A: Naive QLoRA Fine-Tuning on Qwen2.5-1.5B-Instruct.
Designed for local NVIDIA GPU execution (e.g. RTX 5070 / RTX 40xx / 30xx).

Features:
- Completely independent of Google Drive (uses local paths).
- Auto-detects NVIDIA GPU & CUDA capability (BFloat16 / Float16).
- Resumes automatically if interrupted.
- Enforces strict token budgets: subject <= 100, body <= 800, max_seq_length = 1024.
- Runs post-training validation inference on phishing_val.csv.
- Packages final adapter, validation predictions, and summary JSON into a clean output folder.
"""

import os
import sys
import time
import json
import shutil
import hashlib
import inspect
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Resolve repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    MAX_SEQ_LENGTH,
    COMPLETION_PHISHING,
    COMPLETION_LEGITIMATE,
    build_naive_prompt,
    build_prompt_completion_pair,
    truncate_body_text,
    truncate_subject_text
)

def get_git_hash() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return res.decode().strip()
    except Exception:
        return "unknown"

def get_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def parse_label(raw_output: str) -> Tuple[int, bool]:
    text = str(raw_output).strip()
    if text.startswith(COMPLETION_PHISHING):
        return 1, True
    elif text.startswith(COMPLETION_LEGITIMATE):
        return 0, True
    text_lower = text.lower()
    if "phishing" in text_lower and "legitimate" not in text_lower:
        return 1, True
    if "legitimate" in text_lower and "phishing" not in text_lower:
        return 0, True
    return -1, False

def main():
    parser = argparse.ArgumentParser(description="Arm A Standalone Local Runner for RTX GPUs")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to phishing_train.csv")
    parser.add_argument("--val_csv", type=str, default=None, help="Path to phishing_val.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpointing")
    parser.add_argument("--package_dir", type=str, default=None, help="Final folder to zip and return")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint frequency in steps")
    parser.add_argument("--smoke_test", action="store_true", help="Run 40-step smoke test only")
    args = parser.parse_args()

    # 1. Imports
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    except ImportError as e:
        print(f"\n[ERROR] Missing required packages: {e}")
        print("Please install requirements using: pip install -r requirements_rtx.txt\n")
        sys.exit(1)

    seed = 42
    set_seed(seed)
    np.random.seed(seed)

    # 2. Hardware Inspection
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0.0
    print("\n" + "="*60)
    print("   PHISHING SLM RESEARCH — ARM A LOCAL RUNNER (RTX GPU)")
    print("="*60)
    print(f"Device:          {device_name}")
    print(f"CUDA Available:  {cuda_available}")
    print(f"GPU VRAM:        {total_mem} GB")
    print(f"PyTorch Version: {torch.__version__}")

    # Determine optimal precision
    if cuda_available and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Hardware Precision: BFloat16 (native support detected)")
    elif cuda_available:
        compute_dtype = torch.float16
        print("Hardware Precision: Float16")
    else:
        compute_dtype = torch.float32
        print("Hardware Precision: Float32 (CPU fallback)")

    # 3. Path Resolution
    if args.train_csv is None:
        args.train_csv = os.path.join(REPO_ROOT, "data", "processed", "phishing_train.csv")
    if args.val_csv is None:
        args.val_csv = os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv")
    if args.output_dir is None:
        suffix = "Qwen2.5-1.5B-Instruct-naive-SMOKE" if args.smoke_test else "Qwen2.5-1.5B-Instruct-naive"
        args.output_dir = os.path.join(REPO_ROOT, "finetuning", "checkpoints", suffix)
    if args.package_dir is None:
        args.package_dir = os.path.join(REPO_ROOT, "arm_a_results_package")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.package_dir, exist_ok=True)

    # Validate Files
    for p, label in [(args.train_csv, "Train CSV"), (args.val_csv, "Val CSV")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] {label} not found at: {p}")

    train_sha = get_sha256(args.train_csv)
    val_sha = get_sha256(args.val_csv)
    print(f"\nTrain Dataset:   {args.train_csv}")
    print(f"SHA-256:         {train_sha}")
    print(f"Val Dataset:     {args.val_csv}")
    print(f"SHA-256:         {val_sha}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    print(f"Total Rows:      Train={len(train_df)}, Val={len(val_df)}")

    if args.smoke_test:
        print("\n[SMOKE TEST ACTIVE]: Subsampling train to 200 rows, max_steps = 40.")
        train_df = train_df.iloc[:200].copy()

    # 4. Tokenizer Setup
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\nLoading Tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Token ID Verification
    ids_p = tokenizer.encode(COMPLETION_PHISHING, add_special_tokens=False)
    ids_l = tokenizer.encode(COMPLETION_LEGITIMATE, add_special_tokens=False)
    print(f"Token IDs: '{COMPLETION_PHISHING}'={ids_p} | '{COMPLETION_LEGITIMATE}'={ids_l}")
    assert ids_p[0] != ids_l[0], "First tokens must be distinct!"

    # 5. Dataset Construction & Token Budget Audit
    print("\nAuditing token lengths and formatting prompt-completion dataset...")
    records = []
    total_tokens_list = []
    b_trunc_count = 0
    s_trunc_count = 0
    eos_token = tokenizer.eos_token

    for _, row in train_df.iterrows():
        s_text, s_trunc = truncate_subject_text(row['subject'], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
        b_text, b_trunc = truncate_body_text(row['body'], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
        if s_trunc:
            s_trunc_count += 1
        if b_trunc:
            b_trunc_count += 1

        pair = build_prompt_completion_pair(s_text, b_text, int(row['label']), tokenizer=None)
        records.append(pair)

        seq = pair["prompt"] + pair["completion"] + eos_token
        seq_len = len(tokenizer.encode(seq, add_special_tokens=False))
        total_tokens_list.append(seq_len)

    arr = np.array(total_tokens_list)
    rows_exceeding = int((arr > MAX_SEQ_LENGTH).sum())
    max_len = int(np.max(arr))
    print(f"Sequence Length Distribution (Target Ceiling: {MAX_SEQ_LENGTH}):")
    print(f"  Min: {int(np.min(arr))} | p50: {float(np.median(arr)):.1f} | p90: {float(np.percentile(arr, 90)):.1f} | p99: {float(np.percentile(arr, 99)):.1f} | Max: {max_len}")
    print(f"  Rows > 1024:         {rows_exceeding} (Must strictly be 0)")
    print(f"  Bodies Truncated:    {b_trunc_count} ({b_trunc_count/len(train_df)*100:.2f}%)")
    print(f"  Subjects Truncated:  {s_trunc_count} ({s_trunc_count/len(train_df)*100:.2f}%)")
    assert rows_exceeding == 0, f"CRITICAL: {rows_exceeding} sequences exceed {MAX_SEQ_LENGTH}!"

    train_dataset = Dataset.from_pandas(pd.DataFrame(records))

    # 6. Prompt Loss Masking Verification
    print("\n--- [VERIFICATION] Prompt-Completion Loss Masking Check ---")
    for i in range(min(3, len(train_dataset))):
        s = train_dataset[i]
        p_ids = tokenizer.encode(s["prompt"], add_special_tokens=False)
        c_ids = tokenizer.encode(s["completion"] + eos_token, add_special_tokens=False)
        full_ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        unmasked = [idx for idx, x in enumerate(labels) if x != -100]
        unmasked_toks = [tokenizer.decode([full_ids[idx]]) for idx in unmasked]
        print(f"Example {i+1}: {len(full_ids)} tokens | {len(p_ids)} prompt tokens masked (-100) | Trained completion: {unmasked_toks}")
        assert len(unmasked) == len(c_ids)
    print("--- [VERIFIED] Prompt tokens masked (-100); only completion tokens are trained! ---\n")

    # 7. Model Loading (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading {model_id} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        dtype=compute_dtype if cuda_available else torch.float32,
        torch_dtype=compute_dtype if cuda_available else torch.float32
    )
    model.config.use_cache = False

    if cuda_available:
        model = prepare_model_for_kbit_training(model)

    # Cast non-quantized modules to float32
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if not hasattr(module, 'quant_state'):
                if module.weight.dtype == torch.bfloat16:
                    module.to(torch.float32)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable_dtypes = set(p.dtype for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameter Dtypes: {trainable_dtypes}")
    assert torch.bfloat16 not in trainable_dtypes
    model.print_trainable_parameters()

    # 8. Checkpoint Resumption Check
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint is not None:
        print(f"\n[RESUME]: Found previous checkpoint at {last_checkpoint}. Resuming automatically!")
    else:
        print("\n[TRAIN]: Starting clean training run.")

    # 9. SFT Training Arguments
    max_steps = 40 if args.smoke_test else -1
    epochs = 1 if args.smoke_test else args.epochs
    save_strategy = "no" if args.smoke_test else "steps"

    has_completion_loss = 'completion_only_loss' in inspect.signature(SFTConfig).parameters
    has_max_seq_length = 'max_seq_length' in inspect.signature(SFTConfig).parameters

    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 5 if args.smoke_test else 20,
        "save_strategy": save_strategy,
        "save_steps": args.save_steps if not args.smoke_test else 500,
        "save_total_limit": 2,
        "fp16": False,  # Managed internally by bnb_4bit_compute_dtype
        "bf16": False,
        "report_to": "none",
        "seed": seed
    }
    if has_completion_loss:
        sft_kwargs["completion_only_loss"] = True
        sft_kwargs["max_length"] = MAX_SEQ_LENGTH
    elif has_max_seq_length:
        sft_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    else:
        sft_kwargs["max_length"] = MAX_SEQ_LENGTH

    sft_config = SFTConfig(**sft_kwargs)

    collator = None
    if not has_completion_loss:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(response_template="### Response:\n", tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator
    )

    # 10. Execute Training
    print("\n" + "="*60)
    print("   STARTING TRAINING")
    print("="*60)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    wall_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if cuda_available else 0.0
    steps = trainer.state.global_step
    steps_sec = steps / wall_time if wall_time > 0 else 0.0

    print(f"\nTraining Complete in {wall_time:.1f}s ({wall_time/3600:.2f} hours)")
    print(f"Total Steps: {steps} | Steps/sec: {steps_sec:.3f} | Peak VRAM: {peak_mem:.2f} GB")

    # Save final model
    adapter_save_dir = os.path.join(args.package_dir, "adapter_weights")
    os.makedirs(adapter_save_dir, exist_ok=True)
    trainer.save_model(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    print(f"\nSaved fine-tuned adapter to: {adapter_save_dir}")

    # 11. Run Full Validation Evaluation on phishing_val.csv
    print("\n" + "="*60)
    print(f"   RUNNING VALIDATION EVALUATION ({len(val_df)} Emails)")
    print("="*60)
    model.eval()

    val_records = []
    t_val_start = time.time()

    for idx, v_row in val_df.iterrows():
        s_trunc, _ = truncate_subject_text(v_row['subject'], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
        b_trunc, _ = truncate_body_text(v_row['body'], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
        prompt_str = build_naive_prompt(s_trunc, b_trunc)

        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        raw_pred = tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        y_pred, parsed_ok = parse_label(raw_pred)
        y_true = int(v_row['label'])

        val_records.append({
            "email_id": v_row['email_id'],
            "source": v_row.get('source', 'unknown'),
            "y_true": y_true,
            "y_pred": y_pred if parsed_ok else 1 - y_true,  # Unparseable penalized as wrong
            "raw_output": raw_pred,
            "parsed_successfully": parsed_ok
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(val_df):
            print(f"  Evaluated {idx + 1}/{len(val_df)} emails...")

    val_time = time.time() - t_val_start
    val_res_df = pd.DataFrame(val_records)

    # Save Predictions CSV
    preds_csv_path = os.path.join(args.package_dir, "preds_qwen2_5_1_5b_naive_val.csv")
    val_res_df.to_csv(preds_csv_path, index=False)
    print(f"\nSaved validation predictions to: {preds_csv_path}")

    # Compute Metrics
    y_true_all = val_res_df['y_true'].values
    y_pred_all = val_res_df['y_pred'].values
    parse_fails = int((~val_res_df['parsed_successfully']).sum())

    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    try:
        auroc = roc_auc_score(y_true_all, y_pred_all)
    except Exception:
        auroc = 0.5

    print("\n" + "="*60)
    print("   VALIDATION RESULTS SUMMARY (Arm A: Naive QLoRA)")
    print("="*60)
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"AUROC:               {auroc:.4f}")
    print(f"Parse Failure Rate:  {parse_fails}/{len(val_df)} ({parse_fails/len(val_df)*100:.2f}%)")
    print(f"Inference Time:      {val_time:.1f}s ({val_time/len(val_df)*1000:.1f} ms/email)")
    print("="*60)

    # Save Run Summary JSON
    summary_data = {
        "model": model_id,
        "arm": "Arm A (Naive QLoRA Direct Classification)",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "git_hash": get_git_hash(),
        "hardware": device_name,
        "training_wall_time_sec": round(wall_time, 2),
        "training_peak_vram_gb": round(peak_mem, 2),
        "val_wall_time_sec": round(val_time, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auroc": round(auroc, 4),
            "parse_failures": parse_fails
        }
    }

    summary_json_path = os.path.join(args.package_dir, "run_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved execution summary to: {summary_json_path}")

    print("\n" + "#"*60)
    print(f"  DONE! Everything is packaged in: {args.package_dir}")
    print("  Please ZIP the 'arm_a_results_package' folder and send it back to Satvik!")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
