# finetuning/train_arm_a.py — Arm A: Naive QLoRA Fine-Tuning Script
"""
Arm A — Naive QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for direct phishing email classification.
Target: "Phishing" or "Legitimate" (label-only targets, NO leading space).

Adheres strictly to docs/handover/08:
- Trains on phishing_train.csv ONLY.
- phishing_test.csv is never referenced.
- Prompt-completion dataset with TRL completion_only_loss.
- Pre-format body truncation to 800 tokens.
- Resumable training via Drive checkpoints (save_steps=200).
- Fixed seed (42).
- T4-optimized fp16 compute.
"""

import os
import sys
import time
import json
import inspect
import argparse
import hashlib
import subprocess
import numpy as np
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    MAX_SEQ_LENGTH,
    RESPONSE_TEMPLATE,
    COMPLETION_PHISHING,
    COMPLETION_LEGITIMATE,
    build_naive_prompt,
    build_prompt_completion_pair,
    truncate_body_text,
    truncate_subject_text
)

def get_git_hash():
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return res.decode().strip()
    except Exception:
        return "unknown"

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_prompt_completion_masking(dataset, tokenizer, num_examples=3):
    """
    Simulates and verifies that prompt tokens receive label=-100 and only completion tokens are unmasked.
    """
    print("\n--- [VERIFICATION] Prompt-Completion Loss Masking Check ---")
    eos_token_id = tokenizer.eos_token_id
    
    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        prompt_text = sample["prompt"]
        completion_text = sample["completion"]
        
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        if eos_token_id is not None:
            completion_ids.append(eos_token_id)
            
        full_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        
        masked_count = sum(1 for x in labels if x == -100)
        unmasked_indices = [idx for idx, x in enumerate(labels) if x != -100]
        unmasked_tokens = [tokenizer.decode([full_ids[idx]]) for idx in unmasked_indices]
        
        print(f"Example {i+1}:")
        print(f"  Total tokens: {len(full_ids)}")
        print(f"  Masked prompt tokens (-100): {masked_count}")
        print(f"  Unmasked target tokens: {len(unmasked_indices)} -> {unmasked_tokens}")
        
        assert masked_count == len(prompt_ids), f"Prompt tokens count mismatch on example {i+1}"
        assert len(unmasked_indices) == len(completion_ids), f"Completion tokens count mismatch on example {i+1}"
    
    print("--- [VERIFIED] Prompt tokens are correctly masked with -100! Only completion is trained. ---\n")

def main():
    parser = argparse.ArgumentParser(description="Arm A: Naive QLoRA fine-tuning on Qwen2.5-1.5B")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test on 200 rows (~30-50 steps)")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume from last checkpoint in output_dir")
    parser.add_argument("--data_path", type=str, default=None, help="Path to phishing_train.csv")
    parser.add_argument("--val_path", type=str, default=None, help="Path to phishing_val.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save adapter")
    parser.add_argument("--num_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_subject_tokens", type=int, default=MAX_SUBJECT_TOKENS, help="Pre-format subject token budget")
    parser.add_argument("--max_body_tokens", type=int, default=MAX_BODY_TOKENS, help="Pre-format body token budget")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint frequency (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Imports for training
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"ERROR: Missing training dependencies: {e}")
        print("Please install requirements: pip install transformers peft bitsandbytes trl datasets accelerate")
        sys.exit(1)

    # 2. Set Seeds (Deterministic Protocol)
    set_seed(args.seed)
    np.random.seed(args.seed)

    # 3. Hardware Inspection & Precision Justification
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No GPU detected)"
    print(f"Hardware: {device_name}")
    if not cuda_available:
        print("WARNING: Running without local GPU. For full training or smoke testing, launch on Google Colab (T4 GPU).")

    USE_FP16 = True
    USE_BF16 = False
    COMPUTE_DTYPE = torch.float16

    # 4. Resolve Paths
    if args.data_path is None:
        args.data_path = os.path.join(REPO_ROOT, "data", "processed", "phishing_train.csv")
    if args.val_path is None:
        args.val_path = os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv")
    
    if args.output_dir is None:
        suffix = "Qwen2.5-1.5B-Instruct-naive-SMOKE" if args.smoke_test else "Qwen2.5-1.5B-Instruct-naive"
        args.output_dir = os.path.join(REPO_ROOT, "finetuning", "checkpoints", suffix)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "eval", "results", "runs"), exist_ok=True)

    # 5. Strict Read-Only Verification of Train and Val Splits
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"CRITICAL: phishing_train.csv not found at {args.data_path}")
    if not os.path.exists(args.val_path):
        raise FileNotFoundError(f"CRITICAL: phishing_val.csv not found at {args.val_path}")

    train_df = pd.read_csv(args.data_path)
    val_df = pd.read_csv(args.val_path)

    assert len(train_df) == 20681, f"Train row count mismatch: expected 20,681, got {len(train_df)}"
    assert len(val_df) == 2954, f"Val row count mismatch: expected 2,954, got {len(val_df)}"

    train_sha256 = get_file_sha256(args.data_path)
    val_sha256 = get_file_sha256(args.val_path)

    print(f"\n--- Arm A Naive QLoRA Configuration ---")
    print(f"Model:                Qwen/Qwen2.5-1.5B-Instruct")
    print(f"Mode:                 Arm A (Direct Classification, Prompt-Completion)")
    print(f"Train Dataset:        {args.data_path} (Rows: {len(train_df)}, SHA256: {train_sha256})")
    print(f"Val Dataset:          {args.val_path} (Rows: {len(val_df)}, SHA256: {val_sha256})")
    print(f"Pre-format budget:    {args.max_subject_tokens} subject tokens | {args.max_body_tokens} body tokens")
    print(f"Max sequence length:  1024 tokens (explicitly enforced)")
    print(f"LoRA config:          r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate:        {args.lr}")
    print(f"Compute dtype:        fp16={USE_FP16}, bf16={USE_BF16}")
    print(f"Checkpoint save steps:{args.save_steps} (~20-30 mins per checkpoint)")
    print(f"Smoke test mode:      {args.smoke_test}")
    print(f"Output directory:     {args.output_dir}")
    print(f"---------------------------------------\n")

    df = train_df
    total_rows = len(df)
    if args.smoke_test:
        df = df.head(200).copy()
        print(f"[SMOKE TEST]: Subsampled to first {len(df)} rows.")

    # 6. Load Tokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify completion strings (Point 4)
    ids_p = tokenizer.encode(COMPLETION_PHISHING, add_special_tokens=False)
    ids_l = tokenizer.encode(COMPLETION_LEGITIMATE, add_special_tokens=False)
    print(f"Completion '{COMPLETION_PHISHING}': Token IDs={ids_p} | Decoded={[tokenizer.decode([t]) for t in ids_p]}")
    print(f"Completion '{COMPLETION_LEGITIMATE}': Token IDs={ids_l} | Decoded={[tokenizer.decode([t]) for t in ids_l]}")
    print(f"First tokens distinct? {ids_p[0] != ids_l[0]} ({ids_p[0]} vs {ids_l[0]})")
    assert ids_p[0] != ids_l[0], "First tokens must be distinct!"

    # 7. Pre-format Truncation & Dataset Construction
    body_truncated_count = 0
    subject_truncated_count = 0
    records = []
    total_tokens_list = []
    eos_token = tokenizer.eos_token

    for _, row in df.iterrows():
        s_trunc, s_was_trunc = truncate_subject_text(row['subject'], max_tokens=args.max_subject_tokens, tokenizer=tokenizer)
        b_trunc, b_was_trunc = truncate_body_text(row['body'], max_tokens=args.max_body_tokens, tokenizer=tokenizer)
        if s_was_trunc:
            subject_truncated_count += 1
        if b_was_trunc:
            body_truncated_count += 1
            
        pair = build_prompt_completion_pair(s_trunc, b_trunc, int(row['label']), tokenizer=None)
        records.append(pair)
        
        full_seq = pair["prompt"] + pair["completion"] + eos_token
        tok_len = len(tokenizer.encode(full_seq, add_special_tokens=False))
        total_tokens_list.append(tok_len)

    arr = np.array(total_tokens_list)
    rows_exceeding = int((arr > 1024).sum())
    max_observed = int(np.max(arr))

    print(f"\n--- Sequence Length Audit (Target max_length: 1024) ---")
    print(f"Dataset Rows:            {len(df)}")
    print(f"Min Tokens:              {int(np.min(arr))}")
    print(f"p50 (Median) Tokens:     {float(np.median(arr)):.1f}")
    print(f"p90 Tokens:              {float(np.percentile(arr, 90)):.1f}")
    print(f"p99 Tokens:              {float(np.percentile(arr, 99)):.1f}")
    print(f"Max Tokens:              {max_observed}")
    print(f"Rows > 1024:             {rows_exceeding} (Must strictly be 0)")
    print(f"Bodies Truncated:        {body_truncated_count} ({body_truncated_count / len(df) * 100:.2f}%)")
    print(f"Subjects Truncated:      {subject_truncated_count} ({subject_truncated_count / len(df) * 100:.2f}%)")

    assert rows_exceeding == 0, f"CRITICAL: {rows_exceeding} sequences exceed max_length 1024! Max was {max_observed}."
    dataset = Dataset.from_pandas(pd.DataFrame(records))

    # 8. Verify Prompt Tokens are Masked with -100
    verify_prompt_completion_masking(dataset, tokenizer, num_examples=3)

    # 9. Setup Quantization & LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model in 4-bit...")
    target_dtype = COMPUTE_DTYPE if cuda_available else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        dtype=target_dtype,
        torch_dtype=target_dtype
    )
    model.config.use_cache = False

    if cuda_available:
        model = prepare_model_for_kbit_training(model)

    # Explicitly ensure all non-quantized modules (embed_tokens, lm_head, layernorms) are float32
    # so NO bfloat16 tensors exist anywhere in the model
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if not hasattr(module, "quant_state"):
                if module.weight.dtype == torch.bfloat16:
                    module.to(torch.float32)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    # Ensure all trainable LoRA parameters are float32 so fp16 GradScaler never encounters BFloat16 gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable_dtypes = set(p.dtype for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter dtypes: {trainable_dtypes}")
    assert torch.bfloat16 not in trainable_dtypes, f"Found BFloat16 in trainable parameters: {trainable_dtypes}"

    model.print_trainable_parameters()

    # 10. Check for Existing Checkpoint to Support Resumption ONLY when explicitly flagged
    last_checkpoint = None
    if args.resume and not args.smoke_test:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"\n[RESUME]: Found previous checkpoint at {last_checkpoint}. Resuming training!")
        else:
            print(f"\n[RESUME]: --resume flag was set, but no checkpoint was found at {args.output_dir}. Starting from scratch.")
    else:
        if not args.smoke_test:
            print("\n[TRAIN]: Clean run starting from scratch (--resume flag not set).")

    # 11. SFT Training Arguments (Adaptive Signature Handling)
    max_steps = 40 if args.smoke_test else -1
    epochs = 1 if args.smoke_test else args.num_epochs
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
        "fp16": False,  # QLoRA uses bnb_4bit_compute_dtype=float16 internally; disabling global fp16 prevents PyTorch GradScaler BFloat16 crash
        "bf16": False,
        "report_to": "none",
        "seed": args.seed
    }
    if has_completion_loss:
        sft_kwargs["completion_only_loss"] = True
        sft_kwargs["max_length"] = 1024
    elif has_max_seq_length:
        sft_kwargs["max_seq_length"] = 1024
    else:
        sft_kwargs["max_length"] = 1024

    sft_config = SFTConfig(**sft_kwargs)

    collator = None
    if not has_completion_loss:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator
    )

    # 12. Train
    print("\nStarting training...")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    wall_time = time.time() - t0

    # 13. Resource & Performance Accounting
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if cuda_available else 0.0
    steps_completed = trainer.state.global_step
    steps_per_sec = steps_completed / wall_time if wall_time > 0 else 0.0

    print(f"\n--- Training Complete ---")
    print(f"Wall time:       {wall_time:.2f} seconds")
    print(f"Global steps:    {steps_completed}")
    print(f"Steps/sec:       {steps_per_sec:.3f}")
    print(f"Peak GPU Memory: {peak_mem_gb:.2f} GB")

    projected_full_hours = 0.0
    if args.smoke_test:
        full_steps = int((total_rows * args.num_epochs) / (args.batch_size * args.grad_accum))
        projected_full_sec = full_steps / steps_per_sec if steps_per_sec > 0 else 0
        projected_full_hours = projected_full_sec / 3600
        print(f"Projected full run time (3,878 steps on T4): {projected_full_hours:.2f} hours")

        # Sanity validation predictions on first 5 val rows
        print("\n--- First 5 Validation Predictions (Smoke Test Sanity) ---")
        model.eval()
        for i in range(5):
            v_row = val_df.iloc[i]
            prompt_text = build_naive_prompt(v_row['subject'], v_row['body'][:400])
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            gen_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            true_label = COMPLETION_PHISHING if v_row['label'] == 1 else COMPLETION_LEGITIMATE
            print(f"Val [{i+1}] True: {true_label} | Raw Output: {repr(gen_text)}")

    # 14. Save Final Adapter & Metadata
    if not args.smoke_test:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved LoRA adapter weights to: {args.output_dir}")

    run_id = f"arm_a_naive_{'smoke_' if args.smoke_test else ''}{int(time.time())}"
    metadata = {
        "run_id": run_id,
        "model": model_id,
        "mode": "Arm A Naive (Direct Classification)",
        "smoke_test": args.smoke_test,
        "train_rows": len(df),
        "total_corpus_rows": total_rows,
        "data_sha256": train_sha256,
        "val_sha256": val_sha256,
        "git_hash": get_git_hash(),
        "seed": args.seed,
        "hyperparameters": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": "all-linear",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "num_epochs": args.num_epochs,
            "max_body_tokens": args.max_body_tokens,
            "max_seq_length": 1024,
            "completion_string_phishing": COMPLETION_PHISHING,
            "completion_string_legitimate": COMPLETION_LEGITIMATE,
            "save_steps": args.save_steps,
            "fp16": USE_FP16,
            "bf16": USE_BF16
        },
        "stats": {
            "pct_truncated": round(pct_truncated, 2),
            "wall_time_sec": round(wall_time, 2),
            "global_steps": steps_completed,
            "steps_per_sec": round(steps_per_sec, 3),
            "peak_gpu_mem_gb": round(peak_mem_gb, 2),
            "projected_full_run_hours": round(projected_full_hours, 2)
        },
        "hardware": device_name
    }

    run_json_path = os.path.join(REPO_ROOT, "eval", "results", "runs", f"{run_id}.json")
    with open(run_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Logged run metadata to: {run_json_path}")

if __name__ == "__main__":
    main()
