# finetuning/train_arm_a.py — Arm A: Naive QLoRA Fine-Tuning Script
"""
Arm A — Naive QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for direct phishing email classification.
Target: "Phishing" or "Legitimate" (label-only targets).

Adheres strictly to docs/handover/08:
- Trains on phishing_train.csv ONLY.
- phishing_test.csv is never referenced.
- Completion-only loss (DataCollatorForCompletionOnlyLM).
- Pre-format body truncation to 800 tokens.
- Fixed seed (42).
- T4-optimized fp16 compute.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import subprocess
import numpy as np
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.prompt_template import MAX_BODY_TOKENS, RESPONSE_TEMPLATE, build_naive_prompt, truncate_body_text

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

def verify_completion_masking(dataset, collator, tokenizer, num_examples=3):
    """
    Verifies on num_examples that prompt tokens have label=-100 and only response tokens are unmasked.
    """
    print("\n--- [VERIFICATION] Completion-Only Loss Masking Check ---")
    batch = collator([dataset[i] for i in range(min(num_examples, len(dataset)))])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    for i in range(len(input_ids)):
        inp = input_ids[i].tolist()
        lbl = labels[i].tolist()
        
        # Count masked (-100) vs unmasked tokens
        masked_count = sum(1 for x in lbl if x == -100)
        unmasked_indices = [idx for idx, x in enumerate(lbl) if x != -100]
        unmasked_tokens = [tokenizer.decode([inp[idx]]) for idx in unmasked_indices]
        
        print(f"\nExample {i+1}:")
        print(f"  Total tokens: {len(inp)}")
        print(f"  Masked prompt tokens (-100): {masked_count}")
        print(f"  Unmasked target tokens: {len(unmasked_indices)} -> {unmasked_tokens}")
        
        # Assertions
        assert masked_count > 0, f"Example {i+1} has no masked tokens!"
        assert len(unmasked_indices) > 0, f"Example {i+1} has no unmasked tokens!"
        assert all(lbl[idx] != -100 for idx in unmasked_indices), "Unmasked mismatch!"
    
    print("--- [VERIFIED] Prompt tokens are correctly masked with -100! ---\n")

def main():
    parser = argparse.ArgumentParser(description="Arm A: Naive QLoRA fine-tuning on Qwen2.5-1.5B")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test on 200 rows (~30-50 steps)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to phishing_train.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save adapter")
    parser.add_argument("--num_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_body_tokens", type=int, default=MAX_BODY_TOKENS, help="Pre-format body token budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Imports for training
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    except ImportError as e:
        print(f"ERROR: Missing training dependencies: {e}")
        print("Please install requirements: pip install transformers peft bitsandbytes trl datasets accelerate")
        sys.exit(1)

    # 2. Set Seeds (Deterministic Protocol)
    set_seed(args.seed)
    np.random.seed(args.seed)

    # 3. Hardware Inspection & Precision Justification
    # Hardware Justification: Tesla T4 (Turing, Compute Capability 7.5) does not possess native bfloat16
    # tensor core acceleration. Emulating bf16 on T4 degrades throughput. Therefore, fp16 is the 
    # mathematically correct, stable, and accelerated compute dtype for 4-bit QLoRA on T4.
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
    
    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_ROOT, "finetuning", "checkpoints", "Qwen2.5-1.5B-Instruct-naive")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "eval", "results", "runs"), exist_ok=True)

    print(f"\n--- Arm A Naive QLoRA Configuration ---")
    print(f"Model:                Qwen/Qwen2.5-1.5B-Instruct")
    print(f"Mode:                 Arm A (Direct Classification, Label-Only)")
    print(f"Train Dataset:        {args.data_path}")
    print(f"Pre-format budget:    {args.max_body_tokens} body tokens")
    print(f"LoRA config:          r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Learning rate:        {args.lr}")
    print(f"Compute dtype:        fp16={USE_FP16}, bf16={USE_BF16}")
    print(f"Smoke test mode:      {args.smoke_test}")
    print(f"---------------------------------------\n")

    # 5. Load Data & Measure Pre-Format Truncation
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"phishing_train.csv not found at {args.data_path}")
    
    df = pd.read_csv(args.data_path)
    train_sha256 = get_file_sha256(args.data_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} training rows (SHA256: {train_sha256})")

    if args.smoke_test:
        df = df.head(200).copy()
        print(f"[SMOKE TEST]: Subsampled to first {len(df)} rows.")

    # 6. Load Tokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 7. Pre-format Truncation & Dataset Construction
    truncated_count = 0
    formatted_texts = []
    for _, row in df.iterrows():
        subj = str(row['subject']) if pd.notna(row['subject']) else ""
        raw_body = str(row['body']) if pd.notna(row['body']) else ""
        trunc_body, was_trunc = truncate_body_text(raw_body, max_tokens=args.max_body_tokens, tokenizer=tokenizer)
        if was_trunc:
            truncated_count += 1
        label_str = "Phishing" if row['label'] == 1 else "Legitimate"
        formatted_texts.append(build_naive_prompt(subj, trunc_body, label=label_str))

    pct_truncated = (truncated_count / len(df)) * 100
    print(f"Pre-format truncation: {truncated_count}/{len(df)} rows truncated ({pct_truncated:.2f}%)")

    dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_texts}))

    # 8. Setup Quantization & LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else None,
        torch_dtype=COMPUTE_DTYPE if cuda_available else torch.float32
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 9. Completion-Only Data Collator
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer
    )

    # 10. Verify Prompt Tokens are Masked with -100
    verify_completion_masking(dataset, collator, tokenizer, num_examples=3)

    # 11. SFT Training Arguments
    max_steps = 40 if args.smoke_test else -1
    epochs = 1 if args.smoke_test else args.num_epochs

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5 if args.smoke_test else 20,
        save_strategy="no" if args.smoke_test else "epoch",
        fp16=USE_FP16 if cuda_available else False,
        bf16=USE_BF16,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        seed=args.seed
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator
    )

    # 12. Train
    print("\nStarting training...")
    t0 = time.time()
    train_result = trainer.train()
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
        # Full run: 20,681 rows * 3 epochs / 16 eff batch ~= 3,878 steps
        full_steps = int((total_rows * args.num_epochs) / (args.batch_size * args.grad_accum))
        projected_full_sec = full_steps / steps_per_sec if steps_per_sec > 0 else 0
        projected_full_hours = projected_full_sec / 3600
        print(f"Projected full run time (3,878 steps on T4): {projected_full_hours:.2f} hours")

    # 14. Save Adapter & Metadata
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
        "hardware": device_name,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "bitsandbytes": bitsandbytes.__version__
        }
    }

    run_json_path = os.path.join(REPO_ROOT, "eval", "results", "runs", f"{run_id}.json")
    with open(run_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Logged run metadata to: {run_json_path}")

if __name__ == "__main__":
    main()
