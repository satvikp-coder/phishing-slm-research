#!/usr/bin/env python3
"""
src/step6_latency.py — Deployment Latency & Throughput Benchmark for Arm A
Supports Prof. Amit Das's two-tier architecture:
- Real-time online filtering (single forward pass scoring, A2) vs generation (A1)
- Batched background offline clearance (batch sizes 1, 8, 16, 32, length-sorted)
- Full precision per-email timings, repeats, p50, IQR, p95, p99, peak VRAM, and backlog arithmetic.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import pandas as pd

KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import get_kit_root, log_message, get_power_status, get_system_hardware_info
from src.prompt_template import (
    MAX_BODY_TOKENS,
    MAX_SUBJECT_TOKENS,
    MAX_SEQ_LENGTH,
    truncate_body_text,
    truncate_subject_text,
    build_naive_prompt
)

def query_gpu_telemetry():
    """Queries GPU temperature, graphics clock, and memory usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = [float(p.strip()) for p in out.strip().split(",")]
        return {
            "temp_c": parts[0] if len(parts) > 0 else None,
            "clock_mhz": parts[1] if len(parts) > 1 else None,
            "mem_used_mb": parts[2] if len(parts) > 2 else None,
            "mem_total_mb": parts[3] if len(parts) > 3 else None,
        }
    except Exception:
        return {"temp_c": None, "clock_mhz": None, "mem_used_mb": None, "mem_total_mb": None}

def locate_model_path(model_dir: str) -> str:
    for root, dirs, files in os.walk(model_dir):
        if "config.json" in files:
            return root
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Step 6: Latency & Throughput Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of emails for benchmark (e.g. 50 in quick test)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repeats")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warm-up emails discarded")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    raw_out_path = os.path.join(output_dir, "latency_raw.csv")
    summary_out_path = os.path.join(output_dir, "latency_summary.csv")
    meta_out_path = os.path.join(output_dir, "latency_metadata.json")

    log_message("=" * 60)
    log_message("STEP 6: LATENCY & THROUGHPUT BENCHMARK (ARM A)")
    log_message(f"Repeats: {args.repeats} | Warmup: {args.warmup} | Limit: {args.limit}")
    log_message("=" * 60)

    # 1. Load benchmark sample
    sample_csv = os.path.join(kit_root, "data", "latency_sample_ids.csv")
    val_csv = os.path.join(kit_root, "data", "phishing_val.csv")

    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing validation data: {val_csv}")

    val_df = pd.read_csv(val_csv)

    if os.path.exists(sample_csv):
        sample_df = pd.read_csv(sample_csv)
        # Merge to get subject and body
        bench_df = pd.merge(sample_df[["email_id", "bucket"]], val_df, on="email_id")
        log_message(f"Loaded stratified sample from {sample_csv} ({len(bench_df)} emails across 5 buckets)")
    else:
        log_message(f"latency_sample_ids.csv not found; falling back to first rows of {val_csv}")
        bench_df = val_df.head(500).copy()
        bench_df["bucket"] = "all"

    if args.limit is not None and args.limit > 0:
        bench_df = bench_df.head(args.limit).copy()
        log_message(f"Limiting benchmark sample to first {len(bench_df)} emails.")

    # 2. Hardware and Environment Telemetry
    hw = get_system_hardware_info()
    power_start = get_power_status()
    gpu_start = query_gpu_telemetry()

    log_message(f"\nPower Status at Start: {'Plugged In (AC)' if power_start['ac_plugged_in'] else 'Battery'}")
    if gpu_start["temp_c"] is not None:
        log_message(f"GPU Telemetry at Start: Temp={gpu_start['temp_c']} C | Clock={gpu_start['clock_mhz']} MHz | VRAM Used={gpu_start['mem_used_mb']} MB")

    # 3. Model Loading
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    cuda_available = torch.cuda.is_available() and not hw["blackwell_unsupported"]

    model_cache = os.path.join(kit_root, "model_cache")
    adapter_path = os.path.join(kit_root, "adapter")
    base_model_path = locate_model_path(model_cache)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = model_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = model_cache

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For batched inference with causal LM, set padding side to left
    tokenizer.padding_side = "left"

    tok_p = tokenizer.encode("Phishing", add_special_tokens=False)[0]
    tok_l = tokenizer.encode("Legitimate", add_special_tokens=False)[0]

    model = None
    load_mode = "unknown"

    if cuda_available:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
            model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
            load_mode = "4bit_bnb_gpu"
        except Exception:
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
                model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
                load_mode = "fp16_gpu"
            except Exception:
                pass

    if model is None:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=None,
            torch_dtype=torch.float32,
            local_files_only=True
        ).to("cpu")
        model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
        load_mode = "cpu_float32"

    model.eval()
    device = next(model.parameters()).device
    log_message(f"Model loaded successfully on {device} (load_mode: {load_mode})")

    def sync():
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()

    # Pre-build prompts for timing stages
    prepared_items = []
    for _, row in bench_df.iterrows():
        s_trunc, _ = truncate_subject_text(row["subject"], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
        b_trunc, _ = truncate_body_text(row["body"], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
        prompt = build_naive_prompt(s_trunc, b_trunc)
        prepared_items.append({
            "email_id": row["email_id"],
            "bucket": row.get("bucket", "all"),
            "subject_raw": str(row["subject"]) if pd.notna(row["subject"]) else "",
            "body_raw": str(row["body"]) if pd.notna(row["body"]) else "",
            "prompt": prompt
        })

    # CPU protocol check: "If there is no GPU, run A2 on 20 emails on CPU and label it."
    is_cpu_mode = (device.type == "cpu")
    if is_cpu_mode:
        log_message("\n[PROTOCOL NOTE] No GPU active. Running Condition A2 on 20 emails on CPU per protocol specification.")
        prepared_items = prepared_items[:20]
        args.warmup = min(args.warmup, 2)
        args.repeats = 1

    # Warmup
    warmup_n = min(args.warmup, len(prepared_items))
    if warmup_n > 0:
        log_message(f"\nRunning {warmup_n} warm-up passes (discarded from metrics)...")
        for i in range(warmup_n):
            item = prepared_items[i]
            inputs = tokenizer(item["prompt"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                sync()
                _ = model(**inputs)
                sync()
        log_message("Warmup complete.")

    raw_records = []
    summary_records = []

    # 4. Condition A2: Single Forward Pass Logit Scoring (Batch Size 1)
    cond_name = "A2_single_forward_cpu_b1" if is_cpu_mode else "A2_single_forward_b1"
    log_message(f"\n--- Measuring Condition A2: Real-time Single Forward Pass ({'CPU ' if is_cpu_mode else ''}Batch 1) ---")
    a2_preds = {}

    for r in range(args.repeats):
        repeat_times = []
        for item in prepared_items:
            # End-to-end timing
            sync()
            t0 = time.perf_counter()

            # 1. Truncation & prompt build
            s_tr, _ = truncate_subject_text(item["subject_raw"], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
            b_tr, _ = truncate_body_text(item["body_raw"], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
            p = build_naive_prompt(s_tr, b_tr)

            # 2. Tokenization
            inputs = tokenizer(p, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 3. Model forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                score = float(logits[tok_p] - logits[tok_l])
                pred = 1 if score > 0.0 else 0

            sync()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if r == 0:
                a2_preds[item["email_id"]] = pred

            repeat_times.append(dt_ms)
            raw_records.append({
                "condition": cond_name,
                "repeat": r + 1,
                "email_id": item["email_id"],
                "bucket": item["bucket"],
                "latency_ms": dt_ms
            })

        log_message(f"  Repeat {r + 1}/{args.repeats}: p50={np.median(repeat_times):.2f} ms | p95={np.percentile(repeat_times, 95):.2f} ms")

    # 5. Condition A1: Autoregressive Generation (Batch Size 1) (GPU only)
    a1_preds = {}
    a1_a2_pct = None
    disagreements_a1_a2 = []

    if not is_cpu_mode:
        log_message("\n--- Measuring Condition A1: Autoregressive Generation until EOS (Batch 1) ---")
        for r in range(args.repeats):
            repeat_times = []
            for item in prepared_items:
                sync()
                t0 = time.perf_counter()

                s_tr, _ = truncate_subject_text(item["subject_raw"], max_tokens=MAX_SUBJECT_TOKENS, tokenizer=tokenizer)
                b_tr, _ = truncate_body_text(item["body_raw"], max_tokens=MAX_BODY_TOKENS, tokenizer=tokenizer)
                p = build_naive_prompt(s_tr, b_tr)

                inputs = tokenizer(p, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=4,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
                    pred = 1 if "phishing" in decoded else (0 if "legitimate" in decoded else -1)

                sync()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if r == 0:
                    a1_preds[item["email_id"]] = pred

                repeat_times.append(dt_ms)
                raw_records.append({
                    "condition": "A1_generation_b1",
                    "repeat": r + 1,
                    "email_id": item["email_id"],
                    "bucket": item["bucket"],
                    "latency_ms": dt_ms
                })

            log_message(f"  Repeat {r + 1}/{args.repeats}: p50={np.median(repeat_times):.2f} ms | p95={np.percentile(repeat_times, 95):.2f} ms")

        # Agreement check between A1 and A2
        a1_a2_matches = sum(1 for eid in a2_preds if a2_preds[eid] == a1_preds.get(eid))
        a1_a2_total = len(a2_preds)
        a1_a2_pct = (a1_a2_matches / a1_a2_total * 100.0) if a1_a2_total > 0 else 0.0
        disagreements_a1_a2 = [eid for eid in a2_preds if a2_preds[eid] != a1_preds.get(eid)]

        log_message("\n--- A1 vs A2 Label Agreement ---")
        log_message(f"Agreement Rate: {a1_a2_pct:.2f}% ({a1_a2_matches}/{a1_a2_total})")
        log_message(f"Disagreements: {disagreements_a1_a2}")
    else:
        log_message("\n[INFO] Skipped Condition A1 (Autoregressive Generation) on CPU per protocol specification.")

    # 6. Condition D: Batched Offline Background Clearance (Batch Sizes 8, 16, 32) (GPU only)
    batch_sizes = [8, 16, 32]
    batched_results = {}

    if not is_cpu_mode:
        log_message("\n--- Measuring Condition D: Batched Background Clearance (Length-Sorted) ---")
        sorted_items = sorted(prepared_items, key=lambda x: len(tokenizer.encode(x["prompt"], add_special_tokens=False)))

        for bsz in batch_sizes:
            log_message(f"\nEvaluating Batch Size = {bsz} (Length-Sorted)...")
            batch_latencies = []
            emails_processed = 0
            oom_occurred = False
            peak_vram_gb = 0.0

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t_batch_start = time.perf_counter()

            try:
                for i in range(0, len(sorted_items), bsz):
                    chunk = sorted_items[i:i + bsz]
                    prompts = [item["prompt"] for item in chunk]

                    sync()
                    t_chunk0 = time.perf_counter()

                    enc = tokenizer(prompts, padding=True, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}

                    with torch.no_grad():
                        out = model(**enc)
                        logits_last = out.logits[:, -1, :]
                        scores = logits_last[:, tok_p] - logits_last[:, tok_l]
                        _ = (scores > 0.0).long()

                    sync()
                    dt_chunk = (time.perf_counter() - t_chunk0) * 1000.0
                    per_email_ms = dt_chunk / len(chunk)

                    for item in chunk:
                        batch_latencies.append(per_email_ms)
                        raw_records.append({
                            "condition": f"D_batched_b{bsz}",
                            "repeat": 1,
                            "email_id": item["email_id"],
                            "bucket": item["bucket"],
                            "latency_ms": per_email_ms
                        })
                    emails_processed += len(chunk)

                total_sec = time.perf_counter() - t_batch_start
                throughput_eps = emails_processed / total_sec
                throughput_eph = throughput_eps * 3600.0

                if torch.cuda.is_available() and device.type == "cuda":
                    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

                batched_results[bsz] = {
                    "throughput_emails_per_sec": round(throughput_eps, 2),
                    "throughput_emails_per_hour": round(throughput_eph, 1),
                    "peak_vram_gb": peak_vram_gb,
                    "p50_latency_ms": round(float(np.median(batch_latencies)), 2),
                    "p95_latency_ms": round(float(np.percentile(batch_latencies, 95)), 2),
                    "oom": False
                }

                log_message(f"  Batch {bsz} Results: {throughput_eps:.2f} emails/s ({throughput_eph:.0f} emails/hr) | p50={np.median(batch_latencies):.2f} ms/email | Peak VRAM={peak_vram_gb} GB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_message(f"  CUDA Out of Memory at batch size {bsz}!")
                    oom_occurred = True
                    batched_results[bsz] = {"oom": True, "error": "CUDA Out Of Memory"}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
    else:
        log_message("\n[INFO] Skipped Condition D (Batched Clearance) on CPU per protocol specification.")

    # 7. Two-Tier Deployment Backlog Clearance Calculations
    log_message("\n--- Two-Tier Deployment Arithmetic ---")
    if not is_cpu_mode and batched_results:
        best_bsz = max([b for b, res in batched_results.items() if not res.get("oom", False)], default=1)
        best_throughput_sec = batched_results[best_bsz]["throughput_emails_per_sec"]
    else:
        a2_times = [r["latency_ms"] for r in raw_records if "A2" in r["condition"]]
        best_throughput_sec = (1000.0 / np.median(a2_times)) if a2_times else 0.5
        best_bsz = 1

    backlog_clearance = {}
    for n_mail in [100, 1000, 10000]:
        sec = n_mail / best_throughput_sec
        backlog_clearance[f"{n_mail}_emails_day"] = {
            "seconds": round(sec, 2),
            "formatted": f"{sec:.2f} s" if sec < 60 else f"{sec/60:.2f} min"
        }
        log_message(f"  N = {n_mail:,} emails/day backlog clears in: {backlog_clearance[f'{n_mail}_emails_day']['formatted']}")

    # Telemetry at End
    power_end = get_power_status()
    gpu_end = query_gpu_telemetry()
    log_message(f"\nPower Status at End: {'Plugged In (AC)' if power_end['ac_plugged_in'] else 'Battery'}")
    if gpu_end["temp_c"] is not None:
        log_message(f"GPU Telemetry at End: Temp={gpu_end['temp_c']} C | Clock={gpu_end['clock_mhz']} MHz | VRAM Used={gpu_end['mem_used_mb']} MB")

    # 8. Compile Summaries (p50, IQR, p95, p99)
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(raw_out_path, index=False)
    log_message(f"\nSaved raw per-email timings to: {raw_out_path}")

    for cond, grp in raw_df.groupby("condition"):
        times = grp["latency_ms"].values
        q25, q75 = np.percentile(times, 25), np.percentile(times, 75)
        summary_records.append({
            "condition": cond,
            "n_records": len(times),
            "repeats": args.repeats if "b1" in cond else 1,
            "p50_ms": round(float(np.median(times)), 2),
            "iqr_ms": round(float(q75 - q25), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "throughput_eps": round(float(1000.0 / np.median(times)), 2),
            "peak_vram_gb": hw["gpu_vram_gb"] if cuda_available else 0.0,
            "hardware": hw["gpu_model"] if cuda_available else "CPU",
            "load_mode": load_mode
        })

    sum_df = pd.DataFrame(summary_records)
    sum_df.to_csv(summary_out_path, index=False)
    log_message(f"Saved latency summary to: {summary_out_path}")

    # Metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "load_mode": load_mode,
        "telemetry_start": {"power": power_start, "gpu": gpu_start},
        "telemetry_end": {"power": power_end, "gpu": gpu_end},
        "agreement_a1_vs_a2": {
            "agreement_pct": round(a1_a2_pct, 2),
            "evaluated": a1_a2_total,
            "disagreements": disagreements_a1_a2
        },
        "batched_results": batched_results,
        "two_tier_clearance": backlog_clearance,
        "assumptions": [
            "Figures are strictly hardware-specific to this machine.",
            "Calculations exclude network transfer latency and email-server spooling/queue I/O.",
            "Batching uses length-sorted buckets to minimize padding overhead."
        ]
    }

    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log_message(f"Saved latency metadata to: {meta_out_path}\n")

if __name__ == "__main__":
    main()
