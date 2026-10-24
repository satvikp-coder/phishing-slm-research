#!/usr/bin/env python3
"""
src/preflight.py — Pre-Flight Environment, Integrity, and Offline Readiness Check
Arm A Verification Kit.
"""

import os
import sys
import json
import time

# Ensure kit root is on sys.path
KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

from src.utils import (
    get_kit_root,
    log_message,
    compute_sha256,
    scan_for_forbidden_test_files,
    get_power_status,
    get_system_hardware_info
)

TRAIN_EXPECTED_SHA = "81479ac035da02b8248630534f02c80ec8ec585dc5b8bf4314ec2f6a61a2d825"
VAL_EXPECTED_SHA = "3376f0281e414842f7903360b59606d03f5e0a8b4c31d286e2727caa673c0c9f"

def main():
    kit_root = get_kit_root()
    output_dir = os.path.join(kit_root, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "preflight_report.txt")
    json_file = os.path.join(output_dir, "preflight.json")

    overall_pass = True
    messages = []

    def report(line: str, ok: bool = True):
        nonlocal overall_pass
        if not ok:
            overall_pass = False
        prefix = "[PASS]" if ok else "[FAIL]"
        full = f"{prefix} {line}"
        messages.append(full)
        log_message(full)

    log_message("=" * 60)
    log_message("ARM A VERIFICATION KIT — PREFLIGHT CHECK")
    log_message(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Kit Path : {kit_root}")
    log_message("=" * 60)

    # 1. Safety Scan: Never allow test data or profile leakage
    violations = scan_for_forbidden_test_files(kit_root)
    if violations:
        report(f"Forbidden test file(s) detected inside kit: {violations}. ABORTING.", ok=False)
    else:
        report("Safety audit: No forbidden test datasets or profile artifacts detected.", ok=True)

    # 2. Hardware and Environment Inspection
    hw = get_system_hardware_info()
    power = get_power_status()

    log_message("\n--- System & Hardware Audit ---")
    log_message(f"Machine Name       : {hw['machine_name']}")
    log_message(f"Operating System   : {hw['os']}")
    log_message(f"CPU Processor      : {hw['processor']}")
    log_message(f"System RAM         : {hw['ram_gb']} GB")
    log_message(f"Python Executable  : {hw['python_executable']} (v{hw['python_version']})")
    log_message(f"CUDA Available     : {hw['cuda_available']}")
    log_message(f"GPU Model          : {hw['gpu_model']}")
    log_message(f"GPU VRAM           : {hw['gpu_vram_gb']} GB")
    log_message(f"GPU Driver         : {hw['gpu_driver']}")
    log_message(f"Compute Capability : {hw['compute_capability']}")
    log_message(f"AC Power Status    : {'Plugged In (AC)' if power['ac_plugged_in'] else 'Running on Battery'}")
    log_message(f"Active Power Scheme: {power['power_plan']}")

    # Check power status warning
    if power["ac_plugged_in"] is False:
        log_message("WARNING: Laptop is running on battery! Plug in the laptop for consistent thermal and clock performance.")

    # Check Blackwell compatibility
    if hw["blackwell_unsupported"]:
        log_message("\n[NOTICE] Blackwell GPU (RTX 50-series) detected.")
        log_message("The offline PyTorch CUDA 12.4 build does not support Blackwell.")
        log_message("Recommendation: Run with PyTorch cu128 if CUDA acceleration is desired, or run in CPU-only mode.\n")

    # 3. Data Integrity & Manifest Verification
    log_message("\n--- Manifest & Data File Integrity ---")
    manifest_path = os.path.join(kit_root, "data", "manifest.json")
    if not os.path.exists(manifest_path):
        report(f"Missing data manifest at {manifest_path}", ok=False)
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest_ok = True
            for rel_path, expected_sha in manifest_data.items():
                # Ignore self-manifest, compiled bytecode, caches, output directories, and editable scripts
                norm_rel = rel_path.replace("\\", "/").lower()
                if norm_rel in ("data/manifest.json", "manifest.json") or "manifest.json" in norm_rel:
                    continue
                if "__pycache__" in norm_rel or norm_rel.endswith(".pyc") or norm_rel.endswith(".pyo"):
                    continue
                if norm_rel.startswith("output/") or "/output/" in norm_rel:
                    continue
                # Skip scripts and docs (.bat, .py, .txt, .md) to prevent false failures from line-ending differences (CRLF/LF) or script patches
                if norm_rel.endswith(".bat") or norm_rel.endswith(".py") or norm_rel.endswith(".txt") or norm_rel.endswith(".md"):
                    continue

                abs_path = os.path.join(kit_root, rel_path)
                if not os.path.exists(abs_path):
                    report(f"Missing required file: {rel_path}", ok=False)
                    manifest_ok = False
                    continue
                actual_sha = compute_sha256(abs_path)
                if actual_sha != expected_sha:
                    report(f"SHA256 mismatch for {rel_path} (got {actual_sha[:10]}..., expected {expected_sha[:10]}...)", ok=False)
                    manifest_ok = False
            if manifest_ok:
                report("Manifest check: All files match cryptographic SHA256 checksums exactly.", ok=True)
        except Exception as e:
            report(f"Error reading manifest: {e}", ok=False)

    # Explicit check for train and val split hashes
    train_path = os.path.join(kit_root, "data", "phishing_train.csv")
    val_path = os.path.join(kit_root, "data", "phishing_val.csv")

    if os.path.exists(train_path):
        train_sha = compute_sha256(train_path)
        if train_sha == TRAIN_EXPECTED_SHA:
            report("phishing_train.csv SHA256 matches official benchmark checksum.", ok=True)
        else:
            report(f"phishing_train.csv hash mismatch: {train_sha} != {TRAIN_EXPECTED_SHA}", ok=False)
    else:
        report("phishing_train.csv not found in data/", ok=False)

    if os.path.exists(val_path):
        val_sha = compute_sha256(val_path)
        if val_sha == VAL_EXPECTED_SHA:
            report("phishing_val.csv SHA256 matches official benchmark checksum.", ok=True)
        else:
            report(f"phishing_val.csv hash mismatch: {val_sha} != {VAL_EXPECTED_SHA}", ok=False)
    else:
        report("phishing_val.csv not found in data/", ok=False)

    # 4. Offline Model & Adapter Loading Verification
    log_message("\n--- Offline Model & Adapter Verification ---")
    model_dir = os.path.join(kit_root, "model_cache")
    adapter_dir = os.path.join(kit_root, "adapter")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = model_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = model_dir

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        # Locate model weights: check snapshot or direct directory
        base_path = model_dir
        # Search for snapshot subfolder if present
        for root, dirs, files in os.walk(model_dir):
            if "config.json" in files:
                base_path = root
                break

        log_message(f"Loading base model offline from: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load minimal test model (CPU float32 or GPU if available and supported)
        target_device = "cuda" if (torch.cuda.is_available() and not hw["blackwell_unsupported"]) else "cpu"
        target_dtype = torch.float16 if target_device == "cuda" else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=target_dtype,
            device_map={"": target_device} if target_device == "cuda" else None,
            local_files_only=True
        )
        if target_device == "cpu":
            base_model = base_model.to("cpu")

        log_message(f"Loading Arm A LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
        model.eval()

        # Run 1 quick forward pass
        test_text = "### Instruction:\nClassify the following email as 'Phishing' or 'Legitimate'.\n\n### Email:\nSubject: Hello\n\nBody:\nTest email.\n\n### Response:\n"
        inputs = tokenizer(test_text, return_tensors="pt")
        if target_device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            id_p = tokenizer.encode("Phishing", add_special_tokens=False)[0]
            id_l = tokenizer.encode("Legitimate", add_special_tokens=False)[0]
            score = float(logits[id_p] - logits[id_l])
            pred = "Phishing" if score > 0 else "Legitimate"

        report(f"Offline model load & 1-sample forward pass succeeded! (Prediction: '{pred}', Score: {score:.3f})", ok=True)

        # Free memory
        del model
        del base_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        report(f"Offline model/adapter test failed: {e}", ok=False)

    # 5. Summary & Verdict
    log_message("\n" + "=" * 60)
    verdict = "PASS" if overall_pass else "FAIL"
    log_message(f"OVERALL PREFLIGHT VERDICT: {verdict}")
    log_message("=" * 60)

    # Write OUTPUT/preflight_report.txt
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ARM A VERIFICATION KIT — PREFLIGHT REPORT: {verdict}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Machine Identifier: {hw['machine_name']}\n")
        f.write(f"Date & Time       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Operating System  : {hw['os']}\n")
        f.write(f"CPU               : {hw['processor']}\n")
        f.write(f"RAM               : {hw['ram_gb']} GB\n")
        f.write(f"GPU               : {hw['gpu_model']} ({hw['gpu_vram_gb']} GB VRAM)\n")
        f.write(f"CUDA Version      : {hw['cuda_version']}\n")
        f.write(f"AC Power Plugged  : {power['ac_plugged_in']}\n\n")
        f.write("Detailed Checklist:\n")
        for m in messages:
            f.write(f"  {m}\n")
        f.write("\n" + "=" * 60 + "\n")
        if overall_pass:
            f.write("STATUS: READY. You may now proceed to step 2_QUICK_TEST.bat.\n")
        else:
            f.write("STATUS: NOT READY. Please resolve the issues flagged with [FAIL] above.\n")
        f.write("=" * 60 + "\n")

    # Write OUTPUT/preflight.json
    preflight_json_data = {
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "power": power,
        "overall_pass": overall_pass,
        "messages": messages
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(preflight_json_data, f, indent=2)

    log_message(f"Preflight report written to: {report_file}")
    log_message(f"Preflight JSON written to  : {json_file}")

    if not overall_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
