#!/usr/bin/env python3
"""
eval/kit/build_kit.py — Automated Assembly Script for Arm A Verification Kit
Builds self-contained standalone kit at kits/armA_verification_kit/
Copies data, adapter, model cache, prerequisites, wheels, and scripts.
Generates manifest.json with SHA256 hashes for all kit files.
Runs safety audit to strictly ensure NO test datasets or user profile files are included.
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import subprocess
from typing import Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIT_SRC_DIR = os.path.join(REPO_ROOT, "eval", "kit")
TARGET_KIT_DIR = os.path.join(REPO_ROOT, "kits", "armA_verification_kit")

sys.path.insert(0, KIT_SRC_DIR)
from src.utils import compute_sha256, scan_for_forbidden_test_files, log_message

PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

def copy_file_checked(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied: {os.path.relpath(src, REPO_ROOT)} -> {os.path.relpath(dst, REPO_ROOT)}")

def get_dir_size_gb(path: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024**3)

def main():
    t_start = time.time()
    print("=" * 70)
    print("ARM A VERIFICATION KIT — AUTOMATED BUILD & PACKAGING")
    print("=" * 70)
    print(f"Source Kit Scripts : {KIT_SRC_DIR}")
    print(f"Target Kit Path    : {TARGET_KIT_DIR}")
    print("=" * 70)

    # 1. Clean or create target directory
    os.makedirs(TARGET_KIT_DIR, exist_ok=True)
    out_dir = os.path.join(TARGET_KIT_DIR, "OUTPUT")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Copy Batch Files and Guides
    print("\n[Stage 1/7] Copying batch runners, guides, and requirements...")
    top_level_files = [
        "START_HERE.txt",
        "1_CHECK_FIRST.bat",
        "2_QUICK_TEST.bat",
        "3_RUN_FULL.bat",
        "4_PACK_RESULTS.bat",
        "requirements-kit.txt"
    ]
    for fname in top_level_files:
        src = os.path.join(KIT_SRC_DIR, fname)
        dst = os.path.join(TARGET_KIT_DIR, fname)
        copy_file_checked(src, dst)

    # Copy src/ directory
    src_target = os.path.join(TARGET_KIT_DIR, "src")
    if os.path.exists(src_target):
        shutil.rmtree(src_target)
    shutil.copytree(os.path.join(KIT_SRC_DIR, "src"), src_target)
    print(f"  Copied src/ pipeline scripts to {os.path.relpath(src_target, REPO_ROOT)}")

    # 3. Assemble data/ directory (Validation & Train splits, baseline predictions, sample IDs)
    print("\n[Stage 2/7] Assembling benchmark data files...")
    data_target = os.path.join(TARGET_KIT_DIR, "data")
    os.makedirs(data_target, exist_ok=True)

    # Note: phishing_test.csv is STRICTLY EXCLUDED!
    data_files = [
        (os.path.join(REPO_ROOT, "data", "processed", "phishing_train.csv"), os.path.join(data_target, "phishing_train.csv")),
        (os.path.join(REPO_ROOT, "data", "processed", "phishing_val.csv"), os.path.join(data_target, "phishing_val.csv")),
        (os.path.join(REPO_ROOT, "eval", "results", "latency_sample_ids.csv"), os.path.join(data_target, "latency_sample_ids.csv")),
        (os.path.join(REPO_ROOT, "eval", "results", "preds_qwen2_5_1_5b_naive_val.csv"), os.path.join(data_target, "preds_qwen2_5_1_5b_naive_val.csv")),
        (os.path.join(REPO_ROOT, "eval", "results", "preds_tf-idf_plus_logisticregression_val.csv"), os.path.join(data_target, "preds_tf-idf_plus_logisticregression_val.csv")),
        (os.path.join(REPO_ROOT, "eval", "results", "preds_tf-idf_plus_linearsvm_val.csv"), os.path.join(data_target, "preds_tf-idf_plus_linearsvm_val.csv")),
    ]
    for src, dst in data_files:
        copy_file_checked(src, dst)

    # 4. Assemble adapter/ directory (Arm A LoRA Checkpoint)
    print("\n[Stage 3/7] Copying Arm A LoRA adapter checkpoint...")
    adapter_src = os.path.join(REPO_ROOT, "finetuning", "checkpoints", "Qwen2.5-1.5B-Instruct-naive")
    adapter_target = os.path.join(TARGET_KIT_DIR, "adapter")
    os.makedirs(adapter_target, exist_ok=True)

    for item in os.listdir(adapter_src):
        sp = os.path.join(adapter_src, item)
        dp = os.path.join(adapter_target, item)
        if os.path.isfile(sp):
            shutil.copy2(sp, dp)
    print(f"  Copied adapter files to {os.path.relpath(adapter_target, REPO_ROOT)}")

    # Write note explaining checkpoint
    with open(os.path.join(adapter_target, "NOTE_ARM_A_CHECKPOINT.txt"), "w") as f:
        f.write("Arm A Final Adapter Checkpoint\n")
        f.write("Trained for 3 epochs on 20,681 emails from phishing_train.csv.\n")
        f.write("LoRA hyperparameters: r=16, alpha=32, lr=2e-4, max_length=1024.\n")

    # 5. Assemble model_cache/ (Base Qwen2.5-1.5B-Instruct weights)
    print("\n[Stage 4/7] Copying offline Qwen2.5-1.5B-Instruct model cache...")
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    model_cache_target = os.path.join(TARGET_KIT_DIR, "model_cache")
    os.makedirs(model_cache_target, exist_ok=True)

    # Locate snapshot
    snapshot_dir = None
    snapshots_path = os.path.join(hf_cache_dir, "snapshots")
    if os.path.exists(snapshots_path):
        snaps = os.listdir(snapshots_path)
        if snaps:
            snapshot_dir = os.path.join(snapshots_path, snaps[0])

    if snapshot_dir and os.path.exists(snapshot_dir):
        print(f"  Found snapshot at: {snapshot_dir}")
        for f in os.listdir(snapshot_dir):
            sp = os.path.join(snapshot_dir, f)
            dp = os.path.join(model_cache_target, f)
            if os.path.isfile(sp) and not os.path.exists(dp):
                copy_file_checked(sp, dp)

    # 6. Assemble prereq/ (Python 3.11 installer)
    print("\n[Stage 5/7] Checking official Python 3.11 installer in prereq/...")
    prereq_target = os.path.join(TARGET_KIT_DIR, "prereq")
    os.makedirs(prereq_target, exist_ok=True)
    installer_path = os.path.join(prereq_target, "python-3.11.9-amd64.exe")

    if not os.path.exists(installer_path) or os.path.getsize(installer_path) < 1000000:
        print(f"  Downloading from {PYTHON_INSTALLER_URL}...")
        try:
            urllib.request.urlretrieve(PYTHON_INSTALLER_URL, installer_path)
            print(f"  Downloaded: {installer_path} ({os.path.getsize(installer_path)/(1024**2):.1f} MB)")
        except Exception as e:
            print(f"  Notice: Could not download installer directly ({e}).")
    else:
        print(f"  Python installer already present: {installer_path}")

    # 7. Check wheels/
    print("\n[Stage 6/7] Checking offline wheelhouse in wheels/...")
    wheels_target = os.path.join(TARGET_KIT_DIR, "wheels")
    os.makedirs(wheels_target, exist_ok=True)
    wheel_files = [f for f in os.listdir(wheels_target) if f.endswith(".whl")]
    print(f"  Wheelhouse contains {len(wheel_files)} wheels.")

    # 8. Generate data/manifest.json (Cryptographic SHA256 for all kit files)
    print("\n[Stage 7/7] Generating data/manifest.json with SHA256 checksums...")
    manifest = {}
    for root, dirs, files in os.walk(TARGET_KIT_DIR):
        # Exclude OUTPUT, temporary files, virtualenv, and pycache
        if "OUTPUT" in root or ".venv" in root or "__pycache__" in root:
            continue
        for f in files:
            if f == "manifest.json" or f.endswith(".pyc") or f.endswith(".pyo"):
                continue
            fp = os.path.join(root, f)
            rel_p = os.path.relpath(fp, TARGET_KIT_DIR)
            manifest[rel_p] = compute_sha256(fp)

    manifest_json_path = os.path.join(data_target, "manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Generated manifest.json containing {len(manifest)} verified files.")

    # 9. Safety Audit: Scan for forbidden files
    print("\n" + "=" * 70)
    print("MANDATORY SAFETY AUDIT: Scanning kit for forbidden files...")
    violations = scan_for_forbidden_test_files(TARGET_KIT_DIR)
    if violations:
        print(f"CRITICAL SAFETY VIOLATION! Forbidden files detected in kit: {violations}")
        print("ABORTING AND REMOVING KIT.")
        shutil.rmtree(TARGET_KIT_DIR)
        sys.exit(1)
    else:
        print("SAFETY AUDIT PASSED: ZERO forbidden test datasets, keys, or profile files found.")
    print("=" * 70)

    # 10. Summary
    kit_size_gb = get_dir_size_gb(TARGET_KIT_DIR)
    elapsed = time.time() - t_start
    print(f"\nArm A Verification Kit built successfully!")
    print(f"Location    : {TARGET_KIT_DIR}")
    print(f"Total Size  : {kit_size_gb:.2f} GB")
    print(f"Time Taken  : {elapsed:.1f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()
