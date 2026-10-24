# src/utils.py — Helper Utilities for Arm A Verification Kit
import os
import sys
import json
import time
import ctypes
import hashlib
import platform
import subprocess
from typing import Dict, Any, List, Tuple, Optional

# Files allowed to contain the word "test" (case-insensitive)
ALLOWED_TEST_PATTERNS = {
    "2_quick_test.bat",
    "quick_test",
    "binomtest",
    "paired_tests",
    "test_helpers",
    ".quick_test_passed"
}

def get_kit_root() -> str:
    """Returns absolute path to the root of the kit (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def log_message(msg: str, log_file: Optional[str] = None):
    """Prints message to stdout and appends to OUTPUT/run_log.txt."""
    print(msg)
    if log_file is None:
        kit_root = get_kit_root()
        out_dir = os.path.join(kit_root, "OUTPUT")
        os.makedirs(out_dir, exist_ok=True)
        log_file = os.path.join(out_dir, "run_log.txt")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_for_forbidden_test_files(kit_root: str) -> List[str]:
    """
    Scans the entire kit directory for any file containing 'test'
    other than authorized test runner batch files and helpers.
    """
    violations = []
    for root, _, files in os.walk(kit_root):
        # Ignore .venv directory to avoid false positives in site-packages
        if ".venv" in root or "venv" in root:
            continue
        for fname in files:
            lower = fname.lower()
            if "test" in lower:
                allowed = any(pattern in lower for pattern in ALLOWED_TEST_PATTERNS)
                if not allowed:
                    violations.append(os.path.relpath(os.path.join(root, fname), kit_root))
    return violations

def get_power_status() -> Dict[str, Any]:
    """
    Queries Windows power status (Plugged In vs Battery) and active power scheme.
    """
    res = {
        "ac_plugged_in": None,
        "battery_percent": None,
        "power_plan": "unknown"
    }

    # Query AC power using Windows SYSTEM_POWER_STATUS
    try:
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("Reserved1", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]
        sps = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
            res["ac_plugged_in"] = (sps.ACLineStatus == 1)
            res["battery_percent"] = sps.BatteryLifePercent if sps.BatteryLifePercent <= 100 else None
    except Exception:
        pass

    # Query Power Plan via powercfg
    try:
        out = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, stderr=subprocess.DEVNULL)
        res["power_plan"] = out.strip()
    except Exception:
        pass

    return res

def get_system_hardware_info() -> Dict[str, Any]:
    """
    Gathers detailed hardware & environment metadata.
    """
    info = {
        "machine_name": platform.node(),
        "os": f"{platform.system()} {platform.release()} (Version: {platform.version()})",
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "ram_gb": None,
        "cuda_available": False,
        "gpu_model": "None (CPU Execution)",
        "gpu_vram_gb": 0.0,
        "gpu_driver": "None",
        "cuda_version": "None",
        "compute_capability": None,
        "blackwell_unsupported": False,
        "packages": {}
    }

    # RAM
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            info["ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
    except Exception:
        pass

    # Installed package versions
    for pkg in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "scikit-learn", "scipy", "pandas", "numpy"]:
        try:
            m = __import__(pkg.replace("-", "_"))
            info["packages"][pkg] = getattr(m, "__version__", "installed")
        except ImportError:
            info["packages"][pkg] = "not installed"

    # PyTorch and GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["compute_capability"] = f"sm_{props.major}{props.minor}"
            info["cuda_version"] = torch.version.cuda
            # Check if Blackwell (sm_100, sm_120) with cu124
            if props.major >= 10:
                # If torch CUDA version is < 12.8, Blackwell requires newer torch
                info["blackwell_unsupported"] = True
    except Exception:
        pass

    # GPU Driver & Temp via nvidia-smi
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,clocks.current.graphics", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = [p.strip() for p in smi_out.strip().split(",")]
        if len(parts) >= 1:
            info["gpu_driver"] = parts[0]
        if len(parts) >= 3:
            info["gpu_temp_c"] = float(parts[1])
            info["gpu_clock_mhz"] = float(parts[2])
    except Exception:
        pass

    return info
