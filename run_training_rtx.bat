@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   PHISHING SLM RESEARCH — ARM A TRAINING (RTX GPU RUNNER)
echo ============================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 and check "Add to PATH".
    pause
    exit /b 1
)

echo [1/3] Checking environment & installing dependencies...
python -c "import torch; assert torch.cuda.is_available()" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyTorch with CUDA not found. Installing PyTorch with CUDA 12.4...
    pip install torch --index-url https://download.pytorch.org/whl/cu124
)

echo [INFO] Ensuring required libraries are installed...
pip install -r requirements_rtx.txt

echo.
echo [2/3] Checking NVIDIA GPU...
python -c "import torch; print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU DETECTED!')"
echo.

echo [3/3] Starting Training and Validation Evaluation...
echo This will train Arm A (Qwen2.5-1.5B Naive QLoRA) for 3 epochs and evaluate on validation data.
echo Checkpoints will save every 200 steps. If interrupted, simply re-run this script to resume!
echo.

python scripts/run_arm_a_local.py --epochs 3

echo.
echo ============================================================
echo   TRAINING & EVALUATION FINISHED!
echo ============================================================
echo.
echo Please ZIP the 'arm_a_results_package' folder created in this directory
echo and send it back to Satvik.
echo.
pause
