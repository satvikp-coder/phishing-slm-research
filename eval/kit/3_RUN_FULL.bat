@echo off
setlocal enabledelayedexpansion

:: =====================================================================
:: 3_RUN_FULL.bat — Full Validation & Latency Pipeline
:: Arm A Verification Kit
:: All paths relative to this script's directory (%~dp0)
:: =====================================================================

cd /d "%~dp0"
title Arm A Kit - Step 3: Full Run

echo =====================================================================
echo  ARM A VERIFICATION KIT -- STEP 3: FULL COMPREHENSIVE RUN
echo =====================================================================
echo.

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment not found.
    echo Please run 1_CHECK_FIRST.bat first!
    echo.
    pause
    exit /b 1
)

:: Safety Check: Refuse to start unless quick test passed
if not exist "%~dp0OUTPUT\.quick_test_passed" (
    echo [ERROR] Quick test has NOT been completed successfully on this machine!
    echo.
    echo Safety policy requires running 2_QUICK_TEST.bat first to verify that
    echo dependencies, memory, and model loading work properly before launching
    echo the full multi-minute run.
    echo.
    echo Please double-click 2_QUICK_TEST.bat first.
    echo.
    pause
    exit /b 1
)

echo Quick test verification marker verified. Starting full pipeline...
echo.

echo [Stage 1/7] Scoring all 2,954 validation emails with Arm A...
"%VENV_PY%" "%~dp0src\step1_score.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 1 scoring failed.
    pause
    exit /b 1
)

echo.
echo [Stage 2/7] Computing Classification Metrics and AUROC across full validation set...
"%VENV_PY%" "%~dp0src\step2_metrics.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 2 metrics failed.
    pause
    exit /b 1
)

echo.
echo [Stage 3/7] Computing Per-Source Subpopulation Breakdown...
"%VENV_PY%" "%~dp0src\step3_persource.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 3 per-source analysis failed.
    pause
    exit /b 1
)

echo.
echo [Stage 4/7] Running Train-to-Validation Near-Duplicate Leakage Analysis...
"%VENV_PY%" "%~dp0src\step4_neardup.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 4 near-duplicate analysis failed.
    pause
    exit /b 1
)

echo.
echo [Stage 5/7] Running Exact McNemar Tests and 1,000 Bootstrap Resamples...
"%VENV_PY%" "%~dp0src\step5_paired.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 5 paired tests failed.
    pause
    exit /b 1
)

echo.
echo [Stage 6/7] Running Latency and Throughput Benchmark (500 Stratified Emails, 3 Repeats)...
"%VENV_PY%" "%~dp0src\step6_latency.py" --repeats 3 --warmup 20
if %errorlevel% neq 0 (
    echo [ERROR] Step 6 latency benchmark failed.
    pause
    exit /b 1
)

echo.
echo [Stage 7/7] Generating Final Comprehensive Report (OUTPUT\ARM_A_CORRECTED_RESULTS.md)...
"%VENV_PY%" "%~dp0src\make_report.py"
if %errorlevel% neq 0 (
    echo [ERROR] Report generation failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo  SUCCESS: FULL RUN COMPLETE!
echo =====================================================================
echo  All metrics, tables, and reports have been generated in OUTPUT\.
echo  Now proceed to Step 4: Double-click 4_PACK_RESULTS.bat to create the
echo  transfer ZIP archive for bringing back the results.
echo =====================================================================
echo.
pause
exit /b 0
