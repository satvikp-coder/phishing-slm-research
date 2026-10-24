@echo off
setlocal enabledelayedexpansion

:: =====================================================================
:: 2_QUICK_TEST.bat — Fast End-to-End Pipeline Verification (50 emails)
:: Arm A Verification Kit
:: All paths relative to this script's directory (%~dp0)
:: =====================================================================

cd /d "%~dp0"
title Arm A Kit - Step 2: Quick Test (50 Emails)

echo =====================================================================
echo  ARM A VERIFICATION KIT -- STEP 2: QUICK TEST (50 EMAILS)
echo =====================================================================
echo  This step runs the entire verification pipeline on a small sample
echo  of 50 validation emails to verify everything works end-to-end.
echo  Estimated time: 2 to 8 minutes (depending on CPU/GPU).
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

:: Clean previous quick test marker
if exist "%~dp0OUTPUT\.quick_test_passed" del "%~dp0OUTPUT\.quick_test_passed"

echo [Stage 1/7] Running Pre-Flight Check...
"%VENV_PY%" "%~dp0src\preflight.py"
if %errorlevel% neq 0 (
    echo [ERROR] Pre-flight failed. Aborting quick test.
    pause
    exit /b 1
)

echo.
echo [Stage 2/7] Scoring 50 validation emails with Arm A (Single Forward Pass)...
"%VENV_PY%" "%~dp0src\step1_score.py" --limit 50
if %errorlevel% neq 0 (
    echo [ERROR] Step 1 scoring failed.
    pause
    exit /b 1
)

echo.
echo [Stage 3/7] Computing Classification Metrics and AUROC...
"%VENV_PY%" "%~dp0src\step2_metrics.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 2 metrics failed.
    pause
    exit /b 1
)

echo.
echo [Stage 4/7] Computing Per-Source Subpopulation Metrics...
"%VENV_PY%" "%~dp0src\step3_persource.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 3 per-source analysis failed.
    pause
    exit /b 1
)

echo.
echo [Stage 5/7] Running Near-Duplicate Leakage Analysis...
"%VENV_PY%" "%~dp0src\step4_neardup.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 4 near-duplicate analysis failed.
    pause
    exit /b 1
)

echo.
echo [Stage 6/7] Running Paired Significance Tests and Bootstrap CIs...
"%VENV_PY%" "%~dp0src\step5_paired.py"
if %errorlevel% neq 0 (
    echo [ERROR] Step 5 paired tests failed.
    pause
    exit /b 1
)

echo.
echo [Stage 7/7] Running Latency and Throughput Benchmark (50 emails)...
"%VENV_PY%" "%~dp0src\step6_latency.py" --limit 50 --repeats 1 --warmup 5
if %errorlevel% neq 0 (
    echo [ERROR] Step 6 latency benchmark failed.
    pause
    exit /b 1
)

echo.
echo Generating Verification Markdown Report...
"%VENV_PY%" "%~dp0src\make_report.py"
if %errorlevel% neq 0 (
    echo [ERROR] Report generation failed.
    pause
    exit /b 1
)

:: Create pass marker
echo QUICK_TEST_PASSED > "%~dp0OUTPUT\.quick_test_passed"

echo.
echo =====================================================================
echo  SUCCESS: QUICK TEST PASSED!
echo =====================================================================
echo  All 7 pipeline stages completed successfully.
echo  You may now run the full verification: Double-click 3_RUN_FULL.bat.
echo =====================================================================
echo.
pause
exit /b 0
