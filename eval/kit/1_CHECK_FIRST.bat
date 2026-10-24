@echo off
setlocal enabledelayedexpansion

:: =====================================================================
:: 1_CHECK_FIRST.bat — Pre-Flight Environment & Integrity Check
:: Arm A Verification Kit
:: All paths relative to this script's directory (%~dp0)
:: =====================================================================

cd /d "%~dp0"
title Arm A Kit - Step 1: Pre-Flight Check

echo =====================================================================
echo  ARM A VERIFICATION KIT -- STEP 1: PRE-FLIGHT CHECK
echo =====================================================================
echo.

:: Ensure OUTPUT directory exists
if not exist "%~dp0OUTPUT" mkdir "%~dp0OUTPUT"

:: Locate Python (Python 3.11 is required for CUDA GPU acceleration)
set "PY_CMD="

:: First check if .venv already exists
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=%~dp0.venv\Scripts\python.exe"
        goto :python_found
    ) else (
        echo [WARNING] Existing .venv uses Python != 3.11.
        echo NVIDIA CUDA GPU wheels require Python 3.11.
        echo Rebuilding .venv with Python 3.11 is recommended for GPU acceleration.
        echo.
    )
)

:: Search specifically for Python 3.11
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=py -3.11"
        goto :python_found
    )
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :python_found
)

if exist "%ProgramFiles%\Python311\python.exe" (
    set "PY_CMD=%ProgramFiles%\Python311\python.exe"
    goto :python_found
)

:: Check if default 'python' is 3.11
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
        goto :python_found
    )
)

:: If Python 3.11 not found, check if ANY python is present
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo [NOTICE] Found Python, but it is not version 3.11.
    echo The offline NVIDIA GPU CUDA acceleration requires Python 3.11.
    echo.
    echo If you want FULL GPU acceleration on your NVIDIA GPU:
    echo   1. Run the installer: "%~dp0prereq\python-3.11.9-amd64.exe"
    echo   2. Delete the .venv folder (if present)
    echo   3. Run 1_CHECK_FIRST.bat again.
    echo.
    set /p USE_FALLBACK="Do you want to continue with CPU-only mode using existing Python? (Y/N): "
    if /i "!USE_FALLBACK!"=="Y" (
        set "PY_CMD=python"
        goto :python_found
    )
)

echo [ERROR] Python 3.11 is required for offline GPU verification.
echo.
echo Please run the official installer included in the kit:
echo   "%~dp0prereq\python-3.11.9-amd64.exe"
echo.
echo (Make sure to check "Add python.exe to PATH" during installation).
echo After installation, run 1_CHECK_FIRST.bat again.
echo.
pause
exit /b 1

:python_found
echo Found Python: %PY_CMD%
echo.

:: Setup virtual environment if not present
if not exist "%~dp0.venv" (
    echo Creating fresh local virtual environment .venv...
    %PY_CMD% -m venv "%~dp0.venv"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

:: Check if packages are installed
"%VENV_PY%" -c "import torch, transformers, peft, sklearn, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Installing dependencies offline from kit wheelhouse...
    if "%1"=="--online" (
        "%VENV_PIP%" install -r "%~dp0requirements-kit.txt"
    ) else (
        "%VENV_PIP%" install --no-index --find-links "%~dp0wheels" -r "%~dp0requirements-kit.txt"
        if !errorlevel! neq 0 (
            echo.
            echo [NOTICE] Offline install encountered an issue. If internet access is available,
            echo you can retry with online fallback:
            echo   1_CHECK_FIRST.bat --online
            echo.
        )
    )
)

:: Check if PyTorch has CUDA acceleration enabled
"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [NOTICE] PyTorch is currently running in CPU mode.
    echo Your Legion Pro 5 has an NVIDIA GPU. For 50x faster GPU verification:
    set /p INSTALL_CUDA="Download official PyTorch CUDA 12.4 for your NVIDIA GPU? (Y/N): "
    if /i "!INSTALL_CUDA!"=="Y" (
        echo Downloading and installing PyTorch with CUDA support...
        "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

echo.
echo Running pre-flight system, integrity, and offline readiness checks...
echo.
"%VENV_PY%" "%~dp0src\preflight.py"
set "PRE_EXIT=%errorlevel%"

echo.
if %PRE_EXIT% equ 0 (
    echo =====================================================================
    echo  PRE-FLIGHT STATUS: PASS
    echo  You are ready to proceed to Step 2: Double-click 2_QUICK_TEST.bat.
    echo =====================================================================
) else (
    echo =====================================================================
    echo  PRE-FLIGHT STATUS: FAIL
    echo  Please inspect OUTPUT\preflight_report.txt for details.
    echo =====================================================================
)

echo.
pause
exit /b %PRE_EXIT%
