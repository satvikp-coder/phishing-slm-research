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

:: Locate Python
set "PY_CMD="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
    goto :python_found
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=py -3.11"
        goto :python_found
    )
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :python_found
)

echo [ERROR] No compatible Python installation found on this computer.
echo.
echo Please run the official installer located in the kit:
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
