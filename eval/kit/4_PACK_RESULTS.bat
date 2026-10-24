@echo off
setlocal enabledelayedexpansion

:: =====================================================================
:: 4_PACK_RESULTS.bat — Results Packaging Utility
:: Arm A Verification Kit
:: All paths relative to this script's directory (%~dp0)
:: =====================================================================

cd /d "%~dp0"
title Arm A Kit - Step 4: Pack Results

echo =====================================================================
echo  ARM A VERIFICATION KIT -- STEP 4: PACK RESULTS
echo =====================================================================
echo.

if not exist "%~dp0OUTPUT" (
    echo [ERROR] No OUTPUT directory found!
    echo Please run 2_QUICK_TEST.bat or 3_RUN_FULL.bat first.
    echo.
    pause
    exit /b 1
)

:: Get computer name and timestamp
set "M_NAME=%COMPUTERNAME%"
if "%M_NAME%"=="" set "M_NAME=unknown_host"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%I"
if "%DT%"=="" (
    set "TIMESTAMP=%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
    set "TIMESTAMP=!TIMESTAMP: =0!"
) else (
    set "TIMESTAMP=%DT:~0,8%_%DT:~8,6%"
)

set "ZIP_NAME=RESULTS_%M_NAME%_%TIMESTAMP%.zip"
set "ZIP_PATH=%~dp0%ZIP_NAME%"

echo Packaging OUTPUT directory into:
echo   %ZIP_NAME%
echo.

powershell -NoProfile -Command "Compress-Archive -Path '%~dp0OUTPUT\*' -DestinationPath '%ZIP_PATH%' -Force"

if %errorlevel% neq 0 (
    echo [ERROR] Failed to create ZIP archive.
    echo You can manually copy the entire OUTPUT folder instead.
    echo.
    pause
    exit /b 1
)

echo =====================================================================
echo  RESULTS PACKED SUCCESSFULLY!
echo =====================================================================
echo.
echo  File location:
echo    %ZIP_PATH%
echo.
echo  INSTRUCTIONS:
echo  1. Copy this ZIP file onto your USB drive.
echo  2. Bring it back to Satvik / Prof. Das.
echo  3. Delete this verification kit folder from this computer afterwards.
echo =====================================================================
echo.
pause
exit /b 0
