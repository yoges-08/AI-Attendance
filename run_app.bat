@echo off
cd /d "%~dp0"
title PowerFace V6 - AI Face Attendance Dashboard
echo ========================================================
echo   Launching PowerFace V6 - AI Face Attendance System
echo ========================================================
"%~dp0venv\Scripts\python.exe" "%~dp0powerface_gui.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application closed with code %ERRORLEVEL%.
)
echo.
pause
