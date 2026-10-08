@echo off
cd /d "%~dp0"
title PowerFace V6 - Standalone CLI Video Stream
echo ========================================================
echo   Launching PowerFace V6 - Standalone CLI Mode
echo ========================================================
"%~dp0venv\Scripts\python.exe" "%~dp0ai_face_detection.py"
echo.
pause
