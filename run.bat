@echo off
title GitHub Codebase RAG
set PYTHON="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set SCRIPT_DIR=%~dp0

echo Starting GitHub RAG Desktop App...
cd /d "%SCRIPT_DIR%"
%PYTHON% app.py

echo.
echo ========================================================
echo App has closed. If it crashed, read the error above.
echo ========================================================
pause
