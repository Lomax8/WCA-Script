@echo off
title Water Contact Angle (WCA) Analyzer UI
cd /d "%~dp0"
echo ===================================================
echo   Starting Water Contact Angle (WCA) Analyzer UI
echo ===================================================
echo.
echo Launching local server at http://127.0.0.1:5000 ...
.\.venv\Scripts\python.exe app.py
pause
