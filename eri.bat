@echo off
REM ============================================================
REM LivePartner / Eri -- one-click launcher
REM
REM Double-click to start. The Python console stays open behind
REM the Qt window so you can see startup logs / catch errors.
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"
title LivePartner (Eri)

REM Make src/ importable without installing the package.
set "PYTHONPATH=%~dp0src"

REM Belt-and-braces: keep HuggingFace caches on D: so they don't
REM dribble onto C:. __main__.py also sets these defensively.
if "%HF_HOME%"=="" set "HF_HOME=%~dp0.hf-cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM Use the project's venv python directly -- no need to activate.
".\.venv\Scripts\python.exe" -m livepartner

REM Pause only on crash so the user can read the traceback;
REM clean exit just closes the window when they close the UI.
if errorlevel 1 (
    echo.
    echo === LivePartner exited with code %errorlevel% ===
    pause
)