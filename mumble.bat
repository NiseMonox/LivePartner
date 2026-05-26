@echo off
REM ============================================================
REM Mumble server -- foreground mode for LivePartner local use
REM
REM Run this BEFORE eri.bat so Eri has a server to connect to.
REM Close this window (Ctrl+C or X) to stop the server.
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"
title Mumble Server (LivePartner)

REM Try standard Mumble install paths; fall back to PATH.
set "MUMBLE_EXE=C:\Program Files\Mumble\server\mumble-server.exe"
if not exist "%MUMBLE_EXE%" set "MUMBLE_EXE=C:\Program Files (x86)\Mumble\server\mumble-server.exe"
if not exist "%MUMBLE_EXE%" set "MUMBLE_EXE=mumble-server.exe"

echo Starting Mumble server...
echo   binary : %MUMBLE_EXE%
echo   cwd    : %CD%
echo   data   : mumble-server.sqlite (in cwd)
echo   port   : 64738 (TCP+UDP, default)
echo.
echo Connect from this PC via 127.0.0.1
echo Connect from LAN     via your laptop IP (run ipconfig to find it)
echo.
echo Ctrl+C or close window to stop.
echo ------------------------------------------------------------

REM -fg: run in foreground so closing window stops the server.
REM Server reads mumble-server.ini next to the exe if present.
"%MUMBLE_EXE%" -fg

if errorlevel 1 (
    echo.
    echo === Mumble server exited with code %errorlevel% ===
    pause
)