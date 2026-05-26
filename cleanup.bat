@echo off
chcp 65001 >nul
cd /d "%~dp0"
title LivePartner cleanup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cleanup.ps1" %*
echo.
pause