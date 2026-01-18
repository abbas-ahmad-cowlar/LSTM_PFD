@echo off
REM =============================================================================
REM LSTM PFD Dashboard - Quick Start
REM =============================================================================
REM Double-click this file to start the dashboard with Celery worker
REM =============================================================================

echo.
echo =============================================
echo    LSTM PFD Dashboard - Starting...
echo =============================================
echo.

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "start_dashboard.ps1"

pause
