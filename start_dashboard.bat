@echo off
REM Dashboard Launcher
REM Sets up minimal environment variables for development

set DATABASE_URL=sqlite:///dashboard.db
set SECRET_KEY=dev_secret_key
set JWT_SECRET_KEY=dev_jwt_key
set DEBUG=True
set SKIP_CONFIG_VALIDATION=True
set PYTHONPATH=%CD%

echo Starting LSTM PFD Dashboard...
echo Database: %DATABASE_URL%
echo Debug Mode: %DEBUG%

python packages/dashboard/app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard failed to start.
    pause
)
