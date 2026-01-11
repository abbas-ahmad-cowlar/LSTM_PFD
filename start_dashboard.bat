@echo off
REM Dashboard Launcher
REM Sets up minimal environment variables for development

set DATABASE_URL=sqlite:///dashboard.db
set SECRET_KEY=dev_secret_key
set JWT_SECRET_KEY=dev_jwt_key
set DEBUG=True
set SKIP_CONFIG_VALIDATION=True
set PYTHONPATH=%CD%\packages\dashboard;%CD%
set CELERY_ALWAYS_EAGER=True

echo Starting LSTM PFD Dashboard...
echo Database: %DATABASE_URL%
echo Debug Mode: %DEBUG%
echo Celery Eager Mode: %CELERY_ALWAYS_EAGER%

echo Seeding database...
python packages/dashboard/database/seed_data.py

python packages/dashboard/app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard failed to start.
    pause
)
