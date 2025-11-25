@echo off
REM Sequential training script for all recommended models
REM Trains: CNN1D → Attention → MultiScale

echo ========================================
echo Sequential Model Training
echo ========================================
echo.
echo This will train 3 models sequentially:
echo   1. CNN1D (baseline)
echo   2. Attention CNN
echo   3. MultiScale CNN
echo.
echo Estimated total time: 9-12 hours (CPU only)
echo.
pause

REM Common parameters
set SIGNAL_LENGTH=102400
set EPOCHS=50
set DATA_DIR=data/raw/bearing_data
set SEED=42
set BASE_CHECKPOINT_DIR=results/checkpoints_full

REM ========================================
REM 1. Train CNN1D (Baseline)
REM ========================================
echo.
echo ========================================
echo [1/3] Training CNN1D (Baseline)
echo ========================================
echo Start time: %TIME%
echo.

venv\Scripts\python.exe scripts/train_cnn.py ^
  --model cnn1d ^
  --signal-length %SIGNAL_LENGTH% ^
  --epochs %EPOCHS% ^
  --batch-size 32 ^
  --lr 0.001 ^
  --scheduler cosine ^
  --data-dir %DATA_DIR% ^
  --seed %SEED% ^
  --checkpoint-dir %BASE_CHECKPOINT_DIR% ^
  --early-stopping ^
  --patience 15

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CNN1D training failed!
    pause
    exit /b 1
)

echo.
echo CNN1D training complete! Time: %TIME%
echo.

REM ========================================
REM 2. Train Attention CNN
REM ========================================
echo.
echo ========================================
echo [2/3] Training Attention CNN
echo ========================================
echo Start time: %TIME%
echo.

venv\Scripts\python.exe scripts/train_cnn.py ^
  --model attention ^
  --signal-length %SIGNAL_LENGTH% ^
  --epochs %EPOCHS% ^
  --batch-size 32 ^
  --lr 0.001 ^
  --scheduler cosine ^
  --data-dir %DATA_DIR% ^
  --seed %SEED% ^
  --checkpoint-dir %BASE_CHECKPOINT_DIR% ^
  --early-stopping ^
  --patience 15

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Attention training failed!
    pause
    exit /b 1
)

echo.
echo Attention training complete! Time: %TIME%
echo.

REM ========================================
REM 3. Train MultiScale CNN
REM ========================================
echo.
echo ========================================
echo [3/3] Training MultiScale CNN
echo ========================================
echo Start time: %TIME%
echo.

venv\Scripts\python.exe scripts/train_cnn.py ^
  --model multiscale ^
  --signal-length %SIGNAL_LENGTH% ^
  --epochs %EPOCHS% ^
  --batch-size 24 ^
  --lr 0.001 ^
  --scheduler cosine ^
  --data-dir %DATA_DIR% ^
  --seed %SEED% ^
  --checkpoint-dir %BASE_CHECKPOINT_DIR% ^
  --early-stopping ^
  --patience 15

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MultiScale training failed!
    pause
    exit /b 1
)

echo.
echo MultiScale training complete! Time: %TIME%
echo.

REM ========================================
REM All Done!
REM ========================================
echo.
echo ========================================
echo ALL TRAINING COMPLETE!
echo ========================================
echo.
echo Trained models saved in: %BASE_CHECKPOINT_DIR%
echo   - %BASE_CHECKPOINT_DIR%\cnn1d\
echo   - %BASE_CHECKPOINT_DIR%\attention\
echo   - %BASE_CHECKPOINT_DIR%\multiscale\
echo.
echo Next steps:
echo   1. Check training logs for final accuracies
echo   2. Run evaluation scripts on each model
echo   3. Compare results
echo.
pause
