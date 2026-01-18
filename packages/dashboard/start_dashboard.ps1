# =============================================================================
# LSTM PFD Dashboard - Production Start Script
# =============================================================================
# This script starts all required services for the dashboard:
# 1. Redis (message broker for Celery) - if not already running
# 2. Celery worker (background task processing)
# 3. Dash application (main dashboard)
#
# Usage: .\start_dashboard.ps1
# =============================================================================

param(
    [switch]$SkipRedisCheck,
    [int]$CeleryWorkers = 2,
    [string]$LogLevel = "info"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Colors for output
function Write-Info { param($msg) Write-Host "[INFO] " -NoNewline -ForegroundColor Cyan; Write-Host $msg }
function Write-Success { param($msg) Write-Host "[OK] " -NoNewline -ForegroundColor Green; Write-Host $msg }
function Write-Warning { param($msg) Write-Host "[WARN] " -NoNewline -ForegroundColor Yellow; Write-Host $msg }
function Write-Error { param($msg) Write-Host "[ERROR] " -NoNewline -ForegroundColor Red; Write-Host $msg }

# Banner
Write-Host ""
Write-Host "=============================================" -ForegroundColor Blue
Write-Host "   LSTM PFD Dashboard - Production Start    " -ForegroundColor Blue
Write-Host "=============================================" -ForegroundColor Blue
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "$ScriptDir\app.py")) {
    Write-Error "app.py not found. Please run this script from the dashboard directory."
    exit 1
}

# Set environment variables
$env:SKIP_CONFIG_VALIDATION = "true"

# -----------------------------------------------------------------------------
# Step 1: Check Redis availability
# -----------------------------------------------------------------------------
Write-Info "Checking Redis connection..."

# Redis installation path
$RedisPath = "C:\Program Files\Redis"
$RedisServer = "$RedisPath\redis-server.exe"
$RedisCli = "$RedisPath\redis-cli.exe"

if (-not $SkipRedisCheck) {
    try {
        # Try to ping Redis
        if (Test-Path $RedisCli) {
            $pingResult = & $RedisCli ping 2>$null
            if ($pingResult -eq "PONG") {
                Write-Success "Redis is running on localhost:6379"
            }
            else {
                throw "Redis not responding"
            }
        }
        else {
            # Fallback to Python check
            $redisTest = python -c "
import redis
import sys
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print('connected')
except:
    print('failed')
    sys.exit(1)
" 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Redis is running on localhost:6379"
            }
            else {
                throw "Redis not responding"
            }
        }
    }
    catch {
        Write-Warning "Redis is not running. Starting Redis..."
        
        # Try to start Redis from known installation path
        if (Test-Path $RedisServer) {
            Start-Process -FilePath $RedisServer -WindowStyle Minimized
            Start-Sleep -Seconds 2
            Write-Success "Redis started from $RedisPath"
        }
        else {
            # Try PATH
            $redisInPath = Get-Command redis-server -ErrorAction SilentlyContinue
            if ($redisInPath) {
                Start-Process -FilePath "redis-server" -WindowStyle Minimized
                Start-Sleep -Seconds 2
                Write-Success "Redis started"
            }
            else {
                Write-Warning "Redis not found. Falling back to CELERY_ALWAYS_EAGER mode (synchronous tasks)"
                Write-Warning "Install Redis with: winget install Redis.Redis"
                $env:CELERY_ALWAYS_EAGER = "true"
            }
        }
    }
}

# -----------------------------------------------------------------------------
# Step 2: Start Celery Worker (if not in eager mode)
# -----------------------------------------------------------------------------
if ($env:CELERY_ALWAYS_EAGER -ne "true") {
    Write-Info "Starting Celery worker with $CeleryWorkers workers..."
    
    # Kill any existing Celery workers
    Get-Process -Name "celery" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    
    # Start Celery in a new PowerShell window
    $celeryCmd = "cd '$ScriptDir'; celery -A tasks worker --loglevel=$LogLevel --concurrency=$CeleryWorkers --pool=solo"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $celeryCmd -WindowStyle Normal
    
    Write-Success "Celery worker started in new window"
    Start-Sleep -Seconds 2
}
else {
    Write-Info "Running in EAGER mode - tasks will execute synchronously"
}

# -----------------------------------------------------------------------------
# Step 3: Start Dash Application
# -----------------------------------------------------------------------------
Write-Info "Starting Dash application..."
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "   Dashboard starting on http://127.0.0.1:8050" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

# Run the app
Set-Location $ScriptDir
python app.py

# Cleanup on exit
if ($env:CELERY_ALWAYS_EAGER -ne "true") {
    Write-Info "Stopping Celery workers..."
    Get-Process -Name "celery" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

Write-Info "Dashboard stopped."
