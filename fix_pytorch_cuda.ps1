# Quick fix script to upgrade PyTorch to CUDA version
# Run this if PyTorch was installed with CPU-only version

param(
    [string]$PythonCmd = ""
)

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "PyTorch CUDA Upgrade Script" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Detect Python command
if ($PythonCmd -eq "") {
    $venvPaths = @(
        "$PWD\venv\Scripts\python.exe",
        "$PWD\.venv\Scripts\python.exe",
        "$env:USERPROFILE\.venv\LSTM_PFD\Scripts\python.exe"
    )
    
    foreach ($path in $venvPaths) {
        if (Test-Path $path) {
            $PythonCmd = $path
            Write-Host "Using Python: $PythonCmd" -ForegroundColor Green
            break
        }
    }
    
    if ($PythonCmd -eq "") {
        $PythonCmd = "python"
        Write-Host "Using system Python: $PythonCmd" -ForegroundColor Yellow
    }
}

# Check current PyTorch installation
Write-Host "Checking current PyTorch installation..." -ForegroundColor Cyan
$checkScript = @"
import torch
print(f"VERSION:{torch.__version__}")
print(f"CUDA_AVAILABLE:{torch.cuda.is_available()}")
print(f"IS_CPU_BUILD:{'+cpu' in torch.__version__}")
"@

$tempCheck = "$env:TEMP\check_current_torch.py"
$checkScript | Out-File -FilePath $tempCheck -Encoding UTF8
$checkOutput = & $PythonCmd $tempCheck 2>&1
Remove-Item $tempCheck -ErrorAction SilentlyContinue

$currentVersion = ""
$cudaAvailable = $false
$isCpuBuild = $false

if ($checkOutput) {
    foreach ($line in $checkOutput) {
        if ($line -match "VERSION:(.+)") {
            $currentVersion = $matches[1].Trim()
        }
        if ($line -match "CUDA_AVAILABLE:(.+)") {
            $cudaAvailable = ($matches[1].Trim() -eq "True")
        }
        if ($line -match "IS_CPU_BUILD:(.+)") {
            $isCpuBuild = ($matches[1].Trim() -eq "True")
        }
    }
}

Write-Host "Current PyTorch: $currentVersion" -ForegroundColor White
Write-Host "CUDA Available: $cudaAvailable" -ForegroundColor $(if ($cudaAvailable) { "Green" } else { "Red" })
Write-Host ""

if ($cudaAvailable -and -not $isCpuBuild) {
    Write-Host "✓ PyTorch already has CUDA support!" -ForegroundColor Green
    Write-Host "No upgrade needed." -ForegroundColor Green
    exit 0
}

# Check for GPU
Write-Host "Checking for GPU..." -ForegroundColor Cyan
$hasGPU = $false
try {
    $gpuInfo = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpuInfo) {
        $hasGPU = $true
        Write-Host "✓ NVIDIA GPU detected: $($gpuInfo.Name)" -ForegroundColor Green
    } else {
        Write-Host "✗ No NVIDIA GPU detected" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Could not detect GPU" -ForegroundColor Yellow
}

if (-not $hasGPU) {
    Write-Host ""
    Write-Host "No GPU detected. CPU-only PyTorch is appropriate." -ForegroundColor Yellow
    Write-Host "If you have a GPU, make sure NVIDIA drivers are installed." -ForegroundColor Yellow
    exit 0
}

# Upgrade PyTorch
Write-Host ""
Write-Host "Upgrading PyTorch to CUDA version..." -ForegroundColor Cyan
Write-Host "This will:" -ForegroundColor White
Write-Host "  1. Uninstall current PyTorch (CPU-only)" -ForegroundColor White
Write-Host "  2. Install PyTorch with CUDA 11.8 support (~2.5GB download)" -ForegroundColor White
Write-Host "  3. Verify CUDA is available" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "Continue? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Step 1: Uninstalling current PyTorch..." -ForegroundColor Cyan
& $PythonCmd -m pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null

Write-Host "Step 2: Installing PyTorch with CUDA 11.8..." -ForegroundColor Cyan
Write-Host "  (This may take several minutes - large download)" -ForegroundColor Yellow
& $PythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | Out-Null

if ($?) {
    Write-Host ""
    Write-Host "Step 3: Verifying CUDA availability..." -ForegroundColor Cyan
    $verifyScript = @"
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA Available: True")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
else:
    print("✗ CUDA Available: False")
    print("  PyTorch was installed but CUDA is not accessible")
"@
    $tempVerify = "$env:TEMP\verify_cuda.py"
    $verifyScript | Out-File -FilePath $tempVerify -Encoding UTF8
    $verifyOutput = & $PythonCmd $tempVerify 2>&1
    Remove-Item $tempVerify -ErrorAction SilentlyContinue
    Write-Host $verifyOutput
    
    if ($verifyOutput -match "CUDA Available: True") {
        Write-Host ""
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host "✓ SUCCESS! PyTorch upgraded to CUDA version" -ForegroundColor Green
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run your training scripts and they will use GPU." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "⚠ WARNING: PyTorch was installed but CUDA is not accessible" -ForegroundColor Yellow
        Write-Host "  This might mean:" -ForegroundColor Yellow
        Write-Host "  - CUDA runtime libraries are missing" -ForegroundColor Yellow
        Write-Host "  - NVIDIA drivers need to be updated" -ForegroundColor Yellow
        Write-Host "  - GPU is not compatible with CUDA 11.8" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "✗ ERROR: Failed to install PyTorch with CUDA" -ForegroundColor Red
    Write-Host "  Falling back to CPU version..." -ForegroundColor Yellow
    & $PythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | Out-Null
}

Write-Host ""

