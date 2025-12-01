# ============================================================================
# PHASE 2 EXECUTION SCRIPT
# ============================================================================
# Purpose: Complete Phase 2 execution from start to finish
#          - Verify Phase 0 data exists
#          - Check requirements (PyTorch, GPU if available)
#          - Train 1D CNN models
#          - Evaluate on test set
#          - Validate output
#
# Usage:   .\scripts\run_phase_2.ps1 [-SkipRequirements] [-Model <model>] [-Epochs <number>] [-BatchSize <number>] [-UseGPU]
#
# Author:  Auto-generated for LSTM_PFD project
# Date:    2025-01-23
# ============================================================================

param(
    [switch]$SkipRequirements = $false,
    [string]$Model = "cnn1d",  # Options: cnn1d, attention, attention-lite, multiscale, dilated
    [int]$Epochs = 50,
    [int]$BatchSize = 32,
    [switch]$UseGPU = $false,  # If true, force GPU (if available)
    [string]$DataPath = "",  # If empty, auto-detect
    [string]$CheckpointDir = "checkpoints/phase2",
    [switch]$MixedPrecision = $false,  # Use mixed precision (FP16) training
    [switch]$EarlyStopping = $false,  # Enable early stopping
    [int]$Patience = 10  # Early stopping patience
)

# Set error handling
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Colors for output
function Write-Info { 
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan 
}
function Write-Success { 
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green 
}
function Write-Warning { 
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow 
}
function Write-Error { 
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red 
}
function Write-Step { 
    param([string]$Step, [string]$Message)
    Write-Host "`n[$Step] $Message" -ForegroundColor Magenta 
}

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Change to project root
Set-Location $ProjectRoot
Write-Info -Message "Project Root: $ProjectRoot"

# ============================================================================
# STEP 1: CHECK PYTHON AND VENV
# ============================================================================
Write-Step -Step "1" -Message "Checking Python and Virtual Environment"

$pythonCmd = $null
$venvPath = $null

# Check for Python
try {
    $pythonVersionOutput = python --version 2>&1
    if ($?) {
        $pythonVersion = $pythonVersionOutput.ToString()
        Write-Success "Python found: $pythonVersion"
        $pythonCmd = "python"
    } else {
        throw "Python command failed"
    }
} catch {
    Write-Error "Python not found. Please install Python 3.8+"
    exit 1
}

# Check Python version
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
$pythonMajor = 0
$pythonMinor = 0
if ($versionMatch) {
    $pythonMajor = [int]$matches[1]
    $pythonMinor = [int]$matches[2]
    if ($pythonMajor -lt 3 -or ($pythonMajor -eq 3 -and $pythonMinor -lt 8)) {
        Write-Error -Message "Python 3.8+ required. Found: $pythonVersion"
        exit 1
    }
    # Warn about Python 3.13+ not having CUDA wheels yet
    if ($pythonMajor -eq 3 -and $pythonMinor -ge 13) {
        Write-Warning -Message "Python $pythonMajor.$pythonMinor detected - PyTorch CUDA wheels may not be available yet"
        Write-Warning -Message "If GPU acceleration fails, consider using Python 3.12 for CUDA support"
        Write-Info -Message "  To fix: Install Python 3.12, then recreate venv with: C:\Python312\python.exe -m venv venv"
    }
}

# Check for virtual environment
$venvPaths = @(
    "$ProjectRoot\venv",
    "$ProjectRoot\.venv",
    "$env:USERPROFILE\.venv\LSTM_PFD"
)

foreach ($path in $venvPaths) {
    if (Test-Path "$path\Scripts\python.exe") {
        $venvPath = $path
        Write-Success -Message "Virtual environment found: $venvPath"
        $pythonCmd = "$venvPath\Scripts\python.exe"
        break
    }
}

if (-not $venvPath) {
    Write-Warning -Message "Virtual environment not found. Creating one..."
    Write-Info -Message "Creating virtual environment at: $ProjectRoot\venv"
    python -m venv "$ProjectRoot\venv"
    
    if (-not $?) {
        Write-Error -Message "Failed to create virtual environment"
        exit 1
    }
    
    $venvPath = "$ProjectRoot\venv"
    $pythonCmd = "$venvPath\Scripts\python.exe"
    Write-Success -Message "Virtual environment created"
}

# Activate virtual environment
Write-Info -Message "Activating virtual environment..."
& "$venvPath\Scripts\Activate.ps1"

# ============================================================================
# STEP 2: GPU DETECTION
# ============================================================================
Write-Step -Step "2" -Message "Detecting GPU Hardware"

$hasGPU = $false
$gpuName = "None"
$cudaVersion = "N/A"
$deviceInfo = "cpu"

try {
    # Check for NVIDIA GPU
    $gpuInfo = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpuInfo) {
        $hasGPU = $true
        $gpuName = $gpuInfo.Name
        Write-Success -Message "NVIDIA GPU detected: $gpuName"
        Write-Info -Message "  Adapter RAM: $([math]::Round($gpuInfo.AdapterRAM / 1GB, 2)) GB"
        Write-Info -Message "  Driver Version: $($gpuInfo.DriverVersion)"
        
        # Try to detect CUDA version from nvidia-smi
        try {
            $nvidiaSmi = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
            if ($?) {
                Write-Success -Message "NVIDIA drivers are properly installed"
                # Infer CUDA compatibility from driver version
                $driverMajor = [int]($gpuInfo.DriverVersion.Split('.')[0])
                if ($driverMajor -ge 522) {
                    $cudaVersion = "12.x"
                    Write-Info -Message "  CUDA Compatibility: $cudaVersion - based on driver version"
                } elseif ($driverMajor -ge 456) {
                    $cudaVersion = "11.x"
                    Write-Info -Message "  CUDA Compatibility: $cudaVersion - based on driver version"
                } else {
                    Write-Warning -Message "  Driver version may be too old for modern CUDA support"
                    $cudaVersion = "Legacy"
                }
                $deviceInfo = "cuda"
            }
        } catch {
            Write-Warning -Message "nvidia-smi not available. GPU detected but CUDA support uncertain."
        }
        
        # GPU is detected and ready for Phase 2 training
        Write-Success -Message "GPU will be used for neural network training - significant speedup expected"
    } else {
        Write-Info -Message "No NVIDIA GPU detected. Training will use CPU - slower performance."
        Write-Warning -Message "  Note: GPU is highly recommended for Phase 2+ deep learning"
    }
} catch {
    Write-Info -Message "GPU detection failed. Assuming CPU-only mode."
}

# ============================================================================
# STEP 3: CHECK REQUIREMENTS
# ============================================================================
Write-Step -Step "3" -Message "Checking Requirements"

if (-not $SkipRequirements) {
    Write-Info -Message "Checking Python packages for Phase 2..."
    
    if (Test-Path "$ProjectRoot\requirements.txt") {
        Write-Info -Message "Installing/updating Python packages..."
        & $pythonCmd -m pip install --upgrade pip --quiet
        
        # Install Phase 2 specific packages (PyTorch is critical)
        Write-Info -Message "Installing Phase 2 packages..."
        
        # Comprehensive PyTorch check and auto-fix
        Write-Info -Message "Checking PyTorch installation and CUDA support..."
        
        $torchCheckScript = @"
import torch
import sys
try:
    version = torch.__version__
    cuda_available = torch.cuda.is_available()
    # Check if it's CPU-only build
    is_cpu_build = '+cpu' in version
    # Also check if CUDA version attribute exists
    has_cuda_version = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
    
    print(f"INSTALLED:True")
    print(f"VERSION:{version}")
    print(f"CUDA_AVAILABLE:{cuda_available}")
    print(f"IS_CPU_BUILD:{is_cpu_build}")
    print(f"HAS_CUDA_VERSION:{has_cuda_version}")
    
    if cuda_available:
        print(f"GPU_NAME:{torch.cuda.get_device_name(0)}")
        print(f"CUDA_VERSION:{torch.version.cuda}")
    
    sys.exit(0)
except ImportError:
    print("INSTALLED:False")
    sys.exit(1)
"@
        $tempTorchCheck = "$env:TEMP\check_torch_detailed.py"
        $torchCheckScript | Out-File -FilePath $tempTorchCheck -Encoding UTF8
        $torchCheckOutput = & $pythonCmd $tempTorchCheck 2>&1
        Remove-Item $tempTorchCheck -ErrorAction SilentlyContinue
        
        $torchInstalled = $false
        $torchVersion = ""
        $cudaAvailable = $false
        $isCpuBuild = $false
        $hasCudaVersion = $false
        
        if ($torchCheckOutput) {
            foreach ($line in $torchCheckOutput) {
                if ($line -match "INSTALLED:(.+)") {
                    $torchInstalled = ($matches[1].Trim() -eq "True")
                }
                if ($line -match "VERSION:(.+)") {
                    $torchVersion = $matches[1].Trim()
                }
                if ($line -match "CUDA_AVAILABLE:(.+)") {
                    $cudaAvailable = ($matches[1].Trim() -eq "True")
                }
                if ($line -match "IS_CPU_BUILD:(.+)") {
                    $isCpuBuild = ($matches[1].Trim() -eq "True")
                }
                if ($line -match "HAS_CUDA_VERSION:(.+)") {
                    $hasCudaVersion = ($matches[1].Trim() -eq "True")
                }
            }
        }
        
        if (-not $torchInstalled) {
            Write-Warning -Message "PyTorch not found. Installing appropriate version..."
            
            # Auto-install based on GPU detection
            if ($hasGPU -and $cudaVersion -ne "Legacy") {
                Write-Info -Message "  Installing PyTorch with CUDA support for GPU acceleration..."
                Write-Info -Message "  This is a large download ~2.5GB and may take several minutes..."
                
                # Try CUDA 12.4 first (for modern drivers), fallback to 12.1, then 11.8
                $cudaVersionsToTry = @("cu124", "cu121", "cu118")
                $cudaInstallSuccess = $false
                
                foreach ($cudaVer in $cudaVersionsToTry) {
                    Write-Info -Message "  Trying PyTorch with $cudaVer..."
                    $installOutput = & $pythonCmd -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cudaVer" 2>&1
                    
                    # Verify installation worked and CUDA is available
                    $verifyResult = & $pythonCmd -c "import torch; print(torch.cuda.is_available())" 2>&1
                    if ($verifyResult -eq "True") {
                        Write-Success -Message "  PyTorch with $cudaVer installed successfully!"
                        $cudaVerify = & $pythonCmd -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1
                        Write-Info -Message "  $cudaVerify"
                        $cudaInstallSuccess = $true
                        break
                    } else {
                        Write-Warning -Message "  $cudaVer not available for Python $pythonMajor.$pythonMinor, trying next..."
                        & $pythonCmd -m pip uninstall torch torchvision torchaudio -y --quiet 2>&1 | Out-Null
                    }
                }
                
                if (-not $cudaInstallSuccess) {
                    Write-Warning -Message "  No CUDA-enabled PyTorch available for Python $pythonMajor.$pythonMinor"
                    Write-Warning -Message "  This is likely because Python $pythonMajor.$pythonMinor is too new."
                    Write-Warning -Message "  Falling back to CPU version (training will be slower)..."
                    Write-Info -Message "  TIP: For GPU support, use Python 3.12: C:\Python312\python.exe -m venv venv"
                    & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
                }
            } else {
                Write-Info -Message "  Installing PyTorch CPU-only..."
                & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
            }
        } else {
            # PyTorch is installed - check if it needs upgrade
            Write-Info -Message "  PyTorch found: $torchVersion"
            
            if ($hasGPU -and $cudaVersion -ne "Legacy") {
                # Check if we need to upgrade from CPU-only to CUDA version
                $needsUpgrade = $false
                $upgradeReason = ""
                
                if ($isCpuBuild) {
                    $needsUpgrade = $true
                    $upgradeReason = "CPU-only build detected (+cpu in version)"
                } elseif (-not $cudaAvailable) {
                    $needsUpgrade = $true
                    $upgradeReason = "CUDA not available despite GPU being present"
                } elseif (-not $hasCudaVersion) {
                    $needsUpgrade = $true
                    $upgradeReason = "PyTorch doesn't have CUDA version info"
                }
                
                if ($needsUpgrade) {
                    Write-Warning -Message "  GPU detected but PyTorch doesn't have proper CUDA support"
                    Write-Info -Message "  Reason: $upgradeReason"
                    Write-Info -Message "  Auto-upgrading PyTorch to CUDA version for GPU acceleration..."
                    Write-Info -Message "  This may take a few minutes - larger download (~2.5GB)..."
                    
                    # Uninstall existing PyTorch
                    Write-Info -Message "  Step 1/2: Uninstalling existing CPU-only PyTorch..."
                    & $pythonCmd -m pip uninstall torch torchvision torchaudio -y --quiet 2>&1 | Out-Null
                    
                    # Try CUDA versions in order: 12.4, 12.1, 11.8
                    Write-Info -Message "  Step 2/2: Installing PyTorch with CUDA support..."
                    $cudaVersionsToTry = @("cu124", "cu121", "cu118")
                    $cudaUpgradeSuccess = $false
                    
                    foreach ($cudaVer in $cudaVersionsToTry) {
                        Write-Info -Message "    Trying $cudaVer..."
                        $installOutput = & $pythonCmd -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cudaVer" 2>&1
                        
                        # Verify CUDA actually works
                        $verifyResult = & $pythonCmd -c "import torch; print(torch.cuda.is_available())" 2>&1
                        if ($verifyResult -eq "True") {
                            $cudaVerify = & $pythonCmd -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('CUDA Version:', torch.version.cuda)" 2>&1
                            Write-Host $cudaVerify
                            Write-Success -Message "  PyTorch successfully upgraded to CUDA version ($cudaVer) - GPU acceleration enabled!"
                            $deviceInfo = "cuda"
                            $cudaUpgradeSuccess = $true
                            break
                        } else {
                            Write-Warning -Message "    $cudaVer not available for Python $pythonMajor.$pythonMinor"
                            & $pythonCmd -m pip uninstall torch torchvision torchaudio -y --quiet 2>&1 | Out-Null
                        }
                    }
                    
                    if (-not $cudaUpgradeSuccess) {
                        Write-Warning -Message "  No CUDA-enabled PyTorch available for Python $pythonMajor.$pythonMinor"
                        Write-Warning -Message "  This is likely because your Python version is too new for PyTorch CUDA wheels."
                        Write-Warning -Message "  Reinstalling CPU version - training will be slower (10-15 hours vs 2-3 hours)"
                        Write-Info -Message ""
                        Write-Info -Message "  =========================================="
                        Write-Info -Message "  TO ENABLE GPU ACCELERATION:"
                        Write-Info -Message "  1. Install Python 3.12 from python.org"
                        Write-Info -Message "  2. Delete the venv folder"
                        Write-Info -Message "  3. Create new venv: C:\Python312\python.exe -m venv venv"
                        Write-Info -Message "  4. Run this script again"
                        Write-Info -Message "  =========================================="
                        Write-Info -Message ""
                        & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
                    }
                } else {
                    Write-Success -Message "  ??? PyTorch already has CUDA support - ready for GPU acceleration"
                    if ($torchCheckOutput -match "GPU_NAME:(.+)") {
                        $gpuName = $matches[1].Trim()
                        Write-Info -Message "    GPU: $gpuName"
                    }
                    $deviceInfo = "cuda"
                }
            } else {
                Write-Info -Message "  PyTorch installed (CPU version - no GPU detected)"
            }
        }
        
        # Update deviceInfo after PyTorch installation/upgrade
        if ($hasGPU -and $cudaVersion -ne "Legacy") {
            $cudaFinalCheck = & $pythonCmd -c "import torch; print(torch.cuda.is_available())" 2>&1
            if ($cudaFinalCheck -eq "True") {
                $deviceInfo = "cuda"
            }
        }
        
        # Install other Phase 2 packages
        $phase2Packages = @(
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "h5py>=3.8.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
            "jsonschema>=4.0.0",
            "pywavelets>=1.4.0",
            "python-dotenv>=1.0.0",
            "joblib>=1.3.0",
            "pillow>=10.0.0",
            "tabulate>=0.9.0",
            "tensorboard>=2.10.0"
        )
        
        foreach ($pkg in $phase2Packages) {
            Write-Info -Message "  Verifying $pkg..."
            & $pythonCmd -m pip install $pkg --quiet 2>&1 | Out-Null
        }
        
        # Verify core packages
        Write-Info -Message "Verifying Phase 2 packages..."
        $testImport = @"
import sys
missing = []
try:
    import torch
    print("??? torch")
except ImportError:
    missing.append("torch")
    print("??? torch MISSING")

try:
    import numpy
    print("??? numpy")
except ImportError:
    missing.append("numpy")
    print("??? numpy MISSING")

try:
    import h5py
    print("??? h5py")
except ImportError:
    missing.append("h5py")
    print("??? h5py MISSING")

try:
    import sklearn
    print("??? scikit-learn")
except ImportError:
    missing.append("scikit-learn")
    print("??? scikit-learn MISSING")

if missing:
    print(f"ERROR: Missing critical packages: {', '.join(missing)}")
    print("Phase 2 cannot proceed without these packages.")
    sys.exit(1)
else:
    print("SUCCESS: All Phase 2 packages are installed")
    sys.exit(0)
"@
        $tempTest = "$env:TEMP\test_phase2_imports.py"
        $testImport | Out-File -FilePath $tempTest -Encoding UTF8
        $verifyOutput = & $pythonCmd $tempTest 2>&1
        Write-Host $verifyOutput
        
        if ($?) {
            Write-Success -Message "Phase 2 packages verified"
        } else {
            Write-Error -Message "CRITICAL: Phase 2 packages are missing. Please install them manually"
            Remove-Item $tempTest -ErrorAction SilentlyContinue
            exit 1
        }
        Remove-Item $tempTest -ErrorAction SilentlyContinue
    } else {
        Write-Warning -Message "requirements.txt not found, continuing anyway..."
    }
} else {
    Write-Info -Message "Skipping requirements check (--SkipRequirements flag)"
}

# ============================================================================
# STEP 4: CHECK FOR PHASE 0 DATA
# ============================================================================
Write-Step -Step "4" -Message "Checking for Phase 0 Data"

# Check for HDF5 dataset files
$possibleDataPaths = @(
    "$ProjectRoot\data\processed\dataset.h5",
    "$ProjectRoot\data\processed\signals_cache.h5"
)

$hdf5Path = $null

# Use custom path if provided
if ($DataPath -ne "") {
    if (Test-Path $DataPath) {
        $hdf5Path = $DataPath
        Write-Success -Message "Using custom data path: $hdf5Path"
    } else {
        Write-Error -Message "Custom data path not found: $DataPath"
        exit 1
    }
} else {
    # Auto-detect
    foreach ($path in $possibleDataPaths) {
        if (Test-Path $path) {
            $hdf5Path = $path
            Write-Success -Message "HDF5 dataset found: $hdf5Path"
            break
        }
    }
}

if (-not $hdf5Path) {
    Write-Error -Message "HDF5 dataset not found. Phase 0 must be completed first."
    Write-Info -Message "Expected files:"
    foreach ($path in $possibleDataPaths) {
        Write-Host "  - $path" -ForegroundColor Yellow
    }
    Write-Info -Message "`nPlease run Phase 0 first:"
    Write-Host "  .\scripts\run_phase_0.ps1" -ForegroundColor Yellow
    exit 1
}

# Validate HDF5 file structure
Write-Info -Message "Validating HDF5 file structure..."
$hdf5PathPython = $hdf5Path -replace '\\', '/'
$validateScript = @"
import sys
import h5py
from pathlib import Path

hdf5_path = Path(r"$hdf5PathPython")

if not hdf5_path.exists():
    print("ERROR: HDF5 file not found")
    sys.exit(1)

try:
    with h5py.File(hdf5_path, 'r') as f:
        # Check structure
        required_groups = ['train', 'val', 'test']
        for group in required_groups:
            if group not in f:
                print(f"ERROR: Missing group: {group}")
                sys.exit(1)
            
            if 'signals' not in f[group] or 'labels' not in f[group]:
                print(f"ERROR: Missing datasets in {group}")
                sys.exit(1)
        
        # Get statistics
        train_count = f['train']['signals'].shape[0]
        val_count = f['val']['signals'].shape[0]
        test_count = f['test']['signals'].shape[0]
        signal_length = f['train']['signals'].shape[1]
        fs = f.attrs.get('sampling_rate', 20480)
        num_classes = f.attrs.get('num_classes', 'unknown')
        
        print(f"??? HDF5 file structure valid")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Signal length: {signal_length}")
        print(f"  Sampling rate: {fs}")
        print(f"  Number of classes: {num_classes}")
        
        if train_count > 0 and val_count > 0 and test_count > 0:
            print("??? Dataset validation passed")
            sys.exit(0)
        else:
            print("ERROR: Empty dataset splits")
            sys.exit(1)
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

$tempValidate = "$env:TEMP\phase2_validate_hdf5.py"
$validateScript | Out-File -FilePath $tempValidate -Encoding UTF8

$validateOutput = & $pythonCmd $tempValidate 2>&1
Write-Host $validateOutput

if (-not $?) {
    Write-Error -Message "HDF5 file validation failed"
    Remove-Item $tempValidate -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item $tempValidate -ErrorAction SilentlyContinue
Write-Success -Message "HDF5 dataset validated successfully"

# ============================================================================
# STEP 5: VERIFY GPU IN PYTORCH
# ============================================================================
Write-Step -Step "5" -Message "Verifying GPU in PyTorch"

$gpuCheckScript = @"
import torch
import sys
if torch.cuda.is_available():
    print(f"??? GPU available: {torch.cuda.get_device_name(0)}")
    print(f"??? CUDA version: {torch.version.cuda}")
    print(f"??? GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    sys.exit(0)
else:
    print("??? GPU not available - will use CPU (training will be slower)")
    print("  Expected training time: 10-15 hours (CPU) vs 2-3 hours (GPU)")
    sys.exit(1)
"@

$tempGpuCheck = "$env:TEMP\check_gpu.py"
$gpuCheckScript | Out-File -FilePath $tempGpuCheck -Encoding UTF8
$gpuOutput = & $pythonCmd $tempGpuCheck 2>&1
Write-Host $gpuOutput
$gpuAvailable = $?
Remove-Item $tempGpuCheck -ErrorAction SilentlyContinue

if ($gpuAvailable) {
    Write-Success -Message "GPU will be used for training - expected time: 2-3 hours"
    $deviceInfo = "cuda"
} else {
    Write-Warning -Message "GPU not available - CPU training will be significantly slower, expected time: 10-15 hours"
    $deviceInfo = "cpu"
}

# ============================================================================
# STEP 6: RUN PHASE 2 CNN TRAINING
# ============================================================================
Write-Step -Step "6" -Message "Running Phase 2 CNN Training"

Write-Info -Message "Phase 2 will:"
Write-Host "  1. Load data from HDF5 file" -ForegroundColor White
Write-Host "  2. Create $Model model" -ForegroundColor White
Write-Host "  3. Train for $Epochs epochs (batch size: $BatchSize)" -ForegroundColor White
Write-Host "  4. Evaluate on test set" -ForegroundColor White
Write-Host "  5. Save best model checkpoint" -ForegroundColor White
Write-Host ""

# Build command arguments
$deviceArg = if ($UseGPU -and $gpuAvailable) { "cuda" } else { "auto" }
$mixedPrecArg = if ($MixedPrecision) { "--mixed-precision" } else { "" }
$earlyStopArg = if ($EarlyStopping) { "--early-stopping --patience $Patience" } else { "" }

$projectRootPython = $ProjectRoot -replace '\\', '/'
$hdf5PathPython = $hdf5Path -replace '\\', '/'
$checkpointDirPython = $CheckpointDir -replace '\\', '/'

Write-Info -Message "Training Configuration:"
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Epochs: $Epochs" -ForegroundColor White
Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
Write-Host "  Device: $deviceArg" -ForegroundColor White
Write-Host "  Data: $hdf5Path" -ForegroundColor White
Write-Host "  Checkpoints: $CheckpointDir" -ForegroundColor White
Write-Host ""

# Estimate training time
if ($gpuAvailable) {
    $estimatedHours = [math]::Round(($Epochs * 3) / 60, 1)
    Write-Info -Message "Estimated training time: ~$estimatedHours hours with GPU"
} else {
    $estimatedHours = [math]::Round(($Epochs * 12) / 60, 1)
    Write-Warning -Message "Estimated training time: ~$estimatedHours hours with CPU - consider using GPU"
}

Write-Host ""
Write-Info -Message "Starting CNN training..."
Write-Host ""

# Execute training script
$trainScript = "$ProjectRoot\scripts\train_cnn.py"
$trainArgs = @(
    "--model", $Model,
    "--data-path", $hdf5PathPython,
    "--epochs", $Epochs.ToString(),
    "--batch-size", $BatchSize.ToString(),
    "--checkpoint-dir", $checkpointDirPython,
    "--device", $deviceArg
)

if ($MixedPrecision) {
    $trainArgs += "--mixed-precision"
}

if ($EarlyStopping) {
    $trainArgs += "--early-stopping"
    $trainArgs += "--patience"
    $trainArgs += $Patience.ToString()
}

& $pythonCmd $trainScript @trainArgs

$trainExitCode = $LASTEXITCODE

if ($trainExitCode -eq 0 -or $trainExitCode -eq $null) {
    Write-Success -Message "Phase 2 CNN training completed successfully"
} else {
    Write-Warning -Message "Phase 2 training completed with warnings (exit code: $trainExitCode)"
    Write-Info -Message "Check the output above for details"
}

# ============================================================================
# STEP 7: VALIDATE OUTPUT
# ============================================================================
Write-Step -Step "7" -Message "Validating Phase 2 Output"

$checkpointDirPath = Join-Path $ProjectRoot $CheckpointDir
$modelCheckpointDir = Join-Path $checkpointDirPath $Model

if (Test-Path $modelCheckpointDir) {
    Write-Info -Message "Checking output files..."
    
    # Look for best model checkpoint
    $bestCheckpoint = Get-ChildItem -Path $modelCheckpointDir -Filter "*_best.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if ($bestCheckpoint) {
        Write-Success -Message "  ??? Best model checkpoint: $($bestCheckpoint.Name)"
    } else {
        Write-Warning -Message "  ??? Best model checkpoint not found"
    }
    
    # Check for any checkpoint files
    $allCheckpoints = @(Get-ChildItem -Path $modelCheckpointDir -Filter "*.pth" -ErrorAction SilentlyContinue)
    if ($allCheckpoints.Count -gt 0) {
        Write-Success -Message "  [OK] Found $($allCheckpoints.Count) checkpoint file(s)"
    } else {
        Write-Warning -Message "  [WARN] No checkpoint files found"
    }
    
    Write-Success -Message "Output validation complete"
} else {
    Write-Warning -Message "Checkpoint directory not found: $modelCheckpointDir"
}

# ============================================================================
# SUMMARY
# ============================================================================
$separator = "=" * 70
Write-Host "`n$separator" -ForegroundColor Cyan
Write-Success -Message "PHASE 2 EXECUTION COMPLETE"
Write-Host "$separator" -ForegroundColor Cyan
Write-Host ""

Write-Info -Message "Hardware Configuration:"
if ($hasGPU -and $gpuAvailable) {
    Write-Host "  GPU: $gpuName" -ForegroundColor Green
    Write-Host "  CUDA: $cudaVersion" -ForegroundColor Green
    Write-Host "  Training Device: GPU-accelerated" -ForegroundColor Green
} else {
    Write-Host "  GPU: Not used" -ForegroundColor Yellow
    Write-Host "  Training Device: CPU-only" -ForegroundColor Yellow
}
Write-Host ""

Write-Info -Message "Model Checkpoints:"
Write-Host "  $modelCheckpointDir" -ForegroundColor White
Write-Host ""

Write-Info -Message "Next Steps:"
Write-Host "  1. Review training logs for accuracy metrics" -ForegroundColor White
Write-Host "  2. Evaluate model: python scripts/evaluate_cnn.py --checkpoint <path>" -ForegroundColor White
Write-Host "  3. Run Phase 3: Advanced CNNs - ResNet, EfficientNet" -ForegroundColor White
Write-Host "  4. Or use dashboard: cd dash_app; python app.py" -ForegroundColor White
Write-Host ""

if ($trainExitCode -eq 0 -or $trainExitCode -eq $null) {
    Write-Success -Message "Phase 2 is complete! Checkpoint saved for future use."
} else {
    Write-Warning -Message "Phase 2 completed with warnings. Review logs before proceeding."
}

Write-Host ""





