# ============================================================================
# PHASE 1 EXECUTION SCRIPT
# ============================================================================
# Purpose: Complete Phase 1 execution from start to finish
#          - Verify Phase 0 data exists
#          - Check requirements
#          - Run classical ML pipeline
#          - Feature extraction, selection, training, evaluation
#          - Validate output
#
# Usage:   .\scripts\run_phase_1.ps1 [-SkipRequirements] [-SkipHyperOpt] [-NTrials <number>]
#
# Author:  Auto-generated for LSTM_PFD project
# Date:    2025-01-23
# ============================================================================

param(
    [switch]$SkipRequirements = $false,
    [switch]$SkipHyperOpt = $false,
    [int]$NTrials = 50,
    [string]$DataPath = "",  # If empty, auto-detect
    [string]$OutputDir = "results/phase1"
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
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Error -Message "Python 3.8+ required. Found: $pythonVersion"
        exit 1
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
    } else {
        Write-Info -Message "No NVIDIA GPU detected. Will use CPU for computations."
    }
} catch {
    Write-Info -Message "GPU detection failed. Assuming CPU-only mode."
}

# ============================================================================
# STEP 3: CHECK REQUIREMENTS
# ============================================================================
Write-Step -Step "3" -Message "Checking Requirements"

if (-not $SkipRequirements) {
    Write-Info -Message "Running requirements check..."
    
    # Check if requirements.txt exists and install packages
    if (Test-Path "$ProjectRoot\requirements.txt") {
        Write-Info -Message "Installing/updating Python packages..."
        & $pythonCmd -m pip install --upgrade pip --quiet
        
        # Install Phase 1 specific packages
        Write-Info -Message "Installing Phase 1 packages..."
        
        $phase1Packages = @(
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "h5py>=3.8.0",
            "optuna>=3.1.0",
            "pywavelets>=1.4.0",
            "tqdm>=4.65.0",
            "joblib>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        )
        
        foreach ($pkg in $phase1Packages) {
            Write-Info -Message "  Installing $pkg..."
            & $pythonCmd -m pip install $pkg --quiet 2>&1 | Out-Null
            if (-not $?) {
                Write-Warning -Message "  Warning: Failed to install $pkg"
            }
        }
        
        # Verify core packages
        Write-Info -Message "Verifying Phase 1 packages..."
        $testImport = @"
import sys
missing = []
try:
    import numpy
    print("✓ numpy")
except ImportError:
    missing.append("numpy")
    print("✗ numpy MISSING")

try:
    import scipy
    print("✓ scipy")
except ImportError:
    missing.append("scipy")
    print("✗ scipy MISSING")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError:
    missing.append("scikit-learn")
    print("✗ scikit-learn MISSING")

try:
    import h5py
    print("✓ h5py")
except ImportError:
    missing.append("h5py")
    print("✗ h5py MISSING")

try:
    import optuna
    print("✓ optuna")
except ImportError:
    missing.append("optuna")
    print("✗ optuna MISSING")

try:
    import pywt
    print("✓ pywavelets")
except ImportError:
    missing.append("pywavelets")
    print("✗ pywavelets MISSING")

if missing:
    print(f"ERROR: Missing critical packages: {', '.join(missing)}")
    print("Phase 1 cannot proceed without these packages.")
    sys.exit(1)
else:
    print("SUCCESS: All Phase 1 packages are installed")
    sys.exit(0)
"@
        $tempTest = "$env:TEMP\test_phase1_imports.py"
        $testImport | Out-File -FilePath $tempTest -Encoding UTF8
        $verifyOutput = & $pythonCmd $tempTest 2>&1
        Write-Host $verifyOutput
        
        if ($?) {
            Write-Success -Message "Phase 1 packages verified"
        } else {
            Write-Error -Message "CRITICAL: Phase 1 packages are missing. Please install them manually"
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

# Check for HDF5 dataset files (multiple possible locations)
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
        
        print(f"✓ HDF5 file structure valid")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Signal length: {signal_length}")
        print(f"  Sampling rate: {fs}")
        print(f"  Number of classes: {num_classes}")
        
        if train_count > 0 and val_count > 0 and test_count > 0:
            print("✓ Dataset validation passed")
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

$tempValidate = "$env:TEMP\phase1_validate_hdf5.py"
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
# STEP 5: RUN PHASE 1 PIPELINE
# ============================================================================
Write-Step -Step "5" -Message "Running Phase 1 Classical ML Pipeline"

Write-Info -Message "Phase 1 will:"
Write-Host "  1. Extract 36 features from signals" -ForegroundColor White
Write-Host "  2. Select top 15 features using MRMR" -ForegroundColor White
Write-Host "  3. Normalize features" -ForegroundColor White
if (-not $SkipHyperOpt) {
    Write-Host "  4. Optimize hyperparameters ($NTrials trials)" -ForegroundColor White
} else {
    Write-Host "  4. Skip hyperparameter optimization (using defaults)" -ForegroundColor White
}
Write-Host "  5. Train all models (SVM, RF, NN, GBM)" -ForegroundColor White
Write-Host "  6. Select best model based on validation accuracy" -ForegroundColor White
Write-Host "  7. Evaluate on test set" -ForegroundColor White
Write-Host ""

$projectRootPython = $ProjectRoot -replace '\\', '/'
$hdf5PathPython = $hdf5Path -replace '\\', '/'
$outputDirPython = $OutputDir -replace '\\', '/'
$optimizeFlag = if ($SkipHyperOpt) { "False" } else { "True" }

$phase1Script = @"
import sys
import os
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(r"$projectRootPython")
if not project_root.exists():
    print(f"ERROR: Project root not found: {project_root}")
    sys.exit(1)
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

print("=" * 70)
print("PHASE 1: CLASSICAL ML PIPELINE")
print("=" * 70)
print()

# Check GPU availability (for any PyTorch-based components)
import torch
device = torch.device("$deviceInfo")
print(f"Device configuration:")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Device: cuda (GPU acceleration enabled)")
else:
    print(f"  Device: cpu (CPU-only mode)")
print()

from pipelines.classical_ml_pipeline import ClassicalMLPipeline
import h5py
import numpy as np

# Load dataset from HDF5
hdf5_path = Path(r"$hdf5PathPython")
print(f"Loading dataset from: {hdf5_path}")
print()

with h5py.File(hdf5_path, 'r') as f:
    # Load all splits
    X_train = f['train']['signals'][:]
    y_train = f['train']['labels'][:]
    X_val = f['val']['signals'][:]
    y_val = f['val']['labels'][:]
    X_test = f['test']['signals'][:]
    y_test = f['test']['labels'][:]
    fs = f.attrs.get('sampling_rate', 20480)
    
    print("Dataset loaded:")
    train_count = len(X_train)
    val_count = len(X_val)
    test_count = len(X_test)
    print(f"  Train: {train_count} samples")
    print(f"  Val:   {val_count} samples")
    print(f"  Test:  {test_count} samples")
    print(f"  Sampling rate: {fs} Hz")
    print()

# Concatenate all splits (pipeline will re-split internally)
all_signals = np.concatenate([X_train, X_val, X_test], axis=0)
all_labels = np.concatenate([y_train, y_val, y_test], axis=0)

total_signals = len(all_signals)
print(f"Total signals: {total_signals}")
print()

# Initialize pipeline
pipeline = ClassicalMLPipeline(random_state=42)

# Create output directory
output_dir = Path(r"$outputDirPython")
output_dir.mkdir(parents=True, exist_ok=True)

# Run pipeline
print("Starting Phase 1 pipeline execution...")
print("=" * 70)
print()

start_time = time.time()

results = pipeline.run(
    signals=all_signals,
    labels=all_labels,
    fs=fs,
    optimize_hyperparams=$optimizeFlag,
    n_trials=$NTrials,
    save_dir=output_dir
)

elapsed_time = time.time() - start_time

print()
print("=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)
print(f"Execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print()
print("Results Summary:")
best_model = results['best_model']
train_acc = results['train_accuracy']
val_acc = results['val_accuracy']
test_acc = results['test_accuracy']
selected_count = len(results['selected_features'])
print(f"  Best Model: {best_model}")
print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Selected Features: {selected_count}")
print(f"  Results saved to: {output_dir}")
print()
print("=" * 70)

# Check if results meet baseline expectations
if results['test_accuracy'] >= 0.93:
    print("✓ SUCCESS: Test accuracy meets baseline (>= 93%)")
    sys.exit(0)
elif results['test_accuracy'] >= 0.85:
    print("⚠ WARNING: Test accuracy is acceptable but below baseline (>= 85%, < 93%)")
    print("  Consider running with more hyperparameter optimization trials")
    sys.exit(0)
else:
    print("✗ WARNING: Test accuracy is low (< 85%)")
    print("  This may indicate issues with data quality or model training")
    print("  Check the results in the output directory for details")
    sys.exit(1)
"@

$tempPhase1 = "$env:TEMP\run_phase1_pipeline.py"
$phase1Script | Out-File -FilePath $tempPhase1 -Encoding UTF8

Write-Info -Message "Executing Phase 1 pipeline - this may take 20-40 minutes..."
Write-Info -Message "  Feature extraction: ~3-5 minutes"
if (-not $SkipHyperOpt) {
    Write-Info -Message "  Hyperparameter optimization: ~10-20 minutes"
}
Write-Info -Message "  Model training: ~2-5 minutes"
Write-Info -Message "  Evaluation: ~1 minute"
Write-Host ""

& $pythonCmd $tempPhase1

$phase1ExitCode = $LASTEXITCODE

if ($phase1ExitCode -eq 0 -or $phase1ExitCode -eq $null) {
    Write-Success -Message "Phase 1 pipeline executed successfully"
} else {
    Write-Warning -Message "Phase 1 pipeline completed with warnings (exit code: $phase1ExitCode)"
    Write-Info -Message "Check the output above for details"
}

Remove-Item $tempPhase1 -ErrorAction SilentlyContinue

# ============================================================================
# STEP 6: VALIDATE OUTPUT
# ============================================================================
Write-Step -Step "6" -Message "Validating Phase 1 Output"

$outputDirPath = Join-Path $ProjectRoot $OutputDir

if (Test-Path $outputDirPath) {
    Write-Info -Message "Checking output files..."
    $resultsFile = Join-Path $outputDirPath "results.json"
    $modelFile = Join-Path $outputDirPath "best_model.pkl"
    $selectorFile = Join-Path $outputDirPath "feature_selector.pkl"
    $normalizerFile = Join-Path $outputDirPath "normalizer.pkl"
    
    if (Test-Path $resultsFile) {
        Write-Success -Message "  ✓ results.json"
    } else {
        Write-Warning -Message "  ✗ results.json - missing"
    }
    if (Test-Path $modelFile) {
        Write-Success -Message "  ✓ best_model.pkl"
    } else {
        Write-Warning -Message "  ✗ best_model.pkl - missing"
    }
    if (Test-Path $selectorFile) {
        Write-Success -Message "  ✓ feature_selector.pkl"
    } else {
        Write-Warning -Message "  ✗ feature_selector.pkl - missing"
    }
    if (Test-Path $normalizerFile) {
        Write-Success -Message "  ✓ normalizer.pkl"
    } else {
        Write-Warning -Message "  ✗ normalizer.pkl - missing"
    }
    Write-Success -Message "Output validation complete"
} else {
    Write-Warning -Message "Output directory not found: $outputDirPath"
}

# ============================================================================
# SUMMARY
# ============================================================================
$separator = "=" * 70
Write-Host "`n$separator" -ForegroundColor Cyan
Write-Success -Message "PHASE 1 EXECUTION COMPLETE"
Write-Host "$separator" -ForegroundColor Cyan
Write-Host ""

Write-Info -Message "Hardware Configuration:"
if ($hasGPU) {
    Write-Host "  GPU: $gpuName" -ForegroundColor Green
    Write-Host "  CUDA: $cudaVersion" -ForegroundColor Green
    Write-Host "  Device: GPU-accelerated for PyTorch components" -ForegroundColor Green
} else {
    Write-Host "  GPU: Not detected" -ForegroundColor Yellow
    Write-Host "  Device: CPU-only" -ForegroundColor Yellow
}
Write-Host ""

Write-Info -Message "Output Location:"
Write-Host "  $outputDirPath" -ForegroundColor White
Write-Host ""

Write-Info -Message "Next Steps:"
Write-Host "  1. Review results: Check $OutputDir\results.json" -ForegroundColor White
Write-Host "  2. Run Phase 2: Deep learning with 1D CNNs" -ForegroundColor White
Write-Host "  3. Or use dashboard: cd dash_app; python app.py" -ForegroundColor White
Write-Host ""

if ($phase1ExitCode -eq 0 -or $phase1ExitCode -eq $null) {
    Write-Success -Message "Phase 1 is ready! You can proceed to Phase 2."
} else {
    Write-Warning -Message "Phase 1 completed with warnings. Review results before proceeding."
}

Write-Host ""

