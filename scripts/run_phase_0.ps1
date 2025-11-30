# ============================================================================
# PHASE 0 EXECUTION SCRIPT
# ============================================================================
# Purpose: Complete Phase 0 execution from start to finish
#          - Setup environment
#          - Verify requirements
#          - Create directory structure
#          - Generate or import data
#          - Validate output
#
# Usage:   .\scripts\run_phase_0.ps1 [-SkipRequirements] [-UseExistingData] [-NumSignals <number>]
#
# Author:  Auto-generated for LSTM_PFD project
# Date:    2025-11-23
# ============================================================================

param(
    [switch]$SkipRequirements = $false,
    [switch]$UseExistingData = $false,
    [int]$NumSignals = 130,
    [string]$DataMode = "generate"  # "generate" or "import"
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
# STEP 2: CHECK REQUIREMENTS
# ============================================================================
Write-Step -Step "2" -Message "Checking Requirements"

if (-not $SkipRequirements) {
    Write-Info -Message "Running requirements check..."
    & $pythonCmd "$ProjectRoot\check_requirements.py"
    
    if (-not $?) {
        Write-Warning -Message "Some requirements may be missing. Continuing anyway..."
    }
    
    # Check if requirements.txt exists and install packages
    if (Test-Path "$ProjectRoot\requirements.txt") {
        Write-Info -Message "Installing/updating Python packages..."
        & $pythonCmd -m pip install --upgrade pip --quiet
        
        # Install core Phase 0 packages first (essential for data generation)
        # These are the minimum packages needed to run Phase 0 data generation
        Write-Info -Message "Installing core packages for Phase 0..."
        Write-Info -Message "  Note: Installing PyTorch CPU-only (smaller, faster install)"
        
        # Install PyTorch CPU-only first (it's large, so do it separately)
        Write-Info -Message "  Installing PyTorch (CPU-only, this may take a few minutes)..."
        & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
        
        # Install other core packages
        $corePackages = @(
            "numpy>=1.24.0",
            "scipy>=1.10.0", 
            "pandas>=2.0.0",
            "h5py>=3.8.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",           # For config/data_config.py
            "jsonschema>=4.0.0",     # For config/base_config.py validation
            "matplotlib>=3.7.0"      # For utils/visualization_utils.py (imported via utils/__init__.py)
        )
        foreach ($pkg in $corePackages) {
            Write-Info -Message "  Installing $pkg..."
            & $pythonCmd -m pip install $pkg --quiet 2>&1 | Out-Null
            if (-not $?) {
                Write-Warning -Message "  Warning: Failed to install $pkg"
            }
        }
        
        # Now try to install remaining packages from requirements.txt
        # Continue even if some packages fail (like scikit-image needing C compiler)
        Write-Info -Message "Installing remaining packages from requirements.txt..."
        Write-Info -Message "  (Some packages may fail - this is okay if core packages are installed)"
        
        # Temporarily change error action to continue, then restore
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        
        # Run pip install - errors won't stop the script
        & $pythonCmd -m pip install -r "$ProjectRoot\requirements.txt" --no-warn-script-location 2>&1 | Out-Null
        
        # Restore error action
        $ErrorActionPreference = $oldErrorAction
        
        # Always continue - verification step will check if core packages are installed
        Write-Info -Message "Package installation attempt completed (checking results next...)"
        
        # Verify core packages are installed (critical check)
        Write-Info -Message "Verifying core packages for Phase 0..."
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
    import pandas
    print("✓ pandas")
except ImportError:
    missing.append("pandas")
    print("✗ pandas MISSING")

try:
    import h5py
    print("✓ h5py")
except ImportError:
    missing.append("h5py")
    print("✗ h5py MISSING")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError:
    missing.append("scikit-learn")
    print("✗ scikit-learn MISSING")

try:
    import yaml
    print("✓ pyyaml")
except ImportError:
    missing.append("pyyaml")
    print("✗ pyyaml MISSING")

try:
    import jsonschema
    print("✓ jsonschema")
except ImportError:
    missing.append("jsonschema")
    print("✗ jsonschema MISSING")

try:
    import torch
    print("✓ torch (PyTorch)")
except ImportError:
    missing.append("torch")
    print("✗ torch MISSING")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError:
    missing.append("matplotlib")
    print("✗ matplotlib MISSING")

if missing:
    print(f"ERROR: Missing critical packages: {', '.join(missing)}")
    print("Phase 0 cannot proceed without these packages.")
    sys.exit(1)
else:
    print("SUCCESS: All core packages are installed")
    sys.exit(0)
"@
        $tempTest = "$env:TEMP\test_imports.py"
        $testImport | Out-File -FilePath $tempTest -Encoding UTF8
        $verifyOutput = & $pythonCmd $tempTest 2>&1
        Write-Host $verifyOutput
        
        if ($?) {
            Write-Success -Message "Core packages verified - Phase 0 can proceed"
        } else {
            Write-Error -Message "CRITICAL: Core packages are missing. Please install them manually:"
            Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
            Write-Host "  pip install numpy scipy pandas h5py scikit-learn tqdm pyyaml jsonschema matplotlib" -ForegroundColor Yellow
            Write-Host "  Then run this script again." -ForegroundColor Yellow
            Remove-Item $tempTest -ErrorAction SilentlyContinue
            exit 1
        }
        Remove-Item $tempTest -ErrorAction SilentlyContinue
    }
} else {
    Write-Info -Message "Skipping requirements check (--SkipRequirements flag)"
}

# ============================================================================
# STEP 3: CREATE DIRECTORY STRUCTURE
# ============================================================================
Write-Step -Step "3" -Message "Creating Directory Structure"

$directories = @(
    "data\raw\bearing_data\normal",
    "data\raw\bearing_data\ball_fault",
    "data\raw\bearing_data\inner_race",
    "data\raw\bearing_data\outer_race",
    "data\raw\bearing_data\combined",
    "data\raw\bearing_data\imbalance",
    "data\raw\bearing_data\misalignment",
    "data\raw\bearing_data\oil_whirl",
    "data\raw\bearing_data\cavitation",
    "data\raw\bearing_data\looseness",
    "data\raw\bearing_data\oil_deficiency",
    "data\processed",
    "data\spectrograms\stft",
    "data\spectrograms\cwt",
    "data\spectrograms\wvd",
    "checkpoints\phase1",
    "checkpoints\phase2",
    "checkpoints\phase3",
    "checkpoints\phase4",
    "checkpoints\phase5",
    "checkpoints\phase6",
    "checkpoints\phase7",
    "checkpoints\phase8",
    "checkpoints\phase9",
    "logs",
    "results",
    "visualizations",
    "models"
)

$created = 0
$existing = 0

foreach ($dir in $directories) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        $created++
    } else {
        $existing++
    }
}

$dirMsg = "Directory structure ready ($created created, $existing existing)"
Write-Success $dirMsg

# ============================================================================
# STEP 4: CHECK FOR EXISTING DATA
# ============================================================================
Write-Step -Step "4" -Message "Checking for Existing Data"

$hdf5Path = "$ProjectRoot\data\processed\dataset.h5"
$matDir = "$ProjectRoot\data\raw\bearing_data"

$hasHDF5 = Test-Path $hdf5Path
$hasMatFiles = $false

# Count MAT files
if (Test-Path $matDir) {
    $matFiles = Get-ChildItem -Path $matDir -Recurse -Filter "*.mat" -ErrorAction SilentlyContinue
    $matCount = ($matFiles | Measure-Object).Count
    $hasMatFiles = $matCount -gt 0
    
    if ($hasMatFiles) {
        Write-Info -Message "Found $matCount MAT files in data\raw\bearing_data\"
    }
}

if ($hasHDF5) {
    Write-Success -Message "HDF5 dataset found: $hdf5Path"
    Write-Info -Message "You can skip data generation by using existing data."
}

# ============================================================================
# STEP 5: DATA GENERATION OR IMPORT
# ============================================================================
Write-Step -Step "5" -Message "Data Generation/Import"

if ($UseExistingData -and $hasHDF5) {
    Write-Info -Message "Using existing HDF5 dataset (--UseExistingData flag)"
    Write-Success -Message "Phase 0 data already available"
} elseif ($hasMatFiles -and ($DataMode -eq "import")) {
    Write-Info -Message "Importing MAT files to HDF5 format..."
    
    $importScript = "$ProjectRoot\scripts\import_mat_dataset.py"
    if (Test-Path $importScript) {
        & $pythonCmd $importScript `
            --mat_dir $matDir `
            --output $hdf5Path `
            --split-ratios 0.7 0.15 0.15
        
        if ($?) {
            Write-Success -Message "MAT files imported successfully"
        } else {
            Write-Error -Message "Failed to import MAT files"
            exit 1
        }
    } else {
        Write-Error -Message "Import script not found: $importScript"
        exit 1
    }
} else {
    Write-Info -Message "Generating synthetic dataset..."
    Write-Info -Message "  Signals per fault: $NumSignals"
    $totalSignals = $NumSignals * 11
    Write-Info -Message "  Total signals: $totalSignals (11 fault types)"
    
    # Create Python script for data generation
    # Convert Windows path to Python-compatible format
    $projectRootPython = $ProjectRoot -replace '\\', '/'
    $genScript = @"
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(r"$projectRootPython")
if not project_root.exists():
    print(f"ERROR: Project root not found: {project_root}")
    sys.exit(1)
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from data.signal_generator import SignalGenerator
from config.data_config import DataConfig
import time

print("=" * 70)
print("PHASE 0: DATA GENERATION")
print("=" * 70)
print()

# Configure data generation
config = DataConfig(
    num_signals_per_fault=$NumSignals,
    rng_seed=42
)

print(f"Configuration:")
print(f"  Signals per fault: {config.num_signals_per_fault}")
print(f"  Total fault types: {len(config.fault.get_fault_list())}")
print(f"  Random seed: {config.rng_seed}")
print()

# Generate dataset
print("Generating synthetic bearing fault signals...")
start_time = time.time()

generator = SignalGenerator(config)
dataset = generator.generate_dataset()

generation_time = time.time() - start_time

print()
print(f"✓ Generated {len(dataset['signals'])} signals in {generation_time:.1f} seconds")
print()

# Save as HDF5 with automatic train/val/test splits
print("Saving to HDF5 format...")
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

saved_paths = generator.save_dataset(
    dataset,
    output_dir=output_dir,
    format='hdf5',
    train_val_test_split=(0.7, 0.15, 0.15)
)

# Extract the HDF5 path from the dictionary
hdf5_path = saved_paths.get('hdf5')
if hdf5_path is None:
    print("ERROR: HDF5 path not found in saved_paths")
    print(f"Available keys: {list(saved_paths.keys())}")
    sys.exit(1)

print()
print("=" * 70)
print("✓ PHASE 0 COMPLETE")
print("=" * 70)
print(f"Dataset saved to: {hdf5_path}")
print()

# Verify the HDF5 file
import h5py
# Convert Path object to string if needed
hdf5_path_str = str(hdf5_path)
with h5py.File(hdf5_path_str, 'r') as f:
    train_count = f['train']['signals'].shape[0]
    val_count = f['val']['signals'].shape[0]
    test_count = f['test']['signals'].shape[0]
    total = train_count + val_count + test_count
    
    print("Dataset Statistics:")
    print(f"  Train: {train_count} samples ({train_count/total*100:.1f}%)")
    print(f"  Val:   {val_count} samples ({val_count/total*100:.1f}%)")
    print(f"  Test:  {test_count} samples ({test_count/total*100:.1f}%)")
    print(f"  Total: {total} samples")
    print()
"@

    $tempScript = "$env:TEMP\phase0_generate_data.py"
    $genScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    & $pythonCmd $tempScript
    
    if (-not $?) {
        Write-Error -Message "Data generation failed"
        Remove-Item $tempScript -ErrorAction SilentlyContinue
        exit 1
    }
    
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    Write-Success -Message "Data generation complete"
}

# ============================================================================
# STEP 6: VALIDATE OUTPUT
# ============================================================================
Write-Step -Step "6" -Message "Validating Output"

if (Test-Path $hdf5Path) {
    Write-Info -Message "Validating HDF5 dataset..."
    
    # Convert Windows path to Python-compatible format
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
        num_classes = f.attrs.get('num_classes', 'unknown')
        
        print(f"✓ HDF5 file structure valid")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Signal length: {signal_length}")
        print(f"  Number of classes: {num_classes}")
        
        if train_count > 0 and val_count > 0 and test_count > 0:
            print("✓ Dataset validation passed")
            sys.exit(0)
        else:
            print("ERROR: Empty dataset splits")
            sys.exit(1)
            
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"@

    $tempValidate = "$env:TEMP\phase0_validate.py"
    $validateScript | Out-File -FilePath $tempValidate -Encoding UTF8
    
    & $pythonCmd $tempValidate
    
    if ($?) {
        Write-Success -Message "Dataset validation passed"
    } else {
        Write-Warning -Message "Dataset validation had issues (but file exists)"
    }
    
    Remove-Item $tempValidate -ErrorAction SilentlyContinue
} else {
    Write-Warning -Message "HDF5 dataset not found at expected location"
}

# ============================================================================
# STEP 7: RUN BASIC TESTS (OPTIONAL)
# ============================================================================
Write-Step -Step "7" -Message "Running Basic Tests (Optional)"

$runTests = Read-Host "Run unit tests for Phase 0? (y/N)"
if ($runTests -eq "y" -or $runTests -eq "Y") {
    Write-Info -Message "Running Phase 0 unit tests..."
    
    $testFile = "$ProjectRoot\tests\test_data_generation.py"
    if (Test-Path $testFile) {
        & $pythonCmd -m pytest $testFile -v --tb=short
        
        if ($?) {
            Write-Success -Message "All tests passed"
        } else {
            Write-Warning -Message "Some tests failed (this is okay for now)"
        }
    } else {
        Write-Warning -Message "Test file not found: $testFile"
    }
} else {
    Write-Info -Message "Skipping tests (you can run them later with: pytest tests/test_data_generation.py)"
}

# ============================================================================
# SUMMARY
# ============================================================================
$separator = "=" * 70
Write-Host "`n$separator" -ForegroundColor Cyan
Write-Success -Message "PHASE 0 EXECUTION COMPLETE"
Write-Host "$separator" -ForegroundColor Cyan
Write-Host ""

Write-Info -Message "Next Steps:"
Write-Host "  1. Verify dataset: Check data/processed/dataset.h5"
Write-Host "  2. Run Phase 1: python scripts/train_classical_ml.py"
Write-Host "  3. Or use dashboard: cd dash_app && python app.py"
Write-Host ""

Write-Info -Message "Dataset Location:"
Write-Host "  $hdf5Path" -ForegroundColor White
Write-Host ""

# Check for both possible filenames
$hdf5PathAlt = "$ProjectRoot\data\processed\signals_cache.h5"
if (Test-Path $hdf5Path) {
    Write-Success -Message "Phase 0 is ready for Phase 1!"
    Write-Info -Message "Dataset file found: $hdf5Path"
} elseif (Test-Path $hdf5PathAlt) {
    Write-Success -Message "Phase 0 is ready for Phase 1!"
    Write-Info -Message "Dataset file found: $hdf5PathAlt"
} else {
    Write-Warning -Message "Phase 0 may not have completed successfully. Check errors above."
    Write-Info -Message "Expected file at: $hdf5Path or $hdf5PathAlt"
}

Write-Host ""

