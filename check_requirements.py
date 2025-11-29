"""
Check which required software and packages are installed for LSTM_PFD project.
"""
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 8)
    installed = (version.major, version.minor)
    status = "✅" if installed >= required else "❌"
    return {
        "name": "Python",
        "required": f"{required[0]}.{required[1]}+",
        "installed": f"{installed[0]}.{installed[1]}.{version.micro}",
        "status": status,
        "installed_bool": installed >= required
    }

def check_command(command, name):
    """Check if a command exists."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return {
                "name": name,
                "required": "Yes",
                "installed": version,
                "status": "[OK]",
                "installed_bool": True
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "name": name,
        "required": "Yes",
        "installed": "NOT INSTALLED",
                "status": "[MISSING]",
        "installed_bool": False
    }

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "installed (version unknown)")
            return {
                "name": package_name,
                "required": "Yes",
                "installed": version,
                "status": "[OK]",
                "installed_bool": True
            }
    except (ImportError, AttributeError):
        pass
    
    return {
        "name": package_name,
        "required": "Yes",
        "installed": "NOT INSTALLED",
                "status": "[MISSING]",
        "installed_bool": False
    }

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        
        return {
            "name": "PyTorch",
            "required": "2.0.0+",
            "installed": version,
            "status": "[OK]",
            "installed_bool": True,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version
        }
    except ImportError:
        return {
            "name": "PyTorch",
            "required": "2.0.0+",
            "installed": "NOT INSTALLED",
                "status": "[MISSING]",
            "installed_bool": False,
            "cuda_available": False,
            "cuda_version": "N/A"
        }

def main():
    """Main function to check all requirements."""
    print("=" * 70)
    print("LSTM_PFD Software Requirements Check")
    print("=" * 70)
    print()
    
    results = []
    
    # System Requirements
    print("SYSTEM REQUIREMENTS")
    print("-" * 70)
    results.append(check_python_version())
    results.append(check_command("git", "Git"))
    results.append(check_command("docker", "Docker"))
    results.append(check_command("docker-compose", "Docker Compose"))
    results.append(check_command("psql", "PostgreSQL (CLI)"))
    results.append(check_command("redis-cli", "Redis (CLI)"))
    
    # Check for NVIDIA GPU
    nvidia_check = check_command("nvidia-smi", "NVIDIA GPU/CUDA")
    if not nvidia_check["installed_bool"]:
        nvidia_check["note"] = "Optional - CPU training works but is slower"
    results.append(nvidia_check)
    
    # PyTorch (special check)
    pytorch_info = check_pytorch_cuda()
    results.append(pytorch_info)
    
    # Core Python Packages
    print("\nCORE PYTHON PACKAGES")
    print("-" * 70)
    core_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("h5py", "h5py"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("tqdm", "tqdm"),
    ]
    
    for pkg_name, import_name in core_packages:
        results.append(check_python_package(pkg_name, import_name))
    
    # ML/DL Packages
    print("\nMACHINE LEARNING PACKAGES")
    print("-" * 70)
    ml_packages = [
        ("xgboost", "xgboost"),
        ("optuna", "optuna"),
        ("shap", "shap"),
        ("lime", "lime"),
        ("captum", "captum"),
    ]
    
    for pkg_name, import_name in ml_packages:
        results.append(check_python_package(pkg_name, import_name))
    
    # Dashboard Packages
    print("\nDASHBOARD PACKAGES")
    print("-" * 70)
    dashboard_packages = [
        ("dash", "dash"),
        ("dash-bootstrap-components", "dash_bootstrap_components"),
        ("flask", "flask"),
        ("sqlalchemy", "sqlalchemy"),
        ("psycopg2-binary", "psycopg2"),
        ("redis", "redis"),
        ("celery", "celery"),
    ]
    
    for pkg_name, import_name in dashboard_packages:
        results.append(check_python_package(pkg_name, import_name))
    
    # API/Deployment Packages
    print("\nAPI/DEPLOYMENT PACKAGES")
    print("-" * 70)
    api_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
    ]
    
    for pkg_name, import_name in api_packages:
        results.append(check_python_package(pkg_name, import_name))
    
    # Testing Packages
    print("\nTESTING PACKAGES")
    print("-" * 70)
    test_packages = [
        ("pytest", "pytest"),
        ("pytest-cov", "pytest_cov"),
    ]
    
    for pkg_name, import_name in test_packages:
        results.append(check_python_package(pkg_name, import_name))
    
    # Print Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    installed = [r for r in results if r["installed_bool"]]
    missing = [r for r in results if not r["installed_bool"]]
    
    print(f"\n[OK] Installed: {len(installed)}/{len(results)}")
    print(f"[MISSING] Missing: {len(missing)}/{len(results)}")
    
    if missing:
        print("\n[MISSING] MISSING SOFTWARE/PACKAGES:")
        print("-" * 70)
        for item in missing:
            note = f" ({item.get('note', '')})" if 'note' in item else ""
            print(f"  • {item['name']:30} {note}")
    
    # Special notes
    print("\nNOTES:")
    print("-" * 70)
    
    if pytorch_info.get("cuda_available"):
        print(f"  [OK] CUDA is available - GPU training enabled")
        print(f"     CUDA Version: {pytorch_info.get('cuda_version', 'N/A')}")
    else:
        print(f"  [WARN] CUDA not available - Will use CPU (slower but works)")
    
    if not check_command("psql", "")["installed_bool"]:
        print(f"  [INFO] PostgreSQL CLI not found, but Docker can provide it")
        print(f"     Use: docker-compose up (in dash_app/) to start PostgreSQL")
    
    if not check_command("redis-cli", "")["installed_bool"]:
        print(f"  [INFO] Redis CLI not found, but Docker can provide it")
        print(f"     Use: docker-compose up (in dash_app/) to start Redis")
    
    print("\n" + "=" * 70)
    print("INSTALLATION RECOMMENDATIONS")
    print("=" * 70)
    
    if not pytorch_info["installed_bool"]:
        print("\n1. Install PyTorch (CRITICAL):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   (or use 'cpu' instead of 'cu118' for CPU-only)")
    
    if missing:
        print("\n2. Install missing Python packages:")
        print("   pip install -r requirements.txt")
        print("   pip install -r dash_app/requirements.txt")
        print("   pip install -r requirements-deployment.txt  # (optional, for API)")
        print("   pip install -r requirements-test.txt  # (optional, for testing)")
    
    if not check_command("docker", "")["installed_bool"]:
        print("\n3. Install Docker Desktop for Windows:")
        print("   https://www.docker.com/products/docker-desktop/")
        print("   (Required for dashboard with PostgreSQL/Redis)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

