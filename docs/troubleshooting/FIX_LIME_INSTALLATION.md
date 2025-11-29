# Fixing LIME Installation Issue

## Problem
`lime` package requires `scikit-image` which needs to be compiled from source on Python 3.14 (no pre-built wheel available). This requires a C compiler.

## Solution Options

### Option 1: Install Visual Studio Build Tools (Recommended)

1. **Download Visual Studio Build Tools:**
   - Go to: https://visualstudio.microsoft.com/downloads/
   - Download "Build Tools for Visual Studio 2022"
   - Or direct link: https://aka.ms/vs/17/release/vs_buildtools.exe

2. **Install:**
   - Run the installer
   - Select "Desktop development with C++" workload
   - This includes MSVC compiler, Windows SDK, etc.
   - Install size: ~6 GB

3. **Restart terminal and try again:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Install Pre-built scikit-image (Easier)

Try installing a pre-built wheel directly:

```bash
# Install scikit-image from a wheel (if available)
pip install scikit-image

# Then install lime
pip install lime
```

### Option 3: Skip LIME for Now (Quick Fix)

LIME is only needed for Phase 7 (Explainable AI) and is optional. You can:

1. **Comment out LIME in requirements.txt:**
   ```bash
   # Edit requirements.txt and comment this line:
   # lime>=0.2.0.1
   ```

2. **Install everything else:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install LIME later when needed:**
   - After installing Visual Studio Build Tools
   - Or use alternative XAI methods (SHAP, Captum) which are already installing

### Option 4: Use Python 3.11 or 3.12 (Alternative)

Python 3.14 is very new and some packages don't have pre-built wheels yet. Consider:

1. **Create new virtual environment with Python 3.11:**
   ```bash
   # Download Python 3.11 from python.org
   py -3.11 -m venv venv311
   venv311\Scripts\activate
   pip install -r requirements.txt
   ```

## Recommended Action

**For now:** Use Option 3 (skip LIME) to continue with the project. You can install it later when you reach Phase 7.

**For later:** Install Visual Studio Build Tools if you want full XAI capabilities.

## Verify Installation

After fixing, verify:
```bash
python -c "import lime; print('LIME installed successfully')"
```

