#!/usr/bin/env python3
"""Fix problematic indented imports."""

import re
from pathlib import Path
import py_compile

# List of files with compilation errors (remaining after second pass)
ERROR_FILES = [
    "evaluation/cnn_evaluator.py",
    "evaluation/evaluator.py",
    "evaluation/physics_interpretability.py",
    "evaluation/pinn_evaluator.py",
    "evaluation/spectrogram_evaluator.py",
    "evaluation/time_vs_frequency_comparison.py",
    "experiments/cnn_experiment.py",
    "experiments/compare_experiments.py",
    "experiments/ensemble_comparison.py",
    "experiments/experiment_manager.py",
    "experiments/hyperparameter_tuner.py",
    "experiments/pinn_ablation.py",
    "training/bayesian_optimizer.py",
    "training/callbacks.py",
    "training/cnn_callbacks.py",
    "training/cnn_optimizer.py",
    "training/cnn_schedulers.py",
    "training/cnn_trainer.py",
    "training/grid_search.py",
    "training/physics_loss_functions.py",
    "training/pinn_trainer.py",
    "training/random_search.py",
]

def fix_file(filepath):
    """Remove indented/problematic constant imports from file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    removed_count = 0

    # Find the last legitimate top-level import (usually within first 100 lines)
    # We'll remove any constant imports after this point
    last_import_line = 0
    for i, line in enumerate(lines[:100]):
        stripped = line.strip()
        if stripped.startswith('from ') or stripped.startswith('import '):
            # Skip the problematic import itself
            is_const_import = ('from utils.constants import' in line and
                              any(const in line for const in ['NUM_CLASSES', 'SIGNAL_LENGTH', 'SAMPLING_RATE']))
            if not is_const_import:
                last_import_line = i

    for i, line in enumerate(lines):
        # Check if this is the problematic import line
        # Match any variation (might have 1, 2, or all 3 constants)
        is_const_import = ('from utils.constants import' in line and
                          any(const in line for const in ['NUM_CLASSES', 'SIGNAL_LENGTH', 'SAMPLING_RATE']))

        if is_const_import:
            # Remove if:
            # 1. It's indented (not at column 0)
            # 2. It's after the import section (line > last_import_line + 10)
            # 3. Previous line has open paren (multi-line import)
            # 4. Previous import was indented but this one isn't (inside block)

            is_indented = line.startswith(' ') or line.startswith('\t')
            is_after_imports = i > last_import_line + 10
            prev_has_open_paren = i > 0 and '(' in lines[i-1] and ')' not in lines[i-1]

            # Check if previous line is an indented import (we're in a block)
            prev_is_indented_import = False
            if i > 0:
                prev_line = lines[i-1].strip()
                prev_indented = lines[i-1].startswith(' ') or lines[i-1].startswith('\t')
                if prev_indented and (prev_line.startswith('import ') or prev_line.startswith('from ')):
                    prev_is_indented_import = True

            if is_indented:
                print(f"  Line {i+1}: Removing indented import: {line.strip()}")
                removed_count += 1
                continue
            elif is_after_imports:
                print(f"  Line {i+1}: Removing import outside import section: {line.strip()}")
                removed_count += 1
                continue
            elif prev_has_open_paren:
                print(f"  Line {i+1}: Removing import inside multi-line statement: {line.strip()}")
                removed_count += 1
                continue
            elif prev_is_indented_import:
                print(f"  Line {i+1}: Removing import inside code block: {line.strip()}")
                removed_count += 1
                continue

        fixed_lines.append(line)

    # Write back
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)

    return removed_count

def verify_file(filepath):
    """Check if file compiles."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True
    except:
        return False

def main():
    """Fix all error files."""
    project_root = Path(__file__).parent

    print(f"Fixing {len(ERROR_FILES)} files with compilation errors...\n")

    fixed_count = 0
    still_broken = []

    for rel_path in ERROR_FILES:
        filepath = project_root / rel_path

        if not filepath.exists():
            print(f"❌ {rel_path}: File not found")
            continue

        print(f"Fixing {rel_path}...")
        removed = fix_file(filepath)

        # Verify it compiles now
        if verify_file(filepath):
            print(f"  ✅ Fixed! Removed {removed} line(s)\n")
            fixed_count += 1
        else:
            print(f"  ❌ Still has errors after fix\n")
            still_broken.append(rel_path)

    print("=" * 80)
    print(f"✅ Fixed: {fixed_count}/{len(ERROR_FILES)}")

    if still_broken:
        print(f"❌ Still broken: {len(still_broken)}")
        for path in still_broken:
            print(f"  - {path}")
        return 1
    else:
        print("🎉 All files fixed and compile successfully!")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
