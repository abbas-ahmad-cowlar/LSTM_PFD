#!/usr/bin/env python3
"""Check syntax of all Python files in the project."""

import py_compile
import sys
from pathlib import Path

def check_file(filepath):
    """Check if a Python file compiles."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Check all Python files."""
    project_root = Path(__file__).parent

    # Find all Python files
    python_files = list(project_root.rglob("*.py"))

    # Exclude venv, .git, __pycache__
    python_files = [
        f for f in python_files
        if not any(part.startswith('.') or part in ['venv', '__pycache__', 'env']
                   for part in f.parts)
    ]

    errors = []
    for filepath in sorted(python_files):
        success, error = check_file(filepath)
        if not success:
            rel_path = filepath.relative_to(project_root)
            errors.append((rel_path, error))

    if errors:
        print(f"❌ COMPILATION ERRORS FOUND: {len(errors)}")
        for path, error in errors:
            # Extract just the line info
            error_lines = error.split('\n')
            error_msg = error_lines[0] if error_lines else error
            print(f"  {path}: {error_msg}")
        return 1
    else:
        print(f"✅ ALL {len(python_files)} PYTHON FILES COMPILE SUCCESSFULLY")
        return 0

if __name__ == "__main__":
    sys.exit(main())
