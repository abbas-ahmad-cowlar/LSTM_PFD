# Contributing to LSTM_PFD

Thank you for your interest in contributing to LSTM_PFD! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## 🤝 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

**Our Standards**:

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (optional, but recommended for deep learning)
- 16GB+ RAM (32GB recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/LSTM_PFD.git
cd LSTM_PFD
```

3. Add upstream remote:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/LSTM_PFD.git
```

---

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-deployment.txt
pip install -r requirements-test.txt
```

### 3. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Check code style
black --check .
flake8 .
```

---

## 🔄 Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions**:

- `feature/feature-name` - New features
- `fix/bug-name` - Bug fixes
- `docs/doc-name` - Documentation updates
- `refactor/refactor-name` - Code refactoring
- `test/test-name` - Test additions/updates

### 2. Make Changes

- Write clear, concise code
- Follow the project's code style (see [Code Style](#code-style))
- Add/update tests for your changes
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_features.py -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature X to improve Y

- Detailed description of change
- Why the change was made
- Any breaking changes"
```

**Commit message guidelines**:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests after first line

### 5. Keep Your Branch Up to Date

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch
git rebase upstream/main
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=. --cov-report=term-missing --cov-report=html

# Run in parallel
pytest -n auto

# Run specific markers
pytest -m unit
pytest -m integration
pytest -m slow
```

### Writing Tests

**Test Structure**:

```python
import pytest

class TestFeatureName:
    """Test suite for FeatureName."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = ...

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

**Test Guidelines**:

- Each test should test one thing
- Use descriptive test names
- Use fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU tests with `@pytest.mark.gpu`

### Coverage Goals

- Target: **>90% code coverage**
- Critical modules: **>95% coverage**
- Focus on testing edge cases and error handling

---

## 🎨 Code Style

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Maximum line length: 100 characters
# Use 4 spaces for indentation
# Use double quotes for strings

# Good
def calculate_features(signal: np.ndarray, fs: int = 20480) -> Dict[str, float]:
    """
    Calculate features from signal.

    Args:
        signal: Input signal array
        fs: Sampling frequency

    Returns:
        Dictionary of features
    """
    features = {}
    features["mean"] = np.mean(signal)
    return features
```

### Formatting Tools

```bash
# Auto-format code
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def process_signals(
    signals: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process signals with optional normalization."""
    ...
```

### Documentation

**Docstring Format** (Google style):

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001
) -> Dict[str, List[float]]:
    """
    Train PyTorch model.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Dictionary containing training history with keys:
            - 'train_loss': Training losses per epoch
            - 'val_accuracy': Validation accuracies per epoch

    Raises:
        ValueError: If epochs < 1 or lr <= 0

    Example:
        >>> model = create_cnn1d(num_classes=11)
        >>> history = train_model(model, train_loader, epochs=50)
        >>> print(f"Final accuracy: {history['val_accuracy'][-1]:.4f}")
    """
    ...
```

---

## 📤 Submitting Changes

### 1. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to the repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR template:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained/improved

## Checklist

- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated (if applicable)
```

### 3. Code Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

### 4. After Merge

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Delete your feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**:
   ```
   1. Run command X
   2. Input Y
   3. See error Z
   ```
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - PyTorch version
   - GPU (if applicable)
6. **Logs/Screenshots**: Relevant error messages or screenshots

### Feature Requests

When requesting features, include:

1. **Problem**: What problem does this feature solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Any alternative solutions considered?
4. **Additional Context**: Screenshots, examples, etc.

---

## 📚 Additional Resources

### Project Structure

```
LSTM_PFD/
├── packages/
│   ├── core/               # Core ML engine
│   │   ├── models/         # Model architectures
│   │   ├── training/       # Training pipeline
│   │   ├── evaluation/     # Metrics & evaluation
│   │   ├── features/       # Feature extraction
│   │   └── explainability/ # XAI methods
│   ├── dashboard/          # Enterprise dashboard
│   │   ├── layouts/        # UI layouts
│   │   ├── services/       # Backend services
│   │   ├── callbacks/      # Dash callbacks
│   │   └── tasks/          # Celery async tasks
│   └── deployment/         # Deployment utilities
├── data/                   # Data engineering
├── config/                 # Configuration
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── scripts/               # Utility & research scripts
├── deploy/                # Deployment scripts
└── docs/                  # Documentation
```

### Helpful Commands

```bash
# Run linters
black . && isort . && flake8 .

# Run all tests with coverage
pytest --cov=. --cov-report=html

# Build documentation
cd docs && make html

# Run benchmarks
python tests/benchmarks/benchmark_suite.py

# Start API server
uvicorn api.main:app --reload
```

### Getting Help

- **GitHub Issues**: https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues
- **Discussions**: https://github.com/abbas-ahmad-cowlar/LSTM_PFD/discussions

---

## 🏆 Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project README

Thank you for contributing to LSTM_PFD! 🎉

---

**Last Updated**: February 2026
