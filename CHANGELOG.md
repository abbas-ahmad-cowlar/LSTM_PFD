# Changelog

All notable changes to the LSTM_PFD project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Documentation Overhaul (IDB 0.0)**: Complete project-wide documentation rewrite
  - Rewrote `README.md` — removed unverified claims, added Mermaid architecture diagram
  - Created `docs/index.md` as centralized navigation hub
  - Created `docs/ARCHITECTURE.md` with 5-domain architecture diagrams
  - Created `docs/GETTING_STARTED.md` with verified setup instructions
  - Created `docs/DOCUMENTATION_STANDARDS.md` — templates and conventions
  - Updated `CONTRIBUTING.md` — fixed project structure, removed placeholder contacts
  - Archived 40+ legacy markdown files to `docs/archive/`
  - Created `docs/archive/ARCHIVE_MASTER_INDEX.md` covering all 48 archived files
  - All 18 IDB sub-teams completed module-level documentation
- Fixed documentation path references: `dash_app/` → `packages/dashboard/`
- Reorganized documentation structure with `docs/user-guide/phases/`
- Cleaned up temporary files from repository root

### Removed

- Deleted temp/log files: `error_log.txt`, `test_log.txt`, etc.
- Removed empty `EXECUTION_DOCS/` directory
- Removed duplicate `dashboard.db` from root

---

## [1.0.0] - 2025-11-28

### Added

- **Phase 11C**: Enterprise Dashboard with XAI integration
  - Data Generation (synthetic + MAT import)
  - Training monitoring with real-time progress
  - XAI Dashboard (SHAP, LIME, Integrated Gradients, Grad-CAM)
  - Experiment management and comparison
- **Phase 10**: QA & Integration
- **Phase 9**: Production deployment with quantization and ONNX export
- **Phase 8**: Ensemble learning
- **Phase 7**: Explainable AI module
- **Phase 6**: Physics-Informed Neural Networks (PINN)
- **Phase 5**: Time-Frequency Analysis (STFT, CWT, WVD)
- **Phase 4**: Transformer architecture for signal classification
- **Phase 3**: Advanced CNNs (ResNet, EfficientNet)
- **Phase 2**: 1D CNN baseline
- **Phase 1**: Classical ML baseline (SVM, Random Forest)
- **Phase 0**: Data generation and preprocessing pipeline

### Technical Details

- 11 fault types classification
- Accuracy: `[PENDING — run experiment to fill]`
- Inference latency: `[PENDING — run experiment to fill]`
- Docker and Kubernetes deployment support
- REST API with JWT authentication
