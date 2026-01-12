# Changelog

All notable changes to the LSTM_PFD project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
- **Phase 10**: QA & Integration with 90%+ test coverage
- **Phase 9**: Production deployment with quantization and ONNX export
- **Phase 8**: Ensemble learning achieving 98-99% accuracy
- **Phase 7**: Explainable AI module
- **Phase 6**: Physics-Informed Neural Networks (PINN)
- **Phase 5**: Time-Frequency Analysis (STFT, CWT, WVD)
- **Phase 4**: Transformer architecture for signal classification
- **Phase 3**: Advanced CNNs (ResNet, EfficientNet)
- **Phase 2**: 1D CNN baseline
- **Phase 1**: Classical ML baseline (SVM, Random Forest)
- **Phase 0**: Data generation and preprocessing pipeline

### Technical Details

- 11 fault types classified with up to 99% accuracy
- <50ms inference latency
- Docker and Kubernetes deployment support
- REST API with JWT authentication
