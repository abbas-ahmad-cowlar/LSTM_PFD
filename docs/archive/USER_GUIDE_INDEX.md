# User Guide

This guide covers all aspects of using LSTM PFD, from the web dashboard to command-line tools.

## Choose Your Interface

<div class="grid cards" markdown>

- :material-monitor-dashboard:{ .lg .middle } **Dashboard**

  ***

  Web-based interface for visual ML operations.

  - No coding required
  - Real-time training monitoring
  - Interactive XAI visualizations

  [:octicons-arrow-right-24: Dashboard Guide](dashboard/overview.md)

- :material-console:{ .lg .middle } **Command Line**

  ***

  Scripts and automation for power users.

  - Full control over training
  - Batch processing
  - CI/CD integration

  [:octicons-arrow-right-24: CLI Guide](cli/overview.md)

- :material-school:{ .lg .middle } **Phases**

  ***

  Step-by-step learning path through all 11 phases.

  - Classical ML to Deep Learning
  - PINN and XAI
  - Production deployment

  [:octicons-arrow-right-24: Phase Guide](phases/overview.md)

</div>

## Quick Reference

### Common Tasks

| Task                | Dashboard                | CLI                                |
| ------------------- | ------------------------ | ---------------------------------- |
| Generate Data       | Data Explorer → Generate | `python scripts/run_phase0.py`     |
| Train Model         | Experiments → New        | `python scripts/train_cnn.py`      |
| View Results        | Experiment → Results     | `python scripts/evaluate_model.py` |
| Explain Predictions | XAI Dashboard            | `python scripts/explain.py`        |
| Deploy Model        | Deployment               | `python scripts/export_onnx.py`    |

### Model Selection Guide

| Your Need     | Recommended Model          | Accuracy |
| ------------- | -------------------------- | -------- |
| Fast baseline | Random Forest (Phase 1)    | 95-96%   |
| Best accuracy | Stacked Ensemble (Phase 8) | 98-99%   |
| Interpretable | PINN (Phase 6)             | 97-98%   |
| Low latency   | ResNet-18 INT8 (Phase 9)   | 96-97%   |

## Next Steps

1. **New to ML?** Start with [Phase 1 - Classical ML](phases/phase-1.md)
2. **Have experience?** Jump to [Phase 6 - PINN](phases/phase-6.md)
3. **Need deployment?** See [Phase 9 - Deployment](phases/phase-9.md)
