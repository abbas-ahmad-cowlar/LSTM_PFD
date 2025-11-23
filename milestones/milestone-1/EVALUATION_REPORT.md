# Model Evaluation Report

## Overview
This report summarizes the evaluation results for the `cnn1d` and `attention` models on the bearing fault diagnosis dataset.

## Models Evaluated
1.  **CNN1D**
    - Checkpoint: `results/checkpoints/cnn1d_cpu/cnn1d/cnn1d_20251123_083425_best.pth`
    - Architecture: Basic 1D CNN

2.  **Attention CNN**
    - Checkpoint: `results/checkpoints/attention_cpu/attention/attention_20251123_143212_best.pth`
    - Architecture: Attention-based 1D CNN

## Performance Summary

| Model | Accuracy (Macro) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|------------------|-------------------|----------------|------------------|
| **CNN1D** | ~38.37% | N/A | N/A | N/A |
| **Attention** | ~32.66% | N/A | N/A | N/A |

> [!WARNING]
> **Performance Alert**: The observed accuracy is significantly lower than the expected 96-97% reported in the README. This suggests a potential issue with:
> - Data preprocessing mismatch (e.g., normalization parameters).
> - Dataset split differences (evaluating on data the model wasn't trained for, or data distribution shift).
> - Model loading configuration (though architecture mismatch was resolved).

## Visualizations

### CNN1D
**Confusion Matrix**
![CNN1D Confusion Matrix](results/evaluation/cnn1d/confusion_matrix.png)

**ROC Curves**
![CNN1D ROC Curves](results/evaluation/cnn1d/roc_curves.png)

### Attention CNN
**Confusion Matrix**
![Attention Confusion Matrix](results/evaluation/attention/confusion_matrix.png)

**ROC Curves**
![Attention ROC Curves](results/evaluation/attention/roc_curves.png)

## Failure Analysis
Both models exhibited high error rates. The confusion matrices should be consulted to identify specific classes that are being misclassified. Common patterns often include confusion between similar fault types (e.g., different severity levels of the same fault).

## Recommendations
1.  **Verify Data Preprocessing**: Ensure that the normalization statistics (mean, std) used during training are exactly the same as those used for evaluation.
2.  **Check Data Splits**: Verify that the test set is distinct from the training set but follows the same distribution.
3.  **Inspect Training Logs**: Review the original training logs to confirm the high accuracy was achieved on a validation set that is comparable to this test set.
