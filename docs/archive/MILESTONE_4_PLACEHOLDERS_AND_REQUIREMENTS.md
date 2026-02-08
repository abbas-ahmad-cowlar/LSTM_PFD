# Placeholders and Requirements Document

**Milestone 4: Comprehensive Research Report**

This document lists all placeholders, missing data, and requirements that must be filled in after completing experimental training and evaluation for all three milestones.

---

## Document Status

- **Report Length:** Currently ~40-50 pages with placeholders
- **Sections:** All written, awaiting experimental results
- **Figures:** 0/25+ created (all placeholders)
- **Tables:** Structure complete, data pending
- **Bibliography:** Complete (40+ references)
- **Placeholders:** ~150+ \todo{} markers

---

## 1. Performance Metrics to Fill

### 1.1 Overall Accuracy Results

**Location:** `sections/results.tex`, Table `tab:overall_results`

| Metric | Current | Required |
|--------|---------|----------|
| CNN Best Accuracy | \todo{XX.XX ± X.XX} | Actual value ± std dev |
| LSTM Best Accuracy | \todo{XX.XX ± X.XX} | Actual value ± std dev |
| Hybrid Best Accuracy | \todo{XX.XX ± X.XX} | Actual value ± std dev |
| CNN Macro-F1 | \todo{X.XXX} | Actual value |
| LSTM Macro-F1 | \todo{X.XXX} | Actual value |
| Hybrid Macro-F1 | \todo{X.XXX} | Actual value |
| CNN Parameters | \todo{XX.XM} | Actual count in millions |
| LSTM Parameters | \todo{X.XM} | Actual count in millions |
| Hybrid Parameters | \todo{XX.XM} | Actual count in millions |

### 1.2 CNN Architecture Comparison

**Location:** `sections/results.tex`, Table `tab:cnn_results`

For each architecture (15+ total), provide:
- Parameters (M)
- Test Accuracy (%)
- Precision (macro avg)
- Recall (macro avg)
- F1-score (macro avg)
- Inference time (ms/sample)

**Architectures:**
- CNN-1D
- ResNet-18
- ResNet-34
- ResNet-50
- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B4
- [Additional architectures as implemented]

### 1.3 LSTM Configuration Results

**Location:** `sections/results.tex`, Table `tab:lstm_results`

| Configuration | Required Data |
|--------------|---------------|
| Vanilla LSTM (hidden=128) | Params, Acc%, F1, Train time, Inf. time |
| Vanilla LSTM (hidden=256) | Params, Acc%, F1, Train time, Inf. time |
| BiLSTM (hidden=128) | Params, Acc%, F1, Train time, Inf. time |
| BiLSTM (hidden=256) | Params, Acc%, F1, Train time, Inf. time |

### 1.4 Hybrid Recommended Configurations

**Location:** `sections/results.tex`, Table `tab:hybrid_recommended`

| Configuration | Required Data |
|--------------|---------------|
| Recommended 1 (ResNet34+BiLSTM) | Params, Acc% ± std, F1, Inf. time |
| Recommended 2 (EfficientNet-B2+BiLSTM) | Params, Acc% ± std, F1, Inf. time |
| Recommended 3 (ResNet18+LSTM) | Params, Acc% ± std, F1, Inf. time |

### 1.5 Per-Class Performance

**Location:** `sections/results.tex`, Table `tab:cnn_per_class`

For the best model from each approach, provide per-class metrics:

| Fault Class | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Healthy | \todo{} | \todo{} | \todo{} | \todo{} |
| Misalignment | \todo{} | \todo{} | \todo{} | \todo{} |
| Imbalance | \todo{} | \todo{} | \todo{} | \todo{} |
| Bearing Clearance | \todo{} | \todo{} | \todo{} | \todo{} |
| Lubrication Issue | \todo{} | \todo{} | \todo{} | \todo{} |
| Cavitation | \todo{} | \todo{} | \todo{} | \todo{} |
| Wear | \todo{} | \todo{} | \todo{} | \todo{} |
| Oil Whirl | \todo{} | \todo{} | \todo{} | \todo{} |
| Mixed Fault 1 | \todo{} | \todo{} | \todo{} | \todo{} |
| Mixed Fault 2 | \todo{} | \todo{} | \todo{} | \todo{} |
| Mixed Fault 3 | \todo{} | \todo{} | \todo{} | \todo{} |

**Repeat for:** CNN best, LSTM best, Hybrid best

---

## 2. Ablation Study Results

### 2.1 CNN Backbone Ablation

**Location:** `sections/results.tex`, Table `tab:ablation_cnn`

For each CNN backbone (with fixed LSTM: BiLSTM-256):

| CNN Backbone | CNN Params | Total Params | Accuracy | Δ from Best |
|--------------|------------|--------------|----------|-------------|
| CNN-1D | \todo{} | \todo{} | \todo{} | \todo{} |
| ResNet-18 | \todo{} | \todo{} | \todo{} | \todo{} |
| ResNet-34 | \todo{} | \todo{} | \todo{} | \todo{} |
| ResNet-50 | \todo{} | \todo{} | \todo{} | \todo{} |
| EfficientNet-B0 | \todo{} | \todo{} | \todo{} | \todo{} |
| EfficientNet-B2 | \todo{} | \todo{} | \todo{} | \todo{} |
| EfficientNet-B4 | \todo{} | \todo{} | \todo{} | \todo{} |

### 2.2 LSTM Type Ablation

**Location:** `sections/results.tex`, Table `tab:ablation_lstm_type`

For each LSTM configuration (with fixed CNN: ResNet-34):

| LSTM Type | Hidden Size | Params | Accuracy | Inference Time |
|-----------|-------------|--------|----------|----------------|
| Vanilla LSTM | 128 | \todo{} | \todo{} | \todo{} |
| Vanilla LSTM | 256 | \todo{} | \todo{} | \todo{} |
| BiLSTM | 128 | \todo{} | \todo{} | \todo{} |
| BiLSTM | 256 | \todo{} | \todo{} | \todo{} |

### 2.3 Temporal Pooling Ablation

**Location:** `sections/results.tex`, Table `tab:ablation_pooling`

| Pooling Method | Additional Params | Accuracy | Δ from Mean |
|----------------|-------------------|----------|-------------|
| Mean | 0 | \todo{} | 0.00 |
| Max | 0 | \todo{} | \todo{} |
| Last Timestep | 0 | \todo{} | \todo{} |
| Attention | \todo{XXK} | \todo{} | \todo{} |

### 2.4 CNN Freezing Ablation

**Location:** `sections/results.tex`, Table `tab:ablation_freezing`

| Training Strategy | Accuracy | Training Time | GPU Memory |
|------------------|----------|---------------|------------|
| Frozen CNN | \todo{} | \todo{} | \todo{} |
| Fine-tuned CNN | \todo{} | \todo{} | \todo{} |
| Improvement | \todo{} | \todo{} | \todo{} |

---

## 3. Computational Metrics

### 3.1 Training Efficiency

**Location:** `sections/results.tex`, Table `tab:training_efficiency`

| Approach | Time/Epoch (s) | Total Time (h) | GPU Mem (GB) | Convergence Epoch |
|----------|----------------|----------------|--------------|-------------------|
| CNN (ResNet-34) | \todo{} | \todo{} | \todo{} | \todo{} |
| LSTM (BiLSTM-256) | \todo{} | \todo{} | \todo{} | \todo{} |
| Hybrid (ResNet34+BiLSTM) | \todo{} | \todo{} | \todo{} | \todo{} |

### 3.2 Inference Efficiency

**Location:** `sections/results.tex`, Table `tab:inference_efficiency`

| Approach | GPU (ms) | CPU (ms) | Throughput (samples/s) | Model Size (MB) |
|----------|----------|----------|------------------------|-----------------|
| CNN | \todo{} | \todo{} | \todo{} | \todo{} |
| LSTM | \todo{} | \todo{} | \todo{} | \todo{} |
| Hybrid | \todo{} | \todo{} | \todo{} | \todo{} |

---

## 4. Robustness Analysis

### 4.1 Performance Under Noise

**Location:** `sections/results.tex`, Table `tab:noise_robustness`

| Approach | Clean | σ=0.01 | σ=0.05 | σ=0.1 |
|----------|-------|--------|--------|-------|
| CNN | \todo{} | \todo{} | \todo{} | \todo{} |
| LSTM | \todo{} | \todo{} | \todo{} | \todo{} |
| Hybrid | \todo{} | \todo{} | \todo{} | \todo{} |

---

## 5. Statistical Analysis

### 5.1 Statistical Significance

**Location:** `sections/results.tex`, Table `tab:cnn_vs_lstm`

| Metric | Best CNN | Best LSTM | Difference | p-value | Significant? |
|--------|----------|-----------|------------|---------|--------------|
| Accuracy (%) | \todo{} ± \todo{} | \todo{} ± \todo{} | \todo{} | \todo{} | \todo{} |
| Macro-F1 | \todo{} ± \todo{} | \todo{} ± \todo{} | \todo{} | \todo{} | \todo{} |

### 5.2 Multiple Training Runs

**Required:** For each configuration, run \todo{3-5} times with different random seeds

Report:
- Mean ± standard deviation
- Best run
- Worst run
- 95% confidence interval

---

## 6. Hardware Specifications

**Location:** `sections/experimental_setup.tex`

Fill in actual hardware used:

```latex
\paragraph{Hardware Configuration}
\begin{itemize}
    \item \textbf{GPU:} \todo{NVIDIA GPU model, XX GB VRAM}
    \item \textbf{CPU:} \todo{XX cores, XX GHz}
    \item \textbf{RAM:} \todo{XX GB}
    \item \textbf{Storage:} \todo{SSD configuration}
\end{itemize}

\paragraph{Software Stack}
\begin{itemize}
    \item \textbf{Operating System:} \todo{Linux distribution and version}
    \item \textbf{CUDA:} \todo{Version}
    \item \textbf{cuDNN:} \todo{Version}
\end{itemize}
```

---

## 7. Figures to Create

### 7.1 Architecture Diagrams

**Location:** Appendix A

1. **CNN Architectures** - Detailed block diagrams for:
   - ResNet-18 (1D adapted)
   - ResNet-34 (1D adapted)
   - ResNet-50 (1D adapted)
   - EfficientNet-B0 (1D adapted)
   - EfficientNet-B2 (1D adapted)
   - EfficientNet-B4 (1D adapted)

2. **LSTM Architectures**:
   - Vanilla LSTM cell diagram
   - BiLSTM architecture diagram
   - Gating mechanism visualization

3. **Hybrid Architecture**:
   - Overall flow diagram (Signal → CNN → LSTM → Classification)
   - Integration strategy visualization
   - Temporal pooling methods diagram

**Format:** PDF (vector graphics), Width: 6-7 inches

### 7.2 Performance Comparison Figures

**Location:** `sections/results.tex`

1. **Figure: Overall accuracy comparison** (bar chart)
   - X-axis: Approach (CNN, LSTM, Hybrid)
   - Y-axis: Accuracy (%)
   - Error bars showing standard deviation
   - Color-coded bars

2. **Figure: Accuracy vs. parameters scatter plot**
   - X-axis: Number of parameters (log scale)
   - Y-axis: Test accuracy (%)
   - Different markers for CNN/LSTM/Hybrid
   - Annotation for best models

3. **Figure: Radar chart** - Multi-dimensional comparison
   - Dimensions: Accuracy, Speed, Model Size, Robustness, Training Time
   - Three lines: CNN (blue), LSTM (green), Hybrid (red)

**Format:** PDF or PNG (300 DPI), Width: 5-6 inches

### 7.3 Confusion Matrices

**Location:** `sections/results.tex`

Create 11×11 confusion matrices for:
1. Best CNN model
2. Best LSTM model
3. Best Hybrid model

**Format:** Heatmap with:
- Color scale (0% to 100%)
- Cell annotations (percentages)
- Class labels on both axes
- Title indicating the model

**Tool:** Matplotlib/Seaborn in Python
**Export:** PDF, Size: 5×5 inches

### 7.4 Training Curves

**Location:** `sections/results.tex` and Appendix B

For each best model (CNN, LSTM, Hybrid):

1. **Training/Validation Loss Curve**
   - X-axis: Epoch (0-75)
   - Y-axis: Loss (log scale)
   - Two lines: Training (solid), Validation (dashed)

2. **Training/Validation Accuracy Curve**
   - X-axis: Epoch (0-75)
   - Y-axis: Accuracy (%)
   - Two lines: Training (solid), Validation (dashed)

3. **Learning Rate Schedule**
   - X-axis: Epoch (0-75)
   - Y-axis: Learning rate (log scale)
   - Single line showing cosine annealing

**Format:** PDF, Size: 6×4 inches each

### 7.5 Ablation Study Visualizations

**Location:** `sections/results.tex`

1. **CNN Backbone Ablation** (line plot)
   - X-axis: CNN complexity (params or FLOPs)
   - Y-axis: Hybrid accuracy (%)
   - Points labeled with architecture names

2. **Pooling Method Comparison** (bar chart)
   - X-axis: Pooling method (Mean, Max, Last, Attention)
   - Y-axis: Accuracy (%)
   - Horizontal line showing baseline

3. **Attention Weights Visualization** (heatmap)
   - X-axis: Time steps
   - Y-axis: Sample index or fault class
   - Color intensity: Attention weight
   - Multiple subfigures for different fault classes

**Format:** PDF, Size varies (4-6 inches width)

### 7.6 Robustness Analysis

**Location:** `sections/results.tex`

1. **Accuracy vs. Noise Level** (line plot)
   - X-axis: Noise level σ (0, 0.01, 0.05, 0.1, 0.2)
   - Y-axis: Accuracy (%)
   - Three lines: CNN, LSTM, Hybrid
   - Markers at data points

**Format:** PDF, Size: 6×4 inches

### 7.7 Per-Class Performance

**Location:** `sections/results.tex`

1. **F1-Score by Fault Class** (bar chart)
   - X-axis: Fault class (11 classes)
   - Y-axis: F1-score
   - Grouped bars for CNN/LSTM/Hybrid
   - Horizontal line at 0.9 for reference

**Format:** PDF, Size: 7×4 inches

---

## 8. Abstract Completions

**Location:** `sections/abstract.tex`

Fill in specific quantitative results:

```latex
Our CNN-based approach (Milestone 1) employs 15+ architectures
including ResNet and EfficientNet families, achieving
\todo{XX.XX\%} classification accuracy through effective spatial
feature extraction.

The LSTM-based approach (Milestone 2) utilizes both unidirectional
and bidirectional recurrent networks, demonstrating \todo{XX.XX\%}
accuracy by capturing temporal dependencies in sequential data.

Most notably, our hybrid CNN-LSTM framework (Milestone 3) introduces
a configurable architecture allowing arbitrary combinations of CNN
backbones with LSTM types, achieving \todo{XX.XX\%} accuracy while
providing flexibility for application-specific optimization.
```

---

## 9. Results Discussion Paragraphs

**Location:** `sections/results.tex`

Multiple paragraphs marked with \todo{} that require:

### 9.1 Key Findings Summary

Example template:
```
"The hybrid approach achieved XX.XX% accuracy, outperforming pure
CNN (XX.XX%) by X.X percentage points and pure LSTM (XX.XX%) by
X.X points. Statistical testing confirms significance (p<0.05 for
both comparisons)."
```

### 9.2 Architectural Insights

Example template:
```
"ResNet-34 provided optimal accuracy-efficiency balance. Deeper
variants (ResNet-50) showed only marginal gains (XX.XX% vs XX.XX%),
suggesting that excessive depth may not be beneficial for this
dataset size."
```

### 9.3 Comparative Analysis

Example template:
```
"CNN achieved marginally higher accuracy (XX.XX%) compared to LSTM
(XX.XX%), but the difference of X.X percentage points is not
statistically significant (p=X.XXX). However, CNNs demonstrate
significantly faster inference (XX.X ms vs. XX.X ms)..."
```

---

## 10. Conclusion Summary Points

**Location:** `sections/conclusion.tex`

Fill in final summary with actual numbers:

```latex
\begin{enumerate}
    \item \textbf{Overall Performance:} \todo{[Hybrid/CNN/LSTM]
    achieved the highest test accuracy of XX.XX\%, establishing
    new state-of-the-art on this dataset.}

    \item \textbf{CNN Findings:} \todo{[ResNet-XX/EfficientNet-XX]
    provided the best accuracy-efficiency trade-off among CNN
    architectures with XX.XX\% accuracy and XX.XM parameters.}

    \item \textbf{LSTM Findings:} \todo{BiLSTM outperformed vanilla
    LSTM by X.X percentage points, justifying bidirectional
    processing for offline analysis.}

    \item \textbf{Hybrid Advantages:} \todo{The hybrid approach
    improved accuracy by X.X points over pure CNN and X.X points
    over pure LSTM...}

    \item \textbf{Ablation Insights:} \todo{CNN backbone choice
    proved most critical (X.X\% accuracy range), followed by LSTM
    type (X.X\% difference)...}

    \item \textbf{Robustness:} \todo{All approaches maintained
    >XX\% accuracy under moderate noise (σ=0.05)...}

    \item \textbf{Efficiency:} \todo{CNN offered fastest inference
    (XX.X ms), while hybrid sacrificed speed for accuracy...}
\end{enumerate}
```

---

## 11. Methodology Details

**Location:** `sections/methodology_hybrid.tex`

Fill in model specifications:

```latex
\paragraph{Configuration 1: Best Accuracy}

\begin{itemize}
    \item \textbf{Parameters:} \todo{$\sim$XX.X M}
    \item \textbf{Model Size:} \todo{$\sim$XXX MB}
    \item \textbf{Expected Accuracy:} \todo{XX.XX\%}
\end{itemize}
```

Similar for Configurations 2 and 3.

---

## 12. Experimental Timeline

**Location:** `sections/experimental_setup.tex`

Fill in actual compute time:

```latex
The complete experimental program comprises:

\begin{itemize}
    \item \textbf{Milestone 1 (CNN):} \todo{XX} architectures
    $\times$ \todo{3-5} runs $\times$ \todo{XX} hours/run
    $\approx$ \todo{XX} hours total

    \item \textbf{Milestone 2 (LSTM):} \todo{XX} configurations
    $\times$ \todo{3-5} runs $\times$ \todo{XX} hours/run
    $\approx$ \todo{XX} hours total

    \item \textbf{Milestone 3 (Hybrid):} \todo{XX} configurations
    $\times$ \todo{3-5} runs $\times$ \todo{XX} hours/run
    $\approx$ \todo{XX} hours total

    \item \textbf{Total Compute Time:} \todo{XXX-XXX} GPU hours
\end{itemize}

All experiments were conducted \todo{over a period of XX weeks}.
```

---

## 13. Checklist for Completion

### Data Collection Phase

- [ ] Train all CNN architectures (15+)
- [ ] Train all LSTM configurations (4+)
- [ ] Train all hybrid configurations (3 recommended + custom)
- [ ] Run ablation studies (CNN backbone, LSTM type, pooling, freezing)
- [ ] Conduct robustness tests (noise levels)
- [ ] Perform multiple runs with different seeds (3-5 per config)
- [ ] Collect all training logs and metrics
- [ ] Measure inference times on GPU and CPU
- [ ] Count model parameters and sizes

### Figure Creation Phase

- [ ] Generate all confusion matrices (3×11×11)
- [ ] Create training/validation curves (6+ plots)
- [ ] Design architecture diagrams (10+ diagrams)
- [ ] Make comparison charts (bar, scatter, line, radar)
- [ ] Visualize ablation study results (4+ plots)
- [ ] Create robustness analysis plots (1+)
- [ ] Generate per-class performance charts (1+)
- [ ] Export all figures in proper format (PDF/PNG 300DPI)

### LaTeX Update Phase

- [ ] Replace all \todo{XX.XX} with actual accuracies
- [ ] Fill in all parameter counts
- [ ] Update all inference times
- [ ] Complete all table data
- [ ] Add all figure files to figures/
- [ ] Replace \figplaceholder{} with actual figure references
- [ ] Update hardware specifications
- [ ] Fill in experimental timeline
- [ ] Complete abstract quantitative results
- [ ] Write all discussion paragraphs
- [ ] Update conclusion summary points

### Verification Phase

- [ ] Compile LaTeX successfully (no errors)
- [ ] Check page count (40-50 pages)
- [ ] Verify all figures appear correctly
- [ ] Check all cross-references work
- [ ] Ensure bibliography is complete
- [ ] Search for remaining \todo{} markers (should be 0)
- [ ] Proofread for consistency
- [ ] Check table formatting
- [ ] Verify citation formatting
- [ ] Review figure captions and labels

### Final Quality Check

- [ ] All experimental results present
- [ ] No placeholder text remaining
- [ ] Figures are high quality
- [ ] Tables are properly formatted
- [ ] Statistical analysis complete
- [ ] Conclusions supported by results
- [ ] Writing is clear and professional
- [ ] No grammatical errors
- [ ] Consistent notation throughout
- [ ] PDF compiles correctly

---

## 14. Tools and Scripts for Automation

### 14.1 Counting Placeholders

```bash
# Count total \todo markers
grep -r "\\\\todo" report/sections/ | wc -l

# List all \todo markers with context
grep -r "\\\\todo" report/sections/ -n

# Find \todo markers in specific section
grep "\\\\todo" report/sections/results.tex -n
```

### 14.2 Generating Figures from Results

**Python script template:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
results = {
    'CNN': {'acc': 96.73, 'params': 21.0, 'time': 15.2},
    'LSTM': {'acc': 94.21, 'params': 0.4, 'time': 32.1},
    'Hybrid': {'acc': 97.85, 'params': 21.4, 'time': 45.3}
}

# Create comparison bar chart
fig, ax = plt.subplots(figsize=(6, 4))
models = list(results.keys())
accs = [results[m]['acc'] for m in models]
ax.bar(models, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('Overall Performance Comparison')
ax.set_ylim([90, 100])
plt.tight_layout()
plt.savefig('report/figures/overall_comparison.pdf', dpi=300)
plt.close()

# Similar for other plots...
```

### 14.3 Batch Figure Export

```bash
# Convert all .png figures to .pdf
for f in report/figures/*.png; do
    convert "$f" "${f%.png}.pdf"
done

# Optimize PDF sizes
for f in report/figures/*.pdf; do
    gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
       -dPDFSETTINGS=/prepress -dNOPAUSE -dQUIET \
       -dBATCH -sOutputFile="${f%.pdf}_opt.pdf" "$f"
done
```

---

## 15. Timeline Estimate

**Assuming experiments are complete:**

| Phase | Estimated Time |
|-------|----------------|
| Collect all metrics from logs | 2-4 hours |
| Generate all figures (Python) | 4-6 hours |
| Create architecture diagrams | 4-8 hours |
| Update all LaTeX placeholders | 3-5 hours |
| Compile and verify | 1-2 hours |
| Proofread and polish | 2-3 hours |
| **Total** | **16-28 hours** |

---

## 16. Priority Order

If time is limited, fill in this order:

1. **High Priority - Core Results:**
   - Overall accuracy comparison (Table + Figure)
   - Best model from each approach
   - Confusion matrices
   - Abstract quantitative results

2. **Medium Priority - Detailed Analysis:**
   - Per-class performance
   - CNN architecture comparison
   - LSTM configuration results
   - Hybrid configurations

3. **Lower Priority - Supplementary:**
   - Ablation studies
   - Robustness analysis
   - Training curves
   - Computational metrics

4. **Nice to Have:**
   - Architecture diagrams in appendix
   - Additional ablations
   - Extended discussion paragraphs

---

## Contact and Questions

If you encounter issues while filling in placeholders:

1. Check this document for guidance
2. Review the LaTeX comments in section files
3. Ensure data format matches table structure
4. Verify figure paths and filenames
5. Test compile frequently to catch errors early

---

**Document Version:** 1.0
**Last Updated:** [Current date]
**Status:** Complete placeholder documentation
**Total Placeholders:** ~150+ to be filled
