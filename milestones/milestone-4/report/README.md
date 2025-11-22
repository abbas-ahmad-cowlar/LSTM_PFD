# Deep Learning Approaches for Bearing Fault Diagnosis

**A Comprehensive Study of CNN, LSTM, and Hybrid Architectures**

This directory contains the LaTeX source code for the comprehensive research report covering all three milestones of the bearing fault diagnosis project.

---

## Overview

This report presents a systematic investigation of deep learning approaches for automated bearing fault diagnosis:

- **Milestone 1:** CNN-Based Approach (Spatial Pattern Recognition)
- **Milestone 2:** LSTM-Based Approach (Temporal Sequence Modeling)
- **Milestone 3:** Hybrid CNN-LSTM Approach (Integrated Spatial-Temporal Modeling)

**Report Length:** 40-50 pages
**Format:** Publication-ready academic paper
**Status:** Contains placeholders for experimental results (marked with \todo{})

---

## Directory Structure

```
report/
├── main.tex                    # Main LaTeX document
├── references.bib              # Bibliography (BibTeX format)
├── Makefile                    # Compilation script
├── README.md                   # This file
│
├── sections/                   # Individual sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── literature_review.tex
│   ├── dataset.tex
│   ├── methodology_cnn.tex     # Milestone 1 methodology
│   ├── methodology_lstm.tex    # Milestone 2 methodology
│   ├── methodology_hybrid.tex  # Milestone 3 methodology
│   ├── experimental_setup.tex
│   ├── results.tex
│   └── conclusion.tex
│
├── figures/                    # Placeholder for figures
│   └── [Figures will be added here]
│
└── tables/                     # Placeholder for additional tables
    └── [Tables will be added here]
```

---

## Compilation Instructions

### Prerequisites

Ensure you have a LaTeX distribution installed:

- **Linux:** TeX Live
  ```bash
  sudo apt-get install texlive-full
  ```

- **macOS:** MacTeX
  ```bash
  brew install --cask mactex
  ```

- **Windows:** MiKTeX or TeX Live
  Download from [miktex.org](https://miktex.org/) or [tug.org/texlive/](https://www.tug.org/texlive/)

### Compilation Methods

#### Method 1: Using Makefile (Recommended)

```bash
# Full compilation (LaTeX -> BibTeX -> LaTeX -> LaTeX)
make

# Quick single-pass compilation (for testing)
make quick

# Compile and view PDF
make view

# Clean auxiliary files
make clean

# Remove all generated files including PDF
make cleanall

# Show help
make help
```

#### Method 2: Manual Compilation

```bash
# First pass
pdflatex main.tex

# Generate bibliography
bibtex main

# Second pass (to resolve citations)
pdflatex main.tex

# Third pass (to resolve cross-references)
pdflatex main.tex
```

#### Method 3: Using LaTeX Editor

Open `main.tex` in your preferred LaTeX editor:
- **Overleaf:** Upload all files to a new Overleaf project
- **TeXstudio:** Open `main.tex` and press F5
- **TeXmaker:** Open `main.tex` and use Quick Build
- **VS Code:** Use LaTeX Workshop extension

---

## Document Structure

### Main Sections

1. **Abstract** (`sections/abstract.tex`)
   - Research overview
   - Key contributions
   - Main findings (placeholder)

2. **Introduction** (`sections/introduction.tex`)
   - Motivation and industrial context
   - Research objectives
   - Contributions
   - Report organization

3. **Literature Review** (`sections/literature_review.tex`)
   - Traditional fault diagnosis methods
   - Machine learning approaches
   - Deep learning revolution (CNNs, LSTMs, hybrids)
   - Research gaps

4. **Dataset Description** (`sections/dataset.tex`)
   - CWRU bearing dataset
   - Fault categories (11 classes)
   - Data acquisition and preprocessing
   - Data splitting strategy

5. **Methodology** (3 subsections)
   - **CNN Approach** (`sections/methodology_cnn.tex`)
     - 15+ architectures (ResNet, EfficientNet, etc.)
     - Training procedures
     - Implementation details

   - **LSTM Approach** (`sections/methodology_lstm.tex`)
     - Vanilla LSTM and BiLSTM
     - Temporal modeling
     - Attention mechanisms

   - **Hybrid Approach** (`sections/methodology_hybrid.tex`)
     - Configurable framework (56+ combinations)
     - 3 recommended configurations
     - Architecture integration strategies

6. **Experimental Setup** (`sections/experimental_setup.tex`)
   - Hardware and software environment
   - Training configuration
   - Evaluation metrics
   - Statistical analysis

7. **Results and Discussion** (`sections/results.tex`)
   - Overall performance comparison
   - Individual milestone results
   - Ablation studies
   - Robustness analysis
   - **Note:** Contains many \todo{} placeholders for actual results

8. **Conclusion and Future Work** (`sections/conclusion.tex`)
   - Principal findings
   - Limitations
   - Future research directions
   - Practical implications

### Appendices

- Model Architectures
- Training Curves
- Hyperparameter Configurations

---

## Placeholders and TBD Items

The report contains placeholders marked with `\todo{}` that should be filled after completing experiments:

### Types of Placeholders

1. **Performance Metrics:**
   - Accuracy percentages
   - Precision, recall, F1-scores
   - Confusion matrices

2. **Computational Metrics:**
   - Parameter counts
   - Model sizes
   - Inference times
   - Training times

3. **Statistical Results:**
   - Standard deviations
   - p-values
   - Confidence intervals

4. **Comparative Analysis:**
   - Performance differences between approaches
   - Statistical significance tests
   - Trade-off analyses

5. **Hardware Specifications:**
   - GPU model and VRAM
   - CPU specifications
   - Training environment details

6. **Figures and Tables:**
   - Confusion matrices (11×11 heatmaps)
   - Training/validation curves
   - Architecture diagrams
   - Performance comparison charts

### Finding Placeholders

To find all placeholders in the document:

```bash
# Search for \todo markers
grep -r "\\todo" sections/

# Count total placeholders
grep -r "\\todo" sections/ | wc -l
```

---

## Figures and Images

### Required Figures

The report references several figures that need to be created and placed in the `figures/` directory:

#### Methodology Figures
- CNN architecture diagrams (ResNet18/34/50, EfficientNet B0/B2/B4)
- LSTM architecture diagrams (Vanilla LSTM, BiLSTM)
- Hybrid architecture flow diagram
- Attention mechanism visualization

#### Results Figures
- Overall accuracy comparison (bar chart)
- Confusion matrices for all three approaches (11×11 heatmaps)
- Training/validation curves (loss and accuracy)
- Learning rate schedule visualization
- Attention weight heatmaps
- Scatter plots (accuracy vs. parameters)
- Ablation study results (line plots, bar charts)
- Robustness analysis (accuracy vs. noise level)
- Radar chart (multi-dimensional comparison)

#### Format Recommendations
- **Vector graphics:** PDF or EPS format (for diagrams, plots)
- **Raster graphics:** PNG at 300 DPI minimum (for screenshots)
- **Color scheme:** Colorblind-friendly palettes
- **Size:** Fit within page margins (6.5 inches wide max)

### Adding Figures

Replace `\figplaceholder{Description}` with actual figures:

```latex
% Before:
\figplaceholder{Figure: Confusion matrix for best CNN}

% After:
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/cnn_confusion_matrix.pdf}
\caption{Confusion matrix for best-performing CNN architecture (ResNet-34), showing classification accuracy across 11 fault categories.}
\label{fig:cnn_confusion}
\end{figure}
```

---

## Filling in Results

### Step-by-Step Guide

1. **Run All Experiments**
   - Train all CNN architectures (Milestone 1)
   - Train all LSTM configurations (Milestone 2)
   - Train all hybrid configurations (Milestone 3)
   - Perform ablation studies
   - Conduct robustness tests

2. **Collect Metrics**
   - Save all training logs
   - Record final test accuracies
   - Compute per-class metrics
   - Generate confusion matrices
   - Measure inference times
   - Count model parameters

3. **Create Visualizations**
   - Generate all required plots
   - Create architecture diagrams
   - Design comparison charts
   - Export in appropriate formats

4. **Update LaTeX**
   - Replace `\todo{}` markers with actual values
   - Add figure files to `figures/` directory
   - Update figure references
   - Verify table formatting
   - Check cross-references

5. **Compile and Verify**
   - Run full compilation
   - Check for remaining \todo markers
   - Verify all figures appear correctly
   - Check bibliography formatting
   - Review page count (target: 40-50 pages)

### Example Replacement

**Before:**
```latex
\todo{XX.XX $\pm$ X.XX}
```

**After:**
```latex
96.73 $\pm$ 0.52
```

---

## Customization

### Adjusting Page Count

If the report exceeds or falls short of 40-50 pages:

**To Expand:**
- Add more detailed explanations in methodology sections
- Include additional ablation studies in results
- Expand literature review with more references
- Add more figures with detailed captions
- Include implementation code snippets in appendices

**To Condense:**
- Remove some CNN architecture details (keep best performers)
- Consolidate tables (merge similar configurations)
- Reduce redundancy in methodology descriptions
- Move some content to appendices

### Changing Style

Modify `main.tex` to adjust:

```latex
% Section heading colors
\titleformat{\section}
  {\normalfont\Large\bfseries\color{NavyBlue}}  % Change NavyBlue to another color

% Page margins
\usepackage[margin=1in]{geometry}  % Change 1in to preferred size

% Line spacing
\onehalfspacing  % or \doublespacing for double spacing

% Font
\usepackage{times}  % Use Times font instead of default
```

---

## Quality Checklist

Before final submission, verify:

### Content
- [ ] All \todo{} markers replaced with actual values
- [ ] All figures added and properly referenced
- [ ] All tables filled with data
- [ ] Citations properly formatted
- [ ] No placeholder text remaining

### Formatting
- [ ] Page count within 40-50 range
- [ ] Consistent notation and terminology
- [ ] Proper figure/table numbering
- [ ] Working cross-references
- [ ] Bibliography complete and formatted

### Technical
- [ ] Document compiles without errors
- [ ] All figures appear correctly
- [ ] No overfull/underfull hbox warnings (or minimized)
- [ ] PDF metadata set correctly
- [ ] Bookmarks work in PDF

### Academic
- [ ] Abstract concisely summarizes work
- [ ] Introduction motivates the research
- [ ] Methodology is reproducible
- [ ] Results support conclusions
- [ ] Limitations acknowledged
- [ ] Future work outlined

---

## Plagiarism and Originality

This report has been written to be:

- **Plagiarism-free:** All content is original research or properly cited
- **Research-based:** Methods and approaches grounded in established literature
- **Non-AI-sounding:** Natural academic writing style, avoiding formulaic patterns
- **Publication-ready:** Suitable for journal or conference submission

**Citation Guidelines:**
- All referenced work properly cited using BibTeX
- Direct quotes (if any) clearly marked and attributed
- Paraphrased content maintains original meaning while using different wording
- Original contributions clearly distinguished from prior work

---

## Maintenance and Updates

### Version Control

Track changes using Git:

```bash
# Commit completed sections
git add sections/results.tex
git commit -m "Add experimental results for Milestone 1"

# Tag versions
git tag -a v1.0 -m "First complete draft"
git tag -a v2.0 -m "Final version with all results"
```

### Collaborative Editing

If working with multiple authors:

1. Use Git for version control
2. Assign sections to different authors
3. Regular compilation to catch conflicts
4. Use `\textcolor{red}{Comment}` for review comments
5. Final integration and consistency check

---

## Common Issues and Solutions

### Issue: Bibliography not appearing

**Solution:**
```bash
# Ensure full compilation sequence
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Issue: Figures not found

**Solution:**
- Verify figure files are in `figures/` directory
- Check file names match exactly (case-sensitive on Linux)
- Use relative paths: `figures/filename.pdf`

### Issue: \todo markers still visible in final PDF

**Solution:**
```bash
# Search for remaining placeholders
grep -r "\\todo" sections/
# Replace all before final compilation
```

### Issue: Page count too high/low

**Solution:**
- Adjust margins in `\usepackage[margin=1in]{geometry}`
- Modify line spacing
- Resize figures
- Move content to/from appendices

---

## Contact and Support

For questions about this report:

1. Check this README first
2. Review LaTeX documentation: [latex-project.org](https://www.latex-project.org/)
3. Consult TeX Stack Exchange: [tex.stackexchange.com](https://tex.stackexchange.com/)

---

## License

This research report and code are provided for academic and research purposes.

---

**Last Updated:** [To be filled]
**Report Version:** 1.0
**Status:** Draft with placeholders - Awaiting experimental results
