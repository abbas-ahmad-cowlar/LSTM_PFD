# PHASE 11C: ADVANCED ANALYTICS & XAI INTEGRATION

**Duration:** 2 weeks  
**Objective:** Integrate explainable AI capabilities (from Phase 7), add advanced statistical analysis, hyperparameter optimization, and multi-signal comparison tools. Transform dashboard from training tool to complete ML analysis platform.

---

## 11C.1 PRE-DEVELOPMENT DECISIONS

### Decision 1: XAI Integration Strategy

**Challenge:** Phase 7 has multiple XAI methods (SHAP, LIME, Integrated Gradients, CAV). Dashboard needs unified interface.

**Solution: Explanation Manager Architecture**

```
USER REQUEST:
"Explain why model predicted 'Oil Whirl' for Signal #234"
  ↓
EXPLANATION MANAGER:
  ├─ Check cache: Has this signal been explained before?
  │    Yes → Return cached explanation
  │    No  → Continue
  ├─ Determine model type (CNN, Transformer, etc.)
  ├─ Select appropriate XAI method(s):
  │    CNN → SHAP + Grad-CAM
  │    Transformer → Attention weights + Integrated Gradients
  │    Classical ML → SHAP + Feature importance
  ├─ Call Phase 7 explainability modules
  ├─ Format results for visualization
  ├─ Cache explanation (TTL: 1 hour)
  └─ Return formatted explanation

DISPLAY:
  ├─ Attribution map (overlay on signal)
  ├─ Feature importance ranking
  ├─ Textual explanation ("Model focused on high-frequency burst at 2.3s")
  └─ Confidence calibration ("Model is 87% confident, typically 92% accurate at this confidence level")
```

**Key Principles:**

1. **Lazy Computation:** Don't compute explanations until user requests them (expensive operations)
2. **Progressive Disclosure:** Show summary first, detailed analysis on demand
3. **Method Selection:** Automatically choose best XAI method for model type
4. **Caching:** Explanation for Signal #234 with Model #47 never changes → cache aggressively

---

### Decision 2: Hyperparameter Optimization (HPO) Integration

**Challenge:** HPO can run 50-200 experiments. Dashboard must support high-volume experiment management.

**Solution: HPO Campaign Architecture**

```
HPO Campaign Structure:

Campaign (Parent):
  ├─ ID: campaign_123
  ├─ Name: "ResNet Learning Rate Search"
  ├─ Method: Grid Search / Random Search / Bayesian Optimization
  ├─ Search Space: {lr: [1e-5, 1e-3], dropout: [0.1, 0.5]}
  ├─ Budget: 50 experiments
  ├─ Status: running (34/50 complete)
  └─ Best Result: Exp #347 (97.2% accuracy)

Child Experiments:
  ├─ Experiment #347: lr=3e-4, dropout=0.3 → 97.2% ✅ (best)
  ├─ Experiment #348: lr=1e-3, dropout=0.2 → 96.8%
  ├─ Experiment #349: lr=5e-5, dropout=0.4 → 96.1%
  └─ ... (50 total)

Campaign Page:
  ├─ Progress: 34/50 complete (68%)
  ├─ Time: 8h 23m elapsed, 4h 12m remaining
  ├─ Best So Far: 97.2% (Exp #347)
  ├─ Visualization: Parallel coordinates plot (hyperparams vs. accuracy)
  ├─ Actions: Pause Campaign, Stop Early, View Best Model
```

**HPO Methods Supported:**

1. **Grid Search:** Exhaustive (all combinations)
2. **Random Search:** Sample N random configs
3. **Bayesian Optimization:** Use Optuna library (smart sampling)
4. **Hyperband:** Early stopping for bad runs (saves compute)

**Integration Point:** Reuse Phase 11B training infrastructure (each HPO trial = 1 Celery task)

---

### Decision 3: Statistical Analysis Framework

**Challenge:** Users ask "Is Model A significantly better than Model B?"

**Solution: Statistical Testing Suite**

**Tests Implemented:**

1. **McNemar's Test** (Paired, Binary)
   - Use Case: Compare two models on same test set
   - Null Hypothesis: Models have same error rate
   - Output: p-value, conclusion ("Model A is significantly better, p=0.003")

2. **5x2 Cross-Validation** (More Robust)
   - Use Case: Compare models with statistical rigor
   - Method: 5 iterations of 2-fold CV
   - Output: t-statistic, p-value, confidence interval

3. **Bootstrapping** (Non-Parametric)
   - Use Case: Estimate confidence interval for accuracy
   - Method: Resample test set 1000 times
   - Output: 95% CI (e.g., "Accuracy: 96.8% ± 1.2%")

4. **Friedman Test** (Multiple Models)
   - Use Case: Compare 3+ models
   - Output: Ranking, post-hoc pairwise comparisons

**Display Strategy:**
```
Comparison Page:

Model A (ResNet-34):  96.8% ± 1.1%  (Bootstrap 95% CI)
Model B (Transformer): 96.5% ± 1.3%

Statistical Test (McNemar):
  ├─ Test Statistic: χ² = 2.34
  ├─ p-value: 0.126
  └─ Conclusion: No significant difference (p > 0.05)
      → Both models perform similarly. Choose based on other factors (speed, interpretability).

Confusion Matrix Diff:
  [Heatmap showing where models disagree]
  Model A better at: Oil Whirl (78 vs. 71 correct)
  Model B better at: Cavitation (82 vs. 79 correct)
```

---

### Decision 4: Multi-Signal Comparison Tool

**Challenge:** Users want to compare multiple signals side-by-side.

**Solution: Comparison Workspace**

**Features:**

1. **Add to Comparison Cart**
   - From Signal Viewer: Click "Add to Comparison" button
   - Cart: Stores up to 10 signals in session
   - Persistent: Saved in dcc.Store (browser session)

2. **Comparison View (Grid Layout)**
   ```
   ┌─────────────┬─────────────┬─────────────┐
   │  Signal 1   │  Signal 2   │  Signal 3   │
   ├─────────────┼─────────────┼─────────────┤
   │ Time plot   │ Time plot   │ Time plot   │
   ├─────────────┼─────────────┼─────────────┤
   │ Freq plot   │ Freq plot   │ Freq plot   │
   ├─────────────┼─────────────┼─────────────┤
   │ Spectrogram │ Spectrogram │ Spectrogram │
   └─────────────┴─────────────┴─────────────┘
   
   Aligned plots (same x/y axes for easy comparison)
   ```

3. **Overlay Mode**
   - All signals on same plot (different colors)
   - Useful for: Comparing severity progression (mild → moderate → severe)

4. **Difference Plot**
   - Signal A - Signal B (shows what's different)
   - Highlight regions with large differences

5. **Feature Comparison Table**
   - Rows: 36 features (RMS, Kurtosis, etc.)
   - Columns: Signal 1, Signal 2, Signal 3, Δ (difference)
   - Color coding: Red (large difference), Green (similar)

**Use Cases:**
- Compare normal vs. faulty signals
- Compare different severity levels
- Compare different fault types with similar signatures
- Validate data augmentation (original vs. augmented)

---

### Decision 5: Model Interpretation Dashboard

**Challenge:** Transformer attention is complex, ResNet filters are numerous. Need systematic exploration tools.

**Solution: Model Introspection Suite**

**Tools:**

1. **Layer-by-Layer Activations**
   - Select: Signal + Model + Layer
   - Display: Activation map for that layer
   - Interaction: Scrub through layers like video timeline

2. **Filter Gallery (CNN/ResNet)**
   - Grid view: All filters in a layer
   - Click filter: Show activations for that filter across dataset
   - Purpose: Identify "what does filter #23 detect?"

3. **Attention Flow (Transformer)**
   - Animated visualization: How attention propagates through layers
   - Slider: Scrub through time steps
   - Heatmap: Which tokens attend to which

4. **Concept Activation Vectors (CAV)**
   - Define concept: "High-frequency bursts" (select 20 example signals)
   - Train CAV: Linear classifier on activations
   - Test CAV: Score any signal on "high-frequency-ness"
   - Interpretation: "Model uses high-frequency bursts for Oil Whirl classification"

5. **Counterfactual Generator**
   - Input: Signal + Current prediction + Desired prediction
   - Output: Minimal changes to flip prediction
   - Example: "Change amplitude at 2.1-2.3s to flip from 'Normal' to 'Imbalance'"

---

## 11C.2 FILE STRUCTURE ADDITIONS (28 new files)

**New directories and files added to Phase 11A+11B structure:**

```
dash_app/
│
├── layouts/                        # ADD 6 new pages
│   ├── xai_explorer.py             # NEW: Explain individual predictions
│   ├── model_interpretation.py     # NEW: Model introspection tools
│   ├── signal_comparison.py        # NEW: Multi-signal comparison
│   ├── hpo_campaign.py             # NEW: HPO campaign management
│   ├── statistical_analysis.py     # NEW: Statistical model comparison
│   └── advanced_analytics.py       # NEW: Aggregate analytics dashboard
│
├── callbacks/                      # ADD 6 callback files
│   ├── xai_callbacks.py            # Explanation generation, caching
│   ├── model_interpretation_callbacks.py  # Layer selection, visualization
│   ├── signal_comparison_callbacks.py     # Comparison cart, grid layout
│   ├── hpo_callbacks.py            # Campaign creation, progress tracking
│   ├── statistical_callbacks.py    # Test execution, result display
│   └── analytics_callbacks.py      # Dashboard updates, filters
│
├── services/                       # ADD 6 services
│   ├── xai_service.py              # Explanation manager
│   ├── interpretation_service.py   # Model introspection
│   ├── comparison_service.py       # Signal comparison logic
│   ├── hpo_service.py              # HPO campaign orchestration
│   ├── statistics_service.py       # Statistical tests
│   └── analytics_service.py        # Aggregate metrics, trends
│
├── integrations/                   # ADD 2 adapters
│   ├── phase7_xai_adapter.py       # Wraps Phase 7 XAI modules
│   └── optuna_adapter.py           # Hyperparameter optimization
│
├── models/                         # ADD 2 database models
│   ├── hpo_campaign.py             # HPO campaign metadata
│   └── explanation.py              # Cached explanations
│
├── tasks/                          # ADD 2 Celery tasks
│   ├── hpo_tasks.py                # HPO trial execution
│   └── explanation_tasks.py        # Async explanation generation
│
├── utils/                          # ADD 3 utility modules
│   ├── statistical_tests.py        # McNemar, Bootstrap, Friedman
│   ├── visualization_templates.py  # Reusable Plotly templates
│   └── feature_diff.py             # Feature comparison logic
│
└── tests/                          # ADD 3 test files
    ├── test_xai_service.py
    ├── test_hpo_service.py
    └── test_statistics_service.py
```

**Total files added:** 28  
**Total files (11A + 11B + 11C):** 90 + 28 = **118 files**

---

## 11C.3 DETAILED PAGE SPECIFICATIONS

### Page 1: XAI Explorer (`layouts/xai_explorer.py`)

**Purpose:** Explain individual model predictions using Phase 7 XAI techniques

**URL:** `/xai/explain` or `/experiment/{experiment_id}/explain/{signal_id}`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 EXPLAINABLE AI - PREDICTION EXPLANATION                 │
├─────────────────────────────────────────────────────────────┤
│  SELECT MODEL & SIGNAL                                       │
│  Model:  [ResNet34_Standard_v2 ▼]                           │
│  Signal: [Signal #234 ▼]  or  [Upload Custom Signal]       │
│          [🎲 Random Signal]                                  │
│                                                             │
│  [Generate Explanation] (takes 5-10 seconds)                │
├─────────────────────────────────────────────────────────────┤
│  PREDICTION SUMMARY                                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Predicted Class:  Oil Whirl                        │    │
│  │ Confidence:       87.3%                            │    │
│  │ True Class:       Oil Whirl ✅ (correct)           │    │
│  │                                                     │    │
│  │ All Probabilities:                                  │    │
│  │   Oil Whirl      ████████████████████  87.3%       │    │
│  │   Cavitation     ███  6.2%                         │    │
│  │   Oil Whip       ██   3.1%                         │    │
│  │   Normal         █    2.8%                         │    │
│  │   ... (7 more)   <1% each                          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  EXPLANATION METHODS (Tabs)                                 │
│  [SHAP] [Grad-CAM] [Attention] [Feature Importance]        │
│                                                             │
│  TAB: SHAP (SHapley Additive exPlanations)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ATTRIBUTION MAP                                     │    │
│  │ [Signal plot with red/blue overlay]                │    │
│  │ Red regions: Increased Oil Whirl prediction        │    │
│  │ Blue regions: Decreased Oil Whirl prediction       │    │
│  │                                                     │    │
│  │ KEY INSIGHTS:                                       │    │
│  │ • Peak at 2.31s strongly indicates Oil Whirl       │    │
│  │ • Sub-synchronous oscillation (0.42× shaft speed)  │    │
│  │ • High RMS in 1.8-2.5s window (SHAP value: +0.34) │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ TOP FEATURES (by SHAP value)                       │    │
│  │ 1. RMS (1.8-2.5s):        +0.34 ████████████████   │    │
│  │ 2. Spectral Peak (860Hz): +0.21 ██████████         │    │
│  │ 3. Kurtosis:              +0.15 ███████            │    │
│  │ 4. Envelope RMS:          +0.12 ██████             │    │
│  │ 5. Crest Factor:          -0.08 ████ (decreases)  │    │
│  │ ... (show top 10)                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  TAB: Grad-CAM (Gradient-weighted Class Activation)        │
│  [Heatmap overlay on spectrogram showing important regions] │
│                                                             │
│  TAB: Attention Weights (Transformer models only)           │
│  [Attention heatmap: which time steps model focused on]    │
│                                                             │
│  TAB: Feature Importance (Classical ML only)                │
│  [Bar chart: feature contributions from Random Forest]     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  CONFIDENCE CALIBRATION                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Model predicted 87.3% confidence                   │    │
│  │ Historically, at 85-90% confidence:                │    │
│  │   • Accuracy: 92.1% (typically correct)            │    │
│  │   • Calibration: Slightly overconfident (-4.8%)    │    │
│  │                                                     │    │
│  │ [Reliability diagram: predicted vs. actual]        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  SIMILAR SIGNALS                                             │
│  Find signals with similar explanations (similar SHAP patterns)│
│  ┌─────────┬─────────┬─────────┬─────────┐                 │
│  │ Sig #187│ Sig #302│ Sig #421│ Sig #518│                 │
│  │ Oil Whirl│Oil Whirl│Oil Whirl│Oil Whirl│                 │
│  │ 91% sim │ 88% sim │ 86% sim │ 84% sim │                 │
│  │ [View]  │ [View]  │ [View]  │ [View]  │                 │
│  └─────────┴─────────┴─────────┴─────────┘                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  ACTIONS                                                     │
│  [Export Explanation (PDF)]  [Add to Report]               │
│  [Compare with Another Signal]  [Save to Favorites]        │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Automatic Method Selection**
   - CNN/ResNet: SHAP + Grad-CAM
   - Transformer: Attention weights + Integrated Gradients
   - Classical ML: SHAP + Feature importance
   - PINN: Physics consistency + Frequency analysis

2. **Textual Summaries**
   - LLM-generated (GPT-4 via API, optional) or template-based
   - Example: "The model classified this as Oil Whirl due to strong sub-synchronous oscillations at 860 Hz (0.42× shaft speed), which is characteristic of oil whirl instability. The high RMS between 1.8-2.5 seconds further confirms this diagnosis."

3. **Cached Explanations**
   - Cache key: `explanation:{model_id}:{signal_id}:{method}`
   - TTL: 1 hour (explanations don't change)
   - Invalidate: When model retrained

4. **Confidence Calibration**
   - Track historical accuracy at each confidence level
   - Display: "At 87% confidence, model is usually correct 92% of the time"
   - Visual: Reliability diagram (calibration curve)

---

### Page 2: Model Interpretation (`layouts/model_interpretation.py`)

**Purpose:** Deep dive into model internals (filters, activations, attention)

**URL:** `/model-interpretation/{experiment_id}`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 MODEL INTERPRETATION                                    │
│  Model: ResNet34_Standard_v2                                │
├─────────────────────────────────────────────────────────────┤
│  [Overview] [Filter Gallery] [Activations] [Attention] [CAV]│
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: OVERVIEW                                               │
│                                                             │
│  MODEL ARCHITECTURE                                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ [Interactive architecture diagram]                  │    │
│  │ Click layer → Show details                          │    │
│  │                                                     │    │
│  │ Input: [1, 102400]                                  │    │
│  │   ↓                                                 │    │
│  │ Conv1d(1→64, k=7):  [64, 51200]                    │    │
│  │   ↓                                                 │    │
│  │ ResBlock1: [64, 25600]   ← Click for details       │    │
│  │   ↓                                                 │    │
│  │ ResBlock2: [128, 12800]                             │    │
│  │   ↓                                                 │    │
│  │ ... (expand to show all layers)                     │    │
│  │   ↓                                                 │    │
│  │ GlobalAvgPool: [512]                                │    │
│  │   ↓                                                 │    │
│  │ FC: [11] (fault classes)                            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  LAYER STATISTICS                                            │
│  ┌────────────┬─────────┬──────────┬───────────┐          │
│  │   Layer    │ Params  │ Act. Mean│ Act. Std  │          │
│  ├────────────┼─────────┼──────────┼───────────┤          │
│  │ Conv1      │ 448     │ 0.023    │ 0.182     │          │
│  │ ResBlock1  │ 147,584 │ 0.041    │ 0.205     │          │
│  │ ResBlock2  │ 525,824 │ 0.038    │ 0.198     │          │
│  │ ... (all layers)                             │          │
│  └────────────┴─────────┴──────────┴───────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: FILTER GALLERY (CNN/ResNet only)                     │
│                                                             │
│  SELECT LAYER: [Conv1 ▼]                                   │
│                                                             │
│  FILTER GRID (64 filters in this layer)                    │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                        │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │                        │
│  │[plot][plot][plot][plot][plot][plot][plot][plot]│        │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                        │
│  │ 9 │10 │11 │12 │13 │14 │15 │16 │                        │
│  │[plot][plot][plot][plot][plot][plot][plot][plot]│        │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                        │
│  ... (show all 64 filters in 8×8 grid)                     │
│                                                             │
│  CLICK FILTER #23 → Opens detailed view:                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Filter #23 (Conv1)                                  │    │
│  │ [Larger plot of filter weights]                     │    │
│  │                                                     │    │
│  │ What does this filter detect?                       │    │
│  │ • Peaks at: Sample indices 3, 11, 19 (periodic)    │    │
│  │ • Pattern: High-frequency oscillation detector      │    │
│  │                                                     │    │
│  │ Top activating signals:                             │    │
│  │ 1. Signal #234 (Oil Whirl):    Activation = 12.3   │    │
│  │ 2. Signal #412 (Cavitation):   Activation = 11.8   │    │
│  │ 3. Signal #187 (Oil Whirl):    Activation = 11.2   │    │
│  │ ... (top 10)                                        │    │
│  │                                                     │    │
│  │ [View Activations Across Dataset]                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: ACTIVATIONS                                            │
│                                                             │
│  SELECT SIGNAL: [Signal #234 ▼]                            │
│  SELECT LAYER:  [ResBlock2 ▼]                               │
│                                                             │
│  ACTIVATION MAP                                              │
│  [Heatmap: channels × time]                                 │
│  [128 channels, 12800 time steps → downsample for display] │
│                                                             │
│  LAYER SCRUBBER                                              │
│  [Timeline slider: scrub through layers]                    │
│  Input → Conv1 → RB1 → RB2 → ... → FC → Output            │
│           ^                                                 │
│     (currently viewing)                                     │
│                                                             │
│  STATISTICS FOR SELECTED LAYER                               │
│  • Mean activation: 0.038                                   │
│  • Std activation:  0.198                                   │
│  • Sparsity: 23.4% (% of activations near zero)            │
│  • Max activation: 2.341 (channel 47, time 8234)           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: ATTENTION (Transformer models only)                   │
│                                                             │
│  SELECT SIGNAL: [Signal #234 ▼]                            │
│  SELECT LAYER:  [Layer 4 ▼]                                │
│  SELECT HEAD:   [Head 3 / 8 ▼]                             │
│                                                             │
│  ATTENTION HEATMAP                                           │
│  [Matrix: query tokens × key tokens]                       │
│  [Show which time steps attend to which]                   │
│                                                             │
│  ATTENTION FLOW ANIMATION                                    │
│  [Play button: animate attention propagation through time]  │
│                                                             │
│  ATTENTION HEAD ANALYSIS                                     │
│  • Head 1: Focuses on local patterns (±5 time steps)       │
│  • Head 2: Long-range dependencies (100+ time steps)       │
│  • Head 3: Periodic patterns (attends every 20 steps)      │
│  • ... (analyze all 8 heads)                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: CONCEPT ACTIVATION VECTORS (CAV)                      │
│                                                             │
│  DEFINE NEW CONCEPT                                          │
│  Concept Name: [High-frequency bursts________]             │
│  Positive Examples: (Select 20+ signals with this concept)  │
│    [Signal selector with preview]                           │
│    Selected: Signal #23, #45, #67, ... (20 total)         │
│  Negative Examples: (Random signals without this concept)   │
│    [Auto-select 100 random signals]                         │
│  Layer to test: [ResBlock3 ▼]                               │
│  [Train CAV] (takes 10 seconds)                             │
│                                                             │
│  TRAINED CAV: "High-frequency bursts"                        │
│  Trained on: Layer ResBlock3                                │
│  Accuracy: 94.2% (CAV can identify concept)                │
│                                                             │
│  TEST CAV ON SIGNAL                                          │
│  Signal: [Signal #234 ▼]                                   │
│  CAV Score: 0.87 (high presence of "high-frequency bursts")│
│                                                             │
│  TCAV (Testing with CAV)                                     │
│  Question: How important is "high-frequency bursts" for    │
│            predicting "Oil Whirl"?                          │
│  Answer: Very important (TCAV score: 0.73)                 │
│    → 73% of Oil Whirl predictions are influenced by this   │
│       concept                                               │
│                                                             │
│  CONCEPT IMPORTANCE RANKING                                  │
│  For "Oil Whirl" classification:                            │
│  1. High-frequency bursts:    TCAV = 0.73 ████████████████ │
│  2. Sub-sync oscillations:    TCAV = 0.61 ████████████     │
│  3. Low damping:              TCAV = 0.45 █████████        │
│  ... (all defined concepts)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Interactive Architecture Diagram**
   - Click layer → show details (params, activations, gradients)
   - Tooltip on hover: layer specs
   - Expandable: Show residual connections, skip connections

2. **Filter Visualization**
   - 1D filters shown as line plots
   - Color-code: Blue (negative weights), Red (positive weights)
   - Click filter → see top activating signals

3. **Activation Scrubber**
   - Timeline slider to scrub through layers
   - Watch how signal representation evolves

4. **CAV Training**
   - User defines concept by selecting examples
   - System trains linear classifier on activations
   - TCAV quantifies concept importance

---

### Page 3: Signal Comparison (`layouts/signal_comparison.py`)

**Purpose:** Side-by-side comparison of multiple signals

**URL:** `/signal-comparison`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  ⚖️ SIGNAL COMPARISON                                       │
├─────────────────────────────────────────────────────────────┤
│  COMPARISON CART (0/10 signals)                             │
│  [Empty - Add signals from Signal Viewer or Data Explorer]  │
│                                                             │
│  Quick Add:                                                 │
│  [Add by ID: ___] [Add Random] [Load Saved Comparison]     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  (After adding 3+ signals)                                  │
│                                                             │
│  COMPARISON CART (3/10 signals)                             │
│  ┌──────┬───────────┬────────────┬────────┐               │
│  │  #   │   Signal  │ Fault Type │ Remove │               │
│  ├──────┼───────────┼────────────┼────────┤               │
│  │  1   │ Sig #234  │ Oil Whirl  │   🗑️   │               │
│  │  2   │ Sig #187  │ Oil Whirl  │   🗑️   │               │
│  │  3   │ Sig #412  │ Cavitation │   🗑️   │               │
│  └──────┴───────────┴────────────┴────────┘               │
│  [Clear All] [Export Cart] [Save for Later]                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  VIEW MODE                                                   │
│  [● Grid] [○ Overlay] [○ Difference]                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  GRID VIEW                                                   │
│  ┌──────────────┬──────────────┬──────────────┐           │
│  │  Signal #234 │  Signal #187 │  Signal #412 │           │
│  │  Oil Whirl   │  Oil Whirl   │  Cavitation  │           │
│  ├──────────────┼──────────────┼──────────────┤           │
│  │ TIME DOMAIN                                  │           │
│  │ [Aligned time plots, same y-axis scale]     │           │
│  ├──────────────┼──────────────┼──────────────┤           │
│  │ FREQUENCY DOMAIN                             │           │
│  │ [Aligned FFT plots]                          │           │
│  ├──────────────┼──────────────┼──────────────┤           │
│  │ SPECTROGRAM                                  │           │
│  │ [Aligned spectrograms]                       │           │
│  └──────────────┴──────────────┴──────────────┘           │
│                                                             │
│  Interaction: Synchronized zoom (zoom on one → all zoom)   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  OVERLAY VIEW                                                │
│  TIME DOMAIN (ALL SIGNALS)                                  │
│  [Single plot with 3 colored lines]                         │
│  — Signal #234 (Oil Whirl)      [Blue]                     │
│  — Signal #187 (Oil Whirl)      [Green]                    │
│  — Signal #412 (Cavitation)     [Red]                      │
│                                                             │
│  FREQUENCY DOMAIN (ALL SIGNALS)                             │
│  [Single plot with 3 colored lines]                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  DIFFERENCE VIEW (Select 2 signals)                          │
│  Signal A: [Signal #234 ▼]                                 │
│  Signal B: [Signal #187 ▼]                                 │
│                                                             │
│  DIFFERENCE PLOT (A - B)                                     │
│  [Plot showing difference at each time point]               │
│  [Shaded regions: large difference (>0.1 amplitude)]       │
│                                                             │
│  DIFFERENCE STATISTICS                                       │
│  • Mean Absolute Difference: 0.034                          │
│  • Max Difference: 0.187 (at t=2.31s)                      │
│  • Correlation: 0.892 (highly similar)                      │
│  • Regions with large diff: 1.8-2.5s, 3.7-4.1s             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  FEATURE COMPARISON TABLE                                    │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────┐ │
│  │   Feature    │  Sig 234 │  Sig 187 │  Sig 412 │  Δ   │ │
│  ├──────────────┼──────────┼──────────┼──────────┼──────┤ │
│  │ RMS          │  0.234   │  0.241   │  0.187   │ 🔴   │ │
│  │ Kurtosis     │  5.23    │  5.31    │  7.82    │ 🔴   │ │
│  │ Skewness     │  0.12    │  0.09    │  -0.23   │ 🟡   │ │
│  │ Peak Value   │  1.23    │  1.29    │  0.98    │ 🟢   │ │
│  │ ... (36 features)                               │ │
│  └──────────────┴──────────┴──────────┴──────────┴──────┘ │
│  Legend: 🔴 Large diff (>20%)  🟡 Medium (10-20%)  🟢 Small (<10%)│
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  MODEL PREDICTIONS                                           │
│  Model: [ResNet34_Standard_v2 ▼]                           │
│  ┌──────────┬──────────────┬─────────────┬──────────┐     │
│  │  Signal  │  Predicted   │ Confidence  │ Correct? │     │
│  ├──────────┼──────────────┼─────────────┼──────────┤     │
│  │  #234    │  Oil Whirl   │   87.3%     │    ✅    │     │
│  │  #187    │  Oil Whirl   │   91.2%     │    ✅    │     │
│  │  #412    │  Oil Whirl   │   68.4%     │    ❌    │     │
│  │          │  (True: Cav) │             │          │     │
│  └──────────┴──────────────┴─────────────┴──────────┘     │
│  Insight: Signal #412 misclassified - shares features with│
│           Oil Whirl (sub-sync oscillation) but has high   │
│           kurtosis typical of Cavitation.                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  EXPORT OPTIONS                                              │
│  [Download All Plots (ZIP)]  [Export Comparison Report (PDF)]│
│  [Save Comparison (Bookmark)] [Share Link]                  │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Synchronized Interaction**
   - Zoom on one plot → all plots zoom
   - Hover on time point → vertical line appears on all plots
   - Click region → highlight across all signals

2. **Smart Difference Highlighting**
   - Automatically detect regions with large differences
   - Shade regions (red = very different, yellow = somewhat different)
   - Summarize: "Signals differ most at 2.1-2.4s (amplitude spike)"

3. **Feature Delta Visualization**
   - Color-code feature differences
   - Sort by largest difference (show most discriminative features first)

4. **Persistent Comparisons**
   - Save comparison cart (stored in database)
   - Sharable link: `/signal-comparison/ABC123`
   - Use case: Share interesting cases with team

---

### Page 4: HPO Campaign Manager (`layouts/hpo_campaign.py`)

**Purpose:** Manage hyperparameter optimization campaigns

**URL:** `/hpo/campaigns` or `/hpo/campaign/{campaign_id}`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 HYPERPARAMETER OPTIMIZATION                             │
├─────────────────────────────────────────────────────────────┤
│  [Active Campaigns] [Completed] [Create New]                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  ACTIVE CAMPAIGNS (2 running)                               │
│                                                             │
│  Campaign: "ResNet LR + Dropout Search"                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Method: Bayesian Optimization (Optuna)             │    │
│  │ Progress: 34/50 trials  ████████████░░░░  68%      │    │
│  │ Time: 8h 23m elapsed, ~4h 12m remaining            │    │
│  │ Best So Far: 97.2% (Trial #27)                     │    │
│  │ Status: Running (2 trials in progress)             │    │
│  │                                                     │    │
│  │ [View Details] [Pause] [Stop Early]                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Campaign: "Transformer Architecture Search"                │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Method: Grid Search                                 │    │
│  │ Progress: 12/64 trials  ████░░░░░░░░░░░  19%       │    │
│  │ Time: 2h 41m elapsed, ~11h 3m remaining            │    │
│  │ Best So Far: 96.5% (Trial #8)                      │    │
│  │ Status: Running (3 trials in progress)             │    │
│  │                                                     │    │
│  │ [View Details] [Pause] [Stop Early]                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  CREATE NEW HPO CAMPAIGN                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Campaign Name: [________________________]          │    │
│  │ Base Model:    [ResNet ▼]                          │    │
│  │ Dataset:       [BearingFaults_1430_v2 ▼]           │    │
│  │                                                     │    │
│  │ OPTIMIZATION METHOD                                 │    │
│  │ [○ Grid Search] [○ Random Search] [● Bayesian]     │    │
│  │                                                     │    │
│  │ SEARCH SPACE                                        │    │
│  │ Learning Rate:                                      │    │
│  │   [○ Fixed] [● Range] [○ Log-uniform]              │    │
│  │   Min: [1e-5] Max: [1e-3]                          │    │
│  │                                                     │    │
│  │ Dropout Rate:                                       │    │
│  │   [● Range: 0.1 to 0.5]                            │    │
│  │                                                     │    │
│  │ Batch Size:                                         │    │
│  │   [● Categorical: 16, 32, 64, 128]                 │    │
│  │                                                     │    │
│  │ [+ Add Parameter]                                   │    │
│  │                                                     │    │
│  │ BUDGET                                              │    │
│  │ Max Trials: [50]  (estimated 15 hours total)       │    │
│  │ Max Duration: [24] hours (stop after this time)    │    │
│  │ Parallel Trials: [2] (based on GPU availability)   │    │
│  │                                                     │    │
│  │ EARLY STOPPING (optional)                           │    │
│  │ [☑] Stop if no improvement for [10] trials         │    │
│  │                                                     │    │
│  │ [Create Campaign]                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  CAMPAIGN DETAILS VIEW (after clicking "View Details")      │
│                                                             │
│  Campaign: "ResNet LR + Dropout Search"                     │
│  Status: Running (34/50 complete)                           │
│                                                             │
│  [Overview] [Trials] [Visualizations] [Best Model]         │
│                                                             │
│  TAB: OVERVIEW                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ PROGRESS                                            │    │
│  │ 34/50 trials complete (68%)                         │    │
│  │ 2 running, 14 pending                               │    │
│  │                                                     │    │
│  │ TIME                                                │    │
│  │ Elapsed: 8h 23m                                     │    │
│  │ Remaining: ~4h 12m (based on avg trial duration)   │    │
│  │                                                     │    │
│  │ BEST RESULT                                         │    │
│  │ Trial #27: 97.2% accuracy                           │    │
│  │ Hyperparameters:                                    │    │
│  │   lr: 3.2e-4, dropout: 0.31, batch_size: 32        │    │
│  │                                                     │    │
│  │ [View Best Model] [Deploy Best Model]              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ OPTIMIZATION HISTORY                                │    │
│  │ [Line plot: best accuracy over trial number]       │    │
│  │ Shows convergence → plateauing at 97.2%            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  TAB: TRIALS                                                 │
│  [Sortable table of all 34 completed trials]                │
│  ┌──────┬──────┬─────────┬─────────┬───────┬────────┐     │
│  │Trial │  LR  │ Dropout │  Batch  │  Acc  │ Status │     │
│  ├──────┼──────┼─────────┼─────────┼───────┼────────┤     │
│  │  27  │3.2e-4│  0.31   │   32    │ 97.2% │   ✅   │     │
│  │  19  │2.1e-4│  0.28   │   64    │ 97.0% │   ✅   │     │
│  │  31  │5.3e-4│  0.35   │   32    │ 96.8% │   ✅   │     │
│  │ ...  (show all 34 trials)                         │     │
│  │  12  │9.2e-4│  0.48   │  128    │ 92.1% │   ❌   │     │
│  │   5  │1.1e-5│  0.15   │   16    │ FAIL  │   ❌   │     │
│  └──────┴──────┴─────────┴─────────┴───────┴────────┘     │
│  Click row → View full experiment details                   │
│                                                             │
│  TAB: VISUALIZATIONS                                         │
│  PARALLEL COORDINATES PLOT                                   │
│  [Interactive plot: each line = 1 trial]                    │
│  Axes: LR | Dropout | Batch Size | Accuracy                │
│  Color: by accuracy (red = low, green = high)              │
│  Interaction: Brush axes to filter trials                   │
│                                                             │
│  HYPERPARAMETER IMPORTANCE                                   │
│  [Bar chart showing which hyperparameters matter most]      │
│  1. Learning Rate:  0.68 ████████████████  (most important)│
│  2. Dropout Rate:   0.31 ████████                          │
│  3. Batch Size:     0.12 ██                                │
│                                                             │
│  2D SLICES (Contour plots)                                  │
│  [Heatmap: LR vs. Dropout, color = accuracy]               │
│  [Shows optimal region: LR 2e-4 to 4e-4, dropout 0.25-0.35]│
│                                                             │
│  TAB: BEST MODEL                                             │
│  [Detailed results for Trial #27]                           │
│  [All visualizations: confusion matrix, ROC, etc.]          │
│  [Actions: Deploy, Add to Ensemble, Download]              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Optuna Integration**
   - Use Optuna for Bayesian optimization
   - Tree-structured Parzen Estimator (TPE) sampler
   - Pruning: Stop bad trials early (save compute)

2. **Parallel Execution**
   - Multiple trials run simultaneously (if GPUs available)
   - Queue management: High-priority campaigns first

3. **Visualization Suite**
   - Parallel coordinates: See all hyperparameters at once
   - Contour plots: 2D slices of search space
   - Importance: Which hyperparameters matter most?

4. **Smart Early Stopping**
   - Stop if no improvement for N trials
   - Stop if budget exceeded (time or trials)
   - Optuna pruning: Stop bad trials at epoch 10 (don't wait for 100)

---

### Page 5: Statistical Analysis (`layouts/statistical_analysis.py`)

**Purpose:** Statistical comparison of models

**URL:** `/statistics/compare`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  📊 STATISTICAL MODEL COMPARISON                            │
├─────────────────────────────────────────────────────────────┤
│  SELECT MODELS TO COMPARE                                    │
│  Model A: [ResNet34_Standard_v2 ▼]                         │
│  Model B: [Transformer_v1 ▼]                                │
│  [+ Add Model] (compare up to 5 models)                     │
│                                                             │
│  Test Set: [Standard test set (215 signals) ▼]             │
│                                                             │
│  [Run Statistical Tests]                                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  RESULTS                                                     │
│                                                             │
│  ACCURACY WITH CONFIDENCE INTERVALS                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Model A (ResNet34):   96.8% ± 1.1% (95% CI)       │    │
│  │ Model B (Transformer): 96.5% ± 1.3% (95% CI)       │    │
│  │                                                     │    │
│  │ [Forest plot showing CIs]                           │    │
│  │   ResNet    |──●──|                                │    │
│  │   Transf      |───●──|                             │    │
│  │            95%  96%  97%  98%                       │    │
│  │                                                     │    │
│  │ Observation: Confidence intervals overlap          │    │
│  │              → No obvious difference                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  McNEMAR'S TEST (Pairwise comparison)                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Contingency Table:                                  │    │
│  │                Model B Correct  Model B Wrong       │    │
│  │ Model A Correct      198            10              │    │
│  │ Model A Wrong          3             4              │    │
│  │                                                     │    │
│  │ Test Statistic: χ² = 2.54                          │    │
│  │ p-value: 0.111                                      │    │
│  │                                                     │    │
│  │ ✅ CONCLUSION:                                      │    │
│  │ No significant difference (p > 0.05)                │    │
│  │ Models perform similarly on this test set.         │    │
│  │                                                     │    │
│  │ INTERPRETATION:                                     │    │
│  │ • Both models are wrong on 4 samples (overlap)     │    │
│  │ • Model A uniquely correct on 10 samples           │    │
│  │ • Model B uniquely correct on 3 samples            │    │
│  │ → Small advantage to Model A, but not significant  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  CONFUSION MATRIX DIFFERENCE                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ [Heatmap: Model A matrix - Model B matrix]         │    │
│  │ Positive (green): Model A better                   │    │
│  │ Negative (red): Model B better                     │    │
│  │                                                     │    │
│  │ Key Differences:                                    │    │
│  │ • Oil Whirl: Model A +7 correct (green cell)       │    │
│  │ • Cavitation: Model B +4 correct (red cell)        │    │
│  │ • Others: Minimal difference                        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  PER-CLASS ANALYSIS                                          │
│  ┌──────────────┬────────┬────────┬────────┬────────┐     │
│  │ Fault Class  │ Model A│ Model B│  Diff  │ Better │     │
│  ├──────────────┼────────┼────────┼────────┼────────┤     │
│  │ Normal       │ 99.2%  │ 98.5%  │ +0.7%  │   A    │     │
│  │ Misalignment │ 96.8%  │ 97.3%  │ -0.5%  │   B    │     │
│  │ Oil Whirl    │ 92.3%  │ 85.4%  │ +6.9%  │   A ✅ │     │
│  │ Cavitation   │ 94.6%  │ 97.7%  │ -3.1%  │   B ✅ │     │
│  │ ... (11 classes)                                 │     │
│  └──────────────┴────────┴────────┴────────┴────────┘     │
│                                                             │
│  RECOMMENDATION                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 🎯 RECOMMENDATION:                                  │    │
│  │                                                     │    │
│  │ No clear winner overall, but models have           │    │
│  │ complementary strengths:                           │    │
│  │ • Use Model A (ResNet) for Oil Whirl detection     │    │
│  │ • Use Model B (Transformer) for Cavitation         │    │
│  │                                                     │    │
│  │ BEST STRATEGY:                                      │    │
│  │ → Create ensemble combining both models            │    │
│  │    Expected improvement: +1-2% overall accuracy    │    │
│  │                                                     │    │
│  │ [Create Ensemble with These Models]                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  COMPARE 3+ MODELS (Friedman Test)                          │
│  (Only shown when 3+ models selected)                       │
│                                                             │
│  FRIEDMAN TEST (Ranking-based)                               │
│  H₀: All models have same performance                       │
│  Test Statistic: χ² = 12.34                                │
│  p-value: 0.002                                             │
│  Conclusion: Significant difference exists (p < 0.05)       │
│                                                             │
│  AVERAGE RANKINGS (1=best, 5=worst)                         │
│  1. Ensemble (Phase 8):  1.2  ⭐ (best)                    │
│  2. ResNet-34:           2.3                                │
│  3. Transformer:         2.8                                │
│  4. CNN:                 3.9                                │
│  5. Random Forest:       4.8                                │
│                                                             │
│  POST-HOC PAIRWISE COMPARISONS (Bonferroni-corrected)       │
│  Ensemble vs. ResNet:      p=0.023  (significant ✅)        │
│  Ensemble vs. Transformer: p=0.012  (significant ✅)        │
│  ResNet vs. Transformer:   p=0.234  (not significant)      │
│  ... (all pairs)                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Automated Test Selection**
   - 2 models → McNemar's test
   - 3+ models → Friedman test + post-hoc
   - Continuous metrics → Paired t-test

2. **Confidence Intervals**
   - Bootstrap resampling (1000 iterations)
   - Display: Mean ± 95% CI
   - Visual: Forest plot

3. **Effect Size**
   - Not just "significant" but "how much better?"
   - Cohen's d for paired comparisons
   - Interpretation: small, medium, large effect

4. **Actionable Recommendations**
   - LLM-generated (template-based as fallback)
   - Example: "Model A is 3% better on Oil Whirl. If Oil Whirl is critical for your application, choose Model A."

---

### Page 6: Advanced Analytics Dashboard (`layouts/advanced_analytics.py`)

**Purpose:** Aggregate analytics, trends, insights

**URL:** `/analytics`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  📈 ADVANCED ANALYTICS                                      │
├─────────────────────────────────────────────────────────────┤
│  [Overview] [Trends] [Fault Analysis] [Model Performance]  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB: OVERVIEW                                               │
│                                                             │
│  KEY METRICS (last 30 days)                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐│
│  │ Experiments │ Best Model  │ Avg Training│ GPU Hours   ││
│  │     47      │  98.3% acc  │   16.2 min  │   12.3h     ││
│  │  +12 (34%)  │  +1.2%      │   -3.1 min  │  +4.2h      ││
│  └─────────────┴─────────────┴─────────────┴─────────────┘│
│                                                             │
│  ACCURACY TREND (last 50 experiments)                        │
│  [Line chart: accuracy over time]                           │
│  Shows steady improvement from 95% → 98%                    │
│                                                             │
│  MODEL TYPE DISTRIBUTION                                     │
│  [Pie chart: % of experiments by model type]                │
│  ResNet: 45%, Transformer: 25%, CNN: 18%, Other: 12%       │
│                                                             │
│  TAB: TRENDS                                                 │
│                                                             │
│  HYPERPARAMETER TRENDS                                       │
│  What hyperparameters lead to best results?                 │
│  [Scatter plots: each hyperparameter vs. accuracy]          │
│  • Learning Rate: Optimal range 2e-4 to 5e-4               │
│  • Dropout: Higher dropout (0.3-0.4) performs better       │
│  • Batch Size: 32 and 64 outperform 16 and 128            │
│                                                             │
│  TRAINING TIME ANALYSIS                                      │
│  [Box plot: training time by model type]                    │
│  Transformer slowest (median 22 min), CNN fastest (9 min)  │
│                                                             │
│  TAB: FAULT ANALYSIS                                         │
│                                                             │
│  DIFFICULT FAULTS (Lowest accuracy across all models)       │
│  1. Oil Whirl:    92.3% avg  (hardest)                     │
│  2. Cavitation:   94.1% avg                                 │
│  3. Mixed Faults: 94.7% avg                                 │
│  ... (easiest)                                              │
│  11. Normal:       99.2% avg  (easiest)                     │
│                                                             │
│  CONFUSION PATTERNS (Aggregated across models)              │
│  [Heatmap: which faults are confused with which]           │
│  Most common error: Oil Whirl ↔ Oil Whip (23 errors)       │
│                                                             │
│  SEVERITY ANALYSIS                                           │
│  [Bar chart: accuracy by severity level]                    │
│  Incipient: 89.2%, Mild: 95.1%, Moderate: 97.3%, Severe: 98.9%│
│  Insight: Early-stage faults are hardest to detect         │
│                                                             │
│  TAB: MODEL PERFORMANCE                                      │
│                                                             │
│  MODEL RANKINGS (All-time)                                   │
│  ┌──────┬──────────────────┬──────────┬────────┬────────┐ │
│  │ Rank │      Model       │ Accuracy │F1-Score│  Date  │ │
│  ├──────┼──────────────────┼──────────┼────────┼────────┤ │
│  │  1   │ Ensemble_v3      │  98.3%   │ 0.981  │ Jun 15 │ │
│  │  2   │ ResNet50_HPO_27  │  97.2%   │ 0.969  │ Jun 14 │ │
│  │  3   │ PINN_v2          │  97.1%   │ 0.968  │ Jun 12 │ │
│  │ ...  (top 20 models)                                 │ │
│  └──────┴──────────────────┴──────────┴────────┴────────┘ │
│                                                             │
│  ENSEMBLE ANALYSIS                                           │
│  Best ensembles: Which model combinations work best?        │
│  • ResNet + Transformer + PINN: 98.3% (current best)       │
│  • ResNet + CNN + RF: 97.8%                                 │
│  • All Phase 1-8 models: 97.5% (diminishing returns)       │
│                                                             │
│  COMPUTE EFFICIENCY                                          │
│  [Scatter: accuracy vs. training time]                      │
│  Pareto frontier: Highlight models that are both fast      │
│  and accurate                                               │
│  Efficient models: CNN (94.2%, 9 min), ResNet-18 (96.2%, 12 min)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Automated Insights**
   - ML-powered: Detect trends, anomalies, patterns
   - Example: "Accuracy has plateaued at 98% - consider new data or ensemble"

2. **Comparative Analytics**
   - Which model type works best?
   - Which hyperparameters matter most?
   - ROI analysis: Accuracy gain vs. compute cost

3. **Actionable Recommendations**
   - "Oil Whirl is your hardest fault (92% accuracy). Consider collecting more Oil Whirl data or using PINN (physics-informed)."
   - "Your last 5 experiments show diminishing returns. Try ensemble instead of single model tuning."

---

## 11C.4 ACCEPTANCE CRITERIA (Phase 11C Complete When)

✅ **XAI Integration Functional**
- All Phase 7 XAI methods accessible through dashboard
- Explanations generated for all model types (CNN, Transformer, etc.)
- SHAP, Grad-CAM, attention maps working
- Cached explanations (sub-second load time)
- Textual summaries generated

✅ **Model Interpretation Tools Working**
- Filter gallery displays all CNN/ResNet filters
- Activation scrubber allows layer-by-layer exploration
- Attention flow visualization (Transformer)
- CAV training and TCAV scoring functional

✅ **Signal Comparison Operational**
- Comparison cart stores up to 10 signals
- Grid, overlay, and difference views working
- Feature comparison table color-coded
- Synchronized zooming across plots
- Persistent comparisons (save/share)

✅ **HPO Campaigns Running**
- Optuna integration successful
- Grid, random, and Bayesian optimization methods working
- Parallel execution of trials (multiple GPUs)
- Visualization suite (parallel coordinates, contour plots)
- Early stopping and pruning functional

✅ **Statistical Analysis Validated**
- McNemar's test, Friedman test implemented
- Bootstrap confidence intervals accurate
- Confusion matrix difference visualization
- Recommendations generated (template or LLM)

✅ **Analytics Dashboard Insightful**
- Trends identified automatically
- Hyperparameter importance calculated
- Fault difficulty ranking correct
- Model rankings updated in real-time

✅ **Performance Targets Met**
- XAI explanation generation: <10 seconds
- HPO campaign creation: <2 seconds
- Statistical test execution: <5 seconds
- Analytics dashboard load: <2 seconds

✅ **Testing Coverage**
- XAI service: >85% coverage
- HPO service: >80% coverage
- Statistics service: 100% coverage (critical calculations)
- Integration tests: All XAI methods tested with Phase 7 code

✅ **Documentation Complete**
- User guide: "Understanding Model Predictions (XAI)"
- User guide: "Hyperparameter Optimization Best Practices"
- Developer guide: "Adding New XAI Methods"
- Video tutorial: "Advanced Analytics Walkthrough"

---

## 11C.5 RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **XAI computation too slow (>30 sec)** | Medium | High | Async task queue, aggressive caching, GPU acceleration |
| **HPO campaigns fill disk with checkpoints** | High | Medium | Delete intermediate checkpoints (keep best only), cleanup policy |
| **Statistical tests give contradictory results** | Low | Medium | Show all test results, explain assumptions, provide interpretation |
| **Optuna dependency version conflicts** | Low | High | Pin versions, integration tests, fallback to random search |
| **CAV training requires many examples** | Medium | Low | Provide templates (pre-trained CAVs), clear user guidance |
| **Analytics insights are obvious/unhelpful** | Medium | Low | Iterate based on user feedback, add LLM-powered insights (Phase 11D) |

---

## 11C.6 PHASE 11C DELIVERABLES SUMMARY

**6 New Pages:**
1. XAI Explorer (explain predictions)
2. Model Interpretation (filters, activations, attention)
3. Signal Comparison (multi-signal side-by-side)
4. HPO Campaign Manager (hyperparameter optimization)
5. Statistical Analysis (model comparison)
6. Advanced Analytics Dashboard (trends, insights)

**Key Integrations:**
- Phase 7 XAI modules (SHAP, LIME, Grad-CAM, CAV)
- Optuna (Bayesian optimization)
- Statistical testing libraries (scipy.stats)

**Services:**
- Explanation manager
- Model introspection
- Signal comparison logic
- HPO orchestration
- Statistical testing
- Analytics aggregation

**Infrastructure:**
- Explanation caching (Redis)
- HPO trial database schema
- Background tasks for expensive computations

---
