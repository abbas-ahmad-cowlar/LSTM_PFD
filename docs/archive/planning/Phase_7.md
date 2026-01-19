> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
## **PHASE 7: Explainable AI (XAI) & Advanced Interpretability**

### Phase Objective
Implement comprehensive suite of interpretability methods (SHAP, LIME, Integrated Gradients, Concept Activation Vectors) to explain model predictions. Build trust for deployment in safety-critical applications. Create interactive dashboard for operators to understand "why" classifier made a decision.

### Complete File List (10 files)

#### **1. Attribution Methods (4 files)**

**`explainability/shap_explainer.py`**
- **Purpose**: SHAP (SHapley Additive exPlanations) for feature importance
- **Key Functions**:
  - `explain_prediction_deep_shap(model, signal, background_dataset)`:
    ```python
    # DeepSHAP for deep learning models
    explainer = shap.DeepExplainer(model, background_dataset)
    shap_values = explainer.shap_values(signal)
    # shap_values: [n_samples, signal_length] or [n_samples, n_features]
    return shap_values
    ```
  - `explain_prediction_kernel_shap(model, signal)`: Model-agnostic SHAP
  - `plot_shap_waterfall(shap_values, signal, predicted_class)`: Waterfall chart showing contribution
- **Dependencies**: `shap`, `torch`

**`explainability/lime_explainer.py`**
- **Purpose**: LIME (Local Interpretable Model-agnostic Explanations)
- **Key Functions**:
  - `explain_with_lime(model, signal, num_samples=1000)`:
    ```python
    # LIME: Fit local linear model around prediction
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background_features,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=signal_features,
        predict_fn=model.predict_proba
    )
    return explanation
    ```
  - `visualize_lime_explanation(explanation, signal)`: Bar chart of feature importance
- **Dependencies**: `lime`, `sklearn`

**`explainability/integrated_gradients.py`**
- **Purpose**: Integrated Gradients (attribute prediction to input features)
- **Key Functions**:
  - `compute_integrated_gradients(model, signal, baseline, steps=50)`:
    ```python
    # Integrated Gradients: ∫ ∂f/∂x dx from baseline to input
    scaled_inputs = [baseline + (float(i) / steps) * (signal - baseline) for i in range(steps + 1)]
    grads = [torch.autograd.grad(model(x), x)[0] for x in scaled_inputs]
    avg_grads = torch.mean(torch.stack(grads), dim=0)
    integrated_grads = (signal - baseline) * avg_grads
    return integrated_grads
    ```
  - `plot_attribution_map(integrated_grads, signal)`: Overlay attribution on signal
- **Dependencies**: `torch`, `captum`

**`explainability/concept_activation_vectors.py`**
- **Purpose**: CAVs (Concept Activation Vectors) for concept-based explanations
- **Key Functions**:
  - `train_cav(model, concept_examples, random_examples, layer_name)`:
    ```python
    # Extract activations for concept examples (e.g., "high-frequency content")
    concept_acts = model.get_layer_activations(concept_examples, layer_name)
    random_acts = model.get_layer_activations(random_examples, layer_name)
    
    # Train linear classifier to separate concept from random
    cav = train_linear_classifier(concept_acts, random_acts)
    return cav
    ```
  - `compute_tcav_score(model, test_examples, cav, layer_name)`: Quantify concept importance
- **Use Case**: Explain in terms of "high-frequency bursts" rather than raw features
- **Dependencies**: `torch`, `sklearn`

#### **2. Visualization (3 files)**

**`visualization/xai_dashboard.py`**
- **Purpose**: Interactive dashboard for exploring explanations
- **Key Features**:
  - Upload signal → see prediction + confidence
  - Display SHAP values, LIME explanation, Integrated Gradients side-by-side
  - Hover over time region → see local importance
  - Compare explanations across methods
- **Technology**: Streamlit or Dash
- **Dependencies**: `streamlit`, `plotly`, `shap`, `lime`

**`visualization/counterfactual_explanations.py`**
- **Purpose**: "What if" explanations (minimal changes to flip prediction)
- **Key Functions**:
  - `generate_counterfactual(model, signal, target_class)`:
    ```python
    # Find minimal perturbation δ such that model(signal + δ) = target_class
    # Optimization: minimize ||δ||_2 subject to model(signal + δ) = target_class
    delta = torch.zeros_like(signal, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.01)
    
    for step in range(1000):
        pred = model(signal + delta)
        loss = cross_entropy(pred, target_class) + 0.1 * torch.norm(delta)
        loss.backward()
        optimizer.step()
    
    return signal + delta
    ```
  - `visualize_counterfactual(original_signal, counterfactual, changes)`:
    ```python
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(original_signal);  ax1.set_title('Original (Predicted: Misalignment)')
    ax2.plot(counterfactual);  ax2.set_title('Counterfactual (Predicted: Imbalance)')
    ax3.plot(counterfactual - original_signal);  ax3.set_title('Changes Required')
    ```
- **Dependencies**: `torch`, `matplotlib`

**`visualization/saliency_maps.py`**
- **Purpose**: Saliency maps (gradient-based importance)
- **Key Functions**:
  - `compute_saliency(model, signal, target_class)`:
    ```python
    signal.requires_grad = True
    output = model(signal)
    model.zero_grad()
    output[0, target_class].backward()
    saliency = signal.grad.abs()
    return saliency
    ```
  - `smooth_grad(model, signal, target_class, noise_level=0.1, n_samples=50)`: Smooth gradients
  - `plot_saliency_overlay(signal, saliency)`: Heatmap overlay
- **Dependencies**: `torch`, `matplotlib`

#### **3. Model-Agnostic Explanations (2 files)**

**`explainability/partial_dependence.py`**
- **Purpose**: Partial dependence plots (how prediction changes with feature)
- **Key Functions**:
  - `partial_dependence_plot(model, dataset, feature_idx)`:
    ```python
    # Vary feature_idx while keeping others fixed, measure prediction change
    feature_values = np.linspace(feature_min, feature_max, 100)
    predictions = []
    for val in feature_values:
        X_modified = X.copy()
        X_modified[:, feature_idx] = val
        pred = model.predict(X_modified).mean()
        predictions.append(pred)
    
    plt.plot(feature_values, predictions)
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Predicted Probability')
    ```
- **Use Case**: "As Kurtosis increases, probability of Cavitation increases"
- **Dependencies**: `sklearn.inspection`, `matplotlib`

**`explainability/anchors.py`**
- **Purpose**: Anchor explanations (rule-based local explanations)
- **Key Functions**:
  - `find_anchor_rules(model, signal, features)`:
    ```python
    # Find minimal set of feature conditions that guarantee prediction
    # Example: "IF Kurtosis > 5 AND SpectralCentroid > 2000 THEN Cavitation (95% confidence)"
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=fault_names,
        feature_names=feature_names
    )
    explanation = explainer.explain_instance(features, model.predict, threshold=0.95)
    return explanation.names()  # List of rules
    ```
- **Benefit**: Interpretable rules for operators (no ML knowledge required)
- **Dependencies**: `anchor-exp`

#### **4. Trust & Uncertainty (1 file)**

**`explainability/uncertainty_quantification.py`**
- **Purpose**: Quantify model confidence and uncertainty
- **Key Functions**:
  - `monte_carlo_dropout(model, signal, n_samples=50)`:
    ```python
    # Enable dropout during inference, sample predictions
    model.train()  # Enables dropout
    predictions = []
    for _ in range(n_samples):
        pred = model(signal)
        predictions.append(pred.softmax(dim=-1))
    
    # Mean prediction and uncertainty
    mean_pred = torch.stack(predictions).mean(dim=0)
    uncertainty = torch.stack(predictions).std(dim=0)
    return mean_pred, uncertainty
    ```
  - `calibration_plot(model, test_loader)`: Reliability diagram
  - `reject_uncertain_predictions(predictions, uncertainty, threshold)`: Flag low-confidence predictions
- **Use Case**: "Model is 98% confident → proceed. Model is 60% confident → manual inspection required."
- **Dependencies**: `torch`, `sklearn.calibration`

### Architecture Decisions

**1. SHAP vs. LIME vs. Integrated Gradients**
- **Decision**: Implement all three, compare
- **Rationale**:
  - SHAP: Theoretically grounded (Shapley values), slower
  - LIME: Fast, model-agnostic, but unstable
  - Integrated Gradients: Deep learning-specific, fast, stable
- **Recommendation**: Use SHAP for analysis, Integrated Gradients for real-time

**2. Dashboard Technology**
- **Decision**: Streamlit (simpler than Dash)
- **Rationale**: Rapid prototyping, easy deployment
- **Alternative**: Dash (more customizable but complex)

**3. Uncertainty Quantification Method**
- **Decision**: Monte Carlo Dropout (simple, no architecture changes)
- **Alternative**: Ensemble (requires training multiple models)
- **Benefit**: Identify ambiguous cases (mixed faults with low confidence)

### Acceptance Criteria

**Phase 7 Complete When:**

✅ **Attribution methods implemented**
- SHAP, LIME, Integrated Gradients functional
- Produce consistent explanations (high inter-method agreement)

✅ **Interactive dashboard operational**
- Upload signal → instant explanation
- Side-by-side comparison of methods
- Counterfactual generation working

✅ **Explanations validated**
- For known faults, explanations align with domain knowledge
  - Misalignment: High importance on 2X harmonic regions
  - Oil whirl: High importance on sub-synchronous regions
- Quantitative: Explanation faithfulness > 0.8 (using infidelity metric)

✅ **Uncertainty quantification working**
- Monte Carlo Dropout produces calibrated uncertainties
- Calibration plot shows < 5% calibration error
- Can flag ambiguous predictions (mixed faults)

✅ **Documentation complete**
- Tutorial: "Explaining Bearing Fault Predictions with XAI"
- User guide for dashboard

### Estimated Effort

**Total: ~12 days (2.5 weeks) for Phase 7**

---
