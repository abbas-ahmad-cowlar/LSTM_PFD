# User Guide

Complete guide for operators using the LSTM_PFD fault diagnosis system.

**Version:** 1.0.0
**Target Audience:** Maintenance technicians, operators, engineers

---

## Table of Contents

- [Getting Started](#getting-started)
- [Making Predictions](#making-predictions)
- [Understanding Results](#understanding-results)
- [Web Interface](#web-interface)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)

---

## Getting Started

### What is LSTM_PFD?

LSTM_PFD is an AI-powered predictive fault diagnosis system for hydrodynamic bearings. It analyzes vibration signals to detect 11 different fault types with 98%+ accuracy.

### System Access

**Web Interface:**
- URL: `http://your-server:8000`
- Documentation: `http://your-server:8000/docs`

**No login required** (for demo deployments)
Production deployments may require authentication.

### Quick Start (3 steps)

1. **Collect vibration data** (5 seconds @ 20,480 Hz = 102,400 samples)
2. **Upload to system** via web interface or API
3. **Review prediction** and confidence score

---

## Making Predictions

### Method 1: Web Interface

#### Step 1: Navigate to Prediction Page

Open your web browser and go to:
```
http://your-server:8000
```

#### Step 2: Upload Signal

1. Click **"Choose File"** button
2. Select vibration signal file (.csv, .npy, or .txt)
3. Click **"Upload & Predict"**

**Supported formats:**
- CSV: One column of 102,400 values
- NPY: NumPy array saved with `np.save()`
- TXT: Plain text, one value per line

#### Step 3: View Results

Results appear within 1-2 seconds:
- **Fault Type:** e.g., "Inner Race Fault"
- **Confidence:** e.g., 96.7%
- **Probability Chart:** Distribution across all fault classes

### Method 2: Python Script

```python
import requests
import numpy as np

# Load your vibration signal
signal = np.loadtxt('vibration_data.csv')

# Make prediction
response = requests.post(
    'http://your-server:8000/predict',
    json={
        'signal': signal.tolist(),
        'return_probabilities': True
    }
)

result = response.json()
print(f"Fault Detected: {result['class_name']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

### Method 3: Excel/MATLAB Integration

**Excel:**
1. Save vibration data as CSV
2. Use PowerQuery to call API
3. Display results in spreadsheet

**MATLAB:**
```matlab
% Load signal
signal = load('vibration_data.mat');

% Prepare JSON
json = jsonencode(struct('signal', signal));

% Call API
options = weboptions('ContentType', 'json');
result = webwrite('http://your-server:8000/predict', json, options);

% Display
fprintf('Fault: %s\nConfidence: %.1f%%\n', ...
    result.class_name, result.confidence*100);
```

---

## Understanding Results

### Fault Classes

| Class ID | Fault Type | What It Means |
|----------|-----------|---------------|
| 0 | Normal | No fault detected - bearing is healthy |
| 1 | Ball Fault | Ball bearing element defect |
| 2 | Inner Race Fault | Damage to inner race surface |
| 3 | Outer Race Fault | Damage to outer race surface |
| 4 | Imbalance | Rotor imbalance (mass distribution issue) |
| 5 | Misalignment | Shaft/bearing misalignment |
| 6 | Looseness | Mechanical looseness |
| 7 | Oil Whirl | Lubrication instability |
| 8 | Rub | Contact between rotating and stationary parts |
| 9 | Cracked Shaft | Shaft crack |
| 10 | Combined Fault | Multiple simultaneous faults |

### Confidence Scores

The confidence score indicates how certain the system is about its prediction.

**Interpretation:**

| Confidence | Meaning | Action |
|------------|---------|--------|
| **95-100%** | Very High | Fault diagnosis is reliable |
| **85-95%** | High | Diagnosis likely correct |
| **70-85%** | Moderate | Review carefully, consider re-testing |
| **50-70%** | Low | Uncertain - manual inspection recommended |
| **<50%** | Very Low | System unsure - DO NOT rely on this result |

**Example:**
```
Fault: Inner Race Fault
Confidence: 96.7%
Action: High confidence - schedule maintenance
```

### Probability Distribution

When `return_probabilities=True`, you get the full probability distribution:

```json
{
  "probabilities": {
    "0": 0.001,  // 0.1% Normal
    "1": 0.012,  // 1.2% Ball Fault
    "2": 0.967,  // 96.7% Inner Race Fault ← Predicted
    "3": 0.015,  // 1.5% Outer Race Fault
    ...
  }
}
```

**How to use:**
- Check if probabilities are concentrated (good) vs. spread out (uncertain)
- Look at second-highest probability for alternative diagnosis
- If top 2 probabilities are close (e.g., 45% vs 42%), result is ambiguous

---

## Web Interface

### Dashboard Overview

```
┌─────────────────────────────────────────────┐
│  LSTM_PFD Fault Diagnosis System            │
├─────────────────────────────────────────────┤
│                                             │
│  Upload Vibration Signal                    │
│  ┌─────────────────────┐  [Upload]         │
│  │ Choose File...      │                    │
│  └─────────────────────┘                    │
│                                             │
│  Results:                                   │
│  ╔═══════════════════════════════════╗     │
│  ║ Fault Type: Inner Race Fault      ║     │
│  ║ Confidence: 96.7%                 ║     │
│  ║ Severity: WARNING                 ║     │
│  ║                                   ║     │
│  ║ [View Details] [Download Report]  ║     │
│  ╚═══════════════════════════════════╝     │
│                                             │
│  Probability Chart:                         │
│  ████████████████████ Inner Race (96.7%)   │
│  ███ Outer Race (1.5%)                     │
│  ██ Ball Fault (1.2%)                      │
│  █ Other (0.6%)                            │
│                                             │
└─────────────────────────────────────────────┘
```

### Features

- **Drag & Drop:** Drag signal files directly onto upload area
- **Batch Upload:** Upload multiple signals at once
- **History:** View past predictions
- **Export:** Download results as PDF or CSV
- **Explainability:** View which signal features influenced the prediction

### Interpretation Dashboard

Click **"View Details"** to see:
- **Signal Waveform:** Visual plot of your vibration signal
- **Frequency Spectrum:** FFT showing dominant frequencies
- **Feature Importance:** Which features led to this diagnosis
- **Similar Cases:** Past signals with similar patterns
- **Maintenance Recommendations:** Suggested next steps

---

## Best Practices

### 1. Data Collection

**DO:**
- ✅ Collect 5 seconds of data @ 20,480 Hz
- ✅ Ensure sensor is properly mounted
- ✅ Record during steady-state operation
- ✅ Note operating conditions (speed, load, temperature)

**DON'T:**
- ❌ Collect data during startup/shutdown
- ❌ Use loose or improperly mounted sensors
- ❌ Mix data from different machines
- ❌ Ignore environmental noise

### 2. Signal Quality

**Good Signal Characteristics:**
- Clean waveform without clipping
- SNR > 20 dB
- Sampling rate exactly 20,480 Hz
- No missing samples or dropouts

**Check for:**
- Sensor saturation (clipping)
- Electrical noise (60 Hz hum)
- Aliasing (insufficient sampling rate)
- Dropouts or gaps in data

### 3. Result Interpretation

**When to Trust Results:**
- Confidence > 85%
- Signal quality is good
- Operating conditions are normal
- Result matches other indicators (noise, temperature, etc.)

**When to Be Cautious:**
- Confidence < 70%
- Poor signal quality
- Unusual operating conditions
- Contradicts other diagnostics

### 4. Action Based on Results

| Fault Detected | Confidence | Recommended Action |
|---------------|------------|-------------------|
| Normal | >90% | Continue normal operation |
| Any Fault | >90% | Schedule inspection within 1 week |
| Any Fault | 70-90% | Repeat test, monitor closely |
| Any Fault | <70% | Manual inspection recommended |
| Combined Fault | >80% | Immediate inspection required |

---

## Troubleshooting

### Problem: "Invalid signal length" error

**Cause:** Signal has wrong number of samples (not 102,400)

**Solution:**
```python
# Check signal length
print(len(signal))  # Should be 102,400

# Resample if needed
from scipy.signal import resample
signal_resampled = resample(signal, 102400)
```

### Problem: "Signal contains NaN values" error

**Cause:** Missing or corrupted data points

**Solution:**
```python
# Check for NaN
print(np.isnan(signal).sum())

# Remove NaN (simple interpolation)
signal_clean = pd.Series(signal).interpolate().values
```

### Problem: Low confidence (<70%) on multiple tests

**Possible Causes:**
1. Signal quality issue (noise, saturation)
2. Unusual fault pattern not in training data
3. Sensor placement incorrect
4. Machine operating condition unusual

**Solutions:**
1. Check sensor mounting and signal quality
2. Collect multiple signals and compare
3. Consult vibration analysis expert
4. Perform manual inspection

### Problem: Prediction is "Normal" but machine sounds abnormal

**Possible Causes:**
1. Fault frequency outside system's detection range
2. Acoustic noise vs. vibration issue
3. Non-bearing fault (gearbox, motor, etc.)
4. Signal collection error

**Solutions:**
1. Use complementary diagnostics (thermography, ultrasound)
2. Collect signal from different sensor location
3. Check machine for non-bearing issues
4. Verify signal collection procedure

---

## FAQs

### Q1: What if confidence is low (<70%)?

**A:** Low confidence means the system is uncertain. Recommended actions:
1. Re-collect the signal with better sensor placement
2. Check signal quality (no noise, no clipping)
3. Run multiple tests and compare results
4. Perform manual inspection as a precaution

### Q2: How often should I monitor bearings?

**A:** Recommended monitoring frequency:
- **Critical equipment:** Weekly
- **Normal equipment:** Monthly
- **After abnormal event:** Immediately
- **After maintenance:** Before and after

### Q3: Can I use this system for other bearing types?

**A:** The system is trained for hydrodynamic bearings. For rolling element bearings (ball/roller), predictions may be less accurate. Separate models recommended.

### Q4: What to do when multiple faults are detected (Combined Fault)?

**A:** Combined faults indicate serious issues:
1. Stop equipment if safe shutdown is possible
2. Immediate inspection by qualified technician
3. Do not restart until faults are diagnosed and repaired
4. Document all findings for root cause analysis

### Q5: How to update the model with new data?

**A:** Contact system administrator. Model updates require:
1. Labeled ground truth data (confirmed fault types)
2. Minimum 100 samples per fault class
3. Retraining and validation procedure
4. Testing before deployment

### Q6: What sensor specifications are required?

**A:** Recommended accelerometer specifications:
- **Frequency range:** 0-10 kHz minimum
- **Sensitivity:** 100 mV/g
- **Measurement range:** ±50g minimum
- **Mounting:** Stud-mounted (not magnetic)
- **Sampling rate:** 20,480 Hz

### Q7: Can I run this offline (no internet)?

**A:** Yes! The system can be deployed on local servers or edge devices without internet connectivity. See Deployment Guide for offline setup.

### Q8: What if I get "Model not loaded" error?

**A:** This means the inference server is not properly configured:
1. Check server status: `curl http://your-server:8000/health`
2. Restart the service: `docker restart lstm_pfd`
3. Contact system administrator if problem persists

---

## Maintenance Recommendations

### Based on Fault Type

**Inner/Outer Race Fault:**
- Severity: CRITICAL
- Action: Replace bearing within 1-2 weeks
- Downtime: 4-8 hours typical

**Ball Fault:**
- Severity: HIGH
- Action: Schedule replacement within 1 month
- Downtime: 4-6 hours typical

**Imbalance:**
- Severity: MEDIUM
- Action: Rebalance rotor at next planned maintenance
- Downtime: 2-4 hours typical

**Misalignment:**
- Severity: MEDIUM
- Action: Realign and check coupling at next shutdown
- Downtime: 2-4 hours typical

**Oil Whirl:**
- Severity: HIGH
- Action: Check oil level/pressure, may need bearing replacement
- Downtime: Variable

---

## Getting Help

**Documentation:**
- API Reference: `docs/API_REFERENCE.md`
- Deployment Guide: `docs/DEPLOYMENT_GUIDE.md`

**Support:**
- GitHub Issues: https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues
- Email: support@example.com

**Training:**
- User training videos: `https://youtube.com/playlist/...`
- Online course: `https://training.example.com`

---

**End of User Guide**
