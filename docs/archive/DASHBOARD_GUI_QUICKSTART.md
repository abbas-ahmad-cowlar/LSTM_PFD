# GUI Quick Start Guide - No Code Required!

**Welcome!** This guide will show you how to use the LSTM_PFD system entirely through the graphical dashboard—no command-line experience needed.

> **🎯 Goal**: Train a 98% accurate bearing fault diagnosis model using only the web interface.

---

## Table of Contents

- [What You'll Accomplish](#what-youll-accomplish)
- [Prerequisites](#prerequisites)
- [Step 1: Starting the Dashboard](#step-1-starting-the-dashboard)
- [Step 2: Generate Training Data](#step-2-generate-training-data)
- [Step 3: Create an Experiment](#step-3-create-an-experiment)
- [Step 4: Monitor Training](#step-4-monitor-training)
- [Step 5: View Results](#step-5-view-results)
- [Step 6: Explain Predictions](#step-6-explain-predictions)
- [Next Steps](#next-steps)

---

## What You'll Accomplish

By following this guide, you'll:

1. ✅ **Generate a dataset** of synthetic bearing vibration signals
2. ✅ **Train a deep learning model** to detect bearing faults
3. ✅ **Achieve 95-98% accuracy** on fault classification
4. ✅ **Understand predictions** using explainable AI (XAI)

**Time Required**: ~30 minutes (10 min setup + 20 min training)

**No Coding Required**: Everything is done through the web dashboard

---

## Prerequisites

### Software Setup (One-Time)

Ask your system administrator or follow the [installation guide](QUICKSTART.md#installation) to ensure:

- ✅ PostgreSQL database is running
- ✅ Redis server is running
- ✅ Celery worker is running
- ✅ Python dependencies are installed

**Quick Check**:
```bash
# These should all be running (ask admin if unsure)
systemctl status postgresql   # Database
systemctl status redis         # Message queue
celery -A dash_app.tasks status  # Background worker
```

### First-Time Database Setup

```bash
# Only needed once - run these commands
cd /path/to/LSTM_PFD/dash_app
python database/run_migration.py
```

---

## Step 1: Starting the Dashboard

### Option A: Your Administrator Started It

If your admin already started the dashboard, skip to **Step 2**.

**Dashboard URL**: http://localhost:8050 (or the URL your admin provided)

---

### Option B: Start It Yourself

1. **Open Terminal** (or Command Prompt on Windows)

2. **Navigate to Project**:
   ```bash
   cd /path/to/LSTM_PFD/dash_app
   ```

3. **Start the Dashboard**:
   ```bash
   python app.py
   ```

4. **Look for this message**:
   ```
   Dash is running on http://127.0.0.1:8050/
   ```

5. **Open Your Browser**:
   - Go to: http://localhost:8050
   - You should see the LSTM_PFD dashboard home page

**Keep the terminal open** while using the dashboard!

---

## Step 2: Generate Training Data

We'll create a synthetic dataset of bearing vibration signals to train our model.

### 2.1 Navigate to Data Generation

1. Click **"Generate Data"** in the left sidebar
2. You'll see two tabs: **"Generate Data"** and **"Import MAT Files"**
3. Stay on the **"Generate Data"** tab

---

### 2.2 Configure Your Dataset

Fill in the form:

**1. Dataset Name**:
```
my_first_bearing_dataset
```

**2. Number of Signals per Fault**:
```
50
```
_Why 50? It's fast (~5 minutes) but enough for good accuracy. For production, use 100-500._

**3. Select Fault Types**:

Click these checkboxes:
- ✅ Normal (baseline)
- ✅ Ball Fault
- ✅ Inner Race Fault
- ✅ Outer Race Fault
- ✅ Imbalance

_This gives us 5 fault types to detect._

**4. Severity Levels**:

Select:
- ✅ Mild
- ✅ Moderate

_2 severity levels for each fault type._

**5. Noise Layers** (leave defaults):
- ✅ Sensor Noise
- ✅ Quantization Noise
- ✅ Environmental Noise

**6. Operating Conditions** (leave defaults):
- RPM: 1800
- Load: 50%
- Temperature: 40°C

**7. Data Augmentation**:
- ✅ Enable Augmentation
- Augmentation per signal: 2

**8. Output Format**:
- Select: **HDF5** (fastest for training)

---

### 2.3 Generate the Dataset

1. Click the big blue **"Generate Dataset"** button

2. **You'll see**:
   - Progress bar
   - Status messages (e.g., "Generating Ball Fault signals...")
   - Estimated time remaining

3. **Wait ~5 minutes**

4. **Success!**
   - You'll see: ✅ "Dataset generation completed!"
   - Total signals created: **500** (5 faults × 2 severities × 50 signals)

**What just happened?**
- The system created 500 physics-based vibration signals
- Each signal is 102,400 samples (10.24 seconds @ 10 kHz)
- Signals are saved to `data/datasets/my_first_bearing_dataset.h5`

---

## Step 3: Create an Experiment

Now we'll train a deep learning model on our dataset.

### 3.1 Navigate to New Experiment

1. Click **"New Experiment"** in the left sidebar
2. You'll see the Experiment Wizard

---

### 3.2 Configure the Experiment

**Step 1: Basic Information**

```
Experiment Name: my_first_cnn_model
Description: Testing CNN for bearing fault diagnosis
Tags: beginner, cnn, test (comma-separated)
```

Click **"Next"**

---

**Step 2: Dataset Selection**

1. **Dataset Dropdown**: Select `my_first_bearing_dataset`
2. **Train/Val Split**:
   - Training: 80%
   - Validation: 20%
3. **Random Seed**: 42 (for reproducibility)

Click **"Next"**

---

**Step 3: Model Configuration**

1. **Model Type**: Select **"CNN1D - Multi-Scale CNN"**

   _Why CNN? Fast training (~10 min), good accuracy (95-97%)_

2. **Hyperparameters** (use defaults):
   - Learning Rate: 0.001
   - Batch Size: 32
   - Epochs: 50

3. **Optimizer**: Adam (default)

4. **Scheduler**: ReduceLROnPlateau (default)

Click **"Next"**

---

**Step 4: Training Options**

1. **Early Stopping**:
   - ✅ Enable
   - Patience: 10 epochs
   - Metric: validation_loss

2. **Checkpointing**:
   - ✅ Save Best Model

3. **Logging**:
   - Log Interval: 10 batches

Click **"Next"**

---

**Step 5: Review & Launch**

1. Review your configuration
2. Click **"Launch Experiment"**

**What happens**:
- Experiment is created in the database
- Training starts in the background (Celery worker)
- You're redirected to the Training Monitor

---

## Step 4: Monitor Training

You should now be on the **Training Monitor** page.

### 4.1 What You'll See

**Progress Bar**:
```
Epoch 15/50 ████████░░░░░░░░░░░░ 30%
```

**Live Charts** (auto-refresh every 5 seconds):

1. **Loss Curves**:
   - Blue line: Training loss (should decrease)
   - Orange line: Validation loss (should decrease)

2. **Accuracy Curves**:
   - Blue line: Training accuracy (should increase to ~99%)
   - Orange line: Validation accuracy (should reach ~95-97%)

3. **Learning Rate**:
   - Shows how learning rate decreases over time

**Time Estimates**:
```
Elapsed: 8m 23s
Remaining: ~12m 15s
ETA: 14:35:00
```

---

### 4.2 What to Look For

**Good Training**:
- ✅ Training loss decreases smoothly
- ✅ Validation loss decreases (might fluctuate slightly)
- ✅ Training accuracy > 95%
- ✅ Validation accuracy > 90%

**Problems** (if you see these):
- ❌ Validation loss increases = Overfitting (normal after ~30 epochs)
- ❌ Both losses don't decrease = Learning rate too high/low
- ❌ Accuracy stuck at ~20% = Model not learning (rare)

**Don't worry!** Early stopping will save the best model automatically.

---

### 4.3 Wait for Completion

**Total Time**: ~15-20 minutes

You can:
- ✅ Close the browser (training continues)
- ✅ Monitor other experiments
- ✅ Generate more datasets

**When done**, you'll see:
```
✅ Training completed successfully!
Best Epoch: 38
Best Validation Accuracy: 96.4%
```

---

## Step 5: View Results

### 5.1 Navigate to Results

After training completes:
1. Click **"View Results"** button
2. Or go to **"Experiments"** in sidebar → Find your experiment → Click "Results"

---

### 5.2 Explore the Results

**Performance Metrics**:
```
┌─────────────────────────────────────┐
│ Test Accuracy:     96.2%            │
│ Precision:         96.5%            │
│ Recall:            96.0%            │
│ F1-Score:          96.2%            │
│                                     │
│ Training Time:     18m 32s          │
│ Best Epoch:        38/50            │
│ Model Size:        2.3 MB           │
└─────────────────────────────────────┘
```

**Confusion Matrix**:

Hover over the heatmap to see:
- Diagonal = Correct predictions (should be bright)
- Off-diagonal = Mistakes (should be dark)

**Example**:
```
           Predicted
        N   B   I   O   Im
Actual
  N    50   0   0   0   0   ← Perfect!
  B     0  48   1   1   0   ← 48/50 correct
  I     0   0  49   1   0
  O     0   1   0  49   0
  Im    0   0   0   0  50
```

**Per-Class Metrics**:

```
Class           Precision  Recall  F1-Score
─────────────────────────────────────────────
Normal          100%       100%    100%
Ball Fault       96%        96%     96%
Inner Race       98%        98%     98%
Outer Race       98%        98%     98%
Imbalance       100%       100%    100%
```

**Training History**:

Charts showing loss and accuracy progression over 50 epochs.

---

### 5.3 Actions

**Download Model**:
- Click "Download Model" to get the `.pth` file
- Use for deployment or further analysis

**Compare Experiments**:
- Select other experiments
- Click "Compare" to see side-by-side metrics

---

## Step 6: Explain Predictions

Now let's understand **why** the model made specific predictions using XAI.

### 6.1 Navigate to XAI Dashboard

1. Click **"XAI Dashboard"** in the left sidebar (under "Analysis")
2. You'll see the XAI interface

---

### 6.2 Select Model and Signal

**1. Select Model**:
- Dropdown: Choose `my_first_cnn_model`
- You'll see accuracy shown: (96.2% acc)

**2. Select Signal**:
- Dropdown shows signals from your dataset
- Example: `signal_0001 - Ball Fault (Mild)`

**3. Select Method**:
- Choose **"SHAP"** (recommended for first try)

**4. Configure Parameters**:
- Background Samples: 100 (default)
- Leave other settings as-is

---

### 6.3 Generate Explanation

1. Click **"Generate Explanation"**

2. **First Time**: Wait ~20 seconds
   - Shows "Generating explanation..." with spinner

3. **Second Time**: <1 second (cached!)

---

### 6.4 Interpret the Results

**Prediction Display**:
```
┌─────────────────────────────────┐
│ Predicted Class: Ball Fault     │
│ Confidence: 95.3%                │
│                                  │
│ Top 5 Predictions:               │
│  1. Ball Fault      95.3% ████  │
│  2. Inner Race       2.1% █     │
│  3. Normal           1.5% █     │
│  4. Outer Race       0.8% ▌     │
│  5. Imbalance        0.3% ▌     │
└─────────────────────────────────┘
```

**SHAP Visualization**:

You'll see a plot with:
- **Blue line**: Original vibration signal
- **Green/Red overlay**: SHAP attribution
  - **Green**: Increases Ball Fault probability
  - **Red**: Decreases Ball Fault probability

**What This Means**:

The model identified **periodic impulses** in the signal (shown in green) as the key indicator of a ball defect. This makes sense because ball faults create regular impacts as the ball rotates.

**Waterfall Chart**:

Shows the **top 20 time steps** that most influenced the prediction:

```
Time Step 8234:  +0.042 ████████
Time Step 12441: +0.039 ███████
Time Step 4892:  +0.035 ██████
...
```

---

### 6.5 Try Different Methods

**LIME** (Segment-Based):
1. Change Method to **"LIME"**
2. Click "Generate Explanation"
3. See which **segments** of the signal are important
4. Good for understanding temporal patterns

**Integrated Gradients**:
1. Change Method to **"Integrated Gradients"**
2. Similar to SHAP but different algorithm
3. Often faster

**Grad-CAM** (CNN Only):
1. Only works with CNN models (✅ your model is CNN)
2. Shows what the CNN "sees" in the signal
3. Visualizes internal layer activations

---

## Next Steps

Congratulations! 🎉 You've successfully:
- ✅ Generated a dataset
- ✅ Trained a 96% accurate model
- ✅ Understood predictions with XAI

### What to Do Next?

**1. Improve Accuracy** (Try these experiments):

- **More Data**: Generate 200 signals per fault instead of 50
  - Expected: 97-98% accuracy

- **Better Model**: Try "ResNet" instead of "CNN1D"
  - Expected: 98% accuracy

- **Ensemble**: Train 3-5 models and combine (advanced)
  - Expected: 98-99% accuracy

**2. Import Real Data**:

If you have MATLAB `.mat` files:
1. Go to **Data Generation** → **Import MAT Files** tab
2. Upload your files
3. Train models on real-world data

**3. Compare Multiple Experiments**:

1. Train several models (different architectures, hyperparameters)
2. Go to **Experiments** page
3. Select multiple experiments
4. Click **"Compare Selected"**
5. See which performs best

**4. Advanced Features** (Coming Soon):

- **HPO Campaigns**: Automatically find best hyperparameters
- **Deployment**: Export model to ONNX for production
- **API**: Use REST API for real-time predictions

See [DASHBOARD_GAPS.md](docs/DASHBOARD_GAPS.md) for planned features.

---

## Troubleshooting

### Dashboard Won't Load

**Check**:
```bash
# Is the dashboard running?
ps aux | grep "python app.py"
```

**Restart**:
```bash
cd /path/to/LSTM_PFD/dash_app
python app.py
```

---

### Data Generation Stuck

**Check Celery Worker**:
```bash
celery -A dash_app.tasks status
```

If not running:
```bash
cd /path/to/LSTM_PFD/dash_app
celery -A tasks worker --loglevel=info
```

---

### Training Not Starting

**Symptoms**: Experiment stuck in "Pending" status

**Solution**: Same as above - check Celery worker is running

---

### XAI Generation Fails

**Error**: "SHAP library not installed"

**Solution**:
```bash
pip install shap captum
```

---

### Low Accuracy (<90%)

**Possible Causes**:
1. **Too Few Signals**: Try 100+ per fault instead of 50
2. **Model Too Simple**: Try ResNet instead of CNN1D
3. **Not Enough Epochs**: Increase from 50 to 100
4. **Data Quality**: Check dataset has balanced classes

---

## Tips for Best Results

✅ **Start Small**: 50 signals per fault for testing
✅ **Scale Up**: 200+ signals per fault for production
✅ **Use Defaults**: Hyperparameters are pre-tuned
✅ **Monitor Training**: Watch for overfitting
✅ **Compare Models**: Try CNN, ResNet, Transformer
✅ **Cache XAI**: First generation is slow, then instant
✅ **Read Docs**: Check [USAGE_PHASE_11.md](docs/USAGE_PHASE_11.md) for details

---

## Further Learning

- **Detailed Dashboard Guide**: [USAGE_PHASE_11.md](docs/USAGE_PHASE_11.md)
- **Command-Line Workflow**: [QUICKSTART.md](QUICKSTART.md)
- **Missing Features**: [DASHBOARD_GAPS.md](docs/DASHBOARD_GAPS.md)
- **Phase Guides**: See `/USAGE_GUIDES/` directory

---

**Need Help?**
- Check [Troubleshooting](#troubleshooting) section above
- Review [USAGE_PHASE_11.md](docs/USAGE_PHASE_11.md)
- Examine example experiments in dashboard

**Happy Diagnosing! 🔧⚙️**
