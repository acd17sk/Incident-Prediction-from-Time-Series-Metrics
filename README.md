# Incident Prediction from Time-Series Metrics

This project focuses on predicting IT system incidents (anomalies) before they occur by analyzing multivariate time-series data. Given a window of past metrics, the goal is to predict if an incident will happen within a future time horizon.

## 📋 Overview

- **Task**: Binary classification to predict whether an incident will occur within the next $H$ time steps, given the previous $W$ steps of multivariate metrics.
- **Dataset**: Server Machine Dataset (SMD), containing multivariate time series from 28 server machines with point-wise anomaly labels.
- **Hardware Target**: Optimized for CPU-only execution (e.g., MacBook Pro i5).

## 🏗️ Project Structure

- `dataset.py`: Handles data loading from SMD, sliding-window generation, label transformation (point-wise to horizon-based), and oversampling for class imbalance.
- `models.py`: Defines the model architectures:
  - **GRUClassifier**: A GRU-based neural network with LayerNorm and temporal attention.
  - **BaselineClassifier**: A scikit-learn Random Forest wrapper.
- `features.py`: Provides fully vectorized extraction of 14 handcrafted statistical features (mean, skew, trend, etc.) for the baseline model.
- `training.py`: Implements the PyTorch training loop including linear warmup, cosine annealing, and early stopping based on validation F1 score.
- `evaluation.py`: Utilities for threshold sweeping, classification reports, PR/ROC curves, and detection latency analysis.
- `Incident Prediction example.ipynb`: An end-to-end notebook demonstrating data setup, training, and model comparison.

## 🚀 Getting Started

### 1. Requirements
- Python 3.12+
- PyTorch
- Scikit-learn
- NumPy
- Matplotlib

### 2. Setup and Data
The project uses the Server Machine Dataset. You can download it directly within the provided notebook:
```python
# In the notebook:
!git clone [https://github.com/NetManAIOps/OmniAnomaly.git](https://github.com/NetManAIOps/OmniAnomaly.git)
!mv OmniAnomaly/ServerMachineDataset .
```

### 3. Usage
You can run the example notebook to see the full workflow:
1. **Load Data**: Load a specific machine's metrics (e.g., `machine-1-1`).
2. **Preprocessing**: Convert point-wise labels into horizon labels—predicting if an anomaly appears in the interval $[t+1, t+H]$.
3. **Training**: Train the GRU model using `train_model` from `training.py`.
4. **Evaluation**: Compare the GRU model against the Random Forest baseline using F1-score and Precision-Recall curves.

## 🧠 Model Details

### GRU Classifier
- **Input Normalization**: Uses `LayerNorm` per time-step to preserve temporal dynamics across features.
- **Attention**: Employs a `TemporalAttention` layer to compute a weighted average of GRU outputs:
  $$\text{context} = \sum_{i=1}^{W} \alpha_i h_i$$
- **Head**: A small MLP with ReLU activation and Dropout ($p=0.2$) to prevent overfitting.

### Random Forest Baseline
- Extracts **14 features per channel** (Statistical, Trend, Variability, and Tail stats), resulting in a total feature vector length of $N_{features} \times 14$.
- Fast vectorized extraction using NumPy.

## 📊 Evaluation Metrics
Since incident prediction is often highly imbalanced, the project focuses on:
- **F1-Score**: Optimized via threshold sweeping $\arg\max_{\tau} F1(\tau)$.
- **Detection Latency**: Measuring how many steps before an incident the model first fires an alert.
- **Precision-Recall Curves**: To evaluate performance across all operating points.
