"""
evaluation.py – Evaluation utilities for incident prediction.

Provides:
    - collect_predictions   : run a model on a DataLoader → (y_true, y_prob)
    - classification_report : precision, recall, F1 at a given threshold
    - threshold_sweep       : sweep thresholds and return per-threshold metrics
    - find_best_threshold   : pick the threshold that maximises F1
    - detection_latency     : for each true incident region, how many steps
      before the model first fires an alert?
    - Plotting helpers      : loss curves, PR curve, ROC curve, threshold plot,
      detection latency histogram
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the GRU model on a DataLoader and return ground truth + probabilities.

    Returns
    -------
    y_true : np.ndarray, shape (N,)
    y_prob : np.ndarray, shape (N,)   – probabilities in [0, 1]
    """
    model.eval()
    all_y = []
    all_p = []
    for x, y in loader:
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs)
        all_y.append(y.cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def collect_predictions_baseline(
    baseline_model,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions from the sklearn baseline.

    Parameters
    ----------
    baseline_model : BaselineClassifier
    X : feature matrix
    y : true labels

    Returns
    -------
    y_true, y_prob
    """
    y_prob = baseline_model.predict_proba(X)
    return y, y_prob


# ---------------------------------------------------------------------------
# 2.  Metrics at a single threshold
# ---------------------------------------------------------------------------

def classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute key classification metrics at a given threshold.

    Returns dict with: precision, recall, f1, accuracy, n_pos, n_neg,
    true_pos, false_pos, true_neg, false_neg.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# 3.  Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 200,
) -> Dict[str, np.ndarray]:
    """Sweep classification thresholds and compute metrics at each.

    Returns
    -------
    dict with arrays: thresholds, precisions, recalls, f1s
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    precisions = []
    recalls = []
    f1s = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        "thresholds": thresholds,
        "precisions": np.array(precisions),
        "recalls": np.array(recalls),
        "f1s": np.array(f1s),
    }


def find_best_threshold(sweep: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Return (best_threshold, best_f1) from a threshold sweep."""
    idx = np.argmax(sweep["f1s"])
    return float(sweep["thresholds"][idx]), float(sweep["f1s"][idx])


# ---------------------------------------------------------------------------
# 4.  Detection latency
# ---------------------------------------------------------------------------

def detection_latency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    """Analyse how early the model detects each incident region.

    An "incident region" is a contiguous run of y_true == 1.
    For each region, we find the first timestep where y_pred == 1
    (if any) and compute the offset from the region start.

    Returns
    -------
    dict with:
        "latencies"       : list of int – steps from region start to first alert
                            (0 = instant detection)
        "missed_regions"  : int – number of regions with no alert at all
        "total_regions"   : int
        "mean_latency"    : float (over detected regions)
    """
    # Find contiguous incident regions
    regions = []
    in_region = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_region:
            start = i
            in_region = True
        elif y_true[i] == 0 and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(y_true)))

    latencies = []
    missed = 0
    for s, e in regions:
        preds_in_region = y_pred[s:e]
        alert_indices = np.where(preds_in_region == 1)[0]
        if len(alert_indices) > 0:
            latencies.append(int(alert_indices[0]))
        else:
            missed += 1

    return {
        "latencies": latencies,
        "missed_regions": missed,
        "total_regions": len(regions),
        "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5.  Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(history: Dict[str, list], ax: Optional[plt.Axes] = None):
    """Plot training and validation loss curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
):
    """Plot the Precision-Recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, label=f"{label} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
):
    """Plot the ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_threshold_sweep(sweep: Dict[str, np.ndarray], ax: Optional[plt.Axes] = None):
    """Plot precision, recall, and F1 as a function of threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep["thresholds"], sweep["precisions"], label="Precision")
    ax.plot(sweep["thresholds"], sweep["recalls"], label="Recall")
    ax.plot(sweep["thresholds"], sweep["f1s"], label="F1", linewidth=2)

    best_thr, best_f1 = find_best_threshold(sweep)
    ax.axvline(best_thr, color="red", linestyle="--", alpha=0.5,
               label=f"Best thr={best_thr:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Classification Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_detection_latency(latency_info: Dict, ax: Optional[plt.Axes] = None):
    """Histogram of detection latencies (in timesteps from region start)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    lats = latency_info["latencies"]
    if len(lats) == 0:
        ax.text(0.5, 0.5, "No incidents detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return ax
    ax.hist(lats, bins=max(10, len(set(lats))), edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(lats), color="red", linestyle="--",
               label=f"Mean = {np.mean(lats):.1f} steps")
    ax.set_xlabel("Latency (steps from incident start)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Detection Latency  "
        f"({latency_info['total_regions'] - latency_info['missed_regions']}"
        f"/{latency_info['total_regions']} regions detected)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
