"""
features.py – Handcrafted feature extraction for the baseline classifier.

Instead of feeding raw windows to a neural network, we compute a fixed-size
feature vector per window.  A simple sklearn classifier (Random Forest or
Logistic Regression) is then trained on these features.

Feature groups (per input feature / channel):
    - Statistical : mean, std, min, max, median, skew, kurtosis
    - Trend       : linear slope, last – first value
    - Variability : IQR, mean absolute deviation, range
    - Tail stats  : 10th / 90th percentile

For F input features and the groups above, the total feature vector length is
    F × 14.

This version is **fully vectorised** over the batch dimension using numpy,
making extraction ~50-100× faster than a per-sample Python loop.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Vectorised feature extraction  (operates on entire batches at once)
# ---------------------------------------------------------------------------

def _safe_skew(x: np.ndarray) -> np.ndarray:
    """Skewness along axis=1, returning 0 for near-constant rows."""
    # x: (N, W)
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)  # avoid division by zero
    z = (x - m) / s
    return np.mean(z ** 3, axis=1)


def _safe_kurtosis(x: np.ndarray) -> np.ndarray:
    """Excess kurtosis along axis=1, returning 0 for near-constant rows."""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    z = (x - m) / s
    return np.mean(z ** 4, axis=1) - 3.0


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """Extract features from a batch of multivariate windows.

    Parameters
    ----------
    windows : np.ndarray, shape (N, W, F)

    Returns
    -------
    features : np.ndarray, shape (N, F * 14)
    """
    N, W, F = windows.shape

    # Pre-compute the time axis for slope calculation
    t = np.arange(W, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    all_features = []

    for f in range(F):
        # channel: (N, W)
        ch = windows[:, :, f]

        mean = ch.mean(axis=1)                          # (N,)
        std = ch.std(axis=1)                             # (N,)
        minimum = ch.min(axis=1)                         # (N,)
        maximum = ch.max(axis=1)                         # (N,)
        median = np.median(ch, axis=1)                   # (N,)
        skew = _safe_skew(ch)                            # (N,)
        kurt = _safe_kurtosis(ch)                        # (N,)

        # Slope via vectorised linear regression: slope = cov(t, x) / var(t)
        ch_centered = ch - ch.mean(axis=1, keepdims=True)  # (N, W)
        slope = (ch_centered * (t - t_mean)).sum(axis=1) / max(t_var, 1e-12)

        delta = ch[:, -1] - ch[:, 0]                    # (N,)

        p10 = np.percentile(ch, 10, axis=1)              # (N,)
        p25 = np.percentile(ch, 25, axis=1)              # (N,)
        p75 = np.percentile(ch, 75, axis=1)              # (N,)
        p90 = np.percentile(ch, 90, axis=1)              # (N,)
        iqr = p75 - p25                                  # (N,)
        mad = np.mean(np.abs(ch - mean[:, None]), axis=1)  # (N,)
        rng = maximum - minimum                          # (N,)

        # Stack: (N, 14)
        ch_feats = np.stack(
            [mean, std, minimum, maximum, median, skew, kurt,
             slope, delta, iqr, mad, rng, p10, p90],
            axis=1,
        )
        all_features.append(ch_feats)

    # Concatenate across channels: (N, F*14)
    return np.concatenate(all_features, axis=1)


# ---------------------------------------------------------------------------
# Extract from a DataLoader
# ---------------------------------------------------------------------------

def extract_features_from_loader(
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over a PyTorch DataLoader and return (X_feat, y) numpy arrays.

    Uses vectorised batch extraction for speed.

    Parameters
    ----------
    loader : DataLoader yielding (x, y) with x of shape (B, W, F).

    Returns
    -------
    X : np.ndarray, shape (N, F*14)
    y : np.ndarray, shape (N,)
    """
    all_x = []
    all_y = []
    for x_batch, y_batch in loader:
        x_np = x_batch.numpy()           # (B, W, F)
        all_x.append(extract_features_batch(x_np))
        all_y.append(y_batch.numpy())

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Single-window extraction (kept for convenience / debugging)
# ---------------------------------------------------------------------------

def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract features from a single window of shape (W, F).

    Returns shape (F * 14,).
    """
    return extract_features_batch(window[np.newaxis])[0]


# ---------------------------------------------------------------------------
# Feature names (useful for inspection / feature importance plots)
# ---------------------------------------------------------------------------

CHANNEL_FEATURE_NAMES = [
    "mean", "std", "min", "max", "median", "skew", "kurtosis",
    "slope", "delta", "iqr", "mad", "range", "p10", "p90",
]


def get_feature_names(n_channels: int) -> list[str]:
    """Return human-readable feature names for all channels."""
    names = []
    for c in range(n_channels):
        for fname in CHANNEL_FEATURE_NAMES:
            names.append(f"ch{c}_{fname}")
    return names