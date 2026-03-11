"""
dataset.py – Sliding-window dataset for incident prediction.

Provides:
    - load_smd_machine : load a single machine file from the SMD dataset
    - create_horizon_labels : convert point-wise anomaly labels into
      "incident in the next H steps" binary targets
    - TimeSeriesDataset : a PyTorch Dataset that yields (window, label) pairs
      with optional Gaussian noise augmentation
    - build_datasets : convenience function that returns train / val / test
      DataLoaders with chronological split, configurable train stride,
      and minority-class oversampling via WeightedRandomSampler
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1.  Data loading helpers
# ---------------------------------------------------------------------------

def load_smd_machine(
    train_path: str,
    test_path: str,
    label_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single machine from the Server Machine Dataset (SMD).

    Returns
    -------
    train_data : np.ndarray, shape (T_train, F)
    test_data  : np.ndarray, shape (T_test, F)
    test_labels: np.ndarray, shape (T_test,)   – 0 or 1
    """
    train_data = np.loadtxt(train_path, delimiter=",")
    test_data = np.loadtxt(test_path, delimiter=",")
    test_labels = np.loadtxt(label_path, delimiter=",").astype(np.int64)

    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)

    return train_data, test_data, test_labels


# ---------------------------------------------------------------------------
# 2.  Label engineering
# ---------------------------------------------------------------------------

def create_horizon_labels(labels: np.ndarray, horizon: int) -> np.ndarray:
    """Convert point-wise incident labels into look-ahead horizon labels.

    y_h(t) = max(labels[t : t + horizon])

    The last `horizon` timesteps are dropped.
    """
    T = len(labels)
    horizon_labels = np.zeros(T - horizon, dtype=np.int64)
    for t in range(T - horizon):
        horizon_labels[t] = int(labels[t: t + horizon].max())
    return horizon_labels


# ---------------------------------------------------------------------------
# 3.  Normalisation
# ---------------------------------------------------------------------------

def fit_normaliser(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std on the training split."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_normaliser(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-score normalisation using pre-computed statistics."""
    return (data - mean) / std


# ---------------------------------------------------------------------------
# 4.  PyTorch Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for incident prediction.

    Each sample is a tuple:
        x : Tensor of shape (window_size, n_features)
        y : scalar Tensor  (0 or 1)

    Parameters
    ----------
    augment : bool
        If True, apply small Gaussian noise (training regularisation).
    noise_std : float
        Standard deviation of the Gaussian noise (default 0.05).
    """

    def __init__(
        self,
        data: np.ndarray,
        horizon_labels: np.ndarray,
        window_size: int,
        stride: int = 1,
        augment: bool = False,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.data = data
        self.horizon_labels = horizon_labels
        self.window_size = window_size
        self.augment = augment
        self.noise_std = noise_std

        max_start = min(len(data) - window_size, len(horizon_labels) - 1 - window_size)
        max_start = max(max_start, 0)
        self.indices = list(range(0, max_start + 1, stride))

        # Pre-compute labels for all indices (used by WeightedRandomSampler)
        self._labels = np.array(
            [int(horizon_labels[i + window_size]) for i in self.indices]
        )

    @property
    def labels(self) -> np.ndarray:
        """Array of labels for all samples."""
        return self._labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size

        x = torch.tensor(self.data[start:end], dtype=torch.float32)

        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std

        y = torch.tensor(self.horizon_labels[end], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# 5.  Convenience builder
# ---------------------------------------------------------------------------

def build_datasets(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 50,
    horizon: int = 10,
    train_stride: int = 5,
    eval_stride: int = 1,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict:
    """End-to-end builder: normalise → horizon labels → chronological split → DataLoaders.

    Key design choices:
    - **train_stride > 1** reduces redundancy between overlapping windows.
      With stride=1 and W=50, consecutive windows share 49/50 data points,
      creating massive effective duplication.  stride=5 reduces training
      samples ~5× but each sample is more independent.
    - **eval_stride = 1** for val/test to simulate real-time monitoring.
    - **WeightedRandomSampler** oversamples the minority class so each
      epoch has roughly balanced batches.

    Parameters
    ----------
    data         : np.ndarray (T, F)
    labels       : np.ndarray (T,)   point-wise 0/1
    window_size  : int – lookback W
    horizon      : int – look-ahead H
    train_stride : int – stride for training windows (default 5)
    eval_stride  : int – stride for val/test windows (default 1)
    batch_size   : int
    num_workers  : int
    """
    T = len(data)

    # Chronological split: 60 / 20 / 20
    t1 = int(T * 0.6)
    t2 = int(T * 0.8)

    train_data, val_data, test_data = data[:t1], data[t1:t2], data[t2:]
    train_lbl, val_lbl, test_lbl = labels[:t1], labels[t1:t2], labels[t2:]

    # Normalise using training statistics only
    mean, std = fit_normaliser(train_data)
    train_data = apply_normaliser(train_data, mean, std)
    val_data = apply_normaliser(val_data, mean, std)
    test_data = apply_normaliser(test_data, mean, std)

    # Create horizon labels
    train_hlbl = create_horizon_labels(train_lbl, horizon)
    val_hlbl = create_horizon_labels(val_lbl, horizon)
    test_hlbl = create_horizon_labels(test_lbl, horizon)

    # Datasets: larger stride for train, stride=1 for eval
    train_ds = TimeSeriesDataset(train_data, train_hlbl, window_size,
                                 stride=train_stride, augment=True, noise_std=0.05)
    val_ds = TimeSeriesDataset(val_data, val_hlbl, window_size,
                               stride=eval_stride)
    test_ds = TimeSeriesDataset(test_data, test_hlbl, window_size,
                                stride=eval_stride)

    # Weighted sampler: oversample positive windows for balanced batches.
    train_labels = train_ds.labels
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    if n_pos > 0:
        weight_per_class = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
        sample_weights = np.array([weight_per_class[int(l)] for l in train_labels])
        sample_weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "mean": mean,
        "std": std,
        "horizon": horizon,
        "window_size": window_size,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }