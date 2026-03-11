"""
models.py – Model definitions for incident prediction.

Contains:
    1. GRUClassifier      – GRU with LayerNorm, temporal attention, and a
                            properly-sized classification head.
    2. BaselineClassifier  – scikit-learn Random Forest wrapper.

Design notes (v4)
-----------------
- **LayerNorm** on input instead of BatchNorm1d.  BatchNorm across the
  flattened (B*W, F) tensor destroys temporal variance within each
  sequence.  LayerNorm normalises per time-step across features,
  preserving temporal dynamics — the standard for sequential models.
- **hidden_size=64** (up from 16).  With 38 input features, a 16-dim
  hidden state is a severe information bottleneck.
- **Dropout=0.2** in the head.  With hidden_size=64 and head dim 32,
  dropping 50% leaves only 16 active neurons — too few.  0.2 keeps
  ~26 neurons active, a better capacity/regularisation balance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  GRU-based classifier with attention (PyTorch)
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Soft attention over the time axis of GRU outputs.

    Given GRU outputs of shape (B, W, H), computes a weighted average
    over the W time steps, producing a context vector of shape (B, H).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        scores = self.attn(gru_outputs)               # (B, W, 1)
        weights = torch.softmax(scores, dim=1)         # (B, W, 1)
        context = (gru_outputs * weights).sum(dim=1)   # (B, H)
        return context


class GRUClassifier(nn.Module):
    """GRU classifier with attention pooling for incident prediction.

    Architecture
    ------------
    Input (B, W, F)
      → LayerNorm(F) per time-step
      → GRU (1–2 layers)
      → Temporal attention pooling over all time steps
      → FC(H → H//2) → ReLU → Dropout → FC(H//2 → 1)

    Parameters
    ----------
    input_size  : int – number of input features F.
    hidden_size : int – GRU hidden dimension (default 64).
    num_layers  : int – stacked GRU layers (default 1).
    dropout     : float – dropout in the classification head (default 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # LayerNorm across features per time-step — preserves temporal
        # dynamics unlike BatchNorm which was flattening (B*W, F).
        self.input_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, W, F)

        Returns
        -------
        logits : Tensor of shape (B,)
        """
        # LayerNorm is applied per time-step: input (B, W, F), normalises
        # across the last dim (F) for each (batch, timestep) pair.
        x = self.input_norm(x)                        # (B, W, F)

        outputs, _ = self.gru(x)                      # (B, W, H)
        context = self.attention(outputs)              # (B, H)
        logits = self.head(context).squeeze(-1)        # (B,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# 2.  Baseline classifier (sklearn)
# ---------------------------------------------------------------------------

class BaselineClassifier:
    """Random Forest on handcrafted window features."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_