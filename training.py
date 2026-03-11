"""
training.py – Training loop for the GRU classifier (v4).

Key design choices:
    - **Plain BCEWithLogitsLoss** — since the WeightedRandomSampler already
      delivers balanced batches, the loss function sees ~50/50 classes.
      Adding focal loss or pos_weight on top would over-bias toward positives
      (a "double penalty" for imbalance).
    - **Linear warmup + cosine annealing** LR schedule.
    - **AdamW** with weight decay for proper L2 regularisation.
    - **Gradient clipping** (max_norm=1.0).
    - **Early stopping on validation F1** (not loss).

All training runs on **CPU** (targeting a 2018 MacBook Pro i5).
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_val_f1(model: nn.Module, loader: DataLoader, threshold: float = 0.5) -> float:
    """Quick F1 on the validation set for early stopping."""
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_p.append(probs)
            all_y.append(y.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = (np.concatenate(all_p) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch with gradient clipping. Returns average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate on a DataLoader. Returns average loss."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    warmup_epochs: int = 5,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train the GRU model with early stopping on validation F1.

    Uses plain BCEWithLogitsLoss because the DataLoader already delivers
    balanced batches via WeightedRandomSampler.  Adding class-aware loss
    on top of balanced sampling would over-bias toward positives.

    Parameters
    ----------
    model         : nn.Module (GRUClassifier).
    train_loader  : training DataLoader (assumed to use balanced sampling).
    val_loader    : validation DataLoader.
    epochs        : maximum number of epochs.
    lr            : peak learning rate (reached after warmup).
    weight_decay  : L2 regularisation strength.
    patience      : early-stopping patience on val F1.
    warmup_epochs : epochs of linear warmup from lr/10 → lr.
    verbose       : whether to print per-epoch logs.

    Returns
    -------
    history : dict with keys "train_loss", "val_loss", "val_f1",
              "lr", "epoch_time".
    """
    # Plain BCE — no class weighting, because the sampler already balances.
    criterion = nn.BCEWithLogitsLoss()
    if verbose:
        print("[info] Using BCEWithLogitsLoss (batches already balanced by sampler)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup → cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_f1": [],
        "lr": [], "epoch_time": [],
    }

    best_val_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_one_epoch(model, val_loader, criterion)
        val_f1 = _compute_val_f1(model, val_loader, threshold=0.5)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)

        scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_f1={val_f1:.4f} | "
                f"lr={current_lr:.2e}  time={elapsed:.1f}s"
            )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[early stopping] No F1 improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"[info] Restored best model (val_f1={best_val_f1:.4f})")

    return history