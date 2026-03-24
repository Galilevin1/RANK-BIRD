"""
Domain-Adversarial Neural Network (DANN) for microbiome representation learning.

Architecture:
    Input → Encoder → z
    z → Phenotype classifier → phenotype prob  (BCE loss, normal gradient)
    z → GRL → Domain discriminator → domain prob  (BCE loss, reversed gradient)

Total loss = phenotype_BCE(pred, y)  [labeled samples only]
           + λ(t) * domain_BCE(domain_pred, domain_label)  [all samples]

λ(t) ramps from 0 → λ_max over training to stabilise early learning.

Domain labels: train datasets = 0, held-out test dataset = 1.
The encoder is forced to produce representations that look identical across
train and test — removing inter-dataset bias while preserving phenotype signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ──────────────────────────────────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam, dtype=torch.float32))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lam)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class DANN(nn.Module):
    """Domain-Adversarial Neural Network for cross-dataset microbiome alignment."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        # ── Encoder (BatchNorm normalises activations across datasets) ────────
        enc_layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Phenotype classifier head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ── Domain discriminator + GRL ────────────────────────────────────────
        self.grl = GradientReversalLayer()
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            y_pred      – phenotype probability (0–1)
            domain_pred – domain probability (0=train, 1=test)
        """
        z = self.encoder(x)
        y_pred      = self.classifier(z)
        domain_pred = self.domain_discriminator(self.grl(z))
        return y_pred, domain_pred


# ──────────────────────────────────────────────────────────────────────────────
# λ schedule
# ──────────────────────────────────────────────────────────────────────────────

def _lambda_schedule(epoch: int, max_epoch: int, lam_max: float = 1.0) -> float:
    """Standard DANN sigmoid ramp: 0 → lam_max over training."""
    p = epoch / max_epoch
    return lam_max * (2 / (1 + np.exp(-10 * p)) - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_supervised_dae(
    microbiome_dfs: List[pd.DataFrame],
    target_series: pd.Series,
    unlabeled_dfs: Optional[List[pd.DataFrame]] = None,
    latent_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    noise_std: float = 0.0,    # unused in DANN, kept for interface compatibility
    dropout: float = 0.2,
    cls_weight: float = 1.0,   # λ_max for domain adversarial loss
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DANN, StandardScaler, Dict]:
    """
    Train a DANN on labeled microbiome DataFrames, with optional unlabeled
    DataFrames (held-out LODO test set) used for domain adaptation.

    Phenotype loss : labeled train data only            (no label leakage)
    Domain loss    : all data  (train=0, test=1)
                     gradient is reversed into the encoder via GRL so the
                     encoder learns domain-invariant representations

    Args:
        microbiome_dfs : labeled training DataFrames
        target_series  : binary labels aligned with microbiome_dfs
        unlabeled_dfs  : held-out test DataFrames — features used for domain
                         adaptation; labels NEVER seen during training
        cls_weight     : λ_max for the domain adversarial term (ramps 0→λ_max)
        noise_std      : unused, kept for API compatibility with old DAE interface
    Returns:
        (trained_model, fitted_scaler, history_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        mode = "transductive DANN" if unlabeled_dfs else "standard DANN"
        print(f"  [DANN] device={device}  mode={mode}  λ_max={cls_weight}")

    # ── Prepare labeled data ──────────────────────────────────────────────────
    X_labeled = pd.concat(microbiome_dfs, axis=0).values.astype(np.float32)
    y_all     = target_series.values.astype(np.float32).ravel()
    assert len(X_labeled) == len(y_all), "Mismatch between features and labels"

    scaler = StandardScaler()
    idx = np.arange(len(X_labeled))
    np.random.seed(42)
    val_size  = max(1, int(len(idx) * val_split))
    val_idx   = np.random.choice(idx, size=val_size, replace=False)
    train_idx = np.setdiff1d(idx, val_idx)

    scaler.fit(X_labeled[train_idx])
    X_labeled_scaled = scaler.transform(X_labeled)

    X_train = torch.tensor(X_labeled_scaled[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx],             dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(X_labeled_scaled[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y_all[val_idx],               dtype=torch.float32).unsqueeze(1)

    # Domain labels: training datasets = 0
    d_train = torch.zeros(len(X_train), 1, dtype=torch.float32)
    d_val   = torch.zeros(len(X_val),   1, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, d_train), batch_size=batch_size, shuffle=True
    )

    # ── Prepare unlabeled (target) data: domain = 1 ──────────────────────────
    unlabeled_loader = None
    X_unlab_val = None
    if unlabeled_dfs is not None:
        X_unlabeled        = pd.concat(unlabeled_dfs, axis=0).values.astype(np.float32)
        X_unlabeled_scaled = scaler.transform(X_unlabeled)
        X_unlab_t          = torch.tensor(X_unlabeled_scaled, dtype=torch.float32)
        d_unlab            = torch.ones(len(X_unlab_t), 1, dtype=torch.float32)
        unlabeled_loader   = DataLoader(
            TensorDataset(X_unlab_t, d_unlab), batch_size=batch_size, shuffle=True
        )
        # fixed slice for validation domain loss
        X_unlab_val = X_unlab_t[:min(len(X_val), len(X_unlab_t))]

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim = X_labeled.shape[1]
    model = DANN(
        input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    bce_loss = nn.BCELoss()

    best_val_cls  = float("inf")
    best_state    = None
    no_improve    = 0

    history = {
        "train_cls": [], "train_dom": [], "train_total": [],
        "val_cls":   [], "val_dom":   [], "val_total":   [],
        "lambda":    [],
    }

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)
    d_val_dev = d_val.to(device)

    for epoch in range(1, epochs + 1):
        lam = _lambda_schedule(epoch, epochs, lam_max=cls_weight)
        model.grl.lam = lam
        model.train()

        epoch_cls = 0.0
        epoch_dom = 0.0

        # ── Labeled pass: phenotype loss + domain loss ────────────────────────
        for X_b, y_b, d_b in train_loader:
            X_b, y_b, d_b = X_b.to(device), y_b.to(device), d_b.to(device)
            optimizer.zero_grad()
            y_pred, d_pred = model(X_b)
            loss_cls = bce_loss(y_pred, y_b)
            loss_dom = bce_loss(d_pred, d_b)
            (loss_cls + loss_dom).backward()
            optimizer.step()
            epoch_cls += loss_cls.item() * len(X_b)
            epoch_dom += loss_dom.item() * len(X_b)
        epoch_cls /= len(train_idx)
        epoch_dom /= len(train_idx)

        # ── Unlabeled pass: domain loss only (test features, no labels) ───────
        if unlabeled_loader is not None:
            for X_b, d_b in unlabeled_loader:
                X_b, d_b = X_b.to(device), d_b.to(device)
                optimizer.zero_grad()
                _, d_pred = model(X_b)
                bce_loss(d_pred, d_b).backward()
                optimizer.step()

        epoch_total = epoch_cls + epoch_dom

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            yp_val, dp_val_src = model(X_val_dev)
            val_cls     = bce_loss(yp_val, y_val_dev).item()
            val_dom_src = bce_loss(dp_val_src, d_val_dev).item()

            if X_unlab_val is not None:
                X_uv  = X_unlab_val.to(device)
                d_uv  = torch.ones(len(X_uv), 1, dtype=torch.float32).to(device)
                _, dp_tgt = model(X_uv)
                val_dom = (val_dom_src + bce_loss(dp_tgt, d_uv).item()) / 2.0
            else:
                val_dom = val_dom_src

            val_total = val_cls + val_dom

        history["train_cls"].append(epoch_cls)
        history["train_dom"].append(epoch_dom)
        history["train_total"].append(epoch_total)
        history["val_cls"].append(val_cls)
        history["val_dom"].append(val_dom)
        history["val_total"].append(val_total)
        history["lambda"].append(lam)

        scheduler.step(val_total)

        # Track best model by val_cls (classifier generalisation)
        if val_cls < best_val_cls - 1e-5:
            best_val_cls = val_cls
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(
                f"  [DANN] epoch {epoch:03d} | λ={lam:.3f} | "
                f"train  cls={epoch_cls:.4f}  dom={epoch_dom:.4f} | "
                f"val    cls={val_cls:.4f}  dom={val_dom:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  [DANN] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, scaler, history


# ──────────────────────────────────────────────────────────────────────────────
# Loss plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_ae_loss(
    history: Dict[str, List[float]],
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot DANN training curves: phenotype loss, domain loss, and λ schedule.

    Args:
        history:   dict returned by train_supervised_dae
        title:     figure suptitle (e.g. held-out dataset name)
        save_path: if provided, save figure to this path
    Returns:
        fig
    """
    epochs = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("Phenotype loss (BCE)", "train_cls", "val_cls"),
        ("Domain loss (BCE)",    "train_dom", "val_dom"),
        ("Total loss",           "train_total", "val_total"),
    ]

    for ax, (panel_title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.8, color="#2196F3")
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.8, color="#F44336", linestyle="--")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss",  fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # overlay λ ramp on domain loss panel
    ax2 = axes[1].twinx()
    ax2.plot(epochs, history["lambda"], color="#9C27B0", linewidth=1.2,
             linestyle=":", label="λ")
    ax2.set_ylabel("λ", fontsize=9, color="#9C27B0")
    ax2.tick_params(axis="y", labelcolor="#9C27B0")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_datasets(
    model: DANN,
    scaler: StandardScaler,
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> List[pd.DataFrame]:
    """
    Encode each DataFrame with the trained DANN encoder.

    Returns a list of DataFrames with columns latent_0 … latent_{k-1},
    preserving the original sample index of each dataset.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    latent_cols = [f"latent_{i}" for i in range(latent_dim)]
    encoded = []

    with torch.no_grad():
        for df, name in zip(microbiome_dfs, dataset_names):
            X_np     = df.values.astype(np.float32)
            X_scaled = scaler.transform(X_np)
            X_t      = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            z        = model.encode(X_t).cpu().numpy()
            encoded.append(pd.DataFrame(z, index=df.index, columns=latent_cols))

    return encoded
