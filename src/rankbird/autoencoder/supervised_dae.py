"""
Supervised Denoising Autoencoder (SDAE) for microbiome representation learning.

Architecture:
    Input → [Gaussian noise, training only] → Encoder → Latent z
    z → Decoder → Reconstruction   (MSE loss)
    z → Classifier → Phenotype prob (BCE loss)

Total loss = MSE(recon, clean_input) + BCE(pred, y)
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
# Model
# ──────────────────────────────────────────────────────────────────────────────

class SupervisedDAE(nn.Module):
    """Supervised Denoising Autoencoder with a classification head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.noise_std = noise_std

        # ── Encoder (BatchNorm after each linear — normalises across datasets) ─
        enc_layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layers: List[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # ── Classifier head (dropout before final layer to regularise) ────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_recon  – reconstructed input
            y_pred   – phenotype probability (0-1)
            z        – latent representation
        """
        # Add noise only during training
        if self.training and self.noise_std > 0:
            x_noisy = x + torch.randn_like(x) * self.noise_std
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        y_pred = self.classifier(z)
        return x_recon, y_pred, z


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
    noise_std: float = 0.0,
    dropout: float = 0.2,
    cls_weight: float = 5.0,
    patience: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[SupervisedDAE, StandardScaler]:
    """
    Train a Supervised DAE on labeled microbiome DataFrames, with optional
    unlabeled DataFrames (e.g. the held-out LODO test set) used exclusively
    for the reconstruction objective.

    Reconstruction loss : labeled train data  +  unlabeled data (features only)
    Classification loss : labeled train data  only  (no label leakage)

    Args:
        microbiome_dfs : labeled training DataFrames
        target_series  : binary labels for microbiome_dfs
        unlabeled_dfs  : held-out test DataFrames — features used for denoising,
                         labels NEVER seen during training
    Returns:
        (trained_model, fitted_scaler, history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        mode = "transductive" if unlabeled_dfs else "standard"
        print(f"  [SDAE] device={device}  mode={mode}")

    # ── Prepare labeled data ──────────────────────────────────────────────────
    X_labeled = pd.concat(microbiome_dfs, axis=0).values.astype(np.float32)
    y_all     = target_series.values.astype(np.float32).ravel()
    assert len(X_labeled) == len(y_all), "Mismatch between features and labels"

    # Scaler fit on labeled training split only
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

    # Labeled loader: reconstruction + classification
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    # ── Prepare unlabeled data (held-out features, reconstruction only) ───────
    unlabeled_loader = None
    if unlabeled_dfs is not None:
        X_unlabeled = pd.concat(unlabeled_dfs, axis=0).values.astype(np.float32)
        X_unlabeled_scaled = scaler.transform(X_unlabeled)   # transform only, never fit
        X_unlab_t = torch.tensor(X_unlabeled_scaled, dtype=torch.float32)
        unlabeled_loader = DataLoader(
            TensorDataset(X_unlab_t), batch_size=batch_size, shuffle=True
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim = X_labeled.shape[1]
    model = SupervisedDAE(
        input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
        dropout=dropout, noise_std=noise_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    history = {
        "train_mse": [], "train_bce": [], "train_total": [],
        "val_mse":   [], "val_bce":   [], "val_total":   [],
    }

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mse = 0.0
        epoch_bce = 0.0

        # ── Labeled pass: reconstruction + classification ─────────────────────
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            x_recon, y_pred, _ = model(X_b)
            loss_mse = mse_loss(x_recon, X_b)
            loss_bce = bce_loss(y_pred, y_b)
            loss = loss_mse + cls_weight * loss_bce
            loss.backward()
            optimizer.step()
            epoch_mse += loss_mse.item() * len(X_b)
            epoch_bce += loss_bce.item() * len(X_b)
        epoch_mse /= len(train_idx)
        epoch_bce /= len(train_idx)

        # ── Unlabeled pass: reconstruction only (no labels, no leakage) ───────
        if unlabeled_loader is not None:
            for (X_b,) in unlabeled_loader:
                X_b = X_b.to(device)
                optimizer.zero_grad()
                x_recon, _, _ = model(X_b)
                loss_recon = mse_loss(x_recon, X_b)
                loss_recon.backward()
                optimizer.step()

        epoch_total = epoch_mse + cls_weight * epoch_bce

        # Validation
        model.eval()
        with torch.no_grad():
            xr_val, yp_val, _ = model(X_val_dev)
            val_mse   = mse_loss(xr_val, X_val_dev).item()
            val_bce   = bce_loss(yp_val, y_val_dev).item()
            val_total = val_mse + cls_weight * val_bce

        history["train_mse"].append(epoch_mse)
        history["train_bce"].append(epoch_bce)
        history["train_total"].append(epoch_total)
        history["val_mse"].append(val_mse)
        history["val_bce"].append(val_bce)
        history["val_total"].append(val_total)

        scheduler.step(val_total)

        # Track best model by val_bce — classifier generalisation is what matters
        if val_bce < best_val_loss - 1e-5:
            best_val_loss = val_bce
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(
                f"  [SDAE] epoch {epoch:03d} | "
                f"train  total={epoch_total:.4f}  mse={epoch_mse:.4f}  bce={epoch_bce:.4f} | "
                f"val    total={val_total:.4f}  mse={val_mse:.4f}  bce={val_bce:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  [SDAE] early stopping at epoch {epoch}")
            break

    # Restore best weights
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
    Plot training and validation loss curves split into three panels:
        - Total loss (MSE + cls_weight * BCE)
        - Reconstruction loss (MSE)
        - Classification loss (BCE, unweighted)

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
        ("Total loss",            "train_total", "val_total"),
        ("Reconstruction (MSE)",  "train_mse",   "val_mse"),
        ("Classification (BCE)",  "train_bce",   "val_bce"),
    ]

    for ax, (panel_title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.8, color="#2196F3")
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.8, color="#F44336", linestyle="--")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss",  fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

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
    model: SupervisedDAE,
    scaler: StandardScaler,
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> List[pd.DataFrame]:
    """
    Encode each DataFrame with the trained SDAE encoder.

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
            X_np = df.values.astype(np.float32)
            X_scaled = scaler.transform(X_np)
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            z = model.encode(X_t).cpu().numpy()
            encoded.append(pd.DataFrame(z, index=df.index, columns=latent_cols))

    return encoded
