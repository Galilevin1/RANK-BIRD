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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


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

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
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

        # ── Classifier head ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
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
    Train a Supervised DAE on a list of microbiome DataFrames + binary labels.

    Args:
        microbiome_dfs: list of (n_samples_i, n_features) DataFrames (all must share columns)
        target_series:  binary labels aligned with the concatenated DataFrames
        ...
    Returns:
        (trained_model, fitted_scaler)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        print(f"  [SDAE] device={device}")

    # ── Prepare data ─────────────────────────────────────────────────────────
    X_all = pd.concat(microbiome_dfs, axis=0)
    y_all = target_series.values.astype(np.float32).ravel()
    assert len(X_all) == len(y_all), "Mismatch between features and labels"

    # Align feature union — fill missing with 0
    X_np = X_all.values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    idx = np.arange(len(X_np))
    np.random.seed(42)
    val_size = max(1, int(len(idx) * val_split))
    val_idx = np.random.choice(idx, size=val_size, replace=False)
    train_idx = np.setdiff1d(idx, val_idx)

    scaler.fit(X_np[train_idx])
    X_scaled = scaler.transform(X_np)

    X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(X_scaled[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y_all[val_idx],   dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim = X_np.shape[1]
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

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            x_recon, y_pred, _ = model(X_b)
            loss = mse_loss(x_recon, X_b) + cls_weight * bce_loss(y_pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(train_idx)

        # Validation
        model.eval()
        with torch.no_grad():
            xr_val, yp_val, _ = model(X_val_dev)
            val_loss = (
                mse_loss(xr_val, X_val_dev) + cls_weight * bce_loss(yp_val, y_val_dev)
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  [SDAE] epoch {epoch:03d}  train={epoch_loss:.4f}  val={val_loss:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  [SDAE] early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, scaler


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
