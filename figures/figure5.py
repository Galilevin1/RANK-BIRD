"""
Figure 5: Global preprocessing protocol comparison.

Boxplots per protocol (LODO / Internal Validation / Within Learning),
with one box per condition: Original, Normalized, and optionally Autoencoder.

Usage
-----
    from figures.figure5 import plot_figure5
    fig = plot_figure5(results_df_original, results_df_normalized)
    fig = plot_figure5(results_df_original, results_df_normalized, results_df_autoencoder)
    fig.savefig(...)
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


_PROTOCOL_ORDER = ["LODO", "Internal Validation", "Within Learning"]

_PALETTE = {
    "Original":    "#7EB8D4",
    "Normalized":  "#F4A36A",
    "Autoencoder": "#66BB6A",
}
_STRIP_PALETTE = {
    "Original":    "#2a6e8a",
    "Normalized":  "#b5581e",
    "Autoencoder": "#2e7d32",
}


def plot_figure5(
    results_df_original: pd.DataFrame,
    results_df_normalized: pd.DataFrame,
    results_df_autoencoder: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Boxplot comparison across protocols and preprocessing conditions.

    Parameters
    ----------
    results_df_original     : results with no preprocessing
    results_df_normalized   : results after rank normalization
    results_df_autoencoder  : (optional) results after supervised DAE encoding

    Returns
    -------
    fig : matplotlib Figure
    """
    frames = []
    df_orig = results_df_original.copy();  df_orig["condition"] = "Original"
    df_norm = results_df_normalized.copy(); df_norm["condition"] = "Normalized"
    frames = [df_orig, df_norm]

    if results_df_autoencoder is not None and not results_df_autoencoder.empty:
        df_ae = results_df_autoencoder.copy()
        df_ae["condition"] = "Autoencoder"
        frames.append(df_ae)

    combined = pd.concat(frames, ignore_index=True)
    combined["auc"] = pd.to_numeric(combined["auc"], errors="coerce")

    hue_order = [c for c in ["Original", "Normalized", "Autoencoder"]
                 if c in combined["condition"].unique()]
    protocol_order = [p for p in _PROTOCOL_ORDER if p in combined["protocol"].unique()]

    palette       = {k: v for k, v in _PALETTE.items()       if k in hue_order}
    strip_palette = {k: v for k, v in _STRIP_PALETTE.items() if k in hue_order}

    fig_width = 10 + 2 * len(hue_order)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.boxplot(
        data=combined,
        x="protocol", y="auc",
        hue="condition",
        order=protocol_order,
        hue_order=hue_order,
        palette=palette,
        width=0.6,
        fliersize=0,
        linewidth=1.5,
        medianprops=dict(color="#DC143C", linewidth=2.5),
        ax=ax,
    )
    sns.stripplot(
        data=combined,
        x="protocol", y="auc",
        hue="condition",
        order=protocol_order,
        hue_order=hue_order,
        palette=strip_palette,
        dodge=True,
        alpha=0.45, size=5, jitter=True,
        legend=False,
        ax=ax,
    )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Random (0.5)")

    all_auc = combined["auc"].dropna()
    y_min = max(0.0, float(all_auc.min()) - 0.05)
    y_max = min(1.0, float(all_auc.max()) + 0.05)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Protocol", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=13, fontweight="bold")
    title = "Protocol Comparison: " + " vs ".join(hue_order) + " Preprocessing"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=11,
              framealpha=0.9, loc="lower right")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    return fig
