"""
Figure 5: Global Original vs Global Normalized protocol comparison.

Single plot with 6 paired boxplots (interleaved by protocol):
  LODO Original | LODO Normalized |
  Internal Validation Original | Internal Validation Normalized |
  Within Learning Original | Within Learning Normalized

Usage
-----
    from figures.figure5 import plot_figure5
    fig = plot_figure5(results_df_original, results_df_normalized)
    fig.savefig(...)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


_PROTOCOL_ORDER = ["LODO", "Internal Validation", "Within Learning"]

# Colors for Original / Normalized
_PALETTE = {"Original": "#7EB8D4", "Normalized": "#F4A36A"}


def plot_figure5(
    results_df_original: pd.DataFrame,
    results_df_normalized: pd.DataFrame,
) -> plt.Figure:
    """
    Single boxplot with 6 groups: for each protocol, Original then Normalized,
    all on the same y-axis scale.

    Parameters
    ----------
    results_df_original   : results from results_global_original
    results_df_normalized : results from results_global_normalized

    Returns
    -------
    fig : matplotlib Figure
    """
    df_orig = results_df_original.copy()
    df_norm = results_df_normalized.copy()
    df_orig["condition"] = "Original"
    df_norm["condition"] = "Normalized"
    combined = pd.concat([df_orig, df_norm], ignore_index=True)
    combined["auc"] = pd.to_numeric(combined["auc"], errors="coerce")

    # Keep only protocols present in both
    protocol_order = [p for p in _PROTOCOL_ORDER if p in combined["protocol"].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=combined,
        x="protocol", y="auc",
        hue="condition",
        order=protocol_order,
        hue_order=["Original", "Normalized"],
        palette=_PALETTE,
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
        hue_order=["Original", "Normalized"],
        palette={"Original": "#2a6e8a", "Normalized": "#b5581e"},
        dodge=True,
        alpha=0.45, size=5, jitter=True,
        legend=False,
        ax=ax,
    )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Random (0.5)")

    # y-axis range from combined data
    all_auc = combined["auc"].dropna()
    y_min = max(0.0, float(all_auc.min()) - 0.05)
    y_max = min(1.0, float(all_auc.max()) + 0.05)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Protocol", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=13, fontweight="bold")
    ax.set_title(
        "Protocol Comparison: Original vs Normalized Preprocessing",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Merge legend (boxplot hue + random line)
    handles, labels = ax.get_legend_handles_labels()
    # get_legend_handles_labels returns hue handles + random line
    ax.legend(handles=handles, labels=labels, fontsize=11,
              framealpha=0.9, loc="lower right")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    return fig
