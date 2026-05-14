"""
Plotting functions for pairwise LODO + SHAP analysis.

  plot_pairwise_lodo_figure  — 3-panel figure (SHAP grid + pairwise AUC heatmap
                                + feature consistency bar) for one phenotype.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from utils.microbe_names import rename_microbes


def plot_pairwise_lodo_figure(
    phenotype_name: str,
    pairwise_results: pd.DataFrame,
    full_lodo_shap: Dict,
    dataset_names: List[str],
    rename_features: bool = True,
    top_n: int = 10,
) -> plt.Figure:
    """
    Three side-by-side panels for one phenotype.

    Left:   SHAP feature presence grid — which top-N features appear per LODO test dataset.
    Middle: Pairwise LODO AUC heatmap  — AUC(train=col, test=row).
    Right:  Feature consistency barplot — % of test datasets each feature appears in.

    Figure height scales with the number of unique features.
    """
    seen = set()
    for ds in dataset_names:
        if ds in full_lodo_shap:
            for f in full_lodo_shap[ds]["top_10_features"]["feature"]:
                seen.add(f)
    n_features = max(len(seen), 1)

    panel_size = max(12, n_features * 0.5 + 4)
    fig, axes = plt.subplots(
        1, 3, figsize=(panel_size * 3, panel_size),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    _plot_shap_grid(axes[0], full_lodo_shap, dataset_names, rename_features, top_n)
    _plot_pairwise_heatmap(axes[1], pairwise_results, dataset_names)
    _plot_consistency_bar(axes[2], full_lodo_shap, dataset_names, rename_features, top_n)

    fig.suptitle(f"{phenotype_name} — LODO Dataset Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _plot_shap_grid(ax, full_lodo_shap, dataset_names, rename_features, top_n):
    all_features, seen = [], set()
    for ds in dataset_names:
        if ds in full_lodo_shap:
            for f in full_lodo_shap[ds]["top_10_features"]["feature"]:
                if f not in seen:
                    all_features.append(f)
                    seen.add(f)
    if not all_features:
        ax.set_visible(False)
        return

    labels = rename_microbes(all_features) if rename_features else all_features
    presence = np.zeros((len(all_features), len(dataset_names)))
    for j, ds in enumerate(dataset_names):
        if ds in full_lodo_shap:
            top = set(full_lodo_shap[ds]["top_10_features"]["feature"])
            for i, f in enumerate(all_features):
                presence[i, j] = 1.0 if f in top else 0.0

    sns.heatmap(presence, ax=ax,
                xticklabels=dataset_names, yticklabels=labels,
                cmap="Greys", vmin=0, vmax=1,
                linewidths=0.4, linecolor="lightgrey", cbar=False)
    ax.set_title(f"Top {top_n} SHAP Features\nper LODO Test Dataset",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Test Dataset", fontsize=10)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)


def _plot_pairwise_heatmap(ax, pairwise_results, dataset_names):
    n = len(dataset_names)
    matrix = np.full((n, n), np.nan)
    for _, row in pairwise_results.iterrows():
        if row["train_dataset"] in dataset_names and row["test_dataset"] in dataset_names:
            ti = dataset_names.index(row["train_dataset"])
            xi = dataset_names.index(row["test_dataset"])
            matrix[xi, ti] = row["auc"]

    sns.heatmap(matrix, ax=ax,
                xticklabels=dataset_names, yticklabels=dataset_names,
                annot=True, fmt=".2f",
                cmap="coolwarm_r", vmin=0.4, vmax=1.0, center=0.7,
                linewidths=0.4, linecolor="lightgrey",
                cbar_kws={"label": "AUC", "shrink": 0.8})
    ax.set_title("Pairwise LODO AUC\n(row = test, col = train)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Train Dataset", fontsize=10)
    ax.set_ylabel("Test Dataset", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)


def _plot_consistency_bar(ax, full_lodo_shap, dataset_names, rename_features, top_n):
    n_datasets = len(dataset_names)
    feature_counts: dict = {}
    for ds in dataset_names:
        if ds in full_lodo_shap:
            for f in full_lodo_shap[ds]["top_10_features"]["feature"]:
                feature_counts[f] = feature_counts.get(f, 0) + 1
    if not feature_counts:
        ax.set_visible(False)
        return

    sorted_items = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    features, counts = zip(*sorted_items)
    percentages = [c / n_datasets * 100 for c in counts]
    labels = rename_microbes(list(features)) if rename_features else list(features)

    ax.barh(range(len(labels)), percentages, color="#333333")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of LODO Datasets", fontsize=10)
    ax.set_xlim(0, 120)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(f"Feature Consistency\n(top {top_n} SHAP, % datasets)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, pct in enumerate(percentages):
        ax.text(pct + 2, i, f"{pct:.0f}%", va="center", fontsize=9)
