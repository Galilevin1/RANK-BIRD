"""
Figure 3: Microbiome abundance heatmap per phenotype.

Shows all samples (rows) × top-varying common features (columns), z-scored
for visualization, with datasets stacked and separated by horizontal lines.
Generated for both raw and normalized data.

Usage
-----
    from figures.figure3 import plot_figure3, run_figure3

    # generate for one phenotype
    fig_raw, fig_norm = plot_figure3(
        microbiome_dfs, dataset_names, phenotype_name,
        apply_normalization=True   # also produce normalized version
    )
    fig_raw.savefig(...)
    fig_norm.savefig(...)

    # or use the pipeline runner (calls plot_figure3 for every phenotype)
    run_figure3(phenotypes, data_root, figures_dir, apply_normalization=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

from utils.microbe_names import rename_microbes


# ─────────────────────────────────────────────────────────────
# KS significance helper
# ─────────────────────────────────────────────────────────────

def _compute_ks_significance(
    microbiome_dfs: List[pd.DataFrame],
    common_features,
    dataset_names: List[str],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    For each microbial feature perform pairwise KS tests across all dataset
    pairs, apply Benjamini–Hochberg FDR correction, and return:
        ks_df              — full per-feature × per-pair results
        sig_raw_microbes   — features significant before FDR
        sig_fdr_microbes   — features significant after FDR
    """
    rows = []
    for feature in common_features:
        for i, j in combinations(range(len(microbiome_dfs)), 2):
            stat, p = ks_2samp(
                microbiome_dfs[i][feature].values,
                microbiome_dfs[j][feature].values,
            )
            rows.append({
                "microbe":   feature,
                "dataset1":  dataset_names[i],
                "dataset2":  dataset_names[j],
                "ks_stat":   stat,
                "p_value":   p,
                "significant_raw": p < alpha,
            })

    ks_df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(ks_df["p_value"], method='fdr_bh')
    ks_df["p_fdr"]         = p_fdr
    ks_df["significant_fdr"] = reject

    sig_raw = (ks_df.groupby("microbe")["significant_raw"]
               .max().pipe(lambda s: s[s].index.tolist()))
    sig_fdr = (ks_df.groupby("microbe")["significant_fdr"]
               .max().pipe(lambda s: s[s].index.tolist()))

    return ks_df, sig_raw, sig_fdr


# ─────────────────────────────────────────────────────────────
# Single heatmap
# ─────────────────────────────────────────────────────────────

def _make_heatmap(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    max_features: int = 50,
    label_suffix: str = "",
) -> plt.Figure:
    """
    Build one heatmap figure from a list of (possibly already normalized)
    microbiome DataFrames.

    Steps
    -----
    1. Intersect columns → common features
    2. KS test → report significance
    3. Select top-`max_features` by variance
    4. Z-score the combined matrix (for display only)
    5. Rename features
    6. Plot with dataset separator lines
    """
    # 1. Common features
    common = microbiome_dfs[0].columns
    for df in microbiome_dfs[1:]:
        common = common.intersection(df.columns)

    if len(common) == 0:
        raise ValueError(f"No common features for {phenotype_name}")

    # 2. KS significance
    _, _, sig_fdr = _compute_ks_significance(microbiome_dfs, common, dataset_names)
    pct_sig = 100 * len(sig_fdr) / len(common)
    print(f"  KS (FDR): {len(sig_fdr)}/{len(common)} significant microbes "
          f"({pct_sig:.1f}%)")

    # 3. Top features by variance
    if len(common) > max_features:
        all_data = pd.concat([df[common] for df in microbiome_dfs], axis=0)
        common = all_data.var().sort_values(ascending=False).head(max_features).index

    # 4. Z-score combined matrix
    aligned = [df[common] for df in microbiome_dfs]
    X = pd.concat(aligned, axis=0)
    X_z = (X - X.mean()) / (X.std() + 1e-10)
    X_z = X_z.fillna(0)

    # Dataset boundary positions
    boundaries = [0]
    for df in aligned:
        boundaries.append(boundaries[-1] + len(df))

    # 5. Rename features
    X_z.columns = rename_microbes(X_z.columns.tolist())

    # 6. Plot
    n_samples  = len(X_z)
    n_features = len(X_z.columns)
    fig_w = max(14, n_features * 0.28)
    fig_h = max(10, n_samples  * 0.025)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        X_z,
        cmap='RdBu_r', center=0, robust=True,
        vmin=-3, vmax=3,
        yticklabels=False, xticklabels=False,
        cbar=False,
        ax=ax,
    )

    # Separator lines between datasets
    for b in boundaries[1:-1]:
        ax.axhline(y=b, color='black', linewidth=3)

    # Dataset labels on the left
    for i, name in enumerate(dataset_names):
        y_mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(-0.5, y_mid, name,
                ha='right', va='center', fontsize=16, fontweight='bold',
                transform=ax.get_yaxis_transform())

    title = f'{phenotype_name}{label_suffix}'
    ax.set_title(title, fontsize=22, fontweight='bold', pad=18)
    ax.text(0.5, -0.02,
            f'KS significant microbes (FDR): {len(sig_fdr)}/{len(common)} '
            f'({pct_sig:.1f}%)',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=11, color='dimgrey')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

def plot_figure3(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    apply_normalization: bool = False,
    max_features: int = 50,
) -> plt.Figure:
    """
    Generate one figure 3 heatmap for a phenotype.

    If apply_normalization=True, the normalization pipeline is applied first
    and the figure is labelled "(normalized)". Otherwise raw data is used.
    """
    if apply_normalization:
        from src.rankbird.normalization.pipeline import apply_normalization_pipeline
        print(f"  Applying normalization pipeline for {phenotype_name}")
        microbiome_dfs, dataset_names = apply_normalization_pipeline(
            list(microbiome_dfs), list(dataset_names)
        )
        label_suffix = " (normalized)"
    else:
        label_suffix = " (raw)"

    print(f"  Building heatmap for {phenotype_name}{label_suffix}")
    return _make_heatmap(microbiome_dfs, dataset_names, phenotype_name,
                         max_features=max_features, label_suffix=label_suffix)


def run_figure3(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    apply_normalization: bool = True,
    max_features: int = 50,
):
    """
    Run figure 3 for all phenotypes and save results to figures_dir.

    Parameters
    ----------
    phenotypes : list of (phenotype, dtype) tuples
    data_root : str
        Root folder containing phenotype subfolders.
    figures_dir : str
        Where to save the PNG files.
    apply_normalization : bool
        If True, also produce normalized versions.
    max_features : int
        Maximum number of features shown per heatmap.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out = Path(figures_dir)
    data = Path(data_root)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder = data / pheno_str

        if not folder.exists():
            print(f"  Skipping {pheno_str}: folder not found")
            continue

        print(f"\n{'='*50}\nFigure 3: {pheno_str}\n{'='*50}")
        try:
            microbiome_dfs, _, dataset_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        if len(dataset_names) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets")
            continue

        try:
            fig = plot_figure3(
                microbiome_dfs, dataset_names, pheno_str,
                apply_normalization=apply_normalization,
                max_features=max_features,
            )
            safe   = pheno_str.replace(" ", "_")
            mode   = "normalized" if apply_normalization else "raw"
            fig.savefig(out / f"figure3_{safe}_{mode}.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved figure 3 for {pheno_str}")

        except Exception as e:
            print(f"  Error generating figure 3 for {pheno_str}: {e}")
