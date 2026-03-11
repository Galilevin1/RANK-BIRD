"""
Figure 2: Two sub-figures.

Figure 2B — Possible Confounders Correlation Matrix
    Spearman correlation heatmap of dataset-level metadata and performance metrics
    loaded from Data/microbiome_analysis_results.csv.
    Lower triangle: correlation coefficient. Upper triangle: p-value.
    Significant cells (p < alpha) are outlined in black.

    Usage:
        from figures.figure2 import plot_figure2b
        fig = plot_figure2b(csv_path)
        fig.savefig(...)

Figure 2C — Zero-percentage baseline ROC + LODO AUC + Zero-count distributions
    For each phenotype produces one figure with:
      Panel A (top-left, 2 cols): ROC curves using per-sample % zeros as predictor
      Panel B (top-right, 1 col): LODO AUC barplot
      Panels C (bottom rows, 3/row): Zero-count histograms (Control vs Case)
                                      with Mann-Whitney p-value

    Usage:
        from figures.figure2 import plot_figure2c, run_figure2c
        fig = plot_figure2c(microbiome_dfs, target_dfs, dataset_names,
                            phenotype_name, apply_normalization=False)
        run_figure2c(phenotypes, data_root, figures_dir, apply_normalization=False)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# ─────────────────────────────────────────────────────────────
# Zero-percentage baseline
# ─────────────────────────────────────────────────────────────

def _zeros_pct_roc(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> Dict:
    """
    Use per-sample zero-percentage as a naive predictor and compute
    AUC + ROC curve for each dataset.
    """
    auc_scores, roc_curves_, zero_pcts, targets_out = [], [], [], []

    for mb, tgt, name in zip(microbiome_dfs, target_dfs, dataset_names):
        mb  = mb.reset_index(drop=True)
        tgt = tgt.reset_index(drop=True)

        zp = (mb == 0).mean(axis=1).values
        y  = tgt.values.ravel()

        if len(np.unique(y)) < 2:
            auc_scores.append(np.nan)
            roc_curves_.append((np.array([]), np.array([]), np.array([])))
        else:
            try:
                auc = roc_auc_score(y, zp)
                fpr, tpr, thr = roc_curve(y, zp)
                auc_scores.append(auc)
                roc_curves_.append((fpr, tpr, thr))
            except Exception:
                auc_scores.append(np.nan)
                roc_curves_.append((np.array([]), np.array([]), np.array([])))

        zero_pcts.append(zp)
        targets_out.append(y)

    return {
        'dataset_names':    dataset_names,
        'auc_scores':       auc_scores,
        'roc_curves':       roc_curves_,
        'zero_percentages': zero_pcts,
        'targets':          targets_out,
    }


# ─────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────

def _make_figure2(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    label_suffix: str = "",
) -> plt.Figure:
    """Build one Figure 2 for an already-preprocessed phenotype."""
    from evaluation.learning_protocols import lodo_protocol

    auc_results  = _zeros_pct_roc(microbiome_dfs, target_dfs, dataset_names)
    lodo_results = lodo_protocol(microbiome_dfs, target_dfs, dataset_names)

    n = len(dataset_names)
    n_dist_rows  = max(1, int(np.ceil(n / 3)))
    total_rows   = 1 + n_dist_rows
    height_ratios = [1.3] + [1.2] * n_dist_rows

    fig = plt.figure(figsize=(24, 8 + n_dist_rows * 5.5))
    gs  = gridspec.GridSpec(
        total_rows, 3, figure=fig,
        hspace=2.25, wspace=0.35,
        height_ratios=height_ratios,
        top=0.94, bottom=0.06, left=0.05, right=0.98,
    )

    tab10 = plt.cm.tab10(np.arange(10))
    color_map = {name: tab10[i % 10] for i, name in enumerate(dataset_names)}

    # ── Panel A: ROC curves ──────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, :2])
    for i, (fpr, tpr, _) in enumerate(auc_results['roc_curves']):
        auc = auc_results['auc_scores'][i]
        if not np.isnan(auc) and len(fpr):
            ax_roc.plot(fpr, tpr, color=color_map[dataset_names[i]],
                        linewidth=2.5, label=f"{dataset_names[i]} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.8, label='Random')
    ax_roc.set_xlim([0, 1]);  ax_roc.set_ylim([0, 1.25])
    ax_roc.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate',  fontsize=18, fontweight='bold')
    ax_roc.set_title('ROC Curves: Zero-Percentage Predictor', fontsize=18, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_roc.grid(True, alpha=0.3, linestyle='--')

    # ── Panel B: LODO barplot ────────────────────────────────
    ax_lodo = fig.add_subplot(gs[0, 2])
    lodo_aucs  = []
    bar_colors = []
    for name in dataset_names:
        row = lodo_results[lodo_results['test_dataset'] == name]
        lodo_aucs.append(float(row['auc'].values[0]) if len(row) > 0 else 0.0)
        bar_colors.append(color_map[name])

    bars = ax_lodo.bar(range(n), lodo_aucs, color=bar_colors,
                       edgecolor='black', linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, lodo_aucs):
        ax_lodo.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
    ax_lodo.set_xticks(range(n))
    ax_lodo.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=10)
    ax_lodo.set_ylim([0, 1.3])
    ax_lodo.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random')
    ax_lodo.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax_lodo.set_ylabel('AUC',     fontsize=14, fontweight='bold')
    ax_lodo.set_title('LODO Cross-Validation', fontsize=16, fontweight='bold')
    ax_lodo.legend(fontsize=9, loc='upper right')
    ax_lodo.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ── Panels C: Zero-count distributions ──────────────────
    for i, (mb, tgt, name) in enumerate(zip(microbiome_dfs, target_dfs, dataset_names)):
        ax = fig.add_subplot(gs[(i // 3) + 1, i % 3])
        zero_counts = (mb.values == 0).sum(axis=1)
        y = tgt.values.ravel()

        if len(np.unique(y)) >= 2:
            z0, z1 = zero_counts[y == 0], zero_counts[y == 1]
            ax.hist([z0, z1], bins=20, alpha=0.6,
                    label=['Control', 'Case'],
                    color=['lightcoral', 'lightblue'],
                    edgecolor='black', linewidth=0.5)
            ax.legend(fontsize=11)
            if len(z0) > 0 and len(z1) > 0:
                _, p = stats.mannwhitneyu(z0, z1, alternative='two-sided')
                ax.text(0.97, 0.95, f'MW p={p:.4f}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        else:
            ax.hist(zero_counts, bins=20, alpha=0.7,
                    color='lightgray', edgecolor='black')

        ax.set_xlabel('Number of Zeros per Sample', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count',               fontsize=12, fontweight='bold')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'{phenotype_name}{label_suffix}', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────
# Public interface — Figure 2B: Confounder correlation matrix
# ─────────────────────────────────────────────────────────────

_CORR_COLUMNS = [
    'sample_count', 'feature_count',
    'microbes_unique_test', 'microbes_unique_train',
    'similarity_score_all',
    'LODO_auc', 'within_dataset_auc', 'auc_difference',
    'auc_difference_2', 'auc_zeros',
    'mean_age', 'male_percentage',
]


def plot_figure2b(
    csv_path: str,
    alpha: float = 0.05,
) -> plt.Figure:
    """
    Spearman correlation heatmap of dataset-level metadata + performance metrics.

    Lower triangle: Spearman r.  Upper triangle: p-value.
    Significant cells (p < alpha) are outlined in black.

    Parameters
    ----------
    csv_path : path to Data/microbiome_analysis_results.csv
    alpha    : significance threshold for border highlighting
    """
    import seaborn as sns
    from scipy.stats import spearmanr

    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('N/A', np.nan)

    cols = [c for c in _CORR_COLUMNS if c in df.columns]
    corr_df = df[cols].dropna(subset=cols).copy()
    for c in cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')
    corr_df = corr_df.dropna()

    n = len(cols)
    corr_mat = np.full((n, n), np.nan)
    p_mat    = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(corr_df[cols[i]], corr_df[cols[j]])
            corr_mat[i, j] = r
            p_mat[i, j]    = p

    # Annotations: lower triangle = r, upper = p-value
    annot = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            if i < j:
                annot[i, j] = f'p={p_mat[i, j]:.3f}'
            else:
                annot[i, j] = f'{corr_mat[i, j]:.2f}'

    pretty = [c.replace('_', ' ').title() for c in cols]

    fig, ax = plt.subplots(figsize=(18, 14))
    hm = sns.heatmap(
        corr_mat,
        annot=annot, fmt='',
        cmap='coolwarm_r', center=0,
        cbar_kws={'shrink': 0.5},
        annot_kws={'size': 14},
        ax=ax,
    )
    hm.set_xticklabels(pretty, fontsize=16, rotation=45, ha='right')
    hm.set_yticklabels(pretty, fontsize=16, rotation=0)

    # Black border on significant lower-triangle cells
    for i in range(n):
        for j in range(n):
            if i > j and p_mat[i, j] < alpha:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='black', lw=3))

    ax.set_title('Spearman Correlations — Possible Confounders',
                 fontsize=18, fontweight='bold', pad=14)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Public interface — Figure 2C: Zero-% ROC + LODO + distributions
# ─────────────────────────────────────────────────────────────

def plot_figure2c(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    apply_normalization: bool = False,
) -> plt.Figure:
    """
    Generate one Figure 2C for a phenotype.

    If apply_normalization=True the normalization pipeline is applied first
    and target_dfs is re-aligned to match any datasets that survive filtering.
    """
    if apply_normalization:
        from src.rankbird.normalization.pipeline import apply_normalization_pipeline
        print(f"  Applying normalization pipeline for {phenotype_name}")
        orig_names  = list(dataset_names)
        name_to_tgt = {n: t for n, t in zip(orig_names, target_dfs)}
        microbiome_dfs, dataset_names = apply_normalization_pipeline(
            list(microbiome_dfs), list(dataset_names)
        )
        target_dfs = [name_to_tgt[n] for n in dataset_names if n in name_to_tgt]
        label_suffix = " (normalized)"
    else:
        label_suffix = " (raw)"

    print(f"  Building figure 2C for {phenotype_name}{label_suffix}")
    return _make_figure2(microbiome_dfs, target_dfs, dataset_names,
                         phenotype_name, label_suffix=label_suffix)


def run_figure2c(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    apply_normalization: bool = False,
):
    """
    Run Figure 2C for all phenotypes and save PNGs to figures_dir.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out  = Path(figures_dir)
    data = Path(data_root)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = data / pheno_str

        if not folder.exists():
            print(f"  Skipping {pheno_str}: folder not found")
            continue

        print(f"\n{'='*50}\nFigure 2C: {pheno_str}\n{'='*50}")
        try:
            microbiome_dfs, target_dfs, dataset_names = \
                load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        if len(dataset_names) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets")
            continue

        try:
            fig = plot_figure2c(
                microbiome_dfs, target_dfs, dataset_names,
                pheno_str,
                apply_normalization=apply_normalization,
            )
            safe = pheno_str.replace(" ", "_")
            mode = "normalized" if apply_normalization else "raw"
            fig.savefig(out / f"figure2c_{safe}_{mode}.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved figure 2C for {pheno_str}")
        except Exception as e:
            print(f"  Error generating figure 2C for {pheno_str}: {e}")
