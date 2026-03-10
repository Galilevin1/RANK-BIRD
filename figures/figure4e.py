"""
Figure 4E: Jaccard similarity of SHAP features vs LODO performance,
aggregated across all phenotypes.

Requires figure 4 to have been run first so that pairwise_lodo_results.csv,
shap_details.txt and dataset_names.txt exist per phenotype subfolder.

Produces 4 plots saved to figures_out/:
  figure4e_overall_scatter.png      — all pairs, colored by phenotype
  figure4e_per_phenotype_grid.png   — one scatter panel per phenotype
  figure4e_distributions.png        — histograms of Jaccard and LODO AUC
  figure4e_correlations.png         — Pearson r / Spearman ρ per phenotype
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional

from evaluation.pairwise_lodo import _load_shap_details


# ─────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────

def _jaccard(set1: set, set2: set) -> float:
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def _compute_jaccard_matrix(lodo_shap: Dict, dataset_names: List[str]) -> np.ndarray:
    n = len(dataset_names)
    matrix = np.full((n, n), np.nan)
    for i, d1 in enumerate(dataset_names):
        for j, d2 in enumerate(dataset_names):
            if d1 in lodo_shap and d2 in lodo_shap:
                f1 = set(lodo_shap[d1]['top_10_features']['feature'])
                f2 = set(lodo_shap[d2]['top_10_features']['feature'])
                matrix[i, j] = _jaccard(f1, f2)
    return matrix


# ─────────────────────────────────────────────────────────────
# Analysis orchestrator (load-if-exists aware)
# ─────────────────────────────────────────────────────────────

def run_figure4e_analysis(
    figure4_data_dir: str,
    metric: str = 'auc',
) -> Optional[Dict]:
    """
    Load figure 4 results for all phenotype subfolders, compute Jaccard
    similarity between SHAP feature sets, correlate with pairwise LODO
    performance, and return aggregated statistics.

    Returns None if no usable phenotype data is found.
    """
    base = Path(figure4_data_dir)
    phenotype_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir()
        and (d / "pairwise_lodo_results.csv").exists()
        and (d / "shap_details.txt").exists()
        and (d / "dataset_names.txt").exists()
    ])

    if not phenotype_dirs:
        print(f"No figure 4 results found in {figure4_data_dir}. Run figure 4 first.")
        return None

    print(f"\nFigure 4E: found {len(phenotype_dirs)} phenotypes")

    all_jaccard, all_lodo, all_pairs, all_phenos = [], [], [], []
    phenotype_stats: Dict[str, dict] = {}

    for pheno_dir in phenotype_dirs:
        pheno_name = pheno_dir.name.replace("_", " ")
        try:
            pairwise      = pd.read_csv(pheno_dir / "pairwise_lodo_results.csv")
            lodo_shap     = _load_shap_details(pheno_dir / "shap_details.txt")
            dataset_names = (pheno_dir / "dataset_names.txt").read_text().strip().splitlines()

            jaccard_matrix = _compute_jaccard_matrix(lodo_shap, dataset_names)

            pheno_j, pheno_l = [], []
            for _, row in pairwise.iterrows():
                if row['train_dataset'] == row['test_dataset']:
                    continue
                if row['train_dataset'] not in dataset_names or row['test_dataset'] not in dataset_names:
                    continue
                ti = dataset_names.index(row['train_dataset'])
                xi = dataset_names.index(row['test_dataset'])
                j_val = jaccard_matrix[ti, xi]
                l_val = row[metric]
                if np.isnan(j_val) or np.isnan(l_val):
                    continue
                pheno_j.append(j_val)
                pheno_l.append(l_val)
                all_jaccard.append(j_val)
                all_lodo.append(l_val)
                all_pairs.append(f"{row['train_dataset']} → {row['test_dataset']}")
                all_phenos.append(pheno_name)

            if len(pheno_j) > 2:
                pr, pp = stats.pearsonr(pheno_j, pheno_l)
                sr, sp = stats.spearmanr(pheno_j, pheno_l)
                phenotype_stats[pheno_name] = {
                    'n_pairs':    len(pheno_j),
                    'pearson_r':  pr, 'pearson_p':  pp,
                    'spearman_r': sr, 'spearman_p': sp,
                    'mean_jaccard': float(np.mean(pheno_j)),
                    'mean_lodo':    float(np.mean(pheno_l)),
                }
                print(f"  {pheno_name}: {len(pheno_j)} pairs | "
                      f"Pearson r={pr:.3f} | Spearman ρ={sr:.3f}")

        except Exception as e:
            print(f"  Error processing {pheno_name}: {e}")

    if len(all_jaccard) < 3:
        print("Not enough data points for figure 4E.")
        return None

    jac = np.array(all_jaccard)
    lod = np.array(all_lodo)
    pr, pp = stats.pearsonr(jac, lod)
    sr, sp = stats.spearmanr(jac, lod)

    overall_stats = {
        'n_phenotypes':  len(phenotype_stats),
        'n_total_pairs': len(jac),
        'pearson_r':  pr,  'pearson_p':  pp,
        'spearman_r': sr,  'spearman_p': sp,
        'mean_jaccard': float(jac.mean()), 'std_jaccard': float(jac.std()),
        'mean_lodo':    float(lod.mean()), 'std_lodo':    float(lod.std()),
    }

    print(f"\n  Overall — Pearson r={pr:.3f} (p={pp:.2e}) | "
          f"Spearman ρ={sr:.3f} (p={sp:.2e})")

    return {
        'jaccard_scores':   jac,
        'lodo_scores':      lod,
        'pair_labels':      all_pairs,
        'phenotype_labels': all_phenos,
        'phenotype_stats':  phenotype_stats,
        'overall_stats':    overall_stats,
        'metric':           metric,
    }


# ─────────────────────────────────────────────────────────────
# Public plotting entry point
# ─────────────────────────────────────────────────────────────

def plot_figure4e(
    results: Dict,
    output_dir: Optional[str] = None,
) -> List[plt.Figure]:
    """
    Create all 4 figure 4E plots.
    If output_dir is given, saves and closes each figure automatically.
    Returns list of Figure objects (closed if output_dir was provided).
    """
    figs = [
        _plot_overall_scatter(results),
        _plot_per_phenotype_grid(results),
        _plot_distributions(results),
        _plot_correlations(results),
    ]

    if output_dir is not None:
        out = Path(output_dir)
        names = [
            "figure4e_overall_scatter.png",
            "figure4e_per_phenotype_grid.png",
            "figure4e_distributions.png",
            "figure4e_correlations.png",
        ]
        for fig, name in zip(figs, names):
            fig.savefig(out / name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved figure 4E to {out}")

    return figs


# ─────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────

def _phenotype_colors(phenotype_labels):
    unique = sorted(set(phenotype_labels))
    n = len(unique)
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.gist_rainbow
    return {p: cmap(i / max(n - 1, 1)) for i, p in enumerate(unique)}


def _plot_overall_scatter(results: Dict) -> plt.Figure:
    jac    = results['jaccard_scores']
    lod    = results['lodo_scores']
    phens  = results['phenotype_labels']
    ov     = results['overall_stats']
    metric = results['metric']

    colors = _phenotype_colors(phens)
    point_colors = [colors[p] for p in phens]

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.scatter(jac, lod, c=point_colors, alpha=0.6, s=50,
               edgecolors='black', linewidths=0.3)

    x_line = np.linspace(jac.min(), jac.max(), 200)
    for p, c in colors.items():
        mask = np.array(phens) == p
        if mask.sum() > 2:
            z = np.polyfit(jac[mask], lod[mask], 1)
            ax.plot(x_line, np.poly1d(z)(x_line),
                    color=c, alpha=0.6, linewidth=1.5, label=p)

    z_ov = np.polyfit(jac, lod, 1)
    ax.plot(x_line, np.poly1d(z_ov)(x_line),
            'k-', linewidth=3, alpha=0.9, label='Overall fit', zorder=10)

    ax.text(0.05, 0.97,
            f"N phenotypes = {ov['n_phenotypes']}\n"
            f"N pairs = {ov['n_total_pairs']}\n"
            f"Pearson r = {ov['pearson_r']:.3f}  (p={ov['pearson_p']:.2e})\n"
            f"Spearman ρ = {ov['spearman_r']:.3f}  (p={ov['spearman_p']:.2e})",
            transform=ax.transAxes, fontsize=10, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    ax.set_xlabel('Jaccard Similarity (SHAP top-10 features)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Pairwise LODO {metric.upper()}', fontsize=13, fontweight='bold')
    ax.set_title('Feature Similarity vs Cross-Dataset Performance\n(All Phenotypes)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=9, title='Phenotypes', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_per_phenotype_grid(results: Dict) -> plt.Figure:
    jac    = results['jaccard_scores']
    lod    = results['lodo_scores']
    phens  = results['phenotype_labels']
    pstats = results['phenotype_stats']
    metric = results['metric']

    unique = sorted(set(phens))
    n = len(unique)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    colors = _phenotype_colors(phens)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()

    for idx, pheno in enumerate(unique):
        ax = axes[idx]
        mask = np.array(phens) == pheno
        pj, pl = jac[mask], lod[mask]

        ax.scatter(pj, pl, alpha=0.6, s=80,
                   c=colors[pheno], edgecolors='black', linewidths=0.5)

        if len(pj) > 2:
            z = np.polyfit(pj, pl, 1)
            x_line = np.linspace(pj.min(), pj.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, alpha=0.8)
            if pheno in pstats:
                s = pstats[pheno]
                ax.text(0.05, 0.97,
                        f"N={s['n_pairs']}\nr={s['pearson_r']:.3f}\nρ={s['spearman_r']:.3f}",
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Jaccard Similarity', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'LODO {metric.upper()}', fontsize=10, fontweight='bold')
        ax.set_title(pheno, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Jaccard Similarity vs LODO Performance — Per Phenotype',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_distributions(results: Dict) -> plt.Figure:
    jac    = results['jaccard_scores']
    lod    = results['lodo_scores']
    ov     = results['overall_stats']
    metric = results['metric']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(jac, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(ov['mean_jaccard'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean = {ov['mean_jaccard']:.3f}")
    axes[0].set_xlabel('Jaccard Similarity', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Jaccard Similarities', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(lod, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(ov['mean_lodo'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean = {ov['mean_lodo']:.3f}")
    axes[1].set_xlabel(f'LODO {metric.upper()}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Distribution of LODO {metric.upper()}', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Metric Distributions Across All Phenotypes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_correlations(results: Dict) -> plt.Figure:
    pstats = results['phenotype_stats']
    ov     = results['overall_stats']

    phenos  = sorted(pstats.keys())
    pearson = [pstats[p]['pearson_r']  for p in phenos]
    spear   = [pstats[p]['spearman_r'] for p in phenos]
    x = np.arange(len(phenos))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, max(6, len(phenos) * 0.7 + 2)))

    bars1 = ax.barh(x - w / 2, pearson, w, label='Pearson r',  alpha=0.8)
    bars2 = ax.barh(x + w / 2, spear,   w, label='Spearman ρ', alpha=0.8)

    for bars in (bars1, bars2):
        for bar in bars:
            bar.set_color('seagreen' if bar.get_width() >= 0 else 'tomato')

    ax.set_yticks(x)
    ax.set_yticklabels(phenos, fontsize=10)
    ax.set_xlabel('Correlation coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Jaccard–LODO Correlation by Phenotype', fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(ov['pearson_r'], color='navy', linestyle='--', linewidth=2, alpha=0.7,
               label=f"Overall Pearson = {ov['pearson_r']:.3f}")
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig
