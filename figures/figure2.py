"""
Figure 2: Assembled overview figure.

  2A — Spearman correlation heatmap (confounders vs performance metrics)
  2B — ROC curves: zero-% baseline vs LightGBM LODO (CD in main figure)
  2C — Zero-count distributions Case vs Control (CD in main figure)
  2D — Pairwise Wasserstein distance heatmap (CD in main figure)
  2E — |ΔAUC| vs SHAP Jaccard scatter (CD in main figure)

Main figure
-----------
    assemble_figure2(csv_path, cd_microbiome_dfs, cd_target_dfs, cd_dataset_names,
                     figure2e_data, cd_phenotype_name)
    Stats CSVs are saved automatically to figures_out/figure_2/2a–2e/.

Supplementary (one PDF per phenotype)
--------------------------------------
    run_figure2b_supp(phenotypes, data_root, figures_dir)
    run_figure2c_supp(phenotypes, data_root, figures_dir, normalization_approach)
    run_figure2d_supp(phenotypes, data_root, figures_dir)
    run_figure2e_supp(phenotypes, data_root, figures_dir)
    run_figure2e_supp(phenotypes, data_root, figures_dir)
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
# Zero-count distribution helper (embeddable)
# ─────────────────────────────────────────────────────────────

def _plot_zero_distributions(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    ax=None,
) -> plt.Figure:
    """
    Grid of zero-count histograms (Case vs Control) for each dataset,
    laid out inside a single axes area via a nested GridSpec.
    One subplot per dataset, up to 3 per row.
    """
    import matplotlib.gridspec as mgridspec

    n     = len(dataset_names)
    ncols = min(3, n)
    nrows = max(1, int(np.ceil(n / ncols)))

    _standalone = ax is None
    if _standalone:
        fig = plt.figure(figsize=(7 * ncols, 5 * nrows))
        gs  = mgridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.35)
    else:
        fig = ax.figure
        ax.set_visible(False)           # hide the placeholder ax; use its bbox for sub-axes
        bb  = ax.get_position()
        gs  = mgridspec.GridSpec(
            nrows, ncols, figure=fig,
            left=bb.x0, right=bb.x1, bottom=bb.y0, top=bb.y1,
            hspace=0.5, wspace=0.35,
        )

    mw_rows = []
    for i, (mb, tgt, name) in enumerate(zip(microbiome_dfs, target_dfs, dataset_names)):
        sub_ax = fig.add_subplot(gs[i // ncols, i % ncols])
        zero_counts = (mb.values == 0).sum(axis=1)
        y = tgt.values.ravel()

        mw_p = np.nan
        if len(np.unique(y)) >= 2:
            z0, z1 = zero_counts[y == 0], zero_counts[y == 1]
            sub_ax.hist([z0, z1], bins=20, alpha=0.6,
                        label=["Control", "Case"],
                        color=["lightcoral", "lightblue"],
                        edgecolor="black", linewidth=0.5)
            sub_ax.legend(fontsize=9)
            if len(z0) > 0 and len(z1) > 0:
                _, mw_p = stats.mannwhitneyu(z0, z1, alternative="two-sided")
                sub_ax.text(0.97, 0.95, f"MW p={mw_p:.4f}",
                            transform=sub_ax.transAxes, ha="right", va="top",
                            fontsize=9,
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
        else:
            sub_ax.hist(zero_counts, bins=20, alpha=0.7,
                        color="lightgray", edgecolor="black")

        sub_ax.set_xlabel("Zeros per Sample", fontsize=10, fontweight="bold")
        sub_ax.set_ylabel("Count",             fontsize=10, fontweight="bold")
        sub_ax.set_title(name, fontsize=11, fontweight="bold")
        sub_ax.grid(True, alpha=0.3, linestyle="--")
        mw_rows.append({"dataset": name, "mw_pvalue": mw_p})

    mw_df = pd.DataFrame(mw_rows)

    if _standalone:
        plt.tight_layout()
    return fig, mw_df


# ─────────────────────────────────────────────────────────────
# LODO ROC helper
# ─────────────────────────────────────────────────────────────

def _lodo_roc(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> Dict:
    """
    Run LODO with LightGBM and return per-dataset ROC curve data.
    Captures predicted probabilities directly so ROC curves can be plotted.
    """
    import lightgbm as lgb
    from evaluation.data_loading import combine_datasets

    auc_scores, roc_curves_ = [], []

    for test_idx, name in enumerate(dataset_names):
        train_indices = [i for i in range(len(dataset_names)) if i != test_idx]
        if not train_indices:
            auc_scores.append(np.nan)
            roc_curves_.append((np.array([]), np.array([])))
            continue

        X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
        X_test  = microbiome_dfs[test_idx]
        y_test  = target_dfs[test_idx]

        common = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common], X_test[common]

        params = {
            'objective': 'binary', 'metric': 'auc',
            'boosting_type': 'gbdt', 'num_leaves': 31,
            'learning_rate': 0.05, 'feature_fraction': 0.9,
            'bagging_fraction': 0.8, 'bagging_freq': 5,
            'verbose': -1, 'random_state': 42,
        }
        try:
            train_data = lgb.Dataset(X_train, label=y_train.values.ravel())
            valid_data = lgb.Dataset(X_test,  label=y_test.values.ravel())
            model = lgb.train(params, train_data, valid_sets=[valid_data],
                              num_boost_round=1000,
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            y_prob = model.predict(X_test, num_iteration=model.best_iteration)
            y_true = y_test.values.ravel()

            if len(np.unique(y_true)) < 2:
                raise ValueError("single class in test set")

            auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_scores.append(auc)
            roc_curves_.append((fpr, tpr))
        except Exception:
            auc_scores.append(np.nan)
            roc_curves_.append((np.array([]), np.array([])))

    return {
        'dataset_names': dataset_names,
        'auc_scores':    auc_scores,
        'roc_curves':    roc_curves_,
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
    show_title: bool = True,
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
        hspace=0.6, wspace=0.35,
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

    if show_title:
        fig.suptitle(f'{phenotype_name}{label_suffix}', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────
# Public interface — Figure 2B: Confounder correlation matrix
# ─────────────────────────────────────────────────────────────

_CORR_COLUMNS = [
    'sample_count', 'feature_count',
    'microbes_unique_test',
    'microbes_unique_train',
    'jaccard_index',
    'bray_curtis_lodo',
    'leave_one_out_auc',
    'internal_validation_auc',
    'within_dataset_auc',
    'age_median',
    'male_percentage',
]

_CORR_PRETTY = {
    'sample_count':           'Sample\nCount',
    'feature_count':          'Microbes\nCount',
    'microbes_unique_test':   'Microbes\nUnique Test',
    'microbes_unique_train':  'Microbes\nUnique Train',
    'jaccard_index':          'Jaccard\nLODO',
    'bray_curtis_lodo':       'Bray-Curtis\nLODO',
    'leave_one_out_auc':      'LODO',
    'internal_validation_auc':'Mixed\nDataset AUC',
    'within_dataset_auc':     'Within\nDataset AUC',
    'age_median':             'Age\nMedian',
    'male_percentage':        'Male\n%',
}

# Backward-compat rename for CSVs generated before column renaming
_LEGACY_RENAME = {
    'microbes_in_current_not_in_others_pct': 'microbes_unique_test',
    'microbes_in_others_not_in_current_pct': 'microbes_unique_train',
    'similarity_score_all':     'jaccard_index',
    'similarity_score_control': 'jaccard_index_control',
    'similarity_score_test':    'jaccard_index_case',
    'single_dataset_auc':       'within_dataset_auc',
}


def plot_figure2a(
    csv_path: str,
    alpha: float = 0.05,
    ax=None,
) -> plt.Figure:
    """
    Spearman correlation heatmap of dataset-level metadata + performance metrics.

    Lower triangle: Spearman r.  Upper triangle: p-value.
    Significant cells (p < alpha) are outlined in black.

    Parameters
    ----------
    csv_path : path to Data/microbiome_analysis_results.csv
    alpha    : significance threshold for border highlighting
    ax       : existing Axes to draw into; if None a new figure is created
    """
    import seaborn as sns
    from scipy.stats import spearmanr

    df = pd.read_csv(csv_path)
    df = df.rename(columns=_LEGACY_RENAME)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('N/A', np.nan)

    cols = [c for c in _CORR_COLUMNS if c in df.columns]
    num_df = df[cols].copy()
    for c in cols:
        num_df[c] = pd.to_numeric(num_df[c], errors='coerce')

    from statsmodels.stats.multitest import multipletests

    n = len(cols)
    corr_mat = np.full((n, n), np.nan)
    p_mat    = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_mat[i, j] = 1.0
                p_mat[i, j]    = 0.0
                continue
            valid = num_df[[cols[i], cols[j]]].dropna()
            if len(valid) >= 3:
                r, p = spearmanr(valid[cols[i]], valid[cols[j]])
                corr_mat[i, j] = r
                p_mat[i, j]    = p

    # BH FDR correction across all upper-triangle pairs
    q_mat = np.full((n, n), np.nan)
    upper_idx = [(i, j) for i in range(n) for j in range(n) if i < j
                 and not np.isnan(p_mat[i, j])]
    if upper_idx:
        raw_ps = [p_mat[i, j] for i, j in upper_idx]
        _, q_vals, _, _ = multipletests(raw_ps, method="fdr_bh")
        for (i, j), q in zip(upper_idx, q_vals):
            q_mat[i, j] = q
            q_mat[j, i] = q  # mirror for lower-triangle significance check

    # Annotations: lower triangle = r, upper = q-value (FDR corrected)
    annot = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            if i < j:
                annot[i, j] = f'q={q_mat[i, j]:.2f}' if not np.isnan(q_mat[i, j]) else ''
            else:
                annot[i, j] = f'{corr_mat[i, j]:.2f}'

    pretty = [_CORR_PRETTY.get(c, c.replace('_', ' ').title()) for c in cols]

    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(26, 22))
    else:
        fig = ax.figure

    hm = sns.heatmap(
        corr_mat,
        annot=annot, fmt='',
        cmap='coolwarm_r', center=0,
        cbar=_standalone,
        cbar_kws={'shrink': 0.5} if _standalone else {},
        annot_kws={'size': 96},
        ax=ax,
    )
    hm.set_xticklabels(pretty, fontsize=122, rotation=45, ha='right')
    hm.set_yticklabels(pretty, fontsize=122, rotation=0)

    # Black border on significant lower-triangle cells (q < alpha)
    for i in range(n):
        for j in range(n):
            if i > j and not np.isnan(q_mat[i, j]) and q_mat[i, j] < alpha:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='black', lw=3))

    # Build long-format stats table (lower triangle only)
    stats_rows = []
    for i in range(n):
        for j in range(n):
            if i > j:
                stats_rows.append({
                    "var1":        cols[i],
                    "var2":        cols[j],
                    "var1_pretty": pretty[i],
                    "var2_pretty": pretty[j],
                    "spearman_r":  corr_mat[i, j],
                    "p_value":     p_mat[i, j],
                    "q_value":     q_mat[i, j],
                    "significant": bool(not np.isnan(q_mat[i, j]) and q_mat[i, j] < alpha),
                })
    stats_df = pd.DataFrame(stats_rows)

    if _standalone:
        plt.tight_layout()
    return fig, stats_df


def plot_figure2b(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str = "",
    ax=None,
    show_legend: bool = True,
    suppress_xlabel: bool = False,
    inner_width_ratios: list = None,
    fontsize_scale: float = 1.0,
    legend_fontsize_scale: float = 1.0,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Left panel  : ROC curves for the zero-percentage predictor (solid, bold).
    Right panel : Horizontal bars showing LightGBM LODO AUC per dataset.
    Colors are shared between panels.
    Returns (fig, auc_df) where auc_df has columns: dataset, predictor, auc.
    """
    zero_res = _zeros_pct_roc(microbiome_dfs, target_dfs, dataset_names)
    lodo_res = _lodo_roc(microbiome_dfs, target_dfs, dataset_names)

    # Colors matching figure 4D palette (box fill = primary, strip = darker secondary)
    _square   = ["#c49a94", "#c5aee0", "#F4A36A", "#8FD17E",
                 "#5a2a24", "#5a3a7a", "#b5581e", "#2a7a2a"]
    color_map = {name: _square[i % len(_square)] for i, name in enumerate(dataset_names)}

    _standalone = ax is None
    if _standalone:
        fig, (ax_roc, ax_bar) = plt.subplots(
            1, 2, figsize=(34, 12), gridspec_kw={"width_ratios": [2.2, 1]}
        )
    else:
        import matplotlib.gridspec as _mgs
        fig = ax.figure
        ax.set_axis_off()   # hide frame/ticks but keep text (e.g. panel letter) visible
        _bar_ratios = inner_width_ratios if inner_width_ratios is not None else [2.2, 1]
        gs_inner = _mgs.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=ax.get_subplotspec(),
            width_ratios=_bar_ratios, wspace=0.3,
        )
        ax_roc = fig.add_subplot(gs_inner[0, 0])
        ax_bar = fig.add_subplot(gs_inner[0, 1])

    # ── Left: zero-% ROC curves (solid, bold) ───────────────
    for i, name in enumerate(dataset_names):
        fpr_z, tpr_z, _ = zero_res["roc_curves"][i]
        auc_z            = zero_res["auc_scores"][i]
        if len(fpr_z) and not np.isnan(auc_z):
            ax_roc.plot(fpr_z, tpr_z, color=color_map[name], linewidth=14.0,
                        linestyle="-", label=f"{name} (AUC={auc_z:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k:", linewidth=2, alpha=0.6, label="Random")
    ax_roc.set_xlim(0, 1);  ax_roc.set_ylim(0, 1.05)
    ax_roc.set_xlabel("False Positive Rate", fontsize=132*fontsize_scale, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate",  fontsize=132*fontsize_scale, fontweight="bold")
    ax_roc.set_title("Sparsity-Based Classification", fontsize=130*fontsize_scale, fontweight="bold")
    ax_roc.tick_params(labelsize=138*fontsize_scale)
    ax_roc.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: "" if x == 0 else f"{x:.1f}"))
    ax_roc.legend(loc="lower right", fontsize=90*fontsize_scale*legend_fontsize_scale,
                  framealpha=0.9,
                  title="Dataset (AUC)", title_fontsize=88*fontsize_scale*legend_fontsize_scale)
    ax_roc.grid(True, alpha=0.25, linestyle="--")

    # ── Right: LODO AUC horizontal bars ─────────────────────
    lodo_aucs = [lodo_res["auc_scores"][i] for i in range(len(dataset_names))]
    y_pos     = list(range(len(dataset_names)))
    bars      = ax_bar.barh(
        y_pos, lodo_aucs,
        color=[color_map[n] for n in dataset_names],
        edgecolor="black", linewidth=1.2, alpha=0.9,
    )
    ax_bar.axvline(0.5, color="red", linestyle="-", linewidth=5, alpha=1.0, zorder=10, label="Random (0.5)")
    ax_bar.spines["left"].set_color("red")
    ax_bar.spines["left"].set_linewidth(5)
    for bar, auc in zip(bars, lodo_aucs):
        if not np.isnan(auc):
            ax_bar.text(auc + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{auc:.2f}", va="center", ha="left",
                        fontsize=100*fontsize_scale)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("LODO AUC", fontsize=132*fontsize_scale, fontweight="bold")
    ax_bar.set_title("Full Classification", fontsize=130*fontsize_scale, fontweight="bold")
    ax_bar.set_xlim(0.5, 1.07)
    ax_bar.tick_params(axis="x", labelsize=138*fontsize_scale)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: "" if x == 0.5 else f"{x:.1f}"))
    ax_bar.legend(fontsize=90*fontsize_scale, loc="lower right")
    ax_bar.grid(True, alpha=0.25, axis="x", linestyle="--")

    if not show_legend:
        for _ax in (ax_roc, ax_bar):
            lgnd = _ax.get_legend()
            if lgnd:
                lgnd.remove()

    if suppress_xlabel:
        for _ax in (ax_roc, ax_bar):
            _ax.set_xlabel("")
            _ax.tick_params(axis="x", labelbottom=False)

    if _standalone and phenotype_name:
        fig.suptitle(phenotype_name, fontsize=42, fontweight="bold")
        plt.tight_layout()

    auc_rows = []
    for i, name in enumerate(dataset_names):
        auc_rows.append({"dataset": name, "predictor": "zero_pct_baseline",
                         "auc": zero_res["auc_scores"][i]})
        auc_rows.append({"dataset": name, "predictor": "lodo_lightgbm",
                         "auc": lodo_res["auc_scores"][i]})

    return fig, pd.DataFrame(auc_rows)


def run_figure2b_supp(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
) -> None:
    """
    Supplementary: full combined figure per phenotype (ROC curves + LODO barplot
    + zero-count distributions), same layout as the original figure 2.
    Datasets with only one class in target are dropped automatically.
    Output files: supp_figure2b_<phenotype>_<dtype>.pdf
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = Path(data_root) / pheno_str
        if not folder.exists():
            print(f"  Skipping {pheno_str}: folder not found")
            continue
        try:
            mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        filtered = [
            (mb, tgt, name)
            for mb, tgt, name in zip(mb_dfs, tgt_dfs, ds_names)
            if len(np.unique(tgt.values.ravel())) >= 2
        ]
        if not filtered:
            print(f"  Skipping {pheno_str}: no datasets with both classes")
            continue
        dropped = set(ds_names) - {t[2] for t in filtered}
        if dropped:
            print(f"  Dropped single-class datasets: {dropped}")
        mb_dfs, tgt_dfs, ds_names = (list(x) for x in zip(*filtered))

        if len(ds_names) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets after filtering")
            continue

        safe     = pheno_str.replace(" ", "_")
        out_path = out / f"supp_figure2b_{safe}.pdf"
        if out_path.exists():
            print(f"  Skipping supp 2B for {pheno_str}: already saved")
            continue
        print(f"  Building supp 2B for {pheno_str}")
        fig = _make_figure2(mb_dfs, tgt_dfs, ds_names, pheno_str)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved supp 2B for {pheno_str}")


def compute_figure2d_data(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    cache_path: str = "figures_out/figure_2/2d/ks_fractions.csv",
    normalization_approach=None,
) -> pd.DataFrame:
    """
    For each phenotype, compute the fraction of shared taxa that have at least
    one significant inter-dataset KS difference (Benjamini-Hochberg FDR corrected).

    Caches result to cache_path; loads from cache on re-run.
    When normalization_approach is provided, data is normalized before computing KS stats.

    Returns DataFrame with columns:
        phenotype, dtype, n_shared_features, n_datasets, ks_sig_fraction
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from scipy.stats import ks_2samp
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations as _comb

    cache = Path(cache_path)
    if cache.exists():
        print(f"  Loading cached 2D KS data from {cache}")
        return pd.read_csv(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_rows = []

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = Path(data_root) / pheno_str
        if not folder.exists():
            continue
        try:
            mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        filtered = [
            (mb, tgt, n) for mb, tgt, n
            in zip(mb_dfs, tgt_dfs, ds_names)
            if len(np.unique(tgt.values.ravel())) >= 2
        ]
        if len(filtered) < 2:
            continue
        mb_dfs, tgt_dfs_f, ds_names = (list(x) for x in zip(*filtered))

        if normalization_approach is not None:
            try:
                from src.rankbird.normalization.pipeline import apply_normalization_pipeline
                name2tgt = {n: t for n, t in zip(ds_names, tgt_dfs_f)}
                mb_dfs, ds_names = apply_normalization_pipeline(
                    list(mb_dfs), list(ds_names), min_samples_per_dataset=0,
                )
                ds_names = [n for n in ds_names if n in name2tgt]
                mb_dfs   = mb_dfs[:len(ds_names)]
            except Exception as e:
                print(f"  Normalization failed for {pheno_str}: {e}")

        if len(mb_dfs) < 2:
            continue

        common = mb_dfs[0].columns
        for df in mb_dfs[1:]:
            common = common.intersection(df.columns)
        if len(common) == 0:
            continue

        ks_pvals, ks_stats, feat_labels, ds1_labels, ds2_labels = [], [], [], [], []
        for feat in common:
            for i, j in _comb(range(len(mb_dfs)), 2):
                stat, p = ks_2samp(mb_dfs[i][feat].values, mb_dfs[j][feat].values)
                ks_pvals.append(p)
                ks_stats.append(stat)
                feat_labels.append(feat)
                ds1_labels.append(ds_names[i])
                ds2_labels.append(ds_names[j])

        if not ks_pvals:
            continue

        reject, p_adj, _, _ = multipletests(ks_pvals, method="fdr_bh")
        sig_per_feat = pd.Series(reject, index=feat_labels).groupby(level=0).max()
        frac_sig = float(sig_per_feat.mean())

        rows.append({
            "phenotype":          pheno_str,
            "dtype":              dtype,
            "n_shared_features":  int(len(common)),
            "n_datasets":         int(len(ds_names)),
            "ks_sig_fraction":    frac_sig,
        })

        for feat, d1, d2, stat, p, padj, sig in zip(
            feat_labels, ds1_labels, ds2_labels,
            ks_stats, ks_pvals, p_adj, reject,
        ):
            detail_rows.append({
                "phenotype":    pheno_str,
                "dtype":        dtype,
                "feature":      feat,
                "dataset_1":    d1,
                "dataset_2":    d2,
                "ks_statistic": round(float(stat), 6),
                "p_value":      float(p),
                "p_adjusted":   float(padj),
                "significant":  bool(sig),
            })

        print(f"  {pheno_str}: {frac_sig*100:.1f}% of {len(common)} shared features significant")

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)

    detail_path = cache.parent / (cache.stem + "_per_feature.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    print(f"  Saved per-feature KS details → {detail_path}")
    return df


def compute_ks_summary_stats(
    ks_cache_path: str,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Compute summary statistics from the KS fractions CSV for paper reporting.

    Reports: overall range, overall median, median per dtype (Amplicon / Metagenomics).
    Saves to output_path (defaults to same dir as ks_cache_path, named ks_summary_stats.csv).

    Returns a DataFrame with one row per statistic.
    """
    cache = Path(ks_cache_path)
    if not cache.exists():
        print(f"  KS fractions file not found: {cache}")
        return pd.DataFrame()

    df = pd.read_csv(cache)
    df["ks_pct"] = df["ks_sig_fraction"] * 100

    stats = []

    overall_min    = df["ks_pct"].min()
    overall_max    = df["ks_pct"].max()
    overall_median = df["ks_pct"].median()
    stats.append({"stat": "range_min_pct",     "value": round(overall_min,    1)})
    stats.append({"stat": "range_max_pct",     "value": round(overall_max,    1)})
    stats.append({"stat": "median_overall_pct","value": round(overall_median, 1)})

    for dtype_label, dtype_key in [("16S (Amplicon)", "Amplicon"), ("WGS (Metagenomics)", "Metagenomics")]:
        sub = df[df["dtype"] == dtype_key]["ks_pct"]
        if len(sub) > 0:
            stats.append({"stat": f"median_{dtype_key.lower()}_pct", "value": round(sub.median(), 1),
                          "dtype": dtype_label, "n_phenotypes": len(sub)})
            stats.append({"stat": f"min_{dtype_key.lower()}_pct",    "value": round(sub.min(),    1),
                          "dtype": dtype_label, "n_phenotypes": len(sub)})
            stats.append({"stat": f"max_{dtype_key.lower()}_pct",    "value": round(sub.max(),    1),
                          "dtype": dtype_label, "n_phenotypes": len(sub)})

    stats_df = pd.DataFrame(stats)

    out = Path(output_path) if output_path else cache.parent / "ks_summary_stats.csv"
    stats_df.to_csv(out, index=False)

    print(f"\n  KS fraction summary stats:")
    print(f"    Range:           {overall_min:.1f}% – {overall_max:.1f}%")
    print(f"    Median (all):    {overall_median:.1f}%")
    for dtype_key in ["Amplicon", "Metagenomics"]:
        sub = df[df["dtype"] == dtype_key]["ks_pct"]
        if len(sub) > 0:
            print(f"    Median {dtype_key}: {sub.median():.1f}%  (n={len(sub)} phenotypes)")
    print(f"  Saved → {out}")

    return stats_df


def compute_ks_lodo_correlation(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    ks_per_feature_path: str = "figures_out/figure_2/2d/ks_fractions_per_feature.csv",
    output_dir: str = "figures_out/figure_2/2d",
    lodo_cache_path: str = "figures_out/figure_2/2d/lodo_aucs_per_dataset.csv",
) -> pd.DataFrame:
    """
    Correlate per-dataset mean KS distance with LODO AUC.

    For each (phenotype, dataset), computes the mean KS statistic across all
    pairwise comparisons involving that dataset (i.e., how different it is from
    every other dataset on average), then correlates with the LODO AUC when
    that dataset is the held-out test set.

    Saves two files to output_dir:
      ks_lodo_per_dataset.csv   — one row per (phenotype, dataset)
      ks_lodo_correlation.csv   — Spearman r, Pearson r, p-values

    Returns the per-dataset DataFrame.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from evaluation.learning_protocols import lodo_protocol
    from scipy.stats import spearmanr, pearsonr

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_path = out_dir / "ks_lodo_per_dataset.csv"
    stats_path       = out_dir / "ks_lodo_correlation.csv"

    # ── Load per-feature KS data ──────────────────────────────
    ks_path = Path(ks_per_feature_path)
    if not ks_path.exists():
        print(f"  Per-feature KS file not found: {ks_path}")
        print("  Run compute_figure2d_data first to generate it.")
        return pd.DataFrame()

    ks_df = pd.read_csv(ks_path)

    # ── Load or compute LODO AUC per dataset ─────────────────
    lodo_cache = Path(lodo_cache_path)
    if lodo_cache.exists():
        print(f"  Loading cached LODO AUCs from {lodo_cache}")
        lodo_df = pd.read_csv(lodo_cache)
    else:
        lodo_rows = []
        for phenotype, dtype in phenotypes:
            pheno_str = f"{phenotype} {dtype}"
            folder    = Path(data_root) / pheno_str
            if not folder.exists():
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
            except Exception as e:
                print(f"  Error loading {pheno_str}: {e}")
                continue
            filtered = [
                (mb, tgt, n) for mb, tgt, n in zip(mb_dfs, tgt_dfs, ds_names)
                if len(np.unique(tgt.values.ravel())) >= 2
            ]
            if len(filtered) < 2:
                continue
            mb_f, tgt_f, names_f = (list(x) for x in zip(*filtered))
            try:
                res = lodo_protocol(mb_f, tgt_f, names_f)
                for _, row in res.iterrows():
                    lodo_rows.append({
                        "phenotype":  pheno_str,
                        "dtype":      dtype,
                        "dataset":    row["test_dataset"],
                        "lodo_auc":   float(row["auc"]),
                    })
            except Exception as e:
                print(f"  LODO failed for {pheno_str}: {e}")

        lodo_df = pd.DataFrame(lodo_rows)
        lodo_df.to_csv(lodo_cache, index=False)
        print(f"  Saved LODO AUCs → {lodo_cache}")

    if lodo_df.empty:
        print("  No LODO results available.")
        return pd.DataFrame()

    # ── Compute per-dataset mean KS distance ─────────────────
    # For each (phenotype, dataset): average KS statistic over all pairs
    # involving that dataset (across all features).
    ks_long = pd.concat([
        ks_df[["phenotype", "dataset_1", "ks_statistic"]].rename(columns={"dataset_1": "dataset"}),
        ks_df[["phenotype", "dataset_2", "ks_statistic"]].rename(columns={"dataset_2": "dataset"}),
    ], ignore_index=True)
    mean_ks = (
        ks_long
        .groupby(["phenotype", "dataset"])["ks_statistic"]
        .mean()
        .reset_index()
        .rename(columns={"ks_statistic": "mean_ks_distance"})
    )

    # ── Merge ─────────────────────────────────────────────────
    merged = pd.merge(mean_ks, lodo_df, on=["phenotype", "dataset"], how="inner")
    if merged.empty:
        print("  No matching (phenotype, dataset) pairs between KS and LODO data.")
        return pd.DataFrame()

    merged.to_csv(per_dataset_path, index=False)
    print(f"  Saved per-dataset KS vs LODO AUC → {per_dataset_path}")
    print(f"  {len(merged)} data points across {merged['phenotype'].nunique()} phenotypes")

    # ── Correlation ───────────────────────────────────────────
    x = merged["mean_ks_distance"].values
    y = merged["lodo_auc"].values

    sp_r, sp_p = spearmanr(x, y)
    pe_r, pe_p = pearsonr(x, y)

    stats_rows = [
        {"metric": "Spearman r",  "value": round(sp_r, 4), "p_value": round(sp_p, 6)},
        {"metric": "Pearson r",   "value": round(pe_r, 4), "p_value": round(pe_p, 6)},
        {"metric": "n_datapoints", "value": len(merged),   "p_value": np.nan},
        {"metric": "n_phenotypes", "value": merged["phenotype"].nunique(), "p_value": np.nan},
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  KS distance × LODO AUC correlation:")
    print(f"    Spearman r = {sp_r:.4f}  (p = {sp_p:.4e})")
    print(f"    Pearson  r = {pe_r:.4f}  (p = {pe_p:.4e})")
    print(f"    n = {len(merged)} dataset observations")
    print(f"  Saved stats → {stats_path}")

    return merged


def plot_figure2d_ks_bars(
    data_df: pd.DataFrame,
    ax=None,
    fontsize_scale: float = 1.0,
) -> plt.Figure:
    """
    Horizontal bar chart: fraction of shared taxa with significant inter-dataset
    KS difference (FDR-corrected) per phenotype. Bars colored by dtype.
    """
    _standalone = ax is None

    if data_df is None or data_df.empty:
        if _standalone:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No KS data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=16, color="#888")
        ax.set_axis_off()
        return fig

    data_df = data_df.sort_values("ks_sig_fraction", ascending=True).reset_index(drop=True)

    dtype_colors = {"Metagenomics": "lightblue", "Amplicon": "lightcoral"}
    colors = [dtype_colors.get(d, "gray") for d in data_df["dtype"]]

    if _standalone:
        fig, ax = plt.subplots(figsize=(12, max(5, len(data_df) * 0.7)))
    else:
        fig = ax.figure

    y_pos = list(range(len(data_df)))
    bars  = ax.barh(y_pos, data_df["ks_sig_fraction"] * 100,
                    color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)

    for bar, (_, row) in zip(bars, data_df.iterrows()):
        bar_w = bar.get_width()
        if bar_w > 8:
            ax.text(bar_w - 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['ks_sig_fraction']*100:.0f}%",
                    va="center", ha="right", fontsize=124*fontsize_scale, color="black")
        else:
            ax.text(bar_w + 1.0,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['ks_sig_fraction']*100:.0f}%",
                    va="center", ha="left", fontsize=124*fontsize_scale)

    _dtype_display = {"Metagenomics": "WGS", "Amplicon": "16S"}
    labels = [
        r["phenotype"]
        .replace(" Metagenomics", "")
        .replace(" Amplicon", "")
        for _, r in data_df.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=124*fontsize_scale)
    ax.set_xlabel(
        "% Shared Taxa with Significant\nInter-Dataset Difference (KS, FDR)",
        fontsize=124*fontsize_scale, fontweight="bold",
    )
    ax.set_xlim(0, 108)
    ax.tick_params(axis="x", labelsize=120*fontsize_scale)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=_dtype_display.get(d, d))
        for d, c in dtype_colors.items()
        if d in data_df["dtype"].values
    ]
    ax.legend(handles=legend_elements, fontsize=134*fontsize_scale, loc="lower right")

    if _standalone:
        plt.tight_layout()
    return fig


def _plot_umap_raw_panel(
    ax,
    mb_by_dtype: Dict,
    tgt_by_dtype: Dict,
    names_by_dtype: Dict,
    phenotype_name: str = "",
) -> None:
    """Draw the raw UMAP/PCA scatter (left panel of Figure 3D) into an existing ax."""
    from figures.figure3 import _umap_scatter, _DTYPE_COLORS, _CASE_EDGE

    dtypes_present = [d for d in ["Metagenomics", "Amplicon"] if d in mb_by_dtype]
    all_mb, all_tgt, all_dtype_labels = [], [], []
    for dtype in dtypes_present:
        for mb, tgt in zip(mb_by_dtype[dtype], tgt_by_dtype[dtype]):
            all_mb.append(mb)
            all_tgt.append(tgt)
            all_dtype_labels.extend([dtype] * len(mb))

    if len(all_mb) < 2:
        ax.text(0.5, 0.5, "Need ≥2 datasets for UMAP",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="#888")
        return

    common = all_mb[0].columns
    for df in all_mb[1:]:
        common = common.intersection(df.columns)
    if len(common) == 0:
        ax.text(0.5, 0.5, "No common features (raw)",
                ha="center", va="center", transform=ax.transAxes)
        return

    X = pd.concat([df[common] for df in all_mb], axis=0, ignore_index=True)
    tgt_arr = np.concatenate([
        t.iloc[:, 0].values if isinstance(t, pd.DataFrame) else t.values
        for t in all_tgt
    ])
    dt_arr = np.array(all_dtype_labels)

    title = f"Raw — {phenotype_name}" if phenotype_name else "Raw"
    _umap_scatter(ax, X, tgt_arr, dt_arr, title=title)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_DTYPE_COLORS.get(d, "#888"),
                   markeredgecolor="none", markersize=14, label=d)
        for d in dtypes_present
    ] + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888888", markeredgecolor=_CASE_EDGE,
                   markeredgewidth=1.5, markersize=14, label="Case (border)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888888", markeredgecolor="none",
                   markersize=14, label="Control (no border)"),
    ]
    ax.legend(handles=legend_handles, fontsize=24, framealpha=0.9, loc="best")


def assemble_figure2(
    csv_path: str,
    cd_microbiome_dfs: Optional[List[pd.DataFrame]] = None,
    cd_target_dfs: Optional[List[pd.DataFrame]] = None,
    cd_dataset_names: Optional[List[str]] = None,
    figure2d_data: Optional[pd.DataFrame] = None,
    figure2e_data: Optional[Dict] = None,   # kept for backward compat; unused in main figure
    cd_phenotype_name: str = "CD",
    shap_full_lodo: Optional[Dict] = None,
    shap_dataset_names: Optional[List[str]] = None,
    figsize: tuple = (140, 145),
) -> plt.Figure:
    """
    Assemble Figure 2 panels into one combined figure.

      Col 0 (left)  — A (row 0) / B (row 1) / C (row 2) — all same width
      Col 1 (right) — E spanning rows 0–1 (tall) / D (row 2)

    Parameters
    ----------
    csv_path            : path to microbiome_analysis_results.csv
    cd_microbiome_dfs   : microbiome DataFrames for the representative phenotype
    cd_target_dfs       : target DataFrames for the representative phenotype
    cd_dataset_names    : dataset names for the representative phenotype
    figure2d_data       : pre-computed DataFrame from compute_figure2d_data()
    figure2e_data       : unused in main figure (kept for backward compat)
    cd_phenotype_name   : label for the representative phenotype (default "CD")
    shap_full_lodo      : {dataset: {"top_10_features": df}} from _load_shap_details()
    shap_dataset_names  : ordered list of dataset names matching shap_full_lodo keys
    """
    import matplotlib.gridspec as mgridspec

    fig = plt.figure(figsize=figsize)
    gs = mgridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[2.5, 1.5],
        height_ratios=[3.2, 2.2, 2.0],
        hspace=0.42, wspace=0.38,
        left=0.07, right=0.97, top=0.96, bottom=0.05,
    )

    # ── Left column: A / B / C ────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    # ── Right column: E (rows 0–1, tall) / D (row 2) ─────────
    ax_e = fig.add_subplot(gs[0:2, 1])
    ax_d = fig.add_subplot(gs[2, 1])

    def _placeholder(ax, msg):
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()

    # ── Prepare output subfolders ─────────────────────────────
    subdirs = {}
    base = Path("figures_out") / "figure_2"
    for panel in ["2a", "2b", "2c", "2d", "2e"]:
        d = base / panel
        d.mkdir(parents=True, exist_ok=True)
        subdirs[panel] = d

    # ── Panel A ───────────────────────────────────────────────
    _, stats_2a = plot_figure2a(csv_path, ax=ax_a)
    ax_a.text(-0.17, 1.0, "A", transform=ax_a.transAxes,
              fontsize=172, fontweight="bold", va="top", ha="right", clip_on=False)
    stats_2a.to_csv(subdirs["2a"] / "spearman_correlations.csv", index=False)

    # ── Panel B: CD ROC curves ────────────────────────────────
    if cd_microbiome_dfs is not None:
        _, auc_2b = plot_figure2b(cd_microbiome_dfs, cd_target_dfs, cd_dataset_names,
                                   phenotype_name=cd_phenotype_name, ax=ax_b,
                                   inner_width_ratios=[2.1, 1.3])
        auc_2b.to_csv(subdirs["2b"] / f"roc_auc_{cd_phenotype_name}.csv", index=False)
    else:
        _placeholder(ax_b, "CD data not provided")
    ax_b.text(-0.17, 1.0, "B", transform=ax_b.transAxes,
              fontsize=172, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel C: CD microbiome feature heatmap ───────────────
    if cd_microbiome_dfs is not None:
        plot_figure2c(cd_microbiome_dfs, cd_target_dfs,
                      cd_dataset_names, phenotype_name="", ax=ax_c,
                      max_features=50)
    else:
        _placeholder(ax_c, "CD data not provided")
    ax_c.text(-0.17, 1.08, "C", transform=ax_c.transAxes,
              fontsize=172, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel D: KS fraction bar chart (all phenotypes) ──────
    if figure2d_data is not None:
        plot_figure2d_ks_bars(figure2d_data, ax=ax_d)
    else:
        _placeholder(ax_d, "figure2d_data not provided\n(run compute_figure2d_data first)")

    # ── Panel E: SHAP feature presence grid ──────────────────
    if shap_full_lodo and shap_dataset_names:
        from figures.pairwise_lodo_plots import _plot_shap_grid
        _plot_shap_grid(ax_e, shap_full_lodo, shap_dataset_names,
                        rename_features=True, top_n=10)
        ax_e.set_ylabel("Top SHAP Taxa Across Datasets",
                        fontsize=116, fontweight="bold",
                        rotation=270, va="center", labelpad=60)
    else:
        _placeholder(ax_e, "shap_full_lodo not provided\n(run figure 2F supp first)")
    ax_e.text(-0.17, 1.0, "E", transform=ax_e.transAxes,
              fontsize=172, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Extend D right edge to match E's tight bbox (inc. tick labels) ──
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb_e = ax_e.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    pos_d = ax_d.get_position()
    new_d_width = bb_e.x1 - pos_d.x0
    if new_d_width > pos_d.width:
        ax_d.set_position([pos_d.x0, pos_d.y0, new_d_width, pos_d.height])
        pos_d = ax_d.get_position()

    # ── Place D letter using figure coordinates ───────────────
    fig.text(pos_d.x0 - 0.02, pos_d.y1 + 0.005, "D",
             fontsize=172, fontweight="bold", va="bottom", ha="right", clip_on=False)

    return fig


# ─────────────────────────────────────────────────────────────
# Public interface — Figure 2C: Zero-% ROC + LODO + distributions
# ─────────────────────────────────────────────────────────────

def plot_figure2c(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str = "",
    max_features: int = 50,
    ax=None,
    fontsize_scale: float = 1.0,
    title_fontsize_scale: float = None,
) -> plt.Figure:
    """
    Microbiome feature presence/abundance heatmap (samples × top-varying features).

    Datasets are stacked vertically with separator lines. Features are the
    top-`max_features` by cross-dataset variance among common microbes.
    Datasets with only one class in target are dropped before plotting.
    Embeds into an existing axes when ax is provided.
    """
    import seaborn as sns
    from scipy.stats import ks_2samp
    from statsmodels.stats.multitest import multipletests
    from utils.microbe_names import rename_microbes

    # Drop datasets with only one class in target
    filtered = [
        (mb, tgt, name)
        for mb, tgt, name in zip(microbiome_dfs, target_dfs, dataset_names)
        if len(np.unique(tgt.values.ravel())) >= 2
    ]
    if not filtered:
        print(f"  Skipping {phenotype_name}: no datasets with both classes")
        return (ax or plt.subplots(1, 1)[0]).figure
    dropped = set(dataset_names) - {t[2] for t in filtered}
    if dropped:
        print(f"  Dropped single-class datasets: {dropped}")
    microbiome_dfs, _, dataset_names = (list(x) for x in zip(*filtered))

    # Common features across all datasets
    common = microbiome_dfs[0].columns
    for df in microbiome_dfs[1:]:
        common = common.intersection(df.columns)
    if len(common) == 0:
        raise ValueError(f"No common features for {phenotype_name}")

    # KS significance across dataset pairs (FDR corrected)
    ks_rows = []
    from itertools import combinations as _comb
    for feature in common:
        for i, j in _comb(range(len(microbiome_dfs)), 2):
            stat, p = ks_2samp(microbiome_dfs[i][feature].values,
                               microbiome_dfs[j][feature].values)
            ks_rows.append({"microbe": feature, "p_value": p})
    ks_df = pd.DataFrame(ks_rows)
    reject, p_fdr, _, _ = multipletests(ks_df["p_value"], method="fdr_bh")
    ks_df["significant_fdr"] = reject
    sig_fdr = ks_df.groupby("microbe")["significant_fdr"].max()
    sig_fdr = sig_fdr[sig_fdr].index.tolist()
    pct_sig = 100 * len(sig_fdr) / len(common)
    print(f"  KS (FDR): {len(sig_fdr)}/{len(common)} significant ({pct_sig:.1f}%)")

    # Top features by variance
    if len(common) > max_features:
        all_data = pd.concat([df[common] for df in microbiome_dfs], axis=0)
        common = all_data.var().sort_values(ascending=False).head(max_features).index

    # Z-score combined matrix
    aligned = [df[common] for df in microbiome_dfs]
    X = pd.concat(aligned, axis=0)
    X_z = ((X - X.mean()) / (X.std() + 1e-10)).fillna(0)
    boundaries = [0]
    for df in aligned:
        boundaries.append(boundaries[-1] + len(df))
    X_z.columns = rename_microbes(X_z.columns.tolist())

    _standalone = ax is None
    if _standalone:
        n_samples, n_features = len(X_z), len(X_z.columns)
        fig, ax = plt.subplots(figsize=(max(14, n_features * 0.28),
                                        max(10, n_samples * 0.025)))
    else:
        fig = ax.figure

    sns.heatmap(X_z, cmap="RdBu_r", center=0, robust=True,
                vmin=-3, vmax=3,
                yticklabels=False, xticklabels=False,
                cbar=False, ax=ax)
    ax.set_ylabel("")

    for b in boundaries[1:-1]:
        ax.axhline(y=b, color="black", linewidth=3)

    for i, name in enumerate(dataset_names):
        y_mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(-0.01, y_mid, name, ha="right", va="center",
                fontsize=112*fontsize_scale,
                transform=ax.get_yaxis_transform())

    _title_fs = (title_fontsize_scale if title_fontsize_scale is not None else fontsize_scale)
    if phenotype_name:
        ax.set_title(phenotype_name, fontsize=46*_title_fs, fontweight="bold")

    if _standalone:
        plt.tight_layout()
    return fig


def compute_figure2c_chi2_stats(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    output_dir: str = "figures_out/figure_2/global_original",
) -> pd.DataFrame:
    """
    Chi-square prevalence homogeneity test across datasets, per phenotype.

    For each shared microbe in each phenotype, computes:
        χ² = Σ_i (o_i − e_i)² / e_i   (df = K − 1, K = number of datasets)

    where o_i is the observed zero-prevalence fraction in dataset i and
    e_i is the mean zero-prevalence fraction across all K datasets.

    Returns a DataFrame with columns:
        phenotype, dtype, n_shared_microbes, n_datasets,
        n_significant, fraction_significant, chi2_range_min, chi2_range_max

    Saves full per-microbe results to output_dir/chi2_per_microbe.csv
    and summary per phenotype to output_dir/chi2_summary.csv.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from scipy.stats import chi2 as chi2_dist
    from statsmodels.stats.multitest import multipletests

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_microbe_rows = []

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = Path(data_root) / pheno_str
        if not folder.exists():
            continue
        try:
            mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        filtered = [
            (mb, tgt, n) for mb, tgt, n in zip(mb_dfs, tgt_dfs, ds_names)
            if len(np.unique(tgt.values.ravel())) >= 2
        ]
        if len(filtered) < 2:
            continue
        mb_dfs_f, _, ds_names_f = (list(x) for x in zip(*filtered))

        common = mb_dfs_f[0].columns
        for df in mb_dfs_f[1:]:
            common = common.intersection(df.columns)
        if len(common) == 0:
            continue

        K = len(ds_names_f)
        pvals, chi2_scores, microbe_names = [], [], []

        for microbe in common:
            zero_prevs = np.array([
                float((mb[microbe] == 0).mean()) for mb in mb_dfs_f
            ])
            e_bar = zero_prevs.mean()
            if e_bar <= 0 or e_bar >= 1:
                continue
            chi2_stat = float(np.sum((zero_prevs - e_bar) ** 2 / e_bar))
            p = float(1 - chi2_dist.cdf(chi2_stat, df=K - 1))
            pvals.append(p)
            chi2_scores.append(chi2_stat)
            microbe_names.append(microbe)

        if not pvals:
            continue

        reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
        n_sig  = int(reject.sum())
        frac   = n_sig / len(microbe_names)

        for mic, chi2_s, p, padj, sig in zip(
            microbe_names, chi2_scores, pvals, p_adj, reject
        ):
            per_microbe_rows.append({
                "phenotype":  pheno_str,
                "dtype":      dtype,
                "microbe":    mic,
                "chi2":       round(chi2_s, 4),
                "p_value":    p,
                "p_adjusted": padj,
                "significant":bool(sig),
            })

        summary_rows.append({
            "phenotype":          pheno_str,
            "dtype":              dtype,
            "n_datasets":         K,
            "n_shared_microbes":  len(microbe_names),
            "n_significant":      n_sig,
            "fraction_significant": round(frac, 4),
            "chi2_range_min":     round(float(np.min(chi2_scores)), 4),
            "chi2_range_max":     round(float(np.max(chi2_scores)), 4),
        })
        print(f"  {pheno_str}: {n_sig}/{len(microbe_names)} significant "
              f"({frac*100:.1f}%)")

    summary_df    = pd.DataFrame(summary_rows)
    per_microbe_df = pd.DataFrame(per_microbe_rows)

    summary_df.to_csv(out / "chi2_summary.csv", index=False)
    per_microbe_df.to_csv(out / "chi2_per_microbe.csv", index=False)
    print(f"  Saved chi2 stats → {out}")

    return summary_df


def run_figure2c_supp(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    normalization_approach=None,
) -> None:
    """
    Supplementary: one microbiome feature heatmap per phenotype.
    Output files: supp_figure2c_<phenotype>_<dtype>.pdf
    Also computes and saves chi-square prevalence-homogeneity stats per phenotype.
    Datasets with only one class in target are dropped automatically.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out  = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_root)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = data / pheno_str
        if not folder.exists():
            print(f"  Skipping {pheno_str}: folder not found")
            continue
        try:
            mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue
        if len(ds_names) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets")
            continue

        if normalization_approach is not None:
            from src.rankbird.normalization.pipeline import apply_normalization_pipeline
            name_to_tgt = {n: t for n, t in zip(ds_names, tgt_dfs)}
            mb_dfs, ds_names = apply_normalization_pipeline(list(mb_dfs), list(ds_names))
            tgt_dfs = [name_to_tgt[n] for n in ds_names if n in name_to_tgt]

        safe     = pheno_str.replace(" ", "_")
        mode     = normalization_approach or "raw"
        out_path = out / f"supp_figure2c_{safe}_{mode}.pdf"
        if out_path.exists():
            print(f"  Skipping supp 2C for {pheno_str}: already saved")
            continue
        try:
            fig = plot_figure2c(mb_dfs, tgt_dfs, ds_names, pheno_str,
                                fontsize_scale=0.12, title_fontsize_scale=0.45)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved supp 2C for {pheno_str}")
        except Exception as e:
            print(f"  Error generating supp 2C for {pheno_str}: {e}")

    # Compute and save chi-square prevalence-homogeneity stats
    print("\n  Computing chi-square prevalence-homogeneity stats...")
    compute_figure2c_chi2_stats(
        phenotypes, data_root,
        output_dir="figures_out/figure_2/global_original",
    )


# ─────────────────────────────────────────────────────────────
# Public interface — Figure 2D: Pairwise Wasserstein heatmaps per phenotype
# ─────────────────────────────────────────────────────────────

def _hellinger_1d(u: np.ndarray, v: np.ndarray, n_bins: int = 30) -> float:
    """
    Hellinger distance between two 1D empirical distributions.
    Scale-invariant: bins are defined over the combined range of u and v,
    then each histogram is normalized to a probability vector before comparison.
    """
    lo, hi = min(u.min(), v.min()), max(u.max(), v.max())
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(u, bins=bins)
    q, _ = np.histogram(v, bins=bins)
    p = p / p.sum() if p.sum() > 0 else np.ones(n_bins) / n_bins
    q = q / q.sum() if q.sum() > 0 else np.ones(n_bins) / n_bins
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def _pairwise_hellinger(
    mb_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> np.ndarray:
    """
    Pairwise Hellinger distance between datasets, averaged over common microbes.
    Hellinger is scale-invariant (works on normalized probability distributions),
    making it more reliable than Wasserstein for sparse microbiome data.
    """
    n     = len(dataset_names)
    h_mat = np.full((n, n), np.nan)
    np.fill_diagonal(h_mat, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            common = list(set(mb_dfs[i].columns) & set(mb_dfs[j].columns))
            if not common:
                continue
            distances = [
                _hellinger_1d(mb_dfs[i][col].values, mb_dfs[j][col].values)
                for col in common
            ]
            d = float(np.mean(distances))
            h_mat[i, j] = h_mat[j, i] = round(d, 4)

    return h_mat


def _draw_wasserstein_heatmap(
    ax,
    w_mat: np.ndarray,
    names: List[str],
    show_title: str = None,
) -> None:
    """Draw a single Wasserstein distance heatmap panel on ax."""
    import seaborn as sns

    n     = len(names)
    annot = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            if not np.isnan(w_mat[i, j]):
                annot[i, j] = f'{w_mat[i, j]:.4f}'

    sns.heatmap(
        w_mat,
        annot=annot, fmt='',
        cmap='YlOrRd', vmin=0,
        linewidths=0.5,
        xticklabels=names, yticklabels=names,
        ax=ax,
        cbar_kws={'shrink': 0.7, 'label': 'Hellinger distance'},
        annot_kws={'size': 18},
        mask=np.eye(n, dtype=bool),
    )
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(names, rotation=0, fontsize=18)
    if show_title:
        ax.set_title(show_title, fontsize=22, fontweight='bold')


def plot_figure2d(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    dataset_to_phenotype: Dict[str, str],
) -> plt.Figure:
    """
    Combined figure: one Wasserstein distance heatmap per phenotype, 3 per row.
    """
    pheno_groups: Dict[str, List] = {}
    for mb, name in zip(microbiome_dfs, dataset_names):
        pheno = dataset_to_phenotype[name]
        pheno_groups.setdefault(pheno, []).append((name, mb))

    phenotypes = list(pheno_groups.keys())
    n_phenos   = len(phenotypes)
    ncols      = 3
    nrows      = max(1, int(np.ceil(n_phenos / ncols)))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 6 * nrows),
                             squeeze=False)

    for idx, pheno in enumerate(phenotypes):
        ax    = axes[idx // ncols][idx % ncols]
        items = pheno_groups[pheno]
        names = [item[0] for item in items]
        dfs   = [item[1] for item in items]

        if len(dfs) < 2:
            ax.text(0.5, 0.5, 'Only 1 dataset', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(pheno, fontsize=13, fontweight='bold')
            ax.set_axis_off()
            continue

        w_mat = _pairwise_wasserstein(dfs, names)
        _draw_wasserstein_heatmap(ax, w_mat, names, show_title=pheno)

    for idx in range(n_phenos, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()

    plt.tight_layout()
    return fig


def plot_figure2d_single(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str = "",
    ax=None,
) -> plt.Figure:
    """
    Wasserstein distance heatmap for a single phenotype.
    Embeds into an existing ax when provided.
    """
    _standalone = ax is None
    n = len(dataset_names)
    if _standalone:
        fig, ax = plt.subplots(figsize=(max(5, n * 1.8), max(4, n * 1.6)))
    else:
        fig = ax.figure

    if n < 2:
        ax.text(0.5, 0.5, "Need ≥2 datasets", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return fig, pd.DataFrame()

    w_mat = _pairwise_hellinger(microbiome_dfs, dataset_names)
    _draw_wasserstein_heatmap(ax, w_mat, dataset_names, show_title=phenotype_name)
    w_df = pd.DataFrame(w_mat, index=dataset_names, columns=dataset_names)

    if _standalone:
        plt.tight_layout()
    return fig, w_df




# ─────────────────────────────────────────────────────────────
# Figure 2E: delta AUC vs SHAP Jaccard per phenotype
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
                f1 = set(lodo_shap[d1]["top_10_features"]["feature"])
                f2 = set(lodo_shap[d2]["top_10_features"]["feature"])
                matrix[i, j] = _jaccard(f1, f2)
    return matrix


def compute_figure2e_data(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    cache_dir: str = "figures_out/figure_2/2e",
) -> Dict:
    """
    Self-contained: for each phenotype run pairwise LODO + SHAP, then compute
    |ΔAUC| and SHAP Jaccard for every unordered dataset pair.

    Cache files (per phenotype) in cache_dir:
      <pheno>_pairwise_lodo.csv  — all (train→test) AUC results
      <pheno>_shap_details.txt   — top-10 SHAP features per training dataset

    If both files exist the computation is skipped and results are loaded from disk.

    Returns dict keyed by "<phenotype> <dtype>" with keys:
      jaccard, delta_auc, dataset_a, dataset_b
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from evaluation.pairwise_lodo import (
        pairwise_lodo_with_shap, _save_shap_details, _load_shap_details,
    )

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        safe      = pheno_str.replace(" ", "_")
        csv_path  = cache / f"{safe}_pairwise_lodo.csv"
        shap_path = cache / f"{safe}_shap_details.txt"

        # ── Load from cache if available ──────────────────────────────────────
        if csv_path.exists() and shap_path.exists():
            print(f"  Loading cached 2E data for {pheno_str}")
            pairwise  = pd.read_csv(csv_path)
            lodo_shap = _load_shap_details(shap_path)
        else:
            # ── Load raw microbiome data ──────────────────────────────────────
            folder = Path(data_root) / pheno_str
            if not folder.exists():
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
            except Exception as e:
                print(f"  Error loading {pheno_str}: {e}")
                continue

            filtered = [
                (mb, tgt, n)
                for mb, tgt, n in zip(mb_dfs, tgt_dfs, ds_names)
                if len(np.unique(tgt.values.ravel())) >= 2
            ]
            if len(filtered) < 2:
                print(f"  Skipping {pheno_str}: need ≥2 datasets with both classes")
                continue
            mb_dfs, tgt_dfs, ds_names = (list(x) for x in zip(*filtered))

            # ── Run pairwise LODO + SHAP ──────────────────────────────────────
            print(f"  Computing pairwise LODO+SHAP for {pheno_str} ({len(ds_names)} datasets)")
            pairwise, shap_info = pairwise_lodo_with_shap(mb_dfs, tgt_dfs, ds_names)

            # Build per-training-dataset SHAP dict for Jaccard
            # For each dataset A take features from the first A→B run
            lodo_shap = {}
            for val in shap_info.values():
                train_ds = val["train_dataset"]
                if train_ds not in lodo_shap:
                    lodo_shap[train_ds] = {"top_10_features": val["top_10_features"]}

            pairwise.to_csv(csv_path, index=False)
            _save_shap_details(lodo_shap, shap_path)
            print(f"  Cached 2E data → {cache}")

        # ── Compute |ΔAUC| and Jaccard for unordered pairs ────────────────────
        dataset_names = sorted(set(pairwise["train_dataset"].tolist()))
        jaccard_mat   = _compute_jaccard_matrix(lodo_shap, dataset_names)

        auc_lookup = {
            (row["train_dataset"], row["test_dataset"]): float(row["auc"])
            for _, row in pairwise.iterrows()
            if row["train_dataset"] != row["test_dataset"]
        }

        # Each directed pair (train→test): Jaccard(SHAP_train, SHAP_test) vs AUC(train→test)
        rows_j, rows_auc, rows_train, rows_test = [], [], [], []
        n = len(dataset_names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                train_ds = dataset_names[i]
                test_ds  = dataset_names[j]
                auc_val  = auc_lookup.get((train_ds, test_ds), np.nan)
                if np.isnan(auc_val):
                    continue
                j_val = jaccard_mat[i, j]
                if np.isnan(j_val):
                    continue
                rows_j.append(j_val)
                rows_auc.append(auc_val)
                rows_train.append(train_ds)
                rows_test.append(test_ds)

        if rows_j:
            result[pheno_str] = {
                "jaccard":        np.array(rows_j),
                "pairwise_auc":   np.array(rows_auc),
                "train_dataset":  rows_train,
                "test_dataset":   rows_test,
            }
            print(f"  {pheno_str}: {len(rows_j)} directed pairs")

    return result


def plot_figure2e(
    data: Dict,
    ax=None,
    phenotype: str = None,
) -> plt.Figure:
    """
    Main figure mode (phenotype=None): pool all phenotypes into one scatter,
    color by phenotype, marker shape by dtype (circle=Metagenomics, ^=Amplicon).

    Supplementary mode (phenotype=str): single phenotype scatter, color by train dataset.
    """
    if phenotype is not None:
        _standalone = ax is None
        if _standalone:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure
        _draw_2e_scatter(ax, data.get(phenotype, {}), title=phenotype)
        if _standalone:
            plt.tight_layout()
        return fig

    # ── Pooled: all phenotypes, color by phenotype ────────────
    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    if not data:
        ax.text(0.5, 0.5, "No 2E data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#888")
        ax.set_axis_off()
        return fig

    # ── Fixed color scheme ────────────────────────────────────
    _PHENO_COLORS = {
        "ASD Metagenomics": "#2CA02C",   # green
        "ASD Amplicon":     "#98DF8A",   # light green
        "PD Metagenomics":  "#1F77B4",   # blue
        "PD Amplicon":      "#AEC7E8",   # light blue
        "UC Metagenomics":  "#FF69B4",   # hot pink
        "CRC Metagenomics": "#D62728",   # red
    }
    _CD_COLOR    = "#FFBB78"   # warm peach — CD keeps a stable neutral tone
    _EXTRA_POOL  = ["#FF7F0E", "#9467BD", "#8C564B", "#17BECF", "#BCBD22", "#E377C2", "#7F7F7F"]

    phenos  = sorted(data.keys())
    markers = {"Metagenomics": "o", "Amplicon": "^"}
    _ri = 0
    p_color = {}
    for p in phenos:
        if p in _PHENO_COLORS:
            p_color[p] = _PHENO_COLORS[p]
        elif p.split()[0] == "CD":
            p_color[p] = _CD_COLOR
        else:
            p_color[p] = _EXTRA_POOL[_ri % len(_EXTRA_POOL)]
            _ri += 1

    all_j, all_auc = [], []
    for pheno_str, pheno_data in data.items():
        if not pheno_data:
            continue
        jaccard      = pheno_data["jaccard"]
        pairwise_auc = pheno_data["pairwise_auc"]
        dtype        = "Amplicon" if "Amplicon" in pheno_str else "Metagenomics"
        marker       = markers[dtype]
        color        = p_color[pheno_str]

        ax.scatter(jaccard, pairwise_auc,
                   color=color, marker=marker,
                   s=320, alpha=0.85, edgecolors="black", linewidths=0.8,
                   label=pheno_str, zorder=3)
        all_j.extend(jaccard.tolist())
        all_auc.extend(pairwise_auc.tolist())

    all_j   = np.array(all_j)
    all_auc = np.array(all_auc)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=2.5, alpha=0.6)

    if len(all_j) > 2:
        z = np.polyfit(all_j, all_auc, 1)
        x_line = np.linspace(all_j.min(), all_j.max(), 200)
        ax.plot(x_line, np.poly1d(z)(x_line),
                color="dimgray", linestyle="--", linewidth=2.5, alpha=0.8,
                label="Overall trend")
        r, p = stats.spearmanr(all_j, all_auc)
        ax.text(0.97, 0.97, f"ρ={r:.2f}, p={p:.3f}\n(all phenotypes)",
                transform=ax.transAxes, ha="right", va="top", fontsize=34,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax.set_xlabel("SHAP Jaccard (top-10 features)", fontsize=46, fontweight="bold")
    ax.set_ylabel("Pairwise LODO AUC",               fontsize=46, fontweight="bold")
    ax.tick_params(labelsize=38)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Legend: phenotypes + dtype marker guide
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    dtype_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="none",
               markersize=22, label="Metagenomics"),
        Line2D([0], [0], marker="^", color="gray", linestyle="none",
               markersize=22, label="Amplicon (16S)"),
    ]
    ax.legend(handles=handles + dtype_handles,
              fontsize=26, loc="lower right", framealpha=0.9,
              title="Phenotype  (shape = dtype)", title_fontsize=26,
              ncol=2)

    if _standalone:
        plt.tight_layout()
    return fig


def _draw_2e_scatter(ax, pheno_data: Dict, title: str = "") -> None:
    """Draw one delta-AUC vs SHAP-Jaccard scatter panel."""
    if not pheno_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        return

    jaccard       = pheno_data["jaccard"]
    pairwise_auc  = pheno_data["pairwise_auc"]
    train_ds_list = pheno_data["train_dataset"]
    test_ds_list  = pheno_data["test_dataset"]

    all_ds = sorted(set(train_ds_list))
    tab10  = plt.cm.tab10(np.arange(10))
    cmap   = {d: tab10[i % 10] for i, d in enumerate(all_ds)}

    for ds in all_ds:
        mask = np.array([d == ds for d in train_ds_list])
        ax.scatter(jaccard[mask], pairwise_auc[mask],
                   color=cmap[ds], label=ds,
                   s=220, alpha=0.85, edgecolors="black", linewidths=0.7, zorder=3)

    # Annotate each point with its test dataset name
    for x, y, test_ds in zip(jaccard, pairwise_auc, test_ds_list):
        ax.annotate(
            test_ds,
            xy=(x, y), xytext=(4, 4), textcoords="offset points",
            fontsize=13, color="#333333", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"),
        )

    ax.axhline(0.5, color="red", linestyle="--", linewidth=2.0, alpha=0.6, label="Random (0.5)")

    if len(jaccard) > 2:
        z = np.polyfit(jaccard, pairwise_auc, 1)
        x_line = np.linspace(jaccard.min(), jaccard.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line),
                color="gray", linestyle="--", linewidth=2, alpha=0.7)
        r, p = stats.spearmanr(jaccard, pairwise_auc)
        ax.text(0.97, 0.97, f"ρ={r:.2f}, p={p:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=22,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("SHAP Jaccard (top-10 features)", fontsize=28, fontweight="bold")
    ax.set_ylabel("Pairwise LODO AUC",               fontsize=28, fontweight="bold")
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=18, loc="lower right", framealpha=0.9,
              title="Train dataset\n(label = test dataset)", title_fontsize=16)
    ax.grid(True, alpha=0.25, linestyle="--")
    if title:
        ax.set_title(title, fontsize=28, fontweight="bold")


def run_figure2e_supp(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
) -> None:
    """
    Supplementary 2F: full pairwise-LODO + SHAP analysis figure.
    Datasets with only one class (control-only or case-only) are skipped automatically.
    One figure per phenotype saved as supp_figure2e_<phenotype>_<dtype>.pdf.
    """
    from evaluation.pairwise_lodo import run_pairwise_lodo_analysis
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from figures.pairwise_lodo_plots import plot_pairwise_lodo_figure

    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        safe      = pheno_str.replace(" ", "_")
        out_path  = out / f"supp_figure2e_{safe}.pdf"
        if out_path.exists():
            print(f"  Skipping supp 2F for {pheno_str}: already saved")
            continue
        folder = Path(data_root) / pheno_str
        if not folder.exists():
            continue
        try:
            data = run_pairwise_lodo_analysis(
                phenotype=pheno_str,
                data_root=data_root,
                output_dir=str(out / "data"),
                load_function=load_microbiome_datasets_with_targets,
            )
            if data is None:
                continue
            fig = plot_pairwise_lodo_figure(
                phenotype_name=pheno_str,
                pairwise_results=data["pairwise_results"],
                full_lodo_shap=data["full_lodo_shap"],
                dataset_names=data["dataset_names"],
            )
            # Post-process supp only — no changes to pairwise_lodo_plots.py.
            _ax0, _ax1, _ax2 = fig.axes[0], fig.axes[1], fig.axes[2]
            # Move panel 1 y labels to left BEFORE canvas.draw() so the left-side
            # tick Text objects (label1) are instantiated on the first render.
            _ax0.yaxis.tick_left()
            _ax0.yaxis.set_label_position("right")
            _ax0.set_ylabel("Top SHAP Taxa Across Datasets", fontsize=27,
                            labelpad=10, rotation=90, va="center")
            _ax0.tick_params(axis="y", pad=5)
            fig.subplots_adjust(wspace=0.55, left=0.25)
            fig.canvas.draw()
            # Panel 1 x-axis: scale DOWN (hardcoded ~130pt)
            for _item in ([_ax0.xaxis.label] + _ax0.get_xticklabels()):
                if _item.get_text():
                    _item.set_fontsize(_item.get_fontsize() * 0.25)
            # Panel 1 y labels: set to match panel 3 y label size (9pt × 3 = 27pt)
            plt.setp(_ax0.get_yticklabels(), fontsize=27)
            # Panels 2+3: hardcoded 8–12pt — scale UP; title: scale UP
            for _ax in (_ax1, _ax2):
                for _item in ([_ax.title, _ax.xaxis.label, _ax.yaxis.label] +
                               _ax.get_xticklabels() + _ax.get_yticklabels()):
                    if _item.get_text():
                        _item.set_fontsize(_item.get_fontsize() * 3)
                for _t in _ax.texts:
                    _t.set_fontsize(_t.get_fontsize() * 3)
            for _t in fig.texts:
                _t.set_fontsize(_t.get_fontsize() * 3)
            _p1 = _ax1.get_position()
            _p2 = _ax2.get_position()
            _ax2.set_position([_p1.x1 + 0.20, _p2.y0, _p2.width, _p2.height])
            _w, _h = fig.get_size_inches()
            fig.set_size_inches(_w * 1.45, _h)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved supp 2F for {pheno_str}")
        except Exception as e:
            print(f"  Error generating supp 2F for {pheno_str}: {e}")
