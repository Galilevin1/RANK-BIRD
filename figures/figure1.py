"""
Figure 1: Combined paper overview figure.

Layout
------
  ┌──────────┬───────────────┬───────────────┐
  │          │      1a       │      1b       │
  │   1c     │ (pheno grid)  │  (schematic)  │
  │ (papers  ├───────────────┴───────────────┤
  │  vs lgbm)│      1d       │      1e       │
  │          │   (heatmap)   │  (boxplots)   │
  └──────────┴───────────────┴───────────────┘

Usage
-----
    from figures.figure1 import plot_figure1

    fig, stats = plot_figure1(
        results_df, summary_df, papers_csv, phenotypes
    )
    fig.savefig(...)
    stats.to_csv(...)
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# Utility: render an existing Figure as an image on an axis
# ─────────────────────────────────────────────────────────────

def _fig_to_array(fig: plt.Figure, dpi: int = 150) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return plt.imread(buf)


def _place_fig(src_fig: plt.Figure, dst_ax: plt.Axes, dpi: int = 150):
    arr = _fig_to_array(src_fig, dpi=dpi)
    dst_ax.imshow(arr, aspect='auto', interpolation='lanczos')
    dst_ax.set_axis_off()


# ─────────────────────────────────────────────────────────────
# Panel 1b: learning-protocol schematic (drawn programmatically)
# ─────────────────────────────────────────────────────────────

_DS_COLORS  = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
_EDGE       = '#4A5568'
_ARROW_KW   = dict(color=_EDGE, lw=1.5)
_TEXT_KW    = dict(ha='center', va='center', fontsize=9, color='#2D3748')


def _doc(ax, x, y, color, size=0.28):
    """Coloured document icon (rectangle + corner fold)."""
    h = size * 1.25
    fold = size * 0.28
    ax.add_patch(plt.Rectangle(
        (x - size / 2, y - h / 2), size, h,
        facecolor=color, edgecolor=_EDGE, linewidth=0.8, zorder=3,
    ))
    ax.add_patch(plt.Polygon(
        [[x + size/2 - fold, y + h/2],
         [x + size/2,        y + h/2 - fold],
         [x + size/2,        y + h/2]],
        facecolor='white', edgecolor=_EDGE, linewidth=0.8, zorder=4,
    ))


def _box(ax, cx, cy, w, h, text, facecolor='#EBF8FF', fontsize=9):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=facecolor, edgecolor=_EDGE, linewidth=1.3, zorder=3,
    ))
    ax.text(cx, cy, text, zorder=4, **{**_TEXT_KW, 'fontsize': fontsize})


def _arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', connectionstyle=f'arc3,rad={rad}',
                        **_ARROW_KW),
        annotation_clip=False,
    )


def _draw_schematic(ax):
    ax.set_xlim(0, 10);  ax.set_ylim(0, 10);  ax.axis('off')

    # ── Title & shared dataset icons ────────────────────────
    ax.text(5, 9.55, 'Datasets of Same Phenotype',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#1A202C')

    icon_xs = [3.7 + i * 0.65 for i in range(5)]
    for x, c in zip(icon_xs, _DS_COLORS):
        _doc(ax, x, 8.9, c, size=0.28)

    # ── Three diverging arrows ───────────────────────────────
    col_x = [1.6, 5.0, 8.4]
    for cx in col_x:
        _arrow(ax, 5.0, 8.5, cx, 7.95)

    # ── Column headers ───────────────────────────────────────
    for cx, title in zip(col_x, ['Within-\nLearning',
                                   'Internal-\nValidation',
                                   'LODO-\nLearning']):
        ax.text(cx, 7.62, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2D3748')

    # ── WITHIN-LEARNING ─────────────────────────────────────
    cx = 1.6
    _doc(ax, cx, 7.05, _DS_COLORS[0])
    _arrow(ax, cx - 0.15, 6.7, cx - 0.65, 6.1)   # left branch → training
    _arrow(ax, cx + 0.15, 6.7, cx + 0.65, 6.1)   # right branch → test
    ax.text(cx - 0.65, 5.92, 'Training\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    ax.text(cx + 0.65, 5.92, 'Test\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    _arrow(ax, cx - 0.65, 5.38, cx - 0.65, 4.85)
    _box(ax, cx - 0.65, 4.55, 1.1, 0.52, '⚙  Model')
    # eval arc: model → test set
    ax.annotate('', xy=(cx + 0.65, 5.38), xytext=(cx - 0.1, 4.55),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.38',
                                **_ARROW_KW),
                annotation_clip=False)

    # ── INTERNAL-VALIDATION ──────────────────────────────────
    cx = 5.0
    for i, c in enumerate(_DS_COLORS[:4]):
        _doc(ax, cx - 0.45 + i * 0.3, 7.05, c, size=0.24)
    _arrow(ax, cx - 0.3, 6.7, cx - 0.55, 6.1)
    _arrow(ax, cx + 0.3, 6.7, cx + 0.55, 6.1)
    ax.text(cx - 0.55, 5.92, 'Training\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    ax.text(cx + 0.55, 5.92, 'Test\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    _arrow(ax, cx - 0.55, 5.38, cx - 0.55, 4.85)
    _box(ax, cx - 0.55, 4.55, 1.1, 0.52, '⚙  Model')
    ax.annotate('', xy=(cx + 0.55, 5.38), xytext=(cx + 0.0, 4.55),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.38',
                                **_ARROW_KW),
                annotation_clip=False)

    # ── LODO-LEARNING ────────────────────────────────────────
    cx = 8.4
    # 4 training docs close together
    for i, c in enumerate(_DS_COLORS[:4]):
        _doc(ax, cx - 0.55 + i * 0.28, 7.05, c, size=0.22)
    # 1 held-out doc to the right
    _doc(ax, cx + 0.9, 7.05, _DS_COLORS[0], size=0.22)
    ax.text(cx + 0.9, 6.7, 'Left\nout', ha='center', va='center',
            fontsize=7, color='#718096')

    _arrow(ax, cx - 0.15, 6.7, cx - 0.45, 6.1)      # group → training
    _arrow(ax, cx + 0.9,  6.7, cx + 0.9,  6.1)       # lone doc → test
    ax.text(cx - 0.45, 5.92, 'Training\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    ax.text(cx + 0.9, 5.92, 'Test\nset', ha='center', va='top',
            fontsize=8, style='italic', color='#4A5568')
    _arrow(ax, cx - 0.45, 5.38, cx - 0.45, 4.85)
    _box(ax, cx - 0.45, 4.55, 1.1, 0.52, '⚙  Model')
    ax.annotate('', xy=(cx + 0.9, 5.38), xytext=(cx + 0.1, 4.55),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.42',
                                **_ARROW_KW),
                annotation_clip=False)

    # ── Vertical dividers between columns ────────────────────
    for div_x in [3.25, 6.75]:
        ax.axvline(div_x, ymin=0.28, ymax=0.88, color='#CBD5E0',
                   linewidth=0.8, linestyle='--', zorder=0)


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

def plot_figure1(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    papers_csv: str,
    phenotypes: List[Tuple[str, str]],
    render_dpi: int = 150,
):
    """
    Combine panels 1a, 1b, 1c, 1d, 1e into a single Figure 1.

    Parameters
    ----------
    results_df  : per-dataset AUC results (protocol_results_per_dataset.csv)
    summary_df  : aggregated summary (protocol_summary.csv)
    papers_csv  : path to papers AUC CSV
    phenotypes  : list of (phenotype, dtype) tuples
    render_dpi  : DPI used when rasterising sub-figures

    Returns
    -------
    fig    : the composite Figure
    stats  : Mann-Whitney stats DataFrame from panel 1c
    """
    from figures.phenotype_grid import plot_figure_1a
    from figures.papers_vs_lgbm_lodo import plot_auc_horizontal_bars_mann_whitney
    from figures.protocol_comparison_heatmap import plot_protocol_heatmap
    from figures.protocol_comparison_boxplots import plot_protocol_boxplots
    from evaluation.data_loading import build_papers_auc_df

    # ── Composite canvas ─────────────────────────────────────
    # Layout:  ┌──────┬──────┐
    #          │  A   │  B   │
    #          ├──────┼──────┤
    #          │  E   │      │
    #          ├──────┤  C   │
    #          │  D   │      │
    #          └──────┴──────┘
    # Left col: A (top), E (mid), D (bot)
    # Right col: B (top), C spans rows 1-2
    fig = plt.figure(figsize=(34, 26))

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[0.8, 1.35],   # left col narrower → D/A less broad
        height_ratios=[1.3, 1.1, 1.5],  # A shorter, D taller
        hspace=0.08, wspace=0.07,
        left=0.03, right=0.98, top=0.96, bottom=0.03,
    )
    ax_1a = fig.add_subplot(gs[0, 0])
    ax_1b = fig.add_subplot(gs[0, 1])
    ax_1e = fig.add_subplot(gs[1, 0])
    ax_1d = fig.add_subplot(gs[2, 0])
    ax_1c = fig.add_subplot(gs[1:, 1])  # right col, rows 1-2

    # ── Panel A: phenotype grid ──────────────────────────────
    fig_1a, _ = plot_figure_1a()
    _place_fig(fig_1a, ax_1a, dpi=render_dpi)
    plt.close(fig_1a)

    # ── Panel B: schematic ───────────────────────────────────
    _draw_schematic(ax_1b)

    # ── Panel C: papers vs LightGBM (tall right panel) ───────
    df_papers = build_papers_auc_df(papers_csv)
    fig_1c, _, stats = plot_auc_horizontal_bars_mann_whitney(
        df_papers=df_papers,
        df_lightGBM=results_df,
        selected_combinations=phenotypes,
        figsize=(19, 16),
        bar_height=1,
        fdr_alpha=0.05,
    )
    _place_fig(fig_1c, ax_1c, dpi=render_dpi)
    plt.close(fig_1c)

    # ── Panel D: protocol heatmap ────────────────────────────
    fig_1d, _ = plot_protocol_heatmap(summary_df)
    _place_fig(fig_1d, ax_1d, dpi=render_dpi)
    plt.close(fig_1d)

    # ── Panel E: protocol boxplots ───────────────────────────
    fig_1e, _ = plot_protocol_boxplots(results_df)
    _place_fig(fig_1e, ax_1e, dpi=render_dpi)
    plt.close(fig_1e)

    # ── Panel labels ─────────────────────────────────────────
    label_kw = dict(fontsize=18, fontweight='bold', va='top', zorder=10)
    for ax, letter in [(ax_1a, 'A'), (ax_1b, 'B'), (ax_1c, 'C'),
                       (ax_1d, 'D'), (ax_1e, 'E')]:
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, **label_kw)

    return fig, stats
