"""
Figure 1: Assembled overview figure.

Creates all sub-figure panels directly as subplots in one combined figure.

  1a  Paper–Phenotype grid (hardcoded data)
  1b  Schematic image (user-provided PNG, loaded from path_1b)
  1c  Papers vs LightGBM LODO horizontal bars  (requires results_df, papers_df)
  1d  Protocol AUC heatmap   (requires summary_df, results_df, papers_df)
  1e  Protocol AUC boxplots  (requires results_df)
  1f  Cross-dtype LODO on overlap microbes  (requires figure1f_df)

Layout
------
  ┌──────────┬─────────────────┐
  │    1a    │                 │
  ├──────────┤       1b        │
  │    1c    │                 │
  ├──────────┴─────────────────┤
  │    1d    │    1e    │  1f  │
  └──────────┴──────────┴──────┘
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as mgridspec
import pandas as pd
from typing import Optional


def assemble_figure1(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    papers_df: pd.DataFrame = None,
    selected_combinations=None,
    path_1b: Optional[Path] = None,
    figure1f_df: Optional[pd.DataFrame] = None,
    figsize: tuple = (62, 80),
    left_col_ratio: float = 1.15,
    right_col_ratio: float = 1.85,
    top_row_ratio: float = 2.8,
    mid_row_ratio: float = 2.8,
    mid_col_ratios: tuple = (0.85, 0.75),
    section_hspace: float = 0.22,
    top_hspace: float = 0.48,
    top_wspace: float = 0.28,
) -> plt.Figure:
    """
    Assemble panels A–F as live subplots in one combined Figure 1.

    Layout
    ------
      ┌──────────┬─────────────────┐
      │    1a    │                 │
      ├──────────┤       1b        │
      │    1c    │                 │
      ├──────────┴─────────────────┤
      │         1d      │   1e     │
      ├──────────────────────────-─┤
      │              1f            │
      └────────────────────────────┘

    figure1f_df : pre-computed results from compute_figure1f(); if None panel F
                  shows a placeholder.
    """
    from figures.phenotype_grid import plot_figure_1a
    from figures.papers_vs_lgbm_lodo import plot_auc_horizontal_bars_mann_whitney
    from figures.protocol_comparison_heatmap import plot_protocol_heatmap
    from figures.protocol_comparison_boxplots import plot_protocol_boxplots
    from figures.figure1f import plot_figure1f

    top_height = top_row_ratio + mid_row_ratio   # combined inner rows of top section
    mid_height = top_height * 0.48               # 1d + 1e row
    bot_height = top_height * 0.40               # 1f row

    fig = plt.figure(figsize=figsize)
    outer = mgridspec.GridSpec(
        3, 1, figure=fig,
        hspace=section_hspace,
        height_ratios=[top_height, mid_height, bot_height],
        left=0.05, right=0.98, top=0.97, bottom=0.04,
    )

    # ── Section 1: 2 rows × 2 cols; 1b spans both inner rows ────────────────
    gs_top = mgridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0],
        width_ratios=[left_col_ratio, right_col_ratio],
        height_ratios=[top_row_ratio, mid_row_ratio],
        hspace=top_hspace, wspace=top_wspace,
    )

    ax_a = fig.add_subplot(gs_top[0, 0])   # top-left
    ax_b = fig.add_subplot(gs_top[:, 1])   # right, spans both inner rows
    ax_c = fig.add_subplot(gs_top[1, 0])   # bottom-left

    # Shift 1b right without resizing
    _pos_b = ax_b.get_position()
    ax_b.set_position([_pos_b.x0 - 0.02, _pos_b.y0, _pos_b.width, _pos_b.height])

    # Shift 1a and 1c slightly to the left
    _pos_a = ax_a.get_position()
    ax_a.set_position([_pos_a.x0 - 0.03, _pos_a.y0, _pos_a.width, _pos_a.height])
    _pos_c = ax_c.get_position()
    ax_c.set_position([_pos_c.x0 - 0.03, _pos_c.y0, _pos_c.width, _pos_c.height])

    # ── Section 2: 1d left, gap, 1e right (small outer margins) ─────────────
    gs_mid = mgridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1],
        width_ratios=[0.0, mid_col_ratios[0], mid_col_ratios[1], 0.12],
        wspace=0.15,
    )

    ax_d = fig.add_subplot(gs_mid[1])
    ax_e = fig.add_subplot(gs_mid[2])

    # Shift 1d left without resizing
    _pos_d = ax_d.get_position()
    ax_d.set_position([_pos_d.x0 - 0.11, _pos_d.y0, _pos_d.width, _pos_d.height])

    # ── Section 3: 1f slightly left of centre ─────────────────────────────────
    gs_bot = mgridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2],
        width_ratios=[0.07, 1.0, 0.17],
        wspace=0,
    )
    ax_f = fig.add_subplot(gs_bot[1])

    # Shift 1f left without resizing
    _pos_f = ax_f.get_position()
    ax_f.set_position([_pos_f.x0 - 0.05, _pos_f.y0, _pos_f.width, _pos_f.height])

    # ── Panel A: paper–phenotype grid ────────────────────────────────────────
    plot_figure_1a(ax=ax_a)
    ax_a.text(-0.18, 1.0, "A", transform=ax_a.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel B: schematic image ──────────────────────────────────────────────
    ax_b.set_axis_off()
    if path_1b and Path(path_1b).exists():
        img = mpimg.imread(str(path_1b))
        ax_b.imshow(img, aspect="auto", interpolation="bilinear")
    else:
        ax_b.set_facecolor("#E8E8E8")
        ax_b.text(0.5, 0.5, "Schematic\n(place schematic.png in\nfigures_out/figure_1/1b/)",
                  ha="center", va="center", fontsize=13, color="#555555",
                  transform=ax_b.transAxes)
    ax_b.text(0.01, 1.0, "B", transform=ax_b.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="left", clip_on=False)

    # ── Panel C: papers vs LightGBM LODO bars ────────────────────────────────
    if papers_df is not None:
        plot_auc_horizontal_bars_mann_whitney(
            df_papers=papers_df,
            df_lightGBM=results_df,
            selected_combinations=selected_combinations,
            ax=ax_c,
        )
    else:
        ax_c.text(0.5, 0.5, "No papers data", ha="center", va="center",
                  transform=ax_c.transAxes, fontsize=12, color="#888888")
    ax_c.text(-0.18, 1.08, "C", transform=ax_c.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel D: protocol AUC heatmap ─────────────────────────────────────────
    plot_protocol_heatmap(
        summary_df,
        results_df=results_df,
        papers_df=papers_df,
        selected_combinations=selected_combinations,
        ax=ax_d,
    )
    ax_d.text(-0.22, 1.0, "D", transform=ax_d.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel E: protocol AUC boxplots ────────────────────────────────────────
    plot_protocol_boxplots(results_df, ax=ax_e)
    ax_e.text(-0.12, 1.0, "E", transform=ax_e.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel F: cross-dtype LODO on overlap microbes ─────────────────────────
    plot_figure1f(figure1f_df if figure1f_df is not None else pd.DataFrame(), ax=ax_f)
    ax_f.text(-0.05, 1.0, "F", transform=ax_f.transAxes,
              fontsize=52, fontweight="bold", va="top", ha="right", clip_on=False)

    return fig
