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
    figsize: tuple = (125, 101),
    left_col_ratio: float = 2.0,
    right_col_ratio: float = 3.1,
    mid_left_col_ratio: float = 1.0,
    mid_right_col_ratio: float = 1.2,
    top_row_ratio: float = 4.2,
    mid_row_ratio: float = 4.0,
    section_hspace: float = 0.22,
    top_hspace: float = 0.62,
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

    top_height = top_row_ratio + mid_row_ratio
    mid_height = top_height * 0.48

    fig = plt.figure(figsize=figsize)
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        hspace=section_hspace,
        height_ratios=[top_height, mid_height],
        left=0.16, right=0.99, top=0.94, bottom=0.07,
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

    # ── Section 2: D under A/C, E narrower than B ────────────────────────────
    gs_mid = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[mid_left_col_ratio, mid_right_col_ratio],
        wspace=top_wspace,
    )

    ax_d = fig.add_subplot(gs_mid[0])
    ax_e = fig.add_subplot(gs_mid[1])

    # Pin D's right edge to C's right edge; E keeps GridSpec width (decoupled from B)
    _c_pos = ax_c.get_position()
    _b_pos = ax_b.get_position()
    _d_pos = ax_d.get_position()
    _e_pos = ax_e.get_position()
    ax_d.set_position([_d_pos.x0, _d_pos.y0, _c_pos.x1 - _d_pos.x0, _d_pos.height])
    _a_pos_init = ax_a.get_position()
    ax_a.set_position([_a_pos_init.x0, _a_pos_init.y0 + 0.015, _a_pos_init.width, _a_pos_init.height - 0.015])
    _e_trim = 0.03   # narrows E from the left
    _e_shift = 0.06  # shifts E's body left without changing its width or the letter
    ax_e.set_position([_e_pos.x0 + _e_trim - _e_shift, _e_pos.y0,
                       _e_pos.width - _e_trim + 0.02, _e_pos.height])
    # Expand B: up into top margin, down to D/E's top row, right stays at figure edge
    _d_pos_updated = ax_d.get_position()
    ax_b.set_position([_b_pos.x0 - 0.012, _d_pos_updated.y1 + 0.04,
                       _b_pos.width + 0.018, 0.97 - _d_pos_updated.y1 - 0.04])

    # ── Panel A: paper–phenotype grid ────────────────────────────────────────
    plot_figure_1a(ax=ax_a)
    _a_pos = ax_a.get_position()
    fig.text(_a_pos.x0 - 0.22 * _a_pos.width, ax_b.get_position().y1,
             "A", fontsize=110, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel B: schematic image ──────────────────────────────────────────────
    ax_b.set_axis_off()
    if path_1b and Path(path_1b).exists():
        img = mpimg.imread(str(path_1b))
        ax_b.imshow(img, aspect="auto", interpolation="nearest")
    else:
        ax_b.set_facecolor("#E8E8E8")
        ax_b.text(0.5, 0.5, "Schematic\n(place schematic.png in\nfigures_out/figure_1/1b/)",
                  ha="center", va="center", fontsize=13, color="#555555",
                  transform=ax_b.transAxes)
    ax_b.text(-0.02, 1.0, "B", transform=ax_b.transAxes,
              fontsize=110, fontweight="bold", va="top", ha="right", clip_on=False)

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
    ax_c.text(-0.22, 1.2, "C", transform=ax_c.transAxes,
              fontsize=110, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel D: protocol AUC heatmap ─────────────────────────────────────────
    # Align D's WGS/16S boxes with C's: compute the transAxes x for D that
    # maps to the same figure coordinate as C's labels at transAxes x=-0.32.
    _c_pos = ax_c.get_position()
    _d_pos = ax_d.get_position()
    _d_wgs_x = (_c_pos.x0 + (-0.32) * _c_pos.width - _d_pos.x0) / _d_pos.width

    plot_protocol_heatmap(
        summary_df,
        results_df=results_df,
        papers_df=papers_df,
        selected_combinations=selected_combinations,
        ax=ax_d,
        wgs_16s_x=_d_wgs_x,
    )
    ax_d.text(-0.22, 1.06, "D", transform=ax_d.transAxes,
              fontsize=110, fontweight="bold", va="top", ha="right", clip_on=False)

    # ── Panel E: protocol AUC boxplots ────────────────────────────────────────
    plot_protocol_boxplots(results_df, ax=ax_e)
    # Place "E" at the same figure-x as "B" so E is right below B
    _b_pos = ax_b.get_position()
    _e_pos = ax_e.get_position()
    fig.text(_b_pos.x0 - 0.005, _e_pos.y1 + 0.015, "E",
             fontsize=110, fontweight="bold", va="bottom", ha="right",
             clip_on=False)

    return fig
