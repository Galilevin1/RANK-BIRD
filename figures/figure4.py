"""
Figure 4: Normalization effect overview.

  4A — ROC + LODO: original vs RANK-BIRD normalized  (representative phenotype)
  4B — Microbiome feature heatmap: original vs RANK-BIRD normalized
  4C — KS-significant shared taxa bar chart (same as figure 2D)
  4D — Distribution approach comparison: all 4 data conditions
       (Metagenomics | Amplicon | All datasets | Combined)
       reads pre-computed CSVs from data_dir_4d
       (place all relevant results_*.csv files in figures_out/figure_4/4d/)

Layout
------
  ┌───────────────────┬───────────────────┐
  │  4A (Original)    │  4A (RANK-BIRD)   │  row A — ROC + LODO
  ├───────────────────┼───────────────────┤
  │  4B (Original)    │  4B (RANK-BIRD)   │  row B — feature heatmap
  ├───────────────────┴───────────────────┤
  │         4C (KS fraction bar chart)    │  row C — figure-2D style
  ├───────────────────────────────────────┤
  │         4D (distribution, 4-panel)    │  row D — approach comparison
  └───────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# Panel 4D: distribution approach — all data conditions
# ─────────────────────────────────────────────────────────────

def plot_figure4d(base_dir, combined_dir=None, ax=None) -> plt.Figure:
    """
    Four-panel distribution approach comparison.
    Panels: Metagenomics | Amplicon | All datasets | Combined

    base_dir     : directory containing per-dtype CSVs
                   (results_metagenomics_<approach>.csv, results_amplicon_<approach>.csv)
                   — typically investigations/distribution_approach/
    combined_dir : directory containing cross-dtype CSVs
                   (results_combined_<approach>.csv)
                   — typically investigations/distribution_approach_combined/
                   Defaults to base_dir when not provided.
    ax           : embed into this axes' bounding box; if None a new figure is created.
    """
    from experiments.investigate_distribution_approach import (
        _draw_panel, _make_combined_df, APPROACHES,
    )

    base_dir     = Path(base_dir)
    combined_dir = Path(combined_dir) if combined_dir is not None else base_dir
    dtypes       = ["Metagenomics", "Amplicon"]

    # ── Load all results ──────────────────────────────────────
    all_results: dict = {}
    for dtype in dtypes:
        for approach in APPROACHES:
            p = base_dir / f"results_{dtype.lower()}_{approach}.csv"
            if p.exists():
                all_results[(dtype, approach)] = pd.read_csv(p)

    for approach in APPROACHES:
        p = combined_dir / f"results_combined_{approach}.csv"
        if p.exists():
            all_results[("Combined", approach)] = pd.read_csv(p)

    has_combined = any(("Combined", a) in all_results for a in APPROACHES)
    n_panels     = 4 if has_combined else 3

    # ── Create axes ───────────────────────────────────────────
    _standalone = ax is None
    if _standalone:
        fig, _axes = plt.subplots(1, n_panels,
                                  figsize=(9 * n_panels, 6), sharey=False)
        axes = list(_axes) if n_panels > 1 else [_axes]
    else:
        fig = ax.figure
        ax.set_visible(False)
        bb  = ax.get_position()
        gap = 0.015
        pw  = (bb.width - (n_panels - 1) * gap) / n_panels
        axes = [
            fig.add_axes([bb.x0 + i * (pw + gap), bb.y0, pw, bb.height])
            for i in range(n_panels)
        ]

    # ── Build data frames for each panel ─────────────────────
    panel_configs = [
        ("Metagenomics", "Metagenomics"),
        ("Amplicon",     "Amplicon"),
        ("All datasets", None),
    ]
    if has_combined:
        panel_configs.append(("Combined", "Combined"))

    for sub_ax, (title, dtype_key) in zip(axes, panel_configs):
        if dtype_key is None:
            df = _make_combined_df(all_results, dtypes)
        elif dtype_key == "Combined":
            frames = [
                all_results[("Combined", a)].assign(approach=a)
                for a in APPROACHES
                if ("Combined", a) in all_results
                and not all_results[("Combined", a)].empty
            ]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            frames = [
                all_results[(dtype_key, a)].assign(approach=a)
                for a in APPROACHES
                if (dtype_key, a) in all_results
                and not all_results[(dtype_key, a)].empty
            ]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df.empty:
            sub_ax.text(0.5, 0.5, f"{title}\n(no data)",
                        ha="center", va="center",
                        transform=sub_ax.transAxes, fontsize=12, color="#888")
            sub_ax.set_axis_off()
            continue

        _draw_panel(sub_ax, df, title=title,
                    label_fontsize=40, title_fontsize=36,
                    tick_fontsize=32, legend_fontsize=30,
                    box_width=0.75, dot_size=6)

    # ylabel only on leftmost
    for sub_ax in axes[1:]:
        sub_ax.set_ylabel("")
    # xlabel only on middle panel
    for sub_ax in axes:
        sub_ax.set_xlabel("")
    axes[n_panels // 2].set_xlabel("Protocol", fontsize=40, fontweight="bold")

    if _standalone:
        plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Assembled figure 4
# ─────────────────────────────────────────────────────────────

def assemble_figure4(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str = "",
    dtype: str = "Metagenomics",
    figure2d_data: Optional[pd.DataFrame] = None,
    figure2d_norm_data: Optional[pd.DataFrame] = None,
    data_dir_4d: Optional[Path] = None,
    figsize: tuple = (80, 96),
) -> plt.Figure:
    """
    Assemble Figure 4 panels.

    Parameters
    ----------
    microbiome_dfs     : original (raw) microbiome DataFrames for one phenotype
    target_dfs         : corresponding target DataFrames
    dataset_names      : corresponding dataset names
    phenotype_name     : display label for the phenotype
    dtype              : dtype label for the data (unused; kept for API consistency)
    figure2d_data      : pre-computed KS fraction DataFrame (original) for panel 4C left
    figure2d_norm_data : pre-computed KS fraction DataFrame (normalized) for panel 4C right
    data_dir_4d        : flat directory containing all distribution CSVs for panel 4D
                         (results_metagenomics_*.csv, results_amplicon_*.csv,
                          results_combined_*.csv)
    figsize            : overall figure size
    """
    from figures.figure2 import plot_figure2b, plot_figure2c, plot_figure2d_ks_bars

    # ── Normalize data for panels A and B ──────────────────────
    try:
        from src.rankbird.normalization.pipeline import apply_normalization_pipeline
        norm_mb, norm_names = apply_normalization_pipeline(
            list(microbiome_dfs), list(dataset_names), min_samples_per_dataset=0,
        )
        name2tgt   = {n: t for n, t in zip(dataset_names, target_dfs)}
        norm_names = [n for n in norm_names if n in name2tgt]
        norm_mb    = norm_mb[:len(norm_names)]
        norm_tgt   = [name2tgt[n] for n in norm_names]
    except Exception as e:
        print(f"  [Figure 4] Normalization failed: {e}")
        norm_mb, norm_tgt, norm_names = (
            list(microbiome_dfs), list(target_dfs), list(dataset_names)
        )

    # ── Figure layout ──────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    outer = mgridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.5, 1.4, 1.2, 1.0],
        hspace=0.30,
        left=0.05, right=0.97, top=0.97, bottom=0.04,
    )

    gs_a = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.38)
    ax_4a_orig = fig.add_subplot(gs_a[0, 0])
    ax_4a_norm = fig.add_subplot(gs_a[0, 1])

    gs_b = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.38)
    ax_4b_orig = fig.add_subplot(gs_b[0, 0])
    ax_4b_norm = fig.add_subplot(gs_b[0, 1])

    gs_c = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.35)
    ax_4c_orig = fig.add_subplot(gs_c[0, 0])
    ax_4c_norm = fig.add_subplot(gs_c[0, 1])

    ax_4d = fig.add_subplot(outer[3])

    def _placeholder(ax, msg):
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()

    # ── Panel A: ROC curves + LODO bars (orig | norm) ──────────
    try:
        plot_figure2b(
            microbiome_dfs=list(microbiome_dfs),
            target_dfs=list(target_dfs),
            dataset_names=list(dataset_names),
            phenotype_name=f"{phenotype_name} — Original",
            ax=ax_4a_orig,
        )
    except Exception as e:
        _placeholder(ax_4a_orig, f"4A (original) error:\n{e}")
    ax_4a_orig.text(-0.03, 1.02, "A", transform=ax_4a_orig.transAxes,
                    fontsize=42, fontweight="bold", va="bottom", ha="right", clip_on=False)

    try:
        plot_figure2b(
            microbiome_dfs=norm_mb,
            target_dfs=norm_tgt,
            dataset_names=norm_names,
            phenotype_name=f"{phenotype_name} — CIFAR",
            ax=ax_4a_norm,
        )
    except Exception as e:
        _placeholder(ax_4a_norm, f"4A (normalized) error:\n{e}")

    # ── Panel B: microbiome feature heatmap (orig | norm) ───────
    try:
        plot_figure2c(
            microbiome_dfs=list(microbiome_dfs),
            target_dfs=list(target_dfs),
            dataset_names=list(dataset_names),
            phenotype_name=f"{phenotype_name} — Original",
            ax=ax_4b_orig,
        )
    except Exception as e:
        _placeholder(ax_4b_orig, f"4B (original) error:\n{e}")
    ax_4b_orig.text(-0.03, 1.02, "B", transform=ax_4b_orig.transAxes,
                    fontsize=42, fontweight="bold", va="bottom", ha="right", clip_on=False)

    try:
        plot_figure2c(
            microbiome_dfs=norm_mb,
            target_dfs=norm_tgt,
            dataset_names=norm_names,
            phenotype_name=f"{phenotype_name} — CIFAR",
            ax=ax_4b_norm,
        )
    except Exception as e:
        _placeholder(ax_4b_norm, f"4B (normalized) error:\n{e}")

    # ── Panel C: KS fraction bar chart — Original | CIFAR ───────
    _pos_c = ax_4c_orig.get_position()
    if figure2d_data is not None and not figure2d_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_data, ax=ax_4c_orig)
            ax_4c_orig.set_title("Original", fontsize=38, fontweight="bold", pad=8)
        except Exception as e:
            _placeholder(ax_4c_orig, f"4C (original) error:\n{e}")
    else:
        _placeholder(ax_4c_orig, "4C: provide figure2d_data\n(run compute_figure2d_data first)")

    if figure2d_norm_data is not None and not figure2d_norm_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_norm_data, ax=ax_4c_norm)
            ax_4c_norm.set_title("CIFAR", fontsize=38, fontweight="bold", pad=8)
        except Exception as e:
            _placeholder(ax_4c_norm, f"4C (normalized) error:\n{e}")
    else:
        _placeholder(ax_4c_norm, "4C: provide figure2d_norm_data\n(run compute_figure2d_data with normalization)")

    fig.text(_pos_c.x0 - 0.01, _pos_c.y1 + 0.005, "C",
             fontsize=42, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel D: distribution approach (all 4 conditions) ───────
    _pos_d = ax_4d.get_position()
    if data_dir_4d is not None and Path(data_dir_4d).exists():
        try:
            plot_figure4d(base_dir=data_dir_4d, ax=ax_4d)
        except Exception as e:
            _placeholder(ax_4d, f"4D error:\n{e}")
    else:
        _placeholder(
            ax_4d,
            "4D: place distribution CSVs in figures_out/figure_4/4d/\n"
            "(results_metagenomics_*.csv, results_amplicon_*.csv, results_combined_*.csv)",
        )
    fig.text(_pos_d.x0 - 0.01, _pos_d.y1 + 0.005, "D",
             fontsize=42, fontweight="bold", va="bottom", ha="right", clip_on=False)

    return fig
