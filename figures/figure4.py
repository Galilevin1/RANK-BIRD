"""
Figure 4: Normalization effect overview.

  4A — ROC + LODO: original (top) vs RANK-BIRD normalized (bottom)  [left column]
  4B — Microbiome feature heatmap: original (top) vs RANK-BIRD (bottom) [middle column]
  4C — KS-significant shared taxa bar chart: original (wide) | CIFAR (thin, ~0%) [right column]
  4D — Distribution approach comparison: all 4 data conditions, shared legend at top

Layout
------
  ┌──────────────┬──────────────┬──────────────────┐
  │  4A orig     │  4B orig     │  4C orig  │4C nrm│
  ├──────────────┼──────────────┤  (tall)   │(thin)│
  │  4A norm     │  4B norm     │           │      │
  └──────────────┴──────────────┴──────────────────┘
  ┌ legend ─────────────────────────────────────────┐
  │         4D (distribution, 4-panel)              │
  └─────────────────────────────────────────────────┘
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

def plot_figure4d(
    base_dir,
    combined_dir=None,
    ax=None,
    label_fontsize: int = 40,
    title_fontsize: int = 36,
    tick_fontsize: int = 32,
    legend_fontsize: int = 30,
    panel_gap: float = 0.015,
) -> plt.Figure:
    """
    Four-panel distribution approach comparison.
    Panels: Metagenomics | Amplicon | All datasets | Combined

    base_dir     : directory containing per-dtype CSVs
    combined_dir : directory containing cross-dtype CSVs (defaults to base_dir)
    ax           : embed into this axes' bounding box; if None a new figure is created.
    """
    from experiments.investigate_distribution_approach import (
        _draw_panel, _make_combined_df, APPROACHES,
    )

    base_dir     = Path(base_dir)
    combined_dir = Path(combined_dir) if combined_dir is not None else base_dir
    dtypes       = ["Metagenomics", "Amplicon"]

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

    _standalone = ax is None
    if _standalone:
        fig, _axes = plt.subplots(1, n_panels,
                                  figsize=(9 * n_panels, 6), sharey=False)
        axes = list(_axes) if n_panels > 1 else [_axes]
    else:
        fig = ax.figure
        ax.set_visible(False)
        bb  = ax.get_position()
        gap = panel_gap
        pw  = (bb.width - (n_panels - 1) * gap) / n_panels
        axes = [
            fig.add_axes([bb.x0 + i * (pw + gap), bb.y0, pw, bb.height])
            for i in range(n_panels)
        ]

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
                    label_fontsize=label_fontsize,
                    title_fontsize=title_fontsize,
                    tick_fontsize=tick_fontsize,
                    legend_fontsize=legend_fontsize,
                    box_width=0.75, dot_size=6)

    # ylabel only on leftmost
    for sub_ax in axes[1:]:
        sub_ax.set_ylabel("")
    # xlabel only on middle panel
    for sub_ax in axes:
        sub_ax.set_xlabel("")
    axes[n_panels // 2].set_xlabel("Protocol", fontsize=label_fontsize, fontweight="bold")

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
    figsize: tuple = (160, 100),
) -> plt.Figure:
    """
    Assemble Figure 4 panels.

    Parameters
    ----------
    microbiome_dfs     : original (raw) microbiome DataFrames for one phenotype
    target_dfs         : corresponding target DataFrames
    dataset_names      : corresponding dataset names
    phenotype_name     : display label for the phenotype
    dtype              : dtype label (kept for API consistency)
    figure2d_data      : pre-computed KS fraction DataFrame (original) for panel 4C
    figure2d_norm_data : pre-computed KS fraction DataFrame (normalized) for panel 4C thin
    data_dir_4d        : directory containing distribution CSVs for panel 4D
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

    # Two outer rows: top (A/B/C) | D
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.5, 1.6],
        hspace=0.30,
        left=0.05, right=0.97, top=0.97, bottom=0.04,
    )

    # Top: 2 rows × 3 cols  — A (col 0, stacked) | B (col 1, stacked) | C (col 2, tall)
    gs_top = mgridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0],
        width_ratios=[1.0, 0.7, 1.5],
        wspace=0.35, hspace=0.25,
    )
    ax_a_orig = fig.add_subplot(gs_top[0, 0])
    ax_a_norm = fig.add_subplot(gs_top[1, 0])
    ax_b_orig = fig.add_subplot(gs_top[0, 1])
    ax_b_norm = fig.add_subplot(gs_top[1, 1])

    # C spans both rows of col 2; inner split: wide orig | thin norm
    gs_c = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_top[:, 2],
        width_ratios=[5.0, 0.5], wspace=0.06,
    )
    ax_c_orig = fig.add_subplot(gs_c[0, 0])
    ax_c_norm = fig.add_subplot(gs_c[0, 1])

    # D: full-width row
    ax_d_host = fig.add_subplot(outer[1])

    def _placeholder(ax, msg):
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()

    # ── Panel A: ROC + LODO (orig top | norm bottom) ───────────
    # Top: legend shown, x labels suppressed (redundant with bottom panel)
    try:
        plot_figure2b(
            microbiome_dfs=list(microbiome_dfs),
            target_dfs=list(target_dfs),
            dataset_names=list(dataset_names),
            phenotype_name=f"{phenotype_name} — Original",
            ax=ax_a_orig,
            show_legend=True,
            suppress_xlabel=True,
            inner_width_ratios=[2.0, 1.5],
        )
    except Exception as e:
        _placeholder(ax_a_orig, f"4A (original) error:\n{e}")
    ax_a_orig.text(-0.03, 1.02, "A", transform=ax_a_orig.transAxes,
                   fontsize=72, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # Bottom: no legend, x labels visible
    try:
        plot_figure2b(
            microbiome_dfs=norm_mb,
            target_dfs=norm_tgt,
            dataset_names=norm_names,
            phenotype_name=f"{phenotype_name} — CIFAR",
            ax=ax_a_norm,
            show_legend=False,
            suppress_xlabel=False,
            inner_width_ratios=[2.0, 1.5],
        )
    except Exception as e:
        _placeholder(ax_a_norm, f"4A (normalized) error:\n{e}")

    # ── Panel B: feature heatmap (orig top | norm bottom) ───────
    try:
        plot_figure2c(
            microbiome_dfs=list(microbiome_dfs),
            target_dfs=list(target_dfs),
            dataset_names=list(dataset_names),
            phenotype_name="",
            ax=ax_b_orig,
        )
    except Exception as e:
        _placeholder(ax_b_orig, f"4B (original) error:\n{e}")
    ax_b_orig.text(-0.03, 1.02, "B", transform=ax_b_orig.transAxes,
                   fontsize=72, fontweight="bold", va="bottom", ha="right", clip_on=False)

    try:
        plot_figure2c(
            microbiome_dfs=norm_mb,
            target_dfs=norm_tgt,
            dataset_names=norm_names,
            phenotype_name="",
            ax=ax_b_norm,
        )
    except Exception as e:
        _placeholder(ax_b_norm, f"4B (normalized) error:\n{e}")

    # ── Panel C: KS bars — orig (wide) | norm (thin) ────────────
    _pos_c = ax_c_orig.get_position()
    if figure2d_data is not None and not figure2d_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_data, ax=ax_c_orig)
            ax_c_orig.set_title("Original", fontsize=38, fontweight="bold", pad=8)
        except Exception as e:
            _placeholder(ax_c_orig, f"4C (original) error:\n{e}")
    else:
        _placeholder(ax_c_orig, "4C: provide figure2d_data\n(run compute_figure2d_data first)")

    if figure2d_norm_data is not None and not figure2d_norm_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_norm_data, ax=ax_c_norm)
            ax_c_norm.set_title("CIFAR", fontsize=24, fontweight="bold", pad=8)
        except Exception as e:
            _placeholder(ax_c_norm, f"4C (normalized) error:\n{e}")
    else:
        _placeholder(ax_c_norm, "CIFAR\n(~0%)")

    # Strip all axes decoration from the thin norm panel — keep only bars
    ax_c_norm.set_xlabel("")
    ax_c_norm.set_ylabel("")
    ax_c_norm.set_yticklabels([])
    ax_c_norm.tick_params(axis="y", left=False, labelleft=False)
    ax_c_norm.tick_params(axis="x", labelbottom=False, bottom=False)
    lgnd_c = ax_c_norm.get_legend()
    if lgnd_c:
        lgnd_c.remove()

    fig.text(_pos_c.x0 - 0.02, _pos_c.y1 + 0.005, "C",
             fontsize=72, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel D: distribution approach (big text, shared legend) ─
    _pos_d = ax_d_host.get_position()
    _axes_before = {id(a) for a in fig.axes}

    if data_dir_4d is not None and Path(data_dir_4d).exists():
        try:
            plot_figure4d(
                base_dir=data_dir_4d, ax=ax_d_host,
                label_fontsize=96, title_fontsize=88,
                tick_fontsize=80, legend_fontsize=76,
                panel_gap=0.005,
            )
        except Exception as e:
            _placeholder(ax_d_host, f"4D error:\n{e}")
    else:
        _placeholder(
            ax_d_host,
            "4D: place distribution CSVs in figures_out/figure_4/4d/\n"
            "(results_metagenomics_*.csv, results_amplicon_*.csv, results_combined_*.csv)",
        )

    # Collect legend handles from D sub-panels, remove individual legends,
    # add one shared legend horizontally above D
    _d_sub_axes = [a for a in fig.axes if id(a) not in _axes_before]
    _legend_handles, _legend_labels = None, None
    for a in _d_sub_axes:
        lgnd = a.get_legend()
        if lgnd and _legend_handles is None:
            _legend_handles, _legend_labels = a.get_legend_handles_labels()
    for a in _d_sub_axes:
        lgnd = a.get_legend()
        if lgnd:
            lgnd.remove()

    if _legend_handles:
        fig.legend(
            _legend_handles, _legend_labels,
            loc="upper center",
            bbox_to_anchor=((_pos_d.x0 + _pos_d.x1) / 2, _pos_d.y1 + 0.045),
            ncol=len(_legend_handles),
            fontsize=76, framealpha=0.9,
        )

    fig.text(_pos_d.x0 - 0.01, _pos_d.y1 + 0.005, "D",
             fontsize=72, fontweight="bold", va="bottom", ha="right", clip_on=False)

    return fig
