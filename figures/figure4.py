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
    n_cols: int = 2,
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
    n_rows = (n_panels + n_cols - 1) // n_cols   # ceil(n_panels / n_cols)
    if _standalone:
        fig, _axes = plt.subplots(n_rows, n_cols,
                                  figsize=(9 * n_cols, 6 * n_rows), sharey=False)
        axes = list(_axes.flatten())[:n_panels]
    else:
        fig = ax.figure
        ax.set_visible(False)
        bb  = ax.get_position()
        gap = panel_gap
        pw  = (bb.width  - (n_cols - 1) * gap) / n_cols
        ph  = (bb.height - (n_rows - 1) * gap) / n_rows
        axes = [
            fig.add_axes([
                bb.x0 + (i % n_cols) * (pw + gap),
                bb.y0 + (n_rows - 1 - i // n_cols) * (ph + gap),
                pw, ph,
            ])
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

    # ── Tick visibility and shared axis labels ────────────────
    for i, sub_ax in enumerate(axes):
        col = i % n_cols
        row = i // n_cols
        sub_ax.set_ylabel("")
        sub_ax.set_xlabel("")
        # Y ticks: all panels, large labels, tight pad
        sub_ax.tick_params(axis="y", labelsize=tick_fontsize, pad=2)
        # X ticks: bottom row only, large labels, tight pad
        if row == n_rows - 1:
            sub_ax.tick_params(axis="x", labelsize=tick_fontsize, pad=2)
        else:
            sub_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        # Extra breathing room on the categorical x axis
        xl = sub_ax.get_xlim()
        sub_ax.set_xlim(xl[0] - 0.3, xl[1] + 0.3)

    # Shared Y label: vertically centred between the left-column panels
    _left  = [axes[i] for i in range(0, n_panels, n_cols)]
    _ly    = np.mean([a.get_position().y0 + a.get_position().height / 2 for a in _left])
    _lx    = min(a.get_position().x0 for a in _left)
    fig.text(_lx - 0.03, _ly, "AUC",
             fontsize=label_fontsize, fontweight="bold",
             ha="right", va="center", rotation=90)

    # Shared X label: horizontally centred between the bottom-row panels
    _bot   = [axes[i] for i in range(n_panels - n_cols, n_panels)]
    _bx    = np.mean([a.get_position().x0 + a.get_position().width / 2 for a in _bot])
    _by    = min(a.get_position().y0 for a in _bot)
    fig.text(_bx, _by - 0.03, "Protocol",
             fontsize=label_fontsize, fontweight="bold",
             ha="center", va="top")

    if _standalone:
        plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Supplementary: 4A + 4B comparison per phenotype
# ─────────────────────────────────────────────────────────────

def run_figure4ab_supp(
    phenotypes: List,
    data_root: str,
    figures_dir: str,
    normalization_approach=None,
) -> None:
    """
    For each phenotype: one figure with 4A (ROC) and 4B (heatmap),
    original (top row) vs RANK-BIRD normalized (bottom row).
    Output: supp_figure4ab_<phenotype>_<dtype>.pdf
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets
    from figures.figure2 import plot_figure2b, plot_figure2c
    from src.rankbird.normalization.pipeline import apply_normalization_pipeline

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

        filtered = [
            (mb, tgt, name)
            for mb, tgt, name in zip(mb_dfs, tgt_dfs, ds_names)
            if len(np.unique(tgt.values.ravel())) >= 2
        ]
        if len(filtered) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets with both classes")
            continue
        mb_dfs, tgt_dfs, ds_names = (list(x) for x in zip(*filtered))

        safe     = pheno_str.replace(" ", "_")
        out_path = out / f"supp_figure4ab_{safe}.pdf"
        if out_path.exists():
            print(f"  Skipping supp 4AB for {pheno_str}: already saved")
            continue

        try:
            norm_mb, norm_names = apply_normalization_pipeline(list(mb_dfs), list(ds_names))
            name2tgt = {n: t for n, t in zip(ds_names, tgt_dfs)}
            norm_tgt = [name2tgt[n] for n in norm_names if n in name2tgt]
        except Exception as e:
            print(f"  Normalization failed for {pheno_str}: {e}")
            norm_mb, norm_tgt, norm_names = mb_dfs, tgt_dfs, ds_names

        fig, axes = plt.subplots(2, 2, figsize=(28, 20))
        ax_a_orig, ax_b_orig = axes[0]
        ax_a_norm, ax_b_norm = axes[1]

        try:
            plot_figure2b(mb_dfs, tgt_dfs, ds_names,
                          phenotype_name=f"{pheno_str} — Original", ax=ax_a_orig, show_legend=True,
                          fontsize_scale=0.20, legend_fontsize_scale=0.65)
        except Exception as e:
            ax_a_orig.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax_a_orig.transAxes)

        try:
            plot_figure2b(norm_mb, norm_tgt, norm_names,
                          phenotype_name=f"{pheno_str} — CIFAR", ax=ax_a_norm, show_legend=False,
                          fontsize_scale=0.20)
        except Exception as e:
            ax_a_norm.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax_a_norm.transAxes)

        try:
            plot_figure2c(mb_dfs, tgt_dfs, ds_names, phenotype_name="", ax=ax_b_orig,
                          fontsize_scale=0.20)
        except Exception as e:
            ax_b_orig.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax_b_orig.transAxes)

        try:
            plot_figure2c(norm_mb, norm_tgt, norm_names, phenotype_name="", ax=ax_b_norm,
                          fontsize_scale=0.20)
        except Exception as e:
            ax_b_norm.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax_b_norm.transAxes)

        ax_a_orig.set_title("")
        ax_a_norm.set_title("")
        ax_b_orig.set_title("")
        ax_b_norm.set_title("")

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4)
        # Post-process: only thin the ROC lines (linewidth=14 in the base function).
        # All text/legend sizes are set at creation via fontsize_scale so boxes are correct.
        fig.canvas.draw()
        for _ax in fig.axes:
            for _line in _ax.get_lines():
                if _line.get_linewidth() > 5:
                    _line.set_linewidth(3.0)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved supp 4AB for {pheno_str}")


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
    figsize: tuple = (190, 162),
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

    # Outer: 2 rows (height only; each row has its own independent width layout)
    #   Row 0: A (narrower) | D (broader)
    #   Row 1: B (broad)    | C (narrower)
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.4, 1.2],
        hspace=0.38,
        left=0.05, right=0.97, top=0.84, bottom=0.04,
    )

    # Row 0: A (left, narrow) | D (right, broader)
    gs_row0 = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 1.7], wspace=0.28,
    )
    # A: left of row 0 → 2 stacked rows (orig | norm)
    gs_a = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_row0[0, 0], hspace=0.25,
    )
    ax_a_orig = fig.add_subplot(gs_a[0])
    ax_a_norm = fig.add_subplot(gs_a[1])
    # D: right of row 0 — same height as A, legend floats above in GridSpec margin
    ax_d_host = fig.add_subplot(gs_row0[0, 1])

    # Row 1: B (left, broad) | C (right, narrower)
    gs_row1 = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[1.6, 0.9], wspace=0.28,
    )
    # B: left of row 1 → 2 stacked rows (orig | norm)
    gs_b = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_row1[0, 0], hspace=0.25,
    )
    ax_b_orig = fig.add_subplot(gs_b[0])
    ax_b_norm = fig.add_subplot(gs_b[1])
    # C: right of row 1 → wide orig | thin norm
    gs_c = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_row1[0, 1],
        width_ratios=[5.0, 0.5], wspace=0.06,
    )
    ax_c_orig = fig.add_subplot(gs_c[0, 0])
    ax_c_norm = fig.add_subplot(gs_c[0, 1])

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
            fontsize_scale=1.5,
            legend_fontsize_scale=0.45,
        )
    except Exception as e:
        _placeholder(ax_a_orig, f"4A (original) error:\n{e}")

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
            fontsize_scale=1.5,
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
            fontsize_scale=1.5,
        )
    except Exception as e:
        _placeholder(ax_b_orig, f"4B (original) error:\n{e}")
    ax_b_orig.text(-0.03, 1.02, "B", transform=ax_b_orig.transAxes,
                   fontsize=200, fontweight="bold", va="bottom", ha="right", clip_on=False)

    try:
        plot_figure2c(
            microbiome_dfs=norm_mb,
            target_dfs=norm_tgt,
            dataset_names=norm_names,
            phenotype_name="",
            ax=ax_b_norm,
            fontsize_scale=1.5,
        )
    except Exception as e:
        _placeholder(ax_b_norm, f"4B (normalized) error:\n{e}")

    # ── Panel C: KS bars — orig (wide) | norm (thin) ────────────
    _pos_c = ax_c_orig.get_position()
    if figure2d_data is not None and not figure2d_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_data, ax=ax_c_orig, fontsize_scale=1.5)
            ax_c_orig.set_title("")
        except Exception as e:
            _placeholder(ax_c_orig, f"4C (original) error:\n{e}")
    else:
        _placeholder(ax_c_orig, "4C: provide figure2d_data\n(run compute_figure2d_data first)")

    if figure2d_norm_data is not None and not figure2d_norm_data.empty:
        try:
            plot_figure2d_ks_bars(figure2d_norm_data, ax=ax_c_norm, fontsize_scale=1.5)
            ax_c_norm.set_title("")
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
             fontsize=200, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel D: distribution approach (big text, shared legend) ─
    _pos_d = ax_d_host.get_position()
    _axes_before = {id(a) for a in fig.axes}

    if data_dir_4d is not None and Path(data_dir_4d).exists():
        try:
            plot_figure4d(
                base_dir=data_dir_4d, ax=ax_d_host,
                label_fontsize=160, title_fontsize=168,
                tick_fontsize=158, legend_fontsize=148,
                panel_gap=0.030, n_cols=2,
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
            _legend_handles = lgnd.legend_handles
            _legend_labels  = [t.get_text() for t in lgnd.get_texts()]
    for a in _d_sub_axes:
        lgnd = a.get_legend()
        if lgnd:
            lgnd.remove()

    _legend_y = 0.93  # shared y for legend row and letters A/D (above GridSpec top=0.84)

    # Letter D at the start of the legend row; A on the same y
    fig.text(_pos_d.x0 - 0.01, _legend_y, "D",
             fontsize=200, fontweight="bold", va="center", ha="right", clip_on=False)
    _pos_a = ax_a_orig.get_position()
    fig.text(_pos_a.x0 - 0.01, _legend_y, "A",
             fontsize=200, fontweight="bold", va="center", ha="right", clip_on=False)

    if _legend_handles:
        _n_cols = (len(_legend_handles) + 1) // 2  # 2 rows
        fig.legend(
            _legend_handles, _legend_labels,
            loc="upper left",
            bbox_to_anchor=(_pos_d.x0, _legend_y),
            ncol=_n_cols,
            fontsize=168, framealpha=0.9,
        )

    return fig
