"""
Figure 3: Normalization and stability overview.

  3A — Schematic (user-provided image)
  3B — Stability threshold investigation (drawn directly from CSVs)
  3C — Distribution comparison: Original vs RANK-BIRD (Metagenomics | Amplicon | All)
  3D — UMAP scatter: raw vs RANK-BIRD normalized, per phenotype
  3E — Manually provided image (e.g. Taxa_Filtering.pdf)

Main figure
-----------
    assemble_figure3(path_3a, investigation_dir_3b, investigation_dir_3c,
                     data_root_3d, phenotypes_3d, normalization_approach, path_3e)

Supplementary (one file per phenotype for 3D; one combined file for 3C)
------------------------------------------------------------------------
    run_figure3c(investigation_dir, figures_dir)
    run_figure3d(phenotypes, data_root, figures_dir, normalization_approach)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Generic image loader (PNG/JPG + PDF via PyMuPDF)
# ─────────────────────────────────────────────────────────────

def _find_any_image(directory: Path) -> Optional[Path]:
    """Return the first image file (png/jpg/jpeg/pdf) found in directory."""
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.pdf", "*.PDF"):
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_any_image(path: Path) -> Optional[np.ndarray]:
    """Load an image from path. Supports PNG/JPG and PDF (first page)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        # Try PyMuPDF
        try:
            import fitz
            doc  = fitz.open(str(path))
            page = doc[0]
            pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        except ImportError:
            pass
        # Try pdf2image
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(str(path), dpi=150)
            if pages:
                return np.array(pages[0])
        except ImportError:
            pass
        # Try PIL directly (works if Ghostscript is installed)
        try:
            from PIL import Image
            img = Image.open(str(path))
            return np.array(img)
        except Exception:
            pass
        print("  Cannot load PDF — install pymupdf (`pip install pymupdf`) or "
              "pdf2image (`pip install pdf2image`) to render PDF panels.")
        return None
    else:
        return mpimg.imread(str(path))


# ─────────────────────────────────────────────────────────────
# UMAP scatter (shared by 3D panels)
# ─────────────────────────────────────────────────────────────

_DTYPE_COLORS  = {"Metagenomics": "#1f77b4", "Amplicon": "#d62728"}   # blue / red
_CASE_EDGE     = "black"   # border color for case (target=1) samples


def _umap_scatter(
    ax: plt.Axes,
    X: pd.DataFrame,
    targets: np.ndarray,
    dtypes: np.ndarray,
    title: str,
    dataset_labels: Optional[np.ndarray] = None,
    dataset_color_map: Optional[dict] = None,
    label_fontsize: int = 72,
    title_fontsize: int = 76,
    tick_fontsize: int = 60,
) -> None:
    """Run t-SNE on X and draw scatter. Each dataset gets a distinct shade: blue=WGS, red=16S."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    x_label, y_label = "t-SNE 1", "t-SNE 2"

    Xs     = StandardScaler().fit_transform(X.fillna(0))
    coords = reducer.fit_transform(Xs)

    if dataset_labels is not None and dataset_color_map is not None:
        for ds in sorted(set(dataset_labels)):
            color  = dataset_color_map.get(ds, "#888888")
            mask_d = dataset_labels == ds
            for tgt in sorted(set(targets)):
                mask = mask_d & (targets == tgt)
                if mask.sum() == 0:
                    continue
                edge = _CASE_EDGE if int(tgt) == 1 else "none"
                lw   = 0.7 if int(tgt) == 1 else 0
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    color=color, marker="o",
                    s=100, alpha=0.65,
                    edgecolors=edge, linewidths=lw,
                )
    else:
        for dtype in sorted(set(dtypes)):
            color  = _DTYPE_COLORS.get(dtype, "#888888")
            mask_d = dtypes == dtype
            for tgt in sorted(set(targets)):
                mask = mask_d & (targets == tgt)
                if mask.sum() == 0:
                    continue
                edge = _CASE_EDGE if int(tgt) == 1 else "none"
                lw   = 0.7 if int(tgt) == 1 else 0
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, marker="o",
                    s=100, alpha=0.65,
                    edgecolors=edge, linewidths=lw,
                )

    ax.set_xlabel(x_label, fontsize=label_fontsize, labelpad=4)
    ax.set_ylabel(y_label, fontsize=label_fontsize, labelpad=4)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.tick_params(labelsize=tick_fontsize)
    ax.margins(0.05)


# ─────────────────────────────────────────────────────────────
# Figure 3B: stability threshold investigation (direct draw)
# ─────────────────────────────────────────────────────────────

def plot_figure3b(
    investigation_dir: str,
    ax=None,
    plot_mode: str = "combined",
    plot_levels: list = None,
    combined_subdir: str = "stability_threshold_rankbird_wasserstein_combined",
    force_single_row: bool = False,
    stacked: bool = False,
    prefer_filter_only: bool = False,
    thresholds: dict = None,
) -> plt.Figure:
    """
    Draw the stability threshold investigation grid directly onto axes.

    Loads pre-computed CSVs from investigation_dir and calls _plot_sweep_panel
    for each (level × dtype) cell — same pattern as plot_figure3c/plot_figure3d.

    Four columns: Metagenomics | Amplicon | All datasets | Combined (cross-dtype).
    Combined data is loaded from investigation_dir/combined_subdir/.

    Parameters
    ----------
    investigation_dir : path to dir produced by run_stability_investigation
    ax                : embed into this axes' bounding box when provided
    plot_mode         : "mean", "median", or "combined"
    plot_levels       : subset of LEVELS keys to include; None = all
    combined_subdir   : subdirectory name holding cross-dtype CSV files
    """
    from experiments.investigate_stability_threshold import (
        _plot_sweep_panel, LEVELS, _BEST_THRESHOLDS,
        _aggregate_raw_to_sweep, _merge_sweep_dfs, _sum_count_dfs, _merge_orig_aucs,
    )

    DISPLAY_DTYPES = ["Metagenomics", "Amplicon", "All datasets", "Combined"]

    inv_dir        = Path(investigation_dir)
    comb_dir       = inv_dir / combined_subdir
    levels_to_plot = [(l, lbl) for l, lbl in LEVELS
                      if plot_levels is None or l in plot_levels]
    nrows, ncols   = len(levels_to_plot), len(DISPLAY_DTYPES)

    # ── Load CSVs — per-dtype (Metagenomics / Amplicon) ──────
    raw_store, count_store, orig_store = {}, {}, {}
    for level, _ in LEVELS:
        for dd in ["Metagenomics", "Amplicon"]:
            base = f"{dd.lower()}_{level or 'all'}"
            cp = inv_dir / f"microbe_counts_{base}.csv"
            if cp.exists():
                count_store[(level, dd)] = pd.read_csv(cp)
            op = inv_dir / f"original_auc_{base}.csv"
            if op.exists():
                orig_store[(level, dd)] = pd.read_csv(op).iloc[0].to_dict()
            _suffixes = (("_filter_only_raw", "_filter_only", "_full_raw", "_full")
                         if prefer_filter_only
                         else ("_full_raw", "_full", "_filter_only_raw", "_filter_only"))
            for suffix in _suffixes:
                candidate = inv_dir / f"auc_sweep_{base}{suffix}.csv"
                if candidate.exists():
                    raw_store[(level, dd)] = pd.read_csv(candidate)
                    break

    # ── Load CSVs — cross-dtype Combined ─────────────────────
    for level, _ in LEVELS:
        base = f"combined_{level or 'all'}"
        cp = inv_dir / f"microbe_counts_{base}.csv"
        if cp.exists():
            count_store[(level, "Combined")] = pd.read_csv(cp)
        op = inv_dir / f"original_auc_{base}.csv"
        if op.exists():
            orig_store[(level, "Combined")] = pd.read_csv(op).iloc[0].to_dict()
        _suffixes = (("_filter_only_raw", "_filter_only", "_full_raw", "_full")
                     if prefer_filter_only
                     else ("_full_raw", "_full", "_filter_only_raw", "_filter_only"))
        for suffix in _suffixes:
            candidate = inv_dir / f"auc_sweep_{base}{suffix}.csv"
            if candidate.exists():
                raw_store[(level, "Combined")] = pd.read_csv(candidate)
                break

    # ── Build panel_data ──────────────────────────────────────
    panel_data = {}
    for level, _ in LEVELS:
        # Pool Metagenomics + Amplicon raws for "All datasets"
        all_raws = [df for (l, d), df in raw_store.items()
                    if l == level and d in ("Metagenomics", "Amplicon")
                    and df is not None and not df.empty]
        all_raw  = pd.concat(all_raws, ignore_index=True) if all_raws else pd.DataFrame()

        for display_dtype in DISPLAY_DTYPES:
            if display_dtype == "All datasets":
                if all_raw.empty:
                    panel_data[(level, display_dtype)] = None
                    continue
                sweep_df  = (_merge_sweep_dfs(all_raw) if "mean_auc" in all_raw.columns
                             else _aggregate_raw_to_sweep(all_raw))
                count_dfs = [df for (l, d), df in count_store.items()
                             if l == level and d in ("Metagenomics", "Amplicon")]
                count_df  = _sum_count_dfs(count_dfs) if count_dfs else None
                orig_dcts = [d for (l, dd), d in orig_store.items()
                             if l == level and dd in ("Metagenomics", "Amplicon")]
                orig_auc  = _merge_orig_aucs(orig_dcts) if orig_dcts else {}
            else:
                raw_src = raw_store.get((level, display_dtype), pd.DataFrame())
                if raw_src is None or (hasattr(raw_src, "empty") and raw_src.empty):
                    panel_data[(level, display_dtype)] = None
                    continue
                sweep_df = (raw_src if "mean_auc" in raw_src.columns
                            else _aggregate_raw_to_sweep(raw_src))
                if sweep_df.empty:
                    panel_data[(level, display_dtype)] = None
                    continue
                count_df = count_store.get((level, display_dtype))
                orig_auc = orig_store.get((level, display_dtype), {})
                if count_df is None:
                    panel_data[(level, display_dtype)] = None
                    continue

            panel_data[(level, display_dtype)] = (sweep_df, count_df, orig_auc, None)

    # ── Compute display grid ──────────────────────────────────
    total_panels = nrows * ncols
    if nrows == 1 and ncols == 4:
        if force_single_row:
            grid_nrows, grid_ncols = 1, 4
        elif stacked:
            grid_nrows, grid_ncols = 4, 1
        else:
            grid_nrows, grid_ncols = 2, 2
    else:
        grid_nrows, grid_ncols = nrows, ncols

    # ── Create axes ───────────────────────────────────────────
    _standalone  = ax is None
    _LEGEND_FRAC = 0.13

    if _standalone:
        fig, _arr = plt.subplots(grid_nrows, grid_ncols,
                                 figsize=(14 * grid_ncols, 7 * grid_nrows), squeeze=False)
        axes = [list(row) for row in _arr]
    else:
        fig = ax.figure
        ax.set_visible(False)
        bb  = ax.get_position()
        if stacked:
            # Legend goes inside first panel — no right fraction needed
            gs = mgridspec.GridSpec(
                grid_nrows, grid_ncols, figure=fig,
                left=bb.x0, right=bb.x1,
                bottom=bb.y0, top=bb.y1,
                hspace=0.55, wspace=0,
            )
        else:
            gs = mgridspec.GridSpec(
                grid_nrows, grid_ncols, figure=fig,
                left=bb.x0, right=bb.x1 - bb.width * _LEGEND_FRAC,
                bottom=bb.y0, top=bb.y1,
                hspace=0.4, wspace=0.3,
            )
        axes = [[fig.add_subplot(gs[r, c]) for c in range(grid_ncols)] for r in range(grid_nrows)]

    _DTYPE_DISPLAY = {
        "Metagenomics": "WGS",
        "Amplicon":     "16S",
        "All datasets": "All dtypes",
        "Combined":     "Cross dtypes",
    }

    # ── Draw panels (flat enumeration → 2-D grid position) ───
    _legend_lines, _legend_labels = [], []
    panel_seq = [
        (level, level_label, display_dtype)
        for level, level_label in levels_to_plot
        for display_dtype in DISPLAY_DTYPES
    ]
    for flat_idx, (level, level_label, display_dtype) in enumerate(panel_seq):
        grid_r = flat_idx // grid_ncols
        grid_c = flat_idx % grid_ncols
        sub_ax = axes[grid_r][grid_c]
        data   = panel_data.get((level, display_dtype))
        if data is None:
            sub_ax.set_visible(False)
            continue
        sweep_df, count_df, orig_auc, fo_df = data
        _title = _DTYPE_DISPLAY.get(display_dtype, display_dtype)
        _thresh_map = thresholds if thresholds is not None else _BEST_THRESHOLDS
        result = _plot_sweep_panel(
            sub_ax, sweep_df, count_df,
            title=_title,
            original_auc=orig_auc,
            filter_only_df=fo_df,
            plot_mode=plot_mode,
            threshold_percentile=_thresh_map.get(display_dtype),
            show_left_yticks=(grid_c == 0),
            show_right_yticks=(grid_c == grid_ncols - 1),
        )
        if not _legend_lines and result:
            _legend_lines, _legend_labels = result

    # ── Stacked mode: hide x-ticks/labels on all but the bottom panel ──
    if stacked:
        for r in range(grid_nrows - 1):
            for c in range(grid_ncols):
                axes[r][c].tick_params(axis="x", which="both",
                                       bottom=False, labelbottom=False)
                axes[r][c].set_xlabel("")

    # ── Legend ────────────────────────────────────────────────
    if _legend_lines:
        if stacked and _standalone:
            # Legend at bottom of figure, 2 columns
            fig.legend(
                _legend_lines, _legend_labels,
                loc="lower center", bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                fontsize=30, framealpha=0.9,
                title="Legend", title_fontsize=32,
                borderpad=1.2, labelspacing=0.8,
            )
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.22)
        elif stacked:
            # Embedded stacked: legend inside the first (top) panel
            axes[0][0].legend(
                _legend_lines, _legend_labels,
                loc="upper right", fontsize=22, framealpha=0.9,
                title="Legend", title_fontsize=24,
                borderpad=1.2, labelspacing=0.8,
            )
        elif _standalone:
            plt.tight_layout(rect=[0, 0, 1 - _LEGEND_FRAC, 1])
            fig.legend(_legend_lines, _legend_labels,
                       loc="center right", bbox_to_anchor=(0.99, 0.5),
                       fontsize=22, framealpha=0.9,
                       title="Legend", title_fontsize=24,
                       borderpad=1.2, labelspacing=0.8)
        else:
            fig.legend(_legend_lines, _legend_labels,
                       loc="center left",
                       bbox_to_anchor=(bb.x1 - bb.width * _LEGEND_FRAC + 0.005,
                                       bb.y0 + bb.height / 2),
                       fontsize=10, framealpha=0.9,
                       title="Legend", title_fontsize=11)
    return fig


# ─────────────────────────────────────────────────────────────
# Figure 3C: distribution comparison
# ─────────────────────────────────────────────────────────────

def plot_figure3c(investigation_dir: str, ax=None, stacked: bool = False) -> plt.Figure:
    """
    Three-panel distribution comparison: Original vs RANK-BIRD.
    Panels: Metagenomics | Amplicon | All datasets.

    stacked=False (default): panels side by side (standalone figure)
    stacked=True           : panels stacked vertically (used in assembled figure)
    """
    from experiments.investigate_distribution_approach import (
        _make_combined_df, _draw_panel, APPROACH_LABELS,
    )

    inv_dir = Path(investigation_dir)
    dtypes  = ["Metagenomics", "Amplicon"]
    keep    = ["original", "rankbird"]
    keep_labels = {APPROACH_LABELS[a] for a in keep} | {"Random (0.5)"}

    def _fix_legend(sub_ax):
        handles, labels = sub_ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep_labels]
        if filtered:
            h, l = zip(*filtered)
            sub_ax.legend(handles=h, labels=l, fontsize=46, framealpha=0.9,
                          loc="lower right")

    # Load per-dtype results
    all_results: dict = {}
    for dtype in dtypes:
        for approach in keep:
            csv_path = inv_dir / f"results_{dtype.lower()}_{approach}.csv"
            if csv_path.exists():
                all_results[(dtype, approach)] = pd.read_csv(csv_path)
            else:
                print(f"  [SKIP] Missing {csv_path.name}")

    # Check for cross-dtype Combined results (produced when cross_dtype_normalization=True)
    has_cross = all(
        (inv_dir / f"results_combined_{a}.csv").exists() for a in keep
    )
    if has_cross:
        for approach in keep:
            df = pd.read_csv(inv_dir / f"results_combined_{approach}.csv")
            all_results[("Combined", approach)] = df

    # n_panels: Metagenomics | Amplicon | All datasets [| Combined]
    n_panels = 4 if has_cross else 3

    _standalone = ax is None
    if _standalone:
        if stacked:
            fig, _axes = plt.subplots(n_panels, 1, figsize=(8, 6 * n_panels), sharey=False)
        else:
            fig, _axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharey=True)
        axes = list(_axes)
    else:
        fig = ax.figure
        ax.set_visible(False)
        bb  = ax.get_position()
        if stacked:
            gap = 0.012
            ph  = (bb.height - (n_panels - 1) * gap) / n_panels
            axes = [
                fig.add_axes([bb.x0, bb.y1 - (i + 1) * ph - i * gap, bb.width, ph])
                for i in range(n_panels)
            ]
        else:
            gap = 0.015
            pw  = (bb.width - (n_panels - 1) * gap) / n_panels
            axes = [
                fig.add_axes([bb.x0 + i * (pw + gap), bb.y0, pw, bb.height])
                for i in range(n_panels)
            ]

    _panel_kwargs = dict(
        label_fontsize=104, title_fontsize=96,
        tick_fontsize=88, legend_fontsize=80,
        box_width=0.75, dot_size=8,
    )

    # ── Metagenomics and Amplicon panels ──────────────────────
    for sub_ax, dtype in zip(axes[:2], dtypes):
        frames = [
            all_results[(dtype, a)].assign(approach=a)
            for a in keep
            if (dtype, a) in all_results and not all_results[(dtype, a)].empty
        ]
        if frames:
            _DTYPE_LABEL_3C = {"Metagenomics": "WGS", "Amplicon": "16S"}
            _draw_panel(sub_ax, pd.concat(frames, ignore_index=True),
                        title=_DTYPE_LABEL_3C.get(dtype, dtype),
                        **_panel_kwargs)
            lgnd = sub_ax.get_legend()
            if lgnd:
                lgnd.remove()
        else:
            _DTYPE_LABEL_3C = {"Metagenomics": "WGS", "Amplicon": "16S"}
            sub_ax.set_title(f"{_DTYPE_LABEL_3C.get(dtype, dtype)} — no data", fontsize=11)

    # ── All datasets panel (Metagenomics + Amplicon pooled) ───
    combined_df = _make_combined_df(all_results, dtypes)
    if not combined_df.empty:
        combined_df = combined_df[combined_df["approach"].isin(keep)]
        _draw_panel(axes[2], combined_df, title="All dtypes", **_panel_kwargs)
        lgnd = axes[2].get_legend()
        if lgnd:
            lgnd.remove()
    else:
        axes[2].set_title("All dtypes — no data", fontsize=11)

    # ── Combined cross-dtype panel (optional) ─────────────────
    if has_cross:
        cross_frames = [
            all_results[("Combined", a)].assign(approach=a)
            for a in keep
            if ("Combined", a) in all_results and not all_results[("Combined", a)].empty
        ]
        if cross_frames:
            _draw_panel(axes[3], pd.concat(cross_frames, ignore_index=True),
                        title="Cross dtypes", **_panel_kwargs)
            _fix_legend(axes[3])
        else:
            axes[3].set_title("Cross dtypes — no data", fontsize=11)
    else:
        # Legend on the last visible panel
        _fix_legend(axes[2])

    if stacked:
        # xlabel "Protocol" only on bottom panel; ylabel "AUC" on all
        for sub_ax in axes[:-1]:
            sub_ax.set_xlabel("")
        axes[-1].set_xlabel("Protocol", fontsize=96, fontweight="bold")
    else:
        for sub_ax in axes:
            sub_ax.set_xlabel("")
        if _standalone:
            axes[n_panels // 2].set_xlabel("Protocol", fontsize=118, fontweight="bold")
        else:
            # Center "Protocol" across the full 3C area using a figure-level text
            _all_pos = [a.get_position() for a in axes]
            _cx = (_all_pos[0].x0 + _all_pos[-1].x1) / 2
            _cy = _all_pos[0].y0 - 0.04
            fig.text(_cx, _cy, "Protocol", fontsize=118, fontweight="bold",
                     ha="center", va="top")
        for sub_ax in axes[1:]:
            sub_ax.set_ylabel("")
            sub_ax.tick_params(axis="y", left=False, labelleft=False)
        axes[0].set_ylabel("AUC", fontsize=84, fontweight="bold")
        axes[0].tick_params(axis="y", labelsize=88)

    if _standalone:
        plt.tight_layout()
    return fig


def run_figure3c(investigation_dir: str, figures_dir: str) -> None:
    """Generate and save Figure 3C (standalone supplementary)."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "figure3c_original_vs_rankbird.png"
    if out_path.exists():
        print("  Skipping Figure 3C: already saved")
        return
    print("\nBuilding Figure 3C: Original vs RANK-BIRD")
    fig = plot_figure3c(investigation_dir)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 3C")


def run_figure3c_stats_table(
    investigation_dir: str,
    output_path: str,
) -> Optional[pd.DataFrame]:
    """
    Wilcoxon signed-rank test comparing Original vs CIFAR AUC.

    Iterates over every protocol × dtype combination:
      protocols : LODO, Internal Validation, Within Learning
      dtypes    : Metagenomics, Amplicon, All (pooled), Combined (if available)

    For each cell:
      - Aggregate per phenotype (mean + median AUC across datasets)
      - Pair phenotypes present in both approaches
      - Run scipy.stats.wilcoxon (two-sided)
    Then apply Benjamini-Hochberg FDR across all rows.

    Columns: dtype, protocol, n_phenotypes,
             mean_auc_original, mean_auc_rankbird, delta_mean,
             median_auc_original, median_auc_rankbird, delta_median,
             wilcoxon_statistic, p_value, p_value_fdr
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    inv_dir  = Path(investigation_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dtypes = ["Metagenomics", "Amplicon"]
    keep        = ["original", "rankbird"]
    protocols   = ["LODO", "Internal Validation", "Within Learning"]

    # ── Load per-dtype CSVs ───────────────────────────────────
    raw: dict = {}
    for dtype in base_dtypes:
        for approach in keep:
            csv_p = inv_dir / f"results_{dtype.lower()}_{approach}.csv"
            if csv_p.exists():
                df = pd.read_csv(csv_p)
                df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
                raw[(dtype, approach)] = df
            else:
                print(f"  [stats] Missing {csv_p.name} — skipping")

    has_cross = all((inv_dir / f"results_combined_{a}.csv").exists() for a in keep)
    if has_cross:
        for approach in keep:
            df = pd.read_csv(inv_dir / f"results_combined_{approach}.csv")
            df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
            raw[("Combined", approach)] = df

    def _phenotype_stats(df: pd.DataFrame, proto: str):
        """Return (mean_per_phenotype, median_per_phenotype) Series for a protocol."""
        sub = df[df["protocol"] == proto] if "protocol" in df.columns else df
        grp = sub.groupby("phenotype")["auc"]
        return grp.mean().dropna(), grp.median().dropna()

    def _anova_row(orig_df: pd.DataFrame, rb_df: pd.DataFrame,
                   proto: str, common_phenotypes) -> dict:
        """Two-way ANOVA: auc ~ C(phenotype) + C(approach), Type II.
        Uses dataset-level AUC values (not per-phenotype means) so that
        within-cell replication provides the residual error term."""
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        _nan = {"anova_F_approach": float("nan"), "anova_p_approach": float("nan"),
                "anova_F_phenotype": float("nan"), "anova_p_phenotype": float("nan"),
                "anova_df_approach": float("nan"), "anova_df_phenotype": float("nan"),
                "anova_df_residual": float("nan")}
        try:
            has_proto = "protocol" in orig_df.columns
            sub_o = (orig_df[orig_df["protocol"] == proto] if has_proto else orig_df).copy()
            sub_r = (rb_df[rb_df["protocol"]   == proto] if has_proto else rb_df).copy()
            sub_o = sub_o[sub_o["phenotype"].isin(common_phenotypes)][["phenotype", "auc"]].assign(approach="original")
            sub_r = sub_r[sub_r["phenotype"].isin(common_phenotypes)][["phenotype", "auc"]].assign(approach="CIFAR")
            adf = pd.concat([sub_o, sub_r], ignore_index=True)
            adf["auc"] = pd.to_numeric(adf["auc"], errors="coerce")
            adf = adf.dropna(subset=["auc"])
            n_pheno = adf["phenotype"].nunique()
            # Need at least 2 phenotypes, 2 approaches, and residual df > 0
            if n_pheno < 2 or adf["approach"].nunique() < 2 or len(adf) <= n_pheno + 1:
                return _nan
            model = ols("auc ~ C(phenotype) + C(approach)", data=adf).fit()
            tbl   = anova_lm(model, typ=2)
            return {
                "anova_F_approach":   float(tbl.loc["C(approach)",  "F"]),
                "anova_p_approach":   float(tbl.loc["C(approach)",  "PR(>F)"]),
                "anova_F_phenotype":  float(tbl.loc["C(phenotype)", "F"]),
                "anova_p_phenotype":  float(tbl.loc["C(phenotype)", "PR(>F)"]),
                "anova_df_approach":  float(tbl.loc["C(approach)",  "df"]),
                "anova_df_phenotype": float(tbl.loc["C(phenotype)", "df"]),
                "anova_df_residual":  float(tbl.loc["Residual",     "df"]),
            }
        except Exception:
            return _nan

    def _build_row(dtype_label: str, proto: str,
                   orig_df: pd.DataFrame, rb_df: pd.DataFrame) -> dict:
        orig_mean, orig_med = _phenotype_stats(orig_df, proto)
        rb_mean,   rb_med   = _phenotype_stats(rb_df,   proto)
        common = orig_mean.index.intersection(rb_mean.index)
        n = len(common)
        base = {"dtype": dtype_label, "protocol": proto, "n_phenotypes": n}
        _nan_anova = {"anova_F_approach": float("nan"), "anova_p_approach": float("nan"),
                      "anova_F_phenotype": float("nan"), "anova_p_phenotype": float("nan"),
                      "anova_df_approach": float("nan"), "anova_df_phenotype": float("nan"),
                      "anova_df_residual": float("nan")}
        if n < 2:
            return {**base,
                    "mean_auc_original": float("nan"), "mean_auc_cifar": float("nan"),
                    "delta_mean": float("nan"),
                    "median_auc_original": float("nan"), "median_auc_cifar": float("nan"),
                    "delta_median": float("nan"),
                    "wilcoxon_statistic": float("nan"), "p_value": float("nan"),
                    **_nan_anova}
        o_m = orig_mean[common].values
        r_m = rb_mean[common].values
        o_med = orig_med.reindex(common).values
        r_med = rb_med.reindex(common).values
        diff = r_m - o_m
        if (diff == 0).all():
            stat, p = float("nan"), float("nan")
        else:
            stat, p = wilcoxon(diff, alternative="two-sided")
        anova = _anova_row(orig_df, rb_df, proto, common)
        return {**base,
                "mean_auc_original":   float(o_m.mean()),
                "mean_auc_cifar":      float(r_m.mean()),
                "delta_mean":          float(r_m.mean() - o_m.mean()),
                "median_auc_original": float(np.nanmedian(o_med)),
                "median_auc_cifar":    float(np.nanmedian(r_med)),
                "delta_median":        float(np.nanmedian(r_med) - np.nanmedian(o_med)),
                "wilcoxon_statistic":  float(stat),
                "p_value":             float(p),
                **anova}

    rows = []
    for proto in protocols:
        # Metagenomics and Amplicon
        for dtype in base_dtypes:
            if (dtype, "original") in raw and (dtype, "rankbird") in raw:
                rows.append(_build_row(dtype, proto,
                                       raw[(dtype, "original")],
                                       raw[(dtype, "rankbird")]))

        # All pooled
        orig_frames = [raw[(d, "original")] for d in base_dtypes if (d, "original") in raw]
        rb_frames   = [raw[(d, "rankbird")] for d in base_dtypes if (d, "rankbird") in raw]
        if orig_frames and rb_frames:
            rows.append(_build_row("All", proto,
                                   pd.concat(orig_frames, ignore_index=True),
                                   pd.concat(rb_frames,   ignore_index=True)))

        # Combined cross-dtype
        if has_cross and ("Combined", "original") in raw and ("Combined", "rankbird") in raw:
            rows.append(_build_row("Combined", proto,
                                   raw[("Combined", "original")],
                                   raw[("Combined", "rankbird")]))

    if not rows:
        print("  [stats] No data found — table not saved")
        return None

    result_df = pd.DataFrame(rows)

    # ── Benjamini-Hochberg FDR across all rows ────────────────
    def _apply_fdr(df, col_in, col_out):
        mask = df[col_in].notna()
        fdr  = np.full(len(df), float("nan"))
        if mask.any():
            _, vals, _, _ = multipletests(df.loc[mask, col_in].values, method="fdr_bh")
            fdr[mask.values] = vals
        df[col_out] = fdr

    _apply_fdr(result_df, "p_value",        "p_value_fdr")
    _apply_fdr(result_df, "anova_p_approach",  "anova_p_approach_fdr")
    _apply_fdr(result_df, "anova_p_phenotype", "anova_p_phenotype_fdr")

    col_order = [
        "dtype", "protocol", "n_phenotypes",
        "mean_auc_original", "mean_auc_cifar", "delta_mean",
        "median_auc_original", "median_auc_cifar", "delta_median",
        "wilcoxon_statistic", "p_value", "p_value_fdr",
        "anova_df_approach", "anova_df_phenotype", "anova_df_residual",
        "anova_F_approach", "anova_p_approach", "anova_p_approach_fdr",
        "anova_F_phenotype", "anova_p_phenotype", "anova_p_phenotype_fdr",
    ]
    result_df = result_df[col_order]
    result_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Saved Figure 3C stats table → {out_path}")
    return result_df


# ─────────────────────────────────────────────────────────────
# Figure 3D: PCA scatter — raw vs normalized
# ─────────────────────────────────────────────────────────────

def plot_figure3d(
    phenotype_name: str,
    microbiome_dfs_by_dtype: Dict[str, List[pd.DataFrame]],
    target_dfs_by_dtype: Dict[str, List[pd.DataFrame]],
    dataset_names_by_dtype: Dict[str, List[str]],
    ax_left: Optional[plt.Axes] = None,
    ax_right: Optional[plt.Axes] = None,
    parent_fig: Optional[plt.Figure] = None,
    label_fontsize: int = 72,
    title_fontsize: int = 76,
    tick_fontsize: int = 60,
    legend_fontsize: int = 64,
) -> Optional[plt.Figure]:
    """
    Two-panel UMAP scatter for one phenotype: Raw (left) | RANK-BIRD (right).

    If ax_left / ax_right are provided, draws directly into those axes (no new
    figure is created) — use this from assemble_figure3 so text is vector, not
    a rasterised bitmap.  Otherwise creates and returns a standalone figure.
    """
    from src.rankbird.normalization.pipeline import apply_normalization_pipeline

    dtypes_present = [d for d in ["Metagenomics", "Amplicon"]
                      if d in microbiome_dfs_by_dtype]

    all_mb, all_tgt, all_names, all_dtype_labels, all_ds_labels = [], [], [], [], []
    for dtype in dtypes_present:
        for mb, tgt, name in zip(
            microbiome_dfs_by_dtype[dtype],
            target_dfs_by_dtype[dtype],
            dataset_names_by_dtype[dtype],
        ):
            all_mb.append(mb)
            all_tgt.append(tgt)
            all_names.append(name)
            all_dtype_labels.extend([dtype] * len(mb))
            all_ds_labels.extend([name] * len(mb))

    # Build per-dataset color palette: shades of blue for WGS, shades of red for 16S
    import matplotlib.cm as _cm
    dataset_color_map: dict = {}
    for dtype in dtypes_present:
        datasets = sorted(dataset_names_by_dtype[dtype])
        n = len(datasets)
        cmap = _cm.get_cmap("Blues" if dtype == "Metagenomics" else "Reds")
        for i, ds in enumerate(datasets):
            dataset_color_map[ds] = cmap(0.4 + 0.5 * i / max(1, n - 1))

    def _build_arrays(mb_list, tgt_list, dtype_labels, common_cols, ds_labels=None):
        X = pd.concat([df[common_cols] for df in mb_list], axis=0, ignore_index=True)
        tgt_arr = np.concatenate([
            t.iloc[:, 0].values if isinstance(t, pd.DataFrame) else t.values
            for t in tgt_list
        ])
        if ds_labels is not None:
            return X, tgt_arr, np.array(dtype_labels), np.array(ds_labels)
        return X, tgt_arr, np.array(dtype_labels)

    _direct = ax_left is not None and ax_right is not None
    if _direct:
        fig   = parent_fig
        axes  = [ax_left, ax_right]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(40, 17))

    _scatter_kw = dict(label_fontsize=label_fontsize,
                       title_fontsize=title_fontsize,
                       tick_fontsize=tick_fontsize)

    # ── Left: raw ─────────────────────────────────────────────
    common_raw = all_mb[0].columns
    for df in all_mb[1:]:
        common_raw = common_raw.intersection(df.columns)

    if len(common_raw) == 0:
        axes[0].text(0.5, 0.5, "No common features (raw)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Raw", fontsize=title_fontsize, fontweight="bold")
    else:
        X_raw, tgt_raw, dt_raw, ds_raw = _build_arrays(
            all_mb, all_tgt, all_dtype_labels, common_raw, all_ds_labels)
        _umap_scatter(axes[0], X_raw, tgt_raw, dt_raw, title="Raw",
                      dataset_labels=ds_raw, dataset_color_map=dataset_color_map,
                      **_scatter_kw)

    # ── Right: RANK-BIRD normalized ────────────────────────────
    print(f"  [3D] Normalizing {phenotype_name} for RANK-BIRD panel ...")
    try:
        norm_mb, norm_names = apply_normalization_pipeline(
            list(all_mb), list(all_names),
            min_samples_per_dataset=0,
        )
        name2dtype = {n: d for d in dtypes_present
                      for n in dataset_names_by_dtype[d]}
        name2tgt   = {n: t for d in dtypes_present
                      for n, t in zip(dataset_names_by_dtype[d],
                                      target_dfs_by_dtype[d])}
        norm_dtype_labels = []
        norm_ds_labels    = []
        norm_tgt_list     = []
        for mb_n, name_n in zip(norm_mb, norm_names):
            norm_dtype_labels.extend([name2dtype.get(name_n, "Unknown")] * len(mb_n))
            norm_ds_labels.extend([name_n] * len(mb_n))
            norm_tgt_list.append(name2tgt[name_n])

        common_norm = norm_mb[0].columns
        for df in norm_mb[1:]:
            common_norm = common_norm.intersection(df.columns)

        if len(common_norm) == 0:
            axes[1].text(0.5, 0.5, "No common features (normalized)",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("RANK-BIRD", fontsize=title_fontsize, fontweight="bold")
        else:
            X_norm, tgt_norm, dt_norm, ds_norm = _build_arrays(
                norm_mb, norm_tgt_list, norm_dtype_labels, common_norm, norm_ds_labels)
            _umap_scatter(axes[1], X_norm, tgt_norm, dt_norm, title="CIFAR",
                          dataset_labels=ds_norm, dataset_color_map=dataset_color_map,
                          **_scatter_kw)
            axes[1].set_ylabel("")
    except Exception as e:
        import traceback; traceback.print_exc()
        axes[1].text(0.5, 0.5, f"Normalization error:\n{e}",
                     ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=11, color="red")

    # ── Shared legend ──────────────────────────────────────────
    _DTYPE_LABEL_3D = {"Metagenomics": "WGS", "Amplicon": "16S"}
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_DTYPE_COLORS.get(d, "#888"),
                   markeredgecolor="none", markersize=36,
                   label=_DTYPE_LABEL_3D.get(d, d))
        for d in dtypes_present
    ] + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888888", markeredgecolor=_CASE_EDGE,
                   markeredgewidth=2.5, markersize=36, label="Case (border)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888888", markeredgecolor="none",
                   markersize=36, label="Control (no border)"),
    ]

    if _direct and parent_fig is not None:
        _pos_l = axes[0].get_position()
        _pos_r = axes[1].get_position()
        _cx = (_pos_l.x0 + _pos_r.x1) / 2
        _ty = max(_pos_l.y1, _pos_r.y1)
        parent_fig.legend(handles=legend_handles, loc="lower center",
                          ncol=len(legend_handles), fontsize=legend_fontsize,
                          framealpha=0.9,
                          bbox_to_anchor=(_cx, _ty + 0.015),
                          bbox_transform=parent_fig.transFigure)
        return None
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.subplots_adjust(wspace=0.45)
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(legend_handles), fontsize=legend_fontsize,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.90))
        return fig


def run_figure3d(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    normalization_approach: Optional[str] = "rankbird_wasserstein",
) -> None:
    """Generate and save Figure 3D PCA scatter for every phenotype."""
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out  = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_root)

    pheno_map: Dict[str, List[str]] = {}
    for phenotype, dtype in phenotypes:
        pheno_map.setdefault(phenotype, []).append(dtype)

    for phenotype, dtypes_for_pheno in pheno_map.items():
        safe     = phenotype.replace(" ", "_")
        out_path = out / f"figure3d_{safe}.png"
        if out_path.exists():
            print(f"  Skipping 3D for {phenotype}: already saved")
            continue

        print(f"\n{'='*50}\nFigure 3D: {phenotype}\n{'='*50}")
        mb_by_dtype, tgt_by_dtype, name_by_dtype = {}, {}, {}

        for dtype in dtypes_for_pheno:
            pheno_str = f"{phenotype} {dtype}"
            folder    = data / pheno_str
            if not folder.exists():
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
            except Exception as e:
                print(f"  Error loading {pheno_str}: {e}")
                continue
            if mb_dfs:
                mb_by_dtype[dtype]   = mb_dfs
                tgt_by_dtype[dtype]  = tgt_dfs
                name_by_dtype[dtype] = ds_names

        if not mb_by_dtype:
            print(f"  Skipping {phenotype}: no data loaded")
            continue

        try:
            fig = plot_figure3d(
                phenotype_name=phenotype,
                microbiome_dfs_by_dtype=mb_by_dtype,
                target_dfs_by_dtype=tgt_by_dtype,
                dataset_names_by_dtype=name_by_dtype,
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved Figure 3D for {phenotype}")
        except Exception as e:
            print(f"  Error generating 3D for {phenotype}: {e}")


# ─────────────────────────────────────────────────────────────
# Figure 3: assembled
# ─────────────────────────────────────────────────────────────

def assemble_figure3(
    path_3a: Optional[Path] = None,
    investigation_dir_3b: Optional[str] = None,
    investigation_dir_3c: Optional[str] = None,
    data_root_3d: Optional[str] = None,
    phenotypes_3d: Optional[List[Tuple[str, str]]] = None,
    normalization_approach: Optional[str] = "rankbird_wasserstein",
    phenotype_name_3d: str = "PD",
    path_3e: Optional[Path] = None,
    figsize: tuple = (130, 100),
    thresholds_3b: dict = None,
) -> plt.Figure:
    """
    Assemble Figure 3 panels into one combined figure.

    Layout
    ------
      ┌───────────────────┬─────────────┐
      │  3A  (big)        │  3B panel 1 │
      ├───────────────────┤  3B panel 2 │
      │  3C  (side x3)    │  3B panel 3 │
      │                   │  3B panel 4 │
      ├───────────────────┴─────────────┤
      │  3D  (left)   │  3E  (right)    │
      └───────────────────────────────--┘

    Parameters
    ----------
    path_3a              : path to schematic image (placeholder shown if missing)
    investigation_dir_3b : directory with stability-investigation CSVs for 3B
    investigation_dir_3c : directory with distribution-investigation CSVs for 3C
    data_root_3d         : data root for loading phenotype data for 3D
    phenotypes_3d        : list of (phenotype, dtype) tuples
    normalization_approach : passed to plot_figure3d
    phenotype_name_3d    : phenotype name to show in panel 3D (default "PD")
    path_3e              : path to manually provided image for panel E
    figsize              : figure size
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    fig = plt.figure(figsize=figsize)
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[22.0, 8.0],
        hspace=0.22,
        left=0.05, right=0.97, top=0.97, bottom=0.04,
    )

    # ── Top section: left col (A above C) | right col (B stacked) ──
    gs_top = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.5, 0.7], wspace=0.08,
    )
    # Left: A (big) above C
    gs_left = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_top[0, 0],
        height_ratios=[2.5, 1.0], hspace=0.16,
    )
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_c = fig.add_subplot(gs_left[1, 0])

    # Right: B stacked (placeholder; plot_figure3b creates its own sub-axes)
    ax_b = fig.add_subplot(gs_top[0, 1])

    # ── Bottom section: D (left) | E (right) ───────────────────
    gs_bot = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[1.5, 1.0], wspace=0.18,
    )
    ax_d = fig.add_subplot(gs_bot[0, 0])
    ax_e = fig.add_subplot(gs_bot[0, 1])

    # Compute all GridSpec positions upfront (they don't change after draw)
    _pos_a = ax_a.get_position()
    _pos_b = ax_b.get_position()
    _pos_c = ax_c.get_position()
    _pos_d = ax_d.get_position()

    # ── Panel A: schematic (row 0, narrowed) ─────────────────
    ax_a.set_axis_off()
    _p3a = Path(path_3a) if path_3a else None
    _img_a = (_p3a if (_p3a and _p3a.exists()) else
              _find_any_image(_p3a.parent if _p3a else Path("figures_out/figure_3/3a")))
    if _img_a:
        img = _load_any_image(_img_a)
        if img is not None:
            ax_a.imshow(img, aspect="auto", interpolation="bilinear")
    else:
        ax_a.set_facecolor("#E8E8E8")
        ax_a.text(0.5, 0.5,
                  "Schematic\n(place image in\nfigures_out/figure_3/3a/)",
                  ha="center", va="center", fontsize=13, color="#555555",
                  transform=ax_a.transAxes)
    fig.text(_pos_c.x0 - 0.04, _pos_a.y1, "A",
             fontsize=148, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel B: stability investigation (right col, stacked 4×1) ─
    ax_b.set_axis_off()
    if investigation_dir_3b and Path(investigation_dir_3b).exists():
        try:
            import io as _io
            fig_b = plot_figure3b(investigation_dir_3b,
                                   plot_levels=[None],
                                   plot_mode="mean",
                                   stacked=True,
                                   prefer_filter_only=True,
                                   thresholds=thresholds_3b)
            _buf  = _io.BytesIO()
            fig_b.savefig(_buf, dpi=150, bbox_inches="tight",
                          facecolor=fig_b.get_facecolor())
            plt.close(fig_b)
            _buf.seek(0)
            img_b = mpimg.imread(_buf)
            ax_b.imshow(img_b, aspect="auto", interpolation="bilinear")
        except Exception as e:
            print(f"  [3B] Error: {e}")
            ax_b.text(0.5, 0.5, f"3B error:\n{e}", ha="center", va="center",
                      transform=ax_b.transAxes, fontsize=10, color="#888")
    else:
        ax_b.set_facecolor("#E8E8E8")
        ax_b.text(0.5, 0.5,
                  "Stability investigation\n(run \"3b\" first to generate CSVs)",
                  ha="center", va="center", fontsize=13, color="#555555",
                  transform=ax_b.transAxes)
    fig.text(_pos_b.x0 - 0.01, _pos_a.y1, "B",
             fontsize=148, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel C: distribution comparison (row 2, full width) ─
    if investigation_dir_3c and Path(investigation_dir_3c).exists():
        try:
            plot_figure3c(investigation_dir_3c, ax=ax_c)
        except Exception as e:
            print(f"  [3C] Error: {e}")
            ax_c.text(0.5, 0.5, f"3C error:\n{e}", ha="center", va="center",
                      transform=ax_c.transAxes, fontsize=10, color="#888")
            ax_c.set_axis_off()
    else:
        ax_c.set_facecolor("#F0F0F0")
        ax_c.text(0.5, 0.5,
                  "Distribution comparison\n(run \"3c\" first to generate CSVs)",
                  ha="center", va="center", fontsize=13, color="#555555",
                  transform=ax_c.transAxes)
        ax_c.set_axis_off()
    fig.text(_pos_c.x0 - 0.04, _pos_c.y1 + 0.01, "C",
             fontsize=148, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel D: UMAP scatter (row 3, left) ───────────────────
    mb_by_dtype, tgt_by_dtype, name_by_dtype = {}, {}, {}

    if data_root_3d and phenotypes_3d:
        data_root_path = Path(data_root_3d)
        pheno_map: Dict[str, List[str]] = {}
        for pheno, dtype in phenotypes_3d:
            pheno_map.setdefault(pheno, []).append(dtype)

        dtypes_for_rep = pheno_map.get(phenotype_name_3d, [])
        for dtype in dtypes_for_rep:
            pheno_str = f"{phenotype_name_3d} {dtype}"
            folder    = data_root_path / pheno_str
            if not folder.exists():
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(
                    str(folder)
                )
                if mb_dfs:
                    mb_by_dtype[dtype]   = mb_dfs
                    tgt_by_dtype[dtype]  = tgt_dfs
                    name_by_dtype[dtype] = ds_names
            except Exception as e:
                print(f"  [3D] Error loading {pheno_str}: {e}")

    # Draw 3D directly into two sub-axes so text stays vector (not a bitmap)
    ax_d.set_axis_off()
    _d_mx   = 0.04   # horizontal inset margin (figure fraction)
    _d_my   = 0.03   # top inset margin (figure fraction)
    # D has x-tick labels + "t-SNE 1" xlabel below its axes box (~0.028 fig frac),
    # while E only has tick labels (~0.013 fig frac). Raise D by the difference
    # so both visible bottoms land at the same level as E's tick-label bottom.
    _d_yb   = 0.015  # bottom raise (figure fraction)
    _d_x0   = _pos_d.x0 + _d_mx
    _d_y0   = _pos_d.y0 + _d_yb
    _d_wt   = _pos_d.width  - 2 * _d_mx
    _d_ht   = _pos_d.height - _d_my - _d_yb
    _d_gap  = 0.015
    _d_w    = (_d_wt - _d_gap) / 2
    ax_d_left  = fig.add_axes([_d_x0,              _d_y0, _d_w, _d_ht])
    ax_d_right = fig.add_axes([_d_x0 + _d_w + _d_gap, _d_y0, _d_w, _d_ht])

    if mb_by_dtype:
        try:
            plot_figure3d(
                phenotype_name=phenotype_name_3d,
                microbiome_dfs_by_dtype=mb_by_dtype,
                target_dfs_by_dtype=tgt_by_dtype,
                dataset_names_by_dtype=name_by_dtype,
                ax_left=ax_d_left,
                ax_right=ax_d_right,
                parent_fig=fig,
                label_fontsize=96,
                title_fontsize=100,
                tick_fontsize=80,
                legend_fontsize=80,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [3D] Error: {e}")
            ax_d_left.text(0.5, 0.5, f"3D error:\n{e}", ha="center", va="center",
                           transform=ax_d_left.transAxes, fontsize=10, color="red")
    else:
        ax_d_left.set_axis_off()
        ax_d_right.set_axis_off()
        ax_d_left.set_facecolor("#F0F0F0")
        ax_d_left.text(0.5, 0.5,
                       f"UMAP scatter — {phenotype_name_3d}\n(no data loaded)",
                       ha="center", va="center", fontsize=13, color="#555555",
                       transform=ax_d_left.transAxes)
    fig.text(_pos_c.x0 - 0.04, _pos_d.y1 + 0.01, "D",
             fontsize=148, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # ── Panel E: taxa kept vs dropped bar chart ──────────────
    _kept    = {"saliv": 2, "oral": 1}
    _dropped = {"muri": 124, "saliv": 5, "oral": 73, "cutis": 8,
                "aqua": 5, "rhizo": 5, "roden": 4, "dental": 3}
    _cats    = sorted(set(_kept.keys()) | set(_dropped.keys()))
    _x       = np.arange(len(_cats))
    _w       = 0.4
    ax_e.bar(_x - _w / 2, [_kept.get(c, 0)    for c in _cats], _w,
             label="RM",  color="#98df8a", edgecolor="black", linewidth=1.2)
    ax_e.bar(_x + _w / 2, [_dropped.get(c, 0) for c in _cats], _w,
             label="CM", color="#ffbb78", edgecolor="black", linewidth=1.2)
    ax_e.set_xticks(_x)
    ax_e.set_xticklabels([c.capitalize() for c in _cats], fontsize=88)
    ax_e.set_ylabel("Number of taxa", fontsize=104)
    ax_e.tick_params(axis="y", labelsize=88)
    ax_e.legend(fontsize=80)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.text(-0.03, 1.05, "E", transform=ax_e.transAxes,
              fontsize=148, fontweight="bold", va="bottom", ha="right", clip_on=False)

    return fig
