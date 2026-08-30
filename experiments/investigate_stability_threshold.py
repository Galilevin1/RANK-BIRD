"""
Investigation: effect of stability-filter percentile on mean AUC.

For each percentile in [0.1, 0.2, ..., 0.9] and each taxonomic level:
  all                    — all microbes
  genus (g)              — g__<name> present, s__<name> absent
  genus+species (gs)     — g__<name> present (genus-only AND genus+species features)
  family+genus (fg)      — f__<name> AND g__<name> present, s__<name> absent
  family+genus+species (fgs) — f__<name> AND g__<name> present
  order+family+genus     — o__<name> AND f__<name> AND g__<name> present, s__<name> absent
  class+order+family+genus          — c__ AND o__ AND f__ AND g__ present, s__ absent
  phylum+class+order+family+genus   — p__ AND c__ AND o__ AND f__ AND g__ present, s__ absent
  kingdom+phylum+class+order+family+genus — k__ AND p__ AND c__ AND o__ AND f__ AND g__ present, s__ absent

Produces a figure (rows = levels, cols = Metagenomics / Amplicon).
Each subplot: AUC lines (left y-axis) + kept-microbe count & % (right y-axis).
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids tkinter threading errors
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.taxonomy_filter import filter_to_level, has_named as _has_named
from evaluation.data_loading import load_microbiome_datasets_with_targets
from experiments.run_protocols_global_processing import (
    _run_protocols_on_group,
    _run_global_for_dtype,
    _RANKING_NORM_FNS,
)
from src.rankbird.normalization.pipeline import apply_normalization_pipeline
from src.rankbird.normalization.ranking import apply_ranking_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "Data"

PERCENTILES = [round(p, 2) for p in np.arange(0.05, 1.0, 0.05)]
PROTOCOLS   = ["LODO", "Internal Validation", "Within Learning"]
PROTOCOL_COLORS = {
    "LODO":                "#1f77b4",
    "Internal Validation": "#2ca02c",
    "Within Learning":     "#ff7f0e",
}

# (filter_key, display_label)
LEVELS = [
    (None,    "All microbes"),
    # Top-down (phylum → genus)
    ("p",     "Phylum level"),
    ("pc",    "Phylum + Class level"),
    ("pco",   "Phylum + Class + Order level"),
    ("pcof",  "Phylum + Class + Order + Family level"),
    ("pcofg", "Phylum + Class + Order + Family + Genus level"),
    # Bottom-up (genus → family)
    ("g",     "Genus level"),
    ("gs",    "Genus + Species level"),
    ("fg",    "Family + Genus level"),
    ("fgs",   "Family + Genus + Species level"),
    ("ofg",   "Order + Family + Genus level"),
    ("cofg",  "Class + Order + Family + Genus level"),
]


# ── Taxonomy helpers — delegated to shared module ─────────────────────────────
_filter_to_level = filter_to_level


# ── Dataset loader ────────────────────────────────────────────────────────────

def _load_microbiome_for_dtype(phenotypes: list, dtype: str):
    all_dfs, all_names = [], []
    for phenotype, t in phenotypes:
        if dtype != "Combined" and t != dtype:
            continue
        folder = DATA_ROOT / f"{phenotype} {t}"
        dfs, _, names = load_microbiome_datasets_with_targets(folder)
        all_dfs.extend(dfs)
        all_names.extend(names)
    return all_dfs, all_names


# ── Microbe-count sweep (cheap — no model training) ──────────────────────────

def count_kept_microbes_sweep(
    phenotypes: list,
    dtype: str,
    level=None,
) -> pd.DataFrame:
    """
    For each percentile: how many microbes survive the stability filter?
    Returns: [percentile, n_kept, n_total, pct_kept]
    """
    dfs, names = _load_microbiome_for_dtype(phenotypes, dtype)
    dfs = _filter_to_level(dfs, level)

    all_microbes = union_microbes(dfs)
    n_total = len(all_microbes)
    nz_df   = nonzero_percent_by_dataset(dfs, names, all_microbes)

    rows = []
    for pct in PERCENTILES:
        kept = auto_stability_filter(nz_df, percentile=pct)
        rows.append({
            "percentile": pct,
            "n_kept":     len(kept),
            "n_total":    n_total,
            "pct_kept":   100.0 * len(kept) / n_total if n_total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ── AUC sweep ─────────────────────────────────────────────────────────────────

def _sweep_worker(
    pct, pheno_subset, level, norm_approach,
    shuffle_ordered, rank_tie_method, run_lr,
    lgbm_num_threads=0,
):
    """Compute one percentile step — called in parallel by run_stability_sweep."""
    import os
    if lgbm_num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(lgbm_num_threads)

    if level is None:
        results_df, _ = _run_global_for_dtype(
            pheno_subset,
            normalization_approach=norm_approach,
            stability_percentile_global=pct,
            shuffle_ordered=shuffle_ordered,
            rank_tie_method=rank_tie_method,
            run_lr=run_lr,
        )
    else:
        results_df = _run_global_for_dtype_filtered(
            pheno_subset,
            level=level,
            stability_percentile_global=pct,
            normalization_approach=norm_approach,
        )
    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    return pct, results_df


def run_stability_sweep(  # noqa: E302
    phenotypes: list,
    dtype: str,
    level=None,
    filter_only: bool = False,
    normalization_approach: str = "rankbird_wasserstein",
    shuffle_ordered: bool = False,
    rank_tie_method: str = "first",
    run_lr: bool = False,
    n_jobs: int = 1,
    lgbm_num_threads: int = 0,
) -> tuple:
    """
    Sweep stability percentiles, running all protocols at each step.
    filter_only=True applies only stability filtering (no distribution normalization).

    Returns
    -------
    agg_df : DataFrame [percentile, protocol, mean_auc, median_auc, std_auc, n_phenotypes]
    raw_df : DataFrame [percentile, phenotype, dataset, protocol, auc]
             Full per-dataset results at every percentile — use for re-plotting or
             alternative aggregations without recomputing.
    """
    pheno_subset = phenotypes if dtype == "Combined" else [(p, t) for p, t in phenotypes if t == dtype]
    if not pheno_subset:
        raise ValueError(f"No phenotypes for dtype='{dtype}'")

    agg_rows    = []
    raw_rows    = []
    lr_agg_rows = []
    lr_raw_rows = []

    def _agg_for(df, rows, raw_list, pct):
        if df.empty:
            return
        pct_col = df.copy()
        pct_col.insert(0, "percentile", pct)
        raw_list.append(pct_col)
        for protocol in PROTOCOLS:
            sub = df[df["protocol"] == protocol]["auc"].dropna()
            if sub.empty:
                continue
            rows.append({
                "percentile":   pct,
                "protocol":     protocol,
                "mean_auc":     sub.mean(),
                "median_auc":   sub.median(),
                "std_auc":      sub.std(ddof=1),
                "n_phenotypes": df[df["protocol"] == protocol]["phenotype"].nunique(),
            })

    norm_approach = "filter_only" if filter_only else normalization_approach
    print(f"  [{dtype}] level={level}  filter_only={filter_only}  n_jobs={n_jobs}  n_pct={len(PERCENTILES)}")

    pct_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_sweep_worker)(
            pct, pheno_subset, level, norm_approach,
            shuffle_ordered, rank_tie_method, run_lr,
            lgbm_num_threads,
        )
        for pct in PERCENTILES
    )

    for pct, results_df in sorted(pct_results, key=lambda x: x[0]):
        # Split LGBM vs LR if both present
        if "model" in results_df.columns:
            lgbm_df = results_df[results_df["model"] == "LGBM"]
            lr_df   = results_df[results_df["model"] == "LR"]
        else:
            lgbm_df = results_df
            lr_df   = pd.DataFrame()

        _agg_for(lgbm_df, agg_rows,    raw_rows,    pct)
        _agg_for(lr_df,   lr_agg_rows, lr_raw_rows, pct)

    agg_df    = pd.DataFrame(agg_rows)
    raw_df    = pd.concat(raw_rows,    ignore_index=True) if raw_rows    else pd.DataFrame()
    lr_agg_df = pd.DataFrame(lr_agg_rows)
    lr_raw_df = pd.concat(lr_raw_rows, ignore_index=True) if lr_raw_rows else pd.DataFrame()
    return agg_df, raw_df, lr_agg_df, lr_raw_df


def _run_global_for_dtype_filtered(
    phenotypes: list,
    level,
    stability_percentile_global: float = 0.5,
    min_samples_per_dataset: int = 550,
    normalization_approach: str = "rankbird_wasserstein",
) -> pd.DataFrame:
    """_run_global_for_dtype with column-level taxonomic filtering applied first."""
    all_microbiome, all_targets, all_names = [], [], []
    dataset_to_phenotype = {}

    for phenotype, dtype in phenotypes:
        phenotype_str = f"{phenotype} {dtype}"
        folder = DATA_ROOT / phenotype_str
        dfs, tgts, names = load_microbiome_datasets_with_targets(folder)
        for df, y, name in zip(dfs, tgts, names):
            all_microbiome.append(df)
            all_targets.append(y)
            all_names.append(name)
            dataset_to_phenotype[name] = phenotype_str

    # Taxonomic level filter
    all_microbiome = _filter_to_level(all_microbiome, level)

    # Reindex all datasets to the union of columns (mirrors what the stability
    # filter does), so datasets with no locally-present microbes at this level
    # get 0-filled columns instead of 0-column DataFrames.
    union_cols = union_microbes(all_microbiome)
    if not union_cols:
        print(f"  [SKIP] Level '{level}' removed all features for this dtype — no microbes with all required taxonomy levels named.")
        return pd.DataFrame()
    all_microbiome = [df.reindex(columns=union_cols, fill_value=0.0) for df in all_microbiome]

    # name→target map (before normalisation which may drop datasets)
    name_to_target = dict(zip(all_names, all_targets))

    if normalization_approach == "filter_only":
        all_microbes = union_microbes(all_microbiome)
        nz_df = nonzero_percent_by_dataset(all_microbiome, all_names, all_microbes)
        kept = auto_stability_filter(nz_df, percentile=stability_percentile_global)
        all_microbiome = [df.reindex(columns=kept, fill_value=0.0) for df in all_microbiome]

    elif normalization_approach == "rankbird_wasserstein":
        all_microbiome, all_names = apply_normalization_pipeline(
            all_microbiome,
            all_names,
            global_analysis=True,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_global=stability_percentile_global,
        )

    elif normalization_approach in _RANKING_NORM_FNS:
        norm_fn = _RANKING_NORM_FNS[normalization_approach]
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome,
            all_names,
            stability_percentile=stability_percentile_global,
            norm_fn=norm_fn,
            min_size=min_samples_per_dataset,
        )

    # None → no normalization, data passes through as-is

    aligned_targets = [name_to_target[n] for n in all_names if n in name_to_target]

    # Split back by phenotype and run protocols
    records = []
    for phenotype_str in set(dataset_to_phenotype.values()):
        idx = [
            i for i, name in enumerate(all_names)
            if dataset_to_phenotype.get(name) == phenotype_str
        ]
        if not idx:
            continue
        records.append(_run_protocols_on_group(
            [all_microbiome[i] for i in idx],
            [aligned_targets[i] for i in idx],
            [all_names[i] for i in idx],
            phenotype_str,
        ))

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def get_original_mean_auc(
    phenotypes: list,
    dtype: str,
    level=None,
) -> dict:
    """
    Run protocols on raw (non-normalized) data, optionally filtered to a
    taxonomic level.  Returns {protocol: mean_auc} for use as reference lines.
    """
    pheno_subset = phenotypes if dtype == "Combined" else [(p, t) for p, t in phenotypes if t == dtype]
    if not pheno_subset:
        return {}

    if level is None:
        results_df, _ = _run_global_for_dtype(
            pheno_subset, normalization_approach=None
        )
    else:
        results_df = _run_global_for_dtype_filtered(
            pheno_subset, level=level, normalization_approach=None
        )

    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    return {
        protocol: results_df[results_df["protocol"] == protocol]["auc"].dropna().mean()
        for protocol in PROTOCOLS
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_sweep_lines(ax, sweep_df, protocol, color, plot_mode, suffix="", linewidth=2, alpha=1.0):
    """Plot mean/median sweep lines for one protocol onto ax."""
    show_mean   = plot_mode in ("mean",   "combined")
    show_median = plot_mode in ("median", "combined")
    sub = sweep_df[sweep_df["protocol"] == protocol].sort_values("percentile")
    if sub.empty:
        return
    if show_mean:
        label = f"{protocol}{suffix}" if not show_median else f"{protocol} mean{suffix}"
        ax.plot(sub["percentile"], sub["mean_auc"],
                marker="o", linewidth=linewidth, markersize=9,
                color=color, alpha=alpha, label=label)
        ax.fill_between(
            sub["percentile"],
            sub["mean_auc"] - sub["std_auc"],
            sub["mean_auc"] + sub["std_auc"],
            alpha=0.12 * alpha, color=color,
        )
    if show_median and "median_auc" in sub.columns:
        label = f"{protocol}{suffix}" if not show_mean else f"{protocol} median{suffix}"
        ax.plot(sub["percentile"], sub["median_auc"],
                marker="^", linewidth=linewidth,
                markersize=8, linestyle="-" if not show_mean else "--",
                color=color, alpha=alpha,
                label=label)


def _plot_sweep_panel(
    ax: plt.Axes,
    sweep_df: pd.DataFrame,
    count_df: pd.DataFrame,
    title: str,
    original_auc: dict = None,
    filter_only_df: pd.DataFrame = None,
    plot_mode: str = "combined",
    threshold_percentile: float = None,
    show_left_yticks: bool = True,
    show_right_yticks: bool = True,
):
    """AUC lines (left axis) + kept-microbe count & % annotations (right axis).

    Three layers per protocol (same color, different line style/weight):
      - original (no filter, no norm): horizontal dotted line
      - filter-only sweep:             thin solid line  (if filter_only_df provided)
      - full normalization sweep:      thick solid line
    """
    for protocol in PROTOCOLS:
        color = PROTOCOL_COLORS[protocol]

        # 1. Original baseline — horizontal dashed line
        if original_auc and protocol in original_auc:
            ax.axhline(
                original_auc[protocol],
                color=color, linestyle="--", linewidth=3.5, alpha=0.85,
                label=f"{protocol} (original)",
            )

        # 2. Filter-only sweep — thin line (only in combined mode)
        if filter_only_df is not None:
            _plot_sweep_lines(ax, filter_only_df, protocol, color, plot_mode,
                              suffix=" (filter only)", linewidth=3.0, alpha=0.65)

        # 3. Main sweep — thick line; label with "(full)" only when both are shown
        main_suffix = " (full)" if filter_only_df is not None else ""
        _plot_sweep_lines(ax, sweep_df, protocol, color, plot_mode,
                          suffix=main_suffix, linewidth=5.0, alpha=1.0)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=2.5, alpha=0.75,
               label="Random (0.5)")
    if threshold_percentile is not None:
        ax.axvline(threshold_percentile, color="#555555", linestyle="--",
                   linewidth=2.5, alpha=0.75, zorder=1)
        if count_df is not None and not count_df.empty:
            closest = count_df.iloc[
                (count_df["percentile"] - threshold_percentile).abs().argsort().iloc[0]
            ]
            ax.text(
                threshold_percentile + 0.012, 0.04,
                f"{int(closest['n_kept'])} microbes",
                fontsize=20, color="#555555", va="bottom", ha="left",
                transform=ax.get_xaxis_transform(),
            )
    _ylabel = {"mean": "Mean AUC", "median": "Median AUC", "combined": "Mean / Median AUC"}
    ax.set_xlabel("Stability Percentile", fontsize=32)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", rotation=45, labelsize=26)
    ax.set_xlim(PERCENTILES[0] - 0.02, PERCENTILES[-1] + 0.02)
    ax.set_ylim(top=1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.45, linewidth=0.9)
    ax.set_title(title, fontsize=32, fontweight="bold", pad=12)
    if show_left_yticks:
        ax.set_ylabel(_ylabel.get(plot_mode, "AUC"), fontsize=32)
        ax.tick_params(axis="y", labelsize=26)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    # Right axis: kept-microbe count line (no per-point annotations)
    ax2 = ax.twinx()
    if count_df is not None and not count_df.empty:
        ax2.plot(count_df["percentile"], count_df["n_kept"],
                 color="black", linestyle="--", linewidth=4.5,
                 marker="s", markersize=7, label="RM")
    if show_right_yticks:
        ax2.set_ylabel("RM", fontsize=32, color="black")
        ax2.tick_params(axis="y", labelcolor="black", labelsize=26)
    else:
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", right=False, labelright=False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    return lines1 + lines2, labels1 + labels2


# ── Display helpers ──────────────────────────────────────────────────────────

def _merge_sweep_dfs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple already-aggregated sweep DataFrames (e.g. Metagenomics + Amplicon)
    into one by computing n_phenotypes-weighted averages per (percentile, protocol).
    """
    rows = []
    for (pct, prot), grp in df.groupby(['percentile', 'protocol']):
        total_n = grp['n_phenotypes'].sum()
        if total_n == 0:
            continue
        rows.append({
            'percentile':   pct,
            'protocol':     prot,
            'mean_auc':     float((grp['mean_auc']   * grp['n_phenotypes']).sum() / total_n),
            'median_auc':   float((grp['median_auc'] * grp['n_phenotypes']).sum() / total_n),
            'std_auc':      float(grp['std_auc'].mean()),
            'n_phenotypes': int(total_n),
        })
    return pd.DataFrame(rows)


def _aggregate_raw_to_sweep(raw_df: pd.DataFrame, dtype_filter: str = None) -> pd.DataFrame:
    """
    Re-aggregate per-dataset raw sweep results into mean/median/std format for plotting.
    dtype_filter: if "Amplicon" or "Metagenomics", keeps only phenotypes whose name
    contains that word (e.g. "AD Amplicon", "ASD Metagenomics").
    """
    df = raw_df.copy()
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    if dtype_filter:
        df = df[df["phenotype"].str.contains(dtype_filter, na=False)]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for pct in sorted(df["percentile"].unique()):
        pct_df = df[df["percentile"] == pct]
        for protocol in PROTOCOLS:
            sub = pct_df[pct_df["protocol"] == protocol]["auc"].dropna()
            if sub.empty:
                continue
            rows.append({
                "percentile":   pct,
                "protocol":     protocol,
                "mean_auc":     float(sub.mean()),
                "median_auc":   float(sub.median()),
                "std_auc":      float(sub.std(ddof=1)) if len(sub) > 1 else 0.0,
                "n_phenotypes": int(pct_df[pct_df["protocol"] == protocol]["phenotype"].nunique()),
            })
    return pd.DataFrame(rows)


def _sum_count_dfs(count_dfs: list) -> pd.DataFrame:
    """Combine per-dtype count DataFrames by summing n_kept and n_total per percentile."""
    merged = pd.concat(count_dfs, ignore_index=True)
    result = (
        merged.groupby("percentile", as_index=False)
        .agg(n_kept=("n_kept", "sum"), n_total=("n_total", "sum"))
    )
    result["pct_kept"] = 100.0 * result["n_kept"] / result["n_total"].replace(0, np.nan)
    return result.reset_index(drop=True)


def _merge_orig_aucs(orig_dicts: list) -> dict:
    """Average per-protocol original AUC values across multiple dtype dicts."""
    from collections import defaultdict
    sums, counts = defaultdict(float), defaultdict(int)
    for d in orig_dicts:
        for protocol, val in d.items():
            if pd.notna(val):
                sums[protocol] += float(val)
                counts[protocol] += 1
    return {p: sums[p] / counts[p] for p in sums if counts[p] > 0}


# ── Intermediate / per-sweep figure helper ───────────────────────────────────

def _save_interim_figure(
    level, level_label, dtype, raw_df, count_df, orig_auc, fo_raw_df,
    figures_dir, mode_suffix, plot_modes, mode_meta, model_label="LGBM",
):
    """Save a single-panel figure right after one (level, dtype) sweep completes."""
    if raw_df is None or raw_df.empty:
        return
    sweep_df = raw_df if 'mean_auc' in raw_df.columns else _aggregate_raw_to_sweep(raw_df)
    if sweep_df.empty:
        return
    fo_df = None
    if fo_raw_df is not None and not fo_raw_df.empty:
        fo_df = fo_raw_df if 'mean_auc' in fo_raw_df.columns else _aggregate_raw_to_sweep(fo_raw_df)
    level_str = level or "all"
    for pm in plot_modes:
        _, stat_suffix = mode_meta.get(pm, (pm, f"_{pm}"))
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        _plot_sweep_panel(
            ax, sweep_df, count_df,
            title=f"{dtype} — {level_label} [{model_label}]",
            original_auc=orig_auc or {},
            filter_only_df=fo_df,
            plot_mode=pm,
        )
        out_path = figures_dir / f"interim_{dtype.lower()}_{level_str}{mode_suffix}{stat_suffix}_{model_label.lower()}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [interim] {out_path.name}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stability_investigation(
    phenotypes: list,
    output_dir: Path,
    plot_only: bool = False,
    dtype_filter: list = None,
    normalization_modes: list = None,
    plot_mode = "combined",
    plot_levels: list = None,
    compute_levels: list = None,
    normalization_approach: str = "rankbird_wasserstein",
    cross_dtype_normalization: bool = False,
    stability_percentile_global_combined: float = 0.6,
    taxonomy_level_combined=None,
    figures_dir: Path = None,
    shuffle_ordered: bool = False,
    rank_tie_method: str = "first",
    run_lr: bool = False,
    n_jobs: int = 1,
    lgbm_num_threads: int = 0,
):
    """
    Runs the full (dtype × level) sweep and saves CSVs + figure(s) per mode.

    Parameters
    ----------
    phenotypes         : list of (phenotype, dtype) tuples
    output_dir         : directory for CSVs and figures
    plot_only          : if True, skip computation and load existing CSVs
    dtype_filter       : limit to these dtypes, e.g. ["Metagenomics"]. None = all.
                         Ignored when cross_dtype_normalization=True.
    normalization_modes: list of norm modes, e.g. ["full"], ["filter_only"], or both.
    plot_mode          : "mean", "median", "combined", or a list of these to get
                         one figure per mode. Default: "combined".
    plot_levels        : subset of level keys to include in the figure, e.g. ["g", "fg", "ofg"].
                         None = plot all levels in LEVELS.
    compute_levels     : subset of level keys to compute, e.g. [None, "g"].
                         None = compute all levels in LEVELS. Other levels load from disk if available,
                         and are skipped (not plotted) if CSVs are missing.
    cross_dtype_normalization : if True, pool 16S + shotgun together for normalization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir is not None else output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    if normalization_modes is None:
        normalization_modes = ["full"]

    # Computation dtypes (determines which sweep CSVs are produced)
    if cross_dtype_normalization:
        dtypes = ["Combined"]
    else:
        all_dtypes = ["Metagenomics", "Amplicon"]
        dtypes = [d for d in all_dtypes if dtype_filter is None or d in dtype_filter]

    # Figure always shows these 3 columns regardless of computation mode
    DISPLAY_DTYPES = ["Metagenomics", "Amplicon", "All datasets"]

    _mode_meta = {
        "mean":     ("Mean AUC",          "_mean"),
        "median":   ("Median AUC",        "_median"),
        "combined": ("Mean / Median AUC", "_combined"),
    }
    plot_modes     = plot_mode if isinstance(plot_mode, list) else [plot_mode]
    levels_to_plot = [(l, lbl) for l, lbl in LEVELS
                      if plot_levels is None or l in plot_levels]

    for norm_mode in normalization_modes:
        combined_mode = (norm_mode == "full+filter_only")
        filter_only   = (norm_mode == "filter_only")
        mode_suffix   = f"_{norm_mode.replace('+', '_')}" if norm_mode != "full" else ""
        mode_label    = {
            "full":             "Full normalization",
            "filter_only":      "Filter-only",
            "full+filter_only": "Full normalization + Filter-only",
        }.get(norm_mode, norm_mode)

        # raw_store[(level, dtype)]       → LGBM raw sweep DataFrame
        # fo_raw_store[(level, dtype)]    → LGBM filter-only raw sweep DataFrame
        # lr_raw_store[(level, dtype)]    → LR raw sweep DataFrame
        # lr_fo_raw_store[(level, dtype)] → LR filter-only raw sweep DataFrame
        # count_store[(level, dd)]        → microbe count sweep DataFrame
        # orig_store[(level, dd)]         → orig_auc dict
        raw_store       = {}
        fo_raw_store    = {}
        lr_raw_store    = {}
        lr_fo_raw_store = {}
        count_store     = {}
        orig_store      = {}

        for level, level_label in LEVELS:
            should_compute = compute_levels is None or level in compute_levels

            # ── Always compute per-actual-dtype count and baseline ─────────────
            # These are lightweight (count = no training; orig = one pass without filter)
            # and needed to populate the Amplicon / Metagenomics display panels.
            for dd in ["Amplicon", "Metagenomics"]:
                pheno_dd = [(p, t) for p, t in phenotypes if t == dd]
                if not pheno_dd:
                    continue
                base_dd = f"{dd.lower()}_{level or 'all'}"

                count_path_dd = output_dir / f"microbe_counts_{base_dd}.csv"
                if count_path_dd.exists():
                    count_store[(level, dd)] = pd.read_csv(count_path_dd)
                elif should_compute and not plot_only:
                    cdf = count_kept_microbes_sweep(phenotypes, dd, level=level)
                    cdf.to_csv(count_path_dd, index=False)
                    count_store[(level, dd)] = cdf

                orig_path_dd = output_dir / f"original_auc_{base_dd}.csv"
                if orig_path_dd.exists():
                    orig_store[(level, dd)] = pd.read_csv(orig_path_dd).iloc[0].to_dict()
                elif should_compute and not plot_only:
                    oa = get_original_mean_auc(phenotypes, dd, level=level)
                    pd.DataFrame([oa]).to_csv(orig_path_dd, index=False)
                    orig_store[(level, dd)] = oa

            # ── Compute / load sweep for each computation dtype ────────────────
            for dtype in dtypes:
                base = f"{dtype.lower()}_{level or 'all'}"
                print(f"\n=== {mode_label}  |  {level_label}  |  {dtype} ===")

                sweep_full_path = output_dir / f"auc_sweep_{base}_full.csv"
                sweep_fo_path   = output_dir / f"auc_sweep_{base}_filter_only.csv"
                sweep_path      = sweep_fo_path if filter_only else sweep_full_path
                raw_path        = sweep_path.with_name(sweep_path.stem + "_raw.csv")

                if plot_only:
                    load_path = raw_path if raw_path.exists() else sweep_path
                    if load_path.exists():
                        raw_store[(level, dtype)] = pd.read_csv(load_path)
                    else:
                        print(f"  [SKIP] CSV missing ({sweep_path.name}) — recompute without plot_only.")
                    lr_raw_path = raw_path.with_name(raw_path.stem + "_lr.csv")
                    if lr_raw_path.exists():
                        lr_raw_store[(level, dtype)] = pd.read_csv(lr_raw_path)
                    if combined_mode:
                        fo_raw_path = sweep_fo_path.with_name(sweep_fo_path.stem + "_raw.csv")
                        fo_load_path = fo_raw_path if fo_raw_path.exists() else sweep_fo_path
                        if fo_load_path.exists():
                            fo_raw_store[(level, dtype)] = pd.read_csv(fo_load_path)
                        fo_lr_raw_path = fo_raw_path.with_name(fo_raw_path.stem + "_lr.csv")
                        if fo_lr_raw_path.exists():
                            lr_fo_raw_store[(level, dtype)] = pd.read_csv(fo_lr_raw_path)
                else:
                    if sweep_path.exists():
                        print(f"  [LOAD] {sweep_path.name}")
                        load_path = raw_path if raw_path.exists() else sweep_path
                        raw_store[(level, dtype)] = pd.read_csv(load_path)
                        lr_raw_path = raw_path.with_name(raw_path.stem + "_lr.csv")
                        if lr_raw_path.exists():
                            lr_raw_store[(level, dtype)] = pd.read_csv(lr_raw_path)
                    elif should_compute:
                        print(f"  Computing {'filter-only' if filter_only else 'full normalization'} sweep ...")
                        sweep_df, raw_df, lr_sweep_df, lr_raw_df = run_stability_sweep(
                            phenotypes, dtype, level=level,
                            filter_only=filter_only,
                            normalization_approach=normalization_approach,
                            shuffle_ordered=shuffle_ordered,
                            rank_tie_method=rank_tie_method,
                            run_lr=run_lr,
                            n_jobs=n_jobs,
                            lgbm_num_threads=lgbm_num_threads,
                        )
                        sweep_df.to_csv(sweep_path, index=False)
                        raw_df.to_csv(raw_path, index=False)
                        raw_store[(level, dtype)] = raw_df
                        if not lr_sweep_df.empty:
                            lr_sweep_df.to_csv(sweep_path.with_name(sweep_path.stem + "_lr.csv"), index=False)
                            lr_raw_df.to_csv(raw_path.with_name(raw_path.stem + "_lr.csv"), index=False)
                            lr_raw_store[(level, dtype)] = lr_raw_df
                    else:
                        continue

                    if combined_mode:
                        fo_raw_path = sweep_fo_path.with_name(sweep_fo_path.stem + "_raw.csv")
                        if sweep_fo_path.exists() and fo_raw_path.exists():
                            print(f"  [LOAD] {sweep_fo_path.name}")
                            fo_raw_store[(level, dtype)] = pd.read_csv(fo_raw_path)
                            fo_lr_raw_path = fo_raw_path.with_name(fo_raw_path.stem + "_lr.csv")
                            if fo_lr_raw_path.exists():
                                lr_fo_raw_store[(level, dtype)] = pd.read_csv(fo_lr_raw_path)
                        elif should_compute:
                            print(f"  Computing filter-only sweep ...")
                            fo_df, fo_raw_df, fo_lr_df, fo_lr_raw_df = run_stability_sweep(
                                phenotypes, dtype, level=level, filter_only=True,
                                shuffle_ordered=shuffle_ordered,
                                rank_tie_method=rank_tie_method,
                                run_lr=run_lr,
                                n_jobs=n_jobs,
                                lgbm_num_threads=lgbm_num_threads,
                            )
                            fo_df.to_csv(sweep_fo_path, index=False)
                            fo_raw_df.to_csv(fo_raw_path, index=False)
                            fo_raw_store[(level, dtype)] = fo_raw_df
                            if not fo_lr_df.empty:
                                fo_lr_df.to_csv(sweep_fo_path.with_name(sweep_fo_path.stem + "_lr.csv"), index=False)
                                fo_lr_raw_df.to_csv(fo_raw_path.with_name(fo_raw_path.stem + "_lr.csv"), index=False)
                                lr_fo_raw_store[(level, dtype)] = fo_lr_raw_df

                # ── Intermediate figure for this (level, dtype) ───────────────
                _save_interim_figure(
                    level, level_label, dtype,
                    raw_store.get((level, dtype)),
                    count_store.get((level, dtype)),
                    orig_store.get((level, dtype), {}),
                    fo_raw_store.get((level, dtype)),
                    figures_dir, mode_suffix, plot_modes, _mode_meta, "LGBM",
                )
                if (level, dtype) in lr_raw_store:
                    _save_interim_figure(
                        level, level_label, dtype,
                        lr_raw_store.get((level, dtype)),
                        count_store.get((level, dtype)),
                        orig_store.get((level, dtype), {}),
                        lr_fo_raw_store.get((level, dtype)),
                        figures_dir, mode_suffix, plot_modes, _mode_meta, "LR",
                    )

        # ── Build display panel_data (always 3 columns) ───────────────────────
        panel_data = {}  # (level, display_dtype) → (sweep_df, count_df, orig_auc, fo_df)

        for level, level_label in LEVELS:
            # Pool all raw data across computation dtypes for this level
            all_raws    = [df for (l, d), df in raw_store.items()
                           if l == level and df is not None and not df.empty]
            all_fo_raws = [df for (l, d), df in fo_raw_store.items()
                           if l == level and df is not None and not df.empty]
            all_raw    = pd.concat(all_raws,    ignore_index=True) if all_raws    else pd.DataFrame()
            all_fo_raw = pd.concat(all_fo_raws, ignore_index=True) if all_fo_raws else pd.DataFrame()

            for display_dtype in DISPLAY_DTYPES:
                if display_dtype == "All datasets":
                    if all_raw.empty:
                        panel_data[(level, display_dtype)] = None
                        continue
                    if 'mean_auc' in all_raw.columns:
                        sweep_df = _merge_sweep_dfs(all_raw)
                    else:
                        sweep_df = _aggregate_raw_to_sweep(all_raw)
                    count_dfs = [df for (l, d), df in count_store.items() if l == level]
                    count_df  = _sum_count_dfs(count_dfs) if count_dfs else None
                    orig_dcts = [d for (l, dd), d in orig_store.items() if l == level]
                    orig_auc  = _merge_orig_aucs(orig_dcts) if orig_dcts else {}
                    if not all_fo_raw.empty and combined_mode:
                        fo_df = _merge_sweep_dfs(all_fo_raw) if 'mean_auc' in all_fo_raw.columns else _aggregate_raw_to_sweep(all_fo_raw)
                    else:
                        fo_df = None

                else:
                    # For cross_dtype=True the combined raw contains all phenotypes;
                    # for cross_dtype=False each dtype has its own raw.
                    if cross_dtype_normalization:
                        raw_src    = all_raw
                        fo_raw_src = all_fo_raw
                        filt       = display_dtype   # filter combined raw by dtype name
                    else:
                        raw_src    = raw_store.get((level, display_dtype), pd.DataFrame())
                        fo_raw_src = fo_raw_store.get((level, display_dtype), pd.DataFrame())
                        filt       = None            # already dtype-specific, no filter needed

                    if raw_src is None or raw_src.empty:
                        panel_data[(level, display_dtype)] = None
                        continue

                    if 'mean_auc' in raw_src.columns:
                        sweep_df = raw_src
                    else:
                        sweep_df = _aggregate_raw_to_sweep(raw_src, dtype_filter=filt)
                    if sweep_df.empty:
                        panel_data[(level, display_dtype)] = None
                        continue

                    count_df = count_store.get((level, display_dtype))
                    orig_auc = orig_store.get((level, display_dtype), {})
                    fo_df    = None
                    if combined_mode and fo_raw_src is not None and not fo_raw_src.empty:
                        if 'mean_auc' in fo_raw_src.columns:
                            fo_df = fo_raw_src
                        else:
                            fo_df = _aggregate_raw_to_sweep(fo_raw_src, dtype_filter=filt)

                    if count_df is None:
                        panel_data[(level, display_dtype)] = None
                        continue

                panel_data[(level, display_dtype)] = (sweep_df, count_df, orig_auc, fo_df)

        # ── Draw figures (always 3 columns) ───────────────────────────────────
        for pm in plot_modes:
            stat_title, stat_suffix = _mode_meta.get(pm, (pm, f"_{pm}"))

            fig, axes = plt.subplots(
                len(levels_to_plot), len(DISPLAY_DTYPES),
                figsize=(14 * len(DISPLAY_DTYPES), 7 * len(levels_to_plot)),
                squeeze=False,
            )

            _legend_lines, _legend_labels = [], []
            for row_idx, (level, level_label) in enumerate(levels_to_plot):
                for col_idx, display_dtype in enumerate(DISPLAY_DTYPES):
                    ax   = axes[row_idx][col_idx]
                    data = panel_data.get((level, display_dtype))
                    if data is None:
                        ax.set_visible(False)
                        continue
                    sweep_df, count_df, orig_auc, fo_df = data
                    _thresh = _BEST_THRESHOLDS.get(display_dtype)
                    result = _plot_sweep_panel(ax, sweep_df, count_df,
                                      title=f"{display_dtype} — {level_label}",
                                      original_auc=orig_auc,
                                      filter_only_df=fo_df,
                                      plot_mode=pm,
                                      threshold_percentile=_thresh)
                    if not _legend_lines and result:
                        _legend_lines, _legend_labels = result
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            if _legend_lines:
                fig.legend(
                    _legend_lines, _legend_labels,
                    loc="center right",
                    bbox_to_anchor=(0.99, 0.5),
                    fontsize=14,
                    framealpha=0.9,
                    title="Legend",
                    title_fontsize=15,
                )
            out_path = figures_dir / f"stability_threshold_investigation{mode_suffix}{stat_suffix}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

        # ── LR grid figure (same layout, separate file) ───────────────────────
        if lr_raw_store:
            lr_panel_data = {}
            for level, level_label in LEVELS:
                all_lr_raws    = [df for (l, d), df in lr_raw_store.items()
                                  if l == level and df is not None and not df.empty]
                all_lr_fo_raws = [df for (l, d), df in lr_fo_raw_store.items()
                                  if l == level and df is not None and not df.empty]
                all_lr_raw    = pd.concat(all_lr_raws,    ignore_index=True) if all_lr_raws    else pd.DataFrame()
                all_lr_fo_raw = pd.concat(all_lr_fo_raws, ignore_index=True) if all_lr_fo_raws else pd.DataFrame()

                for display_dtype in DISPLAY_DTYPES:
                    if display_dtype == "All datasets":
                        if all_lr_raw.empty:
                            lr_panel_data[(level, display_dtype)] = None
                            continue
                        sweep_df = _merge_sweep_dfs(all_lr_raw) if 'mean_auc' in all_lr_raw.columns else _aggregate_raw_to_sweep(all_lr_raw)
                        count_dfs = [df for (l, d), df in count_store.items() if l == level]
                        count_df  = _sum_count_dfs(count_dfs) if count_dfs else None
                        orig_dcts = [d for (l, dd), d in orig_store.items() if l == level]
                        orig_auc  = _merge_orig_aucs(orig_dcts) if orig_dcts else {}
                        fo_df     = (_merge_sweep_dfs(all_lr_fo_raw) if 'mean_auc' in all_lr_fo_raw.columns
                                     else _aggregate_raw_to_sweep(all_lr_fo_raw)) if (not all_lr_fo_raw.empty and combined_mode) else None
                    else:
                        if cross_dtype_normalization:
                            raw_src    = all_lr_raw
                            fo_raw_src = all_lr_fo_raw
                            filt       = display_dtype
                        else:
                            raw_src    = lr_raw_store.get((level, display_dtype), pd.DataFrame())
                            fo_raw_src = lr_fo_raw_store.get((level, display_dtype), pd.DataFrame())
                            filt       = None
                        if raw_src is None or (hasattr(raw_src, 'empty') and raw_src.empty):
                            lr_panel_data[(level, display_dtype)] = None
                            continue
                        sweep_df = raw_src if 'mean_auc' in raw_src.columns else _aggregate_raw_to_sweep(raw_src, dtype_filter=filt)
                        if sweep_df.empty:
                            lr_panel_data[(level, display_dtype)] = None
                            continue
                        count_df = count_store.get((level, display_dtype))
                        orig_auc = orig_store.get((level, display_dtype), {})
                        fo_df    = None
                        if combined_mode and fo_raw_src is not None and not (hasattr(fo_raw_src, 'empty') and fo_raw_src.empty):
                            fo_df = fo_raw_src if 'mean_auc' in fo_raw_src.columns else _aggregate_raw_to_sweep(fo_raw_src, dtype_filter=filt)
                        if count_df is None:
                            lr_panel_data[(level, display_dtype)] = None
                            continue
                    lr_panel_data[(level, display_dtype)] = (sweep_df, count_df, orig_auc, fo_df)

            for pm in plot_modes:
                stat_title, stat_suffix = _mode_meta.get(pm, (pm, f"_{pm}"))
                fig, axes = plt.subplots(
                    len(levels_to_plot), len(DISPLAY_DTYPES),
                    figsize=(14 * len(DISPLAY_DTYPES), 7 * len(levels_to_plot)),
                    squeeze=False,
                )
                _legend_lines, _legend_labels = [], []
                for row_idx, (level, level_label) in enumerate(levels_to_plot):
                    for col_idx, display_dtype in enumerate(DISPLAY_DTYPES):
                        ax   = axes[row_idx][col_idx]
                        data = lr_panel_data.get((level, display_dtype))
                        if data is None:
                            ax.set_visible(False)
                            continue
                        sweep_df, count_df, orig_auc, fo_df = data
                        _thresh = _BEST_THRESHOLDS.get(display_dtype)
                        result = _plot_sweep_panel(ax, sweep_df, count_df,
                                          title=f"{display_dtype} — {level_label} [LR]",
                                          original_auc=orig_auc,
                                          filter_only_df=fo_df,
                                          plot_mode=pm,
                                          threshold_percentile=_thresh)
                        if not _legend_lines and result:
                            _legend_lines, _legend_labels = result
                plt.tight_layout(rect=[0, 0, 0.82, 1])
                if _legend_lines:
                    fig.legend(
                        _legend_lines, _legend_labels,
                        loc="center right",
                        bbox_to_anchor=(0.99, 0.5),
                        fontsize=14,
                        framealpha=0.9,
                        title="Legend",
                        title_fontsize=15,
                    )
                out_path = figures_dir / f"stability_threshold_investigation{mode_suffix}{stat_suffix}_lr.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Microbe characterization at best thresholds
# ═══════════════════════════════════════════════════════════════════════════════

_BEST_THRESHOLDS = {
    "Metagenomics": 0.1,
    "Amplicon":     0.3,
    "Combined":     0.3,
}


def _get_taxonomy_level(feature: str) -> str:
    for prefix, label in [("s__", "species"), ("g__", "genus"),
                           ("f__", "family"),  ("o__", "order"),
                           ("c__", "class"),   ("p__", "phylum")]:
        if _has_named(feature, prefix):
            return label
    return "unknown"


_LEVEL_PREFIX = {
    "species": "s__", "genus": "g__", "family": "f__",
    "order":   "o__", "class": "c__", "phylum": "p__",
}


def _extract_level_name(feature: str, level: str) -> str:
    """Extract the taxon name at the given level from a feature string."""
    prefix = _LEVEL_PREFIX.get(level)
    if prefix is None:
        return feature
    m = re.search(re.escape(prefix) + r'(\w+)', feature)
    return m.group(1) if m else feature


def _extract_readable_taxa_names(microbes: list) -> list:
    """
    Convert a list of full taxonomy strings to readable genus / genus+species names.
    - Has named genus AND named species → species field content (e.g. 'Lactobacillus acidophilus')
    - Has named genus only             → genus field content  (e.g. 'Prevotella')
    - Neither                          → skip
    Underscores replaced with spaces; result is sorted and deduplicated.
    """
    names = set()
    for feature in microbes:
        has_genus   = _has_named(feature, "g__")
        has_species = _has_named(feature, "s__")
        if not has_genus:
            continue
        prefix = "s__" if has_species else "g__"
        m = re.search(re.escape(prefix) + r"([^;]+)", feature)
        if m:
            readable = re.sub(r"_+", " ", m.group(1)).strip()
            if readable:
                names.add(readable)
    return sorted(names)


def _build_taxa_counts(stats_df: pd.DataFrame, levels: list) -> pd.DataFrame:
    """
    For each microbe, extract its taxon name at its assigned level.
    Returns a long-form DataFrame: [level, taxon_name, status, count].
    """
    rows = []
    for lvl in levels:
        if lvl == "unknown":
            continue
        sub = stats_df[stats_df["taxonomy_level"] == lvl]
        for status in ("kept", "dropped"):
            names = (
                sub[sub["status"] == status]["microbe"]
                .apply(lambda m: _extract_level_name(m, lvl))
            )
            for name, cnt in names.value_counts().items():
                rows.append({"level": lvl, "taxon_name": name, "status": status, "count": cnt})
    return pd.DataFrame(rows)


def _count_unique_taxa(df_group: pd.DataFrame, levels: list) -> pd.Series:
    """For each taxonomy level, count unique taxon names in df_group."""
    counts = {}
    for lvl in levels:
        sub = df_group[df_group["taxonomy_level"] == lvl]
        if sub.empty or lvl == "unknown":
            counts[lvl] = int(sub.shape[0])
            continue
        counts[lvl] = sub["microbe"].apply(
            lambda m: _extract_level_name(m, lvl)
        ).nunique()
    return pd.Series(counts, dtype=int)


def _shannon_diversity(row: np.ndarray) -> float:
    row = row[row > 0]
    if len(row) == 0:
        return 0.0
    p = row / row.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _compute_per_microbe_stats(
    microbiome_dfs: list,
    target_dfs: list,
    dataset_names: list,
    all_microbes: list,
    kept_set: set,
) -> pd.DataFrame:
    """
    Build a per-microbe DataFrame with sparsity, abundance, prevalence,
    CV (stability metric), taxonomy level, MI with target, and kept/dropped label.
    """
    from sklearn.feature_selection import mutual_info_classif

    # Pool all samples
    frames = [df.reindex(columns=all_microbes, fill_value=0.0)
              for df in microbiome_dfs]
    combined = pd.concat(frames, axis=0, ignore_index=True)

    # Per-microbe MI across all datasets
    mi_sums   = {m: [] for m in all_microbes}
    for df, target in zip(microbiome_dfs, target_dfs):
        X = df.reindex(columns=all_microbes, fill_value=0.0)
        y = (target.values if hasattr(target, "values") else np.array(target)).ravel()
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        try:
            mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
            for i, m in enumerate(all_microbes):
                mi_sums[m].append(mi[i])
        except Exception:
            pass

    # CV per microbe across datasets (non-zero fraction per dataset)
    nz_df = nonzero_percent_by_dataset(microbiome_dfs, dataset_names, all_microbes)
    means = nz_df.mean(axis=0)
    stds  = nz_df.std(axis=0, ddof=1)
    cv    = (stds / means.replace(0, np.nan)).fillna(0)

    rows = []
    for m in all_microbes:
        col = combined[m]
        rows.append({
            "microbe":               m,
            "status":                "kept" if m in kept_set else "dropped",
            "taxonomy_level":        _get_taxonomy_level(m),
            "nonzero_fraction":       float((col != 0).mean()),
            "mean_nonzero_abund":    float(col[col > 0].mean()) if (col > 0).any() else 0.0,
            "std_abund":             float(col.std()),
            "prevalence_datasets":   float((nz_df[m] > 0).mean()),
            "cv_across_datasets":    float(cv[m]),
            "mean_MI_with_target":   float(np.mean(mi_sums[m])) if mi_sums[m] else 0.0,
        })

    return pd.DataFrame(rows)


def _plot_characterization(stats_df: pd.DataFrame, dtype: str, output_dir: Path):
    """Six-panel figure comparing kept vs dropped microbes."""
    kept    = stats_df[stats_df["status"] == "kept"]
    dropped = stats_df[stats_df["status"] == "dropped"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Kept vs Dropped Microbes — {dtype}", fontsize=14, fontweight="bold")

    def _box(ax, col, title, ylabel):
        data = [kept[col].dropna().values, dropped[col].dropna().values]
        bp = ax.boxplot(data, labels=["Kept", "Dropped"], patch_artist=True,
                        medianprops=dict(color="red", linewidth=2))
        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][1].set_facecolor("#F9A825")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    _box(axes[0, 0], "nonzero_fraction",      "Non-zero fraction",               "Fraction non-zero")
    _box(axes[0, 1], "cv_across_datasets",  "CV across datasets (stability)",  "CV (std/mean)")
    _box(axes[0, 2], "mean_MI_with_target", "Mean MI with target",             "MI")
    _box(axes[1, 0], "mean_nonzero_abund",  "Mean non-zero abundance",         "Abundance")
    _box(axes[1, 1], "prevalence_datasets", "Prevalence (fraction datasets)",  "Fraction datasets")

    # Taxonomy level distribution — unique taxon names per level
    ax = axes[1, 2]
    levels = ["species", "genus", "family", "order", "class", "phylum"]
    kept_counts    = _count_unique_taxa(kept,    levels).reindex(levels, fill_value=0)
    dropped_counts = _count_unique_taxa(dropped, levels).reindex(levels, fill_value=0)
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w/2, kept_counts.values,    w, label="Kept",    color="#AED6F1")
    ax.bar(x + w/2, dropped_counts.values, w, label="Dropped", color="#F9A825")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=9)
    ax.set_title("Taxonomy level distribution (unique taxa)", fontsize=11, fontweight="bold")
    ax.set_ylabel("# unique taxon names", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"characterization_{dtype.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_alpha_diversity(
    microbiome_dfs: list,
    all_microbes: list,
    kept_set: set,
    dtype: str,
    output_dir: Path,
):
    """Compare per-sample Shannon diversity using all vs kept-only microbes."""
    all_cols  = list(all_microbes)
    kept_cols = [m for m in all_microbes if m in kept_set]

    div_all, div_kept = [], []
    for df in microbiome_dfs:
        X_all  = df.reindex(columns=all_cols,  fill_value=0.0).values
        X_kept = df.reindex(columns=kept_cols, fill_value=0.0).values
        div_all.extend( [_shannon_diversity(r) for r in X_all])
        div_kept.extend([_shannon_diversity(r) for r in X_kept])

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([div_all, div_kept],
                    labels=["All microbes", "Kept microbes"],
                    patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    bp["boxes"][0].set_facecolor("#D5F5E3")
    bp["boxes"][1].set_facecolor("#AED6F1")
    ax.set_title(f"Per-sample Shannon diversity — {dtype}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Shannon index", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"alpha_diversity_{dtype.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Statistical tests & feature importance ────────────────────────────────────

_CONTINUOUS_METRICS = [
    "nonzero_fraction",
    "cv_across_datasets",
    "mean_MI_with_target",
    "mean_nonzero_abund",
    "prevalence_datasets",
]


def _run_statistical_tests(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each continuous metric: Mann-Whitney U test (kept vs dropped).
    Returns a DataFrame with columns:
        metric, U_stat, p_value, p_fdr, significant, median_kept, median_dropped, effect_direction
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    kept    = stats_df[stats_df["status"] == "kept"]
    dropped = stats_df[stats_df["status"] == "dropped"]

    rows = []
    for metric in _CONTINUOUS_METRICS:
        k = kept[metric].dropna().values
        d = dropped[metric].dropna().values
        if len(k) < 2 or len(d) < 2:
            rows.append({"metric": metric, "U_stat": np.nan, "p_value": np.nan,
                         "median_kept": np.nan, "median_dropped": np.nan,
                         "effect_direction": "n/a"})
            continue
        U, p = mannwhitneyu(k, d, alternative="two-sided")
        direction = "kept > dropped" if np.median(k) > np.median(d) else "kept < dropped"
        rows.append({
            "metric":           metric,
            "U_stat":           U,
            "p_value":          p,
            "median_kept":      float(np.median(k)),
            "median_dropped":   float(np.median(d)),
            "effect_direction": direction,
        })

    result = pd.DataFrame(rows)
    valid  = result["p_value"].notna()
    if valid.any():
        _, p_fdr, _, _ = multipletests(result.loc[valid, "p_value"], method="fdr_bh")
        result.loc[valid, "p_fdr"] = p_fdr
    result["significant"] = result.get("p_fdr", result["p_value"]) < 0.05
    return result


def _run_importance_model(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train logistic regression and random forest to predict kept/dropped.
    Returns a DataFrame with columns: metric, logistic_coef (abs), rf_importance, mean_importance.
    Rows sorted by mean_importance descending.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    df = stats_df[_CONTINUOUS_METRICS + ["status"]].dropna()
    X  = df[_CONTINUOUS_METRICS].values
    y  = (df["status"] == "kept").astype(int).values

    if len(np.unique(y)) < 2 or len(y) < 10:
        return pd.DataFrame({"metric": _CONTINUOUS_METRICS})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic regression on standardised features
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    lr     = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_sc, y)
    lr_coef = np.abs(lr.coef_[0])
    lr_auc_scores = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        X, y, cv=cv, scoring="roc_auc"
    )

    # Random forest
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_auc_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X, y, cv=cv, scoring="roc_auc"
    )

    # Normalise both to [0, 1] for a fair combined rank
    lr_norm = lr_coef / (lr_coef.sum() + 1e-12)
    rf_norm = rf_imp  / (rf_imp.sum()  + 1e-12)

    result = pd.DataFrame({
        "metric":          _CONTINUOUS_METRICS,
        "logistic_coef":   lr_coef,
        "rf_importance":   rf_imp,
        "lr_norm":         lr_norm,
        "rf_norm":         rf_norm,
        "mean_importance": (lr_norm + rf_norm) / 2,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    # Attach AUC scores as metadata columns (same value repeated for convenience)
    result["lr_auc_mean"] = lr_auc_scores.mean()
    result["lr_auc_std"]  = lr_auc_scores.std()
    result["rf_auc_mean"] = rf_auc_scores.mean()
    result["rf_auc_std"]  = rf_auc_scores.std()

    return result


def _plot_statistical_summary(
    stats_df: pd.DataFrame,
    test_results: pd.DataFrame,
    importance_df: pd.DataFrame,
    dtype: str,
    output_dir: Path,
):
    """
    Two-panel figure:
      Left  — feature importance (LR + RF bars, sorted by mean importance)
      Right — significance table (metric, medians, p_fdr, direction)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Kept vs Dropped Microbes — Statistical Analysis ({dtype})",
        fontsize=14, fontweight="bold",
    )

    # ── Left: importance bars ────────────────────────────────────────────────
    ax = axes[0]
    if "logistic_coef" in importance_df.columns:
        metrics = importance_df["metric"].tolist()
        x = np.arange(len(metrics))
        w = 0.35
        ax.barh(x + w/2, importance_df["lr_norm"], w,
                label="Logistic Regression (|coef|)", color="#5DADE2", alpha=0.85)
        ax.barh(x - w/2, importance_df["rf_norm"], w,
                label="Random Forest (importance)", color="#F39C12", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(metrics, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised importance", fontsize=11)

        # AUC subtitle
        if "lr_auc_mean" in importance_df.columns:
            lr_auc = importance_df["lr_auc_mean"].iloc[0]
            lr_std = importance_df["lr_auc_std"].iloc[0]
            rf_auc = importance_df["rf_auc_mean"].iloc[0]
            rf_std = importance_df["rf_auc_std"].iloc[0]
            auc_str = (f"5-fold CV AUC — LR: {lr_auc:.3f} ± {lr_std:.3f}  |  "
                       f"RF: {rf_auc:.3f} ± {rf_std:.3f}")
        else:
            auc_str = ""
        ax.set_title(
            f"Feature importance for kept/dropped prediction\n{auc_str}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

    # ── Right: significance table ────────────────────────────────────────────
    ax = axes[1]
    ax.axis("off")

    _METRIC_LABELS = {
        "nonzero_fraction":    "Non-zero\nfraction",
        "cv_across_datasets":  "CV across\ndatasets",
        "mean_MI_with_target": "MI with target\n(mean/dataset)",
        "mean_nonzero_abund":  "Non-zero abund.\n(mean/sample)",
        "prevalence_datasets": "Prevalence\n(frac. datasets)",
    }

    tr = test_results.copy()
    tr["p_fdr"]  = tr.get("p_fdr", pd.Series(np.nan, index=tr.index))
    tr["sig"]    = tr["significant"].map({True: "**", False: ""}).fillna("")

    col_labels = ["Metric", "Median\n(kept)", "Median\n(dropped)", "p (FDR)", "Sig", "Direction"]
    table_data = []
    for _, row in tr.iterrows():
        table_data.append([
            _METRIC_LABELS.get(row["metric"], row["metric"]),
            f"{row['median_kept']:.4f}"    if pd.notna(row.get("median_kept"))    else "—",
            f"{row['median_dropped']:.4f}" if pd.notna(row.get("median_dropped")) else "—",
            f"{row['p_fdr']:.2e}"          if pd.notna(row.get("p_fdr"))          else "—",
            row["sig"],
            row.get("effect_direction", ""),
        ])

    col_widths = [0.28, 0.13, 0.13, 0.12, 0.06, 0.22]
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    # Highlight significant rows
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D6EAF8")
            cell.set_text_props(fontweight="bold")
        elif r > 0 and r <= len(table_data):
            sig_val = table_data[r - 1][4]
            cell.set_facecolor("#D5F5E3" if sig_val else "white")

    ax.set_title("Mann-Whitney U test (FDR-corrected)", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = output_dir / f"statistical_summary_{dtype.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_top_taxa_per_level(
    stats_df: pd.DataFrame,
    dtype: str,
    output_dir: Path,
    top_n: int = 15,
):
    """
    For each taxonomy level: horizontal grouped bar chart of the top_n most
    common specific taxon names in kept vs dropped microbes.

    E.g. at family level it shows how many microbes carrying 'Lachnospiraceae'
    are kept vs dropped.
    """
    levels = ["species", "genus", "family", "order", "class", "phylum"]
    taxa_df = _build_taxa_counts(stats_df, levels)

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle(
        f"Top taxa per level: Kept vs Dropped ({dtype})",
        fontsize=14, fontweight="bold",
    )

    for ax, lvl in zip(axes.flat, levels):
        sub = taxa_df[taxa_df["level"] == lvl]
        if sub.empty:
            ax.set_title(lvl, fontsize=11)
            ax.axis("off")
            continue

        # Pivot to wide: index=taxon_name, cols=kept/dropped
        wide = (
            sub.pivot_table(index="taxon_name", columns="status",
                            values="count", aggfunc="sum", fill_value=0)
            .assign(total=lambda d: d.get("kept", 0) + d.get("dropped", 0))
            .sort_values("total", ascending=False)
            .head(top_n)
            .drop(columns="total")
            .sort_values("kept" if "kept" in sub["status"].values else sub["status"].iloc[0],
                         ascending=True)   # ascending so top is at top of chart
        )

        taxa   = wide.index.tolist()
        kept_v = wide.get("kept",    pd.Series(0, index=wide.index)).values
        drop_v = wide.get("dropped", pd.Series(0, index=wide.index)).values

        y = np.arange(len(taxa))
        h = 0.35
        ax.barh(y + h/2, kept_v, h, label="Kept",    color="#AED6F1", alpha=0.9)
        ax.barh(y - h/2, drop_v, h, label="Dropped", color="#F9A825", alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(taxa, fontsize=8)
        ax.set_xlabel("# microbe features", fontsize=9)
        ax.set_title(f"{lvl.capitalize()} level (top {top_n})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"top_taxa_per_level_{dtype.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_microbe_characterization(
    phenotypes: list,
    output_dir: Path,
    threshold_metagenomics: float = None,
    threshold_amplicon: float = None,
    plot_only: bool = False,
    taxonomy_level_metagenomics=None,
    taxonomy_level_amplicon=None,
    cross_dtype_normalization: bool = False,
    stability_percentile_global_combined: float = 0.6,
    taxonomy_level_combined=None,
):
    """
    Step 2: characterize kept vs dropped microbes at the best stability threshold.

    Parameters
    ----------
    phenotypes             : list of (phenotype, dtype) tuples
    output_dir             : directory to write outputs
    threshold_metagenomics : stability percentile for Metagenomics (default 0.25)
    threshold_amplicon     : stability percentile for Amplicon (default 0.40)
    plot_only              : if True, reload existing CSVs and only regenerate figures
    cross_dtype_normalization : if True, pool 16S + shotgun together for characterization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cross_dtype_normalization:
        tax_level = taxonomy_level_combined if str(taxonomy_level_combined) != "None" else None
        thresholds = {"Combined": stability_percentile_global_combined}
        taxonomy_levels = {"Combined": tax_level}
    else:
        thresholds = {
            "Metagenomics": threshold_metagenomics or _BEST_THRESHOLDS["Metagenomics"],
            "Amplicon":     threshold_amplicon     or _BEST_THRESHOLDS["Amplicon"],
        }
        taxonomy_levels = {
            "Metagenomics": taxonomy_level_metagenomics,
            "Amplicon":     taxonomy_level_amplicon,
        }

    for dtype, threshold in thresholds.items():
        print(f"\n=== Characterization: {dtype}  (threshold={threshold}) ===")

        stats_path = output_dir / f"microbe_stats_{dtype.lower()}.csv"

        if plot_only:
            if not stats_path.exists():
                print(f"  [SKIP] {stats_path} not found — run without plot_only first.")
                continue
            print("  [plot_only] Loading existing microbe stats ...")
            stats_df = pd.read_csv(stats_path)

            # Load raw data only for alpha diversity (cheap I/O, no MI)
            all_microbiome, all_names = _load_microbiome_for_dtype(phenotypes, dtype)
            all_microbes = union_microbes(all_microbiome)
            kept_set     = set(stats_df.loc[stats_df["status"] == "kept", "microbe"])

        else:
            # Load all datasets for this dtype
            all_microbiome, all_names = _load_microbiome_for_dtype(phenotypes, dtype)
            # Also load targets for MI
            all_targets = []
            for phenotype, t in phenotypes:
                if t != dtype:
                    continue
                folder = DATA_ROOT / f"{phenotype} {t}"
                _, tgts, names = load_microbiome_datasets_with_targets(folder)
                all_targets.extend(tgts)

            # Taxonomy level filter (applied before stability filter)
            all_microbiome = filter_to_level(all_microbiome, taxonomy_levels[dtype])

            # Compute stability filter
            all_microbes = union_microbes(all_microbiome)
            nz_df        = nonzero_percent_by_dataset(all_microbiome, all_names, all_microbes)
            kept_list    = auto_stability_filter(nz_df, percentile=threshold)
            kept_set     = set(kept_list)
            dropped_list = [m for m in all_microbes if m not in kept_set]

            # Save microbe lists
            pd.DataFrame({"microbe": kept_list,    "status": "kept"}).to_csv(
                output_dir / f"kept_microbes_{dtype.lower()}.csv", index=False)
            pd.DataFrame({"microbe": dropped_list, "status": "dropped"}).to_csv(
                output_dir / f"dropped_microbes_{dtype.lower()}.csv", index=False)
            print(f"  Kept: {len(kept_list)}  |  Dropped: {len(dropped_list)}")

            # Per-microbe statistics
            print("  Computing per-microbe statistics + MI ...")
            stats_df = _compute_per_microbe_stats(
                all_microbiome, all_targets, all_names, all_microbes, kept_set
            )
            stats_df.to_csv(stats_path, index=False)

        # Text files: readable genus / genus+species names (runs in both modes)
        kept_for_txt    = [m for m in all_microbes if m in kept_set]
        dropped_for_txt = [m for m in all_microbes if m not in kept_set]
        for label, microbe_list in [("kept", kept_for_txt), ("dropped", dropped_for_txt)]:
            names = _extract_readable_taxa_names(microbe_list)
            txt_path = output_dir / f"{label}_taxa_names_{dtype.lower()}.txt"
            txt_path.write_text("\n".join(names), encoding="utf-8")
            print(f"  Saved {len(names)} readable names -> {txt_path.name}")

        # Characterization plots
        _plot_characterization(stats_df, dtype, output_dir)

        # Top taxa per level (kept vs dropped)
        print("  Plotting top taxa per level ...")
        _plot_top_taxa_per_level(stats_df, dtype, output_dir)

        # Alpha diversity comparison
        print("  Computing alpha diversity ...")
        _plot_alpha_diversity(all_microbiome, all_microbes, kept_set, dtype, output_dir)

        # Statistical tests + feature importance
        print("  Running statistical tests ...")
        test_results = _run_statistical_tests(stats_df)
        test_results.to_csv(output_dir / f"statistical_tests_{dtype.lower()}.csv", index=False)

        print("  Running importance model ...")
        importance_df = _run_importance_model(stats_df)
        importance_df.to_csv(output_dir / f"feature_importance_{dtype.lower()}.csv", index=False)

        _plot_statistical_summary(stats_df, test_results, importance_df, dtype, output_dir)

    print("\nCharacterization complete.")
