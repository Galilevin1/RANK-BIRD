"""
Figure 1f: Cross-dtype LODO with common microbe features.

For each base phenotype (e.g. "ASD"):
  1. Load ALL datasets — both Amplicon and Metagenomics (when available)
  2. Intersect all feature columns → global-within-phenotype overlap
  3. Filter every dataset to those overlap features
  4. Run LODO treating all datasets (both dtypes) as one pool per phenotype
  5. Plot AUC per dataset, colored by dtype

This reveals whether classification transfers cross-dtype when both use the
same feature space, and whether the dtype or the target drives sample structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_DTYPE_COLORS  = {"Metagenomics": "#4878D0", "Amplicon": "#EE854A"}
_DTYPE_MARKERS = {"Metagenomics": "o",       "Amplicon": "^"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lodo_name_safe(
    mb_list: list,
    tgt_list: list,
    names: list,
) -> pd.DataFrame:
    """
    LODO where the test-set exclusion is by *name*, not by index.

    When leaving out dataset X as the test set, all other datasets that share
    the same name (same study appearing as both Amplicon and Metagenomics, or
    appearing across multiple related phenotypes) are also excluded from
    training.  This prevents data leakage between paired measurements.
    """
    import io, sys
    from evaluation.data_loading import combine_datasets
    from evaluation.lgbm_model import train_lightgbm

    rows = []
    for test_idx, test_name in enumerate(names):
        train_indices = [
            i for i, n in enumerate(names)
            if i != test_idx and n != test_name
        ]
        if not train_indices:
            continue

        excluded = [names[i] for i in range(len(names)) if i != test_idx and names[i] == test_name]
        if excluded:
            print(f"  [name-safe] testing on '{test_name}' → also dropped {len(excluded)} twin(s) from training")

        X_train, y_train = combine_datasets(mb_list, tgt_list, train_indices)
        X_test  = mb_list[test_idx]
        y_test  = tgt_list[test_idx]

        _buf = io.StringIO()
        _old_stdout, sys.stdout = sys.stdout, _buf
        try:
            metrics = train_lightgbm(X_train, y_train, X_test, y_test)
        finally:
            sys.stdout = _old_stdout

        rows.append({"test_idx": test_idx, "test_dataset": test_name, **metrics})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_figure1f(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
) -> pd.DataFrame:
    """
    Global cross-dtype LODO on the shared feature space.

    Steps:
      1. Load EVERY dataset across ALL phenotypes and BOTH dtypes.
      2. Keep microbes present in ≥60 % of datasets (column exists in that
         dataset).  Datasets missing a selected microbe get 0-filled.  Print count.
      3. Filter all datasets to those common features.
      4. Run one global LODO pool: leave one dataset out as the test set,
         train on ALL remaining datasets (all phenotypes + all dtypes mixed).
         Same-name exclusion: when dataset X is the test set, all other
         entries named X are also removed from training.
      5. Record AUC, dtype, and phenotype of the test dataset.

    Returns DataFrame columns:
        base_phenotype, dtype, dataset, n_common_features, auc
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    # ── Step 1: load every dataset from every (phenotype, dtype) folder ───────
    all_mb, all_tgt, all_names, all_dtype_labels, all_pheno_labels = [], [], [], [], []

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = Path(data_root) / pheno_str
        if not folder.exists():
            continue
        try:
            mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception:
            continue
        for mb, tgt, name in zip(mb_dfs, tgt_dfs, ds_names):
            all_mb.append(mb)
            all_tgt.append(tgt)
            all_names.append(name)
            all_dtype_labels.append(dtype)
            all_pheno_labels.append(phenotype)
    total = len(all_mb)
    if total < 2:
        print("  [skip] need ≥2 datasets globally for LODO")
        return pd.DataFrame()

    # ── Step 2: keep microbes present in ≥60 % of datasets ───────────────────
    PREVALENCE_THRESHOLD = 0.60
    all_cols = set()
    for df in all_mb:
        all_cols.update(df.columns)

    col_counts = {col: sum(col in df.columns for df in all_mb) for col in all_cols}
    min_count  = PREVALENCE_THRESHOLD * total
    common     = sorted(col for col, cnt in col_counts.items() if cnt >= min_count)
    n_common   = len(common)
    print(
        f"Microbes present in ≥{int(PREVALENCE_THRESHOLD*100)}% of datasets "
        f"({int(min_count)}/{total}): {n_common}"
    )

    if n_common == 0:
        print("  [skip] zero microbes pass the prevalence threshold")
        return pd.DataFrame()

    # ── Step 3: filter to selected features, fill 0 for missing columns ───────
    mb_filtered = [df.reindex(columns=common, fill_value=0.0) for df in all_mb]

    # ── Step 4: global LODO with name-safe exclusion ──────────────────────────
    try:
        lodo_df = _lodo_name_safe(mb_filtered, all_tgt, all_names)
    except Exception as e:
        print(f"  [error] global LODO failed: {e}")
        return pd.DataFrame()

    # ── Step 5: annotate results by index (handles duplicate names across phenotypes) ──
    rows = []
    for _, row in lodo_df.iterrows():
        idx = int(row["test_idx"])
        rows.append({
            "base_phenotype":    all_pheno_labels[idx],
            "dtype":             all_dtype_labels[idx],
            "dataset":           all_names[idx],
            "n_common_features": n_common,
            "auc":               float(row["auc"]),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1f(
    results_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Strip/dot plot: cross-dtype LODO AUC per phenotype.

    Color  = dtype (Amplicon vs Metagenomics)
    Shape  = dtype (same mapping)
    Shaded = phenotypes that have BOTH dtypes
    """
    _standalone = ax is None

    if results_df.empty:
        if _standalone:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No 1f results", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        return fig

    if _standalone:
        n_phenos = results_df["base_phenotype"].nunique()
        fig, ax = plt.subplots(figsize=(max(10, n_phenos * 1.4), 6))
    else:
        fig = ax.figure

    # derive per-phenotype has_both from the data itself
    pheno_dtype_counts = (
        results_df.groupby("base_phenotype")["dtype"].nunique()
    )
    has_both_map = (pheno_dtype_counts > 1).to_dict()

    # sort: phenotypes with both dtypes first, then alphabetically
    all_phenos = sorted(results_df["base_phenotype"].unique())
    order = sorted(all_phenos,
                   key=lambda p: (not has_both_map.get(p, False), p))

    n_common = int(results_df["n_common_features"].iloc[0])  # global, same for all

    rng = np.random.default_rng(42)
    x_ticks, x_labels = [], []

    for x_pos, base_pheno in enumerate(order):
        sub      = results_df[results_df["base_phenotype"] == base_pheno]
        has_both = has_both_map.get(base_pheno, False)
        dtypes   = sorted(sub["dtype"].unique())

        # shade phenotypes with both dtypes
        if has_both:
            ax.axvspan(x_pos - 0.45, x_pos + 0.45, alpha=0.08,
                       color="gold", zorder=0)

        n_d = len(dtypes)
        for i, dtype in enumerate(dtypes):
            sub_d  = sub[sub["dtype"] == dtype]
            color  = _DTYPE_COLORS.get(dtype, "#888")
            marker = _DTYPE_MARKERS.get(dtype, "o")
            offset = (i - (n_d - 1) / 2) * 0.22

            xs = rng.normal(x_pos + offset, 0.04, len(sub_d))
            ax.scatter(xs, sub_d["auc"].values,
                       c=color, marker=marker,
                       s=280, alpha=0.85, zorder=3, linewidths=0)

            mean_auc = sub_d["auc"].mean()
            ax.plot([x_pos + offset - 0.10, x_pos + offset + 0.10],
                    [mean_auc, mean_auc],
                    color=color, linewidth=8, zorder=4)

        x_ticks.append(x_pos)
        x_labels.append(base_pheno.replace("_", "\n"))

        if x_pos > 0:
            ax.axvline(x_pos - 0.5, color="lightgray", linewidth=0.8, zorder=1)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=40)
    ax.set_ylabel("LODO AUC", fontsize=42, fontweight="bold")
    ax.tick_params(axis="y", labelsize=40)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.set_xlabel("Phenotype", fontsize=56, fontweight="bold")

    legend_handles = [
        plt.Line2D([0], [0], marker=_DTYPE_MARKERS["Metagenomics"], color="w",
                   markerfacecolor=_DTYPE_COLORS["Metagenomics"],
                   markersize=22, label="Metagenomics"),
        plt.Line2D([0], [0], marker=_DTYPE_MARKERS["Amplicon"], color="w",
                   markerfacecolor=_DTYPE_COLORS["Amplicon"],
                   markersize=22, label="Amplicon"),
        mpatches.Patch(color="gold", alpha=0.35, label="Both dtypes"),
    ]
    ax.legend(handles=legend_handles, fontsize=34, framealpha=0.9,
              loc="lower right")

    if _standalone:
        plt.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_figure1f(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    results_cache_path: Optional[str] = None,
) -> None:
    """Compute (or load cached) Figure 1f results and save the PNG."""
    out   = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(results_cache_path) if results_cache_path else out / "figure1f_results.csv"

    if cache.exists():
        print(f"  Loading cached Figure 1f results from {cache.name}")
        results_df = pd.read_csv(cache)
    else:
        results_df = compute_figure1f(phenotypes, data_root)
        results_df.to_csv(cache, index=False)
        print(f"  Saved Figure 1f results → {cache.name}")

    fig = plot_figure1f(results_df)
    fig.savefig(out / "figure1f_cross_dtype_lodo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 1f")
