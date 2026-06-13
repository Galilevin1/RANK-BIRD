import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.transforms import blended_transform_factory


def _dtype_sort_key(phenotype: str) -> int:
    """Metagenomics (shotgun) rows first, then Amplicon (16S)."""
    if "Amplicon" in phenotype:
        return 1
    return 0



def plot_protocol_heatmap(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame = None,
    papers_df: pd.DataFrame = None,
    selected_combinations=None,
    ax=None,
    wgs_16s_x: float = -0.32,
):
    # Sort phenotypes: Amplicon first, then Metagenomics, alphabetically within each group
    phenotype_order = sorted(
        summary_df["phenotype"].unique(),
        key=lambda x: (_dtype_sort_key(x), x),
    )

    pivot = summary_df.pivot(
        index="phenotype",
        columns="protocol",
        values="mean_auc",
    ).reindex(phenotype_order)

    # Rename protocol columns to include "LightGBM" prefix
    pivot = pivot.rename(columns={
        "LODO":                 "LGBM\nLODO",
        "Internal Validation":  "LGBM\nMixed-\nDataset",
        "Within Learning":      "LGBM\nWithin-\nDataset",
    })

    # ── Add Papers LODO column (mean across all paper columns) ──
    if papers_df is not None:
        _pdf = papers_df.dropna(subset=["Phenotype", "Type"], how="all").copy()
        _pdf["Phenotype"] = _pdf["Phenotype"].astype(str)
        _pdf["Type"]      = _pdf["Type"].astype(str)
        _pdf["Group"]     = _pdf["Phenotype"] + " " + _pdf["Type"]
        if selected_combinations:
            _keep = {f"{p} {t}" for p, t in selected_combinations}
            _pdf  = _pdf[_pdf["Group"].isin(_keep)]
        _ignore = {"Phenotype full name", "Phenotype", "Type",
                   "Metadata Mapping", "Dataset", "Notes", "Group"}
        _pcols  = [c for c in _pdf.columns
                   if c not in _ignore and pd.api.types.is_numeric_dtype(_pdf[c])]
        _pmeans = {}
        for _grp, _rows in _pdf.groupby("Group"):
            _aucs = [v for c in _pcols
                     for v in pd.to_numeric(_rows[c], errors="coerce").dropna().tolist()]
            if _aucs:
                _pmeans[_grp] = float(np.mean(_aucs))
        pivot.insert(0, "Papers LODO",
                     pd.Series({ph: _pmeans.get(ph, np.nan) for ph in phenotype_order}))

    # Enforce column order
    col_order = ["Papers LODO", "LGBM\nLODO",
                 "LGBM\nMixed-\nDataset", "LGBM\nWithin-\nDataset"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    annot = pivot.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=(18, max(6, len(phenotype_order) * 0.6)))
    else:
        fig = ax.figure

    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1,
        cbar=_standalone,
        cbar_kws={"label": "Mean AUC"} if _standalone else {},
        annot_kws={"size": 116},
        ax=ax,
    )

    ax.set_aspect("auto")
    ax.set_ylabel("")
    ax.set_xlabel("Protocol", fontsize=116, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=116)

    # Strip " Amplicon" / " Metagenomics" suffix from y-tick labels
    stripped = [
        t.get_text().replace(" Amplicon", "").replace(" Metagenomics", "")
                    .replace("Delivery_mode_", "DM ")
        for t in ax.get_yticklabels()
    ]
    ax.set_yticklabels(stripped, fontsize=108, rotation=0)

    if _standalone:
        ax.figure.axes[-1].yaxis.label.set_size(14)
        ax.figure.axes[-1].tick_params(labelsize=16)

    # Separator between Metagenomics (top) and Amplicon (bottom) blocks
    n_meta  = sum(1 for p in phenotype_order if "Metagenomics" in p)
    n_total = len(phenotype_order)
    if 0 < n_meta < n_total:
        ax.axhline(n_meta, color="black", linewidth=16, zorder=5)

    for y in range(1, pivot.shape[0]):
        if y != n_meta:
            ax.axhline(y, color="black", linewidth=1.5)

    # Section labels: rotated text outside the figure
    if 0 < n_meta < n_total:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        y_mid_meta = n_meta / 2
        y_mid_amp  = n_meta + (n_total - n_meta) * 0.18

        ax.text(wgs_16s_x, y_mid_meta, "WGS",
                transform=trans, fontsize=112, fontweight="bold",
                va="center", ha="center", rotation=0, clip_on=False,
                bbox=dict(boxstyle="round,pad=1.0", facecolor="lightblue",
                          edgecolor="black", linewidth=2.5, alpha=0.8))
        ax.text(wgs_16s_x, y_mid_amp,  "16S",
                transform=trans, fontsize=112, fontweight="bold",
                va="center", ha="center", rotation=0, clip_on=False,
                bbox=dict(boxstyle="round,pad=1.0", facecolor="lightcoral",
                          edgecolor="black", linewidth=2.5, alpha=0.8))

    if _standalone:
        plt.tight_layout()

    return fig, ax, pivot.reset_index()


def compute_lgbm_protocols_per_phenotype_test(
    results_df: pd.DataFrame,
    protocols: list = None,
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests between LightGBM protocols, per phenotype.

    For each (phenotype × protocol pair): all individual dataset AUC values
    in that phenotype are compared between the two protocols.
    Benjamini-Hochberg FDR is applied across all comparisons.

    Returns one row per (phenotype × protocol pair) with:
    phenotype, protocol_1, protocol_2, n_1, n_2,
    mean_1, mean_2, mean_diff, U_statistic, p_value, p_value_fdr
    """
    from itertools import combinations as _combs
    from statsmodels.stats.multitest import multipletests

    if protocols is None:
        protocols = ["LODO", "Internal Validation", "Within Learning"]

    df = results_df.copy()
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")

    rows = []
    for pheno in sorted(df["phenotype"].unique()):
        sub = df[df["phenotype"] == pheno]
        for p1, p2 in _combs(protocols, 2):
            v1 = sub[sub["protocol"] == p1]["auc"].dropna().values
            v2 = sub[sub["protocol"] == p2]["auc"].dropna().values
            row = {
                "phenotype":   pheno,
                "protocol_1":  p1,
                "protocol_2":  p2,
                "n_1":         len(v1),
                "n_2":         len(v2),
                "mean_1":      float(np.mean(v1)) if len(v1) else np.nan,
                "mean_2":      float(np.mean(v2)) if len(v2) else np.nan,
                "mean_diff":   float(np.mean(v1) - np.mean(v2)) if len(v1) and len(v2) else np.nan,
                "U_statistic": np.nan,
                "p_value":     np.nan,
            }
            if len(v1) >= 2 and len(v2) >= 2:
                U, p = mannwhitneyu(v1, v2, alternative="two-sided")
                row["U_statistic"] = float(U)
                row["p_value"]     = float(p)
            rows.append(row)

    result = pd.DataFrame(rows)

    valid = result["p_value"].notna()
    fdr = np.full(len(result), np.nan)
    if valid.any():
        _, fdr_vals, _, _ = multipletests(result.loc[valid, "p_value"].values, method="fdr_bh")
        fdr[valid.values] = fdr_vals
    result["p_value_fdr"] = fdr

    return result


def compute_lodo_gap_dtype_test(
    results_df: pd.DataFrame,
    within_protocol: str = "Within Learning",
    lodo_protocol: str = "LODO",
    phenotype_filter: str = "ASD",
) -> tuple:
    """
    Mann-Whitney U test comparing the LODO gap (within AUC − LODO AUC)
    between WGS (Metagenomics) and 16S (Amplicon) datasets.

    Each individual dataset is one observation:
      gap = within_protocol AUC  −  LODO AUC  (per dataset)

    Group 1: all WGS  dataset gaps
    Group 2: all 16S  dataset gaps

    Parameters
    ----------
    results_df       : long-form DataFrame with columns
                       [phenotype, dataset, protocol, auc]
    within_protocol  : protocol name for the "within" arm
    lodo_protocol    : protocol name for the LODO arm
    phenotype_filter : if given, restrict to rows whose phenotype contains
                       this string (e.g. "ASD" to compare WGS vs 16S within ASD)

    Returns
    -------
    per_dataset_df : one row per dataset with phenotype, dataset, dtype,
                     within_auc, lodo_auc, gap
    test_df        : one-row summary with U, p-value, group means/medians
    """
    results_df = results_df.copy()
    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    if phenotype_filter:
        results_df = results_df[results_df["phenotype"].str.contains(phenotype_filter, na=False)]

    within = (results_df[results_df["protocol"] == within_protocol]
              [["phenotype", "dataset", "auc"]]
              .rename(columns={"auc": "within_auc"})
              .dropna(subset=["within_auc"]))

    lodo = (results_df[results_df["protocol"] == lodo_protocol]
            [["phenotype", "dataset", "auc"]]
            .rename(columns={"auc": "lodo_auc"})
            .dropna(subset=["lodo_auc"]))

    merged = within.merge(lodo, on=["phenotype", "dataset"], how="inner")
    merged["gap"] = merged["within_auc"] - merged["lodo_auc"]
    merged["dtype"] = merged["phenotype"].apply(
        lambda p: "Amplicon (16S)" if "Amplicon" in p else "Metagenomics (WGS)"
    )

    wgs = merged[merged["dtype"] == "Metagenomics (WGS)"]["gap"].values
    s16 = merged[merged["dtype"] == "Amplicon (16S)"]["gap"].values

    label = f"{phenotype_filter} — " if phenotype_filter else ""
    row = {
        "comparison":     f"{label}Metagenomics (WGS) vs Amplicon (16S)",
        "n_wgs":          len(wgs),
        "n_16s":          len(s16),
        "mean_gap_wgs":   float(np.mean(wgs))   if len(wgs) else np.nan,
        "mean_gap_16s":   float(np.mean(s16))   if len(s16) else np.nan,
        "median_gap_wgs": float(np.median(wgs)) if len(wgs) else np.nan,
        "median_gap_16s": float(np.median(s16)) if len(s16) else np.nan,
        "U_statistic":    np.nan,
        "p_value":        np.nan,
    }
    if len(wgs) >= 2 and len(s16) >= 2:
        U, p = mannwhitneyu(wgs, s16, alternative="two-sided")
        row["U_statistic"] = float(U)
        row["p_value"]     = float(p)

    return merged, pd.DataFrame([row])
