import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu
from matplotlib.transforms import blended_transform_factory


def _dtype_sort_key(phenotype: str) -> int:
    """Metagenomics (shotgun) rows first, then Amplicon (16S)."""
    if "Amplicon" in phenotype:
        return 1
    return 0


def _run_protocol_mannwhitney(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Mann-Whitney between protocols using per-phenotype mean AUC."""
    protocols = summary_df["protocol"].unique().tolist()
    rows = []
    for p1, p2 in combinations(protocols, 2):
        v1 = summary_df[summary_df["protocol"] == p1]["mean_auc"].dropna().values
        v2 = summary_df[summary_df["protocol"] == p2]["mean_auc"].dropna().values
        common = min(len(v1), len(v2))
        if common < 2:
            continue
        U, p = mannwhitneyu(v1, v2, alternative="two-sided")
        rows.append({
            "protocol_1":   p1,
            "protocol_2":   p2,
            "n_phenotypes": common,
            "mean_1":       float(np.mean(v1)),
            "mean_2":       float(np.mean(v2)),
            "mean_diff":    float(np.mean(v1) - np.mean(v2)),
            "U_statistic":  float(U),
            "p_value":      float(p),
            "significant":  p < 0.05,
        })
    return pd.DataFrame(rows)


def _run_lgbm_vs_papers_per_phenotype(
    results_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    phenotypes,
    selected_combinations=None,
) -> tuple:
    """
    Per-phenotype Mann-Whitney: LightGBM LODO (per dataset) vs Papers LODO (per dataset).
    Returns (papers_mean_by_pheno, pheno_stats_df).
    """
    # ── prepare papers ─────────────────────────────────────────
    pdf = papers_df.dropna(subset=["Phenotype", "Type"], how="all").copy()
    pdf["Phenotype"] = pdf["Phenotype"].astype(str)
    pdf["Type"]      = pdf["Type"].astype(str)
    pdf["Group"]     = pdf["Phenotype"] + " " + pdf["Type"]
    if selected_combinations:
        keep = {f"{p} {t}" for p, t in selected_combinations}
        pdf = pdf[pdf["Group"].isin(keep)]
    ignore = {"Phenotype full name", "Phenotype", "Type",
              "Metadata Mapping", "Dataset", "Notes", "Group"}
    paper_cols = [c for c in pdf.columns
                  if c not in ignore and pd.api.types.is_numeric_dtype(pdf[c])]

    paper_raw_by_pheno = {}
    for group_name, grp in pdf.groupby("Group"):
        all_aucs = []
        for col in paper_cols:
            all_aucs.extend(pd.to_numeric(grp[col], errors="coerce").dropna().tolist())
        if all_aucs:
            paper_raw_by_pheno[group_name] = all_aucs

    papers_mean_by_pheno = {ph: float(np.mean(v)) for ph, v in paper_raw_by_pheno.items()}

    # ── per-phenotype Mann-Whitney ─────────────────────────────
    rows = []
    for pheno in phenotypes:
        lgbm_v  = results_df[
            (results_df["phenotype"] == pheno) & (results_df["protocol"] == "LODO")
        ]["auc"].dropna().values
        paper_v = np.array(paper_raw_by_pheno.get(pheno, []))
        row = {
            "phenotype":   pheno,
            "n_lgbm":      len(lgbm_v),
            "n_papers":    len(paper_v),
            "mean_lgbm":   float(np.mean(lgbm_v))  if len(lgbm_v)  else np.nan,
            "mean_papers": float(np.mean(paper_v)) if len(paper_v) else np.nan,
            "mean_diff":   np.nan,
            "U_statistic": np.nan,
            "p_value":     np.nan,
            "significant": False,
        }
        if len(lgbm_v) >= 2 and len(paper_v) >= 2:
            U, p = mannwhitneyu(lgbm_v, paper_v, alternative="two-sided")
            row["mean_diff"]   = float(np.mean(lgbm_v) - np.mean(paper_v))
            row["U_statistic"] = float(U)
            row["p_value"]     = float(p)
            row["significant"] = bool(p < 0.05)
        rows.append(row)

    return papers_mean_by_pheno, pd.DataFrame(rows)


def plot_protocol_heatmap(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame = None,
    papers_df: pd.DataFrame = None,
    selected_combinations=None,
    ax=None,
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
        "LODO":                 "LightGBM\nLODO",
        "Internal Validation":  "LightGBM\nInternal\nValidation",
        "Within Learning":      "LightGBM\nWithin Learning",
    })

    # ── Add Papers LODO column + per-phenotype stats ───────────
    pheno_stats_df = pd.DataFrame()
    sig_phenos = set()
    if results_df is not None and papers_df is not None:
        papers_mean, pheno_stats_df = _run_lgbm_vs_papers_per_phenotype(
            results_df, papers_df, phenotype_order, selected_combinations,
        )
        pivot.insert(0, "Papers LODO",
                     pd.Series({ph: papers_mean.get(ph, np.nan) for ph in phenotype_order}))
        sig_phenos = set(
            pheno_stats_df[pheno_stats_df["significant"]]["phenotype"].tolist()
        )

    # Enforce column order
    col_order = ["Papers LODO", "LightGBM\nLODO",
                 "LightGBM\nInternal\nValidation", "LightGBM\nWithin Learning"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    # ── Custom annotation: add * to Papers LODO cells where significant ──
    annot = pivot.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    if "Papers LODO" in pivot.columns:
        for pheno in phenotype_order:
            if pheno in sig_phenos:
                annot.loc[pheno, "Papers LODO"] = annot.loc[pheno, "Papers LODO"] + "*"

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
        vmin=0,
        vmax=1,
        cbar=_standalone,
        cbar_kws={"label": "Mean AUC"} if _standalone else {},
        annot_kws={"size": 28},
        ax=ax,
    )

    ax.set_aspect("auto")
    ax.set_ylabel("")
    ax.set_xlabel("Protocol", fontsize=46, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=28, fontweight="bold")

    # Strip " Amplicon" / " Metagenomics" suffix from y-tick labels
    stripped = [
        t.get_text().replace(" Amplicon", "").replace(" Metagenomics", "")
        for t in ax.get_yticklabels()
    ]
    ax.set_yticklabels(stripped, fontsize=30, fontweight="bold")

    if _standalone:
        ax.figure.axes[-1].yaxis.label.set_size(14)
        ax.figure.axes[-1].tick_params(labelsize=16)

    # Separator between Metagenomics (top) and Amplicon (bottom) blocks
    n_meta  = sum(1 for p in phenotype_order if "Metagenomics" in p)
    n_total = len(phenotype_order)
    if 0 < n_meta < n_total:
        ax.axhline(n_meta, color="black", linewidth=10, zorder=5)

    for y in range(1, pivot.shape[0]):
        if y != n_meta:
            ax.axhline(y, color="black", linewidth=1.5)

    # Section labels: rotated text outside the figure
    if 0 < n_meta < n_total:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        y_mid_meta = n_meta / 2
        y_mid_amp  = n_meta + (n_total - n_meta) / 2

        ax.text(-0.32, y_mid_meta, "Metagenomics\n(Shotgun)",
                transform=trans, fontsize=32, fontweight="bold",
                va="center", ha="center", rotation=0, clip_on=False,
                color="steelblue")
        ax.text(-0.42, y_mid_amp,  "Amplicon\n(16S rRNA)",
                transform=trans, fontsize=32, fontweight="bold",
                va="center", ha="center", rotation=0, clip_on=False,
                color="crimson")

    if _standalone:
        plt.tight_layout()

    stats_df = _run_protocol_mannwhitney(summary_df)

    return fig, ax, pivot.reset_index(), stats_df, pheno_stats_df
