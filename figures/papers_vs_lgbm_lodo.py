import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from evaluation.statistics import compare_auc_mann_whitney_fdr

def plot_auc_horizontal_bars_mann_whitney(df_papers, df_lightGBM,
                                          selected_combinations=None, figsize=(12, 8),
                                          group_gap=1.2, bar_height=0.6, fdr_alpha=0.05,
                                          ax=None):
    """
    Horizontal bar visualization with Mann-Whitney U test + FDR correction.

    Features:
    ---------
    - Horizontal bars show AUC performance from 0 to mean
    - Error bars at the end show ±SE
    - Significance stars for FDR-corrected comparisons vs LightGBM
    - Mann-Whitney U test (non-parametric, appropriate for small samples)
    - Benjamini-Hochberg FDR correction for multiple testing
    - Bootstrap confidence intervals for effect sizes

    Parameters:
    -----------
    df_papers : DataFrame
        paper results
    df_lightGBM : DataFrame
        lightGBM results
    selected_combinations : list of tuples
        List of (phenotype, type) combinations to plot
    figsize : tuple
        Figure size (width, height)
    group_gap : float
        Space between phenotype groups
    bar_height : float
        Height of horizontal bars
    fdr_alpha : float
        FDR significance threshold (default: 0.05)

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    lgbm_results : dict
        Raw LightGBM AUC values per group
    statistical_results : pd.DataFrame
        Detailed statistical test results with p-values, q-values, and effect sizes
    """
    df_papers = df_papers.dropna(subset=["Phenotype", "Type"], how="all").reset_index(drop=True)
    df_papers["Phenotype"] = df_papers["Phenotype"].astype(str)
    df_papers["Type"] = df_papers["Type"].astype(str)
    df_papers["Group"] = df_papers["Phenotype"] + " " + df_papers["Type"]

    if selected_combinations:
        groups_to_plot = [f"{p} {t}" for p, t in selected_combinations]
        df_papers = df_papers[df_papers["Group"].isin(groups_to_plot)]
    else:
        groups_to_plot = df_papers["Group"].unique()

    ignore = {"Phenotype full name", "Phenotype", "Type", "Metadata Mapping", "Dataset", "Notes", "Group"}
    paper_cols = [c for c in df_papers.columns if c not in ignore and pd.api.types.is_numeric_dtype(df_papers[c])]

    # Compute summary statistics AND store raw data for proper statistical testing
    summary = []
    paper_raw_data = {}  # Store actual AUC values for each paper

    for group_name, group_df in df_papers.groupby("Group", sort=False):
        phenotype, dtype = group_name.split(" ", 1) if " " in group_name else (group_name, "")
        print(f"\nProcessing group: {group_name}")

        group_df = df_papers[(df_papers["Phenotype"] == phenotype) & (df_papers["Type"] == dtype)]
        group_df = group_df.dropna(subset=paper_cols, how="all")

        for paper in paper_cols:
            aucs = pd.to_numeric(group_df[paper], errors="coerce").dropna().values
            if len(aucs) > 0:
                summary.append({
                    "Group": group_name,
                    "Paper": paper,
                    "Mean": np.mean(aucs),
                    "SE": sem(aucs)
                })
                # Store raw data for proper statistical testing
                paper_raw_data[(group_name, paper)] = aucs
                print(f"  {paper}: n={len(aucs)}, mean={np.mean(aucs):.3f}")

    lgbm_raw_data = {}

    for group in groups_to_plot:
        aucs = df_lightGBM.loc[
            (df_lightGBM["phenotype"] == group) &
            (df_lightGBM["protocol"] == "LODO"),
            "auc"
        ].dropna().values

        if len(aucs) == 0:
            print(f"⚠️ No LODO results for {group}")
            continue

        lgbm_raw_data[group] = aucs

        summary.append({
            "Group": group,
            "Paper": "LightGBM LODO",
            "Mean": np.mean(aucs),
            "SE": sem(aucs)
        })

    summary_df = pd.DataFrame(summary)

    # Statistical testing with Mann-Whitney U + FDR correction
    print("\n" + "=" * 70)
    print("STATISTICAL TESTING: Mann-Whitney U Test + FDR Correction")
    print("=" * 70)
    print("Method: Non-parametric Mann-Whitney U test (Wilcoxon rank-sum)")
    print("Multiple testing correction: Benjamini-Hochberg FDR")
    print(f"Significance threshold: α = {fdr_alpha}")
    print("=" * 70)

    statistical_results = compare_auc_mann_whitney_fdr(
        summary_df, lgbm_raw_data, paper_raw_data,
        fdr_method='fdr_bh', alpha=fdr_alpha
    )

    if len(statistical_results) > 0:
        print("\n=== Statistical Test Results ===")
        display_cols = ["Group", "Paper", "LightGBM n", "Paper n",
                        "Mean Difference", "95% CI Lower", "95% CI Upper",
                        "p-value", "q-value (FDR)", "Significance"]
        print(statistical_results[display_cols].to_string(index=False))

        # Summary statistics
        n_total = len(statistical_results)
        n_sig_uncorrected = (statistical_results["Significance (uncorrected)"] != "ns").sum()
        n_sig_fdr = (statistical_results["Significance"] != "ns").sum()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total comparisons performed: {n_total}")
        print(f"Significant (uncorrected p<0.05): {n_sig_uncorrected} ({100 * n_sig_uncorrected / n_total:.1f}%)")
        print(f"Significant (FDR-corrected q<0.05): {n_sig_fdr} ({100 * n_sig_fdr / n_total:.1f}%)")
        print("=" * 70)
    else:
        print("\n⚠️ No statistical comparisons could be performed (insufficient data)")
        statistical_results = pd.DataFrame()  # Empty dataframe

    # Create significance lookup dictionary
    sig_lookup = {}
    for _, row in statistical_results.iterrows():
        key = (row["Group"], row["Paper"])
        sig_lookup[key] = row["Significance"]

    # === PLOTTING ===
    papers = sorted(summary_df["Paper"].unique())
    papers = [p for p in papers if p != "LightGBM LODO"] + ["LightGBM LODO"]

    colors_palette = sns.color_palette("husl", len(papers) - 1)
    color_map = {p: c for p, c in zip(papers[:-1], colors_palette)}
    color_map["LightGBM LODO"] = "#2E2E2E"

    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_facecolor('white')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    y_offset = 0
    group_positions = []
    papers_in_legend = set()

    for group_idx, group in enumerate(groups_to_plot):
        gdf = summary_df[summary_df["Group"] == group]
        if gdf.empty:
            continue

        group_start = y_offset
        gdf = gdf.sort_values(by="Paper", key=lambda x: x != "LightGBM LODO")

        for i, row in enumerate(gdf.itertuples()):
            is_lgbm = row.Paper == "LightGBM LODO"

            bar_h = bar_height * 1.3 if is_lgbm else bar_height
            alpha = 1.0 if is_lgbm else 0.85
            edge_width = 2.0 if is_lgbm else 1.2

            # Horizontal bar
            bar = ax.barh(
                y_offset, row.Mean, height=bar_h, left=0,
                color=color_map.get(row.Paper, 'gray'), alpha=alpha,
                edgecolor='white' if is_lgbm else color_map.get(row.Paper, 'gray'),
                linewidth=edge_width, zorder=10 if is_lgbm else 5,
                label=row.Paper if row.Paper not in papers_in_legend else None
            )
            papers_in_legend.add(row.Paper)

            # Error bar
            ax.errorbar(
                row.Mean, y_offset, xerr=row.SE, fmt='none',
                ecolor='black' if is_lgbm else color_map.get(row.Paper, 'gray'),
                elinewidth=2.5 if is_lgbm else 1.8,
                capsize=6 if is_lgbm else 4,
                capthick=2.5 if is_lgbm else 1.8,
                alpha=1.0, zorder=15 if is_lgbm else 10
            )

            # Significance stars (only for FDR-corrected significant results)
            if not is_lgbm:
                sig_key = (row.Group, row.Paper)
                if sig_key in sig_lookup and sig_lookup[sig_key] != "ns":
                    sig_text = sig_lookup[sig_key]
                    star_x = row.Mean + row.SE + 0.02
                    ax.text(star_x, y_offset, sig_text,
                            fontsize=13, fontweight='bold',
                            va='center', ha='left',
                            color=color_map.get(row.Paper, 'gray'))

            y_offset += 1

        group_end = y_offset - 1
        group_mid = (group_start + group_end) / 2
        group_positions.append((group, group_mid))

        # Separator line between groups
        if group != groups_to_plot[-1]:
            ax.axhline(y=y_offset + group_gap / 2 - 0.5, color='gray',
                       linestyle='-', linewidth=0.8, alpha=0.3)

        y_offset += group_gap

    # Group labels
    for group, ymid in group_positions:
        ax.text(-0.05, ymid, group, ha="right", va="center",
                fontsize=42, fontweight="bold",
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray',
                          edgecolor='gray', alpha=0.3))

    # Aesthetics
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=38)
    ax.set_xlabel("LODO AUC (Mean ± SE)", fontsize=42, fontweight="bold")
    ax.set_xlim(0, 1.05)
    if _standalone:
        ax.set_title("Model Performance: LightGBM vs. Published Results\n(Mann-Whitney U Test + FDR Correction)",
                     fontsize=16, fontweight="bold", pad=20)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = {label: handle for handle, label in zip(handles, labels)}

    ordered_labels = []
    if "LightGBM LODO" in by_label:
        ordered_labels.append("LightGBM LODO")
    ordered_labels.extend([l for l in sorted(by_label.keys()) if l != "LightGBM LODO"])
    ordered_handles = [by_label[l] for l in ordered_labels]

    if _standalone:
        legend = ax.legend(ordered_handles, ordered_labels,
                           title="Models/Papers\n(* = sig. vs LightGBM)",
                           title_fontsize=10, fontsize=9,
                           bbox_to_anchor=(1.02, 1), loc="upper left",
                           frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.95)
        sig_text = (f"Significance vs. LightGBM:\n"
                    f"* q<0.05, ** q<0.01, *** q<0.001\n"
                    f"(Mann-Whitney U, Benjamini-Hochberg FDR, α={fdr_alpha})")
        ax.text(0.96, -0.12, sig_text, transform=ax.transAxes,
                fontsize=8, style='italic', ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.tight_layout()
    else:
        ax.legend(ordered_handles, ordered_labels,
                  title="* = sig. vs LightGBM",
                  title_fontsize=8, fontsize=7,
                  loc="lower right", frameon=True, framealpha=0.9)

    return fig, ax, statistical_results