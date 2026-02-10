import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu

# ===================================================================
# Compare papers-lightGBM performances
# ==================================================================

def bootstrap_ci_difference(lgbm_aucs, paper_aucs, n_bootstrap=10000, ci=95):
    """
    Bootstrap confidence interval for difference in means.
    Provides effect size with uncertainty estimate.
    """
    if len(lgbm_aucs) < 2 or len(paper_aucs) < 2:
        return np.nan, np.nan, False

    n_lgbm = len(lgbm_aucs)
    n_paper = len(paper_aucs)

    diffs = []
    for _ in range(n_bootstrap):
        lgbm_sample = np.random.choice(lgbm_aucs, size=n_lgbm, replace=True)
        paper_sample = np.random.choice(paper_aucs, size=n_paper, replace=True)
        diff = np.mean(lgbm_sample) - np.mean(paper_sample)
        diffs.append(diff)

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)

    # If CI doesn't include 0, significant
    significant = not (lower <= 0 <= upper)

    return lower, upper, significant



def compare_auc_mann_whitney_fdr(summary_df, lgbm_raw_data, paper_raw_data,
                                 fdr_method='fdr_bh', alpha=0.05):
    """
    Perform Mann-Whitney U tests comparing each paper's AUCs with LightGBM AUCs.

    Mann-Whitney U is a non-parametric test that:
    - Does not assume normal distribution
    - Works well with small sample sizes
    - Is robust to outliers
    - Is appropriate for bounded data like AUC values

    Then applies Benjamini-Hochberg FDR correction for multiple testing.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with Mean and SE for each paper and group
    lgbm_raw_data : dict
        Dictionary mapping group -> array of actual LightGBM AUC values
    paper_raw_data : dict
        Dictionary mapping (group, paper) -> array of actual paper AUC values
    fdr_method : str
        Method for FDR correction ('fdr_bh' for Benjamini-Hochberg)
    alpha : float
        Family-wise error rate for FDR correction (default: 0.05)

    Returns:
    --------
    pd.DataFrame with:
        - Original p-values from Mann-Whitney U test
        - FDR-corrected q-values
        - Significance calls with stars
        - Bootstrap confidence intervals for effect sizes
        - Sample sizes for each comparison
    """
    results = []

    for group in summary_df["Group"].unique():
        if group not in lgbm_raw_data:
            continue

        lgbm_aucs = np.array(lgbm_raw_data[group])

        group_df = summary_df[summary_df["Group"] == group]

        for _, row in group_df.iterrows():
            if row["Paper"] == "LightGBM LODO":
                continue

            # Get actual paper AUC values
            paper_key = (group, row["Paper"])
            if paper_key not in paper_raw_data:
                continue

            paper_aucs = np.array(paper_raw_data[paper_key])

            # Skip if too few samples (need at least 2 per group)
            if len(lgbm_aucs) < 2 or len(paper_aucs) < 2:
                print(f"⚠️ Skipping {group} - {row['Paper']}: insufficient samples")
                continue

            # Mann-Whitney U test (non-parametric, no normality assumption required)
            try:
                u_stat, p_val = mannwhitneyu(lgbm_aucs, paper_aucs, alternative='two-sided')
            except Exception as e:
                print(f"⚠️ Mann-Whitney test failed for {group} - {row['Paper']}: {e}")
                continue

            # Bootstrap confidence interval for effect size (mean difference)
            ci_lower, ci_upper, _ = bootstrap_ci_difference(lgbm_aucs, paper_aucs)

            results.append({
                "Group": group,
                "Paper": row["Paper"],
                "LightGBM Mean": np.mean(lgbm_aucs),
                "LightGBM n": len(lgbm_aucs),
                "Paper Mean": np.mean(paper_aucs),
                "Paper n": len(paper_aucs),
                "Mean Difference": np.mean(lgbm_aucs) - np.mean(paper_aucs),
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "p-value": p_val,
                "U-statistic": u_stat
            })

    results_df = pd.DataFrame(results)

    # Apply FDR correction across ALL comparisons
    if len(results_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            results_df["p-value"].values,
            alpha=alpha,
            method=fdr_method
        )

        results_df["q-value (FDR)"] = pvals_corrected
        results_df["FDR significant"] = reject

        # Assign significance stars based on FDR-corrected q-values
        def assign_stars(row):
            if not row["FDR significant"]:
                return "ns"
            q = row["q-value (FDR)"]
            if q < 0.001:
                return "***"
            elif q < 0.01:
                return "**"
            elif q < 0.05:
                return "*"
            else:
                return "ns"

        results_df["Significance"] = results_df.apply(assign_stars, axis=1)

        # Also keep original uncorrected significance for reference
        def assign_stars_uncorrected(p_val):
            if p_val < 0.001:
                return "***"
            elif p_val < 0.01:
                return "**"
            elif p_val < 0.05:
                return "*"
            else:
                return "ns"

        results_df["Significance (uncorrected)"] = results_df["p-value"].apply(assign_stars_uncorrected)

    return results_df


# ===================================================================
# Compare protocols
# ==================================================================



def compare_protocols_mann_whitney(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Mann–Whitney U tests between protocols.
    """

    protocols = results_df["protocol"].unique()
    rows = []

    for i in range(len(protocols)):
        for j in range(i + 1, len(protocols)):
            p1, p2 = protocols[i], protocols[j]

            auc1 = results_df.loc[results_df["protocol"] == p1, "auc"].values
            auc2 = results_df.loc[results_df["protocol"] == p2, "auc"].values

            stat, pval = mannwhitneyu(auc1, auc2, alternative="two-sided")

            rows.append({
                "protocol_1": p1,
                "protocol_2": p2,
                "n_1": len(auc1),
                "n_2": len(auc2),
                "mean_1": auc1.mean(),
                "mean_2": auc2.mean(),
                "u_statistic": stat,
                "p_value": pval,
            })

    return pd.DataFrame(rows)


