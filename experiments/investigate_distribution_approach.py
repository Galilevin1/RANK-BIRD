"""
Investigation: distribution-adjustment approach comparison.

After stability filtering + oversampling-and-combining (equal dataset representation),
compares 4 ways to normalize microbiome data on the combined matrix:

  current       — full RANK-BIRD pipeline (per-microbe Wasserstein distributions
                  + quantile mapping back to each sample)
  ranking       — column-wise rank normalization (rank mapped to [0, 1])
  ranking_sig   — ranking + sigmoid transform (soft present/absent-like values)
  ranking_relu  — ranking + hard threshold: bottom fraction zeroed,
                  top fraction keeps original abundance value

In all 3 new approaches the oversampling + combining step is preserved so every
dataset contributes equally to the global rank.  Only original (non-duplicated)
samples are returned to the protocols after normalization.

Produces one CSV per approach × dtype and one comparison figure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_rel, f_oneway, wilcoxon

try:
    from statsmodels.stats.multitest import multipletests as _sm_multipletests
    def _fdr_bh(pvalues: np.ndarray) -> np.ndarray:
        if len(pvalues) == 0:
            return pvalues
        _, corrected, _, _ = _sm_multipletests(pvalues, method="fdr_bh")
        return corrected
except ImportError:
    def _fdr_bh(pvalues: np.ndarray) -> np.ndarray:
        n = len(pvalues)
        if n == 0:
            return pvalues
        idx = np.argsort(pvalues)
        ranked = np.array(pvalues, dtype=float)
        for rank, i in enumerate(idx):
            ranked[i] = pvalues[i] * n / (rank + 1)
        for i in range(n - 2, -1, -1):
            ranked[idx[i]] = min(ranked[idx[i]], ranked[idx[i + 1]])
        return np.minimum(ranked, 1.0)

from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.pipeline import apply_normalization_pipeline
from src.rankbird.normalization.taxonomy_filter import filter_to_level
from src.rankbird.normalization.ranking import (
    rank_normalize, sigmoid_normalize, relu_normalize,
    apply_ranking_pipeline,
    SIGMOID_K, SIGMOID_CENTER, RELU_THRESHOLD,
)
from evaluation.data_loading import load_microbiome_datasets_with_targets
from experiments.run_protocols_global_processing import (
    _run_protocols_on_group, _run_global_for_dtype, _detect_and_shuffle_ordered,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "Data"

# ── Approach catalogue ────────────────────────────────────────────────────────

APPROACHES = ["original", "original_filtered", "rankbird", "ranking", "ranking_sig", "ranking_relu"]

APPROACH_LABELS = {
    "original":          "Original",
    "original_filtered": "Filtering",
    "rankbird":          "CIFAR",
    "ranking":           "Rank-Only",
    "ranking_sig":       "Rank+Sigmoid",
    "ranking_relu":      "Rank+Relu",
}

APPROACH_COLORS = {
    "original":          "#9467bd",
    "original_filtered": "#8c564b",
    "rankbird":          "#1f77b4",
    "ranking":           "#ff7f0e",
    "ranking_sig":       "#2ca02c",
    "ranking_relu":      "#d62728",
}

PROTOCOLS = ["LODO", "Internal Validation", "Within Learning"]

# ── Default hyper-parameters ──────────────────────────────────────────────────
# (imported from src.rankbird.normalization.ranking)

# Oversampling target (same default as main pipeline)
MIN_SAMPLES_PER_DATASET = 550

# Stability percentiles (same defaults as main pipeline)
STABILITY_PERCENTILE = {
    "Metagenomics": 0.25,
    "Amplicon":     0.40,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Step A — Oversample + combine
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# Per-approach runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_approach_for_dtype(
    phenotypes: list,
    dtype: str,
    approach: str,
    stability_percentile: float,
    min_size: int = MIN_SAMPLES_PER_DATASET,
    sigmoid_k: float = SIGMOID_K,
    sigmoid_center: float = SIGMOID_CENTER,
    relu_threshold: float = RELU_THRESHOLD,
    random_state: int = 42,
    taxonomy_level=None,
    shuffle_ordered: bool = False,
    rank_tie_method: str = "first",
    use_optuna: bool = False,
    n_trials: int = 30,
) -> pd.DataFrame:
    """
    Load all datasets for `dtype`, apply `approach`, run all protocols.
    Returns a long-form DataFrame: [phenotype, dataset, protocol, auc].
    """
    all_microbiome, all_targets, all_names = [], [], []
    dataset_to_phenotype: dict = {}

    for phenotype, t in phenotypes:
        if t != dtype:
            continue
        folder = DATA_ROOT / f"{phenotype} {t}"
        dfs, tgts, names = load_microbiome_datasets_with_targets(folder)
        for df, y, name in zip(dfs, tgts, names):
            all_microbiome.append(df)
            all_targets.append(y)
            all_names.append(name)
            dataset_to_phenotype[name] = f"{phenotype} {t}"

    if not all_microbiome:
        return pd.DataFrame()

    # ── Detect (and optionally shuffle) fully-ordered datasets ───────────────
    all_microbiome, all_targets, _ = _detect_and_shuffle_ordered(
        all_microbiome, all_targets, all_names,
        random_state=random_state, apply_shuffle=shuffle_ordered,
    )

    name_to_target = dict(zip(all_names, all_targets))

    # ── Apply normalization approach ──────────────────────────────────────────
    if approach == "original":
        return _run_global_for_dtype(phenotypes, normalization_approach=None,
                                     taxonomy_level=taxonomy_level,
                                     shuffle_ordered=shuffle_ordered,
                                     random_state=random_state,
                                     rank_tie_method=rank_tie_method)

    elif approach == "original_filtered":
        all_microbiome = filter_to_level(all_microbiome, taxonomy_level)
        all_microbes  = union_microbes(all_microbiome)
        nz_df         = nonzero_percent_by_dataset(all_microbiome, all_names, all_microbes)
        kept_microbes = auto_stability_filter(nz_df, percentile=stability_percentile)
        all_microbiome = [df.reindex(columns=kept_microbes, fill_value=0.0)
                          for df in all_microbiome]

    elif approach == "rankbird":
        all_microbiome, all_names = apply_normalization_pipeline(
            all_microbiome,
            all_names,
            global_analysis=True,
            min_samples_per_dataset=min_size,
            stability_percentile_global=stability_percentile,
            taxonomy_level=taxonomy_level,
            rank_tie_method=rank_tie_method,
        )

    elif approach == "ranking":
        from functools import partial as _partial
        _norm = _partial(rank_normalize, tie_method=rank_tie_method)
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome, all_names,
            stability_percentile=stability_percentile,
            norm_fn=_norm,
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    elif approach == "ranking_sig":
        from functools import partial as _partial
        norm_fn = _partial(sigmoid_normalize, k=sigmoid_k, center=sigmoid_center,
                           tie_method=rank_tie_method)
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome, all_names,
            stability_percentile=stability_percentile,
            norm_fn=norm_fn,
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    elif approach == "ranking_relu":
        from functools import partial as _partial
        norm_fn = _partial(relu_normalize, threshold=relu_threshold,
                           tie_method=rank_tie_method)
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome, all_names,
            stability_percentile=stability_percentile,
            norm_fn=norm_fn,
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    else:
        raise ValueError(f"Unknown approach: {approach!r}")

    n_features = all_microbiome[0].shape[1] if all_microbiome else 0
    aligned_targets = [name_to_target[n] for n in all_names if n in name_to_target]

    # ── Run protocols per phenotype ───────────────────────────────────────────
    records = []
    for phenotype_str in set(dataset_to_phenotype.values()):
        idx = [
            i for i, n in enumerate(all_names)
            if dataset_to_phenotype.get(n) == phenotype_str
        ]
        if not idx:
            continue
        records.append(_run_protocols_on_group(
            [all_microbiome[i] for i in idx],
            [aligned_targets[i]  for i in idx],
            [all_names[i]        for i in idx],
            phenotype_str,
            use_optuna=use_optuna,
            n_trials=n_trials,
        ))

    return (pd.concat(records, ignore_index=True) if records else pd.DataFrame()), n_features


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset exclusion helper
# ═══════════════════════════════════════════════════════════════════════════════

def _get_single_class_datasets(phenotypes: list) -> set:
    """
    Return names of datasets whose target has only one unique class.
    These cannot be used for binary classification and are excluded from results.
    """
    single_class = set()
    for phenotype, dtype in phenotypes:
        folder = DATA_ROOT / f"{phenotype} {dtype}"
        if not folder.exists():
            continue
        for subdir in folder.iterdir():
            if not subdir.is_dir():
                continue
            target_file = subdir / "target.csv"
            if not target_file.exists():
                continue
            try:
                target_df = pd.read_csv(target_file, index_col=0)
                if target_df["Tag"].nunique() <= 1:
                    single_class.add(subdir.name)
            except Exception:
                pass
    if single_class:
        print(f"  [EXCL] single-class datasets (excluded): {sorted(single_class)}")
    return single_class


def _drop_single_class(df: pd.DataFrame, single_class_names: set) -> pd.DataFrame:
    """Remove rows belonging to single-class datasets."""
    if df.empty or "dataset" not in df.columns or not single_class_names:
        return df
    return df[~df["dataset"].isin(single_class_names)].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_auc(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    From raw results (phenotype, dataset, protocol, auc) aggregate to
    (protocol, mean_auc, std_auc, n_datasets).
    """
    results_df = results_df.copy()
    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")

    rows = []
    for protocol in PROTOCOLS:
        sub = results_df[results_df["protocol"] == protocol]["auc"].dropna()
        if sub.empty:
            continue
        rows.append({
            "protocol":   protocol,
            "mean_auc":   sub.mean(),
            "median_auc": sub.median(),
            "std_auc":    sub.std(ddof=1) if len(sub) > 1 else 0.0,
            "n_datasets": len(sub),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def _build_panels(all_results: dict, dtypes: list) -> dict:
    """Build panel_name → {approach: results_df} including 'All dtypes' and optionally 'Cross dtypes'."""
    panels: dict = {}
    base_dtypes = [d for d in dtypes if d != "Combined"]
    for dtype in base_dtypes:
        auc_by_approach = {}
        for approach in APPROACHES:
            df = all_results.get((dtype, approach), pd.DataFrame())
            if df.empty:
                continue
            df = df.copy()
            df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
            auc_by_approach[approach] = df
        if auc_by_approach:
            panels[dtype] = auc_by_approach

    all_combined: dict = {}
    for approach in APPROACHES:
        frames = [f for dtype in base_dtypes
                  for f in [all_results.get((dtype, approach), pd.DataFrame())]
                  if not f.empty]
        if frames:
            merged = pd.concat(frames, ignore_index=True).copy()
            merged["auc"] = pd.to_numeric(merged["auc"], errors="coerce")
            all_combined[approach] = merged
    if all_combined:
        panels["All dtypes"] = all_combined

    cross_combined: dict = {}
    for approach in APPROACHES:
        df = all_results.get(("Combined", approach), pd.DataFrame())
        if not df.empty:
            df = df.copy()
            df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
            cross_combined[approach] = df
    if cross_combined:
        panels["Cross dtypes"] = cross_combined

    return panels


def run_statistical_tests(
    all_results: dict,
    dtypes: list,
    alpha: float = 0.05,
) -> dict:
    """
    For each panel × protocol:
      1. One-way ANOVA across all approaches
      2. All-pairwise paired t-tests with Bonferroni correction

    Returns
    -------
    dict with keys:
      "anova"   : DataFrame (panel, protocol, f_statistic, p_value, significant)
      "posthoc" : DataFrame (panel, protocol, approach_1, approach_2,
                             n_pairs, mean_1, mean_2, mean_diff,
                             t_statistic, p_value_raw, p_value_bonferroni,
                             significant, stars)
    """
    panels = _build_panels(all_results, dtypes)

    anova_rows   = []
    posthoc_rows = []

    for panel_name, auc_by_approach in panels.items():
        for protocol in PROTOCOLS:

            # Collect per-dataset AUC series per approach, aligned on dataset
            series_by_approach: dict = {}
            for approach in APPROACHES:
                if approach not in auc_by_approach:
                    continue
                df_p = (auc_by_approach[approach]
                        [auc_by_approach[approach]["protocol"] == protocol]
                        [["dataset", "auc"]].dropna()
                        .set_index("dataset")["auc"])
                series_by_approach[approach] = df_p

            if len(series_by_approach) < 2:
                continue

            # Common datasets across all approaches (for paired tests)
            common = set.intersection(*[set(s.index) for s in series_by_approach.values()])
            if len(common) < 2:
                continue

            aligned = {a: s.loc[list(common)].values
                       for a, s in series_by_approach.items()}

            # ── ANOVA ─────────────────────────────────────────────────────────
            f_stat, p_anova = f_oneway(*aligned.values())
            anova_rows.append({
                "panel":        panel_name,
                "protocol":     protocol,
                "n_datasets":   len(common),
                "n_approaches": len(aligned),
                "f_statistic":  float(f_stat),
                "p_value":      float(p_anova),
                "significant":  p_anova < alpha,
                "stars":        _assign_stars(p_anova),
            })

            # ── Post-hoc: all pairwise paired t-tests + Bonferroni ────────────
            approach_list = list(aligned.keys())
            pairs = [(approach_list[i], approach_list[j])
                     for i in range(len(approach_list))
                     for j in range(i + 1, len(approach_list))]
            n_tests = len(pairs)

            for a1, a2 in pairs:
                try:
                    t_stat, p_raw_t = ttest_rel(aligned[a1], aligned[a2])
                except Exception:
                    continue
                try:
                    diffs = aligned[a1] - aligned[a2]
                    w_stat, p_raw_w = wilcoxon(diffs, alternative="two-sided")
                except Exception:
                    w_stat, p_raw_w = float("nan"), float("nan")
                p_bonf_t = min(float(p_raw_t) * n_tests, 1.0)
                p_bonf_w = min(float(p_raw_w) * n_tests, 1.0)
                posthoc_rows.append({
                    "panel":                 panel_name,
                    "protocol":              protocol,
                    "approach_1":            a1,
                    "approach_2":            a2,
                    "n_pairs":               len(common),
                    "mean_1":                float(aligned[a1].mean()),
                    "mean_2":                float(aligned[a2].mean()),
                    "mean_diff":             float(aligned[a1].mean() - aligned[a2].mean()),
                    "t_statistic":           float(t_stat),
                    "p_value_raw":           float(p_raw_t),
                    "p_value_bonferroni":    p_bonf_t,
                    "significant":           p_bonf_t < alpha,
                    "stars":                 _assign_stars(p_bonf_t),
                    "w_statistic":           float(w_stat),
                    "p_wilcoxon_raw":        float(p_raw_w),
                    "p_wilcoxon_bonferroni": p_bonf_w,
                    "significant_wilcoxon":  p_bonf_w < alpha,
                    "stars_wilcoxon":        _assign_stars(p_bonf_w),
                })

    posthoc_df = pd.DataFrame(posthoc_rows)

    # ── Apply BH FDR correction across all pairwise tests ────────────────────
    if not posthoc_df.empty:
        if "p_value_raw" in posthoc_df.columns:
            p_fdr_t = _fdr_bh(posthoc_df["p_value_raw"].values.astype(float))
            posthoc_df["p_value_fdr"]   = [round(float(v), 4) for v in p_fdr_t]
            posthoc_df["stars_fdr"]     = [_assign_stars(v) for v in p_fdr_t]
        if "p_wilcoxon_raw" in posthoc_df.columns:
            p_fdr_w = _fdr_bh(posthoc_df["p_wilcoxon_raw"].values.astype(float))
            posthoc_df["p_wilcoxon_fdr"]     = [round(float(v), 4) for v in p_fdr_w]
            posthoc_df["stars_wilcoxon_fdr"] = [_assign_stars(v) for v in p_fdr_w]

    return {
        "anova":   pd.DataFrame(anova_rows),
        "posthoc": posthoc_df,
    }


def compute_normalization_comparison(
    all_results: dict,
    dtypes: list,
    baseline: str = "original",
    treatment: str = "rankbird",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each dtype × protocol compare baseline vs treatment (default: original vs RANK-BIRD).

    Per paired dataset:
      - delta_mean   = mean(treatment) - mean(baseline)
      - delta_median = median(treatment) - median(baseline)
      - p_ttest      : paired t-test on per-dataset AUC values (tests mean difference)
      - p_wilcoxon   : Wilcoxon signed-rank test on per-dataset AUC differences (tests median)

    Returns a DataFrame saved as delta_tests.csv in the investigation output dir.
    """
    rows = []
    panels = _build_panels(all_results, dtypes)

    for panel_name, auc_by_approach in panels.items():
        if baseline not in auc_by_approach or treatment not in auc_by_approach:
            continue

        for protocol in PROTOCOLS:
            s_base = (auc_by_approach[baseline]
                      [auc_by_approach[baseline]["protocol"] == protocol]
                      [["dataset", "auc"]].dropna()
                      .set_index("dataset")["auc"])
            s_treat = (auc_by_approach[treatment]
                       [auc_by_approach[treatment]["protocol"] == protocol]
                       [["dataset", "auc"]].dropna()
                       .set_index("dataset")["auc"])

            common = sorted(set(s_base.index) & set(s_treat.index))
            if len(common) < 3:
                continue

            v_base  = s_base.loc[common].values
            v_treat = s_treat.loc[common].values
            diffs   = v_treat - v_base

            # ── Means ──────────────────────────────────────────────────────────
            mean_base  = float(v_base.mean())
            mean_treat = float(v_treat.mean())
            delta_mean = mean_treat - mean_base

            # ── Medians ────────────────────────────────────────────────────────
            median_base  = float(np.median(v_base))
            median_treat = float(np.median(v_treat))
            delta_median = median_treat - median_base

            # ── Paired t-test (mean difference) ───────────────────────────────
            try:
                _, p_ttest = ttest_rel(v_treat, v_base)
            except Exception:
                p_ttest = float("nan")

            # ── Wilcoxon signed-rank (median of differences) ──────────────────
            try:
                _, p_wilcoxon = wilcoxon(diffs, alternative="two-sided")
            except Exception:
                p_wilcoxon = float("nan")

            _treat_label = APPROACH_LABELS.get(treatment, treatment).replace(" ", "_")
            rows.append({
                "panel":        panel_name,
                "protocol":     protocol,
                "n_datasets":   len(common),
                "mean_original":           round(mean_base,  3),
                f"mean_{_treat_label}":    round(mean_treat, 3),
                "delta_mean":              round(delta_mean, 3),
                "p_ttest":                 round(float(p_ttest), 4),
                "sig_ttest":               _assign_stars(p_ttest),
                "median_original":         round(median_base,  3),
                f"median_{_treat_label}":  round(median_treat, 3),
                "delta_median":            round(delta_median, 3),
                "p_wilcoxon":              round(float(p_wilcoxon), 4),
                "sig_wilcoxon":            _assign_stars(p_wilcoxon),
            })

    result_df = pd.DataFrame(rows)

    # ── Apply BH FDR correction across all comparisons ───────────────────────
    if not result_df.empty:
        for p_col, sig_col, fdr_col, stars_fdr_col in [
            ("p_ttest",    "sig_ttest",    "p_ttest_fdr",    "sig_ttest_fdr"),
            ("p_wilcoxon", "sig_wilcoxon", "p_wilcoxon_fdr", "sig_wilcoxon_fdr"),
        ]:
            pvals = result_df[p_col].values.astype(float)
            valid  = ~np.isnan(pvals)
            fdr_vals = np.full(len(pvals), float("nan"))
            if valid.any():
                fdr_vals[valid] = _fdr_bh(pvals[valid])
            result_df[fdr_col]     = [round(float(v), 4) if not np.isnan(v) else float("nan")
                                      for v in fdr_vals]
            result_df[stars_fdr_col] = [_assign_stars(v) if not np.isnan(v) else "nan"
                                        for v in fdr_vals]

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

# Lighter fill colors for boxes, darker dots for stripplot
_BOX_PALETTE = {
    "original":          "#c5aee0",
    "original_filtered": "#c49a94",
    "rankbird":          "#7EB8D4",
    "ranking":           "#F4A36A",
    "ranking_sig":       "#8FD17E",
    "ranking_relu":      "#E87E7E",
}
_STRIP_PALETTE = {
    "original":          "#5a3a7a",
    "original_filtered": "#5a2a24",
    "rankbird":          "#2a6e8a",
    "ranking":           "#b5581e",
    "ranking_sig":       "#2a7a2a",
    "ranking_relu":      "#8a2a2a",
}


def _make_combined_df(all_results: dict, dtypes: list) -> pd.DataFrame:
    """Merge all dtype × approach results into one DataFrame with an 'approach' column."""
    frames = []
    for dtype in dtypes:
        for approach in APPROACHES:
            df = all_results.get((dtype, approach), pd.DataFrame())
            if df.empty:
                continue
            df = df.copy()
            df["approach"] = approach
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_BOX_WIDTH   = 0.45
_N_APPROACHES = len(APPROACHES)
# Seaborn dodge offsets for each hue level, assuming boxes are evenly spaced
# within width, centered at the group x-position.
_HUE_OFFSETS = {
    approach: (_BOX_WIDTH / _N_APPROACHES) * (i - (_N_APPROACHES - 1) / 2)
    for i, approach in enumerate(APPROACHES)
}


def _draw_panel(
    ax,
    data: pd.DataFrame,
    title: str,
    label_fontsize: int = 14,
    title_fontsize: int = 14,
    tick_fontsize: int = 11,
    legend_fontsize: int = 11,
    box_width: float = 0.7,
    dot_size: float = 4,
):
    """Draw one boxplot+stripplot panel onto ax with mean markers."""
    import matplotlib.patches as _mp
    data = data.copy()
    data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
    protocol_order  = [p for p in PROTOCOLS if p in data["protocol"].unique()]
    present_approaches = [a for a in APPROACHES if a in data["approach"].unique()]

    sns.boxplot(
        data=data,
        x="protocol", y="auc",
        hue="approach",
        order=protocol_order,
        hue_order=present_approaches,
        palette={a: _BOX_PALETTE[a] for a in present_approaches},
        width=box_width,
        fliersize=0,
        linewidth=1.4,
        medianprops=dict(color="#DC143C", linewidth=2.5),
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="protocol", y="auc",
        hue="approach",
        order=protocol_order,
        hue_order=present_approaches,
        palette={a: _STRIP_PALETTE[a] for a in present_approaches},
        dodge=True,
        alpha=0.5, size=dot_size, jitter=True,
        legend=False,
        ax=ax,
    )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    all_auc = data["auc"].dropna()
    if not all_auc.empty:
        q05 = float(np.percentile(all_auc, 5))
        q95 = float(np.percentile(all_auc, 95))
        pad = max(0.02, (q95 - q05) * 0.12)
        ax.set_ylim(max(0.0, q05 - pad), min(1.0, q95 + pad))

    # ── Mean markers (white diamond with black border) ────────────────────────
    hue_offsets = {
        a: (box_width / len(present_approaches)) * (i - (len(present_approaches) - 1) / 2)
        for i, a in enumerate(present_approaches)
    }
    for x_pos, protocol in enumerate(protocol_order):
        for approach in present_approaches:
            grp = data[(data["protocol"] == protocol) &
                       (data["approach"] == approach)]["auc"].dropna()
            if grp.empty:
                continue
            ax.scatter(
                x_pos + hue_offsets[approach], grp.mean(),
                marker="D", s=35, color="white", edgecolors="black",
                linewidths=1.4, zorder=5,
            )

    # Tighten x-axis to remove blank space between groups
    ax.set_xlim(-0.5, len(protocol_order) - 0.5)

    _PROTO_DISPLAY = {
        "Internal Validation": "Mixed-\nDatasets",
        "Within Learning":     "Within-\nDatasets",
    }
    ax.set_xticks(range(len(protocol_order)))
    ax.set_xticklabels(
        [_PROTO_DISPLAY.get(p, p) for p in protocol_order],
        fontsize=tick_fontsize, rotation=0,
    )
    ax.set_xlabel("Protocol", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=label_fontsize, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        _mp.Patch(facecolor=_BOX_PALETTE[a], edgecolor="black", linewidth=0.8,
                  label=APPROACH_LABELS.get(a, a))
        for a in present_approaches
    ]
    ax.legend(handles=legend_handles, fontsize=legend_fontsize, framealpha=0.9,
              loc="lower right")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _plot_comparison(
    all_results: dict,
    dtypes: list,
    output_path: Path,
    sigmoid_k: float,
    sigmoid_center: float,
    relu_threshold: float,
):
    """
    Three subplots: one per dtype + one for all datasets combined.
    Each subplot: boxplot + stripplot, x=protocol, hue=approach (5 approaches).
    Stars (vs current) from post-hoc, all on the same row above the boxes.
    Style mirrors figure 5.
    """
    panels = dtypes + ["All datasets"]
    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    # Per-dtype panels
    for ax, dtype in zip(axes, dtypes):
        frames = [
            all_results[(dtype, a)].assign(approach=a)
            for a in APPROACHES
            if not all_results.get((dtype, a), pd.DataFrame()).empty
        ]
        if not frames:
            ax.set_title(f"{dtype} — no data", fontsize=11)
            continue
        _draw_panel(ax, pd.concat(frames, ignore_index=True), title=dtype)

    # Combined panel (all dtypes together)
    combined_all = _make_combined_df(all_results, dtypes)
    if not combined_all.empty:
        _draw_panel(axes[-1], combined_all, title="All datasets")
    else:
        axes[-1].set_title("All datasets — no data", fontsize=11)

    param_str = (
        f"sigmoid k={sigmoid_k}, center={sigmoid_center} | "
        f"relu threshold={relu_threshold}"
    )
    fig.suptitle(
        f"Distribution Approach Comparison\n({param_str})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _compute_delta_results(all_results: dict, dtypes: list) -> dict:
    """
    For each (dtype, approach) where approach != 'original', subtract the
    original AUC from each matching (dataset, protocol) row.
    Returns a dict with the same keys but delta_auc values.
    """
    delta = {}
    for dtype in dtypes:
        orig = all_results.get((dtype, "original"), pd.DataFrame())
        if orig.empty:
            continue
        orig_lookup = orig.set_index(["dataset", "protocol"])["auc"]

        for approach in APPROACHES:
            if approach == "original":
                continue
            df = all_results.get((dtype, approach), pd.DataFrame())
            if df.empty:
                continue
            df = df.copy()
            df["auc"] = df.apply(
                lambda r: r["auc"] - orig_lookup.get((r["dataset"], r["protocol"]), float("nan")),
                axis=1,
            )
            delta[(dtype, approach)] = df
    return delta


def _draw_delta_panel(
    ax,
    data: pd.DataFrame,
    title: str,
    label_fontsize: int = 14,
    title_fontsize: int = 14,
    tick_fontsize: int = 11,
    legend_fontsize: int = 11,
    box_width: float = 0.7,
    dot_size: float = 4,
):
    """Draw one delta-AUC panel (AUC - original) onto ax."""
    import matplotlib.patches as _mp
    data = data.copy()
    data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
    protocol_order     = [p for p in PROTOCOLS if p in data["protocol"].unique()]
    present_approaches = [a for a in APPROACHES if a != "original" and a in data["approach"].unique()]

    sns.boxplot(
        data=data,
        x="protocol", y="auc",
        hue="approach",
        order=protocol_order,
        hue_order=present_approaches,
        palette={a: _BOX_PALETTE[a] for a in present_approaches},
        width=box_width,
        fliersize=0,
        linewidth=1.4,
        medianprops=dict(color="#DC143C", linewidth=2.5),
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="protocol", y="auc",
        hue="approach",
        order=protocol_order,
        hue_order=present_approaches,
        palette={a: _STRIP_PALETTE[a] for a in present_approaches},
        dodge=True,
        alpha=0.5, size=dot_size, jitter=True,
        legend=False,
        ax=ax,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.8, alpha=0.8)

    hue_offsets = {
        a: (box_width / len(present_approaches)) * (i - (len(present_approaches) - 1) / 2)
        for i, a in enumerate(present_approaches)
    }
    for x_pos, protocol in enumerate(protocol_order):
        for approach in present_approaches:
            grp = data[(data["protocol"] == protocol) &
                       (data["approach"] == approach)]["auc"].dropna()
            if grp.empty:
                continue
            ax.scatter(
                x_pos + hue_offsets[approach], grp.mean(),
                marker="D", s=35, color="white", edgecolors="black",
                linewidths=1.4, zorder=5,
            )

    ax.set_xlim(-0.5, len(protocol_order) - 0.5)
    _PROTO_DISPLAY = {
        "Internal Validation": "Mixed-\nDatasets",
        "Within Learning":     "Within-\nDatasets",
    }
    ax.set_xticks(range(len(protocol_order)))
    ax.set_xticklabels(
        [_PROTO_DISPLAY.get(p, p) for p in protocol_order],
        fontsize=tick_fontsize, rotation=0,
    )
    ax.set_xlabel("Protocol", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("ΔAUC (vs Original)", fontsize=label_fontsize, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        _mp.Patch(facecolor=_BOX_PALETTE[a], edgecolor="black", linewidth=0.8,
                  label=APPROACH_LABELS.get(a, a))
        for a in present_approaches
    ]
    ax.legend(handles=legend_handles, fontsize=legend_fontsize, framealpha=0.9,
              loc="lower right")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _plot_delta_comparison(
    all_results: dict,
    dtypes: list,
    output_path: Path,
    sigmoid_k: float,
    sigmoid_center: float,
    relu_threshold: float,
):
    """
    Same layout as _plot_comparison but y-axis is ΔAUC (each approach minus original).
    'original' is dropped; the zero line marks the original performance.
    """
    delta = _compute_delta_results(all_results, dtypes)
    if not delta:
        print("  [delta plot] no delta data — skipping.")
        return

    panels = dtypes + ["All datasets"]
    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, dtype in zip(axes, dtypes):
        frames = [
            delta[(dtype, a)].assign(approach=a)
            for a in APPROACHES
            if (dtype, a) in delta and not delta[(dtype, a)].empty
        ]
        if not frames:
            ax.set_title(f"{dtype} — no data", fontsize=11)
            continue
        _draw_delta_panel(ax, pd.concat(frames, ignore_index=True), title=dtype)

    all_frames = [
        delta[(dtype, a)].assign(approach=a)
        for dtype in dtypes
        for a in APPROACHES
        if (dtype, a) in delta and not delta[(dtype, a)].empty
    ]
    if all_frames:
        _draw_delta_panel(axes[-1], pd.concat(all_frames, ignore_index=True), title="All datasets")
    else:
        axes[-1].set_title("All datasets — no data", fontsize=11)

    param_str = (
        f"sigmoid k={sigmoid_k}, center={sigmoid_center} | "
        f"relu threshold={relu_threshold}"
    )
    fig.suptitle(
        f"Distribution Approach — ΔAUC vs Original\n({param_str})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_delta_plot(output_dir: Path) -> None:
    """
    Load existing results_*.csv files from output_dir and generate the delta
    plot. Use this to add a delta plot to any folder that already has results
    (e.g. distribution_approach_trial with LightGBM results).
    """
    output_dir = Path(output_dir)
    all_results: dict = {}

    for dtype in ["Metagenomics", "Amplicon", "Combined"]:
        for approach in APPROACHES:
            p = output_dir / f"results_{dtype.lower()}_{approach}.csv"
            if p.exists():
                all_results[(dtype, approach)] = pd.read_csv(p)

    present_dtypes = [d for d in ["Metagenomics", "Amplicon", "Combined"]
                      if any((d, a) in all_results for a in APPROACHES)]
    if not all_results:
        print(f"  [generate_delta_plot] no results found in {output_dir}")
        return

    _plot_delta_comparison(
        all_results=all_results,
        dtypes=present_dtypes,
        output_path=output_dir / "distribution_approach_delta.png",
        sigmoid_k=SIGMOID_K,
        sigmoid_center=SIGMOID_CENTER,
        relu_threshold=RELU_THRESHOLD,
    )
    _plot_train_test_delta(
        all_results=all_results,
        dtypes=present_dtypes,
        output_path=output_dir / "distribution_approach_train_test_delta.png",
    )


def _plot_train_test_delta(
    all_results: dict,
    dtypes: list,
    output_path: Path,
):
    """
    Plot train − test AUC delta for all approaches (including original).
    Boxplot shows distribution + median (red line); diamond marker shows mean.
    One column per dtype + All datasets.
    """
    import matplotlib.patches as _mp

    panels = dtypes + ["All datasets"]
    n_panels = len(panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    for col_idx, panel in enumerate(panels):
        ax = axes[0][col_idx]

        if panel == "All datasets":
            frames = [
                all_results[(dtype, a)].assign(approach=a)
                for dtype in dtypes for a in APPROACHES
                if not all_results.get((dtype, a), pd.DataFrame()).empty
            ]
        else:
            frames = [
                all_results[(panel, a)].assign(approach=a)
                for a in APPROACHES
                if not all_results.get((panel, a), pd.DataFrame()).empty
            ]
        if not frames:
            ax.set_visible(False)
            continue

        data = pd.concat(frames, ignore_index=True)
        if "train_auc" not in data.columns or "auc" not in data.columns:
            ax.set_visible(False)
            continue

        data["delta"] = pd.to_numeric(data["train_auc"], errors="coerce") \
                      - pd.to_numeric(data["auc"],       errors="coerce")

        protocol_order     = [p for p in PROTOCOLS if p in data["protocol"].unique()]
        present_approaches = [a for a in APPROACHES if a in data["approach"].unique()]

        # Boxplot — box line = median
        sns.boxplot(
            data=data,
            x="protocol", y="delta",
            hue="approach",
            order=protocol_order,
            hue_order=present_approaches,
            palette={a: _BOX_PALETTE[a] for a in present_approaches},
            width=0.7,
            fliersize=0,
            linewidth=1.2,
            medianprops=dict(color="#DC143C", linewidth=2.5),
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="protocol", y="delta",
            hue="approach",
            order=protocol_order,
            hue_order=present_approaches,
            palette={a: _STRIP_PALETTE[a] for a in present_approaches},
            dodge=True, alpha=0.4, size=3, jitter=True,
            legend=False, ax=ax,
        )

        # Mean markers — white diamond with black border (same style as comparison plot)
        box_width = 0.7
        hue_offsets = {
            a: (box_width / len(present_approaches)) * (i - (len(present_approaches) - 1) / 2)
            for i, a in enumerate(present_approaches)
        }
        for x_pos, protocol in enumerate(protocol_order):
            for approach in present_approaches:
                grp = data[(data["protocol"] == protocol) &
                           (data["approach"] == approach)]["delta"].dropna()
                if grp.empty:
                    continue
                ax.scatter(
                    x_pos + hue_offsets[approach], grp.mean(),
                    marker="D", s=35, color="white", edgecolors="black",
                    linewidths=1.4, zorder=5,
                )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_title(panel, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Protocol", fontsize=11, fontweight="bold")
        ax.set_ylabel("ΔAUC (train − test)" if col_idx == 0 else "",
                      fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if col_idx == n_panels - 1:
            approach_handles = [
                _mp.Patch(facecolor=_BOX_PALETTE[a], edgecolor="black",
                          linewidth=0.8, label=APPROACH_LABELS.get(a, a))
                for a in present_approaches
            ]
            stat_handles = [
                plt.Line2D([0], [0], color="#DC143C", linewidth=2.5, label="Median"),
                plt.Line2D([0], [0], marker="D", color="white", linewidth=0,
                           markersize=7, markeredgecolor="black",
                           markeredgewidth=1.4, label="Mean"),
            ]
            ax.legend(handles=approach_handles + stat_handles,
                      fontsize=8, framealpha=0.9, loc="upper right",
                      title="Approach / Stat")
        else:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    fig.suptitle("Train − Test AUC Delta by Approach\n(positive = overfitting)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_distribution_summary_figure(
    base_dir: Path,
    combined_dir: Path,
    output_path: Path,
) -> None:
    """
    Single comprehensive figure combining per-dtype and cross-dtype results.

    Panels (left → right):
      Metagenomics | Amplicon | All datasets | Combined (cross-dtype)

    Skips any panel whose CSVs are absent so the figure degrades gracefully
    when only one set of results exists.
    """
    base_dir     = Path(base_dir)
    combined_dir = Path(combined_dir)

    all_results: dict = {}

    # Load per-dtype results
    for dtype in ["Metagenomics", "Amplicon"]:
        for approach in APPROACHES:
            p = base_dir / f"results_{dtype.lower()}_{approach}.csv"
            if p.exists():
                all_results[(dtype, approach)] = pd.read_csv(p)

    # Load cross-dtype Combined results
    for approach in APPROACHES:
        p = combined_dir / f"results_combined_{approach}.csv"
        if p.exists():
            all_results[("Combined", approach)] = pd.read_csv(p)

    per_dtypes   = [d for d in ["Metagenomics", "Amplicon"]
                    if any((d, a) in all_results for a in APPROACHES)]
    has_combined = any(("Combined", a) in all_results for a in APPROACHES)

    if not per_dtypes and not has_combined:
        print(f"  [summary figure] no results found in {base_dir} or {combined_dir} — skipping")
        return

    panels = per_dtypes[:]
    if per_dtypes:
        panels.append("All datasets")
    if has_combined:
        panels.append("Combined")

    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "All datasets":
            df = _make_combined_df(all_results, per_dtypes)
        elif panel == "Combined":
            frames = [
                all_results[("Combined", a)].assign(approach=a)
                for a in APPROACHES if ("Combined", a) in all_results
                and not all_results[("Combined", a)].empty
            ]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            frames = [
                all_results[(panel, a)].assign(approach=a)
                for a in APPROACHES if (panel, a) in all_results
                and not all_results[(panel, a)].empty
            ]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df.empty:
            ax.set_title(f"{panel} — no data", fontsize=11)
            continue
        _draw_panel(ax, df, title=panel)

    fig.suptitle("Distribution Approach Comparison — Full Summary",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

_APPROACH_TO_NORM = {
    "original":          None,
    "original_filtered": "filter_only",
    "rankbird":          "rankbird_wasserstein",
    "ranking":           "rankbird_ranking",
    "ranking_sig":       "rankbird_sigmoid",
    "ranking_relu":      "rankbird_relu",
}


def run_distribution_investigation(
    phenotypes: list,
    output_dir: Path,
    plot_only: bool = False,
    stability_percentile_metagenomics: float = STABILITY_PERCENTILE["Metagenomics"],
    stability_percentile_amplicon: float      = STABILITY_PERCENTILE["Amplicon"],
    stability_percentile_global_combined: float = 0.6,
    min_size: int = MIN_SAMPLES_PER_DATASET,
    sigmoid_k: float = SIGMOID_K,
    sigmoid_center: float = SIGMOID_CENTER,
    relu_threshold: float = RELU_THRESHOLD,
    taxonomy_level_metagenomics=None,
    taxonomy_level_amplicon=None,
    taxonomy_level_combined=None,
    shuffle_ordered: bool = False,
    rank_tie_method: str = "first",
    use_optuna: bool = False,
    n_trials: int = 30,
):
    """
    Compare distribution-adjustment approaches on all phenotypes/dtypes.

    Always computes Metagenomics, Amplicon, and Combined (cross-dtype), producing
    four panels: Metagenomics, Amplicon, All dtypes (merged), and Cross dtypes.

    Parameters
    ----------
    phenotypes  : list of (phenotype, dtype) tuples
    output_dir  : directory for CSVs and figure
    plot_only   : if True, load existing CSVs and skip missing ones
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtypes = ["Metagenomics", "Amplicon", "Combined"]
    _tax_combined = taxonomy_level_combined if str(taxonomy_level_combined) != "None" else None
    stability_percentile_by_dtype = {
        "Metagenomics": stability_percentile_metagenomics,
        "Amplicon":     stability_percentile_amplicon,
        "Combined":     stability_percentile_global_combined,
    }
    taxonomy_level_by_dtype = {
        "Metagenomics": taxonomy_level_metagenomics,
        "Amplicon":     taxonomy_level_amplicon,
        "Combined":     _tax_combined,
    }

    # Identify single-class datasets once — excluded from all results
    single_class_names = _get_single_class_datasets(phenotypes)

    # ── One-time upfront order-check across all datasets ─────────────────────
    if not plot_only:
        _all_micro, _all_tgt, _all_names, _all_pheno = [], [], [], []
        for phenotype, dtype in phenotypes:
            folder = DATA_ROOT / f"{phenotype} {dtype}"
            if not folder.exists():
                continue
            try:
                dfs, tgts, names = load_microbiome_datasets_with_targets(folder)
                _all_micro.extend(dfs); _all_tgt.extend(tgts); _all_names.extend(names)
                _all_pheno.extend([f"{phenotype} {dtype}"] * len(names))
            except Exception:
                pass
        _, _, _ordered_info = _detect_and_shuffle_ordered(
            _all_micro, _all_tgt, _all_names,
            apply_shuffle=shuffle_ordered, verbose=True,
        )
        _pheno_for_name = dict(zip(_all_names, _all_pheno))
        pd.DataFrame([
            {
                "dataset":     d["dataset"],
                "phenotype":   _pheno_for_name.get(d["dataset"], ""),
                "first_class": d["first_class"],
                "first_label": "control" if d["first_class"] == 0 else "case",
                "was_shuffled": shuffle_ordered,
            }
            for d in _ordered_info
        ]).to_csv(output_dir / "ordered_datasets.csv", index=False)
        print(f"  [ORDER-CHECK] saved → {output_dir / 'ordered_datasets.csv'}")

    # ── Feature count pre-pass (fast, no ML, always runs) ────────────────────
    feature_count_records: list = []
    for dtype in dtypes:
        pheno_subset_fc = phenotypes if dtype == "Combined" \
                          else [(p, t) for p, t in phenotypes if t == dtype]
        if not pheno_subset_fc:
            continue

        # Load microbiome data only (discard targets)
        _dfs, _names = [], []
        for phenotype, t in pheno_subset_fc:
            folder = DATA_ROOT / f"{phenotype} {t}"
            if not folder.exists():
                continue
            try:
                dfs, _, names = load_microbiome_datasets_with_targets(folder)
                _dfs.extend(dfs); _names.extend(names)
            except Exception:
                pass
        if not _dfs:
            continue

        # Apply taxonomy filter (same as all approaches)
        _tax = taxonomy_level_by_dtype[dtype]
        _filtered = filter_to_level(_dfs, _tax)

        # original: union of all columns
        _union_n = len(set().union(*[set(df.columns) for df in _filtered]))

        # filtered approaches: stability filter on the union
        _all_microbes = union_microbes(_filtered)
        _nz = nonzero_percent_by_dataset(_filtered, _names, _all_microbes)
        _kept = auto_stability_filter(_nz, percentile=stability_percentile_by_dtype[dtype])
        _kept_n = len(_kept)

        for approach in APPROACHES:
            if approach == "original":
                n_feat, ctype = _union_n, "union (varies per fold)"
            else:
                # rankbird uses its own pipeline but same stability threshold as proxy
                n_feat, ctype = _kept_n, "aligned (same for all folds)"
            feature_count_records.append({
                "approach":   approach,
                "dtype":      dtype,
                "threshold":  stability_percentile_by_dtype[dtype],
                "n_features": n_feat,
                "count_type": ctype,
            })

    # "All datasets" rows: sum Metagenomics + Amplicon (separate pools)
    _fc_df = pd.DataFrame(feature_count_records)
    if not _fc_df.empty:
        for approach in APPROACHES:
            sub = _fc_df[(_fc_df["dtype"] != "Combined") &
                         (_fc_df["approach"] == approach)]
            if sub.empty:
                continue
            ctype = "union (varies per fold)" if approach == "original" \
                    else "sum of separate dtype pools"
            feature_count_records.append({
                "approach":   approach,
                "dtype":      "All datasets",
                "threshold":  "mixed",
                "n_features": int(sub["n_features"].sum()),
                "count_type": ctype,
            })
        feat_df = pd.DataFrame(feature_count_records)
        feat_df.to_csv(output_dir / "feature_counts.csv", index=False)
        print(f"\nSaved: {output_dir / 'feature_counts.csv'}")
        print(feat_df.to_string(index=False))

    all_results: dict = {}    # {(dtype, approach): full results_df}

    for dtype in dtypes:
        pheno_subset = phenotypes if dtype == "Combined" else [(p, t) for p, t in phenotypes if t == dtype]
        if not pheno_subset:
            continue

        for approach in APPROACHES:
            tag      = f"{dtype.lower()}_{approach}"
            csv_path = output_dir / f"results_{tag}.csv"
            agg_path = output_dir / f"agg_{tag}.csv"

            print(f"\n=== {dtype}  |  {APPROACH_LABELS[approach]} ===")

            if csv_path.exists():
                print(f"  [LOAD] {csv_path.name}")
                results_df = pd.read_csv(csv_path)
            elif plot_only:
                print(f"  [SKIP] Missing {csv_path.name} — run without plot_only first.")
                continue
            else:
                if dtype == "Combined":
                    # Route directly to _run_global_for_dtype which handles the full
                    # pooled normalization + per-phenotype learning in one shot.
                    results_df, _ = _run_global_for_dtype(
                        pheno_subset,
                        normalization_approach=_APPROACH_TO_NORM[approach],
                        min_samples_per_dataset=min_size,
                        stability_percentile_global=stability_percentile_by_dtype[dtype],
                        taxonomy_level=taxonomy_level_by_dtype[dtype],
                        shuffle_ordered=shuffle_ordered,
                        rank_tie_method=rank_tie_method,
                        use_optuna=use_optuna,
                        n_trials=n_trials,
                    )
                else:
                    results_df, _ = _run_approach_for_dtype(
                        phenotypes=pheno_subset,
                        dtype=dtype,
                        approach=approach,
                        stability_percentile=stability_percentile_by_dtype[dtype],
                        min_size=min_size,
                        sigmoid_k=sigmoid_k,
                        sigmoid_center=sigmoid_center,
                        relu_threshold=relu_threshold,
                        taxonomy_level=taxonomy_level_by_dtype[dtype],
                        shuffle_ordered=shuffle_ordered,
                        rank_tie_method=rank_tie_method,
                        use_optuna=use_optuna,
                        n_trials=n_trials,
                    )
                # Filter single-class datasets before saving
                results_df = _drop_single_class(results_df, single_class_names)
                results_df.to_csv(csv_path, index=False)

            # Aggregation
            agg_df = _aggregate_auc(results_df)
            agg_df.to_csv(agg_path, index=False)

            all_results[(dtype, approach)] = results_df

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\nRunning statistical tests (ANOVA + pairwise paired t-test + Wilcoxon / Bonferroni + FDR) ...")
    stats_dict = run_statistical_tests(all_results, dtypes)

    anova_df   = stats_dict.get("anova",   pd.DataFrame())
    posthoc_df = stats_dict.get("posthoc", pd.DataFrame())

    if not anova_df.empty:
        anova_df.to_csv(output_dir / "stats_anova.csv", index=False)
        print(f"\nANOVA results:")
        print(anova_df[["panel", "protocol", "n_datasets", "f_statistic",
                         "p_value", "stars"]].to_string(index=False))

    if not posthoc_df.empty:
        posthoc_df.to_csv(output_dir / "stats_posthoc.csv", index=False)
        show_cols = ["panel", "protocol", "approach_1", "approach_2",
                     "mean_diff", "t_statistic", "p_value_raw",
                     "p_value_bonferroni", "stars"]
        if "p_value_fdr" in posthoc_df.columns:
            show_cols += ["p_value_fdr", "stars_fdr"]
        if "p_wilcoxon_raw" in posthoc_df.columns:
            show_cols += ["p_wilcoxon_raw", "p_wilcoxon_bonferroni", "stars_wilcoxon",
                          "p_wilcoxon_fdr", "stars_wilcoxon_fdr"]
        print(f"\nPost-hoc (Bonferroni + BH-FDR corrected paired t-test + Wilcoxon):")
        print(posthoc_df[show_cols].to_string(index=False))

    # ── Delta + tests: original vs each approach ──────────────────────────────
    delta_df = compute_normalization_comparison(all_results, dtypes)
    if not delta_df.empty:
        delta_df.to_csv(output_dir / "delta_tests.csv", index=False)
        print("\nOriginal vs RANK-BIRD — delta & statistical tests:")
        print(delta_df.to_string(index=False))

    _plot_comparison(
        all_results=all_results,
        dtypes=dtypes,
        output_path=output_dir / "distribution_approach_comparison.png",
        sigmoid_k=sigmoid_k,
        sigmoid_center=sigmoid_center,
        relu_threshold=relu_threshold,
    )
    _plot_delta_comparison(
        all_results=all_results,
        dtypes=dtypes,
        output_path=output_dir / "distribution_approach_delta.png",
        sigmoid_k=sigmoid_k,
        sigmoid_center=sigmoid_center,
        relu_threshold=relu_threshold,
    )
    _plot_train_test_delta(
        all_results=all_results,
        dtypes=dtypes,
        output_path=output_dir / "distribution_approach_train_test_delta.png",
    )


def print_auc_summary_table(output_dir: Path, approaches=("original", "rankbird")):
    """
    Print a formatted mean ± std / median comparison table for the given
    approaches across all dtypes and protocols, reading from agg_*.csv files.

    Example
    -------
        from experiments.investigate_distribution_approach import print_auc_summary_table
        print_auc_summary_table(Path("figures_out/figure_3/3c"))
    """
    output_dir = Path(output_dir)

    # Discover available dtypes from saved agg files
    dtypes_found = sorted({
        p.stem.split("_", 1)[1].rsplit("_", len(APPROACHES[0].split("_")))[0]
        for p in output_dir.glob("agg_*.csv")
    })
    # Simpler: just check the known combinations
    rows = []
    for p in sorted(output_dir.glob("agg_*.csv")):
        stem = p.stem[len("agg_"):]          # e.g. "metagenomics_rankbird"
        # match approach suffix
        approach = next((a for a in APPROACHES if stem.endswith(a)), None)
        if approach not in approaches:
            continue
        dtype_raw = stem[: len(stem) - len(approach) - 1]   # e.g. "metagenomics"
        dtype = dtype_raw.title()                            # "Metagenomics"
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for _, r in df.iterrows():
            rows.append({
                "dtype":      dtype,
                "approach":   APPROACH_LABELS.get(approach, approach),
                "protocol":   r["protocol"],
                "mean":       round(r["mean_auc"], 3),
                "median":     round(r["median_auc"], 3) if "median_auc" in r else float("nan"),
                "std":        round(r["std_auc"], 3),
                "n":          int(r["n_datasets"]),
            })

    if not rows:
        print("No agg_*.csv files found. Run the investigation first.")
        return

    tbl = pd.DataFrame(rows).sort_values(["dtype", "protocol", "approach"])

    # ── Pretty print ─────────────────────────────────────────────────────────
    col_w = {"dtype": 14, "protocol": 22, "approach": 20, "mean": 7,
             "median": 8, "std": 7, "n": 4}
    header = (f"{'Dtype':<{col_w['dtype']}}  {'Protocol':<{col_w['protocol']}}"
              f"  {'Approach':<{col_w['approach']}}  {'Mean':>{col_w['mean']}}"
              f"  {'Median':>{col_w['median']}}  {'Std':>{col_w['std']}}"
              f"  {'N':>{col_w['n']}}")
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    prev_dtype = prev_proto = None
    for _, r in tbl.iterrows():
        if r["dtype"] != prev_dtype or r["protocol"] != prev_proto:
            if prev_dtype is not None:
                print()
            prev_dtype, prev_proto = r["dtype"], r["protocol"]

        print(f"{r['dtype']:<{col_w['dtype']}}  {r['protocol']:<{col_w['protocol']}}"
              f"  {r['approach']:<{col_w['approach']}}  {r['mean']:>{col_w['mean']}.3f}"
              f"  {r['median']:>{col_w['median']}.3f}  {r['std']:>{col_w['std']}.3f}"
              f"  {r['n']:>{col_w['n']}}")

    print(sep + "\n")

    # ── Delta table ───────────────────────────────────────────────────────────
    delta_path = output_dir / "delta_tests.csv"
    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        cw = {"panel": 16, "protocol": 22, "n": 4,
              "mb": 8, "mt": 8, "dm": 7, "pt": 8, "st": 4,
              "mdb": 8, "mdt": 8, "dmed": 7, "pw": 8, "sw": 4}
        hdr = (f"\n{'Panel':<{cw['panel']}}  {'Protocol':<{cw['protocol']}}"
               f"  {'N':>{cw['n']}}"
               f"  {'Mean(O)':>{cw['mb']}}  {'Mean(R)':>{cw['mt']}}  {'ΔMean':>{cw['dm']}}"
               f"  {'p(t-test)':>{cw['pt']}}  {'':>{cw['st']}}"
               f"  {'Med(O)':>{cw['mdb']}}  {'Med(R)':>{cw['mdt']}}  {'ΔMedian':>{cw['dmed']}}"
               f"  {'p(Wilcox)':>{cw['pw']}}  {'':>{cw['sw']}}")
        sep2 = "-" * (len(hdr) - 1)
        print(sep2)
        print(hdr)
        print(sep2)
        # detect dynamic column names (mean_original + mean_<label>)
        mean_orig_col   = "mean_original"
        median_orig_col = "median_original"
        mean_treat_col   = next((c for c in delta.columns if c.startswith("mean_")   and c != mean_orig_col),   None)
        median_treat_col = next((c for c in delta.columns if c.startswith("median_") and c != median_orig_col), None)
        prev = None
        for _, r in delta.iterrows():
            if r["panel"] != prev:
                if prev is not None:
                    print()
                prev = r["panel"]
            mean_o   = r[mean_orig_col]   if mean_orig_col   in r.index else float("nan")
            mean_t   = r[mean_treat_col]  if mean_treat_col  else float("nan")
            med_o    = r[median_orig_col] if median_orig_col in r.index else float("nan")
            med_t    = r[median_treat_col] if median_treat_col else float("nan")
            print(f"{r['panel']:<{cw['panel']}}  {r['protocol']:<{cw['protocol']}}"
                  f"  {int(r['n_datasets']):>{cw['n']}}"
                  f"  {mean_o:>{cw['mb']}.3f}  {mean_t:>{cw['mt']}.3f}"
                  f"  {r['delta_mean']:>+{cw['dm']}.3f}  {r['p_ttest']:>{cw['pt']}.4f}"
                  f"  {str(r['sig_ttest']):<{cw['st']}}"
                  f"  {med_o:>{cw['mdb']}.3f}  {med_t:>{cw['mdt']}.3f}"
                  f"  {r['delta_median']:>+{cw['dmed']}.3f}  {r['p_wilcoxon']:>{cw['pw']}.4f}"
                  f"  {str(r['sig_wilcoxon']):<{cw['sw']}}")
        print(sep2 + "\n")
