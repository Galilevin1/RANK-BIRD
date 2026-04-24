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
from scipy.stats import ttest_rel, f_oneway

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
    _run_protocols_on_group, _run_global_for_dtype,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "Data"

# ── Approach catalogue ────────────────────────────────────────────────────────

APPROACHES = ["original", "original_filtered", "rankbird", "ranking", "ranking_sig", "ranking_relu"]

APPROACH_LABELS = {
    "original":          "Original",
    "original_filtered": "Original + Filter",
    "rankbird":          "RANK-BIRD",
    "ranking":           "Ranking",
    "ranking_sig":       "Ranking + Sigmoid",
    "ranking_relu":      "Ranking + Relu",
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

    name_to_target = dict(zip(all_names, all_targets))

    # ── Apply normalization approach ──────────────────────────────────────────
    if approach == "original":
        return _run_global_for_dtype(phenotypes, normalization_approach=None,
                                     taxonomy_level=taxonomy_level)

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
        )

    elif approach == "ranking":
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome, all_names,
            stability_percentile=stability_percentile,
            norm_fn=rank_normalize,
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    elif approach == "ranking_sig":
        norm_fn = lambda X: sigmoid_normalize(X, k=sigmoid_k, center=sigmoid_center)
        all_microbiome, all_names = apply_ranking_pipeline(
            all_microbiome, all_names,
            stability_percentile=stability_percentile,
            norm_fn=norm_fn,
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    elif approach == "ranking_relu":
        norm_fn = lambda X: relu_normalize(X, threshold=relu_threshold)
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
        ))

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


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
    """Build panel_name → {approach: results_df} including 'All datasets'."""
    panels: dict = {}
    for dtype in dtypes:
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
        frames = [f for dtype in dtypes
                  for f in [all_results.get((dtype, approach), pd.DataFrame())]
                  if not f.empty]
        if frames:
            merged = pd.concat(frames, ignore_index=True).copy()
            merged["auc"] = pd.to_numeric(merged["auc"], errors="coerce")
            all_combined[approach] = merged
    if all_combined:
        panels["All datasets"] = all_combined

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
                    t_stat, p_raw = ttest_rel(aligned[a1], aligned[a2])
                except Exception:
                    continue
                p_bonf = min(float(p_raw) * n_tests, 1.0)
                sig    = p_bonf < alpha
                posthoc_rows.append({
                    "panel":               panel_name,
                    "protocol":            protocol,
                    "approach_1":          a1,
                    "approach_2":          a2,
                    "n_pairs":             len(common),
                    "mean_1":              float(aligned[a1].mean()),
                    "mean_2":              float(aligned[a2].mean()),
                    "mean_diff":           float(aligned[a1].mean() - aligned[a2].mean()),
                    "t_statistic":         float(t_stat),
                    "p_value_raw":         float(p_raw),
                    "p_value_bonferroni":  p_bonf,
                    "significant":         sig,
                    "stars":               _assign_stars(p_bonf),
                })

    return {
        "anova":   pd.DataFrame(anova_rows),
        "posthoc": pd.DataFrame(posthoc_rows),
    }


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


_BOX_WIDTH   = 0.7
_N_APPROACHES = len(APPROACHES)
# Seaborn dodge offsets for each hue level, assuming boxes are evenly spaced
# within width, centered at the group x-position.
_HUE_OFFSETS = {
    approach: (_BOX_WIDTH / _N_APPROACHES) * (i - (_N_APPROACHES - 1) / 2)
    for i, approach in enumerate(APPROACHES)
}


def _draw_panel(ax, data: pd.DataFrame, title: str):
    """Draw one boxplot+stripplot panel onto ax with mean markers."""
    data = data.copy()
    data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
    protocol_order = [p for p in PROTOCOLS if p in data["protocol"].unique()]

    sns.boxplot(
        data=data,
        x="protocol", y="auc",
        hue="approach",
        order=protocol_order,
        hue_order=APPROACHES,
        palette=_BOX_PALETTE,
        width=_BOX_WIDTH,
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
        hue_order=APPROACHES,
        palette=_STRIP_PALETTE,
        dodge=True,
        alpha=0.4, size=4, jitter=True,
        legend=False,
        ax=ax,
    )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Random (0.5)")

    all_auc = data["auc"].dropna()
    if not all_auc.empty:
        y_min = max(0.0, float(all_auc.min()) - 0.05)
        y_max = min(1.0, float(all_auc.max()) + 0.05)
        ax.set_ylim(y_min, y_max)

    # ── Mean markers (white diamond with black border) ────────────────────────
    for x_pos, protocol in enumerate(protocol_order):
        for approach in APPROACHES:
            grp = data[(data["protocol"] == protocol) &
                       (data["approach"] == approach)]["auc"].dropna()
            if grp.empty:
                continue
            ax.scatter(
                x_pos + _HUE_OFFSETS[approach], grp.mean(),
                marker="D", s=35, color="white", edgecolors="black",
                linewidths=1.4, zorder=5,
            )

    ax.set_xlabel("Protocol", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=12, fontweight="bold")
    # title pushed up to leave room above stars
    ax.set_title(title, fontsize=13, fontweight="bold", pad=28)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    labels = [APPROACH_LABELS.get(l, l) for l in labels]
    ax.legend(handles=handles, labels=labels, fontsize=9,
              framealpha=0.9, loc="lower right")

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
    min_size: int = MIN_SAMPLES_PER_DATASET,
    sigmoid_k: float = SIGMOID_K,
    sigmoid_center: float = SIGMOID_CENTER,
    relu_threshold: float = RELU_THRESHOLD,
    taxonomy_level_metagenomics=None,
    taxonomy_level_amplicon=None,
    cross_dtype_normalization: bool = False,
    stability_percentile_global_combined: float = 0.6,
    taxonomy_level_combined=None,
):
    """
    Compare distribution-adjustment approaches on all phenotypes/dtypes.

    Parameters
    ----------
    phenotypes  : list of (phenotype, dtype) tuples
    output_dir  : directory for CSVs and figure
    plot_only   : if True, skip computation and reload existing CSVs
    cross_dtype_normalization : if True, pool 16S + shotgun together for normalization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cross_dtype_normalization:
        dtypes = ["Combined"]
        stability_percentile_by_dtype = {"Combined": stability_percentile_global_combined}
        tax_level = taxonomy_level_combined if str(taxonomy_level_combined) != "None" else None
        taxonomy_level_by_dtype = {"Combined": tax_level}
    else:
        dtypes = ["Metagenomics", "Amplicon"]
        stability_percentile_by_dtype = {
            "Metagenomics": stability_percentile_metagenomics,
            "Amplicon":     stability_percentile_amplicon,
        }
        taxonomy_level_by_dtype = {
            "Metagenomics": taxonomy_level_metagenomics,
            "Amplicon":     taxonomy_level_amplicon,
        }

    # Identify single-class datasets once — excluded from all results
    single_class_names = _get_single_class_datasets(phenotypes)

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
                    results_df = _run_global_for_dtype(
                        pheno_subset,
                        normalization_approach=_APPROACH_TO_NORM[approach],
                        min_samples_per_dataset=min_size,
                        stability_percentile_global=stability_percentile_by_dtype[dtype],
                        taxonomy_level=taxonomy_level_by_dtype[dtype],
                    )
                else:
                    results_df = _run_approach_for_dtype(
                        phenotypes=pheno_subset,
                        dtype=dtype,
                        approach=approach,
                        stability_percentile=stability_percentile_by_dtype[dtype],
                        min_size=min_size,
                        sigmoid_k=sigmoid_k,
                        sigmoid_center=sigmoid_center,
                        relu_threshold=relu_threshold,
                        taxonomy_level=taxonomy_level_by_dtype[dtype],
                    )
                # Filter single-class datasets before saving
                results_df = _drop_single_class(results_df, single_class_names)
                results_df.to_csv(csv_path, index=False)

            # Aggregation
            agg_df = _aggregate_auc(results_df)
            agg_df.to_csv(agg_path, index=False)

            all_results[(dtype, approach)] = results_df

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\nRunning statistical tests (ANOVA + pairwise paired t-test / Bonferroni) ...")
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
        print(f"\nPost-hoc (Bonferroni-corrected paired t-test):")
        print(posthoc_df[["panel", "protocol", "approach_1", "approach_2",
                           "mean_diff", "t_statistic",
                           "p_value_raw", "p_value_bonferroni", "stars"]]
              .to_string(index=False))

    _plot_comparison(
        all_results=all_results,
        dtypes=dtypes,
        output_path=output_dir / "distribution_approach_comparison.png",
        sigmoid_k=sigmoid_k,
        sigmoid_center=sigmoid_center,
        relu_threshold=relu_threshold,
    )
