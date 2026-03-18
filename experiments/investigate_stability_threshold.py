"""
Investigation: effect of stability-filter percentile on mean AUC.

For each percentile in [0.1, 0.2, ..., 0.9] and each taxonomic level:
  all            — all microbes
  genus          — g__<name> present  AND  s__<name> NOT present
  species+genus  — g__<name> present  (genus-only AND genus+species features)

Produces a 3×2 figure (rows = levels, cols = Metagenomics / Amplicon).
Each subplot: AUC lines (left y-axis) + kept-microbe count & % (right y-axis).
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from evaluation.data_loading import load_microbiome_datasets_with_targets
from experiments.run_protocols_global_processing import (
    _run_protocols_on_group,
    _run_global_for_dtype,
)
from src.rankbird.normalization.pipeline import apply_normalization_pipeline

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
    (None, "All microbes"),
    ("g",  "Genus level"),
    ("s",  "Species + Genus level"),
]


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

def _has_named(feature: str, prefix: str) -> bool:
    """True if 'prefix' is followed by at least one word character."""
    return bool(re.search(re.escape(prefix) + r'\w', feature))


def _keep_at_level(feature: str, level) -> bool:
    if level is None:
        return True
    if level == "g":
        # Genus: named genus present, named species absent
        # (bare 's__' without a name is allowed)
        return _has_named(feature, "g__") and not _has_named(feature, "s__")
    if level == "s":
        # Species+Genus: has a named genus (includes genus-only and genus+species)
        return _has_named(feature, "g__")
    return True


def _filter_to_level(microbiome_dfs: list, level) -> list:
    if level is None:
        return microbiome_dfs
    filtered = []
    for df in microbiome_dfs:
        cols = [c for c in df.columns if _keep_at_level(c, level)]
        filtered.append(df[cols] if cols else df[[]])
    return filtered


# ── Dataset loader ────────────────────────────────────────────────────────────

def _load_microbiome_for_dtype(phenotypes: list, dtype: str):
    all_dfs, all_names = [], []
    for phenotype, t in phenotypes:
        if t != dtype:
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

def run_stability_sweep(
    phenotypes: list,
    dtype: str,
    level=None,
) -> pd.DataFrame:
    """
    Sweep stability percentiles, running all protocols at each step.
    Returns: [percentile, protocol, mean_auc, std_auc, n_phenotypes]
    """
    pheno_subset = [(p, t) for p, t in phenotypes if t == dtype]
    if not pheno_subset:
        raise ValueError(f"No phenotypes for dtype='{dtype}'")

    rows = []
    for pct in PERCENTILES:
        print(f"  [{dtype}] pct={pct:.1f}  level={level} ...")

        if level is None:
            results_df = _run_global_for_dtype(
                pheno_subset,
                apply_normalization=True,
                stability_percentile_global=pct,
            )
        else:
            results_df = _run_global_for_dtype_filtered(
                pheno_subset,
                level=level,
                stability_percentile_global=pct,
            )

        results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")

        for protocol in PROTOCOLS:
            sub = results_df[results_df["protocol"] == protocol]["auc"].dropna()
            if sub.empty:
                continue
            rows.append({
                "percentile":   pct,
                "protocol":     protocol,
                "mean_auc":     sub.mean(),
                "std_auc":      sub.std(ddof=1),
                "n_phenotypes": results_df[
                    results_df["protocol"] == protocol
                ]["phenotype"].nunique(),
            })

    return pd.DataFrame(rows)


def _run_global_for_dtype_filtered(
    phenotypes: list,
    level,
    stability_percentile_global: float = 0.5,
    min_samples_per_dataset: int = 550,
    apply_normalization: bool = True,
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

    # name→target map (before normalisation which may drop datasets)
    name_to_target = dict(zip(all_names, all_targets))

    if apply_normalization:
        all_microbiome, all_names = apply_normalization_pipeline(
            all_microbiome,
            all_names,
            global_analysis=True,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_global=stability_percentile_global,
        )

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
    pheno_subset = [(p, t) for p, t in phenotypes if t == dtype]
    if not pheno_subset:
        return {}

    if level is None:
        results_df = _run_global_for_dtype(
            pheno_subset, apply_normalization=False
        )
    else:
        results_df = _run_global_for_dtype_filtered(
            pheno_subset, level=level, apply_normalization=False
        )

    results_df["auc"] = pd.to_numeric(results_df["auc"], errors="coerce")
    return {
        protocol: results_df[results_df["protocol"] == protocol]["auc"].dropna().mean()
        for protocol in PROTOCOLS
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_sweep_panel(
    ax: plt.Axes,
    sweep_df: pd.DataFrame,
    count_df: pd.DataFrame,
    title: str,
    original_auc: dict = None,
):
    """AUC lines (left axis) + kept-microbe count & % annotations (right axis).
    original_auc: {protocol: mean_auc} drawn as horizontal dashed reference lines.
    """
    # Left: AUC sweep lines
    for protocol in PROTOCOLS:
        sub = sweep_df[sweep_df["protocol"] == protocol].sort_values("percentile")
        if sub.empty:
            continue
        color = PROTOCOL_COLORS[protocol]
        ax.plot(sub["percentile"], sub["mean_auc"],
                marker="o", linewidth=2, markersize=5,
                color=color, label=protocol)
        ax.fill_between(
            sub["percentile"],
            sub["mean_auc"] - sub["std_auc"],
            sub["mean_auc"] + sub["std_auc"],
            alpha=0.10, color=color,
        )
        # Original (no normalization) reference line
        if original_auc and protocol in original_auc:
            ax.axhline(
                original_auc[protocol],
                color=color, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"{protocol} (original)",
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.5,
               label="Random (0.5)")
    ax.set_ylabel("Mean AUC", fontsize=10)
    ax.set_xlabel("Stability Percentile", fontsize=10)
    ax.set_xticks(PERCENTILES[::2])   # every other tick to avoid crowding
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim(PERCENTILES[0] - 0.02, PERCENTILES[-1] + 0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Right: microbe count + % label
    ax2 = ax.twinx()
    ax2.plot(count_df["percentile"], count_df["n_kept"],
             color="black", linestyle="--", linewidth=1.8,
             marker="s", markersize=4, label="# kept microbes")
    ax2.set_ylabel("# kept microbes", fontsize=10, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    for _, row in count_df.iterrows():
        ax2.annotate(
            f"{int(row['n_kept'])} ({row['pct_kept']:.0f}%)",
            xy=(row["percentile"], row["n_kept"]),
            xytext=(0, 6), textcoords="offset points",
            fontsize=6.5, ha="center", color="black",
        )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=8, framealpha=0.85, loc="lower right")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stability_investigation(
    phenotypes: list,
    output_dir: Path,
    plot_only: bool = False,
):
    """
    Runs the full (dtype × level) sweep and saves CSVs + one combined figure.

    Parameters
    ----------
    phenotypes  : list of (phenotype, dtype) tuples
    output_dir  : directory for CSVs and figure
    plot_only   : if True, skip all computation and load existing CSVs instead
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtypes = ["Metagenomics", "Amplicon"]
    fig, axes = plt.subplots(
        len(LEVELS), len(dtypes),
        figsize=(7 * len(dtypes), 5 * len(LEVELS)),
    )

    for row_idx, (level, level_label) in enumerate(LEVELS):
        for col_idx, dtype in enumerate(dtypes):
            ax  = axes[row_idx][col_idx]
            tag = f"{dtype.lower()}_{level or 'all'}"
            print(f"\n=== {level_label}  |  {dtype} ===")

            if plot_only:
                # Load pre-existing CSVs
                count_path = output_dir / f"microbe_counts_{tag}.csv"
                sweep_path = output_dir / f"auc_sweep_{tag}.csv"
                orig_path  = output_dir / f"original_auc_{tag}.csv"
                if not sweep_path.exists() or not count_path.exists():
                    print(f"  [SKIP] Missing CSVs for {tag}, run without plot_only first.")
                    continue
                count_df = pd.read_csv(count_path)
                sweep_df = pd.read_csv(sweep_path)
                orig_auc = pd.read_csv(orig_path).iloc[0].to_dict() if orig_path.exists() else {}
            else:
                count_df = count_kept_microbes_sweep(phenotypes, dtype, level=level)
                count_df.to_csv(output_dir / f"microbe_counts_{tag}.csv", index=False)

                sweep_df = run_stability_sweep(phenotypes, dtype, level=level)
                sweep_df.to_csv(output_dir / f"auc_sweep_{tag}.csv", index=False)

                print(f"  Computing original (no normalization) baseline ...")
                orig_auc = get_original_mean_auc(phenotypes, dtype, level=level)
                pd.DataFrame([orig_auc]).to_csv(
                    output_dir / f"original_auc_{tag}.csv", index=False
                )

            _plot_sweep_panel(ax, sweep_df, count_df,
                              title=f"{dtype} — {level_label}",
                              original_auc=orig_auc)

    fig.suptitle(
        "Stability Filter Percentile: Mean AUC & Kept Microbes",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / "stability_threshold_investigation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Microbe characterization at best thresholds
# ═══════════════════════════════════════════════════════════════════════════════

_BEST_THRESHOLDS = {
    "Metagenomics": 0.25,
    "Amplicon":     0.40,
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
        y = target.values if hasattr(target, "values") else np.array(target)
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
            "sparsity":              float((col == 0).mean()),
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

    _box(axes[0, 0], "sparsity",            "Sparsity (fraction zeros)",       "Fraction zeros")
    _box(axes[0, 1], "cv_across_datasets",  "CV across datasets (stability)",  "CV (std/mean)")
    _box(axes[0, 2], "mean_MI_with_target", "Mean MI with target",             "MI")
    _box(axes[1, 0], "mean_nonzero_abund",  "Mean non-zero abundance",         "Abundance")
    _box(axes[1, 1], "prevalence_datasets", "Prevalence (fraction datasets)",  "Fraction datasets")

    # Taxonomy level distribution — unique taxon names per level
    ax = axes[1, 2]
    levels = ["species", "genus", "family", "order", "class", "phylum", "unknown"]
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
    "sparsity",
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

    df = stats_df[_CONTINUOUS_METRICS + ["status"]].dropna()
    X  = df[_CONTINUOUS_METRICS].values
    y  = (df["status"] == "kept").astype(int).values

    if len(np.unique(y)) < 2 or len(y) < 10:
        return pd.DataFrame({"metric": _CONTINUOUS_METRICS})

    # Logistic regression on standardised features
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    lr     = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_sc, y)
    lr_coef = np.abs(lr.coef_[0])

    # Random forest
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

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
        ax.set_title("Feature importance for kept/dropped prediction", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

    # ── Right: significance table ────────────────────────────────────────────
    ax = axes[1]
    ax.axis("off")

    _METRIC_LABELS = {
        "sparsity":            "Sparsity\n(frac. zeros)",
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
):
    """
    Step 2: characterize kept vs dropped microbes at the best stability threshold.

    Parameters
    ----------
    phenotypes             : list of (phenotype, dtype) tuples
    output_dir             : directory to write outputs
    threshold_metagenomics : stability percentile for Metagenomics (default 0.25)
    threshold_amplicon     : stability percentile for Amplicon (default 0.40)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "Metagenomics": threshold_metagenomics or _BEST_THRESHOLDS["Metagenomics"],
        "Amplicon":     threshold_amplicon     or _BEST_THRESHOLDS["Amplicon"],
    }

    for dtype, threshold in thresholds.items():
        print(f"\n=== Characterization: {dtype}  (threshold={threshold}) ===")

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
        stats_df.to_csv(output_dir / f"microbe_stats_{dtype.lower()}.csv", index=False)

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
