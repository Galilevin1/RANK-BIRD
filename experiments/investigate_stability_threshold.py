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
