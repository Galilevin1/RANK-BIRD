"""
Figure 3: Microbiome abundance heatmap per phenotype.

Shows all samples (rows) × top-varying common features (columns), z-scored
for visualization, with datasets stacked and separated by horizontal lines.
Generated for both raw and normalized data.

Usage
-----
    from figures.figure3 import plot_figure3, run_figure3

    # generate for one phenotype
    fig = plot_figure3(
        microbiome_dfs, dataset_names, phenotype_name,
        normalization_approach="rankbird_wasserstein"
    )
    fig.savefig(...)

    # or use the pipeline runner (calls plot_figure3 for every phenotype)
    run_figure3(phenotypes, data_root, figures_dir, normalization_approach="rankbird_wasserstein")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

from utils.microbe_names import rename_microbes


# ─────────────────────────────────────────────────────────────
# KS significance helper
# ─────────────────────────────────────────────────────────────

def _compute_ks_significance(
    microbiome_dfs: List[pd.DataFrame],
    common_features,
    dataset_names: List[str],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    For each microbial feature perform pairwise KS tests across all dataset
    pairs, apply Benjamini–Hochberg FDR correction, and return:
        ks_df              — full per-feature × per-pair results
        sig_raw_microbes   — features significant before FDR
        sig_fdr_microbes   — features significant after FDR
    """
    rows = []
    for feature in common_features:
        for i, j in combinations(range(len(microbiome_dfs)), 2):
            stat, p = ks_2samp(
                microbiome_dfs[i][feature].values,
                microbiome_dfs[j][feature].values,
            )
            rows.append({
                "microbe":   feature,
                "dataset1":  dataset_names[i],
                "dataset2":  dataset_names[j],
                "ks_stat":   stat,
                "p_value":   p,
                "significant_raw": p < alpha,
            })

    ks_df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(ks_df["p_value"], method='fdr_bh')
    ks_df["p_fdr"]         = p_fdr
    ks_df["significant_fdr"] = reject

    sig_raw = (ks_df.groupby("microbe")["significant_raw"]
               .max().pipe(lambda s: s[s].index.tolist()))
    sig_fdr = (ks_df.groupby("microbe")["significant_fdr"]
               .max().pipe(lambda s: s[s].index.tolist()))

    return ks_df, sig_raw, sig_fdr


# ─────────────────────────────────────────────────────────────
# Single heatmap
# ─────────────────────────────────────────────────────────────

def _make_heatmap(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    max_features: int = 50,
    label_suffix: str = "",
) -> plt.Figure:
    """
    Build one heatmap figure from a list of (possibly already normalized)
    microbiome DataFrames.

    Steps
    -----
    1. Intersect columns → common features
    2. KS test → report significance
    3. Select top-`max_features` by variance
    4. Z-score the combined matrix (for display only)
    5. Rename features
    6. Plot with dataset separator lines
    """
    # 1. Common features
    common = microbiome_dfs[0].columns
    for df in microbiome_dfs[1:]:
        common = common.intersection(df.columns)

    if len(common) == 0:
        raise ValueError(f"No common features for {phenotype_name}")

    # 2. KS significance
    _, _, sig_fdr = _compute_ks_significance(microbiome_dfs, common, dataset_names)
    pct_sig = 100 * len(sig_fdr) / len(common)
    print(f"  KS (FDR): {len(sig_fdr)}/{len(common)} significant microbes "
          f"({pct_sig:.1f}%)")

    # 3. Top features by variance
    if len(common) > max_features:
        all_data = pd.concat([df[common] for df in microbiome_dfs], axis=0)
        common = all_data.var().sort_values(ascending=False).head(max_features).index

    # 4. Z-score combined matrix
    aligned = [df[common] for df in microbiome_dfs]
    X = pd.concat(aligned, axis=0)
    X_z = (X - X.mean()) / (X.std() + 1e-10)
    X_z = X_z.fillna(0)

    # Dataset boundary positions
    boundaries = [0]
    for df in aligned:
        boundaries.append(boundaries[-1] + len(df))

    # 5. Rename features
    X_z.columns = rename_microbes(X_z.columns.tolist())

    # 6. Plot
    n_samples  = len(X_z)
    n_features = len(X_z.columns)
    fig_w = max(14, n_features * 0.28)
    fig_h = max(10, n_samples  * 0.025)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        X_z,
        cmap='RdBu_r', center=0, robust=True,
        vmin=-3, vmax=3,
        yticklabels=False, xticklabels=False,
        cbar=False,
        ax=ax,
    )

    # Separator lines between datasets
    for b in boundaries[1:-1]:
        ax.axhline(y=b, color='black', linewidth=3)

    # Dataset labels on the left
    for i, name in enumerate(dataset_names):
        y_mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(-0.01, y_mid, name,
                ha='right', va='center', fontsize=16, fontweight='bold',
                transform=ax.get_yaxis_transform())

    ax.text(0.5, -0.02,
            f'KS significant microbes (FDR): {len(sig_fdr)}/{len(common)} '
            f'({pct_sig:.1f}%)',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=11, color='dimgrey')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

def plot_figure3(
    microbiome_dfs: List[pd.DataFrame],
    dataset_names: List[str],
    phenotype_name: str,
    normalization_approach=None,
    max_features: int = 50,
) -> plt.Figure:
    """
    Generate one figure 3 heatmap for a phenotype.

    If normalization_approach is not None, the normalization pipeline is applied
    first and the figure is labelled accordingly. Otherwise raw data is used.
    """
    if normalization_approach is not None:
        from src.rankbird.normalization.pipeline import apply_normalization_pipeline
        print(f"  Applying normalization pipeline for {phenotype_name}")
        microbiome_dfs, dataset_names = apply_normalization_pipeline(
            list(microbiome_dfs), list(dataset_names)
        )
        label_suffix = f" ({normalization_approach})"
    else:
        label_suffix = " (raw)"

    print(f"  Building heatmap for {phenotype_name}{label_suffix}")
    return _make_heatmap(microbiome_dfs, dataset_names, phenotype_name,
                         max_features=max_features, label_suffix=label_suffix)


def run_figure3(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    normalization_approach=None,
    max_features: int = 50,
):
    """
    Run figure 3 for all phenotypes and save results to figures_dir.

    Parameters
    ----------
    phenotypes : list of (phenotype, dtype) tuples
    data_root : str
        Root folder containing phenotype subfolders.
    figures_dir : str
        Where to save the PNG files.
    normalization_approach : str or None
        Normalization approach to apply, or None for raw data.
    max_features : int
        Maximum number of features shown per heatmap.
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out = Path(figures_dir)
    data = Path(data_root)

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder = data / pheno_str

        if not folder.exists():
            print(f"  Skipping {pheno_str}: folder not found")
            continue

        print(f"\n{'='*50}\nFigure 3: {pheno_str}\n{'='*50}")
        try:
            microbiome_dfs, _, dataset_names = load_microbiome_datasets_with_targets(str(folder))
        except Exception as e:
            print(f"  Error loading {pheno_str}: {e}")
            continue

        if len(dataset_names) < 2:
            print(f"  Skipping {pheno_str}: need ≥2 datasets")
            continue

        try:
            fig = plot_figure3(
                microbiome_dfs, dataset_names, pheno_str,
                normalization_approach=normalization_approach,
                max_features=max_features,
            )
            safe   = pheno_str.replace(" ", "_")
            mode   = normalization_approach or "raw"
            fig.savefig(out / f"figure3_{safe}_{mode}.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved figure 3 for {pheno_str}")

        except Exception as e:
            print(f"  Error generating figure 3 for {pheno_str}: {e}")


# ─────────────────────────────────────────────────────────────
# Figure 3C: distribution approach — All datasets, original vs RANK-BIRD only
# ─────────────────────────────────────────────────────────────

def plot_figure3c(investigation_dir: str) -> plt.Figure:
    """
    Load existing distribution-investigation CSVs and produce three panels
    (Metagenomics, Amplicon, All datasets) comparing only Original vs RANK-BIRD.
    """
    from experiments.investigate_distribution_approach import (
        _make_combined_df, _draw_panel, APPROACH_LABELS,
    )

    inv_dir = Path(investigation_dir)
    dtypes  = ["Metagenomics", "Amplicon"]
    keep    = ["original", "rankbird"]
    keep_labels = {APPROACH_LABELS[a] for a in keep} | {"Random (0.5)"}

    def _fix_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep_labels]
        if filtered:
            h, l = zip(*filtered)
            ax.legend(handles=h, labels=l, fontsize=9, framealpha=0.9, loc="lower right")

    all_results: dict = {}
    for dtype in dtypes:
        for approach in keep:
            csv_path = inv_dir / f"results_{dtype.lower()}_{approach}.csv"
            if csv_path.exists():
                all_results[(dtype, approach)] = pd.read_csv(csv_path)
            else:
                print(f"  [SKIP] Missing {csv_path.name}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    for ax, dtype in zip(axes[:2], dtypes):
        frames = [
            all_results[(dtype, a)].assign(approach=a)
            for a in keep
            if (dtype, a) in all_results and not all_results[(dtype, a)].empty
        ]
        if frames:
            _draw_panel(ax, pd.concat(frames, ignore_index=True), title=dtype)
            _fix_legend(ax)
        else:
            ax.set_title(f"{dtype} — no data", fontsize=11)

    combined_df = _make_combined_df(all_results, dtypes)
    if not combined_df.empty:
        combined_df = combined_df[combined_df["approach"].isin(keep)]
        _draw_panel(axes[2], combined_df, title="All datasets")
        _fix_legend(axes[2])
    else:
        axes[2].set_title("All datasets — no data", fontsize=11)

    plt.tight_layout()
    return fig


def run_figure3c(investigation_dir: str, figures_dir: str) -> None:
    """Generate and save Figure 3C."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("\nBuilding Figure 3C: Original vs RANK-BIRD (All datasets)")
    fig = plot_figure3c(investigation_dir)
    fig.savefig(out / "figure3c_original_vs_rankbird.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 3C")


# ─────────────────────────────────────────────────────────────
# Figure 3D: PCA scatter per phenotype — target vs dtype/dataset separation
# ─────────────────────────────────────────────────────────────

_DTYPE_MARKERS = {"Metagenomics": "o", "Amplicon": "^"}
_TARGET_COLORS = {0: "#4878D0", 1: "#EE854A"}   # blue=control, orange=case
_TARGET_LABELS = {0: "Control", 1: "Case"}


def _pca_scatter(
    ax: plt.Axes,
    X: pd.DataFrame,
    targets: np.ndarray,
    dtypes: np.ndarray,
    title: str,
) -> None:
    """Run PCA on X and draw scatter on ax. color=target, shape=dtype."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X.fillna(0))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    unique_dtypes = sorted(set(dtypes))
    unique_targets = sorted(set(targets))

    for dtype in unique_dtypes:
        marker = _DTYPE_MARKERS.get(dtype, "s")
        mask_d = dtypes == dtype
        for tgt in unique_targets:
            mask = mask_d & (targets == tgt)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=_TARGET_COLORS.get(int(tgt), "#888888"),
                marker=marker,
                s=18, alpha=0.65, linewidths=0,
            )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)


def plot_figure3d(
    phenotype_name: str,
    microbiome_dfs_by_dtype: Dict[str, List[pd.DataFrame]],
    target_dfs_by_dtype: Dict[str, List[pd.DataFrame]],
    dataset_names_by_dtype: Dict[str, List[str]],
    normalization_approach: Optional[str] = "rankbird_wasserstein",
) -> plt.Figure:
    """
    Generate figure 3d for one phenotype: two PCA scatter panels (raw | normalized).

    Parameters
    ----------
    phenotype_name : str
    microbiome_dfs_by_dtype : {'Metagenomics': [...], 'Amplicon': [...]}
    target_dfs_by_dtype     : matching target DataFrames
    dataset_names_by_dtype  : matching dataset name lists
    normalization_approach  : passed to apply_normalization_pipeline; None skips norm panel
    """
    dtypes_present = [d for d in ["Metagenomics", "Amplicon"] if d in microbiome_dfs_by_dtype]

    # ── flatten all datasets in a consistent order ──────────────────────────
    all_mb, all_tgt, all_names, all_dtypes_label = [], [], [], []
    for dtype in dtypes_present:
        for mb, tgt, name in zip(
            microbiome_dfs_by_dtype[dtype],
            target_dfs_by_dtype[dtype],
            dataset_names_by_dtype[dtype],
        ):
            all_mb.append(mb)
            all_tgt.append(tgt)
            all_names.append(name)
            all_dtypes_label.extend([dtype] * len(mb))

    # ── common features (raw) ───────────────────────────────────────────────
    common_raw = all_mb[0].columns
    for df in all_mb[1:]:
        common_raw = common_raw.intersection(df.columns)

    n_cols = 2 if normalization_approach is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # ── helper: build (X, targets, dtype_arr) arrays ────────────────────────
    def _build_arrays(mb_list, tgt_list, dtype_labels, common_cols):
        X = pd.concat([df[common_cols] for df in mb_list], axis=0, ignore_index=True)
        tgt_arr = np.concatenate([
            tgt.iloc[:, 0].values if isinstance(tgt, pd.DataFrame) else tgt.values
            for tgt in tgt_list
        ])
        dtype_arr = np.array(dtype_labels)
        return X, tgt_arr, dtype_arr

    # ── Panel 1: raw ─────────────────────────────────────────────────────────
    if len(common_raw) == 0:
        axes[0].text(0.5, 0.5, "No common features (raw)", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("Raw", fontsize=10, fontweight="bold")
    else:
        print(f"  [3d] {phenotype_name}: {len(common_raw)} common raw features")
        X_raw, tgt_raw, dt_raw = _build_arrays(all_mb, all_tgt, all_dtypes_label, common_raw)
        _pca_scatter(axes[0], X_raw, tgt_raw, dt_raw, title="Raw")

    # ── Panel 2: normalized ───────────────────────────────────────────────────
    if normalization_approach is not None:
        try:
            from src.rankbird.normalization.pipeline import apply_normalization_pipeline
            norm_mb, norm_names = apply_normalization_pipeline(
                list(all_mb), list(all_names)
            )
            # rebuild dtype labels in the same order as returned names
            name2dtype = {
                name: dtype
                for dtype in dtypes_present
                for name in dataset_names_by_dtype[dtype]
            }
            norm_dtypes = []
            norm_tgt_list = []
            name2tgt = {
                name: tgt
                for dtype in dtypes_present
                for name, tgt in zip(dataset_names_by_dtype[dtype], target_dfs_by_dtype[dtype])
            }
            for mb_norm, name_norm in zip(norm_mb, norm_names):
                dtype_for_name = name2dtype.get(name_norm, "Unknown")
                norm_dtypes.extend([dtype_for_name] * len(mb_norm))
                norm_tgt_list.append(name2tgt[name_norm])

            common_norm = norm_mb[0].columns
            for df in norm_mb[1:]:
                common_norm = common_norm.intersection(df.columns)

            if len(common_norm) == 0:
                axes[1].text(0.5, 0.5, "No common features (normalized)",
                             ha="center", va="center", transform=axes[1].transAxes)
                axes[1].set_title("RANK-BIRD", fontsize=10, fontweight="bold")
            else:
                print(f"  [3d] {phenotype_name}: {len(common_norm)} common normalized features")
                X_norm, tgt_norm, dt_norm = _build_arrays(
                    norm_mb, norm_tgt_list, norm_dtypes, common_norm
                )
                _pca_scatter(axes[1], X_norm, tgt_norm, dt_norm, title="RANK-BIRD")

        except Exception as e:
            print(f"  [3d] Normalization failed for {phenotype_name}: {e}")
            axes[1].text(0.5, 0.5, f"Normalization error:\n{e}",
                         ha="center", va="center", transform=axes[1].transAxes, fontsize=8)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = []
    for tgt_val in [0, 1]:
        legend_handles.append(mpatches.Patch(
            color=_TARGET_COLORS[tgt_val], label=_TARGET_LABELS[tgt_val]
        ))
    for dtype in dtypes_present:
        marker = _DTYPE_MARKERS.get(dtype, "s")
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="grey",
                       linestyle="None", markersize=7, label=dtype)
        )
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(legend_handles), fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(phenotype_name, fontsize=12, fontweight="bold", y=1.06)
    plt.tight_layout()
    return fig


def run_figure3d(
    phenotypes: List[Tuple[str, str]],
    data_root: str,
    figures_dir: str,
    normalization_approach: Optional[str] = "rankbird_wasserstein",
) -> None:
    """
    Generate and save figure 3d for every phenotype.

    Phenotypes that exist in multiple dtypes (Metagenomics + Amplicon) show
    both dtypes combined.  Phenotypes with only one dtype are included too.

    Parameters
    ----------
    phenotypes : list of (phenotype, dtype) tuples
    data_root  : root folder containing "<phenotype> <dtype>" subfolders
    figures_dir: output directory
    normalization_approach : passed to plot_figure3d
    """
    from evaluation.data_loading import load_microbiome_datasets_with_targets

    out  = Path(figures_dir)
    data = Path(data_root)

    # ── group (phenotype, dtype) tuples by base phenotype name ───────────────
    pheno_map: Dict[str, List[str]] = {}
    for phenotype, dtype in phenotypes:
        pheno_map.setdefault(phenotype, []).append(dtype)

    for phenotype, dtypes_for_pheno in pheno_map.items():
        print(f"\n{'='*50}\nFigure 3d: {phenotype}\n{'='*50}")

        mb_by_dtype:   Dict[str, List[pd.DataFrame]] = {}
        tgt_by_dtype:  Dict[str, List[pd.DataFrame]] = {}
        name_by_dtype: Dict[str, List[str]]           = {}

        for dtype in dtypes_for_pheno:
            pheno_str = f"{phenotype} {dtype}"
            folder    = data / pheno_str
            if not folder.exists():
                print(f"  Skipping {pheno_str}: folder not found")
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
            except Exception as e:
                print(f"  Error loading {pheno_str}: {e}")
                continue
            if not mb_dfs:
                continue
            mb_by_dtype[dtype]   = mb_dfs
            tgt_by_dtype[dtype]  = tgt_dfs
            name_by_dtype[dtype] = ds_names

        if not mb_by_dtype:
            print(f"  Skipping {phenotype}: no data loaded")
            continue

        try:
            fig = plot_figure3d(
                phenotype_name=phenotype,
                microbiome_dfs_by_dtype=mb_by_dtype,
                target_dfs_by_dtype=tgt_by_dtype,
                dataset_names_by_dtype=name_by_dtype,
                normalization_approach=normalization_approach,
            )
            safe = phenotype.replace(" ", "_")
            fig.savefig(out / f"figure3d_{safe}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved figure 3d for {phenotype}")
        except Exception as e:
            print(f"  Error generating figure 3d for {phenotype}: {e}")
