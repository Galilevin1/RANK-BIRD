"""
Shared ranking-based normalization functions for RANK-BIRD pipeline variants.

Normalization approaches
------------------------
rank_normalize      — column-wise rank mapped to [0, 1]
sigmoid_normalize   — rank + sigmoid transform (soft present/absent values)
relu_normalize      — rank-based hard threshold (bottom fraction zeroed)

Pipeline
--------
apply_ranking_pipeline — full ranking variant pipeline:
    taxonomy filter → stability filter → oversample + combine →
    per-dataset normalization → extract original samples
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from .stability import union_microbes, nonzero_percent_by_dataset, auto_stability_filter
from .combine import oversample_to_min_size
from .taxonomy_filter import filter_to_level


# ── Default hyper-parameters ──────────────────────────────────────────────────

SIGMOID_K      = 20.0   # steepness; higher → sharper present/absent boundary
SIGMOID_CENTER = 0.5    # rank fraction at which σ = 0.5
RELU_THRESHOLD = 0.5    # fraction to keep (0.5 → top 50 % kept)
CLR_PSEUDOCOUNT = 1e-6  # added before log to handle exact zeros


# ── Column-wise normalization functions ───────────────────────────────────────

def rank_normalize(X: pd.DataFrame, tie_method: str = "first") -> pd.DataFrame:
    """
    For each microbe column rank all rows descending and map to [0, 1]:
        highest value → 1.0,  lowest value → 0.0.

    tie_method : "first"   — ties broken by position (original behaviour)
                 "average" — tied values receive the average of their ranks,
                             so identical values (e.g. zeros) always map to
                             the same output regardless of row order.
    """
    result = X.copy().astype(float)
    N = len(X)
    if N <= 1:
        return result
    for col in X.columns:
        x = X[col].values
        if tie_method == "average":
            r = rankdata(-x, method="average")      # 1-based descending
            result[col] = 1.0 - (r - 1) / (N - 1)
        else:
            sorted_idx = np.argsort(-x)
            ranks = np.empty(N, dtype=float)
            ranks[sorted_idx] = np.arange(N, dtype=float)
            result[col] = 1.0 - ranks / (N - 1)
    return result


def sigmoid_normalize(
    X: pd.DataFrame,
    k: float = SIGMOID_K,
    center: float = SIGMOID_CENTER,
    tie_method: str = "first",
) -> pd.DataFrame:
    """
    Column ranking (→ [0, 1]) followed by sigmoid:
        value = σ(k · (r − center))
    Produces soft present/absent-like values.
    """
    ranked = rank_normalize(X, tie_method=tie_method)
    result = ranked.copy()
    for col in ranked.columns:
        r = ranked[col].values
        result[col] = expit(k * (r - center))
    return result


def relu_normalize(
    X: pd.DataFrame,
    threshold: float = RELU_THRESHOLD,
    tie_method: str = "first",
) -> pd.DataFrame:
    """
    Rank-based hard threshold: rank each column to [0, 1] to decide which
    samples to keep, but return the original abundance value (not the rank):
      - samples with rank >= threshold → keep original abundance
      - samples with rank < threshold  → set to 0
    """
    ranked = rank_normalize(X, tie_method=tie_method)
    result = X.copy().astype(float)
    for col in X.columns:
        r = ranked[col].values
        result[col] = np.where(r >= threshold, X[col].values, 0.0)
    return result


def clr_normalize(
    X: pd.DataFrame,
    pseudocount: float = CLR_PSEUDOCOUNT,
    tie_method: str = "first",   # accepted for API compatibility, unused
) -> pd.DataFrame:
    """
    Centered Log-Ratio normalization for compositional microbiome data.

    CLR(x_i) = log(x_i + ε) − mean_j[log(x_j + ε)]

    Each sample is normalized by its own log geometric mean, capturing
    within-sample taxon ratios rather than absolute abundances.
    tie_method is accepted for API compatibility with other norm functions.
    """
    vals = X.values.astype(float) + pseudocount
    log_vals = np.log(vals)
    geomean_log = log_vals.mean(axis=1, keepdims=True)
    return pd.DataFrame(log_vals - geomean_log, index=X.index, columns=X.columns)


def clr_rank_normalize(
    X: pd.DataFrame,
    pseudocount: float = CLR_PSEUDOCOUNT,
    tie_method: str = "first",
) -> pd.DataFrame:
    """
    CLR followed by rank normalization.

    CLR removes the compositional constraint and captures log-ratio biology;
    rank normalization then maps each feature to [0, 1] so distributions are
    comparable across datasets.
    """
    return rank_normalize(clr_normalize(X, pseudocount=pseudocount), tie_method=tie_method)


def compute_alpha_diversity_features(
    microbiome_dfs: list,
    dataset_names: list,
) -> dict:
    """
    Compute per-sample alpha diversity from raw abundance data.

    Returns
    -------
    dict : {dataset_name: DataFrame with columns __shannon__, __simpson__, __richness__}
           Indexed identically to the input DataFrames.
    """
    result = {}
    for df, name in zip(microbiome_dfs, dataset_names):
        vals = df.values.astype(float)
        row_sums = vals.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        props = vals / row_sums

        log_props = np.where(props > 0, np.log(props), 0.0)
        shannon  = -(props * log_props).sum(axis=1)
        simpson  = 1.0 - (props ** 2).sum(axis=1)
        richness = (vals > 0).sum(axis=1).astype(float)

        result[name] = pd.DataFrame(
            {'__shannon__': shannon, '__simpson__': simpson, '__richness__': richness},
            index=df.index,
        )
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_oversampled_combined(
    microbiome_dfs: list,
    dataset_names: list,
    kept_microbes: list,
    min_size: int = 550,
    random_state: int = 42,
) -> tuple:
    """
    Filter to kept_microbes, oversample each dataset to min_size, concatenate.

    Returns
    -------
    X_combined    : pd.DataFrame  (sum_oversampled × n_microbes)
    n_orig_per_ds : list[int]     original sample count per dataset
    n_os_per_ds   : list[int]     oversampled sample count per dataset
    """
    frames, n_orig_per_ds, n_os_per_ds = [], [], []
    for df in microbiome_dfs:
        Xi = df.reindex(columns=kept_microbes, fill_value=0.0).copy()
        n_orig_per_ds.append(len(Xi))
        Xi_os = oversample_to_min_size(Xi, min_size=min_size, random_state=random_state)
        n_os_per_ds.append(len(Xi_os))
        frames.append(Xi_os)
    X_combined = pd.concat(frames, axis=0, ignore_index=True)
    return X_combined, n_orig_per_ds, n_os_per_ds


def _extract_original_samples(
    X_norm: pd.DataFrame,
    original_dfs: list,
    n_os_per_ds: list,
) -> list:
    """
    Extract only the original (non-duplicated) rows from the normalized combined
    matrix and restore original DataFrame indices.
    """
    normalized_dfs = []
    offset = 0
    for df_orig, n_os in zip(original_dfs, n_os_per_ds):
        n_orig = len(df_orig)
        df_norm = X_norm.iloc[offset : offset + n_orig].copy()
        df_norm.index = df_orig.index
        normalized_dfs.append(df_norm)
        offset += n_os
    return normalized_dfs


# ── Full ranking pipeline ─────────────────────────────────────────────────────

def apply_ranking_pipeline(
    microbiome_dfs: list,
    dataset_names: list,
    stability_percentile: float,
    norm_fn,
    min_size: int = 550,
    taxonomy_level=None,
    random_state: int = 42,
) -> tuple:
    """
    Full ranking-based RANK-BIRD pipeline:
        taxonomy filter → stability filter → oversample + combine →
        per-dataset normalization → extract original samples.

    Parameters
    ----------
    norm_fn        : callable (pd.DataFrame) → pd.DataFrame
                     One of rank_normalize, sigmoid_normalize, relu_normalize.
    taxonomy_level : None = all, "g" = genus only, "s" = genus+species, etc.

    Returns
    -------
    (normalized_dfs, dataset_names)
    """
    # Step 0 — taxonomy filter
    microbiome_dfs = filter_to_level(microbiome_dfs, taxonomy_level)

    # Step 1 — stability filter
    all_microbes  = union_microbes(microbiome_dfs)
    nz_df         = nonzero_percent_by_dataset(microbiome_dfs, dataset_names, all_microbes)
    kept_microbes = auto_stability_filter(nz_df, percentile=stability_percentile)

    # Step 2 — oversample each dataset + combine
    X_combined, _n_orig, n_os_per_ds = _build_oversampled_combined(
        microbiome_dfs, dataset_names, kept_microbes,
        min_size=min_size, random_state=random_state,
    )

    # Step 3 — per-dataset normalization
    blocks, offset = [], 0
    for n_os in n_os_per_ds:
        block = X_combined.iloc[offset : offset + n_os].copy()
        blocks.append(norm_fn(block))
        offset += n_os
    X_norm = pd.concat(blocks, axis=0, ignore_index=True)

    # Step 4 — extract original samples only
    normalized_dfs = _extract_original_samples(X_norm, microbiome_dfs, n_os_per_ds)

    return normalized_dfs, dataset_names
