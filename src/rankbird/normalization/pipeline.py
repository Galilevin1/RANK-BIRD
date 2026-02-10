import numpy as np
from .sanitize import sanitize_feature_names
from .stability import union_microbes, nonzero_percent_by_dataset, auto_stability_filter, filter_microbes_by_dataset_support
from .combine import build_combined_matrix
from .global_distributions import compute_per_microbe_global_distributions
from .rank_map import rank_map_normalize_dataset



# ---------------------------------------------------------------
# 5) MAIN PIPELINE
# ---------------------------------------------------------------

def apply_normalization_pipeline(
    microbiome_dfs,
    dataset_names,
    global_analysis: bool = False,
    min_samples_per_dataset=550, #550
    stability_percentile_local=0.3,
    stability_percentile_global=0.5,
    min_dataset_support=5,
    z_thresh=3.0,
    random_state=42
):
    # ----------------------------
    # STEP 1 — stability filtering
    # ----------------------------
    if global_analysis:
        stability_percentile = stability_percentile_global
    else:
        stability_percentile = stability_percentile_local
    all_microbes = union_microbes(microbiome_dfs)
    nz_df = nonzero_percent_by_dataset(microbiome_dfs, dataset_names, all_microbes)
    kept_microbes = auto_stability_filter(nz_df, percentile=stability_percentile)
    kept_safe, orig2safe, safe2orig = sanitize_feature_names(kept_microbes)

    # ----------------------------
    # STEP 2 — oversample & combine
    # ----------------------------
    X_all, ds_all, orig_indices_map = build_combined_matrix(
        microbiome_dfs, dataset_names,
        kept_microbes, kept_safe, orig2safe,
        min_size=min_samples_per_dataset,
        random_state=random_state
    )

    # ----------------------------
    # STEP 3-4 — per-microbe outlier detection and refined mean
    # ----------------------------
    global_sorted_dict, per_microbe_kept = compute_per_microbe_global_distributions(
        X_all, ds_all,
        dataset_names,
        kept_safe,
        z_thresh=z_thresh
    )

    # ----------------------------
    # STEP 5 — normalize ORIGINAL data
    # ----------------------------
    normalized_microbiome_datasets = []
    normalized_dataset_names = []

    for d in dataset_names:
        # restore original samples
        df_orig = microbiome_dfs[dataset_names.index(d)].copy()
        df_orig = df_orig.reindex(columns=kept_microbes, fill_value=0.0)

        df_orig.columns = [orig2safe[c] for c in df_orig.columns]
        X_orig = df_orig[kept_safe].copy()

        X_norm = rank_map_normalize_dataset(X_orig, global_sorted_dict)

        # restore taxa names
        X_norm.columns = [safe2orig[c] for c in X_norm.columns]

        normalized_microbiome_datasets.append(X_norm)
        normalized_dataset_names.append(d)

    return normalized_microbiome_datasets, normalized_dataset_names
    
    #, {"kept_microbes": kept_microbes, "n_kept": len(kept_microbes), 
    #"global_analysis": global_analysis}