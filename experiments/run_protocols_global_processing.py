import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.rankbird.normalization.pipeline import apply_normalization_pipeline
from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.taxonomy_filter import filter_to_level
from src.rankbird.normalization.ranking import (
    rank_normalize, sigmoid_normalize, relu_normalize,
    apply_ranking_pipeline,
    SIGMOID_K, SIGMOID_CENTER, RELU_THRESHOLD,
)
from src.rankbird.representation.multi_datasets_SPDR import apply_bias_SPDR
from evaluation.data_loading import load_microbiome_datasets_with_targets
from evaluation.learning_protocols import lodo_protocol, internal_validation_protocol, within_dataset_protocol

def _run_protocols_on_group(microbiome_dfs, target_dfs, dataset_names, phenotype_str):
    records = []

    results = {
        "LODO": lodo_protocol(microbiome_dfs, target_dfs, dataset_names),
        "Internal Validation": internal_validation_protocol(microbiome_dfs, target_dfs, dataset_names),
        "Within Learning": within_dataset_protocol(microbiome_dfs, target_dfs, dataset_names),
    }

    for protocol, df in results.items():
        for _, row in df.iterrows():
            records.append({
                "phenotype": phenotype_str,
                "dataset": row["test_dataset"],
                "protocol": protocol,
                "auc": row["auc"],
            })

    return pd.DataFrame(records)


_RANKING_NORM_FNS = {
    "rankbird_ranking":  rank_normalize,
    "rankbird_sigmoid":  sigmoid_normalize,
    "rankbird_relu":     relu_normalize,
}


def _run_global_for_dtype(
    phenotypes,
    normalization_approach=None,
    apply_decompose=False,
    min_samples_per_dataset=550,
    stability_percentile_local=0.3,
    stability_percentile_global=0.5,
    z_thresh=3.0,
    decompose_method='PCA',
    decompose_rank=30,
    taxonomy_level=None,
):
    """
    normalization_approach options
    --------------------------------
    None                  — no normalization, taxonomy filter only
    "filter_only"         — stability filter only, no distribution normalization
    "rankbird_wasserstein"— full RANK-BIRD pipeline (Wasserstein + quantile mapping)
    "rankbird_ranking"    — RANK-BIRD with pure rank normalization
    "rankbird_sigmoid"    — RANK-BIRD with sigmoid normalization
    "rankbird_relu"       — RANK-BIRD with relu normalization
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "Data"

    # ----------------------------------
    # 1. Load ALL data across phenotypes
    # ----------------------------------
    all_microbiome = []
    all_targets = []
    all_dataset_names = []
    dataset_to_phenotype = {}

    for phenotype, dtype in phenotypes:
        phenotype_str = f"{phenotype} {dtype}"
        folder = DATA_ROOT / phenotype_str

        microbiome_dfs, target_dfs, dataset_names = \
            load_microbiome_datasets_with_targets(folder)

        for df, y, name in zip(microbiome_dfs, target_dfs, dataset_names):
            all_microbiome.append(df)
            all_targets.append(y)
            all_dataset_names.append(name)
            dataset_to_phenotype[name] = phenotype_str

    # ----------------------------------
    # 2. GLOBAL preprocessing (once)
    # ----------------------------------

    if normalization_approach == "filter_only":
        # Taxonomy filter → stability filter only (no distribution normalization)
        all_microbiome = filter_to_level(all_microbiome, taxonomy_level)
        all_microbes = union_microbes(all_microbiome)
        nz_df = nonzero_percent_by_dataset(all_microbiome, all_dataset_names, all_microbes)
        kept = auto_stability_filter(nz_df, percentile=stability_percentile_global)
        all_microbiome = [df.reindex(columns=kept, fill_value=0.0) for df in all_microbiome]

    elif normalization_approach == "rankbird_wasserstein":
        all_microbiome, all_dataset_names = apply_normalization_pipeline(
            all_microbiome,
            all_dataset_names,
            global_analysis=True,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_global,
            z_thresh=z_thresh,
            taxonomy_level=taxonomy_level,
        )

    elif normalization_approach in _RANKING_NORM_FNS:
        norm_fn = _RANKING_NORM_FNS[normalization_approach]
        all_microbiome, all_dataset_names = apply_ranking_pipeline(
            all_microbiome,
            all_dataset_names,
            stability_percentile=stability_percentile_global,
            norm_fn=norm_fn,
            min_size=min_samples_per_dataset,
            taxonomy_level=taxonomy_level,
        )

    else:
        # None — no normalization, apply taxonomy filter only
        all_microbiome = filter_to_level(all_microbiome, taxonomy_level)

    if apply_decompose:
        all_microbiome, _, _ = apply_bias_SPDR(all_microbiome, decompose_method, rank=decompose_rank)

    # ----------------------------------
    # 3. Split BACK by phenotype
    # ----------------------------------
    records = []

    for phenotype_str in set(dataset_to_phenotype.values()):
        idx = [
            i for i, name in enumerate(all_dataset_names)
            if dataset_to_phenotype[name] == phenotype_str
        ]

        microbiome_grp = [all_microbiome[i] for i in idx]
        target_grp = [all_targets[i] for i in idx]
        names_grp = [all_dataset_names[i] for i in idx]

        df_grp = _run_protocols_on_group(
            microbiome_grp,
            target_grp,
            names_grp,
            phenotype_str
        )

        records.append(df_grp)

    return pd.concat(records, ignore_index=True)


def run_protocol_benchmark_global_preprocessing(
    phenotypes: list,
    normalization_approach=None,
    apply_decompose: bool = False,
    min_samples_per_dataset=550,
    stability_percentile_local=0.3,
    stability_percentile_global_amplicon=0.40,
    stability_percentile_global_metagenomics=0.25,
    z_thresh=3.0,
    decompose_method='PCA',
    decompose_rank=30,
    taxonomy_level_metagenomics=None,
    taxonomy_level_amplicon=None,
    cross_dtype_normalization: bool = False,
    stability_percentile_global_combined: float = 0.6,
    taxonomy_level_combined=None,
):
    if cross_dtype_normalization:
        # Pool ALL datasets (both dtypes) into one normalization pass
        print(f"\n[Global preprocessing — combined 16S+shotgun]  approach={normalization_approach}")
        tax_level = taxonomy_level_combined if str(taxonomy_level_combined) != "None" else None
        return _run_global_for_dtype(
            phenotypes,
            normalization_approach=normalization_approach,
            apply_decompose=apply_decompose,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_global_combined,
            z_thresh=z_thresh,
            decompose_method=decompose_method,
            decompose_rank=decompose_rank,
            taxonomy_level=tax_level,
        )

    # Default: normalize each dtype separately
    phenotypes_by_dtype = defaultdict(list)
    for phenotype, dtype in phenotypes:
        phenotypes_by_dtype[dtype].append((phenotype, dtype))

    all_records = []

    stability_percentile_by_dtype = {
        "Amplicon":     stability_percentile_global_amplicon,
        "Metagenomics": stability_percentile_global_metagenomics,
    }
    taxonomy_level_by_dtype = {
        "Amplicon":     taxonomy_level_amplicon,
        "Metagenomics": taxonomy_level_metagenomics,
    }

    for dtype, phenotype_list in phenotypes_by_dtype.items():
        print(f"\n[Global preprocessing] dtype={dtype}  approach={normalization_approach}")

        records_dtype = _run_global_for_dtype(
            phenotype_list,
            normalization_approach=normalization_approach,
            apply_decompose=apply_decompose,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_by_dtype.get(dtype, stability_percentile_global_amplicon),
            z_thresh=z_thresh,
            decompose_method=decompose_method,
            decompose_rank=decompose_rank,
            taxonomy_level=taxonomy_level_by_dtype.get(dtype),
        )

        all_records.append(records_dtype)

    if len(all_records) == 0:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)
