import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, mannwhitneyu
from collections import defaultdict
from src.rankbird.normalization.pipeline import apply_normalization_pipeline #### Change after package
from src.rankbird.representation.multi_datasets_SPDR import apply_bias_SPDR   ### Change after package
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


def _run_global_for_dtype(phenotypes,
                          apply_normalization=False,
                          apply_decompose=False,
                          min_samples_per_dataset=550, 
                          stability_percentile_local=0.3,
                          stability_percentile_global=0.5,
                          z_thresh=3.0,
                          decompose_method='PCA',
                          decompose_rank=30
                          ):

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

        microbiome_dfs, target_dfs, dataset_names, dropped_datasets = \
            load_microbiome_datasets_with_targets(folder)

        for df, y, name in zip(microbiome_dfs, target_dfs, dataset_names):
            all_microbiome.append(df)
            all_targets.append(y)
            all_dataset_names.append(name)
            dataset_to_phenotype[name] = phenotype_str

    # ----------------------------------
    # 2. GLOBAL preprocessing (once)
    # ----------------------------------

    if apply_normalization:
        all_microbiome, all_dataset_names = apply_normalization_pipeline(
            all_microbiome,
            all_dataset_names,
            global_analysis=True,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_global,
            z_thresh=z_thresh
        )

    if apply_decompose:
        all_microbiome, eta, beta = apply_bias_SPDR(all_microbiome, 'PCA', rank=decompose_rank)

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
    apply_normalization: bool = False,
    apply_decompose: bool = False,
    min_samples_per_dataset=550,
    stability_percentile_local=0.3,
    stability_percentile_global=0.5,
    z_thresh=3.0,
    decompose_method='PCA',
    decompose_rank=30
):


    phenotypes_by_dtype = defaultdict(list)
    for phenotype, dtype in phenotypes:
        phenotypes_by_dtype[dtype].append((phenotype, dtype))

    all_records = []

    for dtype, phenotype_list in phenotypes_by_dtype.items():
        print(f"\n[Global preprocessing] dtype = {dtype}")

        records_dtype = _run_global_for_dtype(
            phenotype_list,
            apply_normalization=apply_normalization,
            apply_decompose=apply_decompose,
            min_samples_per_dataset=min_samples_per_dataset, 
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_global,
            z_thresh=z_thresh,
            decompose_method=decompose_method,
            decompose_rank=decompose_rank
        )

        all_records.append(records_dtype)

    if len(all_records) == 0:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)
