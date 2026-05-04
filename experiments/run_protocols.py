import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, mannwhitneyu
from src.rankbird.normalization.pipeline import apply_normalization_pipeline #### Change after package
from src.rankbird.representation.multi_datasets_SPDR import bias_SPDR   ### Change after package
from evaluation.data_loading import load_microbiome_datasets_with_targets
from evaluation.learning_protocols import lodo_protocol, internal_validation_protocol, within_dataset_protocol

def run_protocol_benchmark(phenotypes: list,
                            data_root: str = "Data",
                            normalization_approach=None,
                            apply_decompose: bool = False,
                            stability_threshold: float = 1.0,
                            min_samples_per_dataset: int = 30,
                            outlier_z_thresh: float = 3.0,
                            random_state: int = 42,
                            **kwargs):
    """
    Run LODO, Internal Validation, Within-dataset protocols
    for multiple phenotypes, and returns a tidy DataFrame with per-dataset AUCs.

    Parameters
    ----------
    phenotypes : list
        List of phenotype names (e.g. [("CRC", "Metagenomics"), ("PD", "Amplicon")])
    data_root : str
        Root folder containing "{Phenotype} Metagenomic" and "{Phenotype} Amplicon" subfolders.
    normalization_approach : str or None
        Normalization approach: "rankbird_wasserstein", "rankbird_ranking",
        "rankbird_sigmoid", "rankbird_relu", "filter_only", or None (no normalization).
    min_samples_per_dataset : int
        Minimum samples per dataset for oversampling.
    outlier_z_thresh : float
        Z-score threshold for removing outlier datasets.
    random_state : int
        Random state for reproducibility.
    """

    records = []

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / data_root

    for phenotype, dtype in phenotypes:
        phenotype_str = f"{phenotype} {dtype}"
        folder = DATA_ROOT / phenotype_str
        if not folder.exists():
            continue

        # Load microbiome data
        microbiome_dfs, target_dfs, dataset_names = load_microbiome_datasets_with_targets(folder)

        # Apply normalization if requested
        if normalization_approach == "rankbird_wasserstein":
            microbiome_dfs, dataset_names = apply_normalization_pipeline(
                microbiome_dfs, dataset_names)
            save_folder = f"normalized_datasets/"
            os.makedirs(save_folder, exist_ok=True)

            for df, name in zip(microbiome_dfs, dataset_names):
                filename = PROJECT_ROOT / f"{save_folder}{name}_normalized.csv"
                df.to_csv(filename)
                print(f"Saved: {filename}")

        if apply_decompose:
            microbiome_dfs, eta, beta = bias_SPDR(microbiome_dfs, 'PCA', None)

        # Run protocols
        results = {
            "LODO": lodo_protocol(microbiome_dfs, target_dfs, dataset_names),
            "Internal Validation": internal_validation_protocol(microbiome_dfs, target_dfs, dataset_names),
            "Within Learning": within_dataset_protocol(microbiome_dfs, target_dfs, dataset_names)
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