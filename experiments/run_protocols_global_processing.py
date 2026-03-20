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
from evaluation.lgbm_model import train_lightgbm

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


def _run_ae_protocols(
    all_microbiome,
    all_targets,
    all_dataset_names,
    dataset_to_phenotype,
    ae_latent_dim=64,
    ae_epochs=100,
    ae_batch_size=32,
    ae_lr=1e-3,
):
    """
    Run LODO and Internal Validation protocols using a Supervised DAE encoder.

    LODO:               AE trained on all dtype datasets *except* the held-out one (no leakage).
    Internal Validation: AE trained on all dtype datasets, then standard internal validation.
    Within Learning:    Not run here — caller should keep normalized results for that protocol.
    """
    from src.rankbird.autoencoder.supervised_dae import train_supervised_dae, encode_datasets

    records = []
    n = len(all_dataset_names)

    # ── LODO (per-fold AE training) ───────────────────────────────────────────
    print("\n[AE] LODO — per-fold supervised DAE training")
    for i, test_name in enumerate(all_dataset_names):
        train_idx = [j for j in range(n) if j != i]
        train_dfs  = [all_microbiome[j] for j in train_idx]
        train_tgts = pd.concat([all_targets[j].iloc[:, 0] for j in train_idx])

        print(f"  [AE LODO] held-out={test_name}  train_sets={len(train_dfs)}")

        # Align columns: intersection across training DataFrames + test
        common_cols = all_microbiome[i].columns
        for df in train_dfs:
            common_cols = common_cols.intersection(df.columns)
        train_dfs_aligned = [df[common_cols] for df in train_dfs]
        test_df_aligned   = all_microbiome[i][common_cols]

        model, scaler = train_supervised_dae(
            train_dfs_aligned, train_tgts,
            latent_dim=ae_latent_dim, epochs=ae_epochs,
            batch_size=ae_batch_size, lr=ae_lr, verbose=False,
        )

        # Encode test dataset
        test_enc = encode_datasets(model, scaler, [test_df_aligned], [test_name], ae_latent_dim)[0]
        y_test   = all_targets[i]

        # Encode same-phenotype training datasets
        pheno = dataset_to_phenotype[test_name]
        pheno_train_idx = [j for j in train_idx if dataset_to_phenotype[all_dataset_names[j]] == pheno]

        if not pheno_train_idx:
            print(f"  [AE LODO] skip {test_name}: no same-phenotype training datasets")
            continue

        pheno_train_dfs = [all_microbiome[j][common_cols] for j in pheno_train_idx]
        pheno_train_enc = encode_datasets(
            model, scaler, pheno_train_dfs,
            [all_dataset_names[j] for j in pheno_train_idx], ae_latent_dim
        )
        X_train = pd.concat(pheno_train_enc, axis=0)
        y_train = pd.concat([all_targets[j].iloc[:, 0] for j in pheno_train_idx]).to_frame()

        metrics = train_lightgbm(X_train, y_train, test_enc, y_test)
        records.append({
            "phenotype": pheno,
            "dataset":   test_name,
            "protocol":  "LODO",
            "auc":       metrics["auc"],
        })
        print(f"    AUC={metrics['auc']:.4f}")

    # ── Internal Validation (global AE, then standard protocol) ──────────────
    print("\n[AE] Internal Validation — global supervised DAE")
    y_all = pd.concat([t.iloc[:, 0] for t in all_targets])

    # Align columns across all datasets for global training
    common_cols_global = all_microbiome[0].columns
    for df in all_microbiome[1:]:
        common_cols_global = common_cols_global.intersection(df.columns)
    aligned_all = [df[common_cols_global] for df in all_microbiome]

    model_global, scaler_global = train_supervised_dae(
        aligned_all, y_all,
        latent_dim=ae_latent_dim, epochs=ae_epochs,
        batch_size=ae_batch_size, lr=ae_lr, verbose=True,
    )
    encoded_all = encode_datasets(
        model_global, scaler_global, aligned_all, all_dataset_names, ae_latent_dim
    )

    # Split back by phenotype and run internal validation
    for pheno_str in set(dataset_to_phenotype.values()):
        pheno_idx = [
            i for i, name in enumerate(all_dataset_names)
            if dataset_to_phenotype[name] == pheno_str
        ]
        enc_grp    = [encoded_all[i] for i in pheno_idx]
        target_grp = [all_targets[i] for i in pheno_idx]
        names_grp  = [all_dataset_names[i] for i in pheno_idx]

        iv_results = internal_validation_protocol(enc_grp, target_grp, names_grp)
        for _, row in iv_results.iterrows():
            records.append({
                "phenotype": pheno_str,
                "dataset":   row["test_dataset"],
                "protocol":  "Internal Validation",
                "auc":       row["auc"],
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
                          decompose_rank=30,
                          apply_autoencoder=False,
                          ae_latent_dim=64,
                          ae_epochs=100,
                          ae_batch_size=32,
                          ae_lr=1e-3,
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

    # ----------------------------------
    # 4. Supervised DAE protocols (optional)
    # ----------------------------------
    ae_records = None
    if apply_autoencoder:
        print("\n[AE] Running supervised DAE protocols...")
        ae_df = _run_ae_protocols(
            all_microbiome=all_microbiome,
            all_targets=all_targets,
            all_dataset_names=all_dataset_names,
            dataset_to_phenotype=dataset_to_phenotype,
            ae_latent_dim=ae_latent_dim,
            ae_epochs=ae_epochs,
            ae_batch_size=ae_batch_size,
            ae_lr=ae_lr,
        )
        ae_records = ae_df

    return pd.concat(records, ignore_index=True), ae_records


def run_protocol_benchmark_global_preprocessing(
    phenotypes: list,
    apply_normalization: bool = False,
    apply_decompose: bool = False,
    min_samples_per_dataset=550,
    stability_percentile_local=0.3,
    stability_percentile_global_amplicon=0.40,
    stability_percentile_global_metagenomics=0.25,
    z_thresh=3.0,
    decompose_method='PCA',
    decompose_rank=30,
    apply_autoencoder=False,
    ae_latent_dim=64,
    ae_epochs=100,
    ae_batch_size=32,
    ae_lr=1e-3,
):
    phenotypes_by_dtype = defaultdict(list)
    for phenotype, dtype in phenotypes:
        phenotypes_by_dtype[dtype].append((phenotype, dtype))

    all_records = []
    all_ae_records = []

    stability_percentile_by_dtype = {
        "Amplicon":     stability_percentile_global_amplicon,
        "Metagenomics": stability_percentile_global_metagenomics,
    }

    for dtype, phenotype_list in phenotypes_by_dtype.items():
        print(f"\n[Global preprocessing] dtype = {dtype}")

        records_dtype, ae_records_dtype = _run_global_for_dtype(
            phenotype_list,
            apply_normalization=apply_normalization,
            apply_decompose=apply_decompose,
            min_samples_per_dataset=min_samples_per_dataset,
            stability_percentile_local=stability_percentile_local,
            stability_percentile_global=stability_percentile_by_dtype.get(dtype, stability_percentile_global_amplicon),
            z_thresh=z_thresh,
            decompose_method=decompose_method,
            decompose_rank=decompose_rank,
            apply_autoencoder=apply_autoencoder,
            ae_latent_dim=ae_latent_dim,
            ae_epochs=ae_epochs,
            ae_batch_size=ae_batch_size,
            ae_lr=ae_lr,
        )

        all_records.append(records_dtype)
        if ae_records_dtype is not None:
            all_ae_records.append(ae_records_dtype)

    if len(all_records) == 0:
        return pd.DataFrame(), pd.DataFrame()

    ae_df = pd.concat(all_ae_records, ignore_index=True) if all_ae_records else pd.DataFrame()
    return pd.concat(all_records, ignore_index=True), ae_df
