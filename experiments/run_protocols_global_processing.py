import numpy as np
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
    clr_normalize, clr_rank_normalize,
    apply_ranking_pipeline,
    SIGMOID_K, SIGMOID_CENTER, RELU_THRESHOLD,
)
from src.rankbird.representation.multi_datasets_SPDR import apply_bias_SPDR
from evaluation.data_loading import load_microbiome_datasets_with_targets
from evaluation.learning_protocols import (
    lodo_protocol, internal_validation_protocol, within_dataset_protocol,
    lodo_protocol_optuna, internal_validation_protocol_optuna, within_dataset_protocol_optuna,
    lodo_protocol_cross_phenotype, internal_validation_protocol_cross_phenotype,
    lodo_protocol_lr, internal_validation_protocol_lr, within_dataset_protocol_lr,
)


def _detect_and_shuffle_ordered(
    microbiome_dfs: list,
    target_dfs: list,
    dataset_names: list,
    random_state: int = 42,
    apply_shuffle: bool = True,
    verbose: bool = False,
) -> tuple:
    """
    Detect datasets where class labels are fully ordered (all 0s then 1s, or
    all 1s then 0s). If apply_shuffle=True, shuffle those datasets so that
    samples and their corresponding targets are reordered together.

    Set verbose=True to print a summary (suppressed by default to avoid
    repeating the same message for every approach × dtype combination).
    Returns (microbiome_dfs, target_dfs) — possibly modified in-place copies.
    """
    rng = np.random.default_rng(random_state)
    ordered_info = []   # list of {"dataset": name, "first_class": 0_or_1}
    new_microbiome, new_targets = [], []

    for df, y, name in zip(microbiome_dfs, target_dfs, dataset_names):
        tags = y["Tag"].values
        is_sorted_asc  = np.array_equal(tags, np.sort(tags))
        is_sorted_desc = np.array_equal(tags, np.sort(tags)[::-1])
        is_ordered = (is_sorted_asc or is_sorted_desc) and len(np.unique(tags)) > 1

        if is_ordered:
            # is_sorted_asc  → controls (0) come first
            # is_sorted_desc → cases   (1) come first
            ordered_info.append({"dataset": name, "first_class": 0 if is_sorted_asc else 1})

        if is_ordered and apply_shuffle:
            perm = rng.permutation(df.index)
            new_microbiome.append(df.loc[perm])
            new_targets.append(y.loc[perm])
        else:
            new_microbiome.append(df)
            new_targets.append(y)

    if verbose:
        action = "shuffled" if apply_shuffle else "not shuffled"
        if ordered_info:
            print(
                f"\n[ORDER-CHECK] {len(ordered_info)} fully-ordered dataset(s) ({action}): "
                f"{[d['dataset'] for d in ordered_info]}"
            )
        else:
            print(f"\n[ORDER-CHECK] No fully-ordered datasets found.")

    return new_microbiome, new_targets, ordered_info


def _run_protocols_on_group(
    microbiome_dfs, target_dfs, dataset_names, phenotype_str,
    use_optuna: bool = False,
    n_trials: int = 30,
):
    records = []

    if use_optuna:
        results = {
            "LODO": lodo_protocol_optuna(
                microbiome_dfs, target_dfs, dataset_names, n_trials=n_trials),
            "Internal Validation": internal_validation_protocol_optuna(
                microbiome_dfs, target_dfs, dataset_names, n_trials=n_trials),
            "Within Learning": within_dataset_protocol_optuna(
                microbiome_dfs, target_dfs, dataset_names, n_trials=n_trials),
        }
    else:
        results = {
            "LODO": lodo_protocol(microbiome_dfs, target_dfs, dataset_names),
            "Internal Validation": internal_validation_protocol(microbiome_dfs, target_dfs, dataset_names),
            "Within Learning": within_dataset_protocol(microbiome_dfs, target_dfs, dataset_names),
        }

    _METRICS = ["auc", "accuracy", "precision", "recall", "f1"]
    for protocol, df in results.items():
        for _, row in df.iterrows():
            rec = {
                "phenotype": phenotype_str,
                "dataset":   row["test_dataset"],
                "protocol":  protocol,
            }
            for m in _METRICS:
                rec[m]             = row.get(m,           float("nan"))
                rec[f"train_{m}"]  = row.get(f"train_{m}", float("nan"))
            records.append(rec)

    return pd.DataFrame(records)


def _dispatch_protocols(
    all_microbiome: list,
    all_targets: list,
    all_names: list,
    dataset_to_phenotype: dict,
    use_optuna,
    n_trials: int,
) -> pd.DataFrame:
    """Run all three protocols with the correct mode based on use_optuna.

    use_optuna=False          → fixed params, per phenotype
    use_optuna=True           → Optuna per fold per phenotype
    use_optuna="cross_optuna" → Optuna on cross-phenotype pool, final model per-phenotype
    use_optuna="cross_train"  → Optuna AND final model cross-phenotype
    """
    _METRICS = ["auc", "accuracy", "precision", "recall", "f1"]
    records = []

    if use_optuna in ("cross_optuna", "cross_train"):
        cross_train_flag = (use_optuna == "cross_train")

        lodo_df = lodo_protocol_cross_phenotype(
            all_microbiome, all_targets, all_names,
            dataset_to_phenotype, cross_train=cross_train_flag, n_trials=n_trials,
        )
        iv_df = internal_validation_protocol_cross_phenotype(
            all_microbiome, all_targets, all_names,
            dataset_to_phenotype, cross_train=cross_train_flag, n_trials=n_trials,
        )

        for protocol_name, df in [("LODO", lodo_df), ("Internal Validation", iv_df)]:
            for _, row in df.iterrows():
                test_name = row["test_dataset"]
                pheno_str = dataset_to_phenotype.get(test_name, "unknown")
                rec = {"phenotype": pheno_str, "dataset": test_name, "protocol": protocol_name}
                for m in _METRICS:
                    rec[m]            = row.get(m,           float("nan"))
                    rec[f"train_{m}"] = row.get(f"train_{m}", float("nan"))
                records.append(rec)

        # Within: per-dataset — no cross-phenotype pooling here
        for pheno_str in sorted(set(dataset_to_phenotype.values())):
            idx = [i for i, nm in enumerate(all_names) if dataset_to_phenotype[nm] == pheno_str]
            wdf = within_dataset_protocol_optuna(
                [all_microbiome[i] for i in idx],
                [all_targets[i]    for i in idx],
                [all_names[i]      for i in idx],
                n_trials=n_trials,
            )
            for _, row in wdf.iterrows():
                rec = {"phenotype": pheno_str, "dataset": row["test_dataset"],
                       "protocol": "Within Learning"}
                for m in _METRICS:
                    rec[m]            = row.get(m,           float("nan"))
                    rec[f"train_{m}"] = row.get(f"train_{m}", float("nan"))
                records.append(rec)

    else:
        for phenotype_str in set(dataset_to_phenotype.values()):
            idx = [i for i, nm in enumerate(all_names) if dataset_to_phenotype[nm] == phenotype_str]
            df_grp = _run_protocols_on_group(
                [all_microbiome[i] for i in idx],
                [all_targets[i]    for i in idx],
                [all_names[i]      for i in idx],
                phenotype_str,
                use_optuna=use_optuna,
                n_trials=n_trials,
            )
            records.append(df_grp)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    return pd.DataFrame(records)


_RANKING_NORM_FNS = {
    "rankbird_ranking":     rank_normalize,
    "rankbird_sigmoid":     sigmoid_normalize,
    "rankbird_relu":        relu_normalize,
    "rankbird_clr":         clr_normalize,
    "rankbird_clr_ranking": clr_rank_normalize,
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
    data_root: str = "Data",
    shuffle_ordered: bool = False,
    random_state: int = 42,
    rank_tie_method: str = "first",
    use_optuna: bool = False,
    n_trials: int = 30,
    run_lr: bool = False,
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
    DATA_ROOT = PROJECT_ROOT / data_root

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
    # 1b. Detect (and optionally shuffle) fully-ordered datasets
    # ----------------------------------
    all_microbiome, all_targets, _ = _detect_and_shuffle_ordered(
        all_microbiome, all_targets, all_dataset_names,
        random_state=random_state, apply_shuffle=shuffle_ordered,
    )

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
            rank_tie_method=rank_tie_method,
        )

    elif normalization_approach in _RANKING_NORM_FNS:
        _base_norm_fn = _RANKING_NORM_FNS[normalization_approach]
        from functools import partial as _partial
        norm_fn = _partial(_base_norm_fn, tie_method=rank_tie_method)
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
        # Remove features that are all-zero in every sample of every dataset.
        # These are reference-taxonomy ghost species — never detected anywhere —
        # that enter the union column set only because a bioinformatics pipeline
        # listed them. They contribute zero signal and would make the "original"
        # baseline inconsistent with filter_only at pct=1.0 (which drops them via
        # auto_stability_filter's .dropna()).
        _all_microbes = union_microbes(all_microbiome)
        _nz = nonzero_percent_by_dataset(all_microbiome, all_dataset_names, _all_microbes)
        _ever_observed = _nz.mean(axis=0) > 0
        _kept = _ever_observed[_ever_observed].index.tolist()
        all_microbiome = [df.reindex(columns=_kept, fill_value=0.0) for df in all_microbiome]

    if apply_decompose:
        all_microbiome, _, _ = apply_bias_SPDR(all_microbiome, decompose_method, rank=decompose_rank)

    if normalization_approach is None:
        # No alignment across datasets — report union of all feature columns
        n_features = len(set().union(*[set(df.columns) for df in all_microbiome])) if all_microbiome else 0
    else:
        # All datasets reindexed to same columns after filtering
        n_features = all_microbiome[0].shape[1] if all_microbiome else 0

    # ----------------------------------
    # 3. Run protocols
    # ----------------------------------
    lgbm_df = _dispatch_protocols(
        all_microbiome, all_targets, all_dataset_names, dataset_to_phenotype,
        use_optuna=use_optuna, n_trials=n_trials,
    )

    if not run_lr:
        return lgbm_df, n_features

    lgbm_df["model"] = "LGBM"

    _METRICS = ["auc", "accuracy", "precision", "recall", "f1"]
    _apply_clr = normalization_approach not in ("rankbird_clr", "rankbird_clr_ranking")
    _lr_results = []
    for pheno_str in set(dataset_to_phenotype.values()):
        idx  = [i for i, nm in enumerate(all_dataset_names) if dataset_to_phenotype[nm] == pheno_str]
        _mb  = [all_microbiome[i] for i in idx]
        _tgt = [all_targets[i]    for i in idx]
        _nm  = [all_dataset_names[i] for i in idx]
        for protocol_name, lr_fn in [
            ("LODO",                lodo_protocol_lr),
            ("Internal Validation", internal_validation_protocol_lr),
            ("Within Learning",     within_dataset_protocol_lr),
        ]:
            df = lr_fn(_mb, _tgt, _nm, apply_clr=_apply_clr)
            for _, row in df.iterrows():
                rec = {"phenotype": pheno_str, "dataset": row["test_dataset"],
                       "protocol": protocol_name, "model": "LR"}
                for m in _METRICS:
                    rec[m]            = row.get(m,           float("nan"))
                    rec[f"train_{m}"] = row.get(f"train_{m}", float("nan"))
                _lr_results.append(rec)

    lr_df = pd.DataFrame(_lr_results)
    return pd.concat([lgbm_df, lr_df], ignore_index=True), n_features


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
    data_root: str = "Data",
    use_optuna: bool = False,
    n_trials: int = 30,
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
            data_root=data_root,
            use_optuna=use_optuna,
            n_trials=n_trials,
        )

    # Default: normalize each dtype separately
    phenotypes_by_dtype = defaultdict(list)
    for phenotype, dtype in phenotypes:
        phenotypes_by_dtype[dtype].append((phenotype, dtype))

    all_records = []
    feature_counts = {}

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

        records_dtype, n_feat = _run_global_for_dtype(
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
            data_root=data_root,
            use_optuna=use_optuna,
            n_trials=n_trials,
        )

        all_records.append(records_dtype)
        feature_counts[dtype] = n_feat

    if len(all_records) == 0:
        return pd.DataFrame(), {}

    return pd.concat(all_records, ignore_index=True), feature_counts
