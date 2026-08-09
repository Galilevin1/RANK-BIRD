import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from .data_loading import combine_datasets
from .lgbm_model import train_lightgbm, train_lightgbm_optuna, train_lightgbm_optuna_cross

def lodo_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                        dataset_names: List[str]) -> pd.DataFrame:
    """Step 1: Basic LODO learning."""
    print("\n=== Step 1: LODO Learning ===")
    results = []

    for test_idx in range(len(dataset_names)):
        train_indices = [i for i in range(len(dataset_names)) if i != test_idx]

        # Combine training datasets
        X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
        X_test, y_test = microbiome_dfs[test_idx], target_dfs[test_idx]

        # Align features
        common_features = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common_features], X_test[common_features]

        print(f"Training on {[dataset_names[i] for i in train_indices]}, testing on {dataset_names[test_idx]}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test)

        result = {
            'step': 'Step1_LODO',
            'test_dataset': dataset_names[test_idx],
            'train_datasets': ','.join([dataset_names[i] for i in train_indices]),
            **metrics
        }
        results.append(result)
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def internal_validation_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                              dataset_names: List[str]) -> pd.DataFrame:
    """Step 2: Internal validation on training combinations."""
    print("\n=== Step 2: Internal Validation ===")
    results = []

    for test_idx in range(len(dataset_names)):
        train_indices = [i for i in range(len(dataset_names)) if i != test_idx]

        # Combine training datasets
        X_combined, y_combined = combine_datasets(microbiome_dfs, target_dfs, train_indices)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        train_dataset_names = [dataset_names[i] for i in train_indices]
        print(f"Internal validation on {train_dataset_names}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test)

        result = {
            'step': 'Step2_Internal',
            'test_dataset': dataset_names[test_idx],  # This is the external test set context
            'train_datasets': ','.join(train_dataset_names),
            **metrics
        }
        results.append(result)
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def within_dataset_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                         dataset_names: List[str]) -> pd.DataFrame:
    """Step 4: Within-dataset learning."""
    print("\n=== Step 4: Within-Dataset Learning ===")
    results = []

    for idx, dataset_name in enumerate(dataset_names):
        X, y = microbiome_dfs[idx], target_dfs[idx]

        if len(y.value_counts()) < 2:
            print(f"Skipping {dataset_name} - only one class present")
            continue

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Within-dataset learning for {dataset_name}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test)

        result = {
            'step': 'Step4_Within',
            'test_dataset': dataset_name,
            'train_datasets': dataset_name,
            **metrics
        }
        results.append(result)
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


# ── Optuna-tuned variants ──────────────────────────────────────────────────────

def lodo_protocol_optuna(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs:     List[pd.DataFrame],
    dataset_names:  List[str],
    n_trials: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """LODO with Optuna tuning.

    For each test dataset i: val = dataset (i+1) % N, train = all others.
    Optuna optimises on val; final model is evaluated on test.
    Falls back to fixed-param LODO when N < 3.
    """
    print("\n=== LODO (Optuna) ===")
    n = len(dataset_names)
    results = []

    for test_idx in range(n):
        if n < 3:
            train_indices = [i for i in range(n) if i != test_idx]
            X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
            X_test,  y_test  = microbiome_dfs[test_idx], target_dfs[test_idx]
            common = X_train.columns.intersection(X_test.columns)
            metrics = train_lightgbm(X_train[common], y_train, X_test[common], y_test)
        else:
            # Pick val: first candidate after test_idx (round-robin) that has both classes
            val_idx = None
            for offset in range(1, n):
                candidate = (test_idx + offset) % n
                if len(target_dfs[candidate]["Tag"].unique()) >= 2:
                    val_idx = candidate
                    break
            if val_idx is None:
                # No valid val dataset — fall back to standard LODO without Optuna
                train_indices = [i for i in range(n) if i != test_idx]
                X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
                X_test,  y_test  = microbiome_dfs[test_idx], target_dfs[test_idx]
                common = X_train.columns.intersection(X_test.columns)
                metrics = train_lightgbm(X_train[common], y_train, X_test[common], y_test)
                results.append({'test_dataset': dataset_names[test_idx], **metrics})
                print(f"  AUC: {metrics['auc']:.4f}  (no valid val set — fixed params)")
                continue
            train_indices = [i for i in range(n) if i != test_idx and i != val_idx]

            X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
            X_val,   y_val   = microbiome_dfs[val_idx],  target_dfs[val_idx]
            X_test,  y_test  = microbiome_dfs[test_idx], target_dfs[test_idx]

            common = (X_train.columns
                      .intersection(X_val.columns)
                      .intersection(X_test.columns))
            X_train, X_val, X_test = X_train[common], X_val[common], X_test[common]

            print(f"  train={[dataset_names[i] for i in train_indices]}  "
                  f"val={dataset_names[val_idx]}  test={dataset_names[test_idx]}")
            metrics = train_lightgbm_optuna(
                X_train, y_train, X_val, y_val, X_test, y_test,
                n_trials=n_trials, param_space="default", random_state=random_state,
            )

        results.append({'test_dataset': dataset_names[test_idx], **metrics})
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def internal_validation_protocol_optuna(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs:     List[pd.DataFrame],
    dataset_names:  List[str],
    n_trials: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Mixed-datasets internal validation with Optuna tuning.

    For each LODO context i: pool the other N-1 datasets → 60/20/20
    (train/val/test). Optuna optimises on val; evaluate on test.
    """
    print("\n=== Internal Validation (Optuna) ===")
    results = []

    for test_idx in range(len(dataset_names)):
        pool_indices = [i for i in range(len(dataset_names)) if i != test_idx]
        X_pool, y_pool = combine_datasets(microbiome_dfs, target_dfs, pool_indices)

        X_tv, X_test, y_tv, y_test = train_test_split(
            X_pool, y_pool, test_size=0.20, random_state=random_state, stratify=y_pool,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.25, random_state=random_state, stratify=y_tv,
        )

        print(f"  Mixed validation (context={dataset_names[test_idx]})")
        metrics = train_lightgbm_optuna(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=n_trials, param_space="default", random_state=random_state,
        )

        results.append({'test_dataset': dataset_names[test_idx], **metrics})
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def lodo_protocol_cross_phenotype(
    all_microbiome_dfs: List[pd.DataFrame],
    all_target_dfs:     List[pd.DataFrame],
    all_dataset_names:  List[str],
    dataset_to_phenotype: Dict[str, str],
    cross_train: bool = False,
    n_trials: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """LODO with cross-phenotype Optuna.

    For each test dataset:
      val          = another dataset from the same phenotype
      Optuna train = ALL datasets in the dtype except test and val
      Final train  = same-phenotype datasets only       (cross_train=False)
                   = same cross-phenotype Optuna pool   (cross_train=True)
    """
    mode = "cross_train" if cross_train else "cross_optuna"
    print(f"\n=== LODO ({mode}) ===")
    n = len(all_dataset_names)
    results = []

    for test_idx in range(n):
        test_name  = all_dataset_names[test_idx]
        test_pheno = dataset_to_phenotype[test_name]

        same_pheno_others = [
            i for i, nm in enumerate(all_dataset_names)
            if dataset_to_phenotype[nm] == test_pheno and i != test_idx
        ]
        val_idx = None
        for i in same_pheno_others:
            if len(all_target_dfs[i]["Tag"].unique()) >= 2:
                val_idx = i
                break

        if val_idx is None:
            train_idx = same_pheno_others
            if not train_idx:
                continue
            X_tr, y_tr = combine_datasets(all_microbiome_dfs, all_target_dfs, train_idx)
            X_te, y_te = all_microbiome_dfs[test_idx], all_target_dfs[test_idx]
            common = X_tr.columns.intersection(X_te.columns)
            metrics = train_lightgbm(X_tr[common], y_tr, X_te[common], y_te)
            results.append({'test_dataset': test_name, **metrics})
            print(f"  {test_name} AUC: {metrics['auc']:.4f}  (no valid val — fixed params)")
            continue

        optuna_train_idx = [i for i in range(n) if i != test_idx and i != val_idx]
        pheno_train_idx  = [i for i in same_pheno_others if i != val_idx]

        if cross_train or not pheno_train_idx:
            final_train_idx = optuna_train_idx
        else:
            final_train_idx = pheno_train_idx

        X_optuna_tr, y_optuna_tr = combine_datasets(all_microbiome_dfs, all_target_dfs, optuna_train_idx)
        X_final_tr,  y_final_tr  = combine_datasets(all_microbiome_dfs, all_target_dfs, final_train_idx)
        X_val,  y_val  = all_microbiome_dfs[val_idx],  all_target_dfs[val_idx]
        X_test, y_test = all_microbiome_dfs[test_idx], all_target_dfs[test_idx]

        common = (X_optuna_tr.columns
                  .intersection(X_final_tr.columns)
                  .intersection(X_val.columns)
                  .intersection(X_test.columns))
        X_optuna_tr = X_optuna_tr[common]
        X_final_tr  = X_final_tr[common]
        X_val       = X_val[common]
        X_test      = X_test[common]

        print(f"  test={test_name}  val={all_dataset_names[val_idx]}  "
              f"optuna_pool={len(optuna_train_idx)}  final_pool={len(final_train_idx)}")
        metrics = train_lightgbm_optuna_cross(
            X_optuna_tr, y_optuna_tr,
            X_final_tr,  y_final_tr,
            X_val, y_val, X_test, y_test,
            n_trials=n_trials, param_space="default", random_state=random_state,
        )

        results.append({'test_dataset': test_name, **metrics})
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def internal_validation_protocol_cross_phenotype(
    all_microbiome_dfs: List[pd.DataFrame],
    all_target_dfs:     List[pd.DataFrame],
    all_dataset_names:  List[str],
    dataset_to_phenotype: Dict[str, str],
    cross_train: bool = False,
    n_trials: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Internal validation with cross-phenotype Optuna.

    For each phenotype and each context dataset in that phenotype:
      Pool          = same-phenotype remaining N-1 datasets → 60/20/20
      Optuna train  = 60% same-phenotype + ALL other phenotypes in the dtype
      Final train   = 60% same-phenotype only              (cross_train=False)
                    = 60% same-phenotype + ALL others       (cross_train=True)
      Val / Test    = 20% / 20% of same-phenotype pool (no other phenotypes)
    """
    mode = "cross_train" if cross_train else "cross_optuna"
    print(f"\n=== Internal Validation ({mode}) ===")
    results = []

    for phenotype in sorted(set(dataset_to_phenotype.values())):
        pheno_indices = [
            i for i, nm in enumerate(all_dataset_names)
            if dataset_to_phenotype[nm] == phenotype
        ]
        other_indices = [
            i for i in range(len(all_dataset_names))
            if dataset_to_phenotype[all_dataset_names[i]] != phenotype
        ]

        for context_idx in pheno_indices:
            pool_indices = [i for i in pheno_indices if i != context_idx]
            if not pool_indices:
                continue

            X_pool, y_pool = combine_datasets(all_microbiome_dfs, all_target_dfs, pool_indices)

            try:
                X_tv, X_test, y_tv, y_test = train_test_split(
                    X_pool, y_pool, test_size=0.20, random_state=random_state, stratify=y_pool,
                )
                X_pheno_tr, X_val, y_pheno_tr, y_val = train_test_split(
                    X_tv, y_tv, test_size=0.25, random_state=random_state, stratify=y_tv,
                )
            except ValueError:
                continue

            if other_indices:
                X_other, y_other = combine_datasets(all_microbiome_dfs, all_target_dfs, other_indices)
                common = (X_pheno_tr.columns
                          .intersection(X_other.columns)
                          .intersection(X_val.columns)
                          .intersection(X_test.columns))
                X_optuna_tr = pd.concat([X_pheno_tr[common], X_other[common]], ignore_index=True)
                y_optuna_tr = pd.concat([y_pheno_tr,         y_other],         ignore_index=True)
            else:
                common = X_pheno_tr.columns.intersection(X_val.columns).intersection(X_test.columns)
                X_optuna_tr = X_pheno_tr[common]
                y_optuna_tr = y_pheno_tr

            if cross_train:
                X_final_tr = X_optuna_tr
                y_final_tr = y_optuna_tr
            else:
                X_final_tr = X_pheno_tr[common]
                y_final_tr = y_pheno_tr

            X_val  = X_val[common]
            X_test = X_test[common]

            context_name = all_dataset_names[context_idx]
            print(f"  Internal ({phenotype} context={context_name})")
            metrics = train_lightgbm_optuna_cross(
                X_optuna_tr, y_optuna_tr,
                X_final_tr,  y_final_tr,
                X_val, y_val, X_test, y_test,
                n_trials=n_trials, param_space="default", random_state=random_state,
            )

            results.append({'test_dataset': context_name, **metrics})
            print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def within_dataset_protocol_optuna(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs:     List[pd.DataFrame],
    dataset_names:  List[str],
    n_trials: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Within-dataset learning with Optuna tuning.

    Splits each dataset 60/20/20 (train/val/test). Uses a more regularised
    parameter space suited to small datasets.
    """
    print("\n=== Within-Dataset Learning (Optuna) ===")
    results = []

    for idx, name in enumerate(dataset_names):
        X, y = microbiome_dfs[idx], target_dfs[idx]

        if len(y.value_counts()) < 2:
            print(f"  Skipping {name} — only one class present")
            continue

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state, stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.25, random_state=random_state, stratify=y_tv,
        )

        print(f"  {name}")
        metrics = train_lightgbm_optuna(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=n_trials, param_space="within", random_state=random_state,
        )

        results.append({'test_dataset': name, **metrics})
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)
