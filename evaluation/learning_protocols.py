import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from .data_loading import combine_datasets
from .lgbm_model import train_lightgbm, train_lightgbm_optuna

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
