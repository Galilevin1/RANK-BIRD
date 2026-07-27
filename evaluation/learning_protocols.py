import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from .data_loading import combine_datasets
from .lgbm_model import train_lightgbm


def lodo_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                  dataset_names: List[str], lambda1: float = None, lambda2: float = None) -> pd.DataFrame:
    """Step 1: Basic LODO learning."""
    print("\n=== Step 1: LODO Learning ===")
    results = []

    for test_idx in range(len(dataset_names)):
        train_indices = [i for i in range(len(dataset_names)) if i != test_idx]

        X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
        X_test, y_test = microbiome_dfs[test_idx], target_dfs[test_idx]

        common_features = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common_features], X_test[common_features]

        print(f"Training on {[dataset_names[i] for i in train_indices]}, testing on {dataset_names[test_idx]}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test, lambda1=lambda1, lambda2=lambda2)

        results.append({
            'step': 'Step1_LODO',
            'test_dataset': dataset_names[test_idx],
            'train_datasets': ','.join([dataset_names[i] for i in train_indices]),
            **metrics
        })
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def internal_validation_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                                  dataset_names: List[str], lambda1: float = None, lambda2: float = None) -> pd.DataFrame:
    """Step 2: Internal validation on training combinations."""
    print("\n=== Step 2: Internal Validation ===")
    results = []

    for test_idx in range(len(dataset_names)):
        train_indices = [i for i in range(len(dataset_names)) if i != test_idx]

        X_combined, y_combined = combine_datasets(microbiome_dfs, target_dfs, train_indices)

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        train_dataset_names = [dataset_names[i] for i in train_indices]
        print(f"Internal validation on {train_dataset_names}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test, lambda1=lambda1, lambda2=lambda2)

        results.append({
            'step': 'Step2_Internal',
            'test_dataset': dataset_names[test_idx],
            'train_datasets': ','.join(train_dataset_names),
            **metrics
        })
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)


def within_dataset_protocol(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                             dataset_names: List[str], lambda1: float = None, lambda2: float = None) -> pd.DataFrame:
    """Step 4: Within-dataset learning."""
    print("\n=== Step 4: Within-Dataset Learning ===")
    results = []

    for idx, dataset_name in enumerate(dataset_names):
        X, y = microbiome_dfs[idx], target_dfs[idx]

        if len(y.value_counts()) < 2:
            print(f"Skipping {dataset_name} - only one class present")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Within-dataset learning for {dataset_name}")

        metrics = train_lightgbm(X_train, y_train, X_test, y_test, lambda1=lambda1, lambda2=lambda2)

        results.append({
            'step': 'Step4_Within',
            'test_dataset': dataset_name,
            'train_datasets': dataset_name,
            **metrics
        })
        print(f"  AUC: {metrics['auc']:.4f}")

    return pd.DataFrame(results)
