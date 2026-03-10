import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from .lgbm_model import train_lightgbm_with_shap
from .data_loading import combine_datasets


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _feature_overlap(features1: set, features2: set) -> dict:
    common = features1 & features2
    total = len(features1 | features2)
    return {
        'common_features': len(common),
        'unique_to_first': len(features1 - features2),
        'unique_to_second': len(features2 - features1),
        'overlap_percentage': (len(common) / total * 100) if total > 0 else 0.0,
    }


def _save_shap_details(lodo_shap: dict, path: Path):
    with open(path, 'w') as f:
        for test_name, info in lodo_shap.items():
            f.write(f"Test Dataset: {test_name}\n")
            for _, row in info['top_10_features'].iterrows():
                f.write(f"  {row['feature']}: {row['shap_importance']:.6f}\n")
            f.write("\n")


def _load_shap_details(path: Path) -> dict:
    lodo_shap = {}
    current_dataset = None
    features, importances = [], []

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Test Dataset:"):
                if current_dataset and features:
                    lodo_shap[current_dataset] = {
                        'top_10_features': pd.DataFrame(
                            {'feature': features, 'shap_importance': importances}
                        ),
                        'test_dataset': current_dataset,
                    }
                current_dataset = line.split("Test Dataset:")[1].strip()
                features, importances = [], []
            elif line.strip() and current_dataset and ":" in line:
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    features.append(parts[0].strip())
                    importances.append(float(parts[1].strip()))

    if current_dataset and features:
        lodo_shap[current_dataset] = {
            'top_10_features': pd.DataFrame(
                {'feature': features, 'shap_importance': importances}
            ),
            'test_dataset': current_dataset,
        }

    return lodo_shap


# ─────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────

def pairwise_lodo_with_shap(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train on each dataset A, test on each dataset B (A ≠ B).
    Returns (results_df, shap_info) where shap_info maps
    '{train}_to_{test}' -> {'train_dataset', 'test_dataset', 'top_10_features'}.
    """
    print("\n=== Pairwise LODO with SHAP ===")
    results = []
    shap_info = {}
    n = len(dataset_names)

    for train_idx in range(n):
        for test_idx in range(n):
            if train_idx == test_idx:
                continue

            train_name = dataset_names[train_idx]
            test_name  = dataset_names[test_idx]
            pair_key   = f"{train_name}_to_{test_name}"

            X_train = microbiome_dfs[train_idx]
            y_train = target_dfs[train_idx]
            X_test  = microbiome_dfs[test_idx]
            y_test  = target_dfs[test_idx]

            common = X_train.columns.intersection(X_test.columns)
            if len(common) == 0:
                print(f"  No common features: {train_name} → {test_name}, skipping")
                continue

            overlap = _feature_overlap(set(X_train.columns), set(X_test.columns))
            print(f"  {train_name} → {test_name} | common: {len(common)} "
                  f"({overlap['overlap_percentage']:.1f}%)")

            try:
                metrics, feat_imp = train_lightgbm_with_shap(
                    X_train[common], y_train, X_test[common], y_test
                )
                shap_info[pair_key] = {
                    'train_dataset': train_name,
                    'test_dataset':  test_name,
                    'top_10_features': feat_imp.head(10),
                }
                results.append({
                    'train_dataset': train_name,
                    'test_dataset':  test_name,
                    'n_common_features': len(common),
                    'overlap_pct': overlap['overlap_percentage'],
                    **metrics,
                })
                print(f"    AUC: {metrics['auc']:.4f}")
            except Exception as e:
                print(f"    Error: {e}")

    return pd.DataFrame(results), shap_info


def full_lodo_with_shap(
    microbiome_dfs: List[pd.DataFrame],
    target_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train on all datasets except one, test on the left-out dataset.
    Returns (results_df, lodo_shap) where lodo_shap maps
    test_dataset_name -> {'top_10_features', 'all_feature_importance', 'test_dataset', 'train_datasets'}.
    """
    print("\n=== Full LODO with SHAP ===")
    results = []
    lodo_shap = {}
    n = len(dataset_names)

    for test_idx in range(n):
        test_name     = dataset_names[test_idx]
        train_indices = [i for i in range(n) if i != test_idx]
        train_names   = [dataset_names[i] for i in train_indices]

        X_train, y_train = combine_datasets(microbiome_dfs, target_dfs, train_indices)
        X_test  = microbiome_dfs[test_idx]
        y_test  = target_dfs[test_idx]

        common = X_train.columns.intersection(X_test.columns)
        print(f"  test={test_name} | train={train_names} | common: {len(common)}")

        try:
            metrics, feat_imp = train_lightgbm_with_shap(
                X_train[common], y_train, X_test[common], y_test
            )
            lodo_shap[test_name] = {
                'top_10_features':      feat_imp.head(10),
                'all_feature_importance': feat_imp,
                'test_dataset':  test_name,
                'train_datasets': train_names,
            }
            results.append({
                'test_dataset':  test_name,
                'train_datasets': ','.join(train_names),
                'n_common_features': len(common),
                **metrics,
            })
            print(f"    AUC: {metrics['auc']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    return pd.DataFrame(results), lodo_shap


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

def run_figure4_analysis(
    phenotype: str,
    data_root: str,
    output_dir: str,
    load_function,
) -> Dict:
    """
    Run figure 4 analysis for one phenotype.
    Loads from disk if results already exist, otherwise computes and saves.

    Returns dict with keys:
        phenotype, dataset_names, pairwise_results, full_lodo_results, full_lodo_shap
    Returns None if phenotype data is missing or has fewer than 2 datasets.
    """
    phenotype_safe = phenotype.replace(" ", "_").replace("/", "_")
    out = Path(output_dir) / phenotype_safe
    out.mkdir(parents=True, exist_ok=True)

    pairwise_path  = out / "pairwise_lodo_results.csv"
    full_lodo_path = out / "full_lodo_results.csv"
    shap_path      = out / "shap_details.txt"
    names_path     = out / "dataset_names.txt"

    # ── Load from disk if all artefacts exist ──────────────────────────────
    if all(p.exists() for p in [pairwise_path, full_lodo_path, shap_path, names_path]):
        print(f"Loading existing figure 4 results for {phenotype}")
        return {
            'phenotype':         phenotype,
            'dataset_names':     names_path.read_text().strip().splitlines(),
            'pairwise_results':  pd.read_csv(pairwise_path),
            'full_lodo_results': pd.read_csv(full_lodo_path),
            'full_lodo_shap':    _load_shap_details(shap_path),
        }

    # ── Compute ────────────────────────────────────────────────────────────
    folder = f"{data_root}/{phenotype}"
    if not Path(folder).exists():
        print(f"Skipping {phenotype}: folder not found ({folder})")
        return None

    microbiome_dfs, target_dfs, dataset_names = load_function(folder)
    if len(dataset_names) < 2:
        print(f"Skipping {phenotype}: need ≥2 datasets, found {len(dataset_names)}")
        return None

    print(f"\n{'='*60}\nFigure 4 analysis: {phenotype}\nDatasets: {dataset_names}\n{'='*60}")

    pairwise_results, _          = pairwise_lodo_with_shap(microbiome_dfs, target_dfs, dataset_names)
    full_lodo_results, lodo_shap = full_lodo_with_shap(microbiome_dfs, target_dfs, dataset_names)

    # ── Save artefacts ─────────────────────────────────────────────────────
    pairwise_results.to_csv(pairwise_path, index=False)
    full_lodo_results.to_csv(full_lodo_path, index=False)
    _save_shap_details(lodo_shap, shap_path)
    names_path.write_text("\n".join(dataset_names))

    return {
        'phenotype':         phenotype,
        'dataset_names':     dataset_names,
        'pairwise_results':  pairwise_results,
        'full_lodo_results': full_lodo_results,
        'full_lodo_shap':    lodo_shap,
    }
