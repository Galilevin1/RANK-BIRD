import numpy as np
import pandas as pd


def _filter_genus_downstream(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns at genus level or below (contain g__)."""
    cols = [c for c in df.columns if "g__" in c]
    return df[cols] if cols else df.iloc[:, :0]  # empty df if none


def _shannon_entropy(row: np.ndarray) -> float:
    """Shannon entropy of a single sample (relative abundances)."""
    total = row.sum()
    if total == 0:
        return 0.0
    p = row / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _simpson_index(row: np.ndarray) -> float:
    """Simpson diversity index (1 - D) of a single sample."""
    total = row.sum()
    if total == 0:
        return 0.0
    p = row / total
    return float(1.0 - np.sum(p ** 2))


def compute_dataset_quality(
    microbiome_dfs: list,
    dataset_names: list,
) -> pd.DataFrame:
    """
    Compute per-dataset quality metrics restricted to genus-level and downstream features.

    Metrics
    -------
    n_samples           : number of samples in the dataset
    n_unique_microbes   : number of unique genus+ features (columns with g__)
    mean_total_reads    : average sum of all genus+ abundances per sample
    mean_shannon        : average Shannon entropy across samples
    mean_simpson        : average Simpson diversity index (1 - D) across samples

    Parameters
    ----------
    microbiome_dfs  : list of DataFrames (samples × features)
    dataset_names   : list of dataset name strings, same order as microbiome_dfs

    Returns
    -------
    DataFrame with one row per dataset.
    """
    rows = []
    for df, name in zip(microbiome_dfs, dataset_names):
        g_df = _filter_genus_downstream(df)

        n_samples = len(df)
        n_unique  = g_df.shape[1]

        if n_unique == 0:
            rows.append({
                "dataset":           name,
                "n_samples":         n_samples,
                "n_unique_microbes": 0,
                "mean_total_reads":  np.nan,
                "mean_shannon":      np.nan,
                "mean_simpson":      np.nan,
            })
            continue

        values = g_df.values.astype(float)

        mean_total_reads = float(values.sum(axis=1).mean())
        mean_shannon     = float(np.mean([_shannon_entropy(row) for row in values]))
        mean_simpson     = float(np.mean([_simpson_index(row) for row in values]))

        rows.append({
            "dataset":           name,
            "n_samples":         n_samples,
            "n_unique_microbes": n_unique,
            "mean_total_reads":  mean_total_reads,
            "mean_shannon":      mean_shannon,
            "mean_simpson":      mean_simpson,
        })

    return pd.DataFrame(rows)


def flag_quality(
    quality_df: pd.DataFrame,
    min_samples: int = 50,
    min_shannon: float = 1.0,
    min_simpson: float = 0.5,
) -> pd.DataFrame:
    """
    Add a 'recommendation' column ('keep' / 'drop') based on quality thresholds.
    A dataset is flagged 'drop' if it fails ANY criterion.
    """
    df = quality_df.copy()
    drop_mask = (
        (df["n_samples"]    < min_samples) |
        (df["mean_shannon"] < min_shannon) |
        (df["mean_simpson"] < min_simpson)
    )
    df["recommendation"] = np.where(drop_mask, "drop", "keep")
    return df


def run_quality_report(
    phenotypes: list,
    data_root,
    load_fn,
    output_path=None,
    min_samples: int = 50,
    min_shannon: float = 1.0,
    min_simpson: float = 0.5,
) -> pd.DataFrame:
    """
    Load all datasets for the given phenotypes and compute quality metrics.

    Parameters
    ----------
    phenotypes  : list of (phenotype, dtype) tuples
    data_root   : Path to the Data/ folder
    load_fn     : load_microbiome_datasets_with_targets function
    output_path : optional Path/str — if provided, saves the CSV there

    Returns
    -------
    DataFrame with quality metrics for every dataset across all phenotypes.
    """
    from pathlib import Path
    data_root = Path(data_root)

    all_dfs, all_names, all_phenotypes = [], [], []

    for phenotype, dtype in phenotypes:
        folder = data_root / f"{phenotype} {dtype}"
        if not folder.exists():
            print(f"  [SKIP] Folder not found: {folder}")
            continue
        dfs, _, names = load_fn(folder)
        all_dfs.extend(dfs)
        all_names.extend(names)
        all_phenotypes.extend([f"{phenotype} {dtype}"] * len(names))

    quality_df = compute_dataset_quality(all_dfs, all_names)
    quality_df.insert(1, "phenotype", all_phenotypes)
    quality_df = flag_quality(quality_df, min_samples=min_samples, min_shannon=min_shannon, min_simpson=min_simpson)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        quality_df.to_csv(output_path, index=False)
        print(f"Quality report saved to: {output_path}")
        drops = quality_df[quality_df["recommendation"] == "drop"]
        if not drops.empty:
            print(f"  Recommended to drop ({len(drops)} datasets):")
            for _, row in drops.iterrows():
                print(f"    - {row['dataset']} ({row['phenotype']})")

    return quality_df
