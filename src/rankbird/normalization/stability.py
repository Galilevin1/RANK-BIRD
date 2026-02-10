import numpy as np
import pandas as pd
from typing import List

def union_microbes(microbiome_dfs: List[pd.DataFrame]) -> List[str]:
    cols = set()
    for df in microbiome_dfs:
        cols |= set(df.columns)
    return sorted(cols)


def nonzero_percent_by_dataset(microbiome_dfs, dataset_names, all_microbes):
    rows = []
    for df in microbiome_dfs:
        X = df.reindex(columns=all_microbes, fill_value=0.0)
        frac_nonzero = (X != 0).mean(axis=0)
        rows.append(frac_nonzero)
    return pd.DataFrame(rows, index=dataset_names, columns=all_microbes)


def auto_stability_filter(nz_df, percentile=0.7):
    means = nz_df.mean(axis=0)
    stds  = nz_df.std(axis=0, ddof=1)
    ratio = stds / means.replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    thresh = np.nanpercentile(ratio, percentile*100)
    keep = ratio[ratio <= thresh].dropna().index
    print(f"[AUTO] Using stability threshold = {thresh:.4f} ({int(percentile*100)}th pct)")
    print(f"[AUTO] Kept {len(keep)} microbes out of {len(ratio)}")
    return list(keep)


def filter_microbes_by_dataset_support(
    microbiome_dfs,
    min_datasets: int
):
    presence = {}
    for df in microbiome_dfs:
        for m in df.columns:
            if (df[m] != 0).any():
                presence[m] = presence.get(m, 0) + 1

    return [m for m, c in presence.items() if c >= min_datasets]
