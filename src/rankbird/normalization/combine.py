import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1) OVERSAMPLING
# ---------------------------------------------------------------

def oversample_to_min_size(X, min_size=300, random_state=42):
    n = len(X)
    if n >= min_size:
        return X
    need = min_size - n
    rng = np.random.default_rng(random_state)
    add_idx = rng.integers(0, n, size=need)
    return pd.concat([X, X.iloc[add_idx]], axis=0)


# ---------------------------------------------------------------
# 2) COMBINED MATRIX CONSTRUCTION
# ---------------------------------------------------------------

def build_combined_matrix(
    microbiome_dfs,
    dataset_names,
    kept_microbes,
    kept_safe,
    orig2safe,
    min_size=300,
    random_state=42
):
    X_list, ds_list = [], []
    orig_indices_map = {}

    for df, name in zip(microbiome_dfs, dataset_names):
        orig_indices_map[name] = df.index.tolist()

        Xi = df.reindex(columns=kept_microbes, fill_value=0.0)
        Xi.columns = [orig2safe[c] for c in Xi.columns]
        Xi = Xi.reindex(columns=kept_safe)

        # oversampled version used only for mean estimates

        Xi = oversample_to_min_size(Xi, min_size=min_size, random_state=random_state)

        Xi = Xi.copy()
        Xi.index = [f"{name}__{i:06d}" for i in range(len(Xi))]

        X_list.append(Xi)
        ds_list.append(pd.Series(name, index=Xi.index, name="dataset"))

    X_all = pd.concat(X_list, axis=0)
    ds_all = pd.concat(ds_list, axis=0)

    return X_all, ds_all, orig_indices_map