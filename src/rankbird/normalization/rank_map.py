import numpy as np

# ---------------------------------------------------------------
# 4) QUANTILE NORMALIZATION USING REFINED DISTRIBUTIONS
# ---------------------------------------------------------------

def rank_map_normalize_dataset(X_orig, global_sorted_dict):
    X_norm = X_orig.copy()

    for col in X_orig.columns:
        x = X_orig[col].values
        N = len(x)

        sorted_idx = np.argsort(-x)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(N)

        g = global_sorted_dict[col]
        M = len(g)

        if N > 1:
            global_ranks = np.round(ranks * (M-1) / (N-1)).astype(int)
        else:
            global_ranks = np.zeros(N, dtype=int)

        X_norm[col] = g[global_ranks]

    return X_norm