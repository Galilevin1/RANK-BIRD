import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------
# 4) QUANTILE NORMALIZATION USING REFINED DISTRIBUTIONS
# ---------------------------------------------------------------

def rank_map_normalize_dataset(X_orig, global_sorted_dict, tie_method: str = "first"):
    """
    Map each dataset's feature values onto the global reference distribution
    via rank-based lookup.

    tie_method : "first"   — ties broken by position (original behaviour)
                 "average" — tied values receive the average of their ranks,
                             so identical values map to the same reference
                             value regardless of row order.
    """
    X_norm = X_orig.copy()

    for col in X_orig.columns:
        x = X_orig[col].values
        N = len(x)

        if tie_method == "average":
            ranks = rankdata(-x, method="average") - 1   # 0-based descending
        else:
            sorted_idx = np.argsort(-x)
            ranks = np.empty(N, dtype=float)
            ranks[sorted_idx] = np.arange(N, dtype=float)

        g = global_sorted_dict[col]
        M = len(g)

        if N > 1:
            global_ranks = np.round(ranks * (M-1) / (N-1)).astype(int)
        else:
            global_ranks = np.zeros(N, dtype=int)

        X_norm[col] = g[global_ranks]

    return X_norm