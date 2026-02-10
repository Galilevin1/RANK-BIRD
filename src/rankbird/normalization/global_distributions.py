import numpy as np

# ---------------------------------------------------------------
# 3) **PER-MICROBE OUTLIER DETECTION + MEAN DISTRIBUTION**
# ---------------------------------------------------------------

def compute_per_microbe_global_distributions(
    X_all,
    ds_all,
    dataset_names,
    kept_safe,
    z_thresh=3.0,
):
    """
    For EACH microbe:
      1) compute initial mean distribution across datasets
      2) detect outlier datasets for THIS microbe
      3) compute refined mean after removing its outliers
    """

    global_sorted_dict = {}
    microbe_kept_datasets = {}

    M = None  # samples per dataset after oversampling

    for microbe in kept_safe:

        # --------------------------------------------------
        # Collect curves per dataset
        # --------------------------------------------------
        curves = {}
        for d in dataset_names:
            Xi = X_all.loc[ds_all == d, microbe].values
            max_val = np.max(Xi)
            if max_val > 0:
                Xi = Xi / max_val
            if M is None:
                M = len(Xi)
            curves[d] = np.sort(Xi)[::-1]


        # --------------------------------------------------
        # Initial mean
        # --------------------------------------------------
        initial_mean = np.mean(list(curves.values()), axis=0)

        # --------------------------------------------------
        # Distances per dataset
        # --------------------------------------------------

        from scipy.stats import wasserstein_distance
        from scipy.stats import ks_2samp

        # dist = {d: np.sqrt(np.sum((curves[d] - initial_mean)**2))
        #         for d in dataset_names}
        dist = {d: wasserstein_distance(curves[d], initial_mean)
                for d in dataset_names}
        # dist = {}
        # for d in dataset_names:
        #     D_stat, _ = ks_2samp(curves[d], initial_mean)
        #     dist[d] = float(D_stat)

        dist_vals = np.array(list(dist.values()))
        mu, sigma = dist_vals.mean(), dist_vals.std() if dist_vals.std() > 0 else 1.0

        z_scores = {d: (dist[d] - mu) / sigma for d in dataset_names}

        kept_d = [d for d in dataset_names if z_scores[d] <= z_thresh]
        microbe_kept_datasets[microbe] = kept_d

        # --------------------------------------------------
        # Refined mean for THIS microbe
        # --------------------------------------------------
        final_curves = [curves[d] for d in kept_d]
        refined_mean = np.mean(final_curves, axis=0)

        global_sorted_dict[microbe] = refined_mean

    return global_sorted_dict, microbe_kept_datasets
