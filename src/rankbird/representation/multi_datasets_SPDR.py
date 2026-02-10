import numpy as np
import pandas as pd
from .wrappers import initial_components

def bias_SPDR(V_list, alpha, max_iter=100, tol=1e-6):
    """
    Decomposes a list of datasets into modes and dataset-specific vectors.

    Parameters:
        V_list (list of np.ndarray): List of datasets (samples x features for each dataset).
        alpha (np.ndarray): Initial Coefficients (modes x features).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        U (list of np.ndarray): list of datasets after decomposition (samples x modes for each dataset).
        eta (np.ndarray): Dataset-specific vectors (features x datasets).
        beta (list of np.ndarray): Dataset-specific coefficients (samples for each dataset).

    """
    # Initialize
    n_datasets = len(V_list)
    n_modes = alpha.shape[0]
    n_features = alpha.shape[1]

    # Initialize U (list of samples x modes arrays)
    U = [np.zeros((V.shape[0], n_modes)) for V in V_list]

    # Initialize beta (list of samples arrays)
    beta = [np.zeros(V.shape[0]) for V in V_list]

    # Initialize eta (features x datasets)
    eta = np.array([np.mean(V, axis=0) for V in V_list]).T  # Shape (features x datasets)
    eta /= np.linalg.norm(eta, axis=0, keepdims=True)  # Normalize eta

    for iteration in range(max_iter):

        max_change = 0

        # Iterate over datasets
        for j, V in enumerate(V_list):
            current_samples = V.shape[0]  # Number of samples in dataset j

            # Compute U and beta for the current dataset
            for i in range(current_samples):
                residual = V[i] - (U[j][i, :] @ alpha)  # Matrix multiplication
                beta[j][i] = np.dot(residual.T, eta[:, j])
                # Update U
                for k in range(n_modes):
                    U[j][i, k] = V[i].T @ alpha[k,:] - beta[j][i] * (eta[:, j].T @ alpha[k,:].T)

            # Update eta_j for the current dataset
            eta_j_tilde = np.sum(
                [V[i] - (U[j][i, :] @ alpha) for i in range(current_samples)],
                axis=0,
            )
            new_eta_j = eta_j_tilde / np.linalg.norm(eta_j_tilde)
            max_change = max(max_change, np.linalg.norm(new_eta_j - eta[:, j]))
            eta[:, j] = new_eta_j

        # # Check convergence
        # if max_change < tol:
        #     print(f"Converged in {iteration + 1} iterations.")
        #     break

    return U, eta, beta

def apply_bias_SPDR(microbiome_dfs, method, rank=None):

    V_list = [df.to_numpy() for df in microbiome_dfs]
    _, initial_components_combined, _, rank = initial_components(method, pd.concat(microbiome_dfs), rank)
    initial_components_combined = initial_components_combined.to_numpy()
    U_all, eta_all, beta_all = bias_SPDR(V_list, initial_components_combined, max_iter=10, tol=1e-6)
    U_all = [pd.DataFrame(np.squeeze(U)) for U in U_all]

    return U_all, eta_all, beta_all