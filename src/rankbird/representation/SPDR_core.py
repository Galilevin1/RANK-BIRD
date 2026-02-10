import numpy as np

def loss_function(Z, alpha, U):
    """
    Compute reconstruction error of Z and approximate Z = alpha X U.

    Parameters:
        Z (ndarray): A matrix of shape (n_samples, n_features).
        U (ndarray): A matrix of shape (n_modes, n_samples).
        alpha (ndarray): A matrix of shape (n_features, n_modes).

    Returns:
        Error (float): Reconstruction error between Z and approximate Z.

    """

    # Compute reconstructed Z (Z_tilde)
    Z_tilde = alpha @ U

    # Compute element-wise squared differences
    error_matrix = (Z - Z_tilde.T) ** 2  # Transpose Z_tilde to match Z shape

    # Sum all squared differences and scale by 1/2
    return np.log(0.5 * np.sum(error_matrix))

def compute_alpha(Z, U, epsilon = 1e-6):
    """
    Compute alpha given Z and U.

    Parameters:
        Z (ndarray): A matrix of shape (n_samples, n_features).
        U (ndarray): A matrix of shape (n_modes, n_samples).

    Returns:
        alpha (ndarray): A matrix of shape (n_features, n_modes).
    """

    # Compute A (shape: n_features x n_modes)
    A = Z.T @ U.T

    # Compute B (shape: n_modes x n_modes)
    B = U @ U.T #+ epsilon * np.eye(U.shape[0])

    # Inverse of B
    B_inv = np.linalg.inv(B)       # # Psudo invere of B  # B_inv = np.linalg.pinv(B)

    # Compute alpha  (shape: n_features x n_modes)
    alpha = A @ B_inv

    return alpha


def compute_U(Z, alpha, tol=1e-6, max_iter=100):
    """
    Computes U given Z, alpha, and lambda1.

    Parameters:
        Z (ndarray): A matrix of shape (n_samples, n_features).
        alpha (ndarray): A matrix of shape (n_features, n_modes).
        lambda1_init (ndarray, optional): A vector of shape (n_modes,). Initial guess for lambda1.
        tol (float): Tolerance for optimization convergence.
        max_iter (int): Maximum iterations for optimization.

    Returns:
        U (ndarray): A matrix of shape (n_modes, n_samples).
    """
    # Compute C (shape: n_samples x n_modes)
    C = Z @ alpha

    # Compute D (shape: n_modes x n_modes)
    D = alpha.T @ alpha

    # Compute D^-1 (shape: n_modes x n_modes)
    D_inv = np.linalg.pinv(D) # np.linalg.inv(D)

    # Create vector of ones (shape: n_modes x 1)
    ones = np.ones(D.shape[0])
    ones = ones[:, None]  # Reshape to (n_modes, 1)

    # Compute M = 1^T D^(-1) 1
    M = ones.T @ D_inv @ ones
    M_inv = 1 / M

    # Initialize U matrix (n_samples x n_modes)
    U = np.zeros_like(C)

    # Compute U for each sample
    for i in range(C.shape[0]):  # Iterate over samples

        # Get C values of sample i
        C_i = C[i]
        C_i = C_i[:, None]

        # Initial guess for lambda1
        lambda1_init = np.zeros(D_inv.shape[0])

        def calculating_U(lambda1_i, C_i, D_inv, M_inv, ones):
            lambda1_i = lambda1_i[:, None]
            C_i_lambda1_i = C_i - lambda1_i
            U_i = D_inv @ C_i_lambda1_i - M_inv * (ones.T @ D_inv @ C_i_lambda1_i - 1) * D_inv @ ones
            U_i = U_i.squeeze()
            U_i = np.maximum(U_i, 0)
            return U_i

        def calculating_U_without_lambda(C_i, D_inv, M_inv, ones):
            U_i = D_inv @ (C_i - M_inv * (ones.T @ D_inv @ C_i - 1) * ones)
            U_i = U_i.squeeze()
            U_i = np.maximum(U_i, 0)
            return U_i

        def objective_lambda1(lambda1_i):
            """
            Objective function to optimize lambda1 for sample i.
            """
            U_i = calculating_U(lambda1_i, C_i, D_inv, M_inv, ones) # shape: n_modes x 1
            Z_i_tilde = alpha @ U_i # shape: n_features x 1 (n_features x n_modes @ n_modes x 1)
            error = Z[i,:] - Z_i_tilde.T
            return 0.5 * np.sum(error**2)

        def constraint_non_neg_lambda1(lambda1_i):
            """
            Constraint: lambda1_i >= 0
            """
            return lambda1_i

