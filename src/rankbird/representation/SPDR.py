import numpy as np
import pandas as pd
import matplotlib as plt
from .SPDR_core import compute_alpha, compute_U, loss_function

def apply_SPDR(Z_initial, n_components=2, max_iterations=20, visualize=False, initial_alpha=None):
    """
    Decompose matrix Z into a new representation of modes:
    - alpha: components matrix of shape (n_features, n_modes)
    - U: a non-negative uniform distributed matrix of shape (n_samples, n_modes).

    Parameters:
        Z (pd.DataFrame): A matrix of shape (n_samples, n_features).
        n_components (int):
        max_iterations (int):
        visualize (bool, optional):
        initial_alpha (pd.DataFrame, optional): A matrix of shape (n_modes, n_features) as an initial components' matrix.


    Returns:
        alpha (ndarray): A matrix of shape (n_modes, n_features).
        U (ndarray): A matrix of shape (n_samples, n_modes).
    """

    Z = Z_initial.to_numpy()

    # Initialize lost saving list
    loss_list = []

    if initial_alpha is not None:
        alpha = initial_alpha.to_numpy()
        alpha = alpha.T # Transpose to shape (n_features, n_modes)
        # Iterative optimization
        for iteration in range(max_iterations):

            # compute U by alpha
            U = compute_U(Z, alpha)

            # compute alpha bu U
            alpha = compute_alpha(Z, U)

            # calculate the loss function
            loss = loss_function(Z, alpha, U)
            loss_list.append(loss)

            # print(f"Iteration {iteration}: Loss = {loss}")

            # Check if loss is minimized (optional stopping criterion)
            if iteration > 1 and abs(prev_loss - loss) < 1e-6:
                print(f"Convergence reached in {iteration} iterations.")
                break
            prev_loss = loss
    else:
        # Initialize U
        U = np.random.rand(n_components, Z.shape[0])
        U /= np.sum(U, axis=0, keepdims=True)
        # Iterative optimization
        for iteration in range(max_iterations):
            # compute alpha bu U
            alpha = compute_alpha(Z, U)

            # compute U by alpha
            U = compute_U(Z, alpha)

            # calculate the loss function
            loss = loss_function(Z, alpha, U)
            loss_list.append(loss)

            # print(f"Iteration {iteration}: Loss = {loss}")

            # Check if loss is minimized (optional stopping criterion)
            if iteration > 1 and abs(prev_loss - loss) < 1e-6:
                print(f"Convergence reached in {iteration} iterations.")
                break
            prev_loss = loss

    # Visualize loss
    if visualize:
        iterations = np.arange(1, len(loss_list) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, loss_list, label="Loss", color="black")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss vs. Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()

    return U.T, alpha.T