import pandas as pd
from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from .SPDR import apply_SPDR

def initial_components(method, data, rank, reg_lambda=0.3, initial_alpha=None):

    """
    Apply decomposition methods to a dataset.

    Parameters:
    - method (str): The method to apply decomposition with: 'PCA', 'ICA', 'NMF', 'FactorAnalysis', 'LDA', 'SPDR'
    - data (pd.DataFrame): The original DataFrame to transform.
    - rank (int): The rank for decomposition- the number of features in the output.
    - initial_alpha (pd.DataFrame, optional): A matrix of shape (n_modes, n_features) as an initial components' matrix.

    return:
    - pd.DataFrame(decomposed_data, index=data.index): The transformed DataFrame after decomposition by the chosen method and rank.
    - pd.DataFrame(components, columns=data.columns): The components relates to the decomposed data.
    - model (object): The fitted decomposition model.
    """

    if method == 'PCA':
        #model = PCA(n_components=rank)
        if rank is None:
            temp_pca = PCA(n_components=None)
            temp_pca.fit(data)

            cum_var = temp_pca.explained_variance_ratio_.cumsum()
            rank = (cum_var >= 0.95).argmin() + 1   # index → number of comps

        model = PCA(n_components=rank)
    elif method == 'ICA':
        model = FastICA(n_components=rank)
    elif method == 'NMF':
        model = NMF(n_components=rank)
    elif method == 'FactorAnalysis':
        model = FactorAnalysis(n_components=rank)
    elif method == 'LDA':
        model = LatentDirichletAllocation(n_components=rank)
    elif method == 'SPDR':
        model = 'SPDR'
        decomposed_data, components = apply_SPDR(data, n_components=rank, initial_alpha=initial_alpha)
        return pd.DataFrame(decomposed_data, index=data.index), pd.DataFrame(components, columns=data.columns), model

    decomposed_data = model.fit_transform(data)
    components = model.components_

    return pd.DataFrame(decomposed_data, index=data.index), pd.DataFrame(components, columns=data.columns), model, rank