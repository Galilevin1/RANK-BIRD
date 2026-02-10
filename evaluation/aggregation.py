import pandas as pd
from scipy.stats import sem

def aggregate_protocol_performance(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± SEM per phenotype × protocol.
    """

    return (
        results_df
        .groupby(["phenotype", "protocol"])
        .agg(
            mean_auc=("auc", "mean"),
            sem_auc=("auc", sem),
        )
        .reset_index()
    )
