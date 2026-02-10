import pandas as pd
import matplotlib as plt
import seaborn as sns

def plot_protocol_heatmap(summary_df: pd.DataFrame):

    pivot = summary_df.pivot(
        index="phenotype",
        columns="protocol",
        values="mean_auc"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mean AUC"},
        annot_kws={"size": 18},
        ax=ax
    )

    # Axis labels
    ax.set_ylabel("Phenotype", fontsize=20, fontweight="bold")
    ax.set_xlabel("Protocol", fontsize=18, fontweight="bold")

    # Tick styling
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

    # Colorbar styling
    ax.figure.axes[-1].yaxis.label.set_size(14)
    ax.figure.axes[-1].tick_params(labelsize=16)

    # Horizontal separators
    for y in range(1, pivot.shape[0]):
        ax.axhline(y, color="black", linewidth=2)

    plt.tight_layout()
    return fig, ax
