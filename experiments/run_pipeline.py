import pandas as pd
from pathlib import Path

from experiments.run_protocols import run_protocol_benchmark
from experiments.run_protocols_global_processing import run_protocol_benchmark_global_preprocessing
from evaluation.aggregation import aggregate_protocol_performance
from evaluation.statistics import compare_protocols_mann_whitney

from figures.protocol_comparison_heatmap import plot_protocol_heatmap
from figures.protocol_comparison_boxplots import plot_protocol_boxplots
from figures.papers_vs_lgbm_lodo import plot_auc_horizontal_bars_mann_whitney

RESULTS_DIR = Path(f"results")
FIGURES_DIR = Path(f"figures_out")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():

    phenotypes = [
        ("AD", "Amplicon"),
        ("ASD", "Metagenomics"),
        ("ASD", "Amplicon"),
        ("CD", "Metagenomics"),
        ("CRC", "Metagenomics"),
        ("Delivery_mode_6month", "Amplicon"),
        ("Delivery_mode_month", "Amplicon"),
        ("Delivery_mode_year", "Amplicon"),
        ("PD", "Amplicon"),
        ("T2D", "Amplicon"),
        ("UC", "Metagenomics"),
    ]

    results_path = RESULTS_DIR / "protocol_results_per_dataset.csv"
    results_path_global = RESULTS_DIR / "protocol_results_per_dataset_global.csv"
    summary_path = RESULTS_DIR / "protocol_summary.csv"

    compute_global = True

    # # ================================
    # # STEP 1: Compute or load results
    # # ================================
    # if results_path.exists():
    #     print("Loading existing protocol results...")
    #     results_df = pd.read_csv(results_path)
    # else:
    #     print("Running protocol benchmark...")
    #     results_df = run_protocol_benchmark(
    #         phenotypes=phenotypes,
    #         apply_normalization=True,
    #         apply_decompose=False,
    #     )
    #     results_df.to_csv(results_path, index=False)

    # ================================
    # STEP 1.1: Compute or load results
    # ================================
    if results_path_global.exists():
        print("Loading existing protocol results...")
        results_df_global = pd.read_csv(results_path_global)
    else:
        print("Running protocol benchmark...")
        results_df_global = run_protocol_benchmark_global_preprocessing(
            phenotypes=phenotypes,
            apply_normalization=True,
            apply_decompose=False,
            min_samples_per_dataset=400,  # 550
            stability_percentile_local=0.3,
            stability_percentile_global=0.5,
            min_dataset_support=5,
            z_thresh=3.0
        )
        results_df_global.to_csv(results_path_global, index=False)

    results_df = results_df_global.copy()

    # ================================
    # STEP 2: Aggregate
    # ================================
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = aggregate_protocol_performance(results_df)
        summary_df.to_csv(summary_path, index=False)

    # ================================
    # STEP 3: Statistics
    # ================================
    stats_df = compare_protocols_mann_whitney(results_df)
    stats_df.to_csv(RESULTS_DIR / "protocol_stats.csv", index=False)

    # ================================
    # STEP 4: Figures
    # ================================
    fig, ax = plot_protocol_heatmap(summary_df)
    fig.savefig(FIGURES_DIR / "protocol_heatmap.png", dpi=300)

    fig, ax = plot_protocol_boxplots(results_df)
    fig.savefig(FIGURES_DIR / "protocol_boxplots.png", dpi=300)
    #
    # # ================================
    # # STEP 5: Papers vs LightGBM
    # # ================================
    # df_papers = pd.read_csv("Data/Phenotype_Datasets_for_table.csv")
    #
    # fig, ax, stats = plot_auc_horizontal_bars_mann_whitney(
    #     df_papers=df_papers,
    #     df_lightGBM=results_df,
    # )
    # fig.savefig(FIGURES_DIR / "papers_vs_lgbm_lodo.png", dpi=300)
    # stats.to_csv(RESULTS_DIR / "papers_vs_lgbm_stats.csv", index=False)


if __name__ == "__main__":
    main()
