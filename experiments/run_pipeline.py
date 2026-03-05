import pandas as pd
from pathlib import Path

from experiments.run_protocols import run_protocol_benchmark
from experiments.run_protocols_global_processing import run_protocol_benchmark_global_preprocessing
from evaluation.aggregation import aggregate_protocol_performance
from evaluation.statistics import compare_protocols_mann_whitney

from figures.protocol_comparison_heatmap import plot_protocol_heatmap
from figures.protocol_comparison_boxplots import plot_protocol_boxplots
from figures.papers_vs_lgbm_lodo import plot_auc_horizontal_bars_mann_whitney
from figures.phenotype_grid import plot_figure_1a

from pathlib import Path


# ================================
# CONFIGURATION
# ================================
CONFIG = {

    # -----------------------
    # Pipeline control
    # -----------------------
    "run_compute": True,      # run heavy protocol training
    "run_aggregate": True,    # recompute summary
    "run_stats": True,
    "run_figures": ["1a"], # choose which figures to run

    # -----------------------
    # Experiment config
    # -----------------------
    "preprocessing_scope": "global",   # "local" or "global"
    "normalization": True,
    "decomposition": False,
    "decompose_method": "PCA",
    "decompose_rank": 300,      
    "min_samples_per_dataset": 550, 
    "stability_percentile_local": 0.3,
    "stability_percentile_global": 0.5,
    "z_thresh": 3.0,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_results_dir(project_root: Path, config: dict) -> Path:
    scope = config["preprocessing_scope"]

    if not config["normalization"]:
        mode = "original"
    elif config["normalization"] and not config["decomposition"]:
        mode = "normalized"
    else:
        mode = "dim"

    return project_root / "experiments" / f"results_{scope}_{mode}"



FIGURES_DIR = Path(f"figures_out")
FIGURES_DIR.mkdir(exist_ok=True)


def main():

    RESULTS_DIR = build_results_dir(PROJECT_ROOT, CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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



    # # ================================
    # # STEP 1: Compute or load results
    # # ================================

    results_path = RESULTS_DIR / "protocol_results_per_dataset.csv"

    if CONFIG["run_compute"]:

        if CONFIG["preprocessing_scope"] == "local":
            runner = run_protocol_benchmark
        else:
            runner = run_protocol_benchmark_global_preprocessing

        if results_path.exists():
            print("Loading existing protocol results...")
            results_df = pd.read_csv(results_path)
        else:
            print("Running protocol benchmark...")
            results_df = runner(
                phenotypes=phenotypes,
                apply_normalization=CONFIG["normalization"],
                apply_decompose=CONFIG["decomposition"],
                decompose_method=CONFIG.get("decompose_method"),
                decompose_rank=CONFIG.get("decompose_rank"),
                min_samples_per_dataset=CONFIG.get("min_samples_per_dataset"),
                stability_percentile_local=CONFIG.get("stability_percentile_local"),
                stability_percentile_global=CONFIG.get("stability_percentile_global"),
                z_thresh=CONFIG.get("z_thresh"),
            )
            results_df.to_csv(results_path, index=False)


    # ================================
    # STEP 2: Aggregate
    # ================================

    summary_path = RESULTS_DIR / "protocol_summary.csv"

    if CONFIG["run_aggregate"]:
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = aggregate_protocol_performance(results_df)
            summary_df.to_csv(summary_path, index=False)

    # ================================
    # STEP 3: Statistics
    # ================================
    if CONFIG[ "run_stats"]:
        stats_df = compare_protocols_mann_whitney(results_df)
        stats_df.to_csv(RESULTS_DIR / "protocol_stats.csv", index=False)

    # ================================
    # STEP 4: Figures
    # ================================
    if "1a" in CONFIG["run_figures"]:
        fig, ax = plot_figure_1a()
        fig.savefig(FIGURES_DIR / "paper_phenotype_grid.png", dpi=300, bbox_inches='tight')

    if "1d" in CONFIG["run_figures"]:
        summary_df = pd.read_csv(summary_path)
        fig, ax = plot_protocol_heatmap(summary_df)
        fig.savefig(FIGURES_DIR / "protocol_heatmap.png", dpi=300)

    if "1e" in CONFIG["run_figures"]:
        results_df = pd.read_csv(results_path)
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
