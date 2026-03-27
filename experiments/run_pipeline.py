import pandas as pd
from pathlib import Path

from experiments.run_protocols import run_protocol_benchmark
from experiments.run_protocols_global_processing import run_protocol_benchmark_global_preprocessing
from evaluation.aggregation import aggregate_protocol_performance
from evaluation.statistics import compare_protocols_mann_whitney

import matplotlib.pyplot as plt

from figures.protocol_comparison_heatmap import plot_protocol_heatmap
from figures.protocol_comparison_boxplots import plot_protocol_boxplots
from figures.papers_vs_lgbm_lodo import plot_auc_horizontal_bars_mann_whitney
from figures.phenotype_grid import plot_figure_1a
from figures.figure1 import plot_figure1
from figures.figure2 import plot_figure2b, run_figure2c
from figures.figure3 import run_figure3
from figures.figure4 import plot_figure4
from figures.figure4e import run_figure4e_analysis, plot_figure4e
from figures.figure5 import plot_figure5
from experiments.investigate_stability_threshold import (
    run_stability_investigation, run_microbe_characterization,
)
from experiments.investigate_distribution_approach import run_distribution_investigation
from evaluation.data_loading import build_papers_auc_df, load_microbiome_datasets_with_targets
from evaluation.pairwise_lodo import run_figure4_analysis

from pathlib import Path


# ================================
# CONFIGURATION
# ================================
CONFIG = {

    # -----------------------
    # Pipeline control
    # -----------------------
    "run_compute": False,      # run heavy protocol training
    "run_aggregate": False,    # recompute summary
    "run_stats": False,
    "run_figures": [], # "1" (combined), "1a","1c","1d","1e" (individual), "2b","2c","3","4","4e","5"
    "run_investigations": ["distribution_approach"],  # "stability_threshold", "stability_characterization", "distribution_approach"
    "investigations_plot_only": False,   # For  "stability_threshold" and "distribution_approach"invastigations # True = reload CSVs, False = recompute
    "characterization_threshold_metagenomics": 0.25,
    "characterization_threshold_amplicon":     0.40,

    # -----------------------
    # Papers CSV (for figure 1c)
    # -----------------------
    "papers_csv": "Data/Phenotype_Datasets_for_table.csv",

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
    "stability_percentile_global_metagenomics": 0.25,
    "stability_percentile_global_amplicon":     0.40,
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


def _get_mode(config: dict) -> str:
    if not config["normalization"]:
        return "original"
    elif not config["decomposition"]:
        return "normalized"
    return "dim"


def build_figures_dir(base: Path, config: dict, figure: str) -> Path:
    """Return figures_out/<figure>/<scope>_<mode>/ and create it."""
    scope = config["preprocessing_scope"]
    mode  = _get_mode(config)
    path  = base / figure / f"{scope}_{mode}"
    path.mkdir(parents=True, exist_ok=True)
    return path


FIGURES_BASE = Path("figures_out")
FIGURES_BASE.mkdir(exist_ok=True)


def main():

    RESULTS_DIR = build_results_dir(PROJECT_ROOT, CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    FIG1_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1")
    FIG2_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2")
    FIG3_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_3")
    FIG4_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_4")

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
                stability_percentile_global_amplicon=CONFIG.get("stability_percentile_global_amplicon"),
                stability_percentile_global_metagenomics=CONFIG.get("stability_percentile_global_metagenomics"),
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
    if "1" in CONFIG["run_figures"]:
        results_df  = pd.read_csv(results_path)
        summary_df  = pd.read_csv(summary_path)
        fig, stats = plot_figure1(
            results_df=results_df,
            summary_df=summary_df,
            papers_csv=str(PROJECT_ROOT / CONFIG["papers_csv"]),
            phenotypes=phenotypes,
        )
        fig.savefig(FIG1_DIR / "figure1_combined.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        stats.to_csv(RESULTS_DIR / "papers_vs_lgbm_stats.csv", index=False)

    if "1a" in CONFIG["run_figures"]:
        fig, ax = plot_figure_1a()
        fig.savefig(FIG1_DIR / "paper_phenotype_grid.png", dpi=300, bbox_inches='tight')

    if "1c" in CONFIG["run_figures"]:
        results_df = pd.read_csv(results_path)
        df_papers = build_papers_auc_df(PROJECT_ROOT / CONFIG["papers_csv"])
        fig, ax, stats = plot_auc_horizontal_bars_mann_whitney(
            df_papers=df_papers,
            df_lightGBM=results_df,
            selected_combinations=phenotypes,
            figsize=(12, 16),
            bar_height=1,
            fdr_alpha=0.05,
        )
        fig.savefig(FIG1_DIR / "papers_vs_lgbm_lodo.png", dpi=300, bbox_inches='tight')
        stats.to_csv(RESULTS_DIR / "papers_vs_lgbm_stats.csv", index=False)

    if "1d" in CONFIG["run_figures"]:
        summary_df = pd.read_csv(summary_path)
        fig, ax = plot_protocol_heatmap(summary_df)
        fig.savefig(FIG1_DIR / "protocol_heatmap.png", dpi=300)

    if "1e" in CONFIG["run_figures"]:
        results_df = pd.read_csv(results_path)
        fig, ax = plot_protocol_boxplots(results_df)
        fig.savefig(FIG1_DIR / "protocol_boxplots.png", dpi=300)

    if "2b" in CONFIG["run_figures"]:
        fig = plot_figure2b(csv_path=str(PROJECT_ROOT / "Data" / "microbiome_analysis_results.csv"))
        fig.savefig(FIG2_DIR / "figure2b_confounder_correlations.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    if "2c" in CONFIG["run_figures"]:
        run_figure2c(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / "Data"),
            figures_dir=str(FIG2_DIR),
            apply_normalization=CONFIG["normalization"],
        )

    if "3" in CONFIG["run_figures"]:
        run_figure3(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / "Data"),
            figures_dir=str(FIG3_DIR),
            apply_normalization=CONFIG["normalization"],
        )

    if "4" in CONFIG["run_figures"]:
        figure4_data_dir = RESULTS_DIR / "figure4"
        figure4_data_dir.mkdir(exist_ok=True)

        for pt in phenotypes:
            phenotype = f"{pt[0]} {pt[1]}"
            data = run_figure4_analysis(
                phenotype=phenotype,
                data_root=str(PROJECT_ROOT / "Data"),
                output_dir=str(figure4_data_dir),
                load_function=load_microbiome_datasets_with_targets,
            )
            if data is None:
                continue
            fig = plot_figure4(
                phenotype_name=phenotype,
                pairwise_results=data['pairwise_results'],
                full_lodo_shap=data['full_lodo_shap'],
                dataset_names=data['dataset_names'],
            )
            fig.savefig(
                FIG4_DIR / f"figure4_{phenotype.replace(' ', '_')}.png",
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig)


    if "4e" in CONFIG["run_figures"]:
        figure4_data_dir = RESULTS_DIR / "figure4"
        results_4e = run_figure4e_analysis(
            figure4_data_dir=str(figure4_data_dir),
            metric='auc',
        )
        if results_4e is not None:
            plot_figure4e(results_4e, output_dir=str(FIG4_DIR))

    if "stability_threshold" in CONFIG["run_investigations"]:
        inv_dir = PROJECT_ROOT / "investigations" / "stability_threshold"
        run_stability_investigation(
            phenotypes=phenotypes,
            output_dir=inv_dir,
            plot_only=CONFIG.get("investigations_plot_only", False),
        )

    if "stability_characterization" in CONFIG["run_investigations"]:
        char_dir = PROJECT_ROOT / "investigations" / "stability_characterization"
        run_microbe_characterization(
            phenotypes=phenotypes,
            output_dir=char_dir,
            threshold_metagenomics=CONFIG["characterization_threshold_metagenomics"],
            threshold_amplicon=CONFIG["characterization_threshold_amplicon"],
            plot_only=CONFIG.get("investigations_plot_only", False),
        )

    if "distribution_approach" in CONFIG["run_investigations"]:
        dist_dir = PROJECT_ROOT / "investigations" / "distribution_approach"
        run_distribution_investigation(
            phenotypes=phenotypes,
            output_dir=dist_dir,
            plot_only=CONFIG.get("investigations_plot_only", False),
            stability_percentile_metagenomics=CONFIG.get("stability_percentile_global_metagenomics", 0.25),
            stability_percentile_amplicon=CONFIG.get("stability_percentile_global_amplicon", 0.40),
            min_size=CONFIG.get("min_samples_per_dataset", 550),
        )

    if "5" in CONFIG["run_figures"]:
        FIG5_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_5")
        path_orig = PROJECT_ROOT / "experiments" / "results_global_original" / "protocol_results_per_dataset.csv"
        path_norm = PROJECT_ROOT / "experiments" / "results_global_normalized" / "protocol_results_per_dataset.csv"
        if path_orig.exists() and path_norm.exists():
            fig = plot_figure5(
                results_df_original=pd.read_csv(path_orig),
                results_df_normalized=pd.read_csv(path_norm),
            )
            fig.savefig(FIG5_DIR / "figure5_original_vs_normalized.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            print("Figure 5: missing result files — run pipeline with both global_original and global_normalized first.")


if __name__ == "__main__":
    main()
