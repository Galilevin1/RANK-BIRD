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
from figures.figure1 import assemble_figure1
from figures.figure1f import run_figure1f, compute_figure1f
from figures.figure2 import plot_figure2b, run_figure2b, run_figure2d
from figures.figure3 import run_figure3, run_figure3c, run_figure3d
from figures.figure4 import plot_figure4
from figures.figure4e import run_figure4e_analysis, plot_figure4e
from figures.figure5 import plot_figure5
from experiments.investigate_stability_threshold import (
    run_stability_investigation, run_microbe_characterization,
)
from experiments.investigate_distribution_approach import run_distribution_investigation
from evaluation.data_loading import build_papers_auc_df, load_microbiome_datasets_with_targets
from evaluation.dataset_analysis import run_dataset_analysis
from evaluation.pairwise_lodo import run_figure4_analysis
from evaluation.dataset_quality import run_quality_report

from pathlib import Path


# ================================
# CONFIGURATION
# ================================

# Datasets downloaded as-is from published papers — used for fair comparison figures (1a, 1c, 1d, 1e)
phenotypes_papers = [
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

# All datasets from consistent downloading pipeline — used for main pipeline and AUC improvement
phenotypes_pipeline = [
("AD", "Amplicon"),
("ASD", "Metagenomics"),
("ASD", "Amplicon"),
("CD", "Metagenomics"),
("CRC", "Metagenomics"),
("PD", "Metagenomics"),
("PD", "Amplicon"),
("T2D", "Amplicon"),
("UC", "Metagenomics"),
]

# Union of both lists — used when all available datasets are needed (e.g. figure 2a)
_seen = set()
phenotypes_all = []
for _p in phenotypes_pipeline + phenotypes_papers:
    if _p not in _seen:
        _seen.add(_p)
        phenotypes_all.append(_p)



CONFIG = {

    # -----------------------
    # Pipeline control
    # -----------------------
    "run_compute": False,      # run heavy protocol training
    "run_aggregate": False,    # recompute summary
    "run_stats": False,
    "run_figures": ["1"], # "1" (combined), "1a","1c","1d","1e","1f" (individual)
     #                   "2b","2c","3","4","4e","5"
    "run_investigations": [],  # "stability_threshold", "stability_characterization", "distribution_approach"
    "investigations_plot_only": False,   # For  "stability_threshold" and "distribution_approach" invastigations # True = reload CSVs, False = recompute
    "run_quality_report": False,         # compute per-dataset quality metrics (unique microbes, reads, entropy, Simpson)
    "phenotypes": phenotypes_papers,  # phenotypes_pipeline, phenotypes_papers
    "data_folder": "Data_papers",      # "Data" for pipeline phenotypes, "Data_papers" for papers phenotypes
     "preprocessing_scope": "global",   # "local" or "global"
    "normalization_approach": None,  # "rankbird_wasserstein", "rankbird_ranking", "rankbird_sigmoid", "rankbird_relu", "filter_only", None
    "decomposition": False,

    # -----------------------
    # Stability threshold Investigations control
    # -----------------------
    "stability_dtype_filter":        None,  # None = both dtypes; ["Metagenomics"] = shotgun only; ["Amplicon"] = 16S only
    "stability_normalization_modes": ["full+filter_only"],  # "full", "filter_only", "full+filter_only" (both on same plot), or e.g. ["full", "filter_only"] for separate figures
    "stability_plot_mode":           ["mean", "median"],        # "mean", "median", "combined", or list e.g. ["mean", "median"]
    "stability_compute_levels":      [None, "g", "gs"],               # None = compute all levels; e.g. [None, "g"] to compute only those
    "stability_plot_levels":         [None], # None = plot all levels; e.g. ["g", "fg", "ofg"] for a subset

    # -----------------------
    # Papers CSV (for figure 1_supp)
    # -----------------------
    "papers_csv": "Data/Phenotype_Datasets_for_table.csv",

    # -----------------------
    # Experiment config
    # -----------------------
    "min_samples_per_dataset": 550,
    "z_thresh": 3.0,
    "stability_percentile_local": 0.3,
    "stability_percentile_global_metagenomics": 0.1,
    "stability_percentile_global_amplicon":     0.3,
    "taxonomy_level_metagenomics": "gs",    # None = all, "g" = genus only, "gs" = genus+species
    "taxonomy_level_amplicon":     "gs",
    "decompose_method": "PCA",
    "decompose_rank": 300,
    # -----------------------
    # Cross-dtype normalization
    # When True: pool Amplicon + Metagenomics datasets together for stability
    # filtering and distribution normalization (learning stays per phenotype).
    # -----------------------
    "cross_dtype_normalization":           False,
    "stability_percentile_global_combined": 0.3,
    "taxonomy_level_combined":              "gs",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_mode(config: dict) -> str:
    approach = config.get("normalization_approach")
    if not approach or str(approach) == "None":
        approach = "original"
    mode = approach.replace("rankbird_wasserstein", "normalized")  # backwards-compatible name
    if config.get("cross_dtype_normalization"):
        mode += "_combined"
    if config.get("decomposition"):
        mode += "_dim"
    return mode


def build_results_dir(project_root: Path, config: dict) -> Path:
    scope = config["preprocessing_scope"]
    mode  = _get_mode(config)
    return project_root / "experiments" / f"results_{scope}_{mode}"


def build_figures_dir(base: Path, config: dict, figure: str, subfigure: str = None) -> Path:
    """Return figures_out/<figure>/[<subfigure>/]<scope>_<mode>/ and create it."""
    scope = config["preprocessing_scope"]
    mode  = _get_mode(config)
    if subfigure:
        path = base / figure / subfigure / f"{scope}_{mode}"
    else:
        path = base / figure / f"{scope}_{mode}"
    path.mkdir(parents=True, exist_ok=True)
    return path


FIGURES_BASE = Path("figures_out")
FIGURES_BASE.mkdir(exist_ok=True)


def main():

    RESULTS_DIR = build_results_dir(PROJECT_ROOT, CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    FIG3_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_3")
    FIG4_DIR = build_figures_dir(FIGURES_BASE, CONFIG, "figure_4")

    phenotypes = CONFIG["phenotypes"]

    # Normalise run_figures to a set of strings so that e.g. "1f" never
    # accidentally matches the "1" check via Python substring semantics.
    _rf = CONFIG["run_figures"]
    RUN_FIGS: set = set(_rf) if isinstance(_rf, (list, set, tuple)) else {_rf}



    # ================================
    # STEP 0: Dataset quality report
    # ================================
    if CONFIG.get("run_quality_report"):
        run_quality_report(
            phenotypes=phenotypes,
            data_root=PROJECT_ROOT / "Data",
            load_fn=load_microbiome_datasets_with_targets,
            output_path=RESULTS_DIR / "dataset_quality.csv",
        )

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
                normalization_approach=CONFIG.get("normalization_approach"),
                apply_decompose=CONFIG["decomposition"],
                decompose_method=CONFIG.get("decompose_method"),
                decompose_rank=CONFIG.get("decompose_rank"),
                min_samples_per_dataset=CONFIG.get("min_samples_per_dataset"),
                stability_percentile_local=CONFIG.get("stability_percentile_local"),
                stability_percentile_global_amplicon=CONFIG.get("stability_percentile_global_amplicon"),
                stability_percentile_global_metagenomics=CONFIG.get("stability_percentile_global_metagenomics"),
                z_thresh=CONFIG.get("z_thresh"),
                taxonomy_level_metagenomics=CONFIG.get("taxonomy_level_metagenomics"),
                taxonomy_level_amplicon=CONFIG.get("taxonomy_level_amplicon"),
                cross_dtype_normalization=CONFIG.get("cross_dtype_normalization", False),
                stability_percentile_global_combined=CONFIG.get("stability_percentile_global_combined", 0.6),
                taxonomy_level_combined=CONFIG.get("taxonomy_level_combined"),
                data_root=CONFIG.get("data_folder", "Data"),
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
    if "1" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1")
        _summary_df = pd.read_csv(summary_path)
        _results_df = pd.read_csv(results_path)
        _df_papers  = build_papers_auc_df(PROJECT_ROOT / CONFIG["papers_csv"])
        # load or compute 1f results
        _1f_cache = FIGURES_BASE / "figure_1" / "1f" / "figure1f_results.csv"
        if _1f_cache.exists():
            _1f_df = pd.read_csv(_1f_cache)
        else:
            print("  [figure 1f] cache not found — computing now...")
            _1f_df  = compute_figure1f(
                phenotypes=phenotypes,
                data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            )
            _1f_df.to_csv(_1f_cache, index=False)
        fig = assemble_figure1(
            summary_df=_summary_df,
            results_df=_results_df,
            papers_df=_df_papers,
            selected_combinations=phenotypes,
            path_1b=FIGURES_BASE / "figure_1" / "1b" / "schematic.png",
            figure1f_df=_1f_df,
        )
        fig.savefig(_dir / "figure1_combined.pdf", bbox_inches='tight', dpi=300)
        plt.close(fig)

    if "1a" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1a")
        fig, ax = plot_figure_1a()
        fig.savefig(_dir / "paper_phenotype_grid.png", dpi=300, bbox_inches='tight')

    if "1c" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1c")
        _results_df = pd.read_csv(results_path)
        _df_papers  = build_papers_auc_df(PROJECT_ROOT / CONFIG["papers_csv"])
        fig, ax, stats = plot_auc_horizontal_bars_mann_whitney(
            df_papers=_df_papers,
            df_lightGBM=_results_df,
            selected_combinations=phenotypes,
            figsize=(12, 16),
            bar_height=1,
            fdr_alpha=0.05,
        )
        fig.savefig(_dir / "papers_vs_lgbm_lodo.png", dpi=300, bbox_inches='tight')
        stats.to_csv(_dir / "papers_vs_lgbm_stats.csv", index=False)
        plt.close(fig)

    if "1d" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1d")
        _summary_df = pd.read_csv(summary_path)
        _results_df = pd.read_csv(results_path)
        _df_papers  = build_papers_auc_df(PROJECT_ROOT / CONFIG["papers_csv"])
        fig, _, heatmap_df, _, pheno_stats_df = plot_protocol_heatmap(
            _summary_df,
            results_df=_results_df,
            papers_df=_df_papers,
            selected_combinations=phenotypes,
        )
        fig.savefig(_dir / "protocol_heatmap.png", dpi=300)
        heatmap_df.to_csv(_dir / "heatmap_auc.csv", index=False)
        pheno_stats_df.to_csv(_dir / "heatmap_lgbm_vs_papers.csv", index=False)
        plt.close(fig)

    if "1e" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1e")
        _results_df = pd.read_csv(results_path)
        fig, _, stats_df = plot_protocol_boxplots(_results_df)
        fig.savefig(_dir / "protocol_boxplots.png", dpi=300, bbox_inches="tight")
        stats_df.to_csv(_dir / "protocol_pairwise_mannwhitney.csv", index=False)
        plt.close(fig)

    if "1f" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_1", "1f")
        run_figure1f(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
            results_cache_path=str(_dir / "figure1f_results.csv"),
        )

    if "2a" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2a")
        analysis_csv = RESULTS_DIR / "microbiome_analysis_results.csv"
        if CONFIG["run_compute"]:
            all_mb, all_tgt, all_conf, all_names, d2p = [], [], [], [], {}
            seen_datasets = set()
            for phenotype, dtype in phenotypes_all:
                pheno_str = f"{phenotype} {dtype}"
                for data_root in ["Data", "Data_papers"]:
                    folder = PROJECT_ROOT / data_root / pheno_str
                    if not folder.exists():
                        continue
                    mb_dfs, tgt_dfs, ds_names, conf_dfs = load_microbiome_datasets_with_targets(
                        str(folder), include_confounders=True)
                    for mb, tgt, conf, name in zip(mb_dfs, tgt_dfs, conf_dfs, ds_names):
                        if name in seen_datasets:
                            print(f"  Skipping duplicate dataset: {name}")
                            continue
                        seen_datasets.add(name)
                        all_mb.append(mb); all_tgt.append(tgt)
                        all_conf.append(conf); all_names.append(name); d2p[name] = pheno_str
            run_dataset_analysis(
                all_mb, all_tgt, all_names, d2p,
                output_path=str(analysis_csv),
                confounder_dfs=all_conf,
            )
        fig = plot_figure2b(csv_path=str(analysis_csv))
        fig.savefig(_dir / "confounder_correlations.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    if "2d" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2d")
        all_mb2d, all_names2d, d2p2d = [], [], {}
        for phenotype, dtype in phenotypes:
            pheno_str = f"{phenotype} {dtype}"
            folder = PROJECT_ROOT / CONFIG.get("data_folder", "Data") / pheno_str
            if not folder.exists():
                continue
            mb_dfs, _, ds_names = load_microbiome_datasets_with_targets(str(folder))
            for mb, name in zip(mb_dfs, ds_names):
                all_mb2d.append(mb)
                all_names2d.append(name)
                d2p2d[name] = pheno_str
        run_figure2d(all_mb2d, all_names2d, d2p2d, figures_dir=str(_dir))

    if "2b" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2b")
        run_figure2b(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

    if "2c" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2c")
        run_figure3(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

    if "3" in RUN_FIGS:
        run_figure3(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / "Data"),
            figures_dir=str(FIG3_DIR),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

    if "3b" in RUN_FIGS:
        _3b_results_dir = FIGURES_BASE / "figure_3" / "3b"
        run_stability_investigation(
            phenotypes=phenotypes,
            output_dir=_3b_results_dir,
            plot_only=True,
            dtype_filter=CONFIG.get("stability_dtype_filter", None),
            normalization_modes=["filter_only"],
            plot_mode=CONFIG.get("stability_plot_mode", "combined"),
            plot_levels=CONFIG.get("stability_plot_levels", None),
            compute_levels=CONFIG.get("stability_compute_levels", None),
            cross_dtype_normalization=CONFIG.get("cross_dtype_normalization", False),
            stability_percentile_global_combined=CONFIG.get("stability_percentile_global_combined", 0.6),
            taxonomy_level_combined=CONFIG.get("taxonomy_level_combined"),
        )

    if "3c" in RUN_FIGS:
        _3c_data_dir = FIGURES_BASE / "figure_3" / "3c"
        _3c_out_dir  = build_figures_dir(FIGURES_BASE, CONFIG, "figure_3", "3c")
        run_figure3c(
            investigation_dir=str(_3c_data_dir),
            figures_dir=str(_3c_out_dir),
        )

    if "3d" in RUN_FIGS:
        _3d_dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_3", "3d")
        run_figure3d(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_3d_dir),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

    if "4" in RUN_FIGS:
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


    if "4e" in RUN_FIGS:
        figure4_data_dir = RESULTS_DIR / "figure4"
        results_4e = run_figure4e_analysis(
            figure4_data_dir=str(figure4_data_dir),
            metric='auc',
        )
        if results_4e is not None:
            plot_figure4e(results_4e, output_dir=str(FIG4_DIR))

    if "stability_threshold" in CONFIG["run_investigations"]:
        _approach = CONFIG.get("normalization_approach") or "original"
        _combined_suffix = "_combined" if CONFIG.get("cross_dtype_normalization") else ""
        inv_dir = PROJECT_ROOT / "investigations" / f"stability_threshold_{_approach}{_combined_suffix}"
        _dtype_filter = CONFIG.get("stability_dtype_filter", None)
        if isinstance(_dtype_filter, list) and all(d is None or str(d) == "None" for d in _dtype_filter):
            _dtype_filter = None
        run_stability_investigation(
            phenotypes=phenotypes,
            output_dir=inv_dir,
            plot_only=CONFIG.get("investigations_plot_only", False),
            dtype_filter=_dtype_filter,
            normalization_modes=CONFIG.get("stability_normalization_modes", ["full"]),
            plot_mode=CONFIG.get("stability_plot_mode", "combined"),
            plot_levels=CONFIG.get("stability_plot_levels", None),
            compute_levels=CONFIG.get("stability_compute_levels", None),
            cross_dtype_normalization=CONFIG.get("cross_dtype_normalization", False),
            stability_percentile_global_combined=CONFIG.get("stability_percentile_global_combined", 0.6),
            taxonomy_level_combined=CONFIG.get("taxonomy_level_combined"),
        )

    if "stability_characterization" in CONFIG["run_investigations"]:
        _combined_suffix = "_combined" if CONFIG.get("cross_dtype_normalization") else ""
        char_dir = PROJECT_ROOT / "investigations" / f"stability_characterization{_combined_suffix}"
        run_microbe_characterization(
            phenotypes=phenotypes,
            output_dir=char_dir,
            threshold_metagenomics=CONFIG.get("stability_percentile_global_metagenomics"),
            threshold_amplicon=CONFIG.get("stability_percentile_global_amplicon"),
            plot_only=CONFIG.get("investigations_plot_only", False),
            taxonomy_level_metagenomics=CONFIG.get("taxonomy_level_metagenomics"),
            taxonomy_level_amplicon=CONFIG.get("taxonomy_level_amplicon"),
            cross_dtype_normalization=CONFIG.get("cross_dtype_normalization", False),
            stability_percentile_global_combined=CONFIG.get("stability_percentile_global_combined", 0.6),
            taxonomy_level_combined=CONFIG.get("taxonomy_level_combined"),
        )

    if "distribution_approach" in CONFIG["run_investigations"]:
        _combined_suffix = "_combined" if CONFIG.get("cross_dtype_normalization") else ""
        dist_dir = PROJECT_ROOT / "investigations" / f"distribution_approach{_combined_suffix}"
        run_distribution_investigation(
            phenotypes=phenotypes,
            output_dir=dist_dir,
            plot_only=CONFIG.get("investigations_plot_only", False),
            stability_percentile_metagenomics=CONFIG.get("stability_percentile_global_metagenomics", 0.25),
            stability_percentile_amplicon=CONFIG.get("stability_percentile_global_amplicon", 0.40),
            min_size=CONFIG.get("min_samples_per_dataset", 550),
            taxonomy_level_metagenomics=CONFIG.get("taxonomy_level_metagenomics"),
            taxonomy_level_amplicon=CONFIG.get("taxonomy_level_amplicon"),
            cross_dtype_normalization=CONFIG.get("cross_dtype_normalization", False),
            stability_percentile_global_combined=CONFIG.get("stability_percentile_global_combined", 0.6),
            taxonomy_level_combined=CONFIG.get("taxonomy_level_combined"),
        )

    if "5" in RUN_FIGS:
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
