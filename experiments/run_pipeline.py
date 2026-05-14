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
import numpy as np
from figures.figure2 import (
    plot_figure2a,
    assemble_figure2, compute_figure2d_data, compute_figure2e_data,
    run_figure2b_supp, run_figure2c_supp, run_figure2d_supp,
    run_figure2e_supp, run_figure2f_supp,
)
from figures.figure3 import assemble_figure3, run_figure3c, run_figure3d
from figures.figure4 import assemble_figure4
from figures.figure2e_optional import run_figure2e_optional, plot_figure2e_optional
from figures.figure5 import plot_figure5
from experiments.investigate_stability_threshold import (
    run_stability_investigation, run_microbe_characterization,
)
from experiments.investigate_distribution_approach import (
    run_distribution_investigation, print_auc_summary_table,
    make_distribution_summary_figure,
)
from evaluation.data_loading import build_papers_auc_df, load_microbiome_datasets_with_targets
from evaluation.dataset_analysis import run_dataset_analysis
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
    "run_figures": ["1"], # "1" (combined), "1a","1c","1d","1e","1f" (individual panels)
     #                   "2" (combined + all supp), "2b_supp","2c_supp","2d_supp","2f_supp" (supp only)
     #                   "2e_optional" (Jaccard vs LODO AUC, cross-phenotype aggregation)
     #                   "3","4","5"
    "run_investigations": [],  # "stability_threshold", "stability_characterization", "distribution_approach"
    "investigations_plot_only": False,   # For  "stability_threshold" and "distribution_approach" invastigations # True = reload CSVs, False = recompute
    "run_quality_report": False,         # compute per-dataset quality metrics (unique microbes, reads, entropy, Simpson)
    "phenotypes": phenotypes_papers,  # phenotypes_pipeline, phenotypes_papers
    "data_folder": "Data",             # "Data" for pipeline phenotypes, "Data_papers" for papers phenotypes
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
        if not analysis_csv.exists() or CONFIG["run_compute"]:
            print("  Computing microbiome_analysis_results.csv ...")
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
        fig, _ = plot_figure2a(csv_path=str(analysis_csv))
        fig.savefig(_dir / "confounder_correlations.pdf", bbox_inches='tight')
        plt.close(fig)

    if "2b_supp" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2b")
        run_figure2b_supp(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
        )

    if "2c_supp" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2c")
        run_figure2c_supp(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

    if "2d_supp" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2d")
        run_figure2d_supp(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
        )

    if "2f_supp" in RUN_FIGS:
        _dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2f")
        run_figure2f_supp(
            phenotypes=phenotypes,
            data_root=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            figures_dir=str(_dir),
        )

    if "2" in RUN_FIGS:
        analysis_csv = RESULTS_DIR / "microbiome_analysis_results.csv"
        if not analysis_csv.exists() or CONFIG["run_compute"]:
            print("  Computing microbiome_analysis_results.csv ...")
            all_mb, all_tgt, all_conf, all_names, d2p = [], [], [], [], {}
            seen_datasets = set()
            for phenotype, dtype in phenotypes_all:
                pheno_str = f"{phenotype} {dtype}"
                for _dr in ["Data", "Data_papers"]:
                    folder = PROJECT_ROOT / _dr / pheno_str
                    if not folder.exists():
                        continue
                    mb_dfs, tgt_dfs, ds_names, conf_dfs = load_microbiome_datasets_with_targets(
                        str(folder), include_confounders=True)
                    for mb, tgt, conf, name in zip(mb_dfs, tgt_dfs, conf_dfs, ds_names):
                        if name in seen_datasets:
                            continue
                        seen_datasets.add(name)
                        all_mb.append(mb); all_tgt.append(tgt)
                        all_conf.append(conf); all_names.append(name); d2p[name] = pheno_str
            run_dataset_analysis(
                all_mb, all_tgt, all_names, d2p,
                output_path=str(analysis_csv),
                confounder_dfs=all_conf,
            )
        data_root = PROJECT_ROOT / CONFIG.get("data_folder", "Data")

        # Pick the representative phenotype for the main figure panels (B–E).
        # Use the first CD entry from the configured phenotypes list; fall back
        # to the first phenotype overall if CD is not present.
        _cd_entries = [(p, d) for p, d in phenotypes if p == "CD"]
        _rep_pheno, _rep_dtype = _cd_entries[0] if _cd_entries else phenotypes[0]
        _rep_str = f"{_rep_pheno} {_rep_dtype}"

        # Load datasets for the representative phenotype, drop control-only
        _rep_mb, _rep_tgt, _rep_names = [], [], []
        _rep_folder = data_root / _rep_str
        if _rep_folder.exists():
            try:
                _mb_dfs, _tgt_dfs, _ds_names = load_microbiome_datasets_with_targets(str(_rep_folder))
                _filtered = [
                    (mb, tgt, name)
                    for mb, tgt, name in zip(_mb_dfs, _tgt_dfs, _ds_names)
                    if len(np.unique(tgt.values.ravel())) >= 2
                ]
                if _filtered:
                    _mbs, _tgts, _ns = zip(*_filtered)
                    _rep_mb.extend(_mbs); _rep_tgt.extend(_tgts); _rep_names.extend(_ns)
            except Exception as _e:
                print(f"  Error loading {_rep_str}: {_e}")

        _data_root = str(data_root)

        # Figure 2D: KS fraction per phenotype (cached)
        _d_data = compute_figure2d_data(
            phenotypes=phenotypes,
            data_root=_data_root,
        )

        # Figure 2E: pairwise LODO + SHAP (cached per phenotype)
        _e_data = compute_figure2e_data(
            phenotypes=phenotypes,
            data_root=_data_root,
        )

        _fig2_dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2")
        fig = assemble_figure2(
            csv_path=str(analysis_csv),
            cd_microbiome_dfs=_rep_mb or None,
            cd_target_dfs=_rep_tgt or None,
            cd_dataset_names=_rep_names or None,
            figure2d_data=_d_data if not _d_data.empty else None,
            figure2e_data=_e_data,
            cd_phenotype_name=_rep_str,
        )
        fig.savefig(_fig2_dir / "figure2.pdf", bbox_inches="tight")
        plt.close(fig)
        print("  Saved Figure 2")

        # ── Supplementary: one figure per phenotype ───────────
        _supp_dir = str(_fig2_dir / "supp")
        run_figure2b_supp(phenotypes=phenotypes, data_root=_data_root, figures_dir=_supp_dir)
        run_figure2c_supp(phenotypes=phenotypes, data_root=_data_root, figures_dir=_supp_dir,
                          normalization_approach=CONFIG.get("normalization_approach"))
        run_figure2d_supp(phenotypes=phenotypes, data_root=_data_root, figures_dir=_supp_dir)
        run_figure2e_supp(phenotypes=phenotypes, data_root=_data_root, figures_dir=_supp_dir)
        run_figure2f_supp(phenotypes=phenotypes, data_root=_data_root, figures_dir=_supp_dir)

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

    if "3" in RUN_FIGS:
        fig = assemble_figure3(
            path_3a=FIGURES_BASE / "figure_3" / "3a" / "schematic.png",
            investigation_dir_3b=str(FIGURES_BASE / "figure_3" / "3b"),
            investigation_dir_3c=str(FIGURES_BASE / "figure_3" / "3c"),
            data_root_3d=str(PROJECT_ROOT / CONFIG.get("data_folder", "Data")),
            phenotypes_3d=phenotypes,
            normalization_approach=CONFIG.get("normalization_approach"),
            path_3e=FIGURES_BASE / "figure_3" / "3e" / "panel_e.png",
        )
        fig.savefig(FIG3_DIR / "figure3.pdf", bbox_inches="tight")
        plt.close(fig)
        print("  Saved Figure 3")

    if "4" in RUN_FIGS:
        _data_root_4 = PROJECT_ROOT / CONFIG.get("data_folder", "Data")

        # Representative phenotype (CD preferred, else first in list)
        _cd_entries = [(p, d) for p, d in phenotypes if p == "CD"]
        _rep_pheno, _rep_dtype = _cd_entries[0] if _cd_entries else phenotypes[0]
        _rep_str = f"{_rep_pheno} {_rep_dtype}"

        _rep_mb, _rep_tgt, _rep_names = [], [], []
        _rep_folder = _data_root_4 / _rep_str
        if _rep_folder.exists():
            try:
                _mb_dfs, _tgt_dfs, _ds_names = load_microbiome_datasets_with_targets(
                    str(_rep_folder)
                )
                _filtered = [
                    (mb, tgt, name)
                    for mb, tgt, name in zip(_mb_dfs, _tgt_dfs, _ds_names)
                    if len(np.unique(tgt.values.ravel())) >= 2
                ]
                if _filtered:
                    _mbs, _tgts, _ns = zip(*_filtered)
                    _rep_mb.extend(_mbs)
                    _rep_tgt.extend(_tgts)
                    _rep_names.extend(_ns)
            except Exception as _e:
                print(f"  Error loading {_rep_str}: {_e}")

        # Panel 4C: KS fraction data — original and normalized (cached)
        _fig4_dir = FIGURES_BASE / "figure_4"
        _fig4_dir.mkdir(parents=True, exist_ok=True)
        _fig2d_data = compute_figure2d_data(
            phenotypes=phenotypes,
            data_root=str(_data_root_4),
            cache_path=str(_fig4_dir / "4c_ks_fractions.csv"),
        )
        _fig2d_norm_data = compute_figure2d_data(
            phenotypes=phenotypes,
            data_root=str(_data_root_4),
            cache_path=str(_fig4_dir / "4c_ks_fractions_normalized.csv"),
            normalization_approach=CONFIG.get("normalization_approach"),
        )

        # Panel 4D: place all distribution CSVs in figures_out/figure_4/4d/
        _fig4d_dir = _fig4_dir / "4d"
        _fig4d_dir.mkdir(parents=True, exist_ok=True)

        if _rep_mb:
            fig = assemble_figure4(
                microbiome_dfs=_rep_mb,
                target_dfs=_rep_tgt,
                dataset_names=_rep_names,
                phenotype_name=_rep_str,
                dtype=_rep_dtype,
                figure2d_data=_fig2d_data if not _fig2d_data.empty else None,
                figure2d_norm_data=_fig2d_norm_data if not _fig2d_norm_data.empty else None,
                data_dir_4d=_fig4d_dir,
            )
            fig.savefig(FIG4_DIR / "figure4.pdf", bbox_inches="tight")
            plt.close(fig)
            print("  Saved Figure 4")
        else:
            print(f"  Figure 4: no data for representative phenotype {_rep_str}")


    if "2e_optional" in RUN_FIGS:
        _pairwise_data_dir = RESULTS_DIR / "pairwise_lodo"
        _2e_opt_dir = build_figures_dir(FIGURES_BASE, CONFIG, "figure_2", "2e_optional")
        results_2e_opt = run_figure2e_optional(
            pairwise_lodo_dir=str(_pairwise_data_dir),
            metric="auc",
        )
        if results_2e_opt is not None:
            plot_figure2e_optional(results_2e_opt, output_dir=str(_2e_opt_dir))

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
        print(f"  [distribution_approach] output dir: {dist_dir.resolve()}")
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
        print_auc_summary_table(dist_dir)

        # Combined summary figure — produced whenever both per-dtype and
        # cross-dtype result directories exist (regardless of which was just run)
        _base_dir     = PROJECT_ROOT / "investigations" / "distribution_approach"
        _combined_dir = PROJECT_ROOT / "investigations" / "distribution_approach_combined"
        if _base_dir.exists() or _combined_dir.exists():
            make_distribution_summary_figure(
                base_dir=_base_dir,
                combined_dir=_combined_dir,
                output_path=PROJECT_ROOT / "investigations" / "distribution_approach_full_summary.png",
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
