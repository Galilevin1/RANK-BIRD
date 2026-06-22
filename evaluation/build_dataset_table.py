"""
Build the comprehensive per-dataset supplementary table.

Sources:
  - experiments/results_global_original/protocol_results_per_dataset.csv  (raw AUCs per dataset)
  - experiments/results_global_original/microbiome_analysis_results.csv   (quality/demographics)
  - investigations/distribution_approach/results_*_rankbird.csv            (CIFAR AUCs)
  - figures_out/figure_2/2d/ks_fractions.csv                              (KS divergence per phenotype)
  - raw microbiome data (for AUC-zeros / sparsity-based classification)
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Phenotype display names ───────────────────────────────────────────────────

PHENOTYPE_FULL_NAMES = {
    "AD":                   "Alzheimer's Disease",
    "ASD":                  "Autism Spectrum Disorder",
    "CD":                   "Crohn's Disease",
    "CRC":                  "Colorectal Cancer",
    "PD":                   "Parkinson's Disease",
    "T2D":                  "Type 2 Diabetes",
    "UC":                   "Ulcerative Colitis",
    "Delivery_mode_6month": "Delivery Mode (6 months)",
    "Delivery_mode_month":  "Delivery Mode (1 month)",
    "Delivery_mode_year":   "Delivery Mode (1 year)",
}

MODALITY_DISPLAY = {
    "Metagenomics": "WGS",
    "Amplicon":     "16S",
}

PROTOCOL_COL = {
    "LODO":                 "LODO_AUC",
    "Internal Validation":  "Mixed_datasets_AUC",
    "Within Learning":      "Within_dataset_AUC",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_zero_pct_auc(mb_df: pd.DataFrame, target: pd.Series) -> float:
    """Fraction-of-zeros-per-sample as a binary classifier AUC."""
    if not _SKLEARN_OK:
        return float("nan")
    y = target.values.ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    zero_pct = (mb_df == 0).mean(axis=1).values
    try:
        auc = roc_auc_score(y, zero_pct)
        return max(auc, 1 - auc)   # convention: report > 0.5
    except Exception:
        return float("nan")


def _load_raw_auc_zeros(project_root: Path) -> dict:
    """Read pre-computed roc_auc_*.csv files (sparsity AUC per dataset)."""
    result = {}
    search_dirs = [
        project_root / "figures_out" / "figure_2" / "2b",
        project_root / "figures_out" / "supplementary_figures" / "supp_1",
    ]
    for search_dir in search_dirs:
        for csv_file in search_dir.glob("roc_auc_*.csv"):
            pheno_str = csv_file.stem[len("roc_auc_"):]
            try:
                df = pd.read_csv(csv_file)
                df_z = df[df["predictor"] == "zero_pct_baseline"]
                for _, row in df_z.iterrows():
                    result[(pheno_str, row["dataset"])] = row["auc"]
            except Exception:
                pass
    return result


def _load_cifar_aucs(project_root: Path) -> dict:
    """Per-(phenotype, dataset, protocol) CIFAR AUC from distribution_approach."""
    result = {}
    dist_dir = project_root / "investigations" / "distribution_approach"
    for dtype_lower in ["metagenomics", "amplicon", "combined"]:
        p = dist_dir / f"results_{dtype_lower}_rankbird.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                key = (row["phenotype"], row["dataset"], row["protocol"])
                result[key] = row["auc"]
        except Exception:
            pass
    return result


def _load_ks_fractions(project_root: Path) -> dict:
    """Per-phenotype KS divergence fraction."""
    p = project_root / "figures_out" / "figure_2" / "2d" / "ks_fractions.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        return dict(zip(df["phenotype"], df["ks_sig_fraction"]))
    except Exception:
        return {}


def _load_raw_aucs_pipeline(project_root: Path) -> dict:
    """Per-(phenotype, dataset, protocol) raw AUC from pipeline results."""
    result = {}
    p = project_root / "experiments" / "results_global_original" / "protocol_results_per_dataset.csv"
    if not p.exists():
        return result
    try:
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            key = (row["phenotype"], row["dataset"], row["protocol"])
            result[key] = row["auc"]
    except Exception:
        pass
    return result


def _load_quality_metrics(project_root: Path) -> pd.DataFrame:
    """Per-dataset quality metrics including demographics and raw AUCs."""
    p = project_root / "experiments" / "results_global_original" / "microbiome_analysis_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_dataset_table(
    phenotypes_pipeline: list,
    phenotypes_papers:   list,
    output_path: Path,
    data_root: Path = None,
    project_root: Path = PROJECT_ROOT,
    compute_auc_zeros: bool = True,
) -> pd.DataFrame:
    """
    Build and save the comprehensive per-dataset supplementary table.

    Parameters
    ----------
    phenotypes_pipeline : list of (phenotype, dtype) — "Yamas" pipeline datasets
    phenotypes_papers   : list of (phenotype, dtype) — papers comparison datasets
    output_path         : where to save the CSV
    data_root           : raw data root (for computing AUC-zeros on the fly)
    compute_auc_zeros   : if True, compute sparsity AUC for datasets missing a CSV
    """
    if data_root is None:
        data_root = project_root / "Data"
    data_root = Path(data_root)

    pipeline_set = set(phenotypes_pipeline)
    papers_set   = set(phenotypes_papers)
    all_phenos   = sorted(pipeline_set | papers_set, key=lambda x: (x[0], x[1]))

    print("  Loading data sources...")
    quality_df   = _load_quality_metrics(project_root)
    raw_aucs     = _load_raw_aucs_pipeline(project_root)
    cifar_aucs   = _load_cifar_aucs(project_root)
    ks_fractions = _load_ks_fractions(project_root)
    auc_zeros    = _load_raw_auc_zeros(project_root)

    # Optionally compute AUC-zeros from raw data for missing datasets
    if compute_auc_zeros and _SKLEARN_OK:
        from evaluation.data_loading import load_microbiome_datasets_with_targets
        for pheno, dtype in all_phenos:
            pheno_str = f"{pheno} {dtype}"
            folder = data_root / pheno_str
            if not folder.exists():
                continue
            try:
                mb_dfs, tgt_dfs, ds_names = load_microbiome_datasets_with_targets(str(folder))
                for mb, tgt, name in zip(mb_dfs, tgt_dfs, ds_names):
                    if (pheno_str, name) not in auc_zeros:
                        auc_zeros[(pheno_str, name)] = _compute_zero_pct_auc(mb, tgt)
            except Exception as e:
                print(f"    [AUC-zeros] Could not load {pheno_str}: {e}")

    # Index quality_df for fast lookup
    quality_idx = {}
    if not quality_df.empty:
        for _, row in quality_df.iterrows():
            quality_idx[(row["phenotype"], row["project_name"])] = row

    # Determine all datasets per phenotype
    all_datasets: dict = {}   # (pheno, dtype) → [dataset_ids]
    for pheno, dtype in all_phenos:
        pheno_str = f"{pheno} {dtype}"
        datasets = set()
        # From pipeline results
        for (ph, ds, _), _ in raw_aucs.items():
            if ph == pheno_str:
                datasets.add(ds)
        # From CIFAR results
        for (ph, ds, _), _ in cifar_aucs.items():
            if ph == pheno_str:
                datasets.add(ds)
        # From quality file
        if not quality_df.empty:
            for _, row in quality_df[quality_df["phenotype"] == pheno_str].iterrows():
                datasets.add(row["project_name"])
        all_datasets[(pheno, dtype)] = sorted(datasets)

    # Build rows
    rows = []
    for pheno, dtype in all_phenos:
        pheno_str = f"{pheno} {dtype}"
        datasets  = all_datasets.get((pheno, dtype), [])

        for dataset in datasets:
            q = quality_idx.get((pheno_str, dataset), {})

            def _q(col):
                if isinstance(q, dict):
                    return q.get(col)
                try:
                    return q[col] if pd.notna(q[col]) else None
                except Exception:
                    return None

            # Raw AUCs: prefer pipeline results, fall back to quality file
            lodo_raw   = raw_aucs.get((pheno_str, dataset, "LODO"))
            mixed_raw  = raw_aucs.get((pheno_str, dataset, "Internal Validation"))
            within_raw = raw_aucs.get((pheno_str, dataset, "Within Learning"))
            if lodo_raw is None:
                lodo_raw = _q("leave_one_out_auc")
            if mixed_raw is None:
                mixed_raw = _q("internal_validation_auc")
            if within_raw is None:
                within_raw = _q("within_dataset_auc")

            rows.append({
                "Phenotype_full_name":      PHENOTYPE_FULL_NAMES.get(pheno, pheno),
                "Phenotype_short_name":     pheno,
                "Project_ID":               dataset,
                "Sample_count":             _q("sample_count"),
                "Feature_count":            _q("feature_count"),
                "Sequencing_modality":      MODALITY_DISPLAY.get(dtype, dtype),
                "Phenotype":                pheno_str,
                "LODO_AUC_raw":             _round(lodo_raw),
                "LODO_AUC_CIFAR":           _round(cifar_aucs.get((pheno_str, dataset, "LODO"))),
                "Mixed_datasets_AUC_raw":   _round(mixed_raw),
                "Mixed_datasets_AUC_CIFAR": _round(cifar_aucs.get((pheno_str, dataset, "Internal Validation"))),
                "Within_dataset_AUC_raw":   _round(within_raw),
                "Within_dataset_AUC_CIFAR": _round(cifar_aucs.get((pheno_str, dataset, "Within Learning"))),
                "AUC_zeros":                _round(auc_zeros.get((pheno_str, dataset))),
                "KS_divergence_fraction":   _round(ks_fractions.get(pheno_str)),
                "Age_median":               _q("age_median"),
                "Male_percentage":          _q("male_percentage"),
                "Yamas":                    int((pheno, dtype) in pipeline_set),
                "Papers":                   int((pheno, dtype) in papers_set),
            })

    table = pd.DataFrame(rows)
    table = table.sort_values(["Phenotype_full_name", "Sequencing_modality", "Project_ID"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"  Saved dataset table → {output_path}  ({len(table)} rows, {table.columns.tolist().__len__()} columns)")
    return table


def _round(val, ndigits=4):
    try:
        return round(float(val), ndigits) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
    except Exception:
        return None
