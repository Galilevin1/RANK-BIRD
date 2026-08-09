"""
Save stability-filtered (but un-normalized) datasets to disk.

Pipeline applied per dataset group:
    taxonomy filter → stability filter → reindex to kept microbes

No oversampling, no ranking, no Wasserstein normalization.

Output mirrors the Data/ folder structure under a new root so the existing
load_microbiome_datasets_with_targets() can read it directly:

    Data_filtered/
        CRC Metagenomics/
            PRJEB14674/
                microbiome.csv   ← filtered abundances
                target.csv       ← copied from original
        ...
"""

import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.taxonomy_filter import filter_to_level
from evaluation.data_loading import load_microbiome_datasets_with_targets


def save_filtered_datasets(
    phenotypes: list,
    output_root: str = "Data_filtered",
    stability_percentile_metagenomics: float = 0.05,
    stability_percentile_amplicon: float = 0.15,
    stability_percentile_combined: float = 0.10,
    taxonomy_level_metagenomics=None,
    taxonomy_level_amplicon=None,
    taxonomy_level_combined=None,
    cross_dtype_normalization: bool = False,
    data_root: str = "Data",
):
    """
    Load each phenotype/dtype group, apply taxonomy + stability filter, save.

    Parameters
    ----------
    phenotypes                       : list of (phenotype, dtype) tuples
    output_root                      : root folder for filtered output
    stability_percentile_metagenomics: stability threshold for shotgun datasets
    stability_percentile_amplicon    : stability threshold for 16S datasets
    stability_percentile_combined    : stability threshold when cross-dtype pooling
    taxonomy_level_metagenomics      : taxonomy level for shotgun (None = all)
    taxonomy_level_amplicon          : taxonomy level for 16S (None = all)
    taxonomy_level_combined          : taxonomy level when cross-dtype pooling
    cross_dtype_normalization        : if True, pool all dtypes for one shared filter
    data_root                        : source data folder (default "Data")
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_path    = PROJECT_ROOT / data_root
    out_path     = PROJECT_ROOT / output_root
    out_path.mkdir(parents=True, exist_ok=True)

    if cross_dtype_normalization:
        _save_group(
            phenotypes=phenotypes,
            data_path=data_path,
            out_path=out_path,
            stability_percentile=stability_percentile_combined,
            taxonomy_level=taxonomy_level_combined
            if str(taxonomy_level_combined) != "None" else None,
            label="combined",
        )
    else:
        by_dtype = defaultdict(list)
        for phenotype, dtype in phenotypes:
            by_dtype[dtype].append((phenotype, dtype))

        stability_by_dtype = {
            "Metagenomics": stability_percentile_metagenomics,
            "Amplicon":     stability_percentile_amplicon,
        }
        taxonomy_by_dtype = {
            "Metagenomics": taxonomy_level_metagenomics,
            "Amplicon":     taxonomy_level_amplicon,
        }

        for dtype, pheno_list in by_dtype.items():
            _save_group(
                phenotypes=pheno_list,
                data_path=data_path,
                out_path=out_path,
                stability_percentile=stability_by_dtype.get(dtype, 0.15),
                taxonomy_level=taxonomy_by_dtype.get(dtype),
                label=dtype,
            )


def _save_group(
    phenotypes: list,
    data_path: Path,
    out_path: Path,
    stability_percentile: float,
    taxonomy_level,
    label: str,
):
    """Load all datasets in the group, filter, save each one."""

    # ── 1. Load ────────────────────────────────────────────────────────────────
    all_dfs, all_names, all_pheno_strs, orig_folders = [], [], [], []

    for phenotype, dtype in phenotypes:
        pheno_str = f"{phenotype} {dtype}"
        folder    = data_path / pheno_str
        if not folder.exists():
            print(f"  [SKIP] {pheno_str} — folder not found")
            continue

        dfs, _, names = load_microbiome_datasets_with_targets(folder)
        for df, name in zip(dfs, names):
            all_dfs.append(df)
            all_names.append(name)
            all_pheno_strs.append(pheno_str)
            orig_folders.append(folder / name)

    if not all_dfs:
        print(f"  [SKIP] {label} — no datasets loaded")
        return

    # ── 2. Taxonomy filter ─────────────────────────────────────────────────────
    all_dfs = filter_to_level(all_dfs, taxonomy_level)

    # ── 3. Stability filter ────────────────────────────────────────────────────
    all_microbes  = union_microbes(all_dfs)
    nz_df         = nonzero_percent_by_dataset(all_dfs, all_names, all_microbes)
    kept          = auto_stability_filter(nz_df, percentile=stability_percentile)
    all_dfs       = [df.reindex(columns=kept, fill_value=0.0) for df in all_dfs]

    print(f"\n[{label}] kept {len(kept)} microbes after stability filter "
          f"(percentile={stability_percentile})")

    # ── 4. Save ────────────────────────────────────────────────────────────────
    for df, name, pheno_str, src_folder in zip(all_dfs, all_names, all_pheno_strs, orig_folders):
        dest = out_path / pheno_str / name
        dest.mkdir(parents=True, exist_ok=True)

        # Filtered microbiome
        df.to_csv(dest / "microbiome.csv")

        # Copy target.csv from original
        src_target = src_folder / "target.csv"
        if src_target.exists():
            shutil.copy(src_target, dest / "target.csv")

        print(f"  Saved {pheno_str}/{name}  "
              f"({df.shape[0]} samples × {df.shape[1]} microbes)")

    print(f"[{label}] Done → {out_path}")
