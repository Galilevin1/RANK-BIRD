"""
Diagnostic: LightGBM ROC curve — filter_only vs CIFAR on a single dataset.
Plots train and test ROC curves for both approaches side by side.

Usage:
    uv run python -m experiments.roc_lgbm_diagnostic
    uv run python -m experiments.roc_lgbm_diagnostic PRJNA1101026
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sklearn_auc

from evaluation.data_loading import load_microbiome_datasets_with_targets
from src.rankbird.normalization.pipeline import apply_normalization_pipeline
from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.taxonomy_filter import filter_to_level
from experiments.run_pipeline import phenotypes_pipeline, CONFIG

DATA_ROOT  = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "figures_out" / "roc_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset(name: str):
    for phenotype, dtype in phenotypes_pipeline:
        folder = DATA_ROOT / f"{phenotype} {dtype}"
        if not folder.exists():
            continue
        dfs, tgts, names = load_microbiome_datasets_with_targets(folder)
        if name in names:
            idx = names.index(name)
            return phenotype, dtype, dfs, tgts, names, idx
    raise ValueError(f"Dataset '{name}' not found in any phenotype folder.")


def _train_lgbm(X_train, y_train, X_test, y_test):
    """Fit LightGBM; return (prob_train, prob_test)."""
    y_tr = np.asarray(y_train).ravel()
    y_ts = np.asarray(y_test).ravel()

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "seed":             seed,
    }
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_tr),
        valid_sets=[lgb.Dataset(X_test, label=y_ts)],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    prob_train = model.predict(X_train, num_iteration=model.best_iteration)
    prob_test  = model.predict(X_test,  num_iteration=model.best_iteration)
    return prob_train, prob_test


def _roc(y_true, prob):
    fpr, tpr, _ = roc_curve(np.asarray(y_true).ravel(), prob)
    return fpr, tpr, sklearn_auc(fpr, tpr)


def run(dataset_name: str, seed: int = 42):
    print(f"\nDataset: {dataset_name}")
    phenotype, dtype, all_dfs, all_tgts, all_names, tidx = find_dataset(dataset_name)
    print(f"  Phenotype: {phenotype} {dtype}")

    stability_pct = (
        CONFIG["stability_percentile_global_metagenomics"] if dtype == "Metagenomics"
        else CONFIG["stability_percentile_global_amplicon"]
    )
    tax_level = (
        CONFIG["taxonomy_level_metagenomics"] if dtype == "Metagenomics"
        else CONFIG["taxonomy_level_amplicon"]
    )

    X_raw = all_dfs[tidx]
    y     = all_tgts[tidx]

    # ── Filter-only normalization ────────────────────────────────────────────────
    print("  Applying filter-only...")
    dfs_filt   = filter_to_level(all_dfs, tax_level)
    all_microbes = union_microbes(dfs_filt)
    nz_df      = nonzero_percent_by_dataset(dfs_filt, all_names, all_microbes)
    kept       = auto_stability_filter(nz_df, percentile=stability_pct)
    X_filter   = X_raw.reindex(columns=kept, fill_value=0.0)
    print(f"    features kept: {X_filter.shape[1]}")

    # ── CIFAR normalization ──────────────────────────────────────────────────────
    print("  Applying CIFAR...")
    norm_dfs, norm_names = apply_normalization_pipeline(
        all_dfs, all_names,
        global_analysis=True,
        min_samples_per_dataset=CONFIG["min_samples_per_dataset"],
        stability_percentile_global=stability_pct,
        taxonomy_level=tax_level,
    )
    X_cifar = norm_dfs[norm_names.index(dataset_name)]
    print(f"    features kept: {X_cifar.shape[1]}")

    # ── Shared train/test split (same indices for both approaches) ───────────────
    indices = np.arange(len(y))
    tr_idx, ts_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y.values.ravel()
    )

    Xf_tr, Xf_ts = X_filter.iloc[tr_idx], X_filter.iloc[ts_idx]
    Xc_tr, Xc_ts = X_cifar.iloc[tr_idx],  X_cifar.iloc[ts_idx]
    y_tr = y.iloc[tr_idx]
    y_ts = y.iloc[ts_idx]

    # ── Train LGBM ───────────────────────────────────────────────────────────────
    print("  Training LGBM — filter only...")
    pf_tr, pf_ts = _train_lgbm(Xf_tr, y_tr, Xf_ts, y_ts)

    print("  Training LGBM — CIFAR...")
    pc_tr, pc_ts = _train_lgbm(Xc_tr, y_tr, Xc_ts, y_ts)

    # ROC curves
    fpr_f_tr, tpr_f_tr, auc_f_tr = _roc(y_tr, pf_tr)
    fpr_f_ts, tpr_f_ts, auc_f_ts = _roc(y_ts, pf_ts)
    fpr_c_tr, tpr_c_tr, auc_c_tr = _roc(y_tr, pc_tr)
    fpr_c_ts, tpr_c_ts, auc_c_ts = _roc(y_ts, pc_ts)

    print(f"\n  Filter   train AUC={auc_f_tr:.4f}  test AUC={auc_f_ts:.4f}")
    print(f"  CIFAR    train AUC={auc_c_tr:.4f}  test AUC={auc_c_ts:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    COLORS = {"filter_train": "#1565C0", "filter_test": "#90CAF9",
              "cifar_train":  "#B71C1C", "cifar_test":  "#EF9A9A"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Training ROC
    ax = axes[0]
    ax.plot(fpr_f_tr, tpr_f_tr, color=COLORS["filter_train"], lw=2,
            label=f"Filter only  (AUC={auc_f_tr:.3f})")
    ax.plot(fpr_c_tr, tpr_c_tr, color=COLORS["cifar_train"],  lw=2,
            label=f"CIFAR        (AUC={auc_c_tr:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Training ROC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)

    # Right: Test ROC
    ax = axes[1]
    ax.plot(fpr_f_ts, tpr_f_ts, color=COLORS["filter_test"],  lw=2, linestyle="--",
            label=f"Filter only  (AUC={auc_f_ts:.3f})")
    ax.plot(fpr_c_ts, tpr_c_ts, color=COLORS["cifar_test"],   lw=2, linestyle="--",
            label=f"CIFAR        (AUC={auc_c_ts:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Test ROC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"LightGBM  |  Filter Only vs CIFAR  |  {dataset_name}\n"
        f"{phenotype} {dtype}  —  train n={len(tr_idx)}, test n={len(ts_idx)}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / f"roc_lgbm_{dataset_name}_seed{seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "PRJNA1101026"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run(name, seed=seed)
