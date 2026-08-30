"""
Standalone: within-dataset protocol comparing LGBM vs Logistic Regression
across all normalization approaches.

Does not modify any existing script.
Results saved to investigations/within_lgbm_vs_lr/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial as _partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon, ttest_rel
import lightgbm as lgb

from src.rankbird.normalization.stability import (
    union_microbes, nonzero_percent_by_dataset, auto_stability_filter,
)
from src.rankbird.normalization.pipeline import apply_normalization_pipeline
from src.rankbird.normalization.taxonomy_filter import filter_to_level
from src.rankbird.normalization.ranking import (
    rank_normalize, sigmoid_normalize, relu_normalize,
    clr_normalize, clr_rank_normalize, compute_alpha_diversity_features,
    apply_ranking_pipeline,
    SIGMOID_K, SIGMOID_CENTER, RELU_THRESHOLD,
)
from evaluation.data_loading import load_microbiome_datasets_with_targets
from experiments.investigate_distribution_approach import APPROACHES, APPROACH_LABELS
from experiments.run_protocols_global_processing import _detect_and_shuffle_ordered

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "Data"


# ── Model training ────────────────────────────────────────────────────────────

def _metrics(y_true, proba) -> dict:
    pred = (proba > 0.5).astype(int)
    return {
        "auc":       float(roc_auc_score(y_true, proba)),
        "accuracy":  float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
    }


def _train_lgbm(X_train, y_train, X_test, y_test) -> dict:
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "random_state": 42,
    }
    y_tr = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
    y_te = y_test.values.ravel()  if hasattr(y_test,  "values") else np.asarray(y_test).ravel()
    model = lgb.train(params, lgb.Dataset(X_train, label=y_tr),
                      num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    return _metrics(y_te, model.predict(X_test))


def _clr(X_arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    log_x = np.log(X_arr + eps)
    return log_x - log_x.mean(axis=1, keepdims=True)


_CLR_ALREADY = {"clr", "clr_ranking"}


def _train_lr(X_train, y_train, X_test, y_test, approach: str = "") -> dict:
    y_tr = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
    y_te = y_test.values.ravel()  if hasattr(y_test,  "values") else np.asarray(y_test).ravel()
    X_tr_arr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_te_arr = X_test.values  if hasattr(X_test,  "values") else np.asarray(X_test)
    # clr/clr_ranking are already signed log-ratios — applying CLR again produces NaN.
    if approach not in _CLR_ALREADY:
        X_tr_arr = _clr(X_tr_arr)
        X_te_arr = _clr(X_te_arr)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_arr)
    X_te_sc = scaler.transform(X_te_arr)
    clf = LogisticRegression(penalty="elasticnet", C=0.1, l1_ratio=0.5,
                             solver="saga", max_iter=2000, random_state=42)
    clf.fit(X_tr_sc, y_tr)
    return _metrics(y_te, clf.predict_proba(X_te_sc)[:, 1])


# ── Within-dataset protocol ───────────────────────────────────────────────────

def _within_protocol(microbiome_dfs, target_dfs, dataset_names,
                     approach: str = "",
                     test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    records = []
    for df, y, name in zip(microbiome_dfs, target_dfs, dataset_names):
        if len(y["Tag"].unique()) < 2:
            print(f"  [SKIP] {name} — single class")
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                df, y, test_size=test_size, random_state=random_state, stratify=y,
            )
        except ValueError as e:
            print(f"  [SKIP] {name} — {e}")
            continue

        lgbm_m = _train_lgbm(X_tr, y_tr, X_te, y_te)
        lr_m   = _train_lr  (X_tr, y_tr, X_te, y_te, approach=approach)

        print(f"  {name:30s}  LGBM={lgbm_m['auc']:.3f}  LR={lr_m['auc']:.3f}")
        records.append({"dataset": name, "model": "LGBM", **lgbm_m})
        records.append({"dataset": name, "model": "LR",   **lr_m})

    return pd.DataFrame(records)


# ── Normalization ─────────────────────────────────────────────────────────────

def _apply_normalization(approach, microbiome_dfs, names,
                         stability_percentile, min_size, taxonomy_level,
                         rank_tie_method, random_state):
    if approach == "original":
        return filter_to_level(microbiome_dfs, taxonomy_level), names

    if approach == "original_filtered":
        dfs = filter_to_level(microbiome_dfs, taxonomy_level)
        all_microbes = union_microbes(dfs)
        nz_df = nonzero_percent_by_dataset(dfs, names, all_microbes)
        kept  = auto_stability_filter(nz_df, percentile=stability_percentile)
        return [df.reindex(columns=kept, fill_value=0.0) for df in dfs], names

    if approach == "rankbird":
        return apply_normalization_pipeline(
            microbiome_dfs, names, global_analysis=True,
            min_samples_per_dataset=min_size,
            stability_percentile_global=stability_percentile,
            taxonomy_level=taxonomy_level,
            rank_tie_method=rank_tie_method,
        )

    _norm_fn_map = {
        "ranking":      _partial(rank_normalize,     tie_method=rank_tie_method),
        "ranking_sig":  _partial(sigmoid_normalize,  k=SIGMOID_K, center=SIGMOID_CENTER,
                                                     tie_method=rank_tie_method),
        "ranking_relu": _partial(relu_normalize,     threshold=RELU_THRESHOLD,
                                                     tie_method=rank_tie_method),
        "clr":          _partial(clr_normalize,      tie_method=rank_tie_method),
        "clr_ranking":  _partial(clr_rank_normalize, tie_method=rank_tie_method),
    }

    if approach in _norm_fn_map:
        return apply_ranking_pipeline(
            microbiome_dfs, names,
            stability_percentile=stability_percentile,
            norm_fn=_norm_fn_map[approach],
            min_size=min_size,
            random_state=random_state,
            taxonomy_level=taxonomy_level,
        )

    if approach == "ranking_alpha":
        alpha = compute_alpha_diversity_features(microbiome_dfs, names)
        normed, out_names = apply_ranking_pipeline(
            microbiome_dfs, names,
            stability_percentile=stability_percentile,
            norm_fn=_partial(rank_normalize, tie_method=rank_tie_method),
            min_size=min_size, random_state=random_state, taxonomy_level=taxonomy_level,
        )
        normed = [pd.concat([df, alpha[n].reindex(df.index)], axis=1)
                  for df, n in zip(normed, out_names)]
        return normed, out_names

    if approach == "ranking_pa":
        normed, out_names = apply_ranking_pipeline(
            microbiome_dfs, names,
            stability_percentile=stability_percentile,
            norm_fn=_partial(rank_normalize, tie_method=rank_tie_method),
            min_size=min_size, random_state=random_state, taxonomy_level=taxonomy_level,
        )
        normed = [
            pd.concat([df, pd.DataFrame((df.values > 0).astype(float), index=df.index,
                                        columns=[f"__pa_{c}__" for c in df.columns])], axis=1)
            for df in normed
        ]
        return normed, out_names

    raise ValueError(f"Unknown approach: {approach!r}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_within_lgbm_vs_lr(
    phenotypes: list,
    output_dir: Path,
    stability_percentile_metagenomics: float = 0.25,
    stability_percentile_amplicon: float      = 0.40,
    min_size: int  = 550,
    taxonomy_level_metagenomics = None,
    taxonomy_level_amplicon     = None,
    shuffle_ordered: bool = True,
    rank_tie_method: str  = "average",
    random_state: int     = 42,
    test_size: float      = 0.2,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for dtype in ["Metagenomics", "Amplicon"]:
        pheno_subset = [(p, t) for p, t in phenotypes if t == dtype]
        if not pheno_subset:
            continue

        stability_pct = (stability_percentile_metagenomics if dtype == "Metagenomics"
                         else stability_percentile_amplicon)
        tax_level     = (taxonomy_level_metagenomics if dtype == "Metagenomics"
                         else taxonomy_level_amplicon)

        # Load all datasets for this dtype
        all_microbiome, all_targets, all_names, dataset_to_phenotype = [], [], [], {}
        for phenotype, t in pheno_subset:
            folder = DATA_ROOT / f"{phenotype} {t}"
            if not folder.exists():
                continue
            dfs, tgts, names = load_microbiome_datasets_with_targets(folder)
            for df, y, name in zip(dfs, tgts, names):
                all_microbiome.append(df)
                all_targets.append(y)
                all_names.append(name)
                dataset_to_phenotype[name] = f"{phenotype} {t}"

        if not all_microbiome:
            continue

        all_microbiome, all_targets, _ = _detect_and_shuffle_ordered(
            all_microbiome, all_targets, all_names,
            random_state=random_state, apply_shuffle=shuffle_ordered,
        )
        name_to_target = dict(zip(all_names, all_targets))

        print(f"\n{'='*60}\ndtype={dtype}  ({len(all_names)} datasets)\n{'='*60}")

        for approach in APPROACHES:
            csv_path = output_dir / f"within_{dtype.lower()}_{approach}.csv"

            print(f"\n--- {APPROACH_LABELS[approach]} ---")

            if csv_path.exists():
                print(f"  [LOAD] {csv_path.name}")
                df_res = pd.read_csv(csv_path)
            else:
                try:
                    normed, normed_names = _apply_normalization(
                        approach,
                        [df.copy() for df in all_microbiome],
                        list(all_names),
                        stability_percentile=stability_pct,
                        min_size=min_size,
                        taxonomy_level=tax_level,
                        rank_tie_method=rank_tie_method,
                        random_state=random_state,
                    )
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue

                aligned_targets = [name_to_target[n] for n in normed_names if n in name_to_target]
                df_res = _within_protocol(normed, aligned_targets, normed_names,
                                          approach=approach,
                                          test_size=test_size, random_state=random_state)
                if df_res.empty:
                    continue

                df_res["approach"] = approach
                df_res["dtype"]    = dtype
                df_res["phenotype"] = df_res["dataset"].map(dataset_to_phenotype)
                df_res.to_csv(csv_path, index=False)

            all_records.append(df_res)

    if not all_records:
        print("No results produced.")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv(output_dir / "within_all.csv", index=False)

    _plot(combined, output_dir)
    _save_stats(combined, output_dir)

    return combined


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, output_dir: Path):
    dtypes = [d for d in ["Metagenomics", "Amplicon"] if d in df["dtype"].values]
    fig, axes = plt.subplots(1, len(dtypes), figsize=(8 * len(dtypes), 5), squeeze=False)

    for ax, dtype in zip(axes[0], dtypes):
        sub = df[df["dtype"] == dtype].copy()
        sub["approach_label"] = sub["approach"].map(APPROACH_LABELS)
        order = [APPROACH_LABELS[a] for a in APPROACHES if a in sub["approach"].values]

        sns.boxplot(data=sub, x="approach_label", y="auc", hue="model",
                    order=order, ax=ax, width=0.55, fliersize=0,
                    palette={"LGBM": "#4878CF", "LR": "#D65F5F"})
        sns.stripplot(data=sub, x="approach_label", y="auc", hue="model",
                      order=order, ax=ax, dodge=True, alpha=0.45, size=3.5,
                      jitter=True, legend=False,
                      palette={"LGBM": "#2a4a8a", "LR": "#8a2a2a"})

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.3, alpha=0.7)
        ax.set_title(dtype, fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalization approach", fontsize=11)
        ax.set_ylabel("AUC", fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.legend(title="Model", fontsize=9, loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Within-Dataset: LGBM vs Logistic Regression", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "within_lgbm_vs_lr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Statistics ────────────────────────────────────────────────────────────────

def _save_stats(df: pd.DataFrame, output_dir: Path):
    # Mean / median per (dtype, approach, model)
    summary = (df.groupby(["dtype", "approach", "model"])["auc"]
               .agg(mean="mean", median="median", std="std", n="count")
               .reset_index())
    summary["approach_label"] = summary["approach"].map(APPROACH_LABELS)
    summary.to_csv(output_dir / "within_summary.csv", index=False)
    print(f"Saved: {output_dir / 'within_summary.csv'}")

    # Paired LGBM vs LR per (dtype, approach)
    rows = []
    for dtype in df["dtype"].unique():
        for approach in df["approach"].unique():
            sub  = df[(df["dtype"] == dtype) & (df["approach"] == approach)]
            s_lg = sub[sub["model"] == "LGBM"].set_index("dataset")["auc"]
            s_lr = sub[sub["model"] == "LR"  ].set_index("dataset")["auc"]
            common = sorted(set(s_lg.index) & set(s_lr.index))
            if len(common) < 3:
                continue
            v_lg = s_lg.loc[common].values
            v_lr = s_lr.loc[common].values
            diffs = v_lg - v_lr
            try:
                _, p_t = ttest_rel(v_lg, v_lr)
            except Exception:
                p_t = float("nan")
            try:
                _, p_w = wilcoxon(diffs, alternative="two-sided")
            except Exception:
                p_w = float("nan")
            rows.append({
                "dtype":          dtype,
                "approach":       approach,
                "approach_label": APPROACH_LABELS.get(approach, approach),
                "n_datasets":     len(common),
                "mean_lgbm":      round(float(v_lg.mean()), 3),
                "mean_lr":        round(float(v_lr.mean()),  3),
                "delta_mean":     round(float(diffs.mean()), 3),
                "p_ttest":        round(float(p_t), 4),
                "p_wilcoxon":     round(float(p_w), 4),
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "within_lgbm_vs_lr_stats.csv", index=False)
    print(f"Saved: {output_dir / 'within_lgbm_vs_lr_stats.csv'}")
    if not stats_df.empty:
        print("\nLGBM vs LR (delta = LGBM − LR, positive = LGBM wins):")
        print(stats_df.to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from experiments.run_pipeline import phenotypes_pipeline, CONFIG

    run_within_lgbm_vs_lr(
        phenotypes=phenotypes_pipeline,
        output_dir=PROJECT_ROOT / "investigations" / "within_lgbm_vs_lr",
        stability_percentile_metagenomics=CONFIG.get("stability_percentile_global_metagenomics", 0.25),
        stability_percentile_amplicon    =CONFIG.get("stability_percentile_global_amplicon",     0.40),
        min_size              =CONFIG.get("min_samples_per_dataset", 550),
        taxonomy_level_metagenomics=CONFIG.get("taxonomy_level_metagenomics"),
        taxonomy_level_amplicon    =CONFIG.get("taxonomy_level_amplicon"),
        shuffle_ordered=True,
        rank_tie_method="average",
    )
