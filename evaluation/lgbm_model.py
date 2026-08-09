import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

_LGBM_FIXED = {
    'objective':    'binary',
    'metric':       'auc',
    'boosting_type':'gbdt',
    'verbose':      -1,
}


def _suggest_params(trial, space: str) -> dict:
    if space == "within":
        return {
            'num_leaves':        trial.suggest_int  ('num_leaves',        10,   80),
            'max_depth':         trial.suggest_int  ('max_depth',         3,    8),
            'learning_rate':     trial.suggest_float('learning_rate',     0.01, 0.3,  log=True),
            'min_child_samples': trial.suggest_int  ('min_child_samples', 10,   80),
            'min_child_weight':  trial.suggest_float('min_child_weight',  1.0,  100.0, log=True),
            'min_split_gain':    trial.suggest_float('min_split_gain',    0.0,  5.0),
            'feature_fraction':  trial.suggest_float('feature_fraction',  0.4,  1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.5,  1.0),
            'bagging_freq':      trial.suggest_int  ('bagging_freq',      1,    10),
            'lambda_l1':         trial.suggest_float('lambda_l1',         1e-3, 100.0, log=True),
            'lambda_l2':         trial.suggest_float('lambda_l2',         1e-3, 100.0, log=True),
        }
    else:  # "default" — lodo / mixed datasets
        return {
            'num_leaves':        trial.suggest_int  ('num_leaves',        20,   200),
            'max_depth':         trial.suggest_int  ('max_depth',         3,    12),
            'learning_rate':     trial.suggest_float('learning_rate',     0.01, 0.2,  log=True),
            'min_child_samples': trial.suggest_int  ('min_child_samples', 5,    100),
            'min_child_weight':  trial.suggest_float('min_child_weight',  1e-3, 50.0,  log=True),
            'min_split_gain':    trial.suggest_float('min_split_gain',    0.0,  2.0),
            'feature_fraction':  trial.suggest_float('feature_fraction',  0.5,  1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.5,  1.0),
            'bagging_freq':      trial.suggest_int  ('bagging_freq',      1,    10),
            'lambda_l1':         trial.suggest_float('lambda_l1',         1e-8, 100.0, log=True),
            'lambda_l2':         trial.suggest_float('lambda_l2',         1e-8, 100.0, log=True),
        }


def _compute_metrics(y_true, proba) -> dict:
    pred = (proba > 0.5).astype(int)
    return {
        'auc':       roc_auc_score(y_true, proba),
        'accuracy':  accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall':    recall_score(y_true, pred, zero_division=0),
        'f1':        f1_score(y_true, pred, zero_division=0),
    }


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Train LightGBM with fixed hyperparameters, no early stopping."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train.values.ravel()),
                      num_boost_round=200, callbacks=[lgb.log_evaluation(0)])

    train_m = _compute_metrics(y_train.values.ravel(), model.predict(X_train))
    test_m  = _compute_metrics(y_test.values.ravel(),  model.predict(X_test))

    return {**test_m, **{f"train_{k}": v for k, v in train_m.items()}}


def train_lightgbm_optuna(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_val:   pd.DataFrame, y_val:   pd.DataFrame,
    X_test:  pd.DataFrame, y_test:  pd.DataFrame,
    n_trials: int = 30,
    param_space: str = "default",
    random_state: int = 42,
) -> dict:
    """Train LightGBM with Optuna hyperparameter search.

    Optimises on val, never touches test during search. Final model
    is re-trained with the best params (train only, val for early stopping).

    param_space : "default" for lodo/mixed, "within" for within-dataset.
    """
    y_train_arr = y_train.values.ravel()
    y_val_arr   = y_val.values.ravel()
    y_test_arr  = y_test.values.ravel()

    if len(np.unique(y_val_arr)) < 2 or len(np.unique(y_train_arr)) < 2:
        # Single-class train or val — Optuna cannot run; fall back to fixed params
        return train_lightgbm(X_train, y_train, X_test, y_test)

    def objective(trial):
        params = {**_LGBM_FIXED, 'random_state': random_state, **_suggest_params(trial, param_space)}
        train_ds = lgb.Dataset(X_train, label=y_train_arr)
        val_ds   = lgb.Dataset(X_val,   label=y_val_arr, reference=train_ds)
        model = lgb.train(
            params, train_ds, valid_sets=[val_ds],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return roc_auc_score(y_val_arr, model.predict(X_val, num_iteration=model.best_iteration))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        best_params = {**_LGBM_FIXED, 'random_state': random_state, **study.best_params}
        print(f"    [Optuna best] val_auc={study.best_value:.4f}  params={study.best_params}")
    except ValueError:
        # All trials failed — fall back to fixed params
        return train_lightgbm(X_train, y_train, X_test, y_test)

    # Final model: train on all available data (train + val).
    # For LODO this restores the full N-1 training pool; test is never seen.
    # Use a small internal split of the combined data for early stopping so
    # the iteration count is calibrated on the actual N-1 training pool size.
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = np.concatenate([y_train_arr, y_val_arr])

    from sklearn.model_selection import train_test_split as _tts
    X_f_tr, X_f_val, y_f_tr, y_f_val = _tts(
        X_full, y_full, test_size=0.15, random_state=random_state,
        stratify=y_full if len(np.unique(y_full)) >= 2 else None,
    )
    full_tr_ds  = lgb.Dataset(X_f_tr,  label=y_f_tr)
    full_val_ds = lgb.Dataset(X_f_val, label=y_f_val, reference=full_tr_ds)
    model = lgb.train(
        best_params, full_tr_ds, valid_sets=[full_val_ds],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    # Refit on the full combined set at the iteration found above
    best_iteration = model.best_iteration
    model = lgb.train(best_params, lgb.Dataset(X_full, label=y_full),
                      num_boost_round=best_iteration)

    test_m  = _compute_metrics(y_test_arr,  model.predict(X_test))
    train_m = _compute_metrics(y_train_arr, model.predict(X_train))
    return {**test_m, **{f"train_{k}": v for k, v in train_m.items()}}


def train_lightgbm_optuna_cross(
    X_optuna_train: pd.DataFrame, y_optuna_train: pd.DataFrame,
    X_final_train:  pd.DataFrame, y_final_train:  pd.DataFrame,
    X_val:          pd.DataFrame, y_val:          pd.DataFrame,
    X_test:         pd.DataFrame, y_test:         pd.DataFrame,
    n_trials: int = 30,
    param_space: str = "default",
    random_state: int = 42,
) -> dict:
    """Optuna search on cross-phenotype pool; final model on a (possibly different) training set.

    X_optuna_train : large cross-phenotype pool used for HP search
    X_final_train  : training data for the final model
                     (per-phenotype subset for cross_optuna, full pool for cross_train)
    X_val          : same-phenotype validation set (Optuna objective + calibration split)
    X_test         : held-out test set, never seen during HP search
    """
    y_optuna_arr = y_optuna_train.values.ravel()
    y_final_arr  = y_final_train.values.ravel()
    y_val_arr    = y_val.values.ravel()
    y_test_arr   = y_test.values.ravel()

    if len(np.unique(y_val_arr)) < 2 or len(np.unique(y_optuna_arr)) < 2:
        return train_lightgbm(X_final_train, y_final_train, X_test, y_test)

    def objective(trial):
        params = {**_LGBM_FIXED, 'random_state': random_state, **_suggest_params(trial, param_space)}
        train_ds = lgb.Dataset(X_optuna_train, label=y_optuna_arr)
        val_ds   = lgb.Dataset(X_val,          label=y_val_arr, reference=train_ds)
        model = lgb.train(
            params, train_ds, valid_sets=[val_ds], num_boost_round=1000,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return roc_auc_score(y_val_arr, model.predict(X_val, num_iteration=model.best_iteration))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        best_params = {**_LGBM_FIXED, 'random_state': random_state, **study.best_params}
        print(f"    [Optuna best] val_auc={study.best_value:.4f}  params={study.best_params}")
    except ValueError:
        return train_lightgbm(X_final_train, y_final_train, X_test, y_test)

    # Final model: X_final_train + X_val (N-1 of the final training pool).
    # Calibrate best_iteration via internal 85/15 split.
    X_full = pd.concat([X_final_train, X_val], ignore_index=True)
    y_full = np.concatenate([y_final_arr, y_val_arr])

    from sklearn.model_selection import train_test_split as _tts
    X_f_tr, X_f_val, y_f_tr, y_f_val = _tts(
        X_full, y_full, test_size=0.15, random_state=random_state,
        stratify=y_full if len(np.unique(y_full)) >= 2 else None,
    )
    full_tr_ds  = lgb.Dataset(X_f_tr,  label=y_f_tr)
    full_val_ds = lgb.Dataset(X_f_val, label=y_f_val, reference=full_tr_ds)
    model = lgb.train(
        best_params, full_tr_ds, valid_sets=[full_val_ds], num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = model.best_iteration
    model = lgb.train(best_params, lgb.Dataset(X_full, label=y_full),
                      num_boost_round=best_iteration)

    test_m  = _compute_metrics(y_test_arr,  model.predict(X_test))
    train_m = _compute_metrics(y_final_arr, model.predict(X_final_train))
    return {**test_m, **{f"train_{k}": v for k, v in train_m.items()}}


def train_lightgbm_with_shap(X_train: pd.DataFrame, y_train: pd.DataFrame,
                              X_test: pd.DataFrame, y_test: pd.DataFrame):
    """Train LightGBM and return (metrics, feature_importance) where feature_importance
    is a DataFrame with columns ['feature', 'shap_importance'], sorted descending."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train.values.ravel()),
                      num_boost_round=200, callbacks=[lgb.log_evaluation(0)])

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary: positive class

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

    return metrics, feature_importance
