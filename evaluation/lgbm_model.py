import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fit_model(X_train, y_train, lambda1=None, lambda2=None):
    """Fit scaler on train only, then fit logistic regression. Returns (model, scaler, X_train_scaled).

    lambda1 : L1 regularization strength (1/C). None = no L1.
    lambda2 : L2 regularization strength (1/C). None = no L2.
    If both None → no regularization (penalty=None).
    """
    if lambda1 is not None:
        penalty, C = "l1", 1.0 / lambda1
    elif lambda2 is not None:
        penalty, C = "l2", 1.0 / lambda2
    else:
        penalty, C = None, 1.0   # sklearn ignores C when penalty=None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=2000,
        solver="saga",   # saga supports l1, l2, and no penalty
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, np.asarray(y_train).ravel())
    return model, scaler, X_scaled


def _compute_metrics(y_true, proba) -> dict:
    pred = (proba > 0.5).astype(int)
    return {
        "auc":       roc_auc_score(y_true, proba),
        "accuracy":  accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall":    recall_score(y_true, pred, zero_division=0),
        "f1":        f1_score(y_true, pred, zero_division=0),
    }


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame,
                   lambda1: float = None, lambda2: float = None) -> dict:
    """Train Logistic Regression and return test + train metrics.

    lambda1 : L1 regularization strength. None = disabled.
    lambda2 : L2 regularization strength. None = disabled.
    Both None = no regularization.
    """
    model, scaler, X_train_scaled = _fit_model(X_train, y_train, lambda1=lambda1, lambda2=lambda2)

    train_m = _compute_metrics(
        np.asarray(y_train).ravel(),
        model.predict_proba(X_train_scaled)[:, 1],
    )
    X_test_scaled = scaler.transform(X_test)   # scaler already fitted on train — no leakage
    test_m = _compute_metrics(
        np.asarray(y_test).ravel(),
        model.predict_proba(X_test_scaled)[:, 1],
    )

    return {
        **test_m,
        **{f"train_{k}": v for k, v in train_m.items()},
    }


def train_lightgbm_with_shap(X_train: pd.DataFrame, y_train: pd.DataFrame,
                              X_test: pd.DataFrame, y_test: pd.DataFrame,
                              lambda1: float = None, lambda2: float = None):
    """Train Logistic Regression and return (metrics, feature_importance) where
    feature_importance is a DataFrame with columns ['feature', 'shap_importance'],
    sorted descending."""
    model, scaler, X_train_scaled = _fit_model(X_train, y_train, lambda1=lambda1, lambda2=lambda2)
    X_test_scaled = scaler.transform(X_test)

    train_m = _compute_metrics(np.asarray(y_train).ravel(), model.predict_proba(X_train_scaled)[:, 1])
    test_m  = _compute_metrics(np.asarray(y_test).ravel(),  model.predict_proba(X_test_scaled)[:, 1])
    metrics = {**test_m, **{f"train_{k}": v for k, v in train_m.items()}}

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature":         X_test.columns.tolist(),
        "shap_importance": mean_abs_shap,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return metrics, feature_importance
