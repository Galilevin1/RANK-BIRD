import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Train LightGBM model and return metrics."""
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

    train_data = lgb.Dataset(X_train, label=y_train.values.ravel())
    valid_data = lgb.Dataset(X_test, label=y_test.values.ravel())

    model = lgb.train(params, train_data, valid_sets=[valid_data],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics

