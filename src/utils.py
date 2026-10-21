"""Metrics, plotting helpers and submission utilities."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import make_id


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def wmae(y_true, y_pred, weights):
    """Weighted Mean Absolute Error, the official competition metric."""
    y_true, y_pred, weights = map(np.asarray, (y_true, y_pred, weights))
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def evaluate(y_true, y_pred, weights):
    """Return the competition metric together with unweighted MAE / RMSE."""
    return {
        "WMAE": wmae(y_true, y_pred, weights),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_models(models, train, valid, verbose=True):
    """Fit every model on `train`, score it on `valid` and return a results table."""
    import time

    rows, preds = [], {}
    for model in models:
        start = time.time()
        model.fit(train)
        pred = model.predict(valid)
        scores = evaluate(valid["sales"], pred, valid["weight"])
        scores.update(model=model.name, seconds=round(time.time() - start, 1))
        rows.append(scores)
        preds[model.name] = pred
        if verbose:
            print(f"{model.name:<32} WMAE={scores['WMAE']:.3f}  ({scores['seconds']}s)")
    table = (pd.DataFrame(rows)[["model", "WMAE", "MAE", "RMSE", "seconds"]]
             .sort_values("WMAE").reset_index(drop=True))
    return table, preds


def make_submission(test, preds, path):
    """Write a Kaggle submission file with columns `id`, `sales_hat`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sub = pd.DataFrame({"id": make_id(test), "sales_hat": np.clip(preds, 0, None)})
    sub.to_csv(path, index=False)
    print(f"Saved {len(sub):,} rows to {path}")
    return sub


def save_fig(fig, name, fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=120, bbox_inches="tight")


def missing_report(df):
    """Count and percentage of missing values per column."""
    n = df.isna().sum()
    return (pd.DataFrame({"missing": n, "percent": (100 * n / len(df)).round(2)})
            .query("missing > 0").sort_values("missing", ascending=False))


def plot_item_series(train, unique_id, ax=None, valid=None, preds=None):
    """Plot the sales history of one item, optionally with validation predictions."""
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    s = train[train["unique_id"] == unique_id].sort_values("date")
    ax.plot(s["date"], s["sales"], lw=0.8, label="train")
    if valid is not None:
        mask = (valid["unique_id"] == unique_id).to_numpy()
        v = valid[mask]
        ax.plot(v["date"], v["sales"], lw=1.2, label="actual (valid)")
        for name, p in (preds or {}).items():
            ax.plot(v["date"], np.asarray(p)[mask], lw=1, ls="--", label=name)
    ax.set_title(f"unique_id = {unique_id}")
    ax.legend(fontsize=8)
    return ax
