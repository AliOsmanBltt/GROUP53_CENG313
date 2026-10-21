"""Baseline models for Assignment 1.

Every model has the same interface:
    model.fit(train_df)          # train_df contains a `sales` column
    preds = model.predict(df)    # numpy array aligned with df rows
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CALENDAR_FLAGS, CATEGORY_COLS, DISCOUNT_COLS


class BaseForecaster:
    """Shared fallback logic: unseen ids get the global median of train sales."""

    name = "base"

    def fit(self, train):
        self.global_value_ = float(train["sales"].median())
        return self

    def _lookup(self, df, table, keys):
        out = df[keys].merge(table, on=keys, how="left")["pred"]
        return out.fillna(self.global_value_).to_numpy()

    def __repr__(self):
        return self.name


# ---------------------------------------------------------------------------
# Naive / statistical baselines
# ---------------------------------------------------------------------------

class LastValue(BaseForecaster):
    """Predict the last observed sales value of each item."""

    name = "Naive (last value)"

    def fit(self, train):
        super().fit(train)
        last = train.sort_values("date").groupby("unique_id")["sales"].last()
        self.table_ = last.rename("pred").reset_index()
        return self

    def predict(self, df):
        return self._lookup(df, self.table_, ["unique_id"])


class SeasonalNaive(BaseForecaster):
    """Predict the last observed value on the same weekday (lag-7 naive)."""

    name = "Seasonal naive (same weekday)"

    def fit(self, train):
        super().fit(train)
        t = train.assign(dayofweek=train["date"].dt.dayofweek).sort_values("date")
        last = t.groupby(["unique_id", "dayofweek"])["sales"].last()
        self.table_ = last.rename("pred").reset_index()
        self.fallback_ = LastValue().fit(train)
        return self

    def predict(self, df):
        d = df.assign(dayofweek=df["date"].dt.dayofweek)
        out = d[["unique_id", "dayofweek"]].merge(self.table_, on=["unique_id", "dayofweek"],
                                                   how="left")["pred"].to_numpy()
        missing = np.isnan(out)
        out[missing] = self.fallback_.predict(df[missing])
        return out


class MovingAverage(BaseForecaster):
    """Mean sales of each item over the last `window` days of train."""

    def __init__(self, window=28):
        self.window = window
        self.name = f"Moving average ({window}d)"

    def fit(self, train):
        super().fit(train)
        start = train["date"].max() - pd.Timedelta(days=self.window - 1)
        recent = train[train["date"] >= start]
        mean = recent.groupby("unique_id")["sales"].mean()
        # items without recent history fall back to their all-time mean
        full_mean = train.groupby("unique_id")["sales"].mean()
        self.table_ = mean.combine_first(full_mean).rename("pred").reset_index()
        return self

    def predict(self, df):
        return self._lookup(df, self.table_, ["unique_id"])


class DayOfWeekMean(BaseForecaster):
    """Mean sales of each item per weekday over the last `weeks` weeks."""

    def __init__(self, weeks=8):
        self.weeks = weeks
        self.name = f"Weekday mean ({weeks}w)"

    def fit(self, train):
        super().fit(train)
        start = train["date"].max() - pd.Timedelta(weeks=self.weeks)
        recent = train[train["date"] > start]
        recent = recent.assign(dayofweek=recent["date"].dt.dayofweek)
        mean = recent.groupby(["unique_id", "dayofweek"])["sales"].mean()
        self.table_ = mean.rename("pred").reset_index()
        self.fallback_ = MovingAverage(window=self.weeks * 7).fit(train)
        return self

    def predict(self, df):
        d = df.assign(dayofweek=df["date"].dt.dayofweek)
        out = d[["unique_id", "dayofweek"]].merge(self.table_, on=["unique_id", "dayofweek"],
                                                   how="left")["pred"].to_numpy()
        missing = np.isnan(out)
        out[missing] = self.fallback_.predict(df[missing])
        return out


# ---------------------------------------------------------------------------
# Machine-learning baselines (simple features, no lags)
# ---------------------------------------------------------------------------

BASE_FEATURES = (
    ["total_orders", "sell_price_main", "max_discount"] + DISCOUNT_COLS + CALENDAR_FLAGS
    + ["month", "day", "dayofweek", "dayofyear", "weekofyear", "is_weekend"]
)


class TabularFeatures:
    """Builds a numeric feature matrix.

    Besides the raw columns it adds item-level statistics learned on train
    (target encoding of `unique_id` and its weekday profile) and label-encoded
    warehouse / category columns.
    """

    def fit(self, train):
        self.item_stats_ = (train.groupby("unique_id")["sales"]
                            .agg(item_mean="mean", item_median="median", item_std="std")
                            .reset_index())
        t = train.assign(dayofweek=train["date"].dt.dayofweek)
        self.item_dow_ = (t.groupby(["unique_id", "dayofweek"])["sales"].mean()
                          .rename("item_dow_mean").reset_index())
        self.global_mean_ = float(train["sales"].mean())
        self.cat_maps_ = {}
        for col in ["warehouse"] + CATEGORY_COLS:
            if col in train.columns:
                cats = train[col].astype(str).unique()
                self.cat_maps_[col] = {c: i for i, c in enumerate(sorted(cats))}
        return self

    def transform(self, df):
        X = df[["unique_id"]].copy()
        X["dayofweek"] = df["date"].dt.dayofweek.to_numpy()
        X = X.merge(self.item_stats_, on="unique_id", how="left")
        X = X.merge(self.item_dow_, on=["unique_id", "dayofweek"], how="left")
        X["item_dow_mean"] = X["item_dow_mean"].fillna(X["item_mean"])
        for col in ["item_mean", "item_median", "item_dow_mean"]:
            X[col] = X[col].fillna(self.global_mean_)
        X["item_std"] = X["item_std"].fillna(0)
        X = X.drop(columns=["unique_id", "dayofweek"])
        for col in BASE_FEATURES:
            if col in df.columns:
                X[col] = df[col].to_numpy()
        for col, mapping in self.cat_maps_.items():
            X[col + "_code"] = df[col].astype(str).map(mapping).fillna(-1).to_numpy()
        return X.fillna(0).astype("float32")


class SklearnForecaster(BaseForecaster):
    """Wraps any sklearn regressor with TabularFeatures.

    `recent_days` limits training to the most recent part of train (faster and
    closer to the test distribution). `log_target` trains on log1p(sales).
    Evaluation weights are used as sample weights when available.
    """

    def __init__(self, estimator, name, recent_days=365, log_target=True):
        self.estimator = estimator
        self.name = name
        self.recent_days = recent_days
        self.log_target = log_target

    def fit(self, train):
        super().fit(train)
        if self.recent_days:
            start = train["date"].max() - pd.Timedelta(days=self.recent_days)
            fit_df = train[train["date"] > start]
        else:
            fit_df = train
        # item statistics come from the full history, the regressor from the recent window
        self.features_ = TabularFeatures().fit(train)
        X = self.features_.transform(fit_df)
        y = fit_df["sales"].to_numpy()
        if self.log_target:
            y = np.log1p(y)
        sample_weight = fit_df["weight"].to_numpy() if "weight" in fit_df.columns else None
        fit_kwargs = {}
        if sample_weight is not None:
            step = self.estimator.steps[-1][0] if hasattr(self.estimator, "steps") else None
            key = f"{step}__sample_weight" if step else "sample_weight"
            fit_kwargs[key] = sample_weight
        self.estimator.fit(X, y, **fit_kwargs)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, df):
        pred = self.estimator.predict(self.features_.transform(df))
        if self.log_target:
            pred = np.expm1(pred)
        return np.clip(pred, 0, None)


def ridge_baseline(alpha=1.0, **kwargs):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return SklearnForecaster(model, name="Ridge regression", **kwargs)


def random_forest_baseline(n_estimators=100, max_depth=14, max_samples=0.2, seed=42, **kwargs):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  max_samples=max_samples, min_samples_leaf=5,
                                  n_jobs=-1, random_state=seed)
    return SklearnForecaster(model, name="Random forest", **kwargs)


def all_baselines(seed=42):
    """The full list of baselines evaluated in Assignment 1."""
    return [
        LastValue(),
        SeasonalNaive(),
        MovingAverage(window=7),
        MovingAverage(window=28),
        DayOfWeekMean(weeks=8),
        ridge_baseline(),
        random_forest_baseline(seed=seed),
    ]
