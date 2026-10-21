"""Data loading and preprocessing for the Rohlik Sales Forecasting Challenge v2."""

import os
import subprocess
import zipfile

import numpy as np
import pandas as pd

COMPETITION = "rohlik-sales-forecasting-challenge-v2"

DISCOUNT_COLS = [f"type_{i}_discount" for i in range(7)]
CATEGORY_COLS = ["L1_category_name_en", "L2_category_name_en",
                 "L3_category_name_en", "L4_category_name_en"]
CALENDAR_FLAGS = ["holiday", "shops_closed", "winter_school_holidays", "school_holidays"]


def download_data(data_dir="data"):
    """Download and unzip the competition files with the Kaggle API.

    Requires a valid ~/.kaggle/kaggle.json and the competition rules to be
    accepted on the Kaggle website.
    """
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "sales_train.csv")):
        print("Data already present, skipping download.")
        return
    subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", data_dir],
                   check=True)
    with zipfile.ZipFile(os.path.join(data_dir, f"{COMPETITION}.zip")) as zf:
        zf.extractall(data_dir)
    print("Files:", sorted(os.listdir(data_dir)))


def load_raw(data_dir="data"):
    """Read every competition CSV into a dict of DataFrames."""
    def read(name, **kwargs):
        return pd.read_csv(os.path.join(data_dir, name), **kwargs)

    return {
        "train": read("sales_train.csv", parse_dates=["date"]),
        "test": read("sales_test.csv", parse_dates=["date"]),
        "inventory": read("inventory.csv"),
        "calendar": read("calendar.csv", parse_dates=["date"]),
        "weights": read("test_weights.csv"),
    }


def clean_sales(df):
    """Basic cleaning shared by train and test.

    - drops train rows without a sales value
    - clips discounts to [0, 1] (the raw data contains a few negative values)
    - adds `max_discount`, the strongest of the seven discount types
    """
    df = df.copy()
    if "sales" in df.columns:
        df = df.dropna(subset=["sales"])
    present = [c for c in DISCOUNT_COLS if c in df.columns]
    df[present] = df[present].fillna(0).clip(0, 1)
    df["max_discount"] = df[present].max(axis=1)
    return df


def merge_all(sales, inventory, calendar, weights=None):
    """Join product info, calendar flags and evaluation weights onto sales rows."""
    df = sales.merge(inventory.drop(columns=["warehouse"], errors="ignore"),
                     on="unique_id", how="left")
    calendar = calendar.drop_duplicates(subset=["date", "warehouse"])
    df = df.merge(calendar, on=["date", "warehouse"], how="left")
    for col in CALENDAR_FLAGS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if weights is not None:
        df = df.merge(weights, on="unique_id", how="left")
    return df


def add_date_features(df):
    """Calendar features derived from the date column."""
    df = df.copy()
    d = df["date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def prepare(raw):
    """Run the full preprocessing pipeline and return (train, test) frames."""
    train = clean_sales(raw["train"])
    test = clean_sales(raw["test"])
    train = add_date_features(merge_all(train, raw["inventory"], raw["calendar"], raw["weights"]))
    test = add_date_features(merge_all(test, raw["inventory"], raw["calendar"], raw["weights"]))
    return train, test


def time_split(df, horizon_days=14):
    """Hold out the last `horizon_days` of train as validation.

    The Kaggle test set covers the 14 days right after train ends, so the
    validation set mimics that setup.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=horizon_days)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def make_id(df):
    """Build the submission id: `{unique_id}_{YYYY-MM-DD}`."""
    return df["unique_id"].astype(str) + "_" + df["date"].dt.strftime("%Y-%m-%d")


def reduce_memory(df):
    """Downcast numeric columns to save RAM (train has ~4M rows)."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
