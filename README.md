# GROUP53_CENG313 — Rohlik Sales Forecasting

CENG313 Data Science term project, based on the Kaggle competition
[**Rohlik Sales Forecasting Challenge v2**](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2).

## Team

| Name | Student ID | GitHub |
|---|---|---|
| Ali Osman Bulut | 23118080037 | [@AliOsmanBltt](https://github.com/AliOsmanBltt) |

Group 53 · 1 member

## Competition

[Rohlik Group](https://www.rohlik.cz/) is a European online grocery retailer. The goal is to
forecast the **daily sales of each item in each warehouse for the 14 days** after the
training period (2024-06-03 → 2024-06-16).

- **Target:** `sales`, the units sold of an item (`unique_id`) on a given day
- **Metric:** Weighted Mean Absolute Error (WMAE). Each item has a weight from `test_weights.csv`.
- **Submission format:** `id` (`{unique_id}_{YYYY-MM-DD}`), `sales_hat`

## Dataset

| File | Content |
|---|---|
| `sales_train.csv` | daily history 2020-08-01 → 2024-06-02: `sales`, `total_orders`, `sell_price_main`, `availability`, `type_0..6_discount` |
| `sales_test.csv` | the 14 test days (no `sales` or `availability`) |
| `inventory.csv` | product name and the L1–L4 category hierarchy of every `unique_id` |
| `calendar.csv` | per-warehouse holidays, shop closures and school holidays |
| `test_weights.csv` | evaluation weight of each `unique_id` |

The data is not stored in the repository. The notebooks download it with the Kaggle API.

## Repository structure

```
GROUP53_CENG313/
├── README.md
├── requirements.txt
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA and baselines
├── assignment2_solution.ipynb       # Assignment 2: proposed solution (added later)
├── src/
│   ├── data.py      # download, loading, cleaning, merging, date features, time split
│   ├── models.py    # baseline models with a common fit/predict interface
│   └── utils.py     # WMAE metric, evaluation loop, plotting, submission writer
├── figures/         # plots saved by the notebooks
└── submissions/     # generated Kaggle submission files (git-ignored)
```

## How to run

### Google Colab (recommended)
1. Open the notebook in Colab: *File → Open notebook → GitHub* and paste this repository's URL.
2. Accept the competition rules on Kaggle and download your API token
   (*Kaggle → Settings → API → Create New Token* gives you `kaggle.json`).
3. Run the first cell. It clones the repo, asks for `kaggle.json` and downloads the data into `data/`.
4. Run all cells.

### Local
```bash
git clone https://github.com/AliOsmanBltt/GROUP53_CENG313.git
cd GROUP53_CENG313
pip install -r requirements.txt
# put kaggle.json in ~/.kaggle/, then:
jupyter notebook assignment1_eda_baseline.ipynb
```

## Approach

### Assignment 1: EDA and baselines
The EDA covers data quality, the target distribution, trends, weekly and yearly seasonality,
holidays, discounts, availability, categories, evaluation weights and the train/test comparison.

**Validation:** the last 14 days of train are held out, which mirrors the test horizon.

| Baseline | Description |
|---|---|
| Naive (last value) | last observed sales of the item |
| Seasonal naive | last observed value on the same weekday |
| Moving average 7d / 28d | mean sales over the last 7 / 28 days |
| Weekday mean (8w) | per-item, per-weekday mean over the last 8 weeks |
| Ridge regression | calendar, price, discount and item-level statistics, log1p target |
| Random forest | same features, non-linear |

### Assignment 2: proposed solution
*To be added.*

## Results

| Model | Validation WMAE | Kaggle public | Kaggle private |
|---|---|---|---|
| *to be filled after running* | | | |

## Kaggle submission history

| Date | File | Model | Public | Private |
|---|---|---|---|---|
| | | | | |

## Progress log

| Date | Work done |
|---|---|
| 2026-10-21 | Repository created. Data pipeline (`src/data.py`), baseline models (`src/models.py`), metric and helpers (`src/utils.py`) and the Assignment 1 notebook added. |
