# GROUP52_CENG313 — Rohlik Sales Forecasting

[![Open Assignment 1 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ceng313-2026/GROUP52_CENG313/blob/main/assignment1_eda_baseline.ipynb)

CENG313 Data Science term project, based on the Kaggle competition
[**Rohlik Sales Forecasting Challenge v2**](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2).

## Team

| Name | Student ID | GitHub |
|---|---|---|
| Ali Osman Bulut | 23118080037 | [@AliOsmanBltt](https://github.com/AliOsmanBltt) |

Group 52 · 1 member

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
GROUP52_CENG313/
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
1. Click the *Open in Colab* badge above (or *File → Open notebook → GitHub* and paste this repository's URL).
2. Accept the competition rules on Kaggle and download your API token
   (*Kaggle → Settings → API → Create New Token* gives you `kaggle.json`).
3. Run the first cell. It clones the repo, asks for `kaggle.json` and downloads the data into `data/`.
4. Run all cells.

### Local
```bash
git clone https://github.com/ceng313-2026/GROUP52_CENG313.git
cd GROUP52_CENG313
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
| Ridge regression | reduced, non-redundant feature set (see `RIDGE_EXCLUDE`), log1p target |
| Random forest | all calendar, price, discount, category and item-level features, log1p target |

### Assignment 2: proposed solution
*To be added.*

## Results

Validation on the last 14 days of train (2024-05-20 → 2024-06-02):

| Model | Validation WMAE | Kaggle public | Kaggle private |
|---|---|---|---|
| Random forest | **25.66** | **26.561** | **27.076** |
| Weekday mean (8w) | 29.09 | 30.971 | 30.320 |
| Ridge regression | 30.47 | 31.816 | 32.182 |
| Moving average (28d) | 30.79 | | |
| Moving average (7d) | 32.86 | | |
| Seasonal naive | 36.08 | | |
| Naive (last value) | 42.52 | | |

The random forest is the best baseline on validation and on both Kaggle leaderboards, and the
ranking of the three submitted models is the same on validation and on Kaggle. Ridge has a slightly
higher WMAE than in the first run (28.64) because its feature set was reduced to remove an
ill-conditioned matrix; in exchange it trains without numerical warnings and its RMSE dropped
from 168.9 to 150.9.

### Additional EDA and error analysis / Ek EDA ve hata analizi

- **Weekly pattern by category:** all categories peak on Friday, but Meat and fish changes the most (1.18 on Friday → 0.87 on Sunday), while Bakery stays near its average on Saturday (0.99). (**Kategoriye göre haftalık örüntü:** tüm kateegoriler Cuma zirve yapıyor; en çok değişen Meat and fish (Cuma 1.18 → Pazar 0.87), Bakery ise Cumartesi ortalamasına yakın kalıyor (0.99).)
- **Error analysis of the random forest:** Fruit and vegetable is the hardest category (WMAE ≈ 37) and gives about 68% of the total weighted error. (**Randomm forest hata analizi:** en zor kategori Fruit and vegetable (WMAE ≈ 37) ve toplam ağırlıklı hatanınn yaklaşık %68'i buradan geliyor.)

## Kaggle submission history

| Date | File | Model | Validation WMAE | Public | Private |
|---|---|---|---|---|---|
| 2026-11-04 | `a1_random_forest.csv` | Random forest | 25.66 | 26.561 | 27.076 |
| 2026-11-04 | `a1_weekday_mean_8w.csv` | Weekday mean (8w) | 29.09 | 30.971 | 30.320 |
| 2026-11-04 | `a1_ridge_regression.csv` | Ridge regression | 30.47 | 31.816 | 32.182 |

All Assignment 1 submissions are late submissions (the competition has ended).
Screenshot of the scores / Skorların ekran görüntüsü: [`submissions/a1_kaggle_scores.png`](submissions/a1_kaggle_scores.png)
Submission files are generated by section 16 of the Assignment 1 notebook (the folder is git-ignored).

## AI usage / Yapay zeka kullanımı

Claude was used to help with the initial repository setup and code scaffolding (see the 2026-10-21 commits). / Claude, reponun ilk kurulumunda ve kod iskeletinin hazırlanmasında yardımcı olarak kullandım (2026-10-21 commit'leri).

## Progress log

| Date | Work done |
|---|---|
| 2026-10-21 | Repository created. Data pipeline (`src/data.py`), baseline models (`src/models.py`), metric and helpers (`src/utils.py`) and the Assignment 1 notebook added. |
| 2026-10-21 | Notebook run on the real data. Ridge stabilised by log-transforming skewed features. EDA summary and baseline results added. |
| 2026-11-04 | Ridge baseline reworked: redundant and near-constant columns removed (ill-conditioned matrix warning). Pandas 3 compatibility fix in the naive baselines. Submission cells for the top 3 baselines added. |
| 2026-11-04 | Notebook re-run on Google Colab. Top 3 baselines (random forest, weekday mean, Ridge) submitted to Kaggle as late submissions and scores recorded. |
| 2026-11-06 | AI usage note added to the README. |
| 2026-11-07 | Weekday profile per product category added to the EDA. / EDA'ya kategori bazında haftalık satış profili eklendi. |
| 2026-11-09 | Error analysis of the random forest by product category added. / Random forest için kategori bazında hata analizi eklendi. |
| 2026-11-10 | README updated with the new EDA and error analysis findings. / README yeni EDA ve hata analizi bulgularıyla güncellendi. |
| 2026-11-13 | Written interpretation added under every plot, a Preprocessing section added, prediction plot fixed, Kaggle score screenshot added, repository moved to the ceng313-2026 organization. / Her grafiğin altına yorum, Preprocessing bölümü ve Kaggle ekran görüntüsü eklendi; tahmin grafiği düzeltildi; repo ceng313-2026 organizasyonuna taşındı. |
| 2026-11-13 | Repository renamed to GROUP52_CENG313 (correct group ID) and moved to the ceng313-2026 organization; section map added to the notebook. / Repo doğru grup numarasıyla GROUP52_CENG313 olarak adlandırıldı ve ceng313-2026 organizasyonuna taşındı; notebook'a bölüm eşleştirme tablosu eklendi. |
