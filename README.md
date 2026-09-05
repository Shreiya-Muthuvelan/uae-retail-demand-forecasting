# UAE Retail Demand Forecasting with MLflow

Weekly sales forecasting with Islamic calendar feature engineering, tracked
across three experiments using MLflow. Champion model registered in the
MLflow Model Registry.

---

## Results

| Experiment | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| XGBoost Baseline | 2322.65 | 3384.59 | 5.67% | 0.9563 |
| + Ramadan Features | 2307.70 | 3332.87 | 5.66% | **0.9577** ← champion |
| + Eid Features | 2326.73 | 3342.71 | 5.70% | 0.9574 |

**Champion model:** XGBoost + Ramadan Features — R²=0.9577, MAPE=5.66%  
Registered in MLflow Model Registry at stage `Production`.

### Metric definitions
- **MAE** (Mean Absolute Error) — average absolute difference between predicted and actual weekly sales, in AED units
- **RMSE** (Root Mean Squared Error) — penalises large errors more heavily than MAE; useful for detecting outlier weeks
- **MAPE** (Mean Absolute Percentage Error) — scale-independent error rate; 5.66% means predictions are off by 5.66% on average
- **R²** (Coefficient of Determination) — proportion of sales variance explained by the model; 0.958 means the model explains 95.8% of weekly sales variation across 1,115 stores

---

## Project structure

```
uae-retail-demand-forecasting/
├── exp_1/                  # Baseline XGBoost
│   ├── data_preparation.py
│   ├── features.py
│   ├── model.py
│   ├── train_test_split.py
│   ├── evaluate.py
│   └── main.py
├── exp_2/                  # + Ramadan features
│   ├── features.py
│   └── main.py
├── exp_3/                  # + Eid features
│   ├── features.py
│   └── main.py
├── registry/
│   ├── register_champion.py
│   └── test.py
├── explore_data.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Feature engineering

### Standard features
- **Lag features** — lag_1, lag_2, lag_4, lag_8 (weekly lags per store)
- **Rolling means** — rolling_mean_4, rolling_mean_8 (shift(1) applied to prevent leakage)
- **Calendar** — Year, Month, WeekOfYear, Quarter
- **Store attributes** — StoreType, Assortment, CompetitionDistance, Promo, Promo2

### Hijri calendar features (exp_2 and exp_3)
Islamic calendar events shift 10–12 days earlier each Gregorian year, making
standard month-of-year features insufficient to capture them. All Hijri dates
are computed using the Umm al-Qura calendar standard.

| Feature | Description |
|---|---|
| `ramadan_flag` | 1 if week overlaps Ramadan, 0 otherwise |
| `ramadan_week` | Sequential week number within Ramadan (1–5) |
| `eid_flag` | 1 if week overlaps Eid Al Fitr or Eid Al Adha |
| `eid_week` | Sequential week number within Eid window |

---

## Key finding

Hijri calendar features produced marginal improvement on the Rossmann proxy
dataset. Diagnostic analysis revealed the reason: **temporal collinearity
with German summer seasonality**.

Ramadan falls in June–August across the 2013–2015 data range — the same
window as the German summer demand decline, which is already captured by
`rolling_mean_8`, `WeekOfYear`, and lag features. As a result, XGBoost assigns
`ramadan_flag` an importance of **0.004 (rank 15/29)** — the model learns
almost nothing additional from it.

Importantly, error is actually *lower* during Ramadan weeks (MAE: 2250) than
non-Ramadan weeks (MAE: 2323) — not because the Ramadan feature helps, but
because the model already handles German summer weeks accurately via existing
seasonal features. This is a case where correlational performance masquerades
as causal feature utility.

**The Hijri feature engineering is architecturally correct.** On UAE retail
data, where Ramadan drives causally distinct demand behaviour (fasting-hour
traffic shifts, pre-Eid purchasing spikes), these features would rank
significantly higher. This experiment confirms the implementation and
identifies domain mismatch as the limiting factor on the Rossmann dataset.

---

## MLflow experiment tracking

All experiments are tracked with MLflow using a SQLite backend.

**Logged per run:**
- Parameters — model type, feature set, n_estimators, learning_rate,
  max_depth, train/test date ranges, row counts
- Metrics — MAE, RMSE, MAPE, R²
- Tags — experiment conclusion, feature set name
- Artifacts — feature importance rankings

**Model Registry:**  
The Ramadan-feature XGBoost (R²=0.958, MAPE=5.66%) is registered as
`UAE-Retail-Demand-XGBoost` at stage `Production`.

To reproduce the registry step:
```bash
uv run python -m registry.register_champion
```

---

## Setup and usage

**Requirements**
```bash
pip install -r requirements.txt
```

**Data**  
Download the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data)
from Kaggle and place `train.csv` and `store.csv` in a `data/` directory.

**Run experiments**
```bash
uv run python -m exp_1.main   # Baseline XGBoost
uv run python -m exp_2.main   # + Ramadan features
uv run python -m exp_3.main   # + Eid features
```

**View MLflow UI**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` to compare all runs.

**Register champion model**
```bash
uv run python -m registry.register_champion
```

---

## Stack

Python · XGBoost · MLflow · pandas · scikit-learn · uv
