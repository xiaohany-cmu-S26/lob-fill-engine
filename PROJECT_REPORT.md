# LOB Fill Probability Estimator — Project Report

## Overview

This project is **Phase 3 of a four-phase Limit Order Book (LOB) + Market-Making simulator**. It trains and evaluates machine-learning models that estimate the probability a passive limit order placed at the best bid or ask will be **fully filled within 1 second**.

The estimator is designed to feed directly into Phase 4's expected-value quoting engine:

```
EV = P(fill) × spread_capture − (1 − P(fill)) × inventory_cost
```

The design philosophy prioritises **interpretability over raw performance** and applies strict financial ML methodology to avoid data leakage.

---

## Data

**Source:** LOBSTER NASDAQ tick data, July 2023  
**Tickers:** AAPL (5 trading days, ~21 K limit-order events) and CSCO (17 trading days)

Each trading day consists of two aligned files:

| File | Contents |
|---|---|
| Message file | Timestamp, event type (1 = new limit, 2–5 = cancel/execute, 7 = halt), order ID, size (shares), price (integer ticks ÷ 10 000 = USD), direction (+1 buy / −1 sell) |
| Orderbook file | 40 columns: ask/bid price and volume for each of 10 price levels |

**Data cleaning applied:**
- Remove trading-halt rows (`event_type == 7`)
- Drop crossed-book rows (`ask_p1 ≤ bid_p1`)
- Remove the first 60 rows of each day (opening auction artefacts)

---

## Label Construction

At each new-limit-order event a **synthetic passive order** is placed at the back of the queue at the best bid/ask, sized to match the original order.

**Size-aware binary fill label (1-second horizon):**

```
filled = (cumulative execution volume at price ≥ queue_ahead + order_size)
         within the next 1 second
```

Using cumulative executed volume (rather than a simple price-touch) correctly reflects that a 100-share order is harder to fill than a 10-share order at the same queue position.

Implementation uses `numpy.searchsorted` for O(M log N) label computation over the timestamp array.

A **fractional fill variant** also exists:

```
fill_fraction = clip((cum_exec_vol − queue_ahead) / order_size, 0, 1)
```

This continuous [0, 1] target is used in a parallel regression pipeline.

---

## Features

15 features, all computed from historical data only (no forward leakage).

### Microstructure features (8)

| Feature | Description |
|---|---|
| `imbalance` | (bid_vol − ask_vol) / (bid_vol + ask_vol) at top-of-book |
| `depth_imbalance` | Imbalance over all 10 LOB levels |
| `queue_ahead` | Shares in front of the synthetic order at placement |
| `spread_norm` | Bid-ask spread in tick-relative units |
| `local_vol` | Rolling 20-event mid-price return standard deviation |
| `aggressive_flow` | Count of market/aggressive orders in a recent window |
| `direction` | +1 (buy) / −1 (sell) |
| `order_size` | Size of the submitted order |

### Derived / normalised features (3)

| Feature | Description |
|---|---|
| `queue_position_ratio` | `queue_ahead / total_side_depth` |
| `queue_turnover` | `aggressive_flow / queue_ahead` |
| `vol_in_spreads` | `local_vol / spread` |

### Intraday seasonality features (3)

| Feature | Description |
|---|---|
| `time_sin`, `time_cos` | Cyclical encoding of time-of-day (avoids linear distance artefact between 09:31 and 15:59) |
| `time_bucket` | Discrete bucket: 0 = open, 1 = mid-morning, 2 = midday, 3 = close |

### Macro / regime features (2)

| Feature | Description |
|---|---|
| `vol_regime` | Binary high/low volatility; threshold fitted on training set only |
| `day_of_week` | 0–4; expected to add negligible signal at 1-second horizon |

---

## Train / Validation / Test Pipeline

### Split strategy

Strict **chronological 70 / 15 / 15 split by calendar date** — no random shuffling, no future data in training.

### Leakage prevention measures

| Technique | Detail |
|---|---|
| **Purging** | Training rows whose 1-second label window extends into the validation period are dropped |
| **Embargoing** | First 30 seconds of each val/test period are discarded |
| **Vol-regime threshold** | Fitted on training set only, then applied unchanged to val and test |
| **Feature selection** | Performed entirely on training data |
| **Time-series cross-validation** | `TimeSeriesSplit(n_splits=5, gap=500)` throughout |
| **Sample uniqueness weights** | Observations with overlapping label windows are down-weighted (López de Prado, *AFML* Ch. 4) |

---

## Feature Selection

Three complementary methods are applied on the training set only. The final feature set is the **union of Genetic Algorithm and RFECV selections** (conservative; keeps any feature selected by either method).

### 1. Mutual Information

`mutual_info_classif` ranks all 15 features. Expected result: intraday features rank above macro features at the 1-second horizon.

### 2. RFECV (Recursive Feature Elimination with Cross-Validation)

- Base estimator: Logistic Regression  
- CV: `TimeSeriesSplit(n_splits=5, gap=500)`, scoring = ROC-AUC  
- Drops features whose removal does not reduce out-of-fold AUC

### 3. Genetic Algorithm

Binary chromosome (1 = include feature, 0 = exclude).

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 20 |
| Fitness | Mean 3-fold TS AUC − 0.05 × feature fraction |
| Selection | Tournament (size 2) |
| Crossover | Single-point, p = 0.70 |
| Mutation | Bit-flip, p = 0.15 |

The GA includes macro variables as candidates specifically to test whether they survive selection at the 1-second horizon.

---

## Models

All models are interpretable; no deep learning is used.

### Binary fill probability models

| Model | Configuration | Role |
|---|---|---|
| **Logistic Regression** | `Pipeline([StandardScaler, LogReg(C=0.1, max_iter=1000)])` | Interpretable coefficient baseline |
| **Random Forest** | 200 trees, max depth 10, min samples per leaf 50 | Feature importances; robust non-linear baseline |
| **HistGradientBoosting (GBM)** | 200 iterations, max depth 4, lr 0.05, min samples per leaf 50 | Primary model; histogram-based GBM is orders of magnitude faster than vanilla GBM on million-row LOB data |
| **LightGBM** | Bayesian hyperparameter search via Optuna (50 trials), 1000 estimators with early stopping | Best-in-class GBM performance |

### Fractional fill models (parallel pipeline)

| Model | Role |
|---|---|
| **Ridge Regression** | Linear baseline with L2 regularisation |
| **Random Forest Regressor** | Non-linear baseline |
| **HistGradientBoosting Regressor** | Primary regressor |

### Calibration: Temperature Scaling

Post-hoc calibration using a single parameter *T*:

```
p_cal = sigmoid(logit(raw_prob) / T)
```

*T* is fitted on the validation set only. Using no intercept prevents the calibrator from encoding spurious base-rate shifts from validation-set idiosyncrasies.

---

## Evaluation Metrics

| Metric | Notes |
|---|---|
| **ROC-AUC** | Primary; robust to class imbalance (fill rate ≈ 13 % for CSCO, ≈ 52 % for AAPL) |
| **Brier Score** | Measures calibration quality |
| **Calibration curves** | Predicted fill probability vs actual fill rate per probability bin |
| **Aggregate calibration check** | Sum of predicted probabilities ≈ total actual fills |

**Reporting convention:** All metrics are reported as **day-level mean ± std**, not tick-level. Tick-level standard errors are severely underestimated because ~1.2 M intraday rows are not 1.2 M independent samples.

---

## Experiments

### Experiment 1 — Does intraday seasonality matter?

- **Model A:** Microstructure + vol_regime (no time features)  
- **Model B:** Microstructure + TIME + vol_regime  
- **Hypothesis:** B > A by 2–5 % day-level AUC due to open/close fill-rate seasonality

### Experiment 2 — Do macro regime variables add signal at the 1-second horizon?

- **Model A:** Microstructure + TIME  
- **Model B:** Microstructure + TIME + vol_regime + day_of_week  
- **Hypothesis:** Difference < 1 % AUC; macro state is already encoded by local microstructure at sub-second resolution  
- **Complementary test:** Regress Model-A residuals on (vol_regime, day_of_week) with HC0 robust standard errors; expect |t| < 2 if macro adds no signal

### Experiment 3 — Cross-stock generalisation

- Train on AAPL → evaluate on full CSCO dataset (and vice versa)  
- **Motivation:** One month of data per stock is insufficient to evaluate monthly/quarterly macro variation; cross-stock transfer acts as a robustness check. AAPL (large-tick, high-liquidity) and CSCO (lower-priced, different microstructure) are deliberately contrasting

---

## Testing & Evaluation Scripts

| Script | Purpose |
|---|---|
| [train.py](train.py) | Full binary pipeline: data loading, label construction, chronological split, feature selection (MI / RFECV / GA), model training, temperature scaling, experiment evaluation, model serialisation |
| [train_fractional.py](train_fractional.py) | Regression variant for the fractional fill target; mirrors the binary pipeline structure |
| [test_aapl.py](test_aapl.py) | In-stock holdout: evaluates all models on the AAPL 15 % test split; outputs ROC and calibration plots |
| [test_holdout.py](test_holdout.py) | Evaluates on the combined AAPL + CSCO test split; produces per-ticker and combined metrics plus three plots (ROC, calibration, per-day AUC bar chart) |
| [test_oos.py](test_oos.py) | Out-of-sample (OOS): evaluates AAPL-trained models on the entire CSCO dataset; saves per-prediction CSV |

---

## High-Level Estimator Interface

[fill_estimator.py](fill_estimator.py) exposes two usage modes.

**Low-level** (caller supplies pre-computed features):

```python
prob = estimator.predict_proba({
    "imbalance": 0.12, "depth_imbalance": 0.08, "queue_ahead": 300,
    "spread_norm": 0.0002, "local_vol": 0.015, "aggressive_flow": 3.0,
    "direction": 1, "order_size": 100,
    "time_sin": 0.45, "time_cos": 0.89,
    "time_bucket": 1, "day_of_week": 2,
})
```

**High-level** (convenience wrapper for the MM quoting loop):

```python
prob = estimator.estimate(
    date="2023-07-20", time_of_day="11:30:00",
    direction=1, order_size=100, queue_ahead=850,
    spread=0.0002, imbalance=0.12, depth_imbalance=0.08,
    local_vol=0.015, aggressive_flow=3.0,
)
ev = prob * spread_capture - (1 - prob) * inventory_cost
```

`predict_proba_batch(df)` is also available for backtesting over DataFrames.

---

## Outputs

### Saved models (`models/`)

```
fill_estimator.pkl            Primary FillEstimator (GBM + temperature calibrator)
logistic_regression.pkl
random_forest.pkl
gbm.pkl
lightgbm.pkl
ridge_fractional.pkl
random_forest_fractional.pkl
gbm_fractional.pkl
fill_estimator_fractional.pkl
temperature_calibrators.pkl   Dict of TemperatureScaler objects (one per model)
split_dates.json              Train/val/test date boundaries for reproducibility
```

### Plots (`plots/`)

```
mi_ranking.png                Feature ranking by mutual information
roc_curves.png                ROC curves on test set (all models)
calibration.png               Calibration curves on test set
importance_*.png              Feature importances per model
experiment1/2/3.png           Experiment comparisons with error bars
aapl_test_roc.png / _calibration.png
holdout_roc_curves.png / _calibration.png / _day_auc.png
oos_roc_curves.png / _calibration.png / _day_auc.png
frac_*.png                    Fractional fill model plots
```

### CSVs

```
test_predictions.csv          Per-order: date, direction, order_size, queue_ahead,
                              spread_norm, filled, prob_{model} for all models
holdout_predictions.csv       Same format, AAPL + CSCO combined test split
csco_oos_predictions.csv      Same format, full CSCO OOS dataset
order_size_fill_summary.csv   Fill stats by size tier and ticker
```

---

## Key Design Principles

1. **No data leakage** — chronological splits, purging, embargoing, time-series CV, all preprocessing fitted on training data only
2. **Interpretability over accuracy** — logistic regression coefficients, RF/GBM importances; no black-box deep learning
3. **Stationary features** — no raw prices or raw timestamps; every feature is bounded, normalised, or computed over a rolling window
4. **Honest uncertainty** — day-level mean ± std rather than misleading tick-level aggregates; sample uniqueness weighting
5. **Cross-stock robustness** — cross-ticker transfer test compensates for the short one-month data window
6. **Conservative calibration** — temperature scaling with no intercept avoids overfitting to the validation set's base rate
