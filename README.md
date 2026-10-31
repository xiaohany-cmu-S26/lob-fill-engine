# lob-fill-engine

A calibrated, interpretable estimator of **P(fill ≤ 3s)** for passive limit orders placed at the best bid or ask, trained on LOBSTER NASDAQ tick data (AAPL + CSCO, July 2023) with time-series-safe methodology: chronological splits, purging, embargoing, and sample-uniqueness weighting.

Phase 3 of a four-phase Limit Order Book + Market Making simulator. The estimator is the empirical fill layer that feeds an EV-based quoting strategy:

```
EV = P(fill) × spread_capture − (1 − P(fill)) × adverse_cost
```

---

## Headline Results

| Model | Test Day-AUC | Test Day-Brier |
|---|---|---|
| Logistic Regression | 0.7489 | 0.1011 |
| Random Forest | 0.7653 | 0.0986 |
| **HistGradientBoosting (primary)** | **0.7798** | **0.0979** |
| LightGBM (Optuna, 50 trials) | 0.7795 | 0.0972 |

Temperature-calibrated, day-level mean across held-out trading days.

**Out-of-sample on the full CSCO dataset (17 days):** LightGBM 0.7942 tick-AUC, 0.7910 ± 0.0240 day-AUC; GBM 0.7885 ± 0.0243 day-AUC. Per-day AUC is stable across all 17 days.

| Experiment | Result |
|---|---|
| 1 — Intraday seasonality | +0.0076 day-AUC (0.7773 → 0.7849) |
| 2 — Macro regime variables | +0.0005 day-AUC (0.7849 → 0.7854): statistically significant in residuals (t = −8.4, 35.9) but economically negligible |
| 3 — Cross-stock generalization | AAPL→CSCO 0.7468, CSCO→AAPL 0.7603 day-AUC |

---

## Quick Start

```bash
pip install pandas numpy scikit-learn lightgbm optuna statsmodels matplotlib seaborn joblib

python train.py          # full pipeline: labels, splits, feature selection, models, experiments, plots
python test_oos.py       # out-of-sample evaluation on the full CSCO dataset
python lob_visualizer.py # interactive LOB visualizer
```

LOBSTER data is not included (licensed). Place the `*_message_10.csv` / `*_orderbook_10.csv` files under `Data/`.

```python
from fill_estimator import FillEstimator

est = FillEstimator.load("models/fill_estimator.pkl")
prob = est.predict_proba_batch(feature_df)   # P(fill within est.fill_horizon_sec)
```

---

## Data

LOBSTER NASDAQ tick data, July 2023, depth 10.

- **22 stock-days:** AAPL 5 days, CSCO 17 days; 8.7M labeled rows
- Nanosecond timestamps; ~2.35M events on a single AAPL day (~90k events/sec)
- Message file: timestamp, event_type, order_id, size, price, direction
- Orderbook file: 10 levels × (ask price, ask size, bid price, bid size) = 40 columns
- Cleaning: drop halts (type 7), drop crossed books, trim opening-auction rows

**Coverage note:** AAPL message files exist only for Jul 14, 17–20, so the chronological 70/15/15 split puts 3 AAPL days in train, 2 in validation, and 0 in test. Main-pipeline test metrics are therefore CSCO-only; Experiment 3 builds the AAPL and CSCO datasets independently for a true cross-ticker test.

---

## Label Construction

At each new limit-order event, a synthetic passive order is placed at the back of the queue at the best bid (buy) or best ask (sell), sized to the original order.

**Size-aware fill condition (3-second horizon):** the order is filled iff cumulative executed volume at or through the entry price within 3 seconds reaches `queue_ahead + order_size`. This is stricter and more realistic than "price touched my level", which ignores queue position.

Labels are built with `numpy.searchsorted` in O(M log N). A fractional-fill variant (`train_fractional.py`, 1-second horizon) regresses the filled fraction of the order.

---

## Features

13 candidate features, all stationary (bounded, normalized, or rolling-window); no raw prices or timestamps.

| Group | Features |
|---|---|
| Microstructure | imbalance, depth_imbalance, queue_ahead, spread_norm, local_vol, aggressive_flow, order_size, direction |
| Intraday | time_sin, time_cos, time_bucket |
| Regime | vol_regime (threshold fitted on train only), day_of_week |

**Final selection (8, GA ∪ RFECV):** queue_ahead, spread_norm, local_vol, aggressive_flow, order_size, time_sin, time_cos, time_bucket.

**Dropped (5):** imbalance, depth_imbalance, direction, vol_regime, day_of_week. `direction` ranked #1 by mutual information yet was dropped as conditionally redundant: marginal information is not the same as conditional contribution.

---

## Methodology

- Chronological 70/15/15 split by date, no shuffling
- **Purging:** drop training rows whose 3-second label window crosses a split boundary
- **Embargo:** drop the first 30 seconds of each downstream period (matches the longest feature lookback)
- `TimeSeriesSplit(n_splits=5, gap=500)` for all cross-validation
- Sample-uniqueness weights for overlapping label windows (López de Prado, *AFML*)
- Feature selection on training data only: mutual information, RFECV, and a custom genetic algorithm (fitness = AUC − 0.05 × feature fraction)
- Temperature scaling fitted on validation (single parameter, no intercept, rank-invariant)
- Metrics reported at the day level (mean ± std), since tick rows are autocorrelated

---

## Repository Layout

```
train.py                    Binary pipeline: data, labels, splits, selection, models, experiments
train_fractional.py         Fractional-fill regression pipeline
lobster_data.py             LOBSTER loading, cleaning, feature + label construction
fill_estimator.py           FillEstimator interface (load / predict_proba / estimate)
fill_estimator_fractional.py
test_oos.py                 Out-of-sample evaluation on full CSCO dataset
test_holdout.py, test_aapl.py
lob_visualizer.py           Interactive order book visualizer
order_size_summary.py       Fill statistics by order-size tier
models/                     Trained models + temperature calibrators + split dates
plots/                      ROC, calibration, MI ranking, importances, experiments, OOS
docs/                       Project report, paper, and presentation deck
```

---

## Limitations

- Uneven AAPL coverage (0 test days); cross-ticker evidence comes from Experiment 3
- Synthetic order placement is a proxy for true queue position
- Binary label discards time-to-fill information; survival framing is future work
- Static model, no online updating
- One month of data cannot capture longer macro regimes

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, Optuna, statsmodels, matplotlib, seaborn, joblib.

## References

- Avellaneda & Stoikov (2008), High-Frequency Trading in a Limit Order Book
- Cartea, Jaimungal & Penalva, Algorithmic and High-Frequency Trading
- Hasbrouck, Empirical Market Microstructure
- López de Prado (2018), Advances in Financial Machine Learning
- Maglaras, Moallemi & Wang (2022), A Deep Learning Approach to Estimating Fill Probabilities in a Limit Order Book
- Arroyo et al. (2024), Deep Attentive Survival Analysis in Limit Order Books
