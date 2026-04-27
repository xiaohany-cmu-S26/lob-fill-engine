# lob-fill-engine

Limit order fill probability estimator trained on LOBSTER tick data (AAPL, July 2023). Predicts P(your passive order at the best bid/ask is fully filled within 1 second) given order book microstructure, your order size, and intraday seasonality features, with walk-forward validation and Lopez-de-Prado-style purging/embargoing/uniqueness weighting.

Phase 3 of a larger LOB + Market Making simulator (Phases 1–2: matching engine + Avellaneda-Stoikov strategy; Phase 4: EV-based quoting using fill probability).

---

## Quick Start

```bash
# Train models, run experiments, generate all plots and CSVs
python train.py

# Launch interactive LOB visualizer (AAPL & CSCO, LOBSTER format)
python lob_visualizer.py   # → http://127.0.0.1:8050

# Use the estimator in the MM simulator (high-level convenience API)
from fill_estimator import FillEstimator
est = FillEstimator.load("models/fill_estimator.pkl")
prob = est.estimate(
    date="2023-07-20", time_of_day="11:30:00",
    direction=1,                # 1=buy, -1=sell
    order_size=100,             # your order size in shares
    queue_ahead=850,            # shares already in queue ahead of you
    spread=0.0002,              # (ask - bid) / mid
    imbalance=0.12, depth_imbalance=0.08,
    local_vol=0.015, aggressive_flow=3.0,
)
ev = prob * expected_spread_capture - (1 - prob) * inventory_cost
```

---

## Data

**Source:** [LOBSTER](https://lobsterdata.com) — NASDAQ tick data for AAPL (July 2023, ~21 trading days, 10 price levels).

Each trading day produces two aligned CSV files (no header):

| File | Columns |
|------|---------|
| `*_message_10.csv` | timestamp, event_type, order_id, size, price, direction |
| `*_orderbook_10.csv` | ask_p1, ask_v1, bid_p1, bid_v1, … (10 levels × 4 = 40 columns) |

- `event_type`: 1=new limit, 2=cancel partial, 3=cancel full, 4=exec visible, 5=exec hidden, 7=halt
- Prices: integer ticks ÷ 10 000 = USD
- Orderbook row k = LOB state **after** message row k fires

---

## Label Construction

A synthetic passive order is placed at the back of the queue at the best bid (buy) or best ask (sell) at each new limit order event (`event_type == 1`). The order's size matches the original message's size, so the dataset reflects the empirical distribution of submitted order sizes.

**Fill condition (1-second horizon, size-aware):**
- Compute the cumulative execution volume at-or-through the entry price within 1 second.
- Filled iff `cumulative_volume ≥ queue_ahead + order_size`.
- This is more accurate than the simpler "any execution touched the price" condition, because a 10-share order and a 10 000-share order at the same queue position have very different fill chances.

Implementation uses `numpy.searchsorted` for O(M log N) label construction instead of a naïve O(M·N) loop.

---

## Features

All features are computed at time t using only historical information (no future data). The table below documents the stationarity justification for each.

| Feature | Description | Stationary? |
|---------|-------------|-------------|
| `imbalance` | (bid_v1 − ask_v1) / (bid_v1 + ask_v1) | Yes — bounded [−1, 1] |
| `depth_imbalance` | Same ratio summed over top 3 levels | Yes — bounded [−1, 1] |
| `queue_ahead` | Volume at best level on order's own side | Yes — tick-relative |
| `spread_norm` | (ask_p1 − bid_p1) / mid_price | Yes — normalised by mid |
| `local_vol` | Rolling std of mid_price over past 30 s | Yes — local window |
| `aggressive_flow` | Count of executions in past 10 s | Yes — local window |
| `direction` | 1=buy, −1=sell | Yes — categorical |
| `order_size` | Your own order's size, in shares | Yes — distributional |
| `time_sin` | sin(2π × minute_from_open / 390) | Yes — cyclical encoding |
| `time_cos` | cos(2π × minute_from_open / 390) | Yes — cyclical encoding |
| `time_bucket` | 0=open, 1=mid-morning, 2=midday, 3=close | Yes — categorical |
| `vol_regime` | Binary: local_vol > training-set median | Yes — binary |
| `day_of_week` | 0=Mon … 4=Fri | Yes — categorical |

**Not used (non-stationary):** raw price levels (bid_p1, ask_p1, mid_price), raw timestamps as numerics.

`vol_regime` is the only feature requiring a training-set statistic. Its threshold is fitted on the training split only and then applied to validation and test — no leakage.

---

## Training Pipeline

```
discover_files → build_dataset               ← raw data loaded, stateless features computed
      ↓
make_splits (chronological 70/15/15)         ← NO fitting before this step
      ↓
fit_vol_regime(train) → apply_vol_regime     ← only cross-split statistic; train-only fit
      ↓
Feature selection (training set only)
  ├─ Mutual Information ranking
  ├─ RFECV (LogReg, TimeSeriesSplit n=5 gap=500)
  └─ Genetic Algorithm (LogReg fitness, TimeSeriesSplit n=3 gap=500)
      ↓
Primary features = GA ∪ RFECV
      ↓
Train: LogReg  |  Random Forest  |  GBM
      ↓
Evaluate: tick-AUC, day-level AUC ± std, Brier score, calibration curves
```

**Key constraints (no data leakage):**
- All splits are chronological by date — no random shuffling anywhere
- Feature selection is run only on the training set
- `vol_regime` threshold is fitted on training data only
- Purging: training rows whose 1-second label window extends into val are dropped
- Embargoing: first 30 seconds of each val/test period are dropped
- Cross-validation uses `TimeSeriesSplit(n_splits=5, gap=500)` throughout
- **Sample-uniqueness weights** (Lopez de Prado AFML Ch. 4) down-weight training rows whose label horizons overlap, so highly correlated samples don't inflate effective sample size

---

## Feature Selection

Three methods, all operating on the training set only:

1. **Mutual Information** (`mutual_info_classif`) — ranks all 12 features; primary proof that intraday features matter more than macro regime variables at 5-second resolution.

2. **RFECV** — recursive feature elimination with cross-validation. Base estimator: LogisticRegression. CV: `TimeSeriesSplit(n_splits=5, gap=500)`. Scoring: ROC-AUC.

3. **Genetic Algorithm** — binary-chromosome GA (one bit per feature). Fitness = mean TimeSeriesSplit AUC (LogReg) − 0.05 × feature_fraction. Tournament selection, single-point crossover, bit-flip mutation. Including regime features as candidates: if the GA drops them, that is empirical evidence that they add no signal at the 5-second horizon.

**Primary feature set:** conservative union of GA and RFECV selections.

---

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable coefficients; inside `Pipeline([StandardScaler, LogReg])` |
| Random Forest | n_estimators=200, max_depth=10, min_samples_leaf=50 |
| GBM (HistGradientBoosting) | max_iter=200, max_depth=4, lr=0.05 — histogram-based GBM, same quality as sklearn GBM but orders-of-magnitude faster on LOBSTER data |

Interpretability is prioritised over raw accuracy (no deep learning).

---

## Evaluation

| Metric | Why |
|--------|-----|
| ROC-AUC | Primary; robust to class imbalance |
| Brier score | Measures probability calibration |
| Calibration curve | Visual check: predicted fill prob vs. actual fill rate |

Both ROC-AUC and Brier are reported at the **day level** (mean ± std across trading days) in addition to tick-level. Tick-level standard errors are severely underestimated due to intraday autocorrelation; day-level aggregation gives honest uncertainty bands.

---

## Experiments

### Experiment 1 — Does intraday seasonality matter?
- Model A: MICRO + vol_regime (no time features)
- Model B: MICRO + TIME + vol_regime

Both models use GBM. Expected: B > A by 2–5 % AUC. Justifies including `time_sin`, `time_cos`, `time_bucket`.

### Experiment 2 — Do macro regime variables add signal?
- Model A: MICRO + TIME
- Model B: MICRO + TIME + vol_regime + day_of_week

Expected: negligible difference (<1 % AUC). At the 1-second fill horizon, macro effects are subsumed by local microstructure.

Complemented by a **residual regression**: regress `(filled − prob_A)` on `vol_regime` and `day_of_week` with HC0 robust standard errors. If macro variables had any explanatory power left over after Model A, the coefficients would be significant. They aren't — empirical confirmation.

### Experiment 3 — Cross-stock generalization (robustness check)
- Train on AAPL → test on CSCO (and vice versa) with the GBM on `MICRO + TIME + REGIME` features.

The one-month data scope is insufficient for monthly/macro regime variation; cross-stock generalization is the substitute robustness test. If the model carries most of its in-stock day-level AUC over to the held-out stock, it learned real microstructure rather than ticker-specific quirks. AAPL and CSCO are deliberately different (large-tick, high-liquidity AAPL vs. lower-priced CSCO).

---

## Outputs

```
plots/
  mi_ranking.png                   Feature ranking by mutual information
  roc_curves.png                   ROC curves for all 3 models (test set)
  calibration.png                  Calibration curves (test set)
  importance_random_forest.png     RF feature importances
  importance_gbm.png               GBM feature importances
  importance_logistic_regression.png  |LR coefficients|
  experiment1.png                  Exp 1: seasonality comparison
  experiment2.png                  Exp 2: regime variables comparison
  experiment3.png                  Exp 3: cross-stock generalization (AAPL ↔ CSCO)

models/
  fill_estimator.pkl               FillEstimator (GBM) — primary interface
  logistic_regression.pkl          Standalone LR model
  random_forest.pkl                Standalone RF model
  gbm.pkl                          Standalone GBM model

test_predictions.csv               One row per test-set order: timestamp, direction,
                                   order_size, queue_ahead, spread, prob_<model> from
                                   each of LR / RF / GBM, and the realized fill outcome
```

---

## FillEstimator Interface

```python
from fill_estimator import FillEstimator

est = FillEstimator.load("models/fill_estimator.pkl")

# 1. High-level convenience — pass intuitive market inputs
prob = est.estimate(
    date="2023-07-20", time_of_day="11:30:00",
    direction=1, order_size=100, queue_ahead=850, spread=0.0002,
    imbalance=0.12, depth_imbalance=0.08,
    local_vol=0.015, aggressive_flow=3.0,
)
# returns P(fill within est.fill_horizon_sec seconds)

# 2. Low-level — caller pre-computes every encoded feature
prob = est.predict_proba({
    "imbalance": 0.12, "depth_imbalance": 0.08, "queue_ahead": 300,
    "spread_norm": 0.0002, "local_vol": 0.015, "aggressive_flow": 3.0,
    "direction": 1, "order_size": 100,
    "time_sin": 0.45, "time_cos": 0.89,
    "time_bucket": 1, "day_of_week": 2,
})

# 3. Batch prediction for backtesting
probs = est.predict_proba_batch(feature_df)  # returns np.ndarray

# EV-based quoting in the MM simulator
ev = prob * expected_spread_capture - (1 - prob) * inventory_cost
```

---

## References

- Avellaneda & Stoikov (2008) — High-Frequency Trading in a Limit Order Book
- Guéant, Lehalle & Fernandez-Tapia (2013) — Dealing with the Inventory Risk
- Hasbrouck — Empirical Market Microstructure (spread decomposition)
- Cartea, Jaimungal & Penalva — Algorithmic and High-Frequency Trading, Ch. 1–2
- Lopez de Prado (2018) — Advances in Financial Machine Learning, Ch. 7 (purging/embargoing)
- Columbia Deep LOB (2021) — fill probability estimation methodology
