# Empirical Fill Probability for Limit Orders in a Limit Order Book

**CMU Data Science Final Project — David Yu**

---

## 1. Problem and Motivation

A market maker (MM) places passive limit orders at the best bid or ask and earns the bid-ask spread when those orders are filled. The expected value of any quote is

$$\mathrm{EV} \;=\; P_{\text{fill}} \cdot \pi_{\text{spread}} \;-\; (1 - P_{\text{fill}}) \cdot c_{\text{inventory}}$$

so the MM strategy is only as good as its estimate of the fill probability $P_{\text{fill}}$. Closed-form models (Avellaneda–Stoikov 2008; Guéant, Lehalle & Fernandez-Tapia 2013) give an analytic shape but are calibrated to a homogeneous Poisson arrival process and ignore queue position, spread, and local volatility. The aim of this project is to replace that closed-form term with an **empirical, data-driven fill probability estimator** that can plug into the same EV-based quoting rule.

Concretely: given the LOB state at time $t$, the order's side, and its size, predict
$$P\bigl(\text{order is fully filled within } H \text{ seconds}\bigr).$$
We use $H = 1$ s. The estimator is the Phase-3 component of a four-phase LOB+MM simulator (matching engine → AS strategy → fill model → EV-based quoter).

---

## 2. Data and Label Construction

### 2.1 Data
LOBSTER NASDAQ tick data, July 2023, two stocks with deliberately different microstructure: **AAPL** (large-tick, high-liquidity) and **CSCO** (lower-priced, different queue dynamics). One month per stock — five paired AAPL message+orderbook days, seventeen paired CSCO days. Each day produces two row-aligned CSVs: a message file (timestamp, event_type, order_id, size, price, direction) and an orderbook file with the top ten levels of both sides (40 columns).

The data scope (one month) is sufficient for **intraday** and **day-of-week** seasonality, but **not** for monthly/quarterly or macro-regime variation. We address that limitation explicitly with the cross-stock robustness check in Experiment 3.

### 2.2 Synthetic limit orders
At every new-limit-order event (`event_type == 1`) we place a *synthetic* passive order at the back of the queue at the best bid (if direction = buy) or best ask (if sell) with size equal to the original message's order size. This makes the dataset reflect the empirical *distribution* of real submitted sizes rather than any artificial fixed size.

### 2.3 Size-aware fill label
A naive label sets `filled = 1` if any execution touches the entry price within the horizon. That gives a 10-share order and a 10 000-share order the same label, which is wrong: the small order has a much higher chance of clearing.

We use a **size-aware** label: let $Q$ be the queue ahead at submission (`bid_v1` for buys, `ask_v1` for sells) and $S$ the order's own size. Let $V_t$ denote the cumulative execution volume at-or-through the entry price within $H$ seconds. Then
$$\text{filled} \;=\; \mathbf{1}\bigl[V_t \;\geq\; Q + S\bigr].$$
This is the smallest amount of through-traffic that would clear the queue ahead and then fill the order itself.

Implementation uses `numpy.searchsorted` on the timestamp array, giving $O(M \log N)$ rather than the $O(M\,N)$ of a naïve nested loop.

---

## 3. Feature Engineering

All features are computed strictly from information available at time $t$ — no future data, no rolling windows that span split boundaries, and every feature is justified to be approximately stationary over the trading day.

| Feature | Definition | Stationarity rationale |
|---|---|---|
| `imbalance` | $(b_1 - a_1)/(b_1 + a_1)$ | Bounded $[-1,1]$ |
| `depth_imbalance` | Same ratio summed over levels 1–3 | Bounded $[-1,1]$ |
| `queue_ahead` | Top-of-book volume on order's own side | Tick-relative |
| `spread_norm` | $(a_1 - b_1)/m$, $m$ = mid | Normalized by mid |
| `local_vol` | std of mid over past 30 s | Local-window |
| `aggressive_flow` | count of executions in past 10 s | Local-window |
| `direction` | $\pm 1$ | Categorical |
| `order_size` | Order size in shares | Empirical distribution stable |
| `time_sin`, `time_cos` | $\sin / \cos(2\pi \cdot \text{min\_from\_open}/390)$ | Cyclical encoding |
| `time_bucket` | open / mid-morning / midday / close | Categorical |
| `vol_regime` | $\mathbf{1}[\text{local\_vol} > \text{train median}]$ | Binary; threshold trained-only |
| `day_of_week` | 0–4 | Categorical |

**Excluded** as non-stationary: raw price levels (`bid_p1`, `ask_p1`, `mid_price`) and raw timestamps treated as numbers. Time-of-day is encoded *cyclically* (sin/cos) rather than as raw minutes — this matters because, e.g., 09:31 and 15:59 should be far apart in feature space, and a raw scalar would treat them as 89 minutes apart on one side and ~1370 on the other.

`vol_regime` is the only feature requiring a cross-split statistic. Its threshold is the **median of `local_vol` on the training set only**, then applied unchanged to validation and test. Re-fitting it on the full set would leak future information.

---

## 4. Feature Selection Methodology

Three methods, all adapted from Assignment 2 to time-series data — i.e. with chronological cross-validation rather than random k-fold. CV uses `TimeSeriesSplit(n_splits=5, gap=500)`; the gap of 500 events approximates the embargo buffer (Lopez de Prado 2018, Ch. 7). All three are run on the **training set only**.

1. **Mutual information** (`mutual_info_classif`). Ranks every feature against the label. Primary qualitative evidence for the seasonality vs. macro question — it gives a direct, model-free answer to "which features carry the most signal."
2. **RFECV** with logistic-regression base estimator and `TimeSeriesSplit` CV; ROC-AUC scoring. Drops features whose removal does not hurt out-of-fold AUC.
3. **Genetic algorithm** (binary chromosome, one bit per feature). Fitness = mean `TimeSeriesSplit` AUC of a logistic-regression pipeline minus a feature-count penalty $\lambda \cdot (k/n)$. Tournament selection (size 2), single-point crossover (p = 0.70), bit-flip mutation (p = 0.15), 20 generations × 20 population. Crucially, the macro regime variables are *included* in the candidate set so we can read off whether GA discards them.

The model's **primary feature set** is the union of the GA selection and the RFECV selection — a conservative choice that keeps any feature picked by either method.

**Purging and embargoing.** Following Lopez de Prado (AFML Ch. 7), we (i) *purge* training rows whose 1-second label window extends past the validation start, and (ii) *embargo* the first 30 s of each validation/test period to keep the bleed-through information from contaminating the eval split.

**Sample uniqueness weighting.** Two synthetic orders submitted within $H$ seconds of each other share most of the same look-forward stream — their labels are highly correlated, so treating them as i.i.d. inflates the effective sample size. We compute Lopez de Prado's averaged-uniqueness weight $w_i = 1/N_i$ where $N_i$ is the count of overlapping label windows, and pass these as `sample_weight` into all three classifiers.

---

## 5. Model Training and Evaluation

### 5.1 Models (all interpretable)
1. **Logistic regression** inside a `StandardScaler` pipeline — interpretable coefficients.
2. **Random Forest** — `n_estimators=200, max_depth=10, min_samples_leaf=50`. Gives feature-importance signal that is robust to feature scaling.
3. **HistGradientBoosting** (sklearn) — `max_iter=200, max_depth=4, learning_rate=0.05`. Histogram-based GBM is qualitatively equivalent to vanilla GBM but orders of magnitude faster on multi-million-row tick data.

No deep learning. Interpretability is more important than the last 1 % of AUC for this use case: in a prop-shop interview I have to be able to explain *why* the model gives the answer it does, and an LR coefficient table or RF/GBM importance ranking lets me do that.

### 5.2 Splits
Chronological 70 / 15 / 15 split by date; no random shuffling anywhere in the pipeline. Validation is used for training-time decisions (selection, calibration sanity-check); test is the held-out final evaluation.

### 5.3 Metrics
- **ROC-AUC** as the primary metric (robust to class imbalance, which is severe — at $H=1$ s the fill rate is dominated by short-queue / aggressive-flow regimes).
- **Brier score** for calibration.
- Both metrics are reported **per-day mean ± std** in addition to tick-level. Tick-level standard errors are wildly underestimated due to intraday autocorrelation — a 1.2 M-row test set is not 1.2 M independent samples. Day-level aggregation is the honest version.
- **Calibration curve** for visual sanity-check.

---

## 6. Experiments

### 6.1 Experiment 1 — Does intraday seasonality matter?

**Setup.** Two GBM models on the same training data:

- Model A (microstructure only): `MICRO + vol_regime`
- Model B (microstructure + intraday): `MICRO + TIME + vol_regime`

**Hypothesis.** B beats A by 2–5 % day-level AUC, because a 09:30 quote has a fundamentally different fill landscape than a 15:30 quote (open auction overhang vs. close-of-day order imbalance).

**What to look for in the output.** `experiment1.png` shows the day-level AUC bar chart with std error bars; the magnitude of the gap is the answer.

### 6.2 Experiment 2 — Do macro regime variables add signal at $H = 1$ s?

**Setup.** Two GBM models on the same training data:

- Model A: `MICRO + TIME`
- Model B: `MICRO + TIME + vol_regime + day_of_week`

**Hypothesis.** Negligible difference (< 1 % AUC). At a 1-second horizon, macro-regime effects are subsumed by what local microstructure (queue depth, imbalance, recent aggressive flow, local volatility) already encodes.

**Complementary residual regression.** As an additional check we regress the Model-A residuals $(\text{filled}_i - \hat{p}_{A,i})$ on $(\text{vol\_regime}_i, \text{day\_of\_week}_i)$ with HC0 robust standard errors. If macro variables had any explanatory power left over after Model A, the regime coefficients would be statistically significant. If $|t| < 2$ on both, that's empirical evidence that macro-regime variables add nothing the microstructure features didn't already capture.

This is the empirical case for *not* reaching for VIX or other macro features at the 1-second timescale, despite the one-month data limitation.

### 6.3 Experiment 3 — Cross-stock generalization

**Setup.** Train a GBM on AAPL, test on CSCO. Then train on CSCO, test on AAPL. Both directions use the same `MICRO + TIME + REGIME` feature set. `vol_regime` thresholds are fitted on the source-stock training data only.

**Hypothesis.** The model carries most of its in-stock day-level AUC over to the held-out stock. AAPL and CSCO have *deliberately different* microstructures (large-tick high-liquidity vs. lower-priced), so positive cross-stock transfer means the model learned real microstructure dynamics rather than ticker-specific quirks.

**Why this matters.** With only one month of data per stock, we cannot test monthly/macro regime variation directly. Cross-stock generalization is the practical substitute robustness test: if the AAPL-trained model works on CSCO with similar AUC, the *macro state* of July 2023 is not what's being learned — the *local microstructure* is.

---

## 7. Conclusion and Limitations

### 7.1 What this project shows

1. A size-aware fill label combined with queue-ahead and spread features predicts 1-second fill probability well enough to drive an EV-based quoter (see day-level AUC in test results).
2. Intraday seasonality features (`time_sin`, `time_cos`, `time_bucket`) add measurable lift over pure microstructure (Experiment 1).
3. Macro-regime variables (`vol_regime`, `day_of_week`) add essentially no additional signal at $H = 1$ s, both by direct AUC comparison (Experiment 2) and by residual regression (insignificant macro coefficients in HC0-robust OLS).
4. The model generalizes across two stocks with very different microstructure (Experiment 3), suggesting it learned real LOB dynamics rather than month-specific or ticker-specific noise.

### 7.2 Limitations explicitly

- **One-month data scope** is insufficient to test monthly / quarterly / macro-regime variation directly. Experiment 3 is the substitute robustness test, not a replacement for multi-year data.
- **Synthetic-order assumption.** We assume a passive order placed at the back of the queue at the moment of a real new-limit-order event. The real-world MM quoting cadence is not event-driven in the same way, but the resulting feature distribution is matched.
- **No adverse-selection cost.** A high $P_{\text{fill}}$ can mean the price is about to move against you. The current model does not separately model adverse selection; the EV expression in §1 only handles spread vs. inventory, not toxic flow. This is a deliberate Phase-3 scope cut and is meant to be tackled in Phase 4.
- **Single-venue.** LOBSTER is NASDAQ-only. A real MM has to reason about cross-venue fragmentation; that is out of scope here.
- **Sample autocorrelation is mitigated, not eliminated.** Purging, embargoing and uniqueness weights handle the worst of it, but consecutive submissions still share most of their lookahead stream — the day-level standard errors in §5.3 are the honest reporting of that.

### 7.3 Plug-in to the MM simulator

The fitted model is wrapped in a stateless `FillEstimator` class (`fill_estimator.py`) that exposes both a low-level `predict_proba(state_dict)` and a high-level `estimate(date, time_of_day, …)` convenience API. It depends only on `numpy`, `pandas`, `joblib`, and the saved sklearn model — the matching engine and Avellaneda-Stoikov strategy can import it without pulling in the training stack.

---

## References

- Avellaneda, M., & Stoikov, S. (2008). High-Frequency Trading in a Limit Order Book. *Quantitative Finance*.
- Cartea, Á., Jaimungal, S., & Penalva, J. *Algorithmic and High-Frequency Trading*, Cambridge, Ch. 1–2.
- Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). Dealing with the Inventory Risk. *Mathematics and Financial Economics*.
- Hasbrouck, J. *Empirical Market Microstructure*, Oxford.
- Kolm, P. N., Turiel, J., & Westray, N. (2021). Deep Order Flow Imbalance. (Columbia Deep LOB.)
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley, Ch. 4 (sample weights / uniqueness) and Ch. 7 (purging / embargoing).
