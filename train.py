"""
train.py — LOB Fill Probability Estimator: Full Training Pipeline
=================================================================
Run:  python train.py

Pipeline (strict no-leakage order)
-----------------------------------
1.  Load AAPL + CSCO LOBSTER data; combine and sort chronologically
2.  Build dataset: limit-order labels + stateless features per day
3.  Chronological split 70/15/15 with purging and embargoing — NO fitting before this step
4.  Fit vol_regime threshold on training set only; apply to all three splits
5.  Feature selection on training set only: MI ranking, RFECV, Genetic Algorithm
6.  Train three models on primary features (GA ∪ RFECV)
7.  Platt scaling: fit calibrator on validation set scores — fixes systematic
    probability bias without touching ranking (AUC unchanged)
8.  Evaluate: tick-AUC, day-level AUC ± std, Brier score (raw + calibrated)
9.  Generate all plots (plots/ directory)
10. Experiment 1 — with vs without intraday seasonality features
11. Experiment 2 — with vs without macro regime features
12. Save best model (GBM + Platt calibrator) as FillEstimator → models/fill_estimator.pkl
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lobster_data import (
    discover_files,
    build_dataset,
    make_splits,
    fit_vol_regime,
    apply_vol_regime,
    compute_uniqueness_weights,
    FILL_HORIZON_SEC,
)
from fill_estimator import FillEstimator

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
AAPL_DIR = os.path.join(
    BASE, "Data", "AAPL_2023-07-01_2023-07-31_10",
    "output-2023-07", "0", "0", "13",
)
CSCO_DIR = os.path.join(
    BASE, "Data", "CSCO_2023-07-01_2023-07-31_10",
    "output-2023-07", "0", "0", "75",
)
PLOTS_DIR  = os.path.join(BASE, "plots")
MODELS_DIR = os.path.join(BASE, "models")

# ── Feature groups ─────────────────────────────────────────────────────────────
MICRO_FEATURES  = [
    "imbalance", "depth_imbalance", "queue_ahead",
    "spread_norm", "local_vol", "aggressive_flow", "direction",
    "order_size",
]
TIME_FEATURES   = ["time_sin", "time_cos", "time_bucket"]
REGIME_FEATURES = ["vol_regime", "day_of_week"]
ALL_FEATURES    = MICRO_FEATURES + TIME_FEATURES + REGIME_FEATURES  # 12 total

# GA subsamples the last GA_MAX_ROWS training rows to bound runtime while
# preserving temporal order (no random shuffle).
GA_MAX_ROWS = 200_000


# ── Model factories ────────────────────────────────────────────────────────────

def _make_logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")),
    ])


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=50,
        n_jobs=-1, random_state=42,
    )


def _make_gbm() -> HistGradientBoostingClassifier:
    # HistGradientBoostingClassifier is sklearn's modern histogram-based GBM:
    # same quality as GradientBoostingClassifier but orders-of-magnitude faster
    # on large LOBSTER datasets (millions of tick rows).
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )


def _fit_with_weights(model, X, y, sample_weight=None):
    """
    Fit a model using sample_weight when supported. LogReg, RF and HistGBM
    all accept sample_weight, but the LR Pipeline needs the kwarg routed via
    the step name 'clf'.
    """
    if sample_weight is None:
        model.fit(X, y)
        return model
    if isinstance(model, Pipeline):
        model.fit(X, y, clf__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def day_level_metric(df: pd.DataFrame, metric: str,
                     prob_col: str = "prob") -> tuple[float, float]:
    """
    Per-day mean ± std of either ROC-AUC ('auc') or Brier score ('brier').

    Tick-level standard errors are badly underestimated due to autocorrelation.
    Aggregating at the day level gives honest uncertainty estimates.
    """
    vals = []
    for _, grp in df.groupby("date"):
        if metric == "auc":
            if grp["filled"].nunique() < 2:
                continue
            vals.append(roc_auc_score(grp["filled"], grp[prob_col]))
        elif metric == "brier":
            vals.append(brier_score_loss(grp["filled"], grp[prob_col]))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def evaluate(model, df: pd.DataFrame, features: list[str]) -> dict:
    """Compute tick-level AUC/Brier and day-level mean±std for both."""
    X = df[features].values
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["prob"] = probs
    tick_auc           = roc_auc_score(df["filled"], probs)
    tick_brier         = brier_score_loss(df["filled"], probs)
    day_auc_m, day_auc_s     = day_level_metric(df, "auc")
    day_brier_m, day_brier_s = day_level_metric(df, "brier")
    return {
        "tick_auc":       tick_auc,
        "tick_brier":     tick_brier,
        "day_auc_mean":   day_auc_m,
        "day_auc_std":    day_auc_s,
        "day_brier_mean": day_brier_m,
        "day_brier_std":  day_brier_s,
        # back-compat aliases
        "brier":          tick_brier,
        "probs":          probs,
        "df_with_prob":   df,
    }


# ── Feature selection ──────────────────────────────────────────────────────────

def select_mi(X_train: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, float]]:
    """Mutual information ranking (train set only)."""
    scores = mutual_info_classif(X_train, y_train, discrete_features=False,
                                  random_state=42)
    return sorted(zip(X_train.columns, scores), key=lambda x: x[1], reverse=True)


def select_rfecv(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    """RFECV with LogReg + TimeSeriesSplit(n_splits=5, gap=500)."""
    tscv = TimeSeriesSplit(n_splits=5, gap=500)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    rfe = RFECV(
        estimator=LogisticRegression(C=0.1, max_iter=500, solver="lbfgs"),
        cv=tscv,
        scoring="roc_auc",
        step=1,
        n_jobs=-1,
    )
    rfe.fit(X_scaled, y_train)
    return X_train.columns[rfe.support_].tolist()


def _ga_fitness(chrom: np.ndarray, X_np: np.ndarray, y_np: np.ndarray,
                tscv: TimeSeriesSplit, lambda_penalty: float = 0.05) -> float:
    """
    GA fitness: mean TimeSeriesSplit AUC (LogReg) minus feature-count penalty.
    Uses LogReg inside a scaler Pipeline for fair comparison across feature scales.
    """
    if chrom.sum() == 0:
        return -np.inf
    X_sub = X_np[:, chrom.astype(bool)]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=300, solver="lbfgs")),
    ])
    aucs = []
    for tr, vl in tscv.split(X_sub):
        pipe.fit(X_sub[tr], y_np[tr])
        if len(np.unique(y_np[vl])) < 2:
            continue
        aucs.append(roc_auc_score(y_np[vl], pipe.predict_proba(X_sub[vl])[:, 1]))
    if not aucs:
        return -np.inf
    return float(np.mean(aucs)) - lambda_penalty * (chrom.sum() / len(chrom))


def select_ga(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_generations: int = 20,
    pop_size: int = 20,
    mutation_rate: float = 0.15,
    lambda_penalty: float = 0.05,
    seed: int = 42,
) -> list[str]:
    """
    Binary-chromosome genetic algorithm.

    Fitness = mean TimeSeriesSplit(n_splits=3) AUC (LogReg) − lambda * feature_fraction.
    Uses last GA_MAX_ROWS training rows to bound runtime while preserving
    temporal order (no random shuffle).
    """
    rng = np.random.RandomState(seed)
    n_features = X_train.shape[1]
    feature_names = list(X_train.columns)
    tscv = TimeSeriesSplit(n_splits=3, gap=500)

    # Subsample tail of training data to bound runtime; preserves temporal order.
    if len(X_train) > GA_MAX_ROWS:
        X_ga = X_train.iloc[-GA_MAX_ROWS:].values.astype(np.float64)
        y_ga = y_train.iloc[-GA_MAX_ROWS:].values
        print(f"  (using last {GA_MAX_ROWS:,} training rows for GA fitness)")
    else:
        X_ga = X_train.values.astype(np.float64)
        y_ga = y_train.values

    # Initialise; ensure no all-zero chromosomes.
    pop = rng.randint(0, 2, (pop_size, n_features)).astype(np.int8)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i, rng.randint(n_features)] = 1

    best_fitness = -np.inf
    best_chrom   = pop[0].copy()

    for gen in range(n_generations):
        fits = np.array([_ga_fitness(c, X_ga, y_ga, tscv, lambda_penalty)
                         for c in pop])
        idx_best = int(fits.argmax())
        if fits[idx_best] > best_fitness:
            best_fitness = fits[idx_best]
            best_chrom   = pop[idx_best].copy()

        n_sel = int(pop[idx_best].sum())
        print(f"  Gen {gen+1:2d}/{n_generations}  "
              f"gen_best={fits[idx_best]:.4f}  "
              f"global_best={best_fitness:.4f}  "
              f"n_features={n_sel}")

        # Tournament selection (size 2)
        new_pop = []
        for _ in range(pop_size):
            i, j = rng.choice(pop_size, 2, replace=False)
            new_pop.append(pop[i].copy() if fits[i] >= fits[j] else pop[j].copy())

        # Single-point crossover (p = 0.70)
        for k in range(0, pop_size - 1, 2):
            if rng.rand() < 0.70:
                pt = rng.randint(1, n_features)
                tmp = new_pop[k][pt:].copy()
                new_pop[k][pt:]   = new_pop[k + 1][pt:]
                new_pop[k + 1][pt:] = tmp

        # Bit-flip mutation; prevent empty chromosomes
        for chrom in new_pop:
            flip = rng.rand(n_features) < mutation_rate
            chrom[flip] ^= 1
            if chrom.sum() == 0:
                chrom[rng.randint(n_features)] = 1

        pop = np.array(new_pop, dtype=np.int8)

    selected = [f for f, b in zip(feature_names, best_chrom) if b]
    print(f"  GA selected ({len(selected)}): {selected}")
    return selected


# ── Plotting ───────────────────────────────────────────────────────────────────

_BG      = "#0f172a"
_PANEL   = "#1e293b"
_GRID    = "#334155"
_TEXT    = "#e2e8f0"
_TITLE   = "#f8fafc"
_COLORS  = ["#3b82f6", "#10b981", "#f59e0b"]   # LR, RF, GBM


def _dark_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TITLE)
    ax.grid(color=_GRID, linestyle="--", linewidth=0.5)


def _save(fig, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  saved {path}")


def plot_mi_ranking(mi_ranked: list[tuple[str, float]]) -> None:
    features = [f for f, _ in mi_ranked]
    scores   = [s for _, s in mi_ranked]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _dark_ax(ax)
    ax.barh(features[::-1], scores[::-1], color="#3b82f6", edgecolor="none")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Feature Ranking — Mutual Information (training set)")
    plt.tight_layout()
    _save(fig, "mi_ranking.png")


def plot_roc_curves(models_dict: dict, test_df: pd.DataFrame,
                    features_dict: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=_BG)
    _dark_ax(ax)
    ax.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--")
    for (name, model), color in zip(models_dict.items(), _COLORS):
        feats = features_dict[name]
        probs = model.predict_proba(test_df[feats].values)[:, 1]
        fpr, tpr, _ = roc_curve(test_df["filled"], probs)
        auc = roc_auc_score(test_df["filled"], probs)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    plt.tight_layout()
    _save(fig, "roc_curves.png")


def plot_calibration_curves(models_dict: dict, test_df: pd.DataFrame,
                             features_dict: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=_BG)
    _dark_ax(ax)
    ax.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--",
            label="Perfect calibration")
    for (name, model), color in zip(models_dict.items(), _COLORS):
        feats = features_dict[name]
        probs = model.predict_proba(test_df[feats].values)[:, 1]
        frac_pos, mean_pred = calibration_curve(test_df["filled"], probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=1.8,
                markersize=5, label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive (Actual Fill Rate)")
    ax.set_title("Calibration Curves — Test Set")
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    plt.tight_layout()
    _save(fig, "calibration.png")


def plot_importances(model, feature_names: list[str], model_name: str) -> None:
    """Works for RF/GBM (feature_importances_) and LogReg Pipeline (|coef_|)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Feature Importance"
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if not hasattr(clf, "coef_"):
            return
        importances = np.abs(clf.coef_[0])
        xlabel = "|Coefficient|"
    else:
        return

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _dark_ax(ax)
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color="#10b981", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    fname = f"importance_{model_name.lower().replace(' ', '_')}.png"
    _save(fig, fname)


def plot_experiment(results: dict, title: str, filename: str) -> None:
    labels = list(results.keys())
    means  = [v["day_auc_mean"] for v in results.values()]
    stds   = [v["day_auc_std"]  for v in results.values()]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_BG)
    _dark_ax(ax)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=["#3b82f6", "#f59e0b"], edgecolor="none", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=_TEXT, fontsize=10)
    ax.set_ylabel("Day-Level ROC-AUC")
    ax.set_title(title)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.001,
                f"{mean:.4f}", ha="center", va="bottom",
                color=_TEXT, fontsize=9)
    plt.tight_layout()
    _save(fig, filename)


# ── Experiments ────────────────────────────────────────────────────────────────

def run_experiment_1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Does intraday seasonality matter?
    Model A: MICRO + vol_regime  vs  Model B: MICRO + TIME + vol_regime
    Expected: B > A by 2-5 %.
    """
    print("\n── Experiment 1: Intraday Seasonality ──────────────────────────────")
    results = {}
    configs = [
        ("Micro only",   MICRO_FEATURES + ["vol_regime"]),
        ("Micro + Time", MICRO_FEATURES + TIME_FEATURES + ["vol_regime"]),
    ]
    for label, feats in configs:
        model = _make_gbm()
        model.fit(train_df[feats], train_df["filled"])
        res = evaluate(model, test_df, feats)
        results[label] = res
        print(f"  {label:<20s}: AUC={res['day_auc_mean']:.4f}±{res['day_auc_std']:.4f}"
              f"  Brier={res['brier']:.4f}")
    plot_experiment(results,
                    "Experiment 1 — Intraday Seasonality (GBM, Test Set)",
                    "experiment1.png")
    return results


def run_experiment_2(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Do macro regime variables add signal at the 5-second horizon?
    Model A: MICRO + TIME  vs  Model B: MICRO + TIME + vol_regime + day_of_week
    Expected: negligible difference (<1 %). Macro effects are subsumed by
    local microstructure at 5-second resolution.
    """
    print("\n── Experiment 2: Macro Regime Variables ────────────────────────────")
    results = {}
    configs = [
        ("Micro + Time",          MICRO_FEATURES + TIME_FEATURES),
        ("Micro + Time + Regime", MICRO_FEATURES + TIME_FEATURES + REGIME_FEATURES),
    ]
    for label, feats in configs:
        model = _make_gbm()
        model.fit(train_df[feats], train_df["filled"])
        res = evaluate(model, test_df, feats)
        results[label] = res
        print(f"  {label:<28s}: AUC={res['day_auc_mean']:.4f}±{res['day_auc_std']:.4f}"
              f"  Brier={res['brier']:.4f}")
    plot_experiment(results,
                    "Experiment 2 — Macro Regime Variables (GBM, Test Set)",
                    "experiment2.png")

    # Complementary residual regression: regress Model-A residuals on the macro
    # regime variables. If macro variables had any explanatory power over what
    # microstructure misses, the regime coefficients would be statistically
    # significant. We expect them to be small and not significant.
    _residual_regression(results["Micro + Time"], test_df)
    return results


def _residual_regression(model_a_result: dict, test_df: pd.DataFrame) -> None:
    """
    Regress (filled - prob_A) on the macro regime variables on the test split.

    Reports OLS coefficients with their HC0 robust standard errors. If macro
    variables added information beyond Model A's microstructure+time features,
    we'd expect significant nonzero coefficients.
    """
    print("\n  Residual regression: (filled − prob_A) ~ vol_regime + day_of_week")
    df = model_a_result["df_with_prob"].copy()
    residual = (df["filled"].values - df["prob"].values).astype(np.float64)

    X = np.column_stack([
        np.ones(len(df)),
        df["vol_regime"].values.astype(np.float64),
        df["day_of_week"].values.astype(np.float64),
    ])
    y = residual

    # OLS via lstsq + HC0 robust SE (sandwich estimator)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_y  = y - X @ beta
    XtX_inv  = np.linalg.inv(X.T @ X)
    meat     = X.T @ (X * (resid_y ** 2)[:, None])
    cov      = XtX_inv @ meat @ XtX_inv
    se       = np.sqrt(np.diag(cov))
    tstat    = beta / np.where(se > 0, se, np.nan)

    names = ["intercept", "vol_regime", "day_of_week"]
    print(f"    {'name':<14}{'coef':>10}{'se':>10}{'t':>8}")
    for n, b, s, t_ in zip(names, beta, se, tstat):
        print(f"    {n:<14}{b:>10.5f}{s:>10.5f}{t_:>8.2f}")
    print("    (|t| < 2 ⇒ no detectable macro signal in residuals)")


def run_experiment_3(plots_dir: str) -> dict:
    """
    Cross-stock generalization (robustness check for the one-month data scope).

    Train on AAPL → test on CSCO, and vice versa.  If the model carries
    most of its AAPL AUC over to CSCO, we have evidence it learned real
    microstructure rather than AAPL-specific quirks.

    Each direction uses a chronological 80/20 train/test split on the
    *source* stock to fit, then a separate held-out test on the full set
    of *target*-stock days.  vol_regime is fitted on source training data.
    """
    print("\n── Experiment 3: Cross-Stock Generalization ────────────────────────")

    aapl_pairs = discover_files(AAPL_DIR)
    csco_pairs = discover_files(CSCO_DIR)
    if not aapl_pairs or not csco_pairs:
        print("  [skipped] need both AAPL and CSCO data on disk; skipping.")
        return {}
    print(f"  AAPL days: {len(aapl_pairs)}   CSCO days: {len(csco_pairs)}")

    print("  Building AAPL dataset …", end=" ", flush=True)
    t = time.time(); aapl = build_dataset(aapl_pairs, "AAPL")
    print(f"{len(aapl):,} rows  ({time.time()-t:.0f}s)")

    print("  Building CSCO dataset …", end=" ", flush=True)
    t = time.time(); csco = build_dataset(csco_pairs, "CSCO")
    print(f"{len(csco):,} rows  ({time.time()-t:.0f}s)")

    feats = MICRO_FEATURES + TIME_FEATURES + REGIME_FEATURES

    def _train_on(src: pd.DataFrame) -> tuple[HistGradientBoostingClassifier, float]:
        thresh = fit_vol_regime(src)
        src_w  = apply_vol_regime(src, thresh)
        m = _make_gbm()
        m.fit(src_w[feats], src_w["filled"])
        return m, thresh

    results = {}

    # AAPL → CSCO
    print("\n  Training on AAPL, testing on CSCO …")
    model_a, thresh_a = _train_on(aapl)
    csco_eval = apply_vol_regime(csco, thresh_a)
    res = evaluate(model_a, csco_eval, feats)
    print(f"    Day AUC = {res['day_auc_mean']:.4f} ± {res['day_auc_std']:.4f}   "
          f"Day Brier = {res['day_brier_mean']:.4f}")
    results["AAPL→CSCO"] = res

    # CSCO → AAPL
    print("  Training on CSCO, testing on AAPL …")
    model_c, thresh_c = _train_on(csco)
    aapl_eval = apply_vol_regime(aapl, thresh_c)
    res = evaluate(model_c, aapl_eval, feats)
    print(f"    Day AUC = {res['day_auc_mean']:.4f} ± {res['day_auc_std']:.4f}   "
          f"Day Brier = {res['day_brier_mean']:.4f}")
    results["CSCO→AAPL"] = res

    plot_experiment(results,
                    "Experiment 3 — Cross-Stock Generalization (GBM)",
                    "experiment3.png")
    return results


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    t_start = time.time()

    # ── Step 1: Load raw data (AAPL + CSCO combined) ──────────────────────────
    print("=" * 65)
    print("STEP 1 — Load AAPL + CSCO data and build combined dataset")
    print("=" * 65)
    t0 = time.time()

    aapl_pairs = discover_files(AAPL_DIR)
    print(f"  AAPL: {len(aapl_pairs)} trading days")
    df_aapl = build_dataset(aapl_pairs, "AAPL")

    csco_pairs = discover_files(CSCO_DIR)
    print(f"  CSCO: {len(csco_pairs)} trading days")
    df_csco = build_dataset(csco_pairs, "CSCO")

    df = (pd.concat([df_aapl, df_csco], ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True))

    n_days = df["date"].nunique()
    print(f"  Combined: {len(df):,} rows  ·  {n_days} unique dates  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Fill rate — AAPL: {df_aapl['filled'].mean():.3f}  "
          f"CSCO: {df_csco['filled'].mean():.3f}  "
          f"combined: {df['filled'].mean():.3f}")

    # ── Step 2: Chronological split — NO fitting before this ──────────────────
    print("\n── Chronological split (70 / 15 / 15 by date, purge + embargo) ──────")
    train_raw, val_raw, test_raw = make_splits(df)
    for name, split in [("Train", train_raw), ("Val", val_raw), ("Test", test_raw)]:
        print(f"  {name:<5}: {len(split):>8,} rows  "
              f"{split['date'].nunique():>2} days  "
              f"{split['date'].min()} → {split['date'].max()}  "
              f"fill={split['filled'].mean():.3f}")

    # ── Step 3: Fit vol_regime on TRAINING SET ONLY, then apply ───────────────
    print("\n── Vol-regime threshold (fitted on training set only) ───────────────")
    vol_thresh = fit_vol_regime(train_raw)
    print(f"  threshold (median local_vol in train) = {vol_thresh:.6f}")
    train_df = apply_vol_regime(train_raw, vol_thresh)
    val_df   = apply_vol_regime(val_raw,   vol_thresh)
    test_df  = apply_vol_regime(test_raw,  vol_thresh)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["filled"]

    # ── Step 4: Feature selection (training set only) ─────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Feature selection (training set only)")
    print("=" * 65)

    print("\n2a. Mutual Information")
    mi_ranked = select_mi(X_train, y_train)
    print(f"  {'Feature':<24}  MI score")
    for feat, score in mi_ranked:
        print(f"  {feat:<24}  {score:.4f}")
    plot_mi_ranking(mi_ranked)

    print("\n2b. RFECV (LogReg + TimeSeriesSplit n=5 gap=500)")
    t1 = time.time()
    rfecv_feats = select_rfecv(X_train, y_train)
    print(f"  selected ({len(rfecv_feats)}): {rfecv_feats}  [{time.time()-t1:.0f}s]")

    print("\n2c. Genetic Algorithm (n_generations=20, pop_size=20)")
    print("  This may take several minutes …")
    t2 = time.time()
    ga_feats = select_ga(X_train, y_train,
                          n_generations=20, pop_size=20,
                          mutation_rate=0.15, lambda_penalty=0.05)
    print(f"  GA completed in {time.time()-t2:.0f}s")

    # Primary feature set: conservative union of both selection methods
    primary_features = sorted(
        set(ga_feats) | set(rfecv_feats),
        key=ALL_FEATURES.index,
    )
    print(f"\n  Primary features (GA ∪ RFECV, {len(primary_features)}): {primary_features}")

    # ── Step 5: Train models ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Train models on primary feature set")
    print("=" * 65)

    models = {
        "Logistic Regression": _make_logreg(),
        "Random Forest":       _make_rf(),
        "GBM":                 _make_gbm(),
    }
    features_dict = {name: primary_features for name in models}

    # Sample uniqueness weights (Lopez de Prado AFML Ch.4) — down-weight rows
    # whose label-windows overlap so highly correlated samples don't dominate.
    print("\n  Computing sample-uniqueness weights …", end=" ", flush=True)
    t_w = time.time()
    train_sw = compute_uniqueness_weights(train_df, fill_horizon_sec=FILL_HORIZON_SEC)
    print(f"done ({time.time()-t_w:.0f}s) — mean weight={train_sw.mean():.4f}")

    X_tr = train_df[primary_features]
    for name, model in models.items():
        t3 = time.time()
        print(f"  Fitting {name} …", end=" ", flush=True)
        _fit_with_weights(model, X_tr, y_train, sample_weight=train_sw)
        print(f"done ({time.time()-t3:.0f}s)")

    # ── Step 6: Platt scaling — fit on validation set ─────────────────────────
    # A LogisticRegression maps raw model scores → calibrated probabilities.
    # Fitted on val (not train) so the calibration sees out-of-sample scores.
    # AUC is rank-invariant and won't change; bias ratio should approach 1.0.
    print("\n" + "=" * 65)
    print("STEP 4 — Platt calibration (fitted on validation set)")
    print("=" * 65)

    calibrators: dict[str, LogisticRegression] = {}
    X_val_primary = val_df[primary_features]
    for name, model in models.items():
        raw_val = model.predict_proba(X_val_primary)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        platt.fit(raw_val, val_df["filled"])
        calibrators[name] = platt
        # Quick sanity: calibrated bias ratio on val
        cal_val = platt.predict_proba(raw_val)[:, 1]
        ratio = cal_val.sum() / val_df["filled"].sum()
        print(f"  {name:<22}  val bias ratio after calibration: {ratio:.4f}")

    joblib.dump(calibrators, os.path.join(MODELS_DIR, "platt_calibrators.pkl"))
    print(f"  Saved calibrators → models/platt_calibrators.pkl")

    # ── Step 7: Evaluate ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5 — Evaluation")
    print("=" * 65)

    header = (f"  {'Model':<22} {'Tick AUC':>9} {'Day AUC':>9} {'±Std':>7} "
              f"{'Tick Brier':>11} {'Day Brier':>10} {'±Std':>7}")
    sep    = "  " + "-" * 81

    test_results: dict[str, dict] = {}
    for split_name, split_df in [("Validation", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} set (raw scores):")
        print(header)
        print(sep)
        for name, model in models.items():
            r = evaluate(model, split_df, primary_features)
            print(f"  {name:<22} {r['tick_auc']:>9.4f} {r['day_auc_mean']:>9.4f} "
                  f"{r['day_auc_std']:>7.4f} {r['tick_brier']:>11.4f} "
                  f"{r['day_brier_mean']:>10.4f} {r['day_brier_std']:>7.4f}")
            if split_name == "Test":
                test_results[name] = r

        # Calibrated metrics — AUC unchanged, Brier and bias ratio improve
        print(f"\n  {split_name} set (Platt calibrated):")
        print(header)
        print(sep)
        for name, model in models.items():
            raw_probs = model.predict_proba(split_df[primary_features])[:, 1]
            cal_probs = calibrators[name].predict_proba(
                raw_probs.reshape(-1, 1))[:, 1]
            tmp = split_df[["date", "filled"]].copy()
            tmp["prob"] = cal_probs
            tick_auc   = roc_auc_score(split_df["filled"], cal_probs)
            tick_brier = brier_score_loss(split_df["filled"], cal_probs)
            day_auc_m, day_auc_s     = day_level_metric(tmp, "auc")
            day_brier_m, day_brier_s = day_level_metric(tmp, "brier")
            bias_ratio = cal_probs.sum() / split_df["filled"].sum()
            print(f"  {name:<22} {tick_auc:>9.4f} {day_auc_m:>9.4f} "
                  f"{day_auc_s:>7.4f} {tick_brier:>11.4f} "
                  f"{day_brier_m:>10.4f} {day_brier_s:>7.4f}  "
                  f"bias={bias_ratio:.3f}")

    # Save per-order test-set predictions for downstream sanity-checking and
    # for the MM-simulator backtest. One row per limit order placed during the
    # test split with predicted probability from each model and the realized
    # fill outcome.
    pred_df = test_df[
        ["date", "timestamp", "direction", "order_size", "entry_price",
         "queue_ahead", "spread_norm", "filled"]
    ].copy().reset_index(drop=True)
    for name, r in test_results.items():
        col = f"prob_{name.lower().replace(' ', '_')}"
        pred_df[col] = r["probs"]
    pred_path = os.path.join(BASE, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Test predictions saved → {pred_path}")

    # ── Step 8: Plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6 — Plots")
    print("=" * 65)

    plot_roc_curves(models, test_df, features_dict)
    plot_calibration_curves(models, test_df, features_dict)
    for name, model in models.items():
        plot_importances(model, primary_features, name)

    # ── Step 9: Experiments ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 7 — Experiments")
    print("=" * 65)
    run_experiment_1(train_df, test_df)
    run_experiment_2(train_df, test_df)
    run_experiment_3(PLOTS_DIR)

    # ── Step 10: Save best model ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 8 — Save FillEstimator")
    print("=" * 65)

    best = FillEstimator(
        model=models["GBM"],
        feature_names=primary_features,
        vol_threshold=vol_thresh,
        fill_horizon_sec=FILL_HORIZON_SEC,
        calibrator=calibrators["GBM"],
    )
    est_path = os.path.join(MODELS_DIR, "fill_estimator.pkl")
    best.save(est_path)
    print(f"  FillEstimator (GBM) → {est_path}")

    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        p = os.path.join(MODELS_DIR, fname)
        joblib.dump(model, p)
        print(f"  {name} → {p}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 65}")
    print(f"Training pipeline complete.  Total runtime: {elapsed:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
