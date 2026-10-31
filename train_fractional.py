"""
train_fractional.py — LOB Fill Fraction Estimator: Regression Pipeline
=======================================================================
Run:  python train_fractional.py

Companion to train.py (binary fill/cancel classifier).
Predicts E[fill_fraction] ∈ [0, 1] — what fraction of your passive order
is absorbed by executions within 1 second — using regression models.

Why this matters
----------------
Binary labels collapse a 40 %-filled order and a 0 %-filled order into
the same "unfilled" bucket.  The fractional target is:

    fill_fraction = clip((cum_exec_vol_at_price − queue_ahead) / order_size, 0, 1)

For the MM EV calculation this is directly useful:
    ev = fill_fraction * spread_capture − (1 − fill_fraction) * inventory_cost

Pipeline (strict no-leakage order)
-----------------------------------
1.  Load AAPL LOBSTER data with fractional fill labels
2.  Chronological split 70/15/15 — NO fitting before this step
3.  Fit vol_regime threshold on training set only; apply to all splits
4.  Feature selection on training set only:
      MI (mutual_info_regression), RFECV (Ridge, r²), GA (Ridge fitness)
5.  Train Ridge · RandomForestRegressor · HistGradientBoostingRegressor
6.  Evaluate: MAE, RMSE, R² — tick-level and day-level mean ± std
7.  Plots → plots/frac_*.png
8.  Experiment 1 — with vs without intraday time features
9.  Experiment 2 — with vs without macro regime features
10. Save FillEstimatorFractional (GBM) → models/fill_estimator_fractional.pkl
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lobster_data import (
    discover_files,
    build_dataset_fractional,
    make_splits,
    fit_vol_regime,
    apply_vol_regime,
    compute_uniqueness_weights,
    FILL_HORIZON_SEC,
)
from fill_estimator_fractional import FillEstimatorFractional

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
AAPL_DIR = os.path.join(
    BASE, "Data", "AAPL_2023-07-01_2023-07-31_10",
    "output-2023-07", "0", "0", "13",
)
PLOTS_DIR  = os.path.join(BASE, "plots")
MODELS_DIR = os.path.join(BASE, "models")

# ── Feature groups ─────────────────────────────────────────────────────────────

MICRO_FEATURES = [
    "imbalance", "depth_imbalance", "queue_ahead",
    "spread_norm", "local_vol", "aggressive_flow", "direction",
    "order_size",
]
TIME_FEATURES   = ["time_sin", "time_cos", "time_bucket"]
REGIME_FEATURES = ["vol_regime", "day_of_week"]
ALL_FEATURES    = MICRO_FEATURES + TIME_FEATURES + REGIME_FEATURES  # 13 total

TARGET       = "fill_fraction"
GA_MAX_ROWS  = 200_000   # subsample last N rows for GA to bound runtime

# ── Plot style ─────────────────────────────────────────────────────────────────

BG     = "#0f172a"
PANEL  = "#1e293b"
GRID   = "#334155"
TEXT   = "#e2e8f0"
COLORS = ["#38bdf8", "#f472b6", "#4ade80"]   # Ridge, RF, GBM

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":  GRID, "axes.labelcolor": TEXT,
    "xtick.color":     TEXT, "ytick.color":     TEXT,
    "text.color":      TEXT, "grid.color":      GRID,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})

# ── Model factories ────────────────────────────────────────────────────────────

def _make_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])

def _make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=50,
        n_jobs=-1, random_state=42,
    )

def _make_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )

def _fit_with_weights(model, X, y, w=None):
    if w is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        model.fit(X, y, ridge__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model

# ── Evaluation ─────────────────────────────────────────────────────────────────

def _day_metrics(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Per-day mean ± std of MAE, RMSE, R²."""
    maes, rmses, r2s = [], [], []
    for _, grp in df.groupby("date"):
        yt = grp[TARGET].values
        yp = grp[pred_col].values
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        if np.std(yt) > 0:
            r2s.append(r2_score(yt, yp))
    return {
        "mae_mean":  np.mean(maes),  "mae_std":  np.std(maes),
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "r2_mean":   float(np.mean(r2s)) if r2s else float("nan"),
        "r2_std":    float(np.std(r2s))  if r2s else float("nan"),
    }

def evaluate(model, df: pd.DataFrame, features: list[str]) -> dict:
    preds = np.clip(model.predict(df[features].values), 0.0, 1.0)
    y     = df[TARGET].values
    tmp   = df[["date", TARGET]].copy()
    tmp["pred"] = preds
    day   = _day_metrics(tmp)
    return {
        "tick_mae":  mean_absolute_error(y, preds),
        "tick_rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "tick_r2":   r2_score(y, preds),
        **{f"day_{k}": v for k, v in day.items()},
        "preds": preds,
        "df_pred": tmp,
    }

def _hdr():
    print(f"  {'Model':<22}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  "
          f"{'Day-R² mean±std':>20}")
    print(f"  {'-'*70}")

def _row(name: str, res: dict):
    print(
        f"  {name:<22}  {res['tick_mae']:>7.4f}  {res['tick_rmse']:>7.4f}  "
        f"{res['tick_r2']:>7.4f}  "
        f"{res['day_r2_mean']:>8.4f} ± {res['day_r2_std']:.4f}"
    )

# ── Feature selection ──────────────────────────────────────────────────────────

def select_mi(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]:
    scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    return sorted(zip(X.columns, scores), key=lambda t: t[1], reverse=True)

def select_rfecv(X: pd.DataFrame, y: pd.Series) -> list[str]:
    tscv    = TimeSeriesSplit(n_splits=5, gap=500)
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X)
    rfe     = RFECV(Ridge(alpha=1.0), cv=tscv, scoring="r2",
                    min_features_to_select=3, n_jobs=-1)
    rfe.fit(Xs, y)
    return [f for f, s in zip(X.columns, rfe.support_) if s]

def select_ga(
    X: pd.DataFrame, y: pd.Series,
    n_gen: int = 20, pop_size: int = 20,
    mut_rate: float = 0.15, lam: float = 0.05,
) -> list[str]:
    """Binary-chromosome GA; fitness = mean TimeSeriesSplit R² − λ·feat_frac."""
    feats   = list(X.columns)
    n_feats = len(feats)

    if len(X) > GA_MAX_ROWS:
        X_ga = X.iloc[-GA_MAX_ROWS:].copy()
        y_ga = y.iloc[-GA_MAX_ROWS:].copy()
    else:
        X_ga, y_ga = X.copy(), y.copy()

    tscv   = TimeSeriesSplit(n_splits=3, gap=500)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_ga)

    def fitness(chrom: np.ndarray) -> float:
        sel = np.where(chrom)[0]
        if len(sel) == 0:
            return -1.0
        cv = cross_val_score(Ridge(alpha=1.0), Xs[:, sel], y_ga,
                             cv=tscv, scoring="r2")
        return float(np.mean(cv)) - lam * (len(sel) / n_feats)

    rng  = np.random.default_rng(42)
    pop  = rng.integers(0, 2, size=(pop_size, n_feats)).astype(np.int8)
    for j in range(pop_size):
        if pop[j].sum() == 0:
            pop[j, rng.integers(n_feats)] = 1

    best_chrom, best_fit = pop[0].copy(), -np.inf

    for gen in range(n_gen):
        fits = np.array([fitness(c) for c in pop])
        idx  = int(np.argmax(fits))
        if fits[idx] > best_fit:
            best_fit  = fits[idx]
            best_chrom = pop[idx].copy()

        new_pop = []
        while len(new_pop) < pop_size:
            a1, b1 = rng.choice(pop_size, 2, replace=False)
            a2, b2 = rng.choice(pop_size, 2, replace=False)
            p1 = pop[a1] if fits[a1] >= fits[b1] else pop[b1]
            p2 = pop[a2] if fits[a2] >= fits[b2] else pop[b2]

            if rng.random() < 0.70:
                pt = int(rng.integers(1, n_feats))
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in (c1, c2):
                flip = rng.random(n_feats) < mut_rate
                c ^= flip.astype(np.int8)
                if c.sum() == 0:
                    c[int(rng.integers(n_feats))] = 1
                new_pop.append(c)

        pop = np.array(new_pop[:pop_size])
        if (gen + 1) % 5 == 0:
            print(f"    GA gen {gen+1}/{n_gen}  best_fit={best_fit:.4f}  "
                  f"n_feats={int(best_chrom.sum())}")

    return [feats[j] for j in range(n_feats) if best_chrom[j]]

# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_mi(mi_scores: list[tuple[str, float]]):
    names, scores = zip(*mi_scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=COLORS[2], alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mutual Information score")
    ax.set_title("Feature Ranking — Mutual Information (regression, fill fraction)")
    ax.grid(True, axis="x", lw=0.5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "frac_mi_ranking.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")

def _plot_calibration(results: dict, test_df: pd.DataFrame, features: list[str]):
    """Predicted fill fraction vs actual fill fraction, binned."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1, label="Perfect calibration")
    bins = np.linspace(0, 1, 21)
    for (name, res), color in zip(results.items(), COLORS):
        preds = res["preds"]
        actual = test_df[TARGET].values
        bin_idx = np.digitize(preds, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        x_pts, y_pts = [], []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() >= 10:
                x_pts.append(preds[mask].mean())
                y_pts.append(actual[mask].mean())
        ax.plot(x_pts, y_pts, "o-", color=color, lw=1.5, ms=4,
                label=f"{name}  R²={res['tick_r2']:.4f}")
    ax.set_xlabel("Mean Predicted Fill Fraction")
    ax.set_ylabel("Mean Actual Fill Fraction")
    ax.set_title("Calibration — Fill Fraction (test set)")
    ax.legend(fontsize=9); ax.grid(True, lw=0.5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "frac_calibration.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")

def _plot_importances(trained: dict, features: list[str]):
    model_map = {
        "Random Forest": ("frac_importance_random_forest.png", "feature_importances_", None),
        "GBM":           ("frac_importance_gbm.png",           "feature_importances_", None),
        "Ridge":         ("frac_importance_ridge.png",         None,                   "ridge"),
    }
    for name, (fname, imp_attr, step) in model_map.items():
        model = trained[name]
        if imp_attr:
            imps = getattr(model, imp_attr)
        else:
            imps = np.abs(model.named_steps[step].coef_)

        order = np.argsort(imps)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(np.arange(len(features)), imps[order], color=COLORS[2], alpha=0.85)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels([features[i] for i in order], fontsize=9)
        ax.set_xlabel("|coefficient|" if step else "Feature importance")
        ax.set_title(f"Feature Importance — {name} (fractional)")
        ax.grid(True, axis="x", lw=0.5)
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, fname)
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Saved {path}")

def _experiment_bar(label_a: str, r2_a: float, std_a: float,
                    label_b: str, r2_b: float, std_b: float,
                    title: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([0, 1])
    r2s  = [r2_a, r2_b]
    stds = [std_a, std_b]
    bars = ax.bar(x, r2s, color=COLORS[2], alpha=0.85, width=0.5,
                  yerr=stds, capsize=6, error_kw={"color": TEXT, "lw": 1.5})
    ax.set_xticks(x); ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Day-level R² (mean ± std)")
    ax.set_title(title)
    ax.grid(True, axis="y", lw=0.5)
    # Annotate bars
    for bar, r, s in zip(bars, r2s, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, r + s + 0.003,
                f"{r:.4f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")

# ── Experiments ────────────────────────────────────────────────────────────────

def _run_gbm_experiment(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    features: list[str], w: np.ndarray,
) -> dict:
    model = _make_gbm()
    _fit_with_weights(model, train_df[features], train_df[TARGET], w)
    return evaluate(model, test_df, features)

def _experiment1(train_df, val_df, test_df, w, vol_threshold):
    print("\nExperiment 1 — Intraday seasonality (MICRO vs MICRO+TIME) …")
    micro      = MICRO_FEATURES
    micro_time = MICRO_FEATURES + TIME_FEATURES

    res_a = _run_gbm_experiment(train_df, test_df, micro,      w)
    res_b = _run_gbm_experiment(train_df, test_df, micro_time, w)

    print(f"  MICRO only      R²= {res_a['day_r2_mean']:.4f} ± {res_a['day_r2_std']:.4f}")
    print(f"  MICRO + TIME    R²= {res_b['day_r2_mean']:.4f} ± {res_b['day_r2_std']:.4f}")

    _experiment_bar(
        "MICRO", res_a["day_r2_mean"], res_a["day_r2_std"],
        "MICRO+TIME", res_b["day_r2_mean"], res_b["day_r2_std"],
        "Exp 1 — Intraday Seasonality (fractional fill, GBM)",
        "frac_experiment1.png",
    )

def _experiment2(train_df, val_df, test_df, w, vol_threshold):
    print("\nExperiment 2 — Macro regime variables (MICRO+TIME vs +REGIME) …")
    micro_time        = MICRO_FEATURES + TIME_FEATURES
    micro_time_regime = MICRO_FEATURES + TIME_FEATURES + REGIME_FEATURES

    res_a = _run_gbm_experiment(train_df, test_df, micro_time,        w)
    res_b = _run_gbm_experiment(train_df, test_df, micro_time_regime, w)

    print(f"  MICRO+TIME         R²= {res_a['day_r2_mean']:.4f} ± {res_a['day_r2_std']:.4f}")
    print(f"  MICRO+TIME+REGIME  R²= {res_b['day_r2_mean']:.4f} ± {res_b['day_r2_std']:.4f}")

    _experiment_bar(
        "No Regime", res_a["day_r2_mean"], res_a["day_r2_std"],
        "+Regime", res_b["day_r2_mean"], res_b["day_r2_std"],
        "Exp 2 — Macro Regime Variables (fractional fill, GBM)",
        "frac_experiment2.png",
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("=" * 68)
    print("LOB Fill Fraction Estimator — Regression Pipeline")
    print("=" * 68)
    print(f"\nLoading AAPL data from {AAPL_DIR} …")
    files = discover_files(AAPL_DIR)
    if not files:
        raise FileNotFoundError(f"No LOBSTER file pairs found in {AAPL_DIR}")
    print(f"  Found {len(files)} trading days.")

    df = build_dataset_fractional(files, "AAPL")
    fully  = (df[TARGET] == 1.0).mean()
    partial = ((df[TARGET] > 0) & (df[TARGET] < 1)).mean()
    unfilled = (df[TARGET] == 0.0).mean()
    print(f"  Total rows: {len(df):,}")
    print(f"  Fill distribution:  fully={fully:.3f}  partial={partial:.3f}  "
          f"unfilled={unfilled:.3f}")
    print(f"  Mean fill fraction: {df[TARGET].mean():.4f}")

    # ── 2. Split ───────────────────────────────────────────────────────────────
    print("\nSplitting 70/15/15 (chronological, with purge + embargo) …")
    train_df, val_df, test_df = make_splits(df)
    print(f"  Train {len(train_df):,}  Val {len(val_df):,}  Test {len(test_df):,}")

    # ── 3. Vol-regime (train-only fit) ─────────────────────────────────────────
    vol_threshold = fit_vol_regime(train_df)
    print(f"\nVol-regime threshold (train median): {vol_threshold:.6f}")
    train_df = apply_vol_regime(train_df, vol_threshold)
    val_df   = apply_vol_regime(val_df,   vol_threshold)
    test_df  = apply_vol_regime(test_df,  vol_threshold)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]

    # ── 4. Feature selection ───────────────────────────────────────────────────
    print("\nFeature selection (training set only) …")

    print("  Mutual Information …")
    mi_scores = select_mi(X_train, y_train)

    print("  RFECV (Ridge, TimeSeriesSplit n=5, gap=500, scoring=r²) …")
    rfecv_feats = select_rfecv(X_train, y_train)
    print(f"    Selected: {rfecv_feats}")

    print("  Genetic Algorithm (Ridge fitness, TimeSeriesSplit n=3, gap=500) …")
    ga_feats = select_ga(X_train, y_train)
    print(f"    Selected: {ga_feats}")

    primary = sorted(set(ga_feats) | set(rfecv_feats))
    print(f"  Primary (GA ∪ RFECV): {primary}")

    # ── 5. Sample uniqueness weights ───────────────────────────────────────────
    print("\nComputing uniqueness weights …")
    w = compute_uniqueness_weights(train_df)

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print("\nTraining …")
    trained = {
        "Ridge":         _make_ridge(),
        "Random Forest": _make_rf(),
        "GBM":           _make_gbm(),
    }
    for name, model in trained.items():
        print(f"  {name} …")
        _fit_with_weights(model, train_df[primary], y_train, w)

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    print("\n── Validation ──")
    _hdr()
    for name, model in trained.items():
        _row(name, evaluate(model, val_df, primary))

    print("\n── Test set ──")
    _hdr()
    test_results = {}
    for name, model in trained.items():
        res = evaluate(model, test_df, primary)
        _row(name, res)
        test_results[name] = res

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    _plot_mi(mi_scores)
    _plot_calibration(test_results, test_df, primary)
    _plot_importances(trained, primary)

    # ── 9. Experiments ─────────────────────────────────────────────────────────
    _experiment1(train_df, val_df, test_df, w, vol_threshold)
    _experiment2(train_df, val_df, test_df, w, vol_threshold)

    # ── 10. Save ───────────────────────────────────────────────────────────────
    print("\nSaving models …")
    est = FillEstimatorFractional(
        model=trained["GBM"],
        feature_names=primary,
        vol_threshold=vol_threshold,
        fill_horizon_sec=FILL_HORIZON_SEC,
    )
    est.save(os.path.join(MODELS_DIR, "fill_estimator_fractional.pkl"))
    joblib.dump(trained["Ridge"],
                os.path.join(MODELS_DIR, "ridge_fractional.pkl"))
    joblib.dump(trained["Random Forest"],
                os.path.join(MODELS_DIR, "random_forest_fractional.pkl"))
    joblib.dump(trained["GBM"],
                os.path.join(MODELS_DIR, "gbm_fractional.pkl"))
    print("  Saved models/fill_estimator_fractional.pkl (FillEstimatorFractional)")
    print("  Saved models/ridge_fractional.pkl")
    print("  Saved models/random_forest_fractional.pkl")
    print("  Saved models/gbm_fractional.pkl")
    print("\nDone.")


if __name__ == "__main__":
    main()
