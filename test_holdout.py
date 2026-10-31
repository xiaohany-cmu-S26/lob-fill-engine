"""
test_holdout.py — Holdout Evaluation on AAPL and CSCO Test-Split Days
======================================================================
Evaluates the trained models on the held-out test split (last ~15% of
July 2023 trading dates) for BOTH tickers.  Because training now uses
combined AAPL + CSCO data, this is the only valid evaluation: the test
dates are strictly unseen by the training and validation steps.

Reports per-ticker and combined:
  • Tick-level ROC-AUC
  • Day-level AUC mean ± std
  • Day-level Brier score (raw and temperature-calibrated)
  • Aggregate calibration bias ratio

Plots  → plots/holdout_roc_curves.png
         plots/holdout_calibration.png
         plots/holdout_day_auc.png

CSV    → holdout_predictions.csv

Run:  python test_holdout.py
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from lobster_data import discover_files, build_dataset, apply_vol_regime
from fill_estimator import FillEstimator

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
MODELS_DIR = os.path.join(BASE, "models")
PLOTS_DIR  = os.path.join(BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────

BG     = "#0f172a"
PANEL  = "#1e293b"
GRID   = "#334155"
TEXT   = "#e2e8f0"
# LR, RF, GBM colours; AAPL solid, CSCO dashed
COLORS = ["#38bdf8", "#f472b6", "#4ade80"]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
})

# ── Load models and split dates ────────────────────────────────────────────────

print("Loading trained models …")
estimator: FillEstimator = FillEstimator.load(os.path.join(MODELS_DIR, "fill_estimator.pkl"))
lr_model    = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
rf_model    = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
gbm_model   = joblib.load(os.path.join(MODELS_DIR, "gbm.pkl"))
calibrators = joblib.load(os.path.join(MODELS_DIR, "temperature_calibrators.pkl"))

FEATURE_NAMES = estimator.feature_names
VOL_THRESHOLD = estimator.vol_threshold

MODELS = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "GBM":                 gbm_model,
}

split_path = os.path.join(MODELS_DIR, "split_dates.json")
if not os.path.exists(split_path):
    raise FileNotFoundError(
        f"split_dates.json not found at {split_path}. Re-run train.py first."
    )
with open(split_path) as _f:
    split_dates = json.load(_f)

TEST_DATES = set(split_dates["test_dates"])
print(f"Test split: {len(TEST_DATES)} dates  "
      f"({min(TEST_DATES)} → {max(TEST_DATES)})")

# ── Load and filter datasets ───────────────────────────────────────────────────

def _load_test(data_dir: str, ticker: str) -> pd.DataFrame:
    pairs = discover_files(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No file pairs found in {data_dir}")
    test_pairs = [p for p in pairs if p["date"] in TEST_DATES]
    if not test_pairs:
        raise ValueError(
            f"None of the {ticker} days fall in the test split {sorted(TEST_DATES)}."
        )
    print(f"  {ticker}: {len(test_pairs)} test days  "
          f"({test_pairs[0]['date']} → {test_pairs[-1]['date']})")
    df = build_dataset(test_pairs, ticker=ticker)
    df = apply_vol_regime(df, VOL_THRESHOLD)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features for {ticker}: {missing}")
    return df


print("\nLoading test-split data …")
aapl_df = _load_test(AAPL_DIR, "AAPL")
csco_df = _load_test(CSCO_DIR, "CSCO")

print(f"  AAPL rows: {len(aapl_df):,}  fill rate: {aapl_df['filled'].mean():.3f}")
print(f"  CSCO rows: {len(csco_df):,}  fill rate: {csco_df['filled'].mean():.3f}")

# ── Evaluation helpers ─────────────────────────────────────────────────────────

def day_level_auc(df: pd.DataFrame, prob_col: str = "prob") -> tuple[float, float]:
    vals = [roc_auc_score(grp["filled"], grp[prob_col])
            for _, grp in df.groupby("date")
            if grp["filled"].nunique() == 2]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def day_level_brier(df: pd.DataFrame, prob_col: str = "prob") -> tuple[float, float]:
    vals = [brier_score_loss(grp["filled"], grp[prob_col])
            for _, grp in df.groupby("date")]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def evaluate_split(df: pd.DataFrame, label: str) -> dict[str, dict]:
    """Run all three models (raw + calibrated) on df; return results dict."""
    results = {}
    X = df[FEATURE_NAMES]
    y = df["filled"]
    for name, model in MODELS.items():
        raw  = model.predict_proba(X)[:, 1]
        cal  = calibrators[name].predict_proba(raw.reshape(-1, 1))[:, 1]
        tmp  = df[["date", "filled"]].copy()

        tmp["prob"] = raw
        day_auc_m,   day_auc_s   = day_level_auc(tmp)
        day_brier_m, day_brier_s = day_level_brier(tmp)

        tmp["prob"] = cal
        cal_day_auc_m,   cal_day_auc_s   = day_level_auc(tmp)
        cal_day_brier_m, cal_day_brier_s = day_level_brier(tmp)

        results[name] = {
            "raw":          raw,
            "cal":          cal,
            "tick_auc":     roc_auc_score(y, raw),
            "tick_brier":   brier_score_loss(y, raw),
            "day_auc_m":    day_auc_m,
            "day_auc_s":    day_auc_s,
            "day_brier_m":  day_brier_m,
            "day_brier_s":  day_brier_s,
            "cal_tick_auc": roc_auc_score(y, cal),
            "cal_day_auc_m":    cal_day_auc_m,
            "cal_day_auc_s":    cal_day_auc_s,
            "cal_day_brier_m":  cal_day_brier_m,
            "cal_day_brier_s":  cal_day_brier_s,
            "actual_fills": int(y.sum()),
        }
    return results


# ── Run evaluation ─────────────────────────────────────────────────────────────

def _print_results(label: str, df: pd.DataFrame, res: dict) -> None:
    y = df["filled"]
    print(f"\n{'=' * 72}")
    print(f"{label}   ({len(df):,} rows, fill rate {y.mean():.3f})")
    print(f"{'=' * 72}")
    print(f"  {'Model':<22}  {'Tick AUC':>9}  {'Day AUC':>16}  {'Day Brier':>14}")
    print(f"  {'-' * 66}")
    for name, r in res.items():
        print(f"  {name:<22}  {r['tick_auc']:>9.4f}  "
              f"{r['day_auc_m']:>6.4f} ± {r['day_auc_s']:.4f}  "
              f"{r['day_brier_m']:>6.4f} ± {r['day_brier_s']:.4f}")
    print(f"\n  temperature-calibrated:")
    print(f"  {'-' * 66}")
    for name, r in res.items():
        print(f"  {name:<22}  {r['cal_tick_auc']:>9.4f}  "
              f"{r['cal_day_auc_m']:>6.4f} ± {r['cal_day_auc_s']:.4f}  "
              f"{r['cal_day_brier_m']:>6.4f} ± {r['cal_day_brier_s']:.4f}")
    print(f"\n  Aggregate calibration  (actual fills: {r['actual_fills']:,})")
    print(f"  {'Model':<22}  {'Raw ratio':>10}  {'Cal ratio':>10}  "
          f"{'Raw bias':>10}  {'Cal bias':>10}")
    print(f"  {'-' * 68}")
    for name, r in res.items():
        raw_sum = r["raw"].sum();  cal_sum = r["cal"].sum()
        af = r["actual_fills"]
        print(f"  {name:<22}  {raw_sum/af:>10.4f}  {cal_sum/af:>10.4f}  "
              f"{raw_sum - af:>+10.0f}  {cal_sum - af:>+10.0f}")


print("\nEvaluating …")
aapl_res = evaluate_split(aapl_df, "AAPL")
csco_res = evaluate_split(csco_df, "CSCO")

combined_df  = pd.concat([aapl_df, csco_df], ignore_index=True)
combined_res = evaluate_split(combined_df, "Combined")

_print_results("AAPL — test split",     aapl_df,     aapl_res)
_print_results("CSCO — test split",     csco_df,     csco_res)
_print_results("Combined — test split", combined_df, combined_res)

# ── Plot 1: ROC curves (AAPL solid, CSCO dashed) ──────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)

for (name, _), color in zip(MODELS.items(), COLORS):
    for ticker, res, ls in [("AAPL", aapl_res, "-"), ("CSCO", csco_res, "--")]:
        fpr, tpr, _ = roc_curve(
            (aapl_df if ticker == "AAPL" else csco_df)["filled"],
            res[name]["raw"],
        )
        auc = res[name]["tick_auc"]
        ax.plot(fpr, tpr, color=color, lw=1.6, ls=ls,
                label=f"{name} {ticker}  AUC={auc:.4f}")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Holdout ROC Curves — AAPL & CSCO test split")
ax.legend(fontsize=8)
ax.grid(True, lw=0.5)
fig.tight_layout()
p = os.path.join(PLOTS_DIR, "holdout_roc_curves.png")
fig.savefig(p, dpi=150);  plt.close(fig)
print(f"\nSaved {p}")

# ── Plot 2: Calibration curves (combined) ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1, label="Perfect calibration")

for (name, _), color in zip(MODELS.items(), COLORS):
    frac_pos, mean_pred = calibration_curve(
        combined_df["filled"], combined_res[name]["raw"], n_bins=15
    )
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=1.5, ms=4, label=name)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Holdout Calibration — combined test split")
ax.legend(fontsize=9)
ax.grid(True, lw=0.5)
fig.tight_layout()
p = os.path.join(PLOTS_DIR, "holdout_calibration.png")
fig.savefig(p, dpi=150);  plt.close(fig)
print(f"Saved {p}")

# ── Plot 3: Per-day GBM AUC bars, both tickers ────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

for ax, df, res, ticker, color in [
    (axes[0], aapl_df, aapl_res, "AAPL", COLORS[2]),
    (axes[1], csco_df, csco_res, "CSCO", COLORS[0]),
]:
    tmp = df[["date", "filled"]].copy()
    tmp["prob"] = res["GBM"]["raw"]
    day_aucs = {
        d: roc_auc_score(grp["filled"], grp["prob"])
        for d, grp in tmp.groupby("date")
        if grp["filled"].nunique() == 2
    }
    dates = list(day_aucs.keys())
    aucs  = [day_aucs[d] for d in dates]
    x = np.arange(len(dates))
    ax.bar(x, aucs, color=color, alpha=0.85, width=0.6)
    ax.axhline(np.mean(aucs), color=TEXT, lw=1.2, ls="--",
               label=f"Mean = {np.mean(aucs):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"GBM Day-Level AUC — {ticker} (test split)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", lw=0.5)

fig.tight_layout()
p = os.path.join(PLOTS_DIR, "holdout_day_auc.png")
fig.savefig(p, dpi=150);  plt.close(fig)
print(f"Saved {p}")

# ── Save per-row predictions ───────────────────────────────────────────────────

def _pred_df(df: pd.DataFrame, res: dict) -> pd.DataFrame:
    out = df[["date", "ticker", "filled"]].copy().reset_index(drop=True)
    for name in MODELS:
        col = f"prob_{name.lower().replace(' ', '_')}"
        out[col] = res[name]["raw"]
        out[f"{col}_cal"] = res[name]["cal"]
    return out

pred_df = pd.concat([_pred_df(aapl_df, aapl_res),
                     _pred_df(csco_df, csco_res)], ignore_index=True)
out_path = os.path.join(BASE, "holdout_predictions.csv")
pred_df.to_csv(out_path, index=False)
print(f"Saved per-row predictions → {out_path}")

print("\nDone.")
