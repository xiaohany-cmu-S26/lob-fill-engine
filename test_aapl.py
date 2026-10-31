"""
test_aapl.py — In-Stock Test Set Evaluation (AAPL)
===================================================
Reconstructs the same chronological 70/15/15 split used in training and
evaluates all three saved models on the AAPL test split — the 15 % of AAPL
days the models never trained or validated on.

This gives the reference-point AUC / Brier to compare against the
CSCO cross-stock OOS numbers in test_oos.py.

Run:  python test_aapl.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from lobster_data import (
    discover_files, build_dataset, make_splits,
    fit_vol_regime, apply_vol_regime,
)
from fill_estimator import FillEstimator

# ── Paths ─────────────────────────────────────────────────────────────────────

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

BG     = "#0f172a"; PANEL  = "#1e293b"; GRID = "#334155"
TEXT   = "#e2e8f0"; COLORS = ["#38bdf8", "#f472b6", "#4ade80", "#fb923c"]

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":  GRID, "axes.labelcolor": TEXT,
    "xtick.color":     TEXT, "ytick.color":     TEXT,
    "text.color":      TEXT, "grid.color":      GRID,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})

# ── Load models ───────────────────────────────────────────────────────────────

print("Loading trained models …")
estimator: FillEstimator = FillEstimator.load(
    os.path.join(MODELS_DIR, "fill_estimator.pkl"))
lr_model   = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
rf_model   = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
gbm_model  = joblib.load(os.path.join(MODELS_DIR, "gbm.pkl"))
lgbm_model = joblib.load(os.path.join(MODELS_DIR, "lightgbm.pkl"))
calibrators = joblib.load(os.path.join(MODELS_DIR, "temperature_calibrators.pkl"))

FEATURE_NAMES = estimator.feature_names
VOL_THRESHOLD = estimator.vol_threshold

MODELS = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "GBM":                 gbm_model,
    "LightGBM":            lgbm_model,
}

# ── Reconstruct the exact same split used in training ─────────────────────────

print("Loading AAPL + CSCO data to reconstruct exact training split …")
# Training split was done on the combined AAPL+CSCO dataframe sorted by date.
# Splitting AAPL alone would place rows in different partitions — we must
# reproduce the exact same combined sort before calling make_splits.
aapl_files = discover_files(AAPL_DIR)
df_aapl    = build_dataset(aapl_files, "AAPL")
csco_files = discover_files(CSCO_DIR)
df_csco    = build_dataset(csco_files, "CSCO")

df_combined = (pd.concat([df_aapl, df_csco], ignore_index=True)
                 .sort_values("date")
                 .reset_index(drop=True))

_, _, test_raw = make_splits(df_combined)
test_df = test_raw[test_raw["ticker"] == "AAPL"].copy().reset_index(drop=True)

vol_threshold = VOL_THRESHOLD
test_df = apply_vol_regime(test_df, vol_threshold)

print(f"  Total AAPL rows : {len(df_aapl):,}")
print(f"  Test split rows : {len(test_df):,}  "
      f"({len(test_df)/len(df_aapl)*100:.1f} % of AAPL data)")
print(f"  Test fill rate  : {test_df['filled'].mean():.4f}")
print(f"  Test dates      : {test_df['date'].min()} → {test_df['date'].max()}")

missing = [f for f in FEATURE_NAMES if f not in test_df.columns]
if missing:
    raise KeyError(f"Missing features: {missing}")

X_test = test_df[FEATURE_NAMES]
y_test = test_df["filled"]

# ── Evaluation helpers ────────────────────────────────────────────────────────

def day_level_auc(df, prob_col="prob"):
    vals = []
    for _, g in df.groupby("date"):
        if g["filled"].nunique() < 2:
            continue
        vals.append(roc_auc_score(g["filled"], g[prob_col]))
    return float(np.mean(vals)), float(np.std(vals))

def day_level_brier(df, prob_col="prob"):
    vals = [brier_score_loss(g["filled"], g[prob_col])
            for _, g in df.groupby("date")]
    return float(np.mean(vals)), float(np.std(vals))

# ── Evaluate ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"{'Model':<22}  {'Tick AUC':>9}  {'Day AUC':>16}  {'Day Brier':>14}")
print("=" * 65)

results  = {}
prob_df  = test_df[["date", "filled"]].copy()

for name, model in MODELS.items():
    probs = model.predict_proba(X_test)[:, 1]
    prob_df[name] = probs

    tick_auc   = roc_auc_score(y_test, probs)
    tick_brier = brier_score_loss(y_test, probs)
    tmp = prob_df[["date", "filled", name]].rename(columns={name: "prob"})
    day_auc_m, day_auc_s     = day_level_auc(tmp)
    day_brier_m, day_brier_s = day_level_brier(tmp)

    results[name] = {
        "tick_auc": tick_auc, "tick_brier": tick_brier,
        "day_auc_mean": day_auc_m, "day_auc_std": day_auc_s,
        "day_brier_mean": day_brier_m, "day_brier_std": day_brier_s,
        "probs": probs,
    }
    print(f"{name:<22}  {tick_auc:>9.4f}  "
          f"{day_auc_m:>6.4f} ± {day_auc_s:.4f}  "
          f"{day_brier_m:>6.4f} ± {day_brier_s:.4f}")

print("=" * 65)

# ── Calibrated metrics ────────────────────────────────────────────────────────
print(f"\ntemperature-calibrated scores:")
print(f"{'Model':<22}  {'Tick AUC':>9}  {'Day AUC':>16}  {'Day Brier':>14}")
print("-" * 65)
for name, res in results.items():
    cal_probs = calibrators[name].predict_proba(
        res["probs"].reshape(-1, 1))[:, 1]
    results[name]["cal_probs"] = cal_probs
    tmp = prob_df[["date", "filled"]].copy(); tmp["prob"] = cal_probs
    tick_auc   = roc_auc_score(y_test, cal_probs)
    day_auc_m, day_auc_s     = day_level_auc(tmp)
    day_brier_m, day_brier_s = day_level_brier(tmp)
    print(f"{name:<22}  {tick_auc:>9.4f}  "
          f"{day_auc_m:>6.4f} ± {day_auc_s:.4f}  "
          f"{day_brier_m:>6.4f} ± {day_brier_s:.4f}")

# ── Aggregate calibration ─────────────────────────────────────────────────────

actual_fills = int(y_test.sum())
print(f"\nAggregate calibration (AAPL test fill rate: {y_test.mean():.4f})")
print(f"  Actual fills: {actual_fills:,}")
print(f"  {'Model':<22}  {'Raw ratio':>10}  {'Cal ratio':>10}  {'Raw bias':>10}  {'Cal bias':>10}")
print(f"  {'-'*64}")
for name, res in results.items():
    raw_pred = res["probs"].sum()
    cal_pred = res["cal_probs"].sum()
    raw_ratio = raw_pred / actual_fills
    cal_ratio = cal_pred / actual_fills
    print(f"  {name:<22}  {raw_ratio:>10.4f}  {cal_ratio:>10.4f}  "
          f"{raw_pred - actual_fills:>+10.0f}  {cal_pred - actual_fills:>+10.0f}")

# ── Plot: ROC curves ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)
for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    ax.plot(fpr, tpr, color=color, lw=1.8,
            label=f"{name}  AUC={res['tick_auc']:.4f}")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("AAPL Test Set ROC Curves (in-stock holdout)")
ax.legend(fontsize=9); ax.grid(True, lw=0.5)
fig.tight_layout()
path = os.path.join(PLOTS_DIR, "aapl_test_roc.png")
fig.savefig(path, dpi=150); plt.close(fig)
print(f"\nSaved {path}")

# ── Plot: Calibration ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1, label="Perfect")
for (name, res), color in zip(results.items(), COLORS):
    frac, mean_pred = calibration_curve(y_test, res["probs"], n_bins=15)
    ax.plot(mean_pred, frac, "o-", color=color, lw=1.5, ms=4, label=name)
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("AAPL Test Set Calibration (in-stock holdout)")
ax.legend(fontsize=9); ax.grid(True, lw=0.5)
fig.tight_layout()
path = os.path.join(PLOTS_DIR, "aapl_test_calibration.png")
fig.savefig(path, dpi=150); plt.close(fig)
print(f"Saved {path}")

print("\nDone.")
